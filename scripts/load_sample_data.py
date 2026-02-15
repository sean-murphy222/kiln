"""
Load sample MIL-STD chunk data into CHONK for diagnostic testing.

This demonstrates the diagnostic-first workflow:
1. Load existing chunks from a real document
2. Run diagnostics to identify problems
3. Show the "aha moment" when problems are detected
"""

import json
from pathlib import Path
from chonk.core.document import (
    ChonkDocument,
    DocumentMetadata,
    Chunk,
    ChunkMetadata,
    QualityScore,
    Block,
)


def load_milstd_data(json_path: Path) -> ChonkDocument:
    """Load MIL-STD chunk data from JSON file."""

    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    doc_data = data['document']
    chunks_data = data['chunks']

    print(f"Loading: {doc_data['filename']}")
    print(f"  Pages: {doc_data['metadata']['page_count']}")
    print(f"  Words: {doc_data['metadata']['word_count']}")
    print(f"  Chunks: {len(chunks_data)}")

    # Create document metadata
    metadata = DocumentMetadata(
        title=doc_data['metadata'].get('title'),
        author=doc_data['metadata'].get('author'),
        page_count=doc_data['metadata']['page_count'],
        word_count=doc_data['metadata']['word_count'],
        file_size_bytes=doc_data['metadata']['file_size_bytes'],
    )

    # Create chunks
    chunks = []
    for chunk_data in chunks_data:
        quality = QualityScore(
            token_range=chunk_data['quality_details']['token_range'],
            sentence_complete=chunk_data['quality_details']['sentence_complete'],
            hierarchy_preserved=chunk_data['quality_details']['hierarchy_preserved'],
            table_integrity=chunk_data['quality_details']['table_integrity'],
            reference_complete=chunk_data['quality_details']['reference_complete'],
            # overall is calculated automatically as a property
        )

        user_metadata = ChunkMetadata(
            tags=chunk_data['user_metadata']['tags'],
            hierarchy_hint=chunk_data['user_metadata']['hierarchy_hint'],
            notes=chunk_data['user_metadata']['notes'],
            custom=chunk_data['user_metadata']['custom'],
        )

        chunk = Chunk(
            id=chunk_data['id'],
            block_ids=chunk_data['block_ids'],
            content=chunk_data['content'],
            token_count=chunk_data['token_count'],
            hierarchy_path=chunk_data['hierarchy_path'],
            quality=quality,
            system_metadata=chunk_data['system_metadata'],
            user_metadata=user_metadata,
        )
        chunks.append(chunk)

    # Create document (without blocks for now - we only have chunks)
    document = ChonkDocument(
        id=ChonkDocument.generate_id(),
        source_path=Path(doc_data['filename']),
        source_type='pdf',
        blocks=[],  # We don't have block data in this export
        chunks=chunks,
        metadata=metadata,
        loader_used='imported',
        chunker_used=doc_data['chunking_params']['chunker'],
        chunker_config=doc_data['chunking_params'],
    )

    return document


def analyze_chunk_problems(document: ChonkDocument):
    """Run basic diagnostic analysis on chunks."""

    print("\n" + "="*80)
    print("DIAGNOSTIC ANALYSIS")
    print("="*80)

    # 1. Token distribution analysis
    token_counts = [c.token_count for c in document.chunks]
    avg_tokens = sum(token_counts) / len(token_counts)
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)

    print(f"\nToken Distribution:")
    print(f"  Average: {avg_tokens:.1f} tokens")
    print(f"  Min: {min_tokens} tokens")
    print(f"  Max: {max_tokens} tokens")

    # 2. Find very small chunks (potential semantic incompleteness)
    small_chunks = [c for c in document.chunks if c.token_count < 20]
    print(f"\n[!] Very small chunks (<20 tokens): {len(small_chunks)}")
    if small_chunks:
        print("  Sample problematic chunks:")
        for chunk in small_chunks[:5]:
            print(f"    • {chunk.id}: {chunk.token_count} tokens - \"{chunk.content[:60]}...\"")

    # 3. Find very large chunks (potential semantic contamination)
    large_chunks = [c for c in document.chunks if c.token_count > 500]
    print(f"\n[!]  Very large chunks (>500 tokens): {len(large_chunks)}")
    if large_chunks:
        print("  Sample problematic chunks:")
        for chunk in large_chunks[:5]:
            preview = chunk.content[:100].replace('\n', ' ')
            print(f"    • {chunk.id}: {chunk.token_count} tokens - \"{preview}...\"")

    # 4. Find chunks with incomplete sentences (basic heuristic)
    incomplete_chunks = []
    for chunk in document.chunks:
        content = chunk.content.strip()
        # Check if chunk starts with lowercase (might be mid-sentence)
        if content and content[0].islower() and not content[0].isdigit():
            incomplete_chunks.append(chunk)

    print(f"\n[!]  Chunks starting with lowercase (potential fragments): {len(incomplete_chunks)}")
    if incomplete_chunks:
        print("  Sample problematic chunks:")
        for chunk in incomplete_chunks[:5]:
            preview = chunk.content[:80].replace('\n', ' ')
            print(f"    • {chunk.id}: \"{preview}...\"")

    # 5. Find chunks with dangling connectives
    connectives = ['however', 'therefore', 'additionally', 'furthermore', 'moreover', 'consequently']
    dangling_chunks = []
    for chunk in document.chunks:
        content_lower = chunk.content.lower().strip()
        for conn in connectives:
            if content_lower.startswith(conn):
                dangling_chunks.append((chunk, conn))
                break

    print(f"\n[!]  Chunks starting with connectives (semantic incompleteness): {len(dangling_chunks)}")
    if dangling_chunks:
        print("  Sample problematic chunks:")
        for chunk, conn in dangling_chunks[:5]:
            preview = chunk.content[:80].replace('\n', ' ')
            print(f"    • {chunk.id} (starts with '{conn}'): \"{preview}...\"")

    # 6. Find chunks with broken references
    reference_patterns = [
        'see above', 'as mentioned', 'as follows', 'see section',
        'see table', 'see figure', 'as shown', 'as described'
    ]
    reference_chunks = []
    for chunk in document.chunks:
        content_lower = chunk.content.lower()
        for pattern in reference_patterns:
            if pattern in content_lower:
                reference_chunks.append((chunk, pattern))
                break

    print(f"\n[!]  Chunks with references (potential orphaning): {len(reference_chunks)}")
    if reference_chunks:
        print("  Sample chunks with references:")
        for chunk, pattern in reference_chunks[:5]:
            # Find the sentence containing the reference
            content_lower = chunk.content.lower()
            idx = content_lower.find(pattern)
            start = max(0, idx - 30)
            end = min(len(chunk.content), idx + 70)
            preview = chunk.content[start:end].replace('\n', ' ')
            print(f"    • {chunk.id} ('{pattern}'): \"...{preview}...\"")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    total_problems = len(small_chunks) + len(large_chunks) + len(incomplete_chunks) + len(dangling_chunks)
    print(f"Total chunks analyzed: {len(document.chunks)}")
    print(f"Chunks with potential problems: {total_problems}")
    print(f"Problem rate: {total_problems / len(document.chunks) * 100:.1f}%")

    print("\n[*] These chunks all have 'perfect' quality scores (1.0), but diagnostics")
    print("   reveal real problems! This is the 'aha moment' - showing why")
    print("   quality scores alone aren't enough.")


if __name__ == '__main__':
    # Load the MIL-STD data
    json_path = Path('MIL-STD-extraction-chunks-HIERARCHY.json')

    if not json_path.exists():
        print(f"Error: {json_path} not found")
        print("Please run this script from the CHONK root directory")
        exit(1)

    document = load_milstd_data(json_path)

    # Run diagnostic analysis
    analyze_chunk_problems(document)

    print("\n" + "="*80)
    print("Next steps:")
    print("  1. Implement these detection algorithms in src/chonk/diagnostics/")
    print("  2. Add API endpoint: POST /api/diagnostics/analyze")
    print("  3. Connect 'RUN DIAGNOSTICS' button in UI")
    print("  4. Show problems in DiagnosticDashboard")
    print("="*80)
