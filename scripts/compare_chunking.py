"""
Compare old (flat) chunking vs new (hierarchical) chunking.
"""
import json
from pathlib import Path

def main():
    old_file = Path("MIL-STD-extraction-chunks.json")
    new_file = Path("MIL-STD-extraction-chunks-HIERARCHY.json")

    if not old_file.exists() or not new_file.exists():
        print("ERROR: Missing chunk files!")
        return

    with open(old_file, "r", encoding="utf-8") as f:
        old_data = json.load(f)

    with open(new_file, "r", encoding="utf-8") as f:
        new_data = json.load(f)

    old_chunks = old_data["chunks"]
    new_chunks = new_data["chunks"]

    print("=" * 80)
    print("CHUNKING COMPARISON: FLAT vs HIERARCHICAL")
    print("=" * 80)
    print()

    print("OVERVIEW")
    print("-" * 80)
    hierarchy_pct = 100 * new_data['stats']['chunks_with_hierarchy'] / len(new_chunks)
    hierarchy_str = f"{new_data['stats']['chunks_with_hierarchy']} ({hierarchy_pct:.1f}%)"

    print(f"{'Metric':<40} {'Flat':<20} {'Hierarchical':<20}")
    print("-" * 80)
    print(f"{'Total chunks':<40} {len(old_chunks):<20} {len(new_chunks):<20}")
    print(f"{'Avg tokens per chunk':<40} {old_data['stats']['avg_tokens']:<20.1f} {new_data['stats']['avg_tokens']:<20.1f}")
    print(f"{'Chunks with hierarchy context':<40} {'0 (0%)':<20} {hierarchy_str:<20}")
    print(f"{'Avg quality score':<40} {'N/A':<20} {new_data['stats']['avg_quality']:<20.3f}")
    print()

    print("WHY MORE CHUNKS?")
    print("-" * 80)
    print("The hierarchical chunker creates MORE chunks because it respects section")
    print("boundaries. Instead of forcing content together to reach token limits,")
    print("it breaks at logical section boundaries, creating cleaner, more focused chunks.")
    print()
    print("Benefits:")
    print("  - Each chunk has clear semantic meaning (one section = one chunk)")
    print("  - Retrieval finds exact sections, not mixed content")
    print("  - Hierarchy paths provide context (where in doc this came from)")
    print("  - Higher quality scores (complete sections, not partial)")
    print()

    print("EXAMPLE CHUNKS")
    print("=" * 80)
    print()

    # Find a chunk from page 10 in both
    old_chunk_p10 = next((c for c in old_chunks if c.get("content") and "FOREWORD" in c.get("content", "")), old_chunks[4])
    new_chunk_p10 = next((c for c in new_chunks if c.get("hierarchy_path") == "FOREWORD"), new_chunks[4])

    print("OLD (FLAT) CHUNK EXAMPLE:")
    print("-" * 80)
    print(f"ID: {old_chunk_p10['id']}")
    print(f"Tokens: {old_chunk_p10['token_count']}")
    print(f"Hierarchy path: {old_chunk_p10.get('hierarchy_path', 'NONE')}")
    print(f"Content preview:")
    content_preview = old_chunk_p10['content'][:300].replace('\n', ' ')
    print(f"  {content_preview}...")
    print()

    print("NEW (HIERARCHICAL) CHUNK EXAMPLE:")
    print("-" * 80)
    print(f"ID: {new_chunk_p10['id']}")
    print(f"Tokens: {new_chunk_p10['token_count']}")
    print(f"Hierarchy path: {new_chunk_p10.get('hierarchy_path', 'NONE')}")
    print(f"Quality score: {new_chunk_p10.get('quality_score', 0):.3f}")
    quality_details = new_chunk_p10.get('quality_details', {})
    if quality_details:
        print(f"Quality breakdown:")
        print(f"  - Token range: {quality_details.get('token_range', 0):.3f}")
        print(f"  - Sentence complete: {quality_details.get('sentence_complete', 0):.3f}")
        print(f"  - Hierarchy preserved: {quality_details.get('hierarchy_preserved', 0):.3f}")
        print(f"  - Table integrity: {quality_details.get('table_integrity', 0):.3f}")
    print(f"Content preview:")
    content_preview = new_chunk_p10['content'][:300].replace('\n', ' ')
    print(f"  {content_preview}...")
    print()

    # Show hierarchy path examples
    print("HIERARCHY PATH EXAMPLES")
    print("=" * 80)
    print("Hierarchy paths show the document structure, making it easy to understand")
    print("where each chunk came from:")
    print()

    unique_paths = list(set(c.get('hierarchy_path', '') for c in new_chunks if c.get('hierarchy_path')))[:10]
    for i, path in enumerate(unique_paths, 1):
        print(f"{i}. {path}")
    print()

    # Token distribution
    print("TOKEN DISTRIBUTION")
    print("=" * 80)

    old_tokens = [c['token_count'] for c in old_chunks]
    new_tokens = [c['token_count'] for c in new_chunks]

    print(f"{'Range':<20} {'Flat Chunks':<20} {'Hierarchical Chunks':<20}")
    print("-" * 80)

    ranges = [(0, 100), (101, 200), (201, 300), (301, 400), (401, 512), (513, 9999)]
    for min_tok, max_tok in ranges:
        old_count = sum(1 for t in old_tokens if min_tok <= t <= max_tok)
        new_count = sum(1 for t in new_tokens if min_tok <= t <= max_tok)
        range_str = f"{min_tok}-{max_tok if max_tok < 9999 else '512+'}"
        print(f"{range_str:<20} {old_count:<20} {new_count:<20}")
    print()

    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()
    print("Use the HIERARCHICAL chunks for RAG because:")
    print()
    print("1. SEMANTIC COHERENCE: Each chunk is a complete section, not arbitrary")
    print("   text fragments")
    print()
    print("2. BETTER RETRIEVAL: When user asks about 'FOREWORD', you get the exact")
    print("   FOREWORD section, not mixed content")
    print()
    print("3. CONTEXT PRESERVED: Hierarchy paths show document structure, helping")
    print("   the LLM understand where information came from")
    print()
    print("4. QUALITY SCORES: Built-in quality metrics help identify well-formed chunks")
    print()
    print("5. FLEXIBLE: Can still use overlap for continuity if needed")
    print()
    print("=" * 80)

    # Save comparison report
    report_file = Path("chunking_comparison_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("CHUNKING COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Flat chunks: {len(old_chunks)}\n")
        f.write(f"Hierarchical chunks: {len(new_chunks)}\n")
        f.write(f"Improvement: {len(new_chunks) - len(old_chunks)} more semantically coherent chunks\n")
        f.write(f"\nHierarchy coverage: {100*new_data['stats']['chunks_with_hierarchy']/len(new_chunks):.1f}%\n")
        f.write(f"Avg quality score: {new_data['stats']['avg_quality']:.3f}\n")

    print(f"\nReport saved to: {report_file}")
    print()

if __name__ == "__main__":
    main()
