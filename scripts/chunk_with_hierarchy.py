"""
Create hierarchical chunks from properly extracted blocks.
"""
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chonk.chunkers.hierarchy import HierarchyChunker
from chonk.chunkers.base import ChunkerConfig
from chonk.core.document import ChonkDocument, Block, BlockType, BoundingBox, DocumentMetadata

def main():
    blocks_file = Path("MIL-STD-extraction-blocks-HIERARCHY.json")

    if not blocks_file.exists():
        print(f"ERROR: {blocks_file} not found!")
        print("Run extract_with_hierarchy.py first")
        return

    print("=" * 70)
    print("HIERARCHICAL CHUNKING")
    print("=" * 70)
    print(f"[LOADING] {blocks_file.name}...")

    with open(blocks_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct blocks
    blocks = []
    for b in data["blocks"]:
        bbox = None
        if b["bbox"]:
            bbox = BoundingBox(
                x1=b["bbox"]["x1"],
                y1=b["bbox"]["y1"],
                x2=b["bbox"]["x2"],
                y2=b["bbox"]["y2"],
                page=b["bbox"]["page"],
            )

        block = Block(
            id=b["id"],
            type=BlockType(b["type"]),
            content=b["content"],
            bbox=bbox,
            page=b["page"],
            heading_level=b.get("heading_level"),
            metadata=b.get("metadata", {}),
        )
        blocks.append(block)

    # Count headings
    headings = [b for b in blocks if b.type == BlockType.HEADING]
    print(f"[LOADED] {len(blocks)} blocks ({len(headings)} headings)")
    print()

    # Show hierarchy stats
    from collections import Counter
    heading_levels = Counter(h.heading_level for h in headings if h.heading_level)
    print("[HEADING STRUCTURE]")
    for level in sorted(heading_levels.keys()):
        print(f"  Level {level}: {heading_levels[level]} headings")
    print()

    # Create chunker with hierarchy-aware config
    print("[CHUNKING] Creating chunks with hierarchy preservation...")
    print("  Parameters:")
    print("    - Max tokens: 512")
    print("    - Overlap: 50 tokens")
    print("    - Preserve tables: True")
    print("    - Group under headings: True  <-- KEY FEATURE")
    print()

    config = ChunkerConfig(
        max_tokens=512,
        overlap_tokens=50,
        preserve_tables=True,
        group_under_headings=True,  # This is critical!
        heading_weight=2.0,  # Strongly prefer breaking at headings
    )

    chunker = HierarchyChunker(config=config)

    # Chunk the blocks
    chunks = chunker.chunk(blocks)

    print(f"[CHUNKING] Created {len(chunks)} chunks")
    print()

    # Analyze chunk quality
    chunk_sizes = [len(chunk.content.split()) for chunk in chunks]
    token_counts = [chunk.token_count for chunk in chunks]

    # Count chunks with hierarchy paths
    chunks_with_hierarchy = sum(1 for c in chunks if c.hierarchy_path)

    # Analyze quality scores
    quality_scores = [c.quality.overall for c in chunks if c.quality]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

    print("[CHUNK STATS]")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Chunks with hierarchy paths: {chunks_with_hierarchy} ({100*chunks_with_hierarchy/len(chunks):.1f}%)")
    print(f"  Avg quality score: {avg_quality:.3f}")
    print()
    print("  Word count:")
    print(f"    Avg: {sum(chunk_sizes) / len(chunk_sizes):.1f}")
    print(f"    Min: {min(chunk_sizes)}")
    print(f"    Max: {max(chunk_sizes)}")
    print()
    print("  Token count:")
    print(f"    Avg: {sum(token_counts) / len(token_counts):.1f}")
    print(f"    Min: {min(token_counts)}")
    print(f"    Max: {max(token_counts)}")
    print()

    # Save chunks as JSON
    chunks_output = Path("MIL-STD-extraction-chunks-HIERARCHY.json")

    meta_data = data["document"]["metadata"]
    chunks_data = {
        "document": {
            "filename": data["document"]["filename"],
            "metadata": meta_data,
            "chunking_params": {
                "max_tokens": 512,
                "overlap_tokens": 50,
                "preserve_tables": True,
                "group_under_headings": True,
                "heading_weight": 2.0,
                "chunker": "HierarchyChunker",
            },
        },
        "chunks": [
            {
                "id": chunk.id,
                "content": chunk.content,
                "block_ids": chunk.block_ids,
                "token_count": chunk.token_count,
                "hierarchy_path": chunk.hierarchy_path,
                "quality_score": chunk.quality.overall if chunk.quality else 0,
                "quality_details": chunk.quality.to_dict() if chunk.quality else {},
                "system_metadata": chunk.system_metadata,
                "user_metadata": chunk.user_metadata.to_dict() if chunk.user_metadata else {},
            }
            for chunk in chunks
        ],
        "stats": {
            "total_chunks": len(chunks),
            "total_blocks": len(blocks),
            "total_headings": len(headings),
            "chunks_with_hierarchy": chunks_with_hierarchy,
            "avg_tokens": sum(token_counts) / len(chunks) if chunks else 0,
            "avg_words": sum(chunk_sizes) / len(chunks) if chunks else 0,
            "avg_quality": avg_quality,
        }
    }

    with open(chunks_output, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] Chunks saved to: {chunks_output}")
    print(f"        File size: {chunks_output.stat().st_size / (1024*1024):.2f} MB")
    print()

    # Show first few chunks with hierarchy
    chunks_to_show = [c for c in chunks if c.hierarchy_path][:5]
    if chunks_to_show:
        print("[FIRST 5 HIERARCHICAL CHUNKS]")
        print("-" * 70)
        for i, chunk in enumerate(chunks_to_show, 1):
            content_preview = chunk.content[:150].replace('\n', ' ')
            if len(chunk.content) > 150:
                content_preview += "..."

            page_info = ""
            if chunk.system_metadata.get("start_page") and chunk.system_metadata.get("end_page"):
                page_info = f"Pages: {chunk.system_metadata['start_page']}-{chunk.system_metadata['end_page']}"

            print(f"{i}. Chunk {chunk.id}")
            if page_info:
                print(f"   {page_info}")
            print(f"   Tokens: {chunk.token_count}")
            print(f"   Quality: {chunk.quality.overall:.3f}")
            print(f"   Blocks: {len(chunk.block_ids)}")
            print(f"   Hierarchy: {chunk.hierarchy_path}")
            print(f"   Content: {content_preview}")
            print()

    # Compare with old chunking
    old_chunks_file = Path("MIL-STD-extraction-chunks.json")
    if old_chunks_file.exists():
        with open(old_chunks_file, "r", encoding="utf-8") as f:
            old_data = json.load(f)

        old_chunk_count = len(old_data["chunks"])
        old_avg_tokens = old_data["stats"].get("avg_tokens", 0)

        print("=" * 70)
        print("[COMPARISON] Old vs New Chunking")
        print("=" * 70)
        print(f"{'Metric':<30} {'Old':<15} {'New':<15}")
        print("-" * 70)
        print(f"{'Total chunks':<30} {old_chunk_count:<15} {len(chunks):<15}")
        print(f"{'Avg tokens per chunk':<30} {old_avg_tokens:<15.1f} {sum(token_counts)/len(chunks):<15.1f}")
        print(f"{'Chunks with hierarchy':<30} {'0 (0%)':<15} {f'{chunks_with_hierarchy} ({100*chunks_with_hierarchy/len(chunks):.1f}%)':<15}")
        print(f"{'Avg quality score':<30} {'N/A':<15} {avg_quality:<15.3f}")
        print()

    print("=" * 70)
    print("[SUCCESS] Hierarchical chunking complete!")
    print(f"  - {len(chunks)} chunks with intelligent section grouping")
    print(f"  - {chunks_with_hierarchy} chunks ({100*chunks_with_hierarchy/len(chunks):.1f}%) have hierarchy context")
    print(f"  - Avg quality score: {avg_quality:.3f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
