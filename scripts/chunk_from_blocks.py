"""
Create chunks from already-extracted blocks JSON file.
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
    blocks_file = Path("MIL-STD-extraction-blocks.json")

    if not blocks_file.exists():
        print(f"ERROR: {blocks_file} not found!")
        print("Run test_docling_gpu.py first to extract blocks")
        return

    print(f"[LOADING] Reading blocks from {blocks_file.name}...")

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

    print(f"[LOADED] {len(blocks)} blocks")
    print()

    # Reconstruct metadata
    meta_data = data["document"]["metadata"]
    metadata = DocumentMetadata(
        title=meta_data.get("title"),
        author=meta_data.get("author"),
        page_count=meta_data.get("page_count"),
        word_count=meta_data.get("word_count"),
        file_size_bytes=meta_data.get("file_size_bytes"),
    )

    # Create document
    doc = ChonkDocument(
        id="mil-std-test",
        source_path=Path(data["document"]["filename"]),
        source_type="pdf",
        blocks=blocks,
        chunks=[],
        metadata=metadata,
    )

    # Create chunker
    print("[CHUNKING] Creating chunks...")
    print("  Parameters:")
    print("    - Max tokens: 512")
    print("    - Overlap: 50 tokens")
    print("    - Preserve tables: True")
    print("    - Group under headings: True")
    print()

    config = ChunkerConfig(
        max_tokens=512,
        overlap_tokens=50,
        preserve_tables=True,
        group_under_headings=True,
    )

    chunker = HierarchyChunker(config=config)

    # Chunk the blocks (not the document)
    chunks = chunker.chunk(blocks)

    print(f"[CHUNKING] Created {len(chunks)} chunks")
    print()

    # Show chunk stats
    chunk_sizes = [len(chunk.content.split()) for chunk in chunks]
    token_counts = [chunk.token_count for chunk in chunks]

    print("[CHUNK STATS]")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Avg words per chunk: {sum(chunk_sizes) / len(chunk_sizes):.1f}")
    print(f"  Min words: {min(chunk_sizes)}")
    print(f"  Max words: {max(chunk_sizes)}")
    print(f"  Avg tokens: {sum(token_counts) / len(token_counts):.1f}")
    print(f"  Min tokens: {min(token_counts)}")
    print(f"  Max tokens: {max(token_counts)}")
    print()

    # Save chunks as JSON
    chunks_output = Path("MIL-STD-extraction-chunks.json")

    chunks_data = {
        "document": {
            "filename": data["document"]["filename"],
            "metadata": meta_data,
            "chunking_params": {
                "max_tokens": 512,
                "overlap_tokens": 50,
                "preserve_headings": True,
                "preserve_tables": True,
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
                "system_metadata": chunk.system_metadata,
                "user_metadata": chunk.user_metadata.to_dict() if chunk.user_metadata else {},
            }
            for chunk in chunks
        ],
        "stats": {
            "total_chunks": len(chunks),
            "total_blocks": len(blocks),
            "avg_tokens": sum(token_counts) / len(chunks) if chunks else 0,
            "avg_words": sum(chunk_sizes) / len(chunks) if chunks else 0,
        }
    }

    with open(chunks_output, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] Chunks saved to: {chunks_output}")
    print(f"        File size: {chunks_output.stat().st_size / (1024*1024):.2f} MB")
    print()

    # Show first few chunks
    print("[FIRST 3 CHUNKS]")
    print("-" * 70)
    for i, chunk in enumerate(chunks[:3]):
        content_preview = chunk.content[:200].replace('\n', ' ')
        if len(chunk.content) > 200:
            content_preview += "..."

        page_info = ""
        if chunk.system_metadata.get("start_page") and chunk.system_metadata.get("end_page"):
            page_info = f"Pages: {chunk.system_metadata['start_page']}-{chunk.system_metadata['end_page']}"

        print(f"{i+1}. Chunk {chunk.id}")
        if page_info:
            print(f"   {page_info}")
        print(f"   Tokens: {chunk.token_count}")
        print(f"   Blocks: {len(chunk.block_ids)}")
        if chunk.hierarchy_path:
            print(f"   Path: {chunk.hierarchy_path}")
        print(f"   Content: {content_preview}")
        print()

    print("=" * 70)
    print("[COMPLETE] Chunking complete!")
    print(f"  - Input: {len(blocks)} blocks from {blocks_file.name}")
    print(f"  - Output: {len(chunks)} chunks to {chunks_output.name}")
    print("=" * 70)

if __name__ == "__main__":
    main()
