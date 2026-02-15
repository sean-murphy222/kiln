"""
Test docling extraction with GPU on MIL-STD PDF.
"""
import os
import sys
from pathlib import Path

# Force UTF-8 output for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Force GPU usage - override the architecture check
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chonk.extraction.docling_extractor import DoclingExtractor
from chonk.extraction.strategy import ExtractionTier

def main():
    pdf_path = Path(r"C:\Users\Sean Murphy\OneDrive\Desktop\CHONK\MIL-STD-40051-2D Change 1.pdf")

    if not pdf_path.exists():
        print(f"ERROR: PDF not found at {pdf_path}")
        return

    print(f"[FILE] Extracting: {pdf_path.name}")
    print(f"[SIZE] File size: {pdf_path.stat().st_size / (1024*1024):.2f} MB")
    print()

    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[GPU] Device: {torch.cuda.get_device_name(0)}")
            print(f"      Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            print("[WARN] CUDA not available, using CPU")
    except ImportError:
        print("[WARN] PyTorch not installed")
    print()

    # Create extractor
    extractor = DoclingExtractor()

    if not extractor.is_available():
        print("[ERROR] Docling not installed!")
        print("Install with: pip install chonk[enhanced]")
        return

    print(f"[OK] Docling extractor available (tier: {extractor.tier.value})")
    print()
    print("[PROCESSING] Extracting... (this may take a minute)")
    print()

    # Extract
    result = extractor.extract(pdf_path)

    # Print results
    print("=" * 70)
    print(f"[COMPLETE] EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Tier used: {result.tier_used.value}")
    print(f"Blocks extracted: {len(result.blocks)}")
    print()

    # Metadata
    meta = result.metadata
    print("[METADATA]")
    print(f"  Title: {meta.title or 'N/A'}")
    print(f"  Author: {meta.author or 'N/A'}")
    print(f"  Pages: {meta.page_count or 'N/A'}")
    print(f"  Word count: {meta.word_count or 'N/A'}")
    print()

    # Block type breakdown
    from collections import Counter
    block_types = Counter(b.type.value for b in result.blocks)
    print("[BLOCK TYPES]")
    for block_type, count in block_types.most_common():
        print(f"  {block_type}: {count}")
    print()

    # Warnings
    if result.warnings:
        print("[WARNINGS]")
        for warning in result.warnings:
            print(f"  - {warning}")
        print()

    # Extraction info
    if result.extraction_info:
        print("[EXTRACTION INFO]")
        for key, value in result.extraction_info.items():
            print(f"  {key}: {value}")
        print()

    # Show first few blocks
    print("[FIRST 5 BLOCKS]")
    print("-" * 70)
    for i, block in enumerate(result.blocks[:5]):
        content_preview = block.content[:100].replace('\n', ' ')
        if len(block.content) > 100:
            content_preview += "..."
        print(f"{i+1}. [{block.type.value}] {content_preview}")
        if block.heading_level:
            print(f"   (Heading level: {block.heading_level})")
        if block.page:
            print(f"   (Page: {block.page})")
        print()

    print("=" * 70)
    print(f"[DONE] Extracted {len(result.blocks)} blocks from {meta.page_count or '?'} pages")
    print()

    # Save blocks as JSON
    import json
    from dataclasses import asdict

    output_file = Path("MIL-STD-extraction-blocks.json")

    # Convert blocks to dict format
    blocks_data = {
        "document": {
            "filename": pdf_path.name,
            "metadata": {
                "title": meta.title,
                "author": meta.author,
                "page_count": meta.page_count,
                "word_count": meta.word_count,
                "file_size_bytes": meta.file_size_bytes,
            },
            "extraction_info": result.extraction_info,
        },
        "blocks": [
            {
                "id": block.id,
                "type": block.type.value,
                "content": block.content,
                "page": block.page,
                "heading_level": block.heading_level,
                "bbox": {
                    "x1": block.bbox.x1,
                    "y1": block.bbox.y1,
                    "x2": block.bbox.x2,
                    "y2": block.bbox.y2,
                    "page": block.bbox.page,
                } if block.bbox else None,
                "metadata": block.metadata,
            }
            for block in result.blocks
        ],
        "stats": {
            "total_blocks": len(result.blocks),
            "block_types": dict(block_types),
        }
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(blocks_data, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] Blocks saved to: {output_file}")
    print(f"        File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
    print()

    # Now run chunker to create chunks
    print("=" * 70)
    print("[CHUNKING] Creating chunks from blocks...")
    print("=" * 70)

    from chonk.chunkers.hierarchy import HierarchyChunker
    from chonk.core.document import ChonkDocument

    # Create a ChonkDocument object
    doc = ChonkDocument(
        id="mil-std-test",
        title=meta.title or pdf_path.name,
        blocks=result.blocks,
        chunks=[],  # Will be populated by chunker
        metadata=meta,
    )

    # Create chunker with parameters
    chunker = HierarchyChunker(
        max_tokens=512,      # Max tokens per chunk
        overlap_tokens=50,   # Overlap between chunks
        preserve_headings=True,
        preserve_tables=True,
    )

    # Chunk the document
    chunks = chunker.chunk(doc)

    print(f"[CHUNKING] Created {len(chunks)} chunks")
    print()

    # Show chunk stats
    chunk_sizes = [len(chunk.content.split()) for chunk in chunks]
    print("[CHUNK STATS]")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Avg words per chunk: {sum(chunk_sizes) / len(chunk_sizes):.1f}")
    print(f"  Min words: {min(chunk_sizes)}")
    print(f"  Max words: {max(chunk_sizes)}")
    print()

    # Save chunks as JSON
    chunks_output = Path("MIL-STD-extraction-chunks.json")

    chunks_data = {
        "document": {
            "filename": pdf_path.name,
            "metadata": {
                "title": meta.title,
                "author": meta.author,
                "page_count": meta.page_count,
                "word_count": meta.word_count,
            },
            "chunking_params": {
                "max_tokens": 512,
                "overlap_tokens": 50,
                "preserve_headings": True,
                "preserve_tables": True,
            },
        },
        "chunks": [
            {
                "id": chunk.id,
                "content": chunk.content,
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
                "block_ids": chunk.block_ids,
                "token_count": chunk.token_count,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ],
        "stats": {
            "total_chunks": len(chunks),
            "avg_tokens": sum(c.token_count for c in chunks) / len(chunks) if chunks else 0,
        }
    }

    with open(chunks_output, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] Chunks saved to: {chunks_output}")
    print(f"        File size: {chunks_output.stat().st_size / (1024*1024):.2f} MB")
    print()
    print("=" * 70)
    print("[COMPLETE] Extraction and chunking complete!")
    print(f"  - {len(result.blocks)} blocks saved to {output_file.name}")
    print(f"  - {len(chunks)} chunks saved to {chunks_output.name}")
    print("=" * 70)

if __name__ == "__main__":
    main()
