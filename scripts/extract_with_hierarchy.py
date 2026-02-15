"""
Re-extract MIL-STD PDF with corrected hierarchy mapping.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

# Force GPU usage
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

# Force UTF-8 output for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chonk.extraction.docling_extractor import DoclingExtractor

def main():
    pdf_path = Path(r"C:\Users\Sean Murphy\OneDrive\Desktop\CHONK\MIL-STD-40051-2D Change 1.pdf")

    if not pdf_path.exists():
        print(f"ERROR: PDF not found at {pdf_path}")
        return

    print("=" * 70)
    print("RE-EXTRACTING WITH HIERARCHY FIX")
    print("=" * 70)
    print(f"[FILE] {pdf_path.name}")
    print(f"[SIZE] {pdf_path.stat().st_size / (1024*1024):.2f} MB")
    print()

    # Create extractor
    extractor = DoclingExtractor()

    if not extractor.is_available():
        print("[ERROR] Docling not installed!")
        return

    print("[PROCESSING] Extracting with GPU acceleration...")
    print()

    # Extract
    result = extractor.extract(pdf_path)

    # Print results
    print("=" * 70)
    print("[COMPLETE] EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Tier used: {result.tier_used.value}")
    print(f"Blocks extracted: {len(result.blocks)}")
    print()

    # Metadata
    meta = result.metadata
    print("[METADATA]")
    print(f"  Pages: {meta.page_count or 'N/A'}")
    print(f"  Word count: {meta.word_count or 'N/A'}")
    print()

    # Block type breakdown
    block_types = Counter(b.type.value for b in result.blocks)
    print("[BLOCK TYPES]")
    for block_type, count in block_types.most_common():
        print(f"  {block_type}: {count}")
    print()

    # Heading level breakdown
    heading_levels = Counter(
        b.heading_level for b in result.blocks
        if b.type.value == "heading" and b.heading_level
    )
    if heading_levels:
        print("[HEADING LEVELS]")
        for level, count in sorted(heading_levels.items()):
            print(f"  Level {level}: {count}")
        print()

    # Show first few headings
    headings = [b for b in result.blocks if b.type.value == "heading"][:10]
    if headings:
        print("[FIRST 10 HEADINGS]")
        print("-" * 70)
        for i, h in enumerate(headings, 1):
            content = h.content[:80].replace('\n', ' ')
            if len(h.content) > 80:
                content += "..."
            print(f"{i}. [Page {h.page}] Level {h.heading_level}")
            print(f"   {content}")
        print()

    # Save blocks as JSON
    output_file = Path("MIL-STD-extraction-blocks-HIERARCHY.json")

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
            "heading_levels": dict(heading_levels) if heading_levels else {},
        }
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(blocks_data, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] Blocks saved to: {output_file}")
    print(f"        File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
    print()

    # Compare with old extraction
    old_file = Path("MIL-STD-extraction-blocks.json")
    if old_file.exists():
        with open(old_file, "r", encoding="utf-8") as f:
            old_data = json.load(f)

        old_types = Counter(b["type"] for b in old_data["blocks"])

        print("=" * 70)
        print("[COMPARISON] Old vs New Extraction")
        print("=" * 70)
        print(f"{'Type':<15} {'Old Count':<12} {'New Count':<12} {'Change'}")
        print("-" * 70)

        all_types = set(old_types.keys()) | set(block_types.keys())
        for btype in sorted(all_types):
            old_count = old_types.get(btype, 0)
            new_count = block_types.get(btype, 0)
            change = new_count - old_count
            change_str = f"+{change}" if change > 0 else str(change)
            print(f"{btype:<15} {old_count:<12} {new_count:<12} {change_str}")
        print()

    print("=" * 70)
    print("[SUCCESS] Hierarchy extraction complete!")
    print(f"  - {len(result.blocks)} blocks with proper heading structure")
    print(f"  - {block_types.get('heading', 0)} headings properly identified")
    print("=" * 70)

if __name__ == "__main__":
    main()
