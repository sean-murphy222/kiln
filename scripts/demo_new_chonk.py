"""
CHONK Demo - The New Way

This demonstrates CHONK's refocused vision:
- Extract blocks (commodity)
- Build hierarchy (CORE)
- Analyze structure (CORE)
- Compare strategies (CORE)
- Test before embed (KILLER FEATURE)

Run this to see the difference between the old way (flat chunking)
and the new way (hierarchy-first organization).
"""
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chonk.extraction.docling_extractor import DoclingExtractor
from chonk.hierarchy import HierarchyBuilder, HierarchyAnalyzer
from chonk.chunkers.hierarchy import HierarchyChunker
from chonk.chunkers.fixed import FixedChunker
from chonk.chunkers.base import ChunkerConfig
from chonk.comparison import StrategyComparer

def main():
    print("=" * 80)
    print("CHONK DEMO - Hierarchy-First Chunking")
    print("=" * 80)
    print()

    # Load the pre-extracted blocks
    blocks_file = Path("MIL-STD-extraction-blocks-HIERARCHY.json")

    if not blocks_file.exists():
        print("ERROR: Need to run extract_with_hierarchy.py first!")
        print("This demo uses pre-extracted blocks.")
        return

    print("[1] LOADING BLOCKS (Commodity Step)")
    print("-" * 80)

    with open(blocks_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct blocks
    from chonk.core.document import Block, BlockType, BoundingBox

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

    headings = [b for b in blocks if b.type == BlockType.HEADING]
    print(f"✓ Loaded {len(blocks)} blocks ({len(headings)} headings)")
    print()

    # =========================================================================
    # CORE FEATURE #1: Build Hierarchy Tree
    # =========================================================================

    print("[2] BUILD HIERARCHY TREE (CORE Feature)")
    print("-" * 80)
    print("This is what makes CHONK different - seeing document structure")
    print()

    tree = HierarchyBuilder.build_from_blocks(
        blocks=blocks,
        document_id="MIL-STD-40051-1D",
        metadata=data["document"]["metadata"]
    )

    stats = tree.get_statistics()
    print(f"✓ Built hierarchy tree:")
    print(f"  Total sections: {stats['total_nodes']}")
    print(f"  Max depth: {stats['max_depth']}")
    print(f"  Sections with content: {stats['nodes_with_content']}")
    print(f"  Average tokens per section: {stats['avg_tokens_per_node']:.1f}")
    print()

    # Show tree preview
    print("Tree preview (first 3 levels):")
    print()
    tree.print_tree(max_depth=2, show_content=False)
    print()

    # =========================================================================
    # CORE FEATURE #2: Analyze Structure Quality
    # =========================================================================

    print("[3] ANALYZE STRUCTURE (CORE Feature)")
    print("-" * 80)
    print("Check for issues before chunking")
    print()

    analysis = HierarchyAnalyzer.analyze(tree)

    print(f"Quality Score: {analysis.quality_score:.2f}/1.0")
    print()

    if analysis.issues:
        print(f"Issues found: {len(analysis.issues)}")
        for issue in analysis.issues[:5]:
            print(f"  [{issue.severity.upper()}] {issue.message}")
        if len(analysis.issues) > 5:
            print(f"  ... and {len(analysis.issues) - 5} more")
    else:
        print("✓ No issues found!")

    print()

    if analysis.recommendations:
        print("Recommendations:")
        for rec in analysis.recommendations:
            print(f"  {rec}")

    print()

    # =========================================================================
    # CORE FEATURE #3: Compare Chunking Strategies
    # =========================================================================

    print("[4] COMPARE STRATEGIES (CORE Feature)")
    print("-" * 80)
    print("See concrete differences before choosing")
    print()

    # Strategy 1: Hierarchical (NEW WAY)
    hierarchical_config = ChunkerConfig(
        max_tokens=512,
        overlap_tokens=50,
        preserve_tables=True,
        group_under_headings=True,
    )
    hierarchical_chunker = HierarchyChunker(config=hierarchical_config)

    # Strategy 2: Fixed (OLD WAY)
    fixed_config = ChunkerConfig(
        max_tokens=512,
        overlap_tokens=50,
        preserve_tables=True,
        group_under_headings=False,  # DOESN'T respect structure
    )
    fixed_chunker = FixedChunker(config=fixed_config)

    # Compare
    comparison = StrategyComparer.compare(
        blocks=blocks,
        chunkers=[
            ("Hierarchical", hierarchical_chunker),
            ("Fixed", fixed_chunker),
        ]
    )

    print("COMPARISON RESULTS:")
    print()

    for result in comparison.strategies:
        m = result.metrics
        print(f"{result.strategy_name}:")
        print(f"  Chunks: {m.total_chunks}")
        print(f"  Avg tokens: {m.avg_tokens_per_chunk:.1f}")
        print(f"  Quality score: {m.avg_quality_score:.3f}")
        print(f"  Hierarchy preservation: {m.hierarchy_preservation:.1%}")
        print(f"  Chunks with context: {m.chunks_with_context}/{m.total_chunks}")
        print()

    print(comparison.get_recommendation())
    print()

    # =========================================================================
    # KILLER FEATURE: Test Queries Before Embedding
    # =========================================================================

    print("[5] TEST QUERIES (KILLER Feature)")
    print("-" * 80)
    print("See which strategy retrieves better results")
    print()

    test_queries = [
        "What are the maintenance work package requirements?",
        "How should safety information be documented?",
        "What is in the FOREWORD?",
    ]

    print("Test queries:")
    for i, q in enumerate(test_queries, 1):
        print(f"  {i}. {q}")
    print()

    print("Running retrieval tests...")
    print("(This uses embeddings to test actual retrieval quality)")
    print()

    try:
        query_comparison = StrategyComparer.compare_with_queries(
            blocks=blocks,
            chunkers=[
                ("Hierarchical", hierarchical_chunker),
                ("Fixed", fixed_chunker),
            ],
            test_queries=test_queries
        )

        print("QUERY RESULTS:")
        print()

        for strategy_name, results in query_comparison["query_results"].items():
            print(f"{strategy_name} Strategy:")
            for i, result in enumerate(results, 1):
                print(f"  Query {i}: Retrieved {result['retrieved_chunks']} chunks")
                print(f"          Top score: {result['top_score']:.3f}")
            print()

        print(query_comparison["recommendation"])

    except Exception as e:
        print(f"⚠️  Query testing requires embedding model")
        print(f"   Install with: pip install sentence-transformers")
        print(f"   Error: {e}")

    print()

    # =========================================================================
    # EXPORT
    # =========================================================================

    print("[6] EXPORT")
    print("-" * 80)

    best = comparison.best_strategy
    if best:
        output_file = Path(f"demo_output_{best.strategy_name.lower()}.json")

        export_data = {
            "document": data["document"],
            "strategy": best.strategy_name,
            "metrics": best.metrics.to_dict(),
            "chunks": [
                {
                    "id": chunk.id,
                    "content": chunk.content,
                    "hierarchy_path": chunk.hierarchy_path,
                    "token_count": chunk.token_count,
                    "quality_score": chunk.quality.overall if chunk.quality else 0,
                }
                for chunk in best.chunks[:10]  # First 10 for demo
            ],
            "total_chunks": len(best.chunks),
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Exported {len(best.chunks)} chunks to {output_file}")
        print(f"  Strategy: {best.strategy_name}")
        print(f"  Quality: {best.metrics.avg_quality_score:.3f}/1.0")

    print()

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("=" * 80)
    print("SUMMARY - The New CHONK")
    print("=" * 80)
    print()
    print("What we just did:")
    print()
    print("1. ✅ Loaded blocks (commodity - anyone can do this)")
    print("2. 🌟 Built hierarchy tree (CORE - see document structure)")
    print("3. 🌟 Analyzed quality (CORE - find issues before chunking)")
    print("4. 🌟 Compared strategies (CORE - see concrete differences)")
    print("5. 💎 Tested queries (KILLER - validate before embedding)")
    print("6. ✅ Exported best strategy (with confidence)")
    print()
    print("Key Difference:")
    print("  OLD WAY: Extract → Chunk → Hope it works")
    print("  NEW WAY: Extract → Organize → Test → Refine → Export")
    print()
    print("CHONK is not about extraction. It's about organization.")
    print("=" * 80)

if __name__ == "__main__":
    main()
