"""
Test automatic fix system on MIL-STD data.

Demonstrates:
1. Detect problems
2. Plan fixes
3. Preview changes
4. Apply fixes
5. Validate improvements
"""

import json
from pathlib import Path

from chonk.core.document import (
    ChonkDocument,
    DocumentMetadata,
    Chunk,
    ChunkMetadata,
    QualityScore,
)
from chonk.diagnostics import DiagnosticAnalyzer, FixOrchestrator


def load_chunks_from_json(json_path: Path, limit: int | None = None) -> list[Chunk]:
    """Load chunks from MIL-STD JSON file."""
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    chunks_data = data['chunks']
    if limit:
        chunks_data = chunks_data[:limit]

    chunks = []
    for chunk_data in chunks_data:
        quality = QualityScore(
            token_range=chunk_data['quality_details']['token_range'],
            sentence_complete=chunk_data['quality_details']['sentence_complete'],
            hierarchy_preserved=chunk_data['quality_details']['hierarchy_preserved'],
            table_integrity=chunk_data['quality_details']['table_integrity'],
            reference_complete=chunk_data['quality_details']['reference_complete'],
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

    return chunks


def main():
    """Test automatic fix system."""

    print("=" * 80)
    print("AUTOMATIC FIX SYSTEM TEST")
    print("=" * 80)

    # Load test data
    json_path = Path('MIL-STD-extraction-chunks-HIERARCHY.json')
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return

    # Load first 50 chunks (faster for testing)
    print("\nLoading MIL-STD chunks...")
    chunks = load_chunks_from_json(json_path, limit=50)
    print(f"Loaded {len(chunks)} chunks for testing")

    # Create mock document
    doc = ChonkDocument(
        id="test_doc",
        source_path=Path("MIL-STD-40051-2D.pdf"),
        source_type="pdf",
        blocks=[],
        chunks=chunks,
        metadata=DocumentMetadata(
            page_count=555,
            word_count=176294,
        ),
        loader_used="imported",
    )

    # Step 1: Detect problems
    print("\n" + "=" * 80)
    print("STEP 1: DETECT PROBLEMS")
    print("=" * 80)

    analyzer = DiagnosticAnalyzer()
    problems = analyzer.analyze_document(doc)
    stats_before = analyzer.get_statistics(problems)

    print(f"\nTotal problems detected: {stats_before['total_problems']}")
    print(f"Unique chunks with problems: {stats_before['unique_chunks_with_problems']}")
    print(f"\nBy Type:")
    for ptype, count in stats_before['by_type'].items():
        print(f"  {ptype}: {count}")
    print(f"\nBy Severity:")
    for severity, count in stats_before['by_severity'].items():
        print(f"  {severity}: {count}")

    # Step 2: Plan fixes
    print("\n" + "=" * 80)
    print("STEP 2: PLAN AUTOMATIC FIXES")
    print("=" * 80)

    orchestrator = FixOrchestrator()
    plan = orchestrator.plan_fixes(problems, chunks, auto_resolve_conflicts=True)

    print(f"\nFix plan generated:")
    print(f"  Total fix actions: {len(plan.actions)}")
    print(f"  Estimated improvement: {plan.estimated_improvement * 100:.1f}%")

    if plan.conflicts:
        print(f"\nConflicts detected:")
        for conflict in plan.conflicts:
            print(f"  - {conflict}")

    if plan.warnings:
        print(f"\nWarnings:")
        for warning in plan.warnings:
            print(f"  - {warning}")

    # Group actions by type
    by_type = {}
    for action in plan.actions:
        if action.action_type not in by_type:
            by_type[action.action_type] = []
        by_type[action.action_type].append(action)

    print(f"\nPlanned actions by type:")
    for action_type, actions in by_type.items():
        print(f"  {action_type}: {len(actions)} actions")

    # Show sample actions
    print(f"\nSample fix actions:")
    for action in plan.actions[:5]:
        print(f"\n  [{action.action_type.upper()}] (confidence: {action.confidence:.2f})")
        print(f"    {action.description}")
        print(f"    Affects chunks: {action.chunk_ids[:2]}{'...' if len(action.chunk_ids) > 2 else ''}")

    # Step 3: Preview changes
    print("\n" + "=" * 80)
    print("STEP 3: PREVIEW CHANGES")
    print("=" * 80)

    print(f"\nBefore fixes:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Chunks with problems: {stats_before['unique_chunks_with_problems']}")

    # Estimate after
    merge_count = len([a for a in plan.actions if a.action_type == "merge"])
    split_count = len([a for a in plan.actions if a.action_type == "split"])

    estimated_chunks = len(chunks) - merge_count + split_count
    print(f"\nAfter fixes (estimated):")
    print(f"  Total chunks: ~{estimated_chunks}")
    print(f"  Merge operations: {merge_count} (reduces chunk count)")
    print(f"  Split operations: {split_count} (increases chunk count)")

    # Step 4: Execute fixes
    print("\n" + "=" * 80)
    print("STEP 4: EXECUTE FIXES")
    print("=" * 80)

    result = orchestrator.execute_plan(plan, chunks, validate=True)

    print(f"\nExecution result:")
    print(f"  Success: {result.success}")
    print(f"  Chunks before: {result.chunks_before}")
    print(f"  Chunks after: {result.chunks_after}")
    print(f"  Actions applied: {len(result.actions_applied)}")

    if result.errors:
        print(f"\nErrors:")
        for error in result.errors:
            print(f"  - {error}")

    # Step 5: Validate improvements
    print("\n" + "=" * 80)
    print("STEP 5: VALIDATE IMPROVEMENTS")
    print("=" * 80)

    # Re-run diagnostics on fixed chunks
    doc.chunks = result.new_chunks
    problems_after = analyzer.analyze_document(doc)
    stats_after = analyzer.get_statistics(problems_after)

    print(f"\nProblems BEFORE fixes:")
    print(f"  Total: {stats_before['total_problems']}")
    print(f"  By severity: {stats_before['by_severity']}")

    print(f"\nProblems AFTER fixes:")
    print(f"  Total: {stats_after['total_problems']}")
    print(f"  By severity: {stats_after['by_severity']}")

    improvement = len(problems) - len(problems_after)
    reduction_rate = improvement / len(problems) * 100 if len(problems) > 0 else 0

    print(f"\nIMPROVEMENT:")
    print(f"  Problems fixed: {improvement}")
    print(f"  Reduction rate: {reduction_rate:.1f}%")

    # Show sample fixed chunks
    print(f"\nSample fixed chunks:")
    fixed_chunks = [c for c in result.new_chunks if c.is_modified]
    for chunk in fixed_chunks[:3]:
        print(f"\n  Chunk: {chunk.id}")
        print(f"    Modified: {chunk.is_modified}")
        print(f"    Tokens: {chunk.token_count}")
        if "merged_from" in chunk.system_metadata:
            print(f"    Merged from: {chunk.system_metadata['merged_from']}")
        if "split_from" in chunk.system_metadata:
            print(f"    Split from: {chunk.system_metadata['split_from']}")
        print(f"    Content preview: {chunk.content[:80]}...")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n[*] Analyzed {len(chunks)} chunks from MIL-STD-40051-2D")
    print(f"[*] Detected {len(problems)} problems")
    print(f"[*] Planned {len(plan.actions)} automatic fixes")
    print(f"[*] Applied {len(result.actions_applied)} fixes successfully")
    print(f"[*] Reduced problems from {len(problems)} to {len(problems_after)} ({reduction_rate:.1f}% improvement)")

    print(f"\n[*] This demonstrates the complete diagnostic-to-fix workflow:")
    print(f"    1. Detect problems using static analysis")
    print(f"    2. Plan fixes automatically (merge/split strategies)")
    print(f"    3. Preview changes before applying")
    print(f"    4. Execute fixes with conflict resolution")
    print(f"    5. Validate improvements with re-analysis")

    print("\n" + "=" * 80)
    print("READY FOR PRODUCTION")
    print("=" * 80)
    print("API endpoints available:")
    print("  POST /api/diagnostics/preview-fixes  - Preview fix plan")
    print("  POST /api/diagnostics/apply-fixes    - Apply fixes and update document")
    print("\nUI integration:")
    print("  1. 'RUN DIAGNOSTICS' button -> shows problems")
    print("  2. 'FIX PROBLEMS' button -> previews fixes")
    print("  3. 'APPLY FIXES' button -> executes fixes")
    print("  4. Show before/after metrics")
    print("=" * 80)


if __name__ == '__main__':
    main()
