"""
Test the new diagnostic system with MIL-STD data.

Demonstrates:
1. Static analysis (heuristics)
2. Question generation
3. Question-based testing (simulated - shows what questions would be generated)
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
from chonk.diagnostics import DiagnosticAnalyzer
from chonk.diagnostics.question_generator import QuestionGenerator


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


def test_static_analysis(chunks: list[Chunk]):
    """Test static problem detection."""
    print("=" * 80)
    print("STATIC ANALYSIS (Heuristic Detection)")
    print("=" * 80)

    # Create a mock document
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

    analyzer = DiagnosticAnalyzer()
    problems = analyzer.analyze_document(doc)
    stats = analyzer.get_statistics(problems)

    print(f"\nTotal problems detected: {stats['total_problems']}")
    print(f"Chunks with problems: {stats['unique_chunks_with_problems']}")
    print(f"\nBy Type:")
    for ptype, count in stats['by_type'].items():
        print(f"  {ptype}: {count}")
    print(f"\nBy Severity:")
    for severity, count in stats['by_severity'].items():
        print(f"  {severity}: {count}")

    # Show sample problems
    print(f"\nSample detected problems:")
    for problem in problems[:10]:
        print(f"\n[{problem.severity.value.upper()}] {problem.problem_type.value}")
        print(f"  Chunk: {problem.chunk_id}")
        print(f"  Issue: {problem.description}")
        if problem.suggested_fix:
            print(f"  Fix: {problem.suggested_fix}")

    return problems


def test_question_generation(chunks: list[Chunk]):
    """Test automatic question generation."""
    print("\n" + "=" * 80)
    print("QUESTION GENERATION (Automatic Test Suite)")
    print("=" * 80)

    generator = QuestionGenerator()
    questions = generator.generate_all_questions(chunks)

    # Group by type
    by_type = {}
    for q in questions:
        if q.test_type not in by_type:
            by_type[q.test_type] = []
        by_type[q.test_type].append(q)

    print(f"\nTotal questions generated: {len(questions)}")
    print(f"\nBy Test Type:")
    for test_type, type_questions in by_type.items():
        print(f"  {test_type}: {len(type_questions)} questions")

    # Show sample questions from each type
    print(f"\nSample Generated Questions:")
    for test_type, type_questions in by_type.items():
        print(f"\n[{test_type.upper()}]")
        for q in type_questions[:3]:  # Show 3 samples per type
            print(f"  Q: {q.question}")
            print(f"     Expected chunks: {q.expected_chunk_ids[:2]}{'...' if len(q.expected_chunk_ids) > 2 else ''}")
            if q.metadata:
                print(f"     Note: {q.metadata.get('note', '')}")

    return questions


def analyze_question_coverage(chunks: list[Chunk], questions: list):
    """Analyze which chunks are covered by questions."""
    print("\n" + "=" * 80)
    print("QUESTION COVERAGE ANALYSIS")
    print("=" * 80)

    # Count how many questions test each chunk
    chunk_question_count = {}
    for q in questions:
        for chunk_id in q.expected_chunk_ids:
            chunk_question_count[chunk_id] = chunk_question_count.get(chunk_id, 0) + 1

    covered_chunks = len(chunk_question_count)
    total_chunks = len(chunks)
    coverage_rate = covered_chunks / total_chunks * 100

    print(f"\nChunks covered by questions: {covered_chunks}/{total_chunks} ({coverage_rate:.1f}%)")

    # Find chunks with most questions (likely problematic)
    most_questioned = sorted(
        chunk_question_count.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    print(f"\nChunks with most test questions (likely problematic):")
    for chunk_id, count in most_questioned:
        chunk = next((c for c in chunks if c.id == chunk_id), None)
        if chunk:
            preview = chunk.content[:60].replace('\n', ' ')
            print(f"  {chunk_id}: {count} questions - \"{preview}...\"")


def main():
    """Run diagnostic tests on MIL-STD data."""
    json_path = Path('MIL-STD-extraction-chunks-HIERARCHY.json')

    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return

    # Load first 100 chunks for testing (faster)
    print("Loading MIL-STD chunks...")
    chunks = load_chunks_from_json(json_path, limit=100)
    print(f"Loaded {len(chunks)} chunks for testing\n")

    # Run tests
    problems = test_static_analysis(chunks)
    questions = test_question_generation(chunks)
    analyze_question_coverage(chunks, questions)

    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SYSTEM SUMMARY")
    print("=" * 80)
    print(f"\n[*] Analyzed {len(chunks)} chunks from MIL-STD-40051-2D")
    print(f"[*] Detected {len(problems)} problems using static heuristics")
    print(f"[*] Generated {len(questions)} diagnostic questions automatically")
    print(f"\n[*] This demonstrates the diagnostic-first approach:")
    print(f"    1. Static analysis finds obvious problems (size, fragments, references)")
    print(f"    2. Question generation creates retrieval tests automatically")
    print(f"    3. Running these questions against RAG will reveal chunking failures")
    print(f"\n[*] The 'aha moment': All chunks have quality score 1.0, but diagnostics")
    print(f"    reveal {len(problems)} real problems that would break retrieval!")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. API endpoint POST /api/diagnostics/analyze is now available")
    print("2. UI 'RUN DIAGNOSTICS' button can call this endpoint")
    print("3. Question-based testing can be run with include_questions=true")
    print("4. Results show WHERE chunking fails and WHY")
    print("=" * 80)


if __name__ == '__main__':
    main()
