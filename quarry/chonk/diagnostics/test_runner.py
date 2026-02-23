"""
Test runner for question-based chunk diagnostics.

Executes generated questions against retrieval system and analyzes failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chonk.diagnostics.question_generator import GeneratedQuestion, QuestionGenerator
from chonk.testing import RetrievalTester
from chonk.core.document import Chunk, ChonkDocument


@dataclass
class QuestionTestResult:
    """Result of testing a single generated question."""

    question: GeneratedQuestion
    retrieved_chunk_ids: list[str]
    retrieved_scores: list[float]
    status: str  # "pass", "partial", "fail"
    rank_of_expected: list[int] | None = None  # Rank of expected chunks in results
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question.question,
            "test_type": self.question.test_type,
            "expected_chunks": self.question.expected_chunk_ids,
            "retrieved_chunks": self.retrieved_chunk_ids,
            "retrieved_scores": self.retrieved_scores,
            "status": self.status,
            "rank_of_expected": self.rank_of_expected,
            "source_chunk": self.question.source_chunk_id,
            "metadata": self.metadata,
        }


@dataclass
class DiagnosticTestReport:
    """Complete diagnostic test report for a document."""

    document_id: str
    total_tests: int
    passed: int
    partial: int
    failed: int
    results_by_type: dict[str, dict[str, int]]
    worst_chunks: list[dict[str, Any]]
    failed_tests: list[QuestionTestResult]
    all_results: list[QuestionTestResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "summary": {
                "total_tests": self.total_tests,
                "passed": self.passed,
                "partial": self.partial,
                "failed": self.failed,
                "pass_rate": self.passed / self.total_tests if self.total_tests > 0 else 0,
            },
            "by_test_type": self.results_by_type,
            "worst_chunks": self.worst_chunks,
            "failed_tests": [r.to_dict() for r in self.failed_tests[:20]],  # Limit output
        }


class QuestionTestRunner:
    """Run generated questions against retrieval system for diagnostic testing."""

    def __init__(self, retrieval_tester: RetrievalTester):
        self.tester = retrieval_tester
        self.question_generator = QuestionGenerator()

    def run_diagnostic_tests(
        self,
        document: ChonkDocument,
        top_k: int = 5,
    ) -> DiagnosticTestReport:
        """
        Run complete diagnostic test suite on a document.

        1. Generate questions from chunks
        2. Execute each question against retrieval
        3. Analyze which questions failed
        4. Identify worst-performing chunks
        """

        # Generate questions
        questions = self.question_generator.generate_all_questions(document.chunks)

        # Index chunks for retrieval
        if not self.tester.is_indexed:
            self.tester.index_chunks(document.chunks)

        # Run each question
        results = []
        for question in questions:
            result = self._test_question(question, top_k)
            results.append(result)

        # Analyze results
        report = self._analyze_results(document.id, results)

        return report

    def _test_question(
        self,
        question: GeneratedQuestion,
        top_k: int,
    ) -> QuestionTestResult:
        """Test a single generated question against retrieval."""

        # Search for chunks
        search_results = self.tester.search(question.question, top_k)

        retrieved_ids = [r.chunk_id for r in search_results]
        retrieved_scores = [r.score for r in search_results]

        # Check if expected chunks were retrieved
        expected = set(question.expected_chunk_ids)
        actual = set(retrieved_ids)

        # Determine status
        if expected.issubset(actual):
            status = "pass"
        elif expected.intersection(actual):
            status = "partial"
        else:
            status = "fail"

        # Find rank of expected chunks
        rank_of_expected = []
        for expected_id in question.expected_chunk_ids:
            try:
                rank = retrieved_ids.index(expected_id) + 1  # 1-indexed
                rank_of_expected.append(rank)
            except ValueError:
                rank_of_expected.append(None)  # Not in top-k

        return QuestionTestResult(
            question=question,
            retrieved_chunk_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            status=status,
            rank_of_expected=rank_of_expected,
            metadata={
                "expected_found": len(expected.intersection(actual)),
                "expected_total": len(expected),
            },
        )

    def _analyze_results(
        self,
        document_id: str,
        results: list[QuestionTestResult],
    ) -> DiagnosticTestReport:
        """Analyze test results and generate report."""

        total = len(results)
        passed = sum(1 for r in results if r.status == "pass")
        partial = sum(1 for r in results if r.status == "partial")
        failed = sum(1 for r in results if r.status == "fail")

        # Group by test type
        results_by_type = {}
        for result in results:
            test_type = result.question.test_type
            if test_type not in results_by_type:
                results_by_type[test_type] = {"passed": 0, "partial": 0, "failed": 0}

            results_by_type[test_type][result.status] += 1

        # Find worst-performing chunks
        chunk_failures = {}
        for result in results:
            if result.status in ["partial", "fail"]:
                chunk_id = result.question.source_chunk_id
                chunk_failures[chunk_id] = chunk_failures.get(chunk_id, 0) + 1

        worst_chunks = sorted(
            [{"chunk_id": cid, "failure_count": count} for cid, count in chunk_failures.items()],
            key=lambda x: x["failure_count"],
            reverse=True,
        )[:10]

        # Get failed tests
        failed_tests = [r for r in results if r.status == "fail"]

        return DiagnosticTestReport(
            document_id=document_id,
            total_tests=total,
            passed=passed,
            partial=partial,
            failed=failed,
            results_by_type=results_by_type,
            worst_chunks=worst_chunks,
            failed_tests=failed_tests,
            all_results=results,
        )
