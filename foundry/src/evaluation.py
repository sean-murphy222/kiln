"""Competency-based evaluation system for Foundry.

Tests trained LoRA models against held-out test sets and reports
results in plain language that domain experts can understand.
This module intentionally avoids ML jargon (loss, perplexity, F1)
in favor of SME-friendly language (X/10 correct, needs improvement).

The inference step is abstracted behind the ModelInference protocol
so tests can use MockInference without requiring a real model.

Example::

    runner = EvaluationRunner()
    cases = runner.load_test_cases(Path("test_set.jsonl"))
    report = runner.run_evaluation(
        model=my_model,
        test_cases=cases,
        competency_names={"comp_proc": "Procedural Comprehension"},
        model_name="maintenance-lora-v1",
        discipline_id="disc_maint",
    )
    print(report.plain_language_summary)
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class EvaluationError(Exception):
    """Raised for evaluation workflow errors."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ResponseQuality(str, Enum):
    """Quality rating for a single model response.

    Attributes:
        CORRECT: Response closely matches expected answer.
        PARTIALLY_CORRECT: Response captures some key information.
        INCORRECT: Response does not match expected answer.
        NO_RESPONSE: Model produced empty or whitespace-only output.
    """

    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"
    NO_RESPONSE = "no_response"


class CompetencyRating(str, Enum):
    """Overall rating for a competency area.

    Attributes:
        STRONG: High percentage of correct answers.
        ADEQUATE: Acceptable but could improve.
        NEEDS_IMPROVEMENT: Below acceptable threshold.
        WEAK: Significantly below acceptable threshold.
        UNTESTED: No test cases available.
    """

    STRONG = "strong"
    ADEQUATE = "adequate"
    NEEDS_IMPROVEMENT = "needs_improvement"
    WEAK = "weak"
    UNTESTED = "untested"


class EvaluationStatus(str, Enum):
    """Status of an evaluation run.

    Attributes:
        PENDING: Not yet started.
        RUNNING: Currently in progress.
        COMPLETED: Finished successfully.
        FAILED: Finished with an error.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelInference(Protocol):
    """Protocol for model inference abstraction.

    Any class implementing generate() can be used for evaluation,
    enabling mock inference in tests without a real model.
    """

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate a response for the given prompt.

        Args:
            prompt: The input prompt text.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text response.
        """
        ...


@runtime_checkable
class SimilarityScorer(Protocol):
    """Protocol for computing similarity between reference and candidate text."""

    def score(self, reference: str, candidate: str) -> float:
        """Compute similarity score between reference and candidate.

        Args:
            reference: The expected/ideal answer.
            candidate: The model-generated answer.

        Returns:
            Float between 0.0 (no match) and 1.0 (perfect match).
        """
        ...


# ---------------------------------------------------------------------------
# Inference implementations
# ---------------------------------------------------------------------------


class MockInference:
    """Mock model inference for testing.

    Returns pre-configured responses for known prompts,
    or a default response for unknown prompts.

    Args:
        responses: Dict mapping prompt text to response text.
        default_response: Response for prompts not in the dict.

    Example::

        model = MockInference(
            responses={"What is X?": "X is Y."},
            default_response="I don't know.",
        )
        answer = model.generate("What is X?")  # "X is Y."
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default_response: str = "",
    ) -> None:
        self._responses = responses or {}
        self._default = default_response

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Return pre-configured response or default.

        Args:
            prompt: The input prompt text.
            max_tokens: Ignored for mock inference.

        Returns:
            Pre-configured or default response string.
        """
        return self._responses.get(prompt, self._default)


# ---------------------------------------------------------------------------
# Similarity scorer implementations
# ---------------------------------------------------------------------------


class KeywordSimilarityScorer:
    """Bag-of-words Jaccard similarity scorer.

    Tokenizes both texts, normalizes to lowercase, strips punctuation,
    and computes Jaccard similarity: |intersection| / |union|.

    Example::

        scorer = KeywordSimilarityScorer()
        score = scorer.score("remove filter install new", "remove old filter install new")
    """

    def score(self, reference: str, candidate: str) -> float:
        """Compute Jaccard keyword similarity.

        Args:
            reference: The expected/ideal answer text.
            candidate: The model-generated answer text.

        Returns:
            Float between 0.0 and 1.0.
        """
        ref_tokens = self._tokenize(reference)
        cand_tokens = self._tokenize(candidate)

        if not ref_tokens or not cand_tokens:
            return 0.0

        intersection = ref_tokens & cand_tokens
        union = ref_tokens | cand_tokens
        return len(intersection) / len(union)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Normalize and tokenize text into a set of lowercase words.

        Args:
            text: Input text to tokenize.

        Returns:
            Set of normalized word tokens.
        """
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        tokens = cleaned.split()
        return set(tokens)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    """A single test case from a held-out test set.

    Attributes:
        example_id: Unique identifier from Forge.
        question: The test question/instruction.
        expected_answer: The ideal/reference answer.
        competency_id: Which competency this tests.
        discipline_id: Which discipline this belongs to.
    """

    __test__ = False  # Prevent pytest collection

    example_id: str
    question: str
    expected_answer: str
    competency_id: str
    discipline_id: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with all fields.
        """
        return {
            "example_id": self.example_id,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "competency_id": self.competency_id,
            "discipline_id": self.discipline_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestCase:
        """Deserialize from dictionary.

        Args:
            data: Dict with test case fields.

        Returns:
            TestCase instance.
        """
        return cls(
            example_id=data["example_id"],
            question=data["question"],
            expected_answer=data["expected_answer"],
            competency_id=data["competency_id"],
            discipline_id=data["discipline_id"],
        )


@dataclass
class TestResult:
    """Result of evaluating a single test case.

    Attributes:
        test_case: The original test case.
        model_response: What the model actually generated.
        similarity_score: Similarity between expected and actual (0-1).
        quality: Categorized quality rating.
        response_time_ms: How long inference took in milliseconds.
    """

    __test__ = False  # Prevent pytest collection

    test_case: TestCase
    model_response: str
    similarity_score: float
    quality: ResponseQuality
    response_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with all fields including nested test_case.
        """
        return {
            "test_case": self.test_case.to_dict(),
            "model_response": self.model_response,
            "similarity_score": self.similarity_score,
            "quality": self.quality.value,
            "response_time_ms": self.response_time_ms,
        }


@dataclass
class CompetencyScore:
    """Aggregated score for a single competency area.

    Attributes:
        competency_id: Unique competency identifier.
        competency_name: Human-readable competency name.
        total_cases: Total test cases for this competency.
        correct: Number of correct responses.
        partially_correct: Number of partially correct responses.
        incorrect: Number of incorrect responses.
        no_response: Number of empty/no responses.
        rating: Overall competency rating.
        summary: Plain language summary (e.g., "7/10 correct").
    """

    competency_id: str
    competency_name: str
    total_cases: int
    correct: int
    partially_correct: int
    incorrect: int
    no_response: int
    rating: CompetencyRating
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with all fields.
        """
        return {
            "competency_id": self.competency_id,
            "competency_name": self.competency_name,
            "total_cases": self.total_cases,
            "correct": self.correct,
            "partially_correct": self.partially_correct,
            "incorrect": self.incorrect,
            "no_response": self.no_response,
            "rating": self.rating.value,
            "summary": self.summary,
        }


@dataclass
class EvaluationConfig:
    """Configuration for evaluation thresholds.

    Attributes:
        correct_threshold: Similarity score >= this is CORRECT (default 0.7).
        partial_threshold: Similarity score >= this is PARTIALLY_CORRECT (default 0.4).
        strong_threshold: Correct % >= this rates STRONG (default 0.8).
        adequate_threshold: Correct % >= this rates ADEQUATE (default 0.6).
        needs_improvement_threshold: Correct % >= this rates NEEDS_IMPROVEMENT (default 0.4).
        max_tokens: Maximum tokens for model generation (default 512).
    """

    correct_threshold: float = 0.7
    partial_threshold: float = 0.4
    strong_threshold: float = 0.8
    adequate_threshold: float = 0.6
    needs_improvement_threshold: float = 0.4
    max_tokens: int = 512

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with all config fields.
        """
        return {
            "correct_threshold": self.correct_threshold,
            "partial_threshold": self.partial_threshold,
            "strong_threshold": self.strong_threshold,
            "adequate_threshold": self.adequate_threshold,
            "needs_improvement_threshold": self.needs_improvement_threshold,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationConfig:
        """Deserialize from dictionary.

        Args:
            data: Dict with config fields.

        Returns:
            EvaluationConfig instance.
        """
        return cls(
            correct_threshold=data.get("correct_threshold", 0.7),
            partial_threshold=data.get("partial_threshold", 0.4),
            strong_threshold=data.get("strong_threshold", 0.8),
            adequate_threshold=data.get("adequate_threshold", 0.6),
            needs_improvement_threshold=data.get("needs_improvement_threshold", 0.4),
            max_tokens=data.get("max_tokens", 512),
        )


@dataclass
class ModelComparison:
    """Comparison result between two models.

    Attributes:
        model_a_name: Name of the first model (typically baseline).
        model_b_name: Name of the second model (typically finetuned).
        model_a_scores: Competency scores for model A.
        model_b_scores: Competency scores for model B.
        improvements: Competency IDs where B outperforms A.
        regressions: Competency IDs where B underperforms A.
        summary: Plain language comparison summary.
    """

    model_a_name: str
    model_b_name: str
    model_a_scores: dict[str, CompetencyScore]
    model_b_scores: dict[str, CompetencyScore]
    improvements: list[str]
    regressions: list[str]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with comparison data.
        """
        return {
            "model_a_name": self.model_a_name,
            "model_b_name": self.model_b_name,
            "model_a_scores": {k: v.to_dict() for k, v in self.model_a_scores.items()},
            "model_b_scores": {k: v.to_dict() for k, v in self.model_b_scores.items()},
            "improvements": self.improvements,
            "regressions": self.regressions,
            "summary": self.summary,
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report for a model on a discipline.

    Attributes:
        run_id: Unique identifier for this evaluation run.
        model_name: Name of the evaluated model.
        discipline_id: Discipline being evaluated.
        status: Current status of the evaluation.
        competency_scores: Per-competency aggregated scores.
        test_results: Individual test case results.
        total_cases: Total number of test cases.
        overall_correct: Total correct responses.
        overall_accuracy: Fraction of correct responses (0-1).
        overall_rating: Overall competency rating.
        plain_language_summary: SME-friendly top-level summary.
        weak_areas: Competency names needing improvement.
        strong_areas: Competency names performing well.
        started_at: When the evaluation started.
        completed_at: When the evaluation finished (None if not done).
    """

    run_id: str
    model_name: str
    discipline_id: str
    status: EvaluationStatus
    competency_scores: dict[str, CompetencyScore]
    test_results: list[TestResult]
    total_cases: int
    overall_correct: int
    overall_accuracy: float
    overall_rating: CompetencyRating
    plain_language_summary: str
    weak_areas: list[str]
    strong_areas: list[str]
    started_at: datetime
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with full report data.
        """
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "discipline_id": self.discipline_id,
            "status": self.status.value,
            "competency_scores": {k: v.to_dict() for k, v in self.competency_scores.items()},
            "test_results": [r.to_dict() for r in self.test_results],
            "total_cases": self.total_cases,
            "overall_correct": self.overall_correct,
            "overall_accuracy": self.overall_accuracy,
            "overall_rating": self.overall_rating.value,
            "plain_language_summary": self.plain_language_summary,
            "weak_areas": self.weak_areas,
            "strong_areas": self.strong_areas,
            "started_at": self.started_at.isoformat(),
            "completed_at": (self.completed_at.isoformat() if self.completed_at else None),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationReport:
        """Deserialize from dictionary.

        Args:
            data: Dict with report fields.

        Returns:
            EvaluationReport instance.
        """
        competency_scores = _parse_competency_scores(data.get("competency_scores", {}))
        test_results = _parse_test_results(data.get("test_results", []))
        completed_at = data.get("completed_at")

        return cls(
            run_id=data["run_id"],
            model_name=data["model_name"],
            discipline_id=data["discipline_id"],
            status=EvaluationStatus(data["status"]),
            competency_scores=competency_scores,
            test_results=test_results,
            total_cases=data["total_cases"],
            overall_correct=data["overall_correct"],
            overall_accuracy=data["overall_accuracy"],
            overall_rating=CompetencyRating(data["overall_rating"]),
            plain_language_summary=data["plain_language_summary"],
            weak_areas=data.get("weak_areas", []),
            strong_areas=data.get("strong_areas", []),
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=(datetime.fromisoformat(completed_at) if completed_at else None),
        )


# ---------------------------------------------------------------------------
# Deserialization helpers (keep from_dict under 50 lines)
# ---------------------------------------------------------------------------


def _parse_competency_scores(
    raw: dict[str, Any],
) -> dict[str, CompetencyScore]:
    """Parse competency scores from serialized dict.

    Args:
        raw: Dict of competency_id -> serialized CompetencyScore.

    Returns:
        Dict of competency_id -> CompetencyScore instances.
    """
    result: dict[str, CompetencyScore] = {}
    for comp_id, score_data in raw.items():
        result[comp_id] = CompetencyScore(
            competency_id=score_data["competency_id"],
            competency_name=score_data["competency_name"],
            total_cases=score_data["total_cases"],
            correct=score_data["correct"],
            partially_correct=score_data["partially_correct"],
            incorrect=score_data["incorrect"],
            no_response=score_data["no_response"],
            rating=CompetencyRating(score_data["rating"]),
            summary=score_data["summary"],
        )
    return result


def _parse_test_results(raw: list[dict[str, Any]]) -> list[TestResult]:
    """Parse test results from serialized list.

    Args:
        raw: List of serialized TestResult dicts.

    Returns:
        List of TestResult instances.
    """
    results: list[TestResult] = []
    for item in raw:
        tc = TestCase.from_dict(item["test_case"])
        results.append(
            TestResult(
                test_case=tc,
                model_response=item["model_response"],
                similarity_score=item["similarity_score"],
                quality=ResponseQuality(item["quality"]),
                response_time_ms=item["response_time_ms"],
            )
        )
    return results


# ---------------------------------------------------------------------------
# Competency name resolver
# ---------------------------------------------------------------------------


class CompetencyNameResolver:
    """Resolves competency IDs to human-readable names.

    Falls back to the raw ID if no name mapping is available.

    Args:
        competency_names: Dict mapping competency IDs to display names.

    Example::

        resolver = CompetencyNameResolver({"comp_proc": "Procedural Comprehension"})
        name = resolver.resolve("comp_proc")  # "Procedural Comprehension"
    """

    def __init__(self, competency_names: dict[str, str]) -> None:
        self._names = competency_names

    def resolve(self, competency_id: str) -> str:
        """Resolve competency ID to display name.

        Args:
            competency_id: The competency identifier.

        Returns:
            Human-readable name, or the ID itself if not found.
        """
        return self._names.get(competency_id, competency_id)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


class EvaluationRunner:
    """Orchestrates competency-based evaluation of models.

    Runs test cases through a model, scores responses using
    keyword similarity, groups results by competency, and
    generates plain-language reports.

    Args:
        config: Evaluation thresholds and settings.

    Example::

        runner = EvaluationRunner()
        report = runner.run_evaluation(model, cases, names, "my-model", "disc_001")
    """

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self._config = config or EvaluationConfig()

    def run_evaluation(
        self,
        model: ModelInference,
        test_cases: list[TestCase],
        competency_names: dict[str, str],
        model_name: str,
        discipline_id: str,
        scorer: SimilarityScorer | None = None,
    ) -> EvaluationReport:
        """Run a full evaluation of a model against test cases.

        Args:
            model: Model inference implementation.
            test_cases: List of test cases to evaluate.
            competency_names: Map of competency_id to display name.
            model_name: Identifier for the model being evaluated.
            discipline_id: Discipline being evaluated.
            scorer: Optional custom similarity scorer.

        Returns:
            Complete EvaluationReport.

        Raises:
            EvaluationError: If no test cases provided.
        """
        if not test_cases:
            raise EvaluationError("No test cases provided for evaluation")

        started_at = datetime.now()
        run_id = f"eval_{uuid.uuid4().hex[:12]}"
        used_scorer = scorer or KeywordSimilarityScorer()

        results = self._run_all_cases(model, test_cases, used_scorer)
        comp_scores = self._compute_competency_scores(results, competency_names)

        return self._build_report(
            run_id,
            model_name,
            discipline_id,
            comp_scores,
            results,
            started_at,
        )

    def run_comparison(
        self,
        model_a: ModelInference,
        model_b: ModelInference,
        test_cases: list[TestCase],
        competency_names: dict[str, str],
        model_a_name: str,
        model_b_name: str,
        discipline_id: str,
    ) -> ModelComparison:
        """Compare two models on the same test set.

        Args:
            model_a: First model (typically baseline).
            model_b: Second model (typically finetuned).
            test_cases: Shared test cases.
            competency_names: Map of competency_id to display name.
            model_a_name: Display name for model A.
            model_b_name: Display name for model B.
            discipline_id: Discipline being evaluated.

        Returns:
            ModelComparison with improvements and regressions.
        """
        report_a = self.run_evaluation(
            model_a,
            test_cases,
            competency_names,
            model_a_name,
            discipline_id,
        )
        report_b = self.run_evaluation(
            model_b,
            test_cases,
            competency_names,
            model_b_name,
            discipline_id,
        )

        improvements, regressions = self._detect_changes(
            report_a.competency_scores,
            report_b.competency_scores,
        )

        comparison = ModelComparison(
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            model_a_scores=report_a.competency_scores,
            model_b_scores=report_b.competency_scores,
            improvements=improvements,
            regressions=regressions,
            summary="",
        )
        comparison.summary = self._generate_comparison_summary(comparison)
        return comparison

    def load_test_cases(self, jsonl_path: Path) -> list[TestCase]:
        """Load test cases from a Forge-exported JSONL file.

        Expects Alpaca format with metadata containing
        example_id, discipline_id, and competency_id.

        Args:
            jsonl_path: Path to the JSONL file.

        Returns:
            List of TestCase instances.

        Raises:
            EvaluationError: If file not found, empty, or malformed.
        """
        if not jsonl_path.exists():
            raise EvaluationError(f"Test set file not found: {jsonl_path}")

        cases = self._parse_jsonl_file(jsonl_path)

        if not cases:
            raise EvaluationError(f"No test cases loaded from {jsonl_path}")

        return cases

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _evaluate_single(
        self,
        model: ModelInference,
        test_case: TestCase,
        scorer: SimilarityScorer | None = None,
    ) -> TestResult:
        """Evaluate a single test case.

        Args:
            model: Model inference implementation.
            test_case: The test case to evaluate.
            scorer: Similarity scorer (defaults to KeywordSimilarityScorer).

        Returns:
            TestResult with quality classification.
        """
        used_scorer = scorer or KeywordSimilarityScorer()
        start = time.monotonic()
        response = model.generate(test_case.question, self._config.max_tokens)
        elapsed_ms = (time.monotonic() - start) * 1000

        if not response.strip():
            return TestResult(
                test_case=test_case,
                model_response=response,
                similarity_score=0.0,
                quality=ResponseQuality.NO_RESPONSE,
                response_time_ms=elapsed_ms,
            )

        sim = used_scorer.score(test_case.expected_answer, response)
        quality = self._classify_quality(sim)

        return TestResult(
            test_case=test_case,
            model_response=response,
            similarity_score=sim,
            quality=quality,
            response_time_ms=elapsed_ms,
        )

    def _classify_quality(self, similarity: float) -> ResponseQuality:
        """Classify a similarity score into a quality rating.

        Args:
            similarity: Similarity score between 0 and 1.

        Returns:
            ResponseQuality enum value.
        """
        if similarity >= self._config.correct_threshold:
            return ResponseQuality.CORRECT
        if similarity >= self._config.partial_threshold:
            return ResponseQuality.PARTIALLY_CORRECT
        return ResponseQuality.INCORRECT

    def _run_all_cases(
        self,
        model: ModelInference,
        test_cases: list[TestCase],
        scorer: SimilarityScorer,
    ) -> list[TestResult]:
        """Run all test cases through the model.

        Args:
            model: Model inference implementation.
            test_cases: List of test cases.
            scorer: Similarity scorer.

        Returns:
            List of TestResult instances.
        """
        return [self._evaluate_single(model, tc, scorer) for tc in test_cases]

    def _compute_competency_scores(
        self,
        results: list[TestResult],
        competency_names: dict[str, str],
    ) -> dict[str, CompetencyScore]:
        """Group results by competency and compute aggregate scores.

        Args:
            results: List of individual test results.
            competency_names: Map of competency_id to display name.

        Returns:
            Dict of competency_id to CompetencyScore.
        """
        resolver = CompetencyNameResolver(competency_names)
        grouped: dict[str, list[TestResult]] = {}

        for result in results:
            comp_id = result.test_case.competency_id
            grouped.setdefault(comp_id, []).append(result)

        scores: dict[str, CompetencyScore] = {}
        for comp_id, comp_results in grouped.items():
            scores[comp_id] = self._score_competency(
                comp_id,
                resolver.resolve(comp_id),
                comp_results,
            )
        return scores

    def _score_competency(
        self,
        comp_id: str,
        comp_name: str,
        results: list[TestResult],
    ) -> CompetencyScore:
        """Compute score for a single competency.

        Args:
            comp_id: Competency identifier.
            comp_name: Human-readable competency name.
            results: Test results for this competency.

        Returns:
            CompetencyScore with rating and summary.
        """
        counts = self._count_qualities(results)
        total = len(results)
        correct_pct = counts["correct"] / total if total > 0 else 0.0
        rating = self._compute_rating(correct_pct)
        summary = f"{counts['correct']}/{total} correct"

        return CompetencyScore(
            competency_id=comp_id,
            competency_name=comp_name,
            total_cases=total,
            correct=counts["correct"],
            partially_correct=counts["partially_correct"],
            incorrect=counts["incorrect"],
            no_response=counts["no_response"],
            rating=rating,
            summary=summary,
        )

    @staticmethod
    def _count_qualities(results: list[TestResult]) -> dict[str, int]:
        """Count occurrences of each quality level.

        Args:
            results: List of test results.

        Returns:
            Dict with counts for each ResponseQuality.
        """
        counts = {"correct": 0, "partially_correct": 0, "incorrect": 0, "no_response": 0}
        for r in results:
            counts[r.quality.value] += 1
        return counts

    def _compute_rating(self, correct_pct: float) -> CompetencyRating:
        """Compute competency rating from correct percentage.

        Args:
            correct_pct: Fraction of correct answers (0-1).

        Returns:
            CompetencyRating enum value.
        """
        if correct_pct >= self._config.strong_threshold:
            return CompetencyRating.STRONG
        if correct_pct >= self._config.adequate_threshold:
            return CompetencyRating.ADEQUATE
        if correct_pct >= self._config.needs_improvement_threshold:
            return CompetencyRating.NEEDS_IMPROVEMENT
        return CompetencyRating.WEAK

    def _build_report(
        self,
        run_id: str,
        model_name: str,
        discipline_id: str,
        competency_scores: dict[str, CompetencyScore],
        results: list[TestResult],
        started_at: datetime,
    ) -> EvaluationReport:
        """Assemble the final evaluation report.

        Args:
            run_id: Unique run identifier.
            model_name: Name of the evaluated model.
            discipline_id: Discipline identifier.
            competency_scores: Per-competency scores.
            results: All individual test results.
            started_at: When evaluation started.

        Returns:
            Complete EvaluationReport.
        """
        total = len(results)
        correct = sum(1 for r in results if r.quality == ResponseQuality.CORRECT)
        accuracy = correct / total if total > 0 else 0.0
        overall_rating = self._compute_rating(accuracy)

        weak = self._find_weak_areas(competency_scores)
        strong = self._find_strong_areas(competency_scores)

        report = EvaluationReport(
            run_id=run_id,
            model_name=model_name,
            discipline_id=discipline_id,
            status=EvaluationStatus.COMPLETED,
            competency_scores=competency_scores,
            test_results=results,
            total_cases=total,
            overall_correct=correct,
            overall_accuracy=accuracy,
            overall_rating=overall_rating,
            plain_language_summary="",
            weak_areas=weak,
            strong_areas=strong,
            started_at=started_at,
            completed_at=datetime.now(),
        )
        report.plain_language_summary = self._generate_summary(report)
        return report

    @staticmethod
    def _find_weak_areas(
        scores: dict[str, CompetencyScore],
    ) -> list[str]:
        """Identify competencies rated WEAK or NEEDS_IMPROVEMENT.

        Args:
            scores: Per-competency scores.

        Returns:
            List of competency names that need work.
        """
        weak_ratings = {CompetencyRating.WEAK, CompetencyRating.NEEDS_IMPROVEMENT}
        return [s.competency_name for s in scores.values() if s.rating in weak_ratings]

    @staticmethod
    def _find_strong_areas(
        scores: dict[str, CompetencyScore],
    ) -> list[str]:
        """Identify competencies rated STRONG.

        Args:
            scores: Per-competency scores.

        Returns:
            List of competency names performing well.
        """
        return [s.competency_name for s in scores.values() if s.rating == CompetencyRating.STRONG]

    @staticmethod
    def _generate_summary(report: EvaluationReport) -> str:
        """Generate a plain-language summary of the evaluation.

        Args:
            report: The completed evaluation report.

        Returns:
            SME-friendly summary string.
        """
        lines: list[str] = []
        lines.append(
            f"Overall: {report.overall_correct}/{report.total_cases} "
            f"correct ({report.overall_rating.value.replace('_', ' ')})"
        )

        for score in report.competency_scores.values():
            lines.append(f"  {score.competency_name}: {score.summary}")

        if report.strong_areas:
            areas = ", ".join(report.strong_areas)
            lines.append(f"Strong areas: {areas}")

        if report.weak_areas:
            areas = ", ".join(report.weak_areas)
            lines.append(f"Needs more examples: {areas}")

        return "\n".join(lines)

    @staticmethod
    def _generate_comparison_summary(comparison: ModelComparison) -> str:
        """Generate a plain-language comparison summary.

        Args:
            comparison: The model comparison data.

        Returns:
            SME-friendly comparison string.
        """
        lines: list[str] = []
        lines.append(f"Comparing {comparison.model_a_name} vs {comparison.model_b_name}:")

        if comparison.improvements:
            names = [comparison.model_b_scores[c].competency_name for c in comparison.improvements]
            lines.append(f"  Improved in: {', '.join(names)}")

        if comparison.regressions:
            names = [comparison.model_a_scores[c].competency_name for c in comparison.regressions]
            lines.append(f"  Regressed in: {', '.join(names)}")

        if not comparison.improvements and not comparison.regressions:
            lines.append("  No significant changes detected.")

        return "\n".join(lines)

    @staticmethod
    def _detect_changes(
        scores_a: dict[str, CompetencyScore],
        scores_b: dict[str, CompetencyScore],
    ) -> tuple[list[str], list[str]]:
        """Detect improvements and regressions between two score sets.

        Args:
            scores_a: Competency scores for model A.
            scores_b: Competency scores for model B.

        Returns:
            Tuple of (improvements, regressions) as competency ID lists.
        """
        improvements: list[str] = []
        regressions: list[str] = []
        all_ids = set(scores_a.keys()) | set(scores_b.keys())

        for comp_id in all_ids:
            pct_a = _correct_pct(scores_a.get(comp_id))
            pct_b = _correct_pct(scores_b.get(comp_id))

            if pct_b > pct_a:
                improvements.append(comp_id)
            elif pct_b < pct_a:
                regressions.append(comp_id)

        return improvements, regressions

    def _parse_jsonl_file(self, jsonl_path: Path) -> list[TestCase]:
        """Parse a JSONL file into test cases.

        Args:
            jsonl_path: Path to the JSONL file.

        Returns:
            List of TestCase instances.

        Raises:
            EvaluationError: On parse errors or missing fields.
        """
        cases: list[TestCase] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                cases.append(self._parse_jsonl_line(stripped, line_num))
        return cases

    @staticmethod
    def _parse_jsonl_line(line: str, line_num: int) -> TestCase:
        """Parse a single JSONL line into a TestCase.

        Args:
            line: Raw JSON string.
            line_num: Line number for error reporting.

        Returns:
            TestCase instance.

        Raises:
            EvaluationError: On JSON parse error or missing fields.
        """
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"Failed to parse JSONL line {line_num}: {exc}") from exc

        metadata = record.get("metadata")
        if metadata is None:
            raise EvaluationError(f"Line {line_num}: missing 'metadata' field")

        required = ["example_id", "discipline_id", "competency_id"]
        for field_name in required:
            if field_name not in metadata:
                raise EvaluationError(f"Line {line_num}: missing '{field_name}' in metadata")

        return TestCase(
            example_id=metadata["example_id"],
            question=record.get("instruction", ""),
            expected_answer=record.get("output", ""),
            competency_id=metadata["competency_id"],
            discipline_id=metadata["discipline_id"],
        )


def _correct_pct(score: CompetencyScore | None) -> float:
    """Compute correct percentage from a CompetencyScore.

    Args:
        score: CompetencyScore or None.

    Returns:
        Fraction of correct answers, or 0.0 if None.
    """
    if score is None or score.total_cases == 0:
        return 0.0
    return score.correct / score.total_cases


# ---------------------------------------------------------------------------
# Evaluation history
# ---------------------------------------------------------------------------


class EvaluationHistory:
    """Persists and retrieves evaluation reports from disk.

    Reports are saved as JSON files in the history directory,
    named by their run_id.

    Args:
        history_dir: Directory for storing report JSON files.

    Example::

        history = EvaluationHistory(Path("./eval_history"))
        history.save_report(report)
        loaded = history.load_report(report.run_id)
    """

    def __init__(self, history_dir: Path) -> None:
        self._dir = history_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def save_report(self, report: EvaluationReport) -> Path:
        """Save an evaluation report to disk.

        Args:
            report: The report to save.

        Returns:
            Path to the saved JSON file.
        """
        path = self._dir / f"{report.run_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    def load_report(self, run_id: str) -> EvaluationReport:
        """Load an evaluation report by run_id.

        Args:
            run_id: The unique identifier of the evaluation run.

        Returns:
            EvaluationReport instance.

        Raises:
            EvaluationError: If report file not found.
        """
        path = self._dir / f"{run_id}.json"
        if not path.exists():
            raise EvaluationError(f"Report not found: {run_id}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return EvaluationReport.from_dict(data)

    def list_reports(
        self,
        discipline_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List saved report summaries.

        Args:
            discipline_id: Optional filter by discipline.

        Returns:
            List of summary dicts with run_id, model_name,
            discipline_id, overall_accuracy, and status.
        """
        summaries: list[dict[str, Any]] = []
        for path in sorted(self._dir.glob("eval_*.json")):
            summary = self._read_summary(path)
            if summary is None:
                continue
            if discipline_id and summary["discipline_id"] != discipline_id:
                continue
            summaries.append(summary)
        return summaries

    def get_latest(
        self,
        discipline_id: str,
    ) -> EvaluationReport | None:
        """Get the most recently saved report for a discipline.

        Args:
            discipline_id: Discipline to find reports for.

        Returns:
            EvaluationReport or None if no reports exist.
        """
        matching = self.list_reports(discipline_id=discipline_id)
        if not matching:
            return None

        by_time = sorted(matching, key=lambda s: s.get("started_at", ""))
        latest_id = by_time[-1]["run_id"]
        return self.load_report(latest_id)

    @staticmethod
    def _read_summary(path: Path) -> dict[str, Any] | None:
        """Read a report summary from a JSON file.

        Args:
            path: Path to the report JSON file.

        Returns:
            Summary dict, or None on read error.
        """
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return {
                "run_id": data["run_id"],
                "model_name": data["model_name"],
                "discipline_id": data["discipline_id"],
                "overall_accuracy": data["overall_accuracy"],
                "status": data["status"],
                "started_at": data.get("started_at", ""),
            }
        except (json.JSONDecodeError, KeyError):
            return None
