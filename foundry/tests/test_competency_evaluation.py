"""Tests for competency-based evaluation system.

Covers enums, scorers, dataclasses, evaluation runner, model comparison,
evaluation history, JSONL loading, and full integration pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundry.src.evaluation import (
    CompetencyNameResolver,
    CompetencyRating,
    CompetencyScore,
    EvaluationConfig,
    EvaluationError,
    EvaluationHistory,
    EvaluationReport,
    EvaluationRunner,
    EvaluationStatus,
    KeywordSimilarityScorer,
    MockInference,
    ResponseQuality,
    TestCase,
    TestResult,
)

# ---------------------------------------------------------------------------
# 1. TestResponseQuality
# ---------------------------------------------------------------------------


class TestResponseQuality:
    """Tests for ResponseQuality enum values."""

    def test_correct_value(self) -> None:
        """CORRECT enum has string value 'correct'."""
        assert ResponseQuality.CORRECT == "correct"
        assert ResponseQuality.CORRECT.value == "correct"

    def test_partially_correct_value(self) -> None:
        """PARTIALLY_CORRECT enum has string value 'partially_correct'."""
        assert ResponseQuality.PARTIALLY_CORRECT == "partially_correct"

    def test_incorrect_value(self) -> None:
        """INCORRECT enum has string value 'incorrect'."""
        assert ResponseQuality.INCORRECT == "incorrect"

    def test_no_response_value(self) -> None:
        """NO_RESPONSE enum has string value 'no_response'."""
        assert ResponseQuality.NO_RESPONSE == "no_response"

    def test_all_values_present(self) -> None:
        """All four quality levels exist."""
        values = {q.value for q in ResponseQuality}
        assert values == {"correct", "partially_correct", "incorrect", "no_response"}

    def test_is_str_enum(self) -> None:
        """ResponseQuality is a str enum for JSON serialization."""
        assert isinstance(ResponseQuality.CORRECT, str)


# ---------------------------------------------------------------------------
# 2. TestKeywordSimilarityScorer
# ---------------------------------------------------------------------------


class TestKeywordSimilarityScorer:
    """Tests for KeywordSimilarityScorer bag-of-words approach."""

    def test_identical_texts_score_one(self) -> None:
        """Identical texts should score 1.0."""
        scorer = KeywordSimilarityScorer()
        result = scorer.score("hello world", "hello world")
        assert result == pytest.approx(1.0)

    def test_no_overlap_score_zero(self) -> None:
        """Completely different texts should score 0.0."""
        scorer = KeywordSimilarityScorer()
        result = scorer.score("hello world", "goodbye moon")
        assert result == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Partial overlap produces score between 0 and 1."""
        scorer = KeywordSimilarityScorer()
        result = scorer.score("the quick brown fox", "the slow brown dog")
        assert 0.0 < result < 1.0

    def test_case_insensitive(self) -> None:
        """Scoring is case-insensitive."""
        scorer = KeywordSimilarityScorer()
        result = scorer.score("Hello World", "hello world")
        assert result == pytest.approx(1.0)

    def test_empty_reference_scores_zero(self) -> None:
        """Empty reference text should score 0.0."""
        scorer = KeywordSimilarityScorer()
        result = scorer.score("", "hello world")
        assert result == pytest.approx(0.0)

    def test_empty_candidate_scores_zero(self) -> None:
        """Empty candidate text should score 0.0."""
        scorer = KeywordSimilarityScorer()
        result = scorer.score("hello world", "")
        assert result == pytest.approx(0.0)

    def test_both_empty_scores_zero(self) -> None:
        """Both empty should score 0.0."""
        scorer = KeywordSimilarityScorer()
        result = scorer.score("", "")
        assert result == pytest.approx(0.0)

    def test_jaccard_calculation(self) -> None:
        """Verify Jaccard similarity: |intersection| / |union|."""
        scorer = KeywordSimilarityScorer()
        # "a b c" vs "b c d" => intersection={b,c}=2, union={a,b,c,d}=4 => 0.5
        result = scorer.score("a b c", "b c d")
        assert result == pytest.approx(0.5)

    def test_duplicate_words_handled(self) -> None:
        """Duplicate words in input should not inflate score."""
        scorer = KeywordSimilarityScorer()
        result = scorer.score("hello hello hello", "hello")
        assert result == pytest.approx(1.0)

    def test_punctuation_stripped(self) -> None:
        """Punctuation should not affect scoring."""
        scorer = KeywordSimilarityScorer()
        result = scorer.score("hello, world!", "hello world")
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. TestTestCase
# ---------------------------------------------------------------------------


class TestTestCase:
    """Tests for TestCase dataclass."""

    def test_construction(self) -> None:
        """TestCase can be constructed with required fields."""
        tc = TestCase(
            example_id="ex_001",
            question="How to replace filter?",
            expected_answer="Remove old, install new.",
            competency_id="comp_proc",
            discipline_id="disc_maint",
        )
        assert tc.example_id == "ex_001"
        assert tc.competency_id == "comp_proc"

    def test_to_dict(self) -> None:
        """TestCase serializes to dictionary."""
        tc = TestCase(
            example_id="ex_001",
            question="How?",
            expected_answer="Like this.",
            competency_id="comp_proc",
            discipline_id="disc_maint",
        )
        d = tc.to_dict()
        assert d["example_id"] == "ex_001"
        assert d["question"] == "How?"
        assert d["expected_answer"] == "Like this."
        assert d["competency_id"] == "comp_proc"
        assert d["discipline_id"] == "disc_maint"

    def test_from_dict(self) -> None:
        """TestCase can be deserialized from dictionary."""
        data = {
            "example_id": "ex_002",
            "question": "What?",
            "expected_answer": "This.",
            "competency_id": "comp_fault",
            "discipline_id": "disc_maint",
        }
        tc = TestCase.from_dict(data)
        assert tc.example_id == "ex_002"
        assert tc.competency_id == "comp_fault"

    def test_roundtrip(self) -> None:
        """to_dict -> from_dict roundtrip preserves data."""
        tc = TestCase(
            example_id="ex_003",
            question="Why?",
            expected_answer="Because.",
            competency_id="comp_safety",
            discipline_id="disc_maint",
        )
        restored = TestCase.from_dict(tc.to_dict())
        assert restored.example_id == tc.example_id
        assert restored.question == tc.question
        assert restored.expected_answer == tc.expected_answer


# ---------------------------------------------------------------------------
# 4. TestTestResult
# ---------------------------------------------------------------------------


class TestTestResult:
    """Tests for TestResult dataclass."""

    def test_construction(self) -> None:
        """TestResult holds test case, model response, and scoring."""
        tc = TestCase(
            example_id="ex_001",
            question="Q?",
            expected_answer="A.",
            competency_id="comp_proc",
            discipline_id="disc_maint",
        )
        tr = TestResult(
            test_case=tc,
            model_response="A.",
            similarity_score=0.95,
            quality=ResponseQuality.CORRECT,
            response_time_ms=150.0,
        )
        assert tr.quality == ResponseQuality.CORRECT
        assert tr.similarity_score == pytest.approx(0.95)

    def test_to_dict(self) -> None:
        """TestResult serializes with nested test case."""
        tc = TestCase(
            example_id="ex_001",
            question="Q?",
            expected_answer="A.",
            competency_id="comp_proc",
            discipline_id="disc_maint",
        )
        tr = TestResult(
            test_case=tc,
            model_response="A.",
            similarity_score=0.9,
            quality=ResponseQuality.CORRECT,
            response_time_ms=100.0,
        )
        d = tr.to_dict()
        assert d["quality"] == "correct"
        assert d["model_response"] == "A."
        assert "test_case" in d

    def test_quality_correct(self) -> None:
        """CORRECT quality assigned for high similarity."""
        tc = TestCase("ex_1", "Q", "A", "c", "d")
        tr = TestResult(
            test_case=tc,
            model_response="A",
            similarity_score=0.85,
            quality=ResponseQuality.CORRECT,
            response_time_ms=50.0,
        )
        assert tr.quality == ResponseQuality.CORRECT

    def test_quality_no_response(self) -> None:
        """NO_RESPONSE quality for empty response."""
        tc = TestCase("ex_1", "Q", "A", "c", "d")
        tr = TestResult(
            test_case=tc,
            model_response="",
            similarity_score=0.0,
            quality=ResponseQuality.NO_RESPONSE,
            response_time_ms=10.0,
        )
        assert tr.quality == ResponseQuality.NO_RESPONSE


# ---------------------------------------------------------------------------
# 5. TestCompetencyScore
# ---------------------------------------------------------------------------


class TestCompetencyScore:
    """Tests for CompetencyScore dataclass."""

    def test_construction(self) -> None:
        """CompetencyScore holds aggregated metrics and rating."""
        cs = CompetencyScore(
            competency_id="comp_proc",
            competency_name="Procedural Comprehension",
            total_cases=10,
            correct=8,
            partially_correct=1,
            incorrect=1,
            no_response=0,
            rating=CompetencyRating.STRONG,
            summary="8/10 correct",
        )
        assert cs.total_cases == 10
        assert cs.correct == 8
        assert cs.rating == CompetencyRating.STRONG

    def test_to_dict(self) -> None:
        """CompetencyScore serializes properly."""
        cs = CompetencyScore(
            competency_id="comp_fault",
            competency_name="Fault Isolation",
            total_cases=5,
            correct=3,
            partially_correct=1,
            incorrect=1,
            no_response=0,
            rating=CompetencyRating.ADEQUATE,
            summary="3/5 correct",
        )
        d = cs.to_dict()
        assert d["competency_id"] == "comp_fault"
        assert d["rating"] == "adequate"
        assert d["summary"] == "3/5 correct"

    def test_strong_rating(self) -> None:
        """STRONG rating enum is 'strong'."""
        assert CompetencyRating.STRONG == "strong"

    def test_adequate_rating(self) -> None:
        """ADEQUATE rating enum is 'adequate'."""
        assert CompetencyRating.ADEQUATE == "adequate"

    def test_needs_improvement_rating(self) -> None:
        """NEEDS_IMPROVEMENT rating enum is 'needs_improvement'."""
        assert CompetencyRating.NEEDS_IMPROVEMENT == "needs_improvement"

    def test_weak_rating(self) -> None:
        """WEAK rating enum is 'weak'."""
        assert CompetencyRating.WEAK == "weak"

    def test_untested_rating(self) -> None:
        """UNTESTED rating enum is 'untested'."""
        assert CompetencyRating.UNTESTED == "untested"

    def test_all_ratings_present(self) -> None:
        """All five competency ratings exist."""
        values = {r.value for r in CompetencyRating}
        expected = {"strong", "adequate", "needs_improvement", "weak", "untested"}
        assert values == expected


# ---------------------------------------------------------------------------
# 6. TestEvaluationConfig
# ---------------------------------------------------------------------------


class TestEvaluationConfig:
    """Tests for EvaluationConfig defaults and serialization."""

    def test_defaults(self) -> None:
        """Default config has reasonable thresholds."""
        cfg = EvaluationConfig()
        assert cfg.correct_threshold == pytest.approx(0.7)
        assert cfg.partial_threshold == pytest.approx(0.4)
        assert cfg.strong_threshold == pytest.approx(0.8)
        assert cfg.adequate_threshold == pytest.approx(0.6)
        assert cfg.needs_improvement_threshold == pytest.approx(0.4)
        assert cfg.max_tokens == 512

    def test_custom_thresholds(self) -> None:
        """Custom thresholds can be set."""
        cfg = EvaluationConfig(correct_threshold=0.8, partial_threshold=0.5)
        assert cfg.correct_threshold == pytest.approx(0.8)
        assert cfg.partial_threshold == pytest.approx(0.5)

    def test_to_dict(self) -> None:
        """Config serializes to dict."""
        cfg = EvaluationConfig()
        d = cfg.to_dict()
        assert "correct_threshold" in d
        assert "max_tokens" in d

    def test_from_dict(self) -> None:
        """Config can be restored from dict."""
        cfg = EvaluationConfig(correct_threshold=0.9)
        restored = EvaluationConfig.from_dict(cfg.to_dict())
        assert restored.correct_threshold == pytest.approx(0.9)

    def test_roundtrip(self) -> None:
        """to_dict -> from_dict preserves all fields."""
        cfg = EvaluationConfig(
            correct_threshold=0.75,
            partial_threshold=0.35,
            strong_threshold=0.85,
            adequate_threshold=0.65,
            needs_improvement_threshold=0.45,
            max_tokens=256,
        )
        restored = EvaluationConfig.from_dict(cfg.to_dict())
        assert restored.correct_threshold == pytest.approx(0.75)
        assert restored.max_tokens == 256


# ---------------------------------------------------------------------------
# 7. TestMockInference
# ---------------------------------------------------------------------------


class TestMockInference:
    """Tests for MockInference helper class."""

    def test_configured_response(self) -> None:
        """Returns pre-configured response for known prompt."""
        model = MockInference(responses={"hello": "world"})
        assert model.generate("hello") == "world"

    def test_default_response(self) -> None:
        """Returns default response for unknown prompt."""
        model = MockInference(default_response="I don't know")
        assert model.generate("unknown question") == "I don't know"

    def test_empty_default(self) -> None:
        """Default response is empty string when not specified."""
        model = MockInference()
        assert model.generate("anything") == ""

    def test_max_tokens_accepted(self) -> None:
        """generate accepts max_tokens parameter."""
        model = MockInference(responses={"q": "a"})
        result = model.generate("q", max_tokens=128)
        assert result == "a"

    def test_none_responses_uses_default(self) -> None:
        """None responses dict uses default for everything."""
        model = MockInference(responses=None, default_response="fallback")
        assert model.generate("any prompt") == "fallback"


# ---------------------------------------------------------------------------
# 8. TestEvaluationRunner
# ---------------------------------------------------------------------------


class TestEvaluationRunner:
    """Tests for EvaluationRunner evaluation logic."""

    def test_evaluate_single_correct(self, evaluation_runner: EvaluationRunner) -> None:
        """Single evaluation with matching response is CORRECT."""
        model = MockInference(responses={"Q?": "exact answer here"})
        tc = TestCase("ex_1", "Q?", "exact answer here", "comp_proc", "disc_maint")
        result = evaluation_runner._evaluate_single(model, tc)
        assert result.quality == ResponseQuality.CORRECT
        assert result.similarity_score == pytest.approx(1.0)

    def test_evaluate_single_no_response(self, evaluation_runner: EvaluationRunner) -> None:
        """Empty model response gets NO_RESPONSE quality."""
        model = MockInference(default_response="")
        tc = TestCase("ex_1", "Q?", "expected answer", "comp_proc", "disc_maint")
        result = evaluation_runner._evaluate_single(model, tc)
        assert result.quality == ResponseQuality.NO_RESPONSE

    def test_evaluate_single_incorrect(self, evaluation_runner: EvaluationRunner) -> None:
        """Completely wrong answer is INCORRECT."""
        model = MockInference(responses={"Q?": "bananas oranges apples grapes"})
        tc = TestCase(
            "ex_1",
            "Q?",
            "check pressure readings isolate sections inspect fittings",
            "comp_proc",
            "disc_maint",
        )
        result = evaluation_runner._evaluate_single(model, tc)
        assert result.quality == ResponseQuality.INCORRECT

    def test_evaluate_single_records_time(self, evaluation_runner: EvaluationRunner) -> None:
        """Evaluation records response time in milliseconds."""
        model = MockInference(responses={"Q?": "answer"})
        tc = TestCase("ex_1", "Q?", "answer", "comp_proc", "disc_maint")
        result = evaluation_runner._evaluate_single(model, tc)
        assert result.response_time_ms >= 0

    def test_run_evaluation_basic(
        self,
        evaluation_runner: EvaluationRunner,
        competency_names: dict[str, str],
    ) -> None:
        """Full evaluation run produces report with competency scores."""
        model = MockInference(
            responses={
                "Q1": "answer one",
                "Q2": "answer two",
            }
        )
        cases = [
            TestCase("ex_1", "Q1", "answer one", "comp_proc", "disc_maint"),
            TestCase("ex_2", "Q2", "answer two", "comp_proc", "disc_maint"),
        ]
        report = evaluation_runner.run_evaluation(
            model=model,
            test_cases=cases,
            competency_names=competency_names,
            model_name="test-model",
            discipline_id="disc_maint",
        )
        assert report.status == EvaluationStatus.COMPLETED
        assert report.total_cases == 2
        assert report.overall_correct == 2
        assert "comp_proc" in report.competency_scores

    def test_run_evaluation_multiple_competencies(
        self,
        evaluation_runner: EvaluationRunner,
        competency_names: dict[str, str],
    ) -> None:
        """Evaluation groups results by competency."""
        model = MockInference(
            responses={
                "Q1": "correct response one",
                "Q2": "correct response two",
                "Q3": "wrong answer completely",
            }
        )
        cases = [
            TestCase("ex_1", "Q1", "correct response one", "comp_proc", "disc_maint"),
            TestCase("ex_2", "Q2", "correct response two", "comp_fault", "disc_maint"),
            TestCase(
                "ex_3",
                "Q3",
                "check pressure isolate sections inspect fittings",
                "comp_fault",
                "disc_maint",
            ),
        ]
        report = evaluation_runner.run_evaluation(
            model=model,
            test_cases=cases,
            competency_names=competency_names,
            model_name="test-model",
            discipline_id="disc_maint",
        )
        assert "comp_proc" in report.competency_scores
        assert "comp_fault" in report.competency_scores
        assert report.competency_scores["comp_proc"].correct == 1
        assert report.competency_scores["comp_proc"].total_cases == 1

    def test_run_evaluation_empty_cases(
        self,
        evaluation_runner: EvaluationRunner,
        competency_names: dict[str, str],
    ) -> None:
        """Evaluation with no test cases raises error."""
        model = MockInference()
        with pytest.raises(EvaluationError, match="No test cases"):
            evaluation_runner.run_evaluation(
                model=model,
                test_cases=[],
                competency_names=competency_names,
                model_name="test-model",
                discipline_id="disc_maint",
            )

    def test_competency_scoring_strong(self) -> None:
        """Competency with >= 80% correct gets STRONG rating."""
        runner = EvaluationRunner()
        model = MockInference(responses={"Q": "exact match answer"})
        cases = [
            TestCase(f"ex_{i}", "Q", "exact match answer", "comp_proc", "d") for i in range(10)
        ]
        report = runner.run_evaluation(
            model=model,
            test_cases=cases,
            competency_names={"comp_proc": "Procedural"},
            model_name="m",
            discipline_id="d",
        )
        assert report.competency_scores["comp_proc"].rating == CompetencyRating.STRONG

    def test_competency_scoring_weak(self) -> None:
        """Competency with < 40% correct gets WEAK rating."""
        runner = EvaluationRunner()
        model = MockInference(default_response="completely irrelevant gibberish")
        cases = [
            TestCase(
                f"ex_{i}",
                f"Q{i}",
                "check pressure gauges isolate sections systematically inspect fittings",
                "comp_proc",
                "d",
            )
            for i in range(10)
        ]
        report = runner.run_evaluation(
            model=model,
            test_cases=cases,
            competency_names={"comp_proc": "Procedural"},
            model_name="m",
            discipline_id="d",
        )
        assert report.competency_scores["comp_proc"].rating == CompetencyRating.WEAK

    def test_custom_scorer(
        self,
        evaluation_runner: EvaluationRunner,
        competency_names: dict[str, str],
    ) -> None:
        """Evaluation accepts custom SimilarityScorer."""
        scorer = KeywordSimilarityScorer()
        model = MockInference(responses={"Q": "answer"})
        cases = [TestCase("ex_1", "Q", "answer", "comp_proc", "disc_maint")]
        report = evaluation_runner.run_evaluation(
            model=model,
            test_cases=cases,
            competency_names=competency_names,
            model_name="test",
            discipline_id="disc_maint",
            scorer=scorer,
        )
        assert report.status == EvaluationStatus.COMPLETED

    def test_overall_accuracy(self, competency_names: dict[str, str]) -> None:
        """Overall accuracy is computed correctly."""
        runner = EvaluationRunner()
        model = MockInference(
            responses={
                "Q1": "correct answer one",
                "Q2": "wrong stuff entirely",
            }
        )
        cases = [
            TestCase("ex_1", "Q1", "correct answer one", "comp_proc", "disc_maint"),
            TestCase(
                "ex_2",
                "Q2",
                "check pressure isolate sections inspect fittings",
                "comp_proc",
                "disc_maint",
            ),
        ]
        report = runner.run_evaluation(
            model=model,
            test_cases=cases,
            competency_names=competency_names,
            model_name="m",
            discipline_id="disc_maint",
        )
        # 1 correct out of 2
        assert report.overall_accuracy == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 9. TestModelComparison
# ---------------------------------------------------------------------------


class TestModelComparison:
    """Tests for ModelComparison and run_comparison."""

    def test_comparison_detects_improvement(
        self,
        evaluation_runner: EvaluationRunner,
        competency_names: dict[str, str],
    ) -> None:
        """Comparison detects when model B outperforms model A."""
        model_a = MockInference(default_response="completely wrong gibberish nonsense")
        model_b = MockInference(responses={"Q": "exact correct answer"})
        cases = [TestCase("ex_1", "Q", "exact correct answer", "comp_proc", "disc_maint")]
        comparison = evaluation_runner.run_comparison(
            model_a=model_a,
            model_b=model_b,
            test_cases=cases,
            competency_names=competency_names,
            model_a_name="baseline",
            model_b_name="finetuned",
            discipline_id="disc_maint",
        )
        assert "comp_proc" in comparison.improvements

    def test_comparison_detects_regression(
        self,
        evaluation_runner: EvaluationRunner,
        competency_names: dict[str, str],
    ) -> None:
        """Comparison detects when model B underperforms model A."""
        model_a = MockInference(responses={"Q": "exact correct answer"})
        model_b = MockInference(default_response="wrong stuff entirely")
        cases = [TestCase("ex_1", "Q", "exact correct answer", "comp_proc", "disc_maint")]
        comparison = evaluation_runner.run_comparison(
            model_a=model_a,
            model_b=model_b,
            test_cases=cases,
            competency_names=competency_names,
            model_a_name="baseline",
            model_b_name="regressed",
            discipline_id="disc_maint",
        )
        assert "comp_proc" in comparison.regressions

    def test_comparison_to_dict(
        self,
        evaluation_runner: EvaluationRunner,
        competency_names: dict[str, str],
    ) -> None:
        """ModelComparison serializes properly."""
        model_a = MockInference(responses={"Q": "answer"})
        model_b = MockInference(responses={"Q": "answer"})
        cases = [TestCase("ex_1", "Q", "answer", "comp_proc", "disc_maint")]
        comparison = evaluation_runner.run_comparison(
            model_a=model_a,
            model_b=model_b,
            test_cases=cases,
            competency_names=competency_names,
            model_a_name="A",
            model_b_name="B",
            discipline_id="disc_maint",
        )
        d = comparison.to_dict()
        assert d["model_a_name"] == "A"
        assert d["model_b_name"] == "B"
        assert "improvements" in d
        assert "regressions" in d
        assert "summary" in d

    def test_comparison_summary_generated(
        self,
        evaluation_runner: EvaluationRunner,
        competency_names: dict[str, str],
    ) -> None:
        """Comparison has non-empty summary."""
        model_a = MockInference(responses={"Q": "answer"})
        model_b = MockInference(responses={"Q": "answer"})
        cases = [TestCase("ex_1", "Q", "answer", "comp_proc", "disc_maint")]
        comparison = evaluation_runner.run_comparison(
            model_a=model_a,
            model_b=model_b,
            test_cases=cases,
            competency_names=competency_names,
            model_a_name="A",
            model_b_name="B",
            discipline_id="disc_maint",
        )
        assert len(comparison.summary) > 0

    def test_comparison_no_change(
        self,
        evaluation_runner: EvaluationRunner,
        competency_names: dict[str, str],
    ) -> None:
        """Comparison with identical models shows no improvements or regressions."""
        model = MockInference(responses={"Q": "answer"})
        cases = [TestCase("ex_1", "Q", "answer", "comp_proc", "disc_maint")]
        comparison = evaluation_runner.run_comparison(
            model_a=model,
            model_b=model,
            test_cases=cases,
            competency_names=competency_names,
            model_a_name="A",
            model_b_name="B",
            discipline_id="disc_maint",
        )
        assert len(comparison.improvements) == 0
        assert len(comparison.regressions) == 0


# ---------------------------------------------------------------------------
# 10. TestEvaluationReport
# ---------------------------------------------------------------------------


class TestEvaluationReport:
    """Tests for EvaluationReport construction and serialization."""

    def test_overall_accuracy_computed(
        self,
        evaluation_runner: EvaluationRunner,
        mock_model: MockInference,
        sample_test_jsonl: Path,
        competency_names: dict[str, str],
    ) -> None:
        """Report computes overall accuracy across all test cases."""
        cases = evaluation_runner.load_test_cases(sample_test_jsonl)
        report = evaluation_runner.run_evaluation(
            model=mock_model,
            test_cases=cases,
            competency_names=competency_names,
            model_name="test-model",
            discipline_id="disc_maint",
        )
        assert 0.0 <= report.overall_accuracy <= 1.0
        assert report.total_cases == 15

    def test_weak_areas_identified(
        self,
        evaluation_runner: EvaluationRunner,
        mock_model: MockInference,
        sample_test_jsonl: Path,
        competency_names: dict[str, str],
    ) -> None:
        """Report identifies weak competency areas."""
        cases = evaluation_runner.load_test_cases(sample_test_jsonl)
        report = evaluation_runner.run_evaluation(
            model=mock_model,
            test_cases=cases,
            competency_names=competency_names,
            model_name="test-model",
            discipline_id="disc_maint",
        )
        # Parts ID has weak mock responses, should appear in weak_areas
        assert isinstance(report.weak_areas, list)

    def test_strong_areas_identified(
        self,
        evaluation_runner: EvaluationRunner,
        mock_model: MockInference,
        sample_test_jsonl: Path,
        competency_names: dict[str, str],
    ) -> None:
        """Report identifies strong competency areas."""
        cases = evaluation_runner.load_test_cases(sample_test_jsonl)
        report = evaluation_runner.run_evaluation(
            model=mock_model,
            test_cases=cases,
            competency_names=competency_names,
            model_name="test-model",
            discipline_id="disc_maint",
        )
        assert isinstance(report.strong_areas, list)

    def test_plain_language_summary(
        self,
        evaluation_runner: EvaluationRunner,
        mock_model: MockInference,
        sample_test_jsonl: Path,
        competency_names: dict[str, str],
    ) -> None:
        """Report has non-empty plain language summary."""
        cases = evaluation_runner.load_test_cases(sample_test_jsonl)
        report = evaluation_runner.run_evaluation(
            model=mock_model,
            test_cases=cases,
            competency_names=competency_names,
            model_name="test-model",
            discipline_id="disc_maint",
        )
        assert len(report.plain_language_summary) > 0
        # Summary should NOT contain ML jargon
        jargon = ["loss", "perplexity", "f1", "precision", "recall", "epoch"]
        lower_summary = report.plain_language_summary.lower()
        for term in jargon:
            assert term not in lower_summary, f"Summary contains ML jargon: {term}"

    def test_report_to_dict(
        self,
        evaluation_runner: EvaluationRunner,
        competency_names: dict[str, str],
    ) -> None:
        """Report serializes to dict."""
        model = MockInference(responses={"Q": "A"})
        cases = [TestCase("ex_1", "Q", "A", "comp_proc", "disc_maint")]
        report = evaluation_runner.run_evaluation(
            model=model,
            test_cases=cases,
            competency_names=competency_names,
            model_name="test",
            discipline_id="disc_maint",
        )
        d = report.to_dict()
        assert d["status"] == "completed"
        assert d["model_name"] == "test"
        assert "competency_scores" in d
        assert "plain_language_summary" in d

    def test_report_from_dict(
        self,
        evaluation_runner: EvaluationRunner,
        competency_names: dict[str, str],
    ) -> None:
        """Report can be deserialized from dict."""
        model = MockInference(responses={"Q": "A"})
        cases = [TestCase("ex_1", "Q", "A", "comp_proc", "disc_maint")]
        report = evaluation_runner.run_evaluation(
            model=model,
            test_cases=cases,
            competency_names=competency_names,
            model_name="test",
            discipline_id="disc_maint",
        )
        restored = EvaluationReport.from_dict(report.to_dict())
        assert restored.run_id == report.run_id
        assert restored.model_name == report.model_name
        assert restored.overall_accuracy == pytest.approx(report.overall_accuracy)

    def test_report_status_completed(
        self,
        evaluation_runner: EvaluationRunner,
        competency_names: dict[str, str],
    ) -> None:
        """Successful evaluation has COMPLETED status."""
        model = MockInference(responses={"Q": "A"})
        cases = [TestCase("ex_1", "Q", "A", "comp_proc", "disc_maint")]
        report = evaluation_runner.run_evaluation(
            model=model,
            test_cases=cases,
            competency_names=competency_names,
            model_name="test",
            discipline_id="disc_maint",
        )
        assert report.status == EvaluationStatus.COMPLETED

    def test_report_timestamps(
        self,
        evaluation_runner: EvaluationRunner,
        competency_names: dict[str, str],
    ) -> None:
        """Report records start and completion timestamps."""
        model = MockInference(responses={"Q": "A"})
        cases = [TestCase("ex_1", "Q", "A", "comp_proc", "disc_maint")]
        report = evaluation_runner.run_evaluation(
            model=model,
            test_cases=cases,
            competency_names=competency_names,
            model_name="test",
            discipline_id="disc_maint",
        )
        assert report.started_at is not None
        assert report.completed_at is not None
        assert report.completed_at >= report.started_at


# ---------------------------------------------------------------------------
# 11. TestEvaluationHistory
# ---------------------------------------------------------------------------


class TestEvaluationHistory:
    """Tests for EvaluationHistory persistence."""

    def test_save_and_load(
        self,
        temp_dir: Path,
        evaluation_runner: EvaluationRunner,
        competency_names: dict[str, str],
    ) -> None:
        """Can save and load a report by run_id."""
        history = EvaluationHistory(temp_dir / "history")
        model = MockInference(responses={"Q": "A"})
        cases = [TestCase("ex_1", "Q", "A", "comp_proc", "disc_maint")]
        report = evaluation_runner.run_evaluation(
            model=model,
            test_cases=cases,
            competency_names=competency_names,
            model_name="test",
            discipline_id="disc_maint",
        )
        path = history.save_report(report)
        assert path.exists()

        loaded = history.load_report(report.run_id)
        assert loaded.run_id == report.run_id
        assert loaded.model_name == report.model_name

    def test_load_nonexistent_raises(self, temp_dir: Path) -> None:
        """Loading non-existent report raises EvaluationError."""
        history = EvaluationHistory(temp_dir / "history")
        with pytest.raises(EvaluationError, match="not found"):
            history.load_report("nonexistent_id")

    def test_list_reports(
        self,
        temp_dir: Path,
        evaluation_runner: EvaluationRunner,
        competency_names: dict[str, str],
    ) -> None:
        """Can list saved report summaries."""
        history = EvaluationHistory(temp_dir / "history")
        model = MockInference(responses={"Q": "A"})
        cases = [TestCase("ex_1", "Q", "A", "comp_proc", "disc_maint")]

        report1 = evaluation_runner.run_evaluation(
            model=model,
            test_cases=cases,
            competency_names=competency_names,
            model_name="model-v1",
            discipline_id="disc_maint",
        )
        report2 = evaluation_runner.run_evaluation(
            model=model,
            test_cases=cases,
            competency_names=competency_names,
            model_name="model-v2",
            discipline_id="disc_maint",
        )
        history.save_report(report1)
        history.save_report(report2)

        summaries = history.list_reports()
        assert len(summaries) == 2

    def test_list_reports_by_discipline(
        self,
        temp_dir: Path,
        evaluation_runner: EvaluationRunner,
        competency_names: dict[str, str],
    ) -> None:
        """Can filter report list by discipline_id."""
        history = EvaluationHistory(temp_dir / "history")
        model = MockInference(responses={"Q": "A"})
        cases_a = [TestCase("ex_1", "Q", "A", "comp_proc", "disc_maint")]
        cases_b = [TestCase("ex_1", "Q", "A", "comp_proc", "disc_avionics")]

        report_a = evaluation_runner.run_evaluation(
            model=model,
            test_cases=cases_a,
            competency_names=competency_names,
            model_name="m",
            discipline_id="disc_maint",
        )
        report_b = evaluation_runner.run_evaluation(
            model=model,
            test_cases=cases_b,
            competency_names=competency_names,
            model_name="m",
            discipline_id="disc_avionics",
        )
        history.save_report(report_a)
        history.save_report(report_b)

        maint_reports = history.list_reports(discipline_id="disc_maint")
        assert len(maint_reports) == 1
        assert maint_reports[0]["discipline_id"] == "disc_maint"

    def test_get_latest(
        self,
        temp_dir: Path,
        evaluation_runner: EvaluationRunner,
        competency_names: dict[str, str],
    ) -> None:
        """get_latest returns most recent report for discipline."""
        history = EvaluationHistory(temp_dir / "history")
        model = MockInference(responses={"Q": "A"})
        cases = [TestCase("ex_1", "Q", "A", "comp_proc", "disc_maint")]

        report1 = evaluation_runner.run_evaluation(
            model=model,
            test_cases=cases,
            competency_names=competency_names,
            model_name="model-v1",
            discipline_id="disc_maint",
        )
        history.save_report(report1)

        report2 = evaluation_runner.run_evaluation(
            model=model,
            test_cases=cases,
            competency_names=competency_names,
            model_name="model-v2",
            discipline_id="disc_maint",
        )
        history.save_report(report2)

        latest = history.get_latest("disc_maint")
        assert latest is not None
        assert latest.run_id == report2.run_id

    def test_get_latest_no_reports(self, temp_dir: Path) -> None:
        """get_latest returns None when no reports exist."""
        history = EvaluationHistory(temp_dir / "history")
        assert history.get_latest("disc_nonexistent") is None

    def test_history_creates_directory(self, temp_dir: Path) -> None:
        """History directory is created if it does not exist."""
        history_dir = temp_dir / "new_history_dir"
        assert not history_dir.exists()
        EvaluationHistory(history_dir)
        assert history_dir.exists()


# ---------------------------------------------------------------------------
# 12. TestLoadTestCases
# ---------------------------------------------------------------------------


class TestLoadTestCases:
    """Tests for loading test cases from JSONL files."""

    def test_load_from_jsonl(
        self,
        evaluation_runner: EvaluationRunner,
        sample_test_jsonl: Path,
    ) -> None:
        """Loads test cases from JSONL file."""
        cases = evaluation_runner.load_test_cases(sample_test_jsonl)
        assert len(cases) == 15

    def test_loaded_case_fields(
        self,
        evaluation_runner: EvaluationRunner,
        sample_test_jsonl: Path,
    ) -> None:
        """Loaded cases have correct fields from JSONL."""
        cases = evaluation_runner.load_test_cases(sample_test_jsonl)
        first = cases[0]
        assert first.example_id == "ex_001"
        assert first.discipline_id == "disc_maint"
        assert first.competency_id == "comp_proc"
        assert len(first.question) > 0
        assert len(first.expected_answer) > 0

    def test_load_nonexistent_file_raises(
        self,
        evaluation_runner: EvaluationRunner,
        tmp_path: Path,
    ) -> None:
        """Loading non-existent JSONL raises EvaluationError."""
        with pytest.raises(EvaluationError, match="not found"):
            evaluation_runner.load_test_cases(tmp_path / "missing.jsonl")

    def test_load_empty_file_raises(
        self,
        evaluation_runner: EvaluationRunner,
        tmp_path: Path,
    ) -> None:
        """Loading empty JSONL raises EvaluationError."""
        empty_path = tmp_path / "empty.jsonl"
        empty_path.write_text("")
        with pytest.raises(EvaluationError, match="No test cases"):
            evaluation_runner.load_test_cases(empty_path)

    def test_load_invalid_json_raises(
        self,
        evaluation_runner: EvaluationRunner,
        tmp_path: Path,
    ) -> None:
        """Loading malformed JSONL raises EvaluationError."""
        bad_path = tmp_path / "bad.jsonl"
        bad_path.write_text("not valid json\n")
        with pytest.raises(EvaluationError, match="parse"):
            evaluation_runner.load_test_cases(bad_path)

    def test_load_missing_fields_raises(
        self,
        evaluation_runner: EvaluationRunner,
        tmp_path: Path,
    ) -> None:
        """JSONL record missing required fields raises EvaluationError."""
        bad_path = tmp_path / "incomplete.jsonl"
        record = {"instruction": "Q", "output": "A"}  # missing metadata
        bad_path.write_text(json.dumps(record) + "\n")
        with pytest.raises(EvaluationError, match="missing"):
            evaluation_runner.load_test_cases(bad_path)


# ---------------------------------------------------------------------------
# 13. TestCompetencyNameResolver
# ---------------------------------------------------------------------------


class TestCompetencyNameResolver:
    """Tests for CompetencyNameResolver."""

    def test_resolve_known_id(self, competency_names: dict[str, str]) -> None:
        """Resolves known competency ID to name."""
        resolver = CompetencyNameResolver(competency_names)
        assert resolver.resolve("comp_proc") == "Procedural Comprehension"

    def test_resolve_unknown_id(self) -> None:
        """Unknown ID returns the ID itself as fallback."""
        resolver = CompetencyNameResolver({})
        assert resolver.resolve("comp_unknown") == "comp_unknown"


# ---------------------------------------------------------------------------
# 14. TestEvaluationStatus
# ---------------------------------------------------------------------------


class TestEvaluationStatus:
    """Tests for EvaluationStatus enum."""

    def test_all_statuses(self) -> None:
        """All evaluation statuses exist."""
        values = {s.value for s in EvaluationStatus}
        assert values == {"pending", "running", "completed", "failed"}

    def test_is_str_enum(self) -> None:
        """EvaluationStatus is a str enum."""
        assert isinstance(EvaluationStatus.PENDING, str)


# ---------------------------------------------------------------------------
# 15. TestIntegration
# ---------------------------------------------------------------------------


class TestIntegration:
    """Full pipeline integration: load JSONL -> run eval -> report with plain language."""

    def test_full_pipeline(
        self,
        evaluation_runner: EvaluationRunner,
        mock_model: MockInference,
        sample_test_jsonl: Path,
        competency_names: dict[str, str],
        temp_dir: Path,
    ) -> None:
        """End-to-end: load test set, run evaluation, save report, verify output."""
        # Load test cases
        cases = evaluation_runner.load_test_cases(sample_test_jsonl)
        assert len(cases) == 15

        # Run evaluation
        report = evaluation_runner.run_evaluation(
            model=mock_model,
            test_cases=cases,
            competency_names=competency_names,
            model_name="maintenance-lora-v1",
            discipline_id="disc_maint",
        )

        # Verify report structure
        assert report.status == EvaluationStatus.COMPLETED
        assert report.total_cases == 15
        assert len(report.competency_scores) == 5
        assert len(report.test_results) == 15

        # Verify plain language summary
        assert len(report.plain_language_summary) > 0

        # Save to history
        history = EvaluationHistory(temp_dir / "history")
        path = history.save_report(report)
        assert path.exists()

        # Reload and verify
        loaded = history.load_report(report.run_id)
        assert loaded.total_cases == 15
        assert loaded.model_name == "maintenance-lora-v1"

    def test_comparison_pipeline(
        self,
        evaluation_runner: EvaluationRunner,
        mock_model: MockInference,
        sample_test_jsonl: Path,
        competency_names: dict[str, str],
    ) -> None:
        """End-to-end comparison between two models."""
        cases = evaluation_runner.load_test_cases(sample_test_jsonl)

        # Model A: weak responses
        model_a = MockInference(default_response="I am not sure about this topic.")

        # Model B: pre-configured mock (stronger)
        model_b = mock_model

        comparison = evaluation_runner.run_comparison(
            model_a=model_a,
            model_b=model_b,
            test_cases=cases,
            competency_names=competency_names,
            model_a_name="base-model",
            model_b_name="finetuned-lora",
            discipline_id="disc_maint",
        )

        # Finetuned should improve over base in at least some competencies
        assert len(comparison.improvements) > 0
        assert len(comparison.summary) > 0

    def test_report_has_all_competencies(
        self,
        evaluation_runner: EvaluationRunner,
        mock_model: MockInference,
        sample_test_jsonl: Path,
        competency_names: dict[str, str],
    ) -> None:
        """Report covers all 5 competencies from test set."""
        cases = evaluation_runner.load_test_cases(sample_test_jsonl)
        report = evaluation_runner.run_evaluation(
            model=mock_model,
            test_cases=cases,
            competency_names=competency_names,
            model_name="test",
            discipline_id="disc_maint",
        )
        expected_competencies = {
            "comp_proc",
            "comp_fault",
            "comp_safety",
            "comp_parts",
            "comp_tools",
        }
        assert set(report.competency_scores.keys()) == expected_competencies

    def test_each_competency_has_3_cases(
        self,
        evaluation_runner: EvaluationRunner,
        mock_model: MockInference,
        sample_test_jsonl: Path,
        competency_names: dict[str, str],
    ) -> None:
        """Each competency should have exactly 3 test cases."""
        cases = evaluation_runner.load_test_cases(sample_test_jsonl)
        report = evaluation_runner.run_evaluation(
            model=mock_model,
            test_cases=cases,
            competency_names=competency_names,
            model_name="test",
            discipline_id="disc_maint",
        )
        for comp_id, score in report.competency_scores.items():
            assert score.total_cases == 3, f"{comp_id} has {score.total_cases} cases, expected 3"

    def test_competency_summaries_are_plain_language(
        self,
        evaluation_runner: EvaluationRunner,
        mock_model: MockInference,
        sample_test_jsonl: Path,
        competency_names: dict[str, str],
    ) -> None:
        """Each competency summary uses plain language format."""
        cases = evaluation_runner.load_test_cases(sample_test_jsonl)
        report = evaluation_runner.run_evaluation(
            model=mock_model,
            test_cases=cases,
            competency_names=competency_names,
            model_name="test",
            discipline_id="disc_maint",
        )
        for score in report.competency_scores.values():
            # Summary should contain "X/Y correct" pattern
            assert "/" in score.summary, f"Summary missing X/Y pattern: {score.summary}"
            assert "correct" in score.summary.lower(), f"Summary missing 'correct': {score.summary}"
