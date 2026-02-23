"""Tests for Tier 1 classifier evaluation module."""

from __future__ import annotations

from chonk.tier1.classifier import DocumentClassifier
from chonk.tier1.evaluation import (
    ClassAccuracyResult,
    EvaluationReport,
    evaluate_classifier,
)
from chonk.tier1.taxonomy import TRAINABLE_TYPES, DocumentType
from chonk.tier1.training_data import generate_training_corpus


def _get_evaluation_report() -> EvaluationReport:
    """Train and evaluate a classifier, returning the report."""
    corpus = generate_training_corpus(samples_per_type=40)
    clf = DocumentClassifier()
    clf.train(corpus)
    return evaluate_classifier(clf, corpus, n_folds=5)


class TestEvaluateClassifier:
    """Tests for the evaluate_classifier function."""

    def test_returns_evaluation_report(self):
        report = _get_evaluation_report()
        assert isinstance(report, EvaluationReport)

    def test_meets_baseline_accuracy(self):
        """Overall accuracy >= 70%."""
        report = _get_evaluation_report()
        assert report.overall_accuracy >= 0.70, (
            f"Accuracy {report.overall_accuracy:.2%} below 70% baseline"
        )
        assert report.meets_baseline is True

    def test_per_class_results_has_all_types(self):
        """One result per trainable type."""
        report = _get_evaluation_report()
        result_types = {r.document_type for r in report.per_class_results}
        for dt in TRAINABLE_TYPES:
            assert dt in result_types

    def test_per_class_results_count(self):
        """14 per-class results."""
        report = _get_evaluation_report()
        assert len(report.per_class_results) == 14

    def test_confusion_matrix_square(self):
        """Confusion matrix is square with side == 14."""
        report = _get_evaluation_report()
        assert len(report.confusion_matrix) == 14
        for row in report.confusion_matrix:
            assert len(row) == 14

    def test_confusion_matrix_nonnegative(self):
        """All confusion matrix entries are non-negative."""
        report = _get_evaluation_report()
        for row in report.confusion_matrix:
            for val in row:
                assert val >= 0

    def test_n_samples_correct(self):
        """n_samples matches corpus size."""
        report = _get_evaluation_report()
        assert report.n_samples == 560  # 40 * 14

    def test_n_folds_correct(self):
        report = _get_evaluation_report()
        assert report.n_folds == 5

    def test_class_labels_populated(self):
        """class_labels list has 14 entries."""
        report = _get_evaluation_report()
        assert len(report.class_labels) == 14

    def test_summary_lines_nonempty(self):
        """summary_lines() returns formatted output."""
        report = _get_evaluation_report()
        lines = report.summary_lines()
        assert len(lines) > 0
        assert any("accuracy" in line.lower() or "%" in line for line in lines)

    def test_serialization_roundtrip(self):
        """Report can be serialized to dict."""
        report = _get_evaluation_report()
        d = report.to_dict()
        assert "overall_accuracy" in d
        assert "per_class_results" in d
        assert len(d["per_class_results"]) == 14

    def test_per_class_accuracy_bounded(self):
        """Per-class accuracy values are between 0 and 1."""
        report = _get_evaluation_report()
        for result in report.per_class_results:
            assert 0.0 <= result.accuracy <= 1.0
