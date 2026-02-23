"""Tests for foundry.src.diagnostics — failure detection & training diagnostics.

TDD test suite with synthetic metric histories covering:
- MetricSnapshot construction and serialization
- DiagnosticIssue construction and severity
- DiagnosticConfig defaults and custom values
- TrendAnalyzer slope, plateau, direction
- ConvergenceChecker for plateau, no improvement, divergence
- OverfitDetector for train/val gap and val loss increasing
- StabilityChecker for loss spikes and gradient explosion
- DataQualityChecker for curriculum stats
- DiagnosticReport health, summary, forge recommendations
- TrainingDiagnostics full analysis
- Integration test simulating realistic training with overfit
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from foundry.src.diagnostics import (
    ConvergenceChecker,
    DataQualityChecker,
    DiagnosticConfig,
    DiagnosticIssue,
    DiagnosticReport,
    DiagnosticsError,
    IssueCategory,
    IssueSeverity,
    MetricSnapshot,
    OverfitDetector,
    StabilityChecker,
    TrainingDiagnostics,
    TrainingTrend,
    TrendAnalyzer,
)

# ---------------------------------------------------------------------------
# Helper factory functions for synthetic metric histories
# ---------------------------------------------------------------------------


def make_healthy_metrics(epochs: int = 10) -> list[MetricSnapshot]:
    """Create metrics with steadily decreasing train and val loss.

    Both losses decrease smoothly from ~2.0 toward ~0.3 over the epochs.
    """
    metrics: list[MetricSnapshot] = []
    for i in range(epochs):
        t = i / max(epochs - 1, 1)
        train_loss = 2.0 - 1.7 * t
        val_loss = 2.1 - 1.6 * t
        metrics.append(
            MetricSnapshot(
                epoch=i,
                step=i * 100,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=1e-4,
                gradient_norm=1.0 - 0.5 * t,
            )
        )
    return metrics


def make_overfitting_metrics(epochs: int = 10) -> list[MetricSnapshot]:
    """Create metrics where train loss decreases but val loss rises after midpoint."""
    metrics: list[MetricSnapshot] = []
    mid = epochs // 2
    for i in range(epochs):
        t = i / max(epochs - 1, 1)
        train_loss = 2.0 - 1.8 * t  # keeps decreasing
        if i < mid:
            val_loss = 2.1 - 1.0 * t
        else:
            # val loss starts climbing after midpoint
            val_loss = 2.1 - 1.0 * (mid / max(epochs - 1, 1)) + 0.2 * (i - mid)
        metrics.append(
            MetricSnapshot(
                epoch=i,
                step=i * 100,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=1e-4,
                gradient_norm=1.0,
            )
        )
    return metrics


def make_plateaued_metrics(epochs: int = 10) -> list[MetricSnapshot]:
    """Create metrics where loss decreases then flattens."""
    metrics: list[MetricSnapshot] = []
    plateau_start = epochs // 3
    for i in range(epochs):
        if i < plateau_start:
            t = i / max(plateau_start - 1, 1)
            train_loss = 2.0 - 1.0 * t
        else:
            train_loss = 1.0 + 0.0001 * (i - plateau_start)  # essentially flat
        metrics.append(
            MetricSnapshot(
                epoch=i,
                step=i * 100,
                train_loss=train_loss,
                val_loss=train_loss + 0.05,
                learning_rate=1e-4,
            )
        )
    return metrics


def make_diverging_metrics(epochs: int = 10) -> list[MetricSnapshot]:
    """Create metrics where loss initially decreases then increases (diverges)."""
    metrics: list[MetricSnapshot] = []
    turn = epochs // 3
    for i in range(epochs):
        if i <= turn:
            t = i / max(turn, 1)
            train_loss = 2.0 - 0.5 * t
        else:
            # loss starts climbing
            train_loss = 1.5 + 0.3 * (i - turn)
        metrics.append(
            MetricSnapshot(
                epoch=i,
                step=i * 100,
                train_loss=train_loss,
                learning_rate=1e-4,
            )
        )
    return metrics


def make_spiking_metrics(epochs: int = 10) -> list[MetricSnapshot]:
    """Create mostly decreasing metrics with occasional large spikes."""
    metrics: list[MetricSnapshot] = []
    for i in range(epochs):
        t = i / max(epochs - 1, 1)
        train_loss = 2.0 - 1.5 * t
        # Inject spikes at epochs 3 and 7
        if i in (3, 7) and epochs > 7:
            train_loss *= 3.0
        metrics.append(
            MetricSnapshot(
                epoch=i,
                step=i * 100,
                train_loss=train_loss,
                val_loss=train_loss + 0.1,
                learning_rate=1e-4,
                gradient_norm=1.0 if i not in (3, 7) else 50.0,
            )
        )
    return metrics


# ===========================================================================
# TestMetricSnapshot
# ===========================================================================


class TestMetricSnapshot:
    """Tests for MetricSnapshot construction and serialization."""

    def test_construction_required_fields(self) -> None:
        """MetricSnapshot can be created with required fields only."""
        snap = MetricSnapshot(epoch=0, step=0, train_loss=1.5)
        assert snap.epoch == 0
        assert snap.step == 0
        assert snap.train_loss == 1.5
        assert snap.val_loss is None
        assert snap.learning_rate == 0.0
        assert snap.gradient_norm is None

    def test_construction_all_fields(self) -> None:
        """MetricSnapshot can be created with all fields."""
        now = datetime.now()
        snap = MetricSnapshot(
            epoch=5,
            step=500,
            train_loss=0.3,
            val_loss=0.4,
            learning_rate=1e-4,
            gradient_norm=1.2,
            timestamp=now,
        )
        assert snap.epoch == 5
        assert snap.val_loss == 0.4
        assert snap.learning_rate == 1e-4
        assert snap.gradient_norm == 1.2
        assert snap.timestamp == now

    def test_to_dict(self) -> None:
        """to_dict serializes all fields."""
        snap = MetricSnapshot(epoch=1, step=100, train_loss=1.0, val_loss=1.1)
        d = snap.to_dict()
        assert d["epoch"] == 1
        assert d["step"] == 100
        assert d["train_loss"] == 1.0
        assert d["val_loss"] == 1.1
        assert "timestamp" in d

    def test_from_dict_roundtrip(self) -> None:
        """from_dict restores a snapshot from its to_dict output."""
        snap = MetricSnapshot(epoch=3, step=300, train_loss=0.5, val_loss=0.6, learning_rate=2e-5)
        d = snap.to_dict()
        restored = MetricSnapshot.from_dict(d)
        assert restored.epoch == snap.epoch
        assert restored.train_loss == snap.train_loss
        assert restored.val_loss == snap.val_loss
        assert restored.learning_rate == snap.learning_rate

    def test_default_timestamp(self) -> None:
        """Default timestamp is approximately now."""
        before = datetime.now()
        snap = MetricSnapshot(epoch=0, step=0, train_loss=1.0)
        after = datetime.now()
        assert before <= snap.timestamp <= after


# ===========================================================================
# TestDiagnosticIssue
# ===========================================================================


class TestDiagnosticIssue:
    """Tests for DiagnosticIssue construction and severity."""

    def test_construction(self) -> None:
        """DiagnosticIssue stores all fields."""
        issue = DiagnosticIssue(
            category=IssueCategory.CONVERGENCE,
            severity=IssueSeverity.WARNING,
            title="Training seems stuck",
            description="The model stopped learning new patterns.",
            suggestion="Try adding more diverse examples.",
            evidence={"plateau_epochs": 5},
            detected_at_epoch=5,
        )
        assert issue.category == IssueCategory.CONVERGENCE
        assert issue.severity == IssueSeverity.WARNING
        assert issue.detected_at_epoch == 5

    def test_severity_levels_are_distinct(self) -> None:
        """All three severity levels are distinct values."""
        values = {
            IssueSeverity.INFO.value,
            IssueSeverity.WARNING.value,
            IssueSeverity.CRITICAL.value,
        }
        assert len(values) == 3

    def test_to_dict(self) -> None:
        """to_dict serializes the issue."""
        issue = DiagnosticIssue(
            category=IssueCategory.OVERFITTING,
            severity=IssueSeverity.CRITICAL,
            title="Memorizing data",
            description="The model is memorizing training data.",
            suggestion="Add more varied examples.",
            evidence={"gap": 0.5},
            detected_at_epoch=8,
        )
        d = issue.to_dict()
        assert d["category"] == "overfitting"
        assert d["severity"] == "critical"
        assert d["title"] == "Memorizing data"
        assert d["detected_at_epoch"] == 8
        assert d["evidence"]["gap"] == 0.5

    def test_all_categories_exist(self) -> None:
        """All expected issue categories are defined."""
        expected = {
            "CONVERGENCE",
            "OVERFITTING",
            "UNDERFITTING",
            "DATA_QUALITY",
            "CONFIGURATION",
            "STABILITY",
        }
        actual = {c.name for c in IssueCategory}
        assert expected == actual

    def test_all_severities_exist(self) -> None:
        """All expected severity levels are defined."""
        expected = {"INFO", "WARNING", "CRITICAL"}
        actual = {s.name for s in IssueSeverity}
        assert expected == actual


# ===========================================================================
# TestDiagnosticConfig
# ===========================================================================


class TestDiagnosticConfig:
    """Tests for DiagnosticConfig defaults and customization."""

    def test_defaults(self) -> None:
        """Default values match specification."""
        cfg = DiagnosticConfig()
        assert cfg.plateau_patience == 5
        assert cfg.plateau_threshold == 0.001
        assert cfg.overfit_gap_threshold == 0.3
        assert cfg.loss_spike_threshold == 2.0
        assert cfg.min_epochs_for_analysis == 3
        assert cfg.target_train_loss == 0.5

    def test_custom_values(self) -> None:
        """Custom values override defaults."""
        cfg = DiagnosticConfig(
            plateau_patience=10,
            overfit_gap_threshold=0.5,
        )
        assert cfg.plateau_patience == 10
        assert cfg.overfit_gap_threshold == 0.5
        # others stay default
        assert cfg.loss_spike_threshold == 2.0

    def test_to_dict(self) -> None:
        """to_dict serializes config."""
        cfg = DiagnosticConfig()
        d = cfg.to_dict()
        assert d["plateau_patience"] == 5
        assert d["target_train_loss"] == 0.5

    def test_from_dict_roundtrip(self) -> None:
        """from_dict restores config from to_dict output."""
        cfg = DiagnosticConfig(plateau_patience=8, overfit_gap_threshold=0.4)
        d = cfg.to_dict()
        restored = DiagnosticConfig.from_dict(d)
        assert restored.plateau_patience == 8
        assert restored.overfit_gap_threshold == 0.4


# ===========================================================================
# TestTrendAnalyzer
# ===========================================================================


class TestTrendAnalyzer:
    """Tests for TrendAnalyzer trend computation."""

    def test_decreasing_trend(self) -> None:
        """Detects a clear downward trend."""
        values = [2.0, 1.5, 1.0, 0.7, 0.5]
        analyzer = TrendAnalyzer()
        trend = analyzer.analyze(values, "train_loss")
        assert trend.is_decreasing is True
        assert trend.slope < 0
        assert trend.metric_name == "train_loss"

    def test_increasing_trend(self) -> None:
        """Detects an upward trend."""
        values = [0.5, 0.7, 1.0, 1.5, 2.0]
        analyzer = TrendAnalyzer()
        trend = analyzer.analyze(values, "val_loss")
        assert trend.is_decreasing is False
        assert trend.slope > 0

    def test_plateau_detection(self) -> None:
        """Detects a plateau when values stop changing."""
        values = [2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        analyzer = TrendAnalyzer()
        trend = analyzer.analyze(values, "train_loss")
        assert trend.is_plateaued is True
        assert trend.plateau_start_epoch is not None
        assert trend.plateau_start_epoch <= 3

    def test_no_plateau_when_decreasing(self) -> None:
        """No plateau when values keep decreasing."""
        values = [2.0, 1.8, 1.5, 1.2, 0.9, 0.6, 0.3, 0.1]
        analyzer = TrendAnalyzer()
        trend = analyzer.analyze(values, "train_loss")
        assert trend.is_plateaued is False
        assert trend.plateau_start_epoch is None

    def test_slope_computation_linear(self) -> None:
        """Slope of perfectly linear data is correct."""
        values = [10.0, 8.0, 6.0, 4.0, 2.0]  # slope = -2 per step
        analyzer = TrendAnalyzer()
        trend = analyzer.analyze(values, "test")
        assert abs(trend.slope - (-2.0)) < 0.01

    def test_single_value(self) -> None:
        """Single value has zero slope, not plateaued."""
        analyzer = TrendAnalyzer()
        trend = analyzer.analyze([1.0], "test")
        assert trend.slope == 0.0
        assert trend.is_plateaued is False

    def test_two_values(self) -> None:
        """Two values compute slope correctly."""
        analyzer = TrendAnalyzer()
        trend = analyzer.analyze([2.0, 1.0], "test")
        assert trend.slope < 0
        assert trend.is_decreasing is True

    def test_to_dict(self) -> None:
        """TrainingTrend serializes to dict."""
        analyzer = TrendAnalyzer()
        trend = analyzer.analyze([2.0, 1.5, 1.0], "train_loss")
        d = trend.to_dict()
        assert d["metric_name"] == "train_loss"
        assert "slope" in d
        assert "is_decreasing" in d
        assert "is_plateaued" in d


# ===========================================================================
# TestConvergenceChecker
# ===========================================================================


class TestConvergenceChecker:
    """Tests for ConvergenceChecker detecting convergence problems."""

    def test_healthy_training_no_issues(self, diagnostic_config: DiagnosticConfig) -> None:
        """Healthy training produces no convergence issues."""
        checker = ConvergenceChecker(diagnostic_config)
        metrics = make_healthy_metrics(epochs=10)
        issues = checker.check(metrics)
        assert len(issues) == 0

    def test_loss_plateau_detected(self, diagnostic_config: DiagnosticConfig) -> None:
        """Detects loss plateau when training stalls."""
        checker = ConvergenceChecker(diagnostic_config)
        metrics = make_plateaued_metrics(epochs=15)
        issues = checker.check(metrics)
        plateau_issues = [
            i for i in issues if "stuck" in i.title.lower() or "plateau" in i.title.lower()
        ]
        assert len(plateau_issues) >= 1
        assert plateau_issues[0].category == IssueCategory.CONVERGENCE

    def test_no_improvement_detected(self, diagnostic_config: DiagnosticConfig) -> None:
        """Detects when there is no meaningful improvement at all."""
        # Flat from the start
        metrics = [
            MetricSnapshot(epoch=i, step=i * 100, train_loss=2.0 + 0.001 * i) for i in range(10)
        ]
        checker = ConvergenceChecker(diagnostic_config)
        issues = checker.check(metrics)
        # Should find at least one convergence issue
        assert len(issues) >= 1
        cats = {i.category for i in issues}
        assert IssueCategory.CONVERGENCE in cats

    def test_divergence_detected(self, diagnostic_config: DiagnosticConfig) -> None:
        """Detects divergence when loss increases over time."""
        checker = ConvergenceChecker(diagnostic_config)
        metrics = make_diverging_metrics(epochs=10)
        issues = checker.check(metrics)
        diverge_issues = [
            i
            for i in issues
            if "wrong direction" in i.title.lower()
            or "diverge" in i.title.lower()
            or "increasing" in i.title.lower()
        ]
        assert len(diverge_issues) >= 1

    def test_too_few_epochs_skips_analysis(self) -> None:
        """With fewer epochs than min_epochs_for_analysis, returns no issues."""
        config = DiagnosticConfig(min_epochs_for_analysis=5)
        checker = ConvergenceChecker(config)
        metrics = [MetricSnapshot(epoch=i, step=i * 100, train_loss=2.0) for i in range(3)]
        issues = checker.check(metrics)
        assert len(issues) == 0

    def test_plain_language_in_description(self, diagnostic_config: DiagnosticConfig) -> None:
        """Issue descriptions use plain language, not ML jargon."""
        checker = ConvergenceChecker(diagnostic_config)
        metrics = make_plateaued_metrics(epochs=15)
        issues = checker.check(metrics)
        for issue in issues:
            # Should not contain raw ML jargon
            assert "gradient" not in issue.description.lower() or "norm" not in issue.description
            assert "perplexity" not in issue.description.lower()
            assert "f1" not in issue.description.lower()


# ===========================================================================
# TestOverfitDetector
# ===========================================================================


class TestOverfitDetector:
    """Tests for OverfitDetector detecting overfitting patterns."""

    def test_healthy_gap_no_issues(self, diagnostic_config: DiagnosticConfig) -> None:
        """Healthy train/val gap produces no overfitting issues."""
        detector = OverfitDetector(diagnostic_config)
        metrics = make_healthy_metrics(epochs=10)
        issues = detector.check(metrics)
        assert len(issues) == 0

    def test_overfitting_gap_detected(self, diagnostic_config: DiagnosticConfig) -> None:
        """Detects overfitting when val_loss - train_loss exceeds threshold."""
        detector = OverfitDetector(diagnostic_config)
        metrics = make_overfitting_metrics(epochs=10)
        issues = detector.check(metrics)
        overfit_issues = [i for i in issues if i.category == IssueCategory.OVERFITTING]
        assert len(overfit_issues) >= 1

    def test_val_loss_increasing_while_train_decreasing(
        self, diagnostic_config: DiagnosticConfig
    ) -> None:
        """Detects when val loss is increasing while train loss decreases."""
        detector = OverfitDetector(diagnostic_config)
        metrics = make_overfitting_metrics(epochs=10)
        issues = detector.check(metrics)
        assert len(issues) >= 1
        # Should mention memorizing or overfitting in plain language
        descriptions = " ".join(i.description.lower() for i in issues)
        assert "memoriz" in descriptions or "overfit" in descriptions or "pattern" in descriptions

    def test_skips_when_no_val_loss(self, diagnostic_config: DiagnosticConfig) -> None:
        """No issues when val_loss is not available."""
        detector = OverfitDetector(diagnostic_config)
        metrics = [
            MetricSnapshot(epoch=i, step=i * 100, train_loss=2.0 - 0.2 * i) for i in range(10)
        ]
        issues = detector.check(metrics)
        assert len(issues) == 0

    def test_suggestion_links_to_forge(self, diagnostic_config: DiagnosticConfig) -> None:
        """Overfitting suggestions mention adding examples or Forge."""
        detector = OverfitDetector(diagnostic_config)
        metrics = make_overfitting_metrics(epochs=10)
        issues = detector.check(metrics)
        if issues:
            suggestions = " ".join(i.suggestion.lower() for i in issues)
            assert "example" in suggestions or "forge" in suggestions or "curriculum" in suggestions


# ===========================================================================
# TestStabilityChecker
# ===========================================================================


class TestStabilityChecker:
    """Tests for StabilityChecker detecting training instability."""

    def test_stable_training_no_issues(self, diagnostic_config: DiagnosticConfig) -> None:
        """Stable training produces no stability issues."""
        checker = StabilityChecker(diagnostic_config)
        metrics = make_healthy_metrics(epochs=10)
        issues = checker.check(metrics)
        assert len(issues) == 0

    def test_loss_spike_detected(self, diagnostic_config: DiagnosticConfig) -> None:
        """Detects loss spikes above the threshold."""
        checker = StabilityChecker(diagnostic_config)
        metrics = make_spiking_metrics(epochs=10)
        issues = checker.check(metrics)
        spike_issues = [i for i in issues if i.category == IssueCategory.STABILITY]
        assert len(spike_issues) >= 1

    def test_gradient_explosion_detected(self, diagnostic_config: DiagnosticConfig) -> None:
        """Detects gradient explosion from large gradient norms."""
        checker = StabilityChecker(diagnostic_config)
        metrics = make_spiking_metrics(epochs=10)
        issues = checker.check(metrics)
        # Spiking metrics have gradient_norm=50 at spike epochs
        grad_issues = [
            i for i in issues if "unstable" in i.title.lower() or "erratic" in i.title.lower()
        ]
        assert len(grad_issues) >= 1

    def test_severity_is_warning_or_critical(self, diagnostic_config: DiagnosticConfig) -> None:
        """Stability issues are at least WARNING severity."""
        checker = StabilityChecker(diagnostic_config)
        metrics = make_spiking_metrics(epochs=10)
        issues = checker.check(metrics)
        for issue in issues:
            assert issue.severity in (IssueSeverity.WARNING, IssueSeverity.CRITICAL)


# ===========================================================================
# TestDataQualityChecker
# ===========================================================================


class TestDataQualityChecker:
    """Tests for DataQualityChecker evaluating curriculum statistics."""

    def test_sufficient_data_no_issues(self) -> None:
        """Sufficient curriculum data produces no issues."""
        checker = DataQualityChecker()
        stats: dict[str, Any] = {
            "total_examples": 300,
            "competency_counts": {
                "fault_isolation": 80,
                "procedural": 100,
                "parts_identification": 120,
            },
            "answer_lengths": [50, 60, 55, 70, 65] * 60,
        }
        issues = checker.check_curriculum_stats(stats)
        assert len(issues) == 0

    def test_small_curriculum_warning(self) -> None:
        """Warns when total examples are below 50."""
        checker = DataQualityChecker()
        stats: dict[str, Any] = {
            "total_examples": 30,
            "competency_counts": {"fault_isolation": 15, "procedural": 15},
            "answer_lengths": [50] * 30,
        }
        issues = checker.check_curriculum_stats(stats)
        size_issues = [i for i in issues if i.category == IssueCategory.DATA_QUALITY]
        assert len(size_issues) >= 1
        assert any(i.severity == IssueSeverity.WARNING for i in size_issues)

    def test_very_small_curriculum_critical(self) -> None:
        """Critical when total examples are below 20."""
        checker = DataQualityChecker()
        stats: dict[str, Any] = {
            "total_examples": 10,
            "competency_counts": {"fault_isolation": 5, "procedural": 5},
            "answer_lengths": [50] * 10,
        }
        issues = checker.check_curriculum_stats(stats)
        size_issues = [i for i in issues if i.category == IssueCategory.DATA_QUALITY]
        assert any(i.severity == IssueSeverity.CRITICAL for i in size_issues)

    def test_imbalanced_competencies(self) -> None:
        """Detects when competency distribution is heavily imbalanced."""
        checker = DataQualityChecker()
        stats: dict[str, Any] = {
            "total_examples": 200,
            "competency_counts": {
                "fault_isolation": 180,
                "procedural": 10,
                "parts_identification": 10,
            },
            "answer_lengths": [50] * 200,
        }
        issues = checker.check_curriculum_stats(stats)
        balance_issues = [
            i
            for i in issues
            if "balance" in i.title.lower()
            or "uneven" in i.title.lower()
            or "imbalance" in i.title.lower()
        ]
        assert len(balance_issues) >= 1

    def test_answer_length_variance(self) -> None:
        """Detects extreme variance in answer lengths."""
        checker = DataQualityChecker()
        stats: dict[str, Any] = {
            "total_examples": 100,
            "competency_counts": {"fault_isolation": 50, "procedural": 50},
            "answer_lengths": [5] * 50 + [10000] * 50,  # huge variance
        }
        issues = checker.check_curriculum_stats(stats)
        length_issues = [
            i for i in issues if "length" in i.title.lower() or "inconsistent" in i.title.lower()
        ]
        assert len(length_issues) >= 1

    def test_empty_stats_handled(self) -> None:
        """Handles missing or empty stats gracefully."""
        checker = DataQualityChecker()
        stats: dict[str, Any] = {
            "total_examples": 0,
            "competency_counts": {},
            "answer_lengths": [],
        }
        issues = checker.check_curriculum_stats(stats)
        assert len(issues) >= 1  # at least a size issue


# ===========================================================================
# TestDiagnosticReport
# ===========================================================================


class TestDiagnosticReport:
    """Tests for DiagnosticReport dataclass."""

    def test_healthy_report(self) -> None:
        """Report with no issues is healthy."""
        report = DiagnosticReport(
            issues=[],
            trends={},
            overall_health="healthy",
            plain_language_summary="Training is going well.",
            forge_recommendations=[],
        )
        assert report.overall_health == "healthy"
        assert len(report.issues) == 0

    def test_warning_report(self) -> None:
        """Report with WARNING issues has warning health."""
        issue = DiagnosticIssue(
            category=IssueCategory.CONVERGENCE,
            severity=IssueSeverity.WARNING,
            title="Training seems stuck",
            description="Not learning.",
            suggestion="Add examples.",
            evidence={},
            detected_at_epoch=5,
        )
        report = DiagnosticReport(
            issues=[issue],
            trends={},
            overall_health="warning",
            plain_language_summary="Some concerns.",
            forge_recommendations=["Add more examples in Forge."],
        )
        assert report.overall_health == "warning"

    def test_critical_report(self) -> None:
        """Report with CRITICAL issues has critical health."""
        issue = DiagnosticIssue(
            category=IssueCategory.OVERFITTING,
            severity=IssueSeverity.CRITICAL,
            title="Memorizing data",
            description="Overfitting.",
            suggestion="Diversify.",
            evidence={},
            detected_at_epoch=8,
        )
        report = DiagnosticReport(
            issues=[issue],
            trends={},
            overall_health="critical",
            plain_language_summary="Serious problems.",
            forge_recommendations=["Diversify curriculum."],
        )
        assert report.overall_health == "critical"

    def test_to_dict(self) -> None:
        """to_dict serializes the full report."""
        report = DiagnosticReport(
            issues=[],
            trends={},
            overall_health="healthy",
            plain_language_summary="All good.",
            forge_recommendations=[],
        )
        d = report.to_dict()
        assert d["overall_health"] == "healthy"
        assert d["plain_language_summary"] == "All good."
        assert "issues" in d
        assert "trends" in d
        assert "forge_recommendations" in d

    def test_report_with_trends(self) -> None:
        """Report includes serialized trends."""
        trend = TrainingTrend(
            metric_name="train_loss",
            values=[2.0, 1.5, 1.0],
            slope=-0.5,
            is_decreasing=True,
            is_plateaued=False,
            plateau_start_epoch=None,
        )
        report = DiagnosticReport(
            issues=[],
            trends={"train_loss": trend},
            overall_health="healthy",
            plain_language_summary="Good.",
            forge_recommendations=[],
        )
        d = report.to_dict()
        assert "train_loss" in d["trends"]
        assert d["trends"]["train_loss"]["slope"] == -0.5


# ===========================================================================
# TestTrainingDiagnostics
# ===========================================================================


class TestTrainingDiagnostics:
    """Tests for the main TrainingDiagnostics orchestrator."""

    def test_healthy_training_full_analysis(
        self, training_diagnostics: TrainingDiagnostics
    ) -> None:
        """Full analysis of healthy training returns healthy report."""
        metrics = make_healthy_metrics(epochs=10)
        report = training_diagnostics.analyze_training(metrics)
        assert report.overall_health == "healthy"
        assert len(report.issues) == 0
        assert "train_loss" in report.trends

    def test_problematic_training_full_analysis(
        self, training_diagnostics: TrainingDiagnostics
    ) -> None:
        """Full analysis detects overfitting in problematic metrics."""
        metrics = make_overfitting_metrics(epochs=10)
        report = training_diagnostics.analyze_training(metrics)
        assert report.overall_health in ("warning", "critical")
        assert len(report.issues) >= 1

    def test_curriculum_only_analysis(self, training_diagnostics: TrainingDiagnostics) -> None:
        """Curriculum-only analysis works without training metrics."""
        stats: dict[str, Any] = {
            "total_examples": 15,
            "competency_counts": {"fault_isolation": 15},
            "answer_lengths": [50] * 15,
        }
        report = training_diagnostics.analyze_curriculum(stats)
        assert report.overall_health in ("warning", "critical")
        assert len(report.issues) >= 1

    def test_full_analysis_combines_training_and_curriculum(
        self, training_diagnostics: TrainingDiagnostics
    ) -> None:
        """Full analysis combines training metrics and curriculum stats."""
        metrics = make_overfitting_metrics(epochs=10)
        stats: dict[str, Any] = {
            "total_examples": 30,
            "competency_counts": {"fault_isolation": 15, "procedural": 15},
            "answer_lengths": [50] * 30,
        }
        report = training_diagnostics.analyze_full(metrics, stats)
        assert len(report.issues) >= 2  # at least one training + one data issue

    def test_trends_computed_for_available_metrics(
        self, training_diagnostics: TrainingDiagnostics
    ) -> None:
        """Trends are computed for train_loss and val_loss when available."""
        metrics = make_healthy_metrics(epochs=10)
        report = training_diagnostics.analyze_training(metrics)
        assert "train_loss" in report.trends
        assert "val_loss" in report.trends

    def test_trends_only_train_when_no_val(self, training_diagnostics: TrainingDiagnostics) -> None:
        """Only train_loss trend when val_loss is absent."""
        metrics = [
            MetricSnapshot(epoch=i, step=i * 100, train_loss=2.0 - 0.2 * i) for i in range(10)
        ]
        report = training_diagnostics.analyze_training(metrics)
        assert "train_loss" in report.trends
        assert "val_loss" not in report.trends

    def test_plain_language_summary_generated(
        self, training_diagnostics: TrainingDiagnostics
    ) -> None:
        """Summary is non-empty plain language."""
        metrics = make_healthy_metrics(epochs=10)
        report = training_diagnostics.analyze_training(metrics)
        assert len(report.plain_language_summary) > 0
        # No ML jargon in summary
        summary_lower = report.plain_language_summary.lower()
        assert "perplexity" not in summary_lower
        assert "f1 score" not in summary_lower

    def test_forge_recommendations_for_problems(
        self, training_diagnostics: TrainingDiagnostics
    ) -> None:
        """Forge recommendations generated for problematic training."""
        metrics = make_overfitting_metrics(epochs=10)
        report = training_diagnostics.analyze_training(metrics)
        assert len(report.forge_recommendations) >= 1

    def test_empty_metrics_raises(self, training_diagnostics: TrainingDiagnostics) -> None:
        """Empty metrics list raises DiagnosticsError."""
        with pytest.raises(DiagnosticsError):
            training_diagnostics.analyze_training([])

    def test_custom_config(self) -> None:
        """TrainingDiagnostics works with custom config."""
        config = DiagnosticConfig(
            plateau_patience=3,
            overfit_gap_threshold=0.5,
        )
        diag = TrainingDiagnostics(config=config)
        metrics = make_healthy_metrics(epochs=10)
        report = diag.analyze_training(metrics)
        assert report.overall_health == "healthy"


# ===========================================================================
# TestIntegration
# ===========================================================================


class TestIntegration:
    """Integration tests simulating realistic training scenarios."""

    def test_gradual_overfit_detection(self) -> None:
        """Simulate a realistic training run that gradually overfits.

        Training loss keeps improving but val loss diverges. The diagnostics
        should detect overfitting and suggest curriculum improvements.
        """
        epochs = 20
        metrics: list[MetricSnapshot] = []
        for i in range(epochs):
            t = i / (epochs - 1)
            train_loss = 2.0 * (1 - t) ** 1.5  # smooth decrease toward 0
            # Val loss decreases then increases
            if i < 8:
                val_loss = 2.1 - 0.15 * i
            else:
                val_loss = 0.9 + 0.08 * (i - 8)
            metrics.append(
                MetricSnapshot(
                    epoch=i,
                    step=i * 100,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=1e-4 * (1 - 0.5 * t),
                    gradient_norm=1.0 + 0.1 * i,
                )
            )

        diag = TrainingDiagnostics()
        report = diag.analyze_training(metrics)

        # Should detect overfitting
        assert report.overall_health in ("warning", "critical")
        overfit_issues = [i for i in report.issues if i.category == IssueCategory.OVERFITTING]
        assert len(overfit_issues) >= 1

        # Should have forge recommendations
        assert len(report.forge_recommendations) >= 1

        # Summary should be plain language
        assert len(report.plain_language_summary) > 20

    def test_healthy_run_end_to_end(self) -> None:
        """A well-behaved training run gets a clean bill of health."""
        metrics = make_healthy_metrics(epochs=15)
        stats: dict[str, Any] = {
            "total_examples": 350,
            "competency_counts": {
                "fault_isolation": 120,
                "procedural": 120,
                "parts_identification": 110,
            },
            "answer_lengths": [60, 80, 70, 90, 75] * 70,
        }

        diag = TrainingDiagnostics()
        report = diag.analyze_full(metrics, stats)
        assert report.overall_health == "healthy"
        assert len(report.issues) == 0

    def test_multiple_issues_combined(self) -> None:
        """Training with spikes + small curriculum reports multiple issues."""
        metrics = make_spiking_metrics(epochs=10)
        stats: dict[str, Any] = {
            "total_examples": 15,
            "competency_counts": {"fault_isolation": 15},
            "answer_lengths": [50] * 15,
        }

        diag = TrainingDiagnostics()
        report = diag.analyze_full(metrics, stats)
        assert report.overall_health in ("warning", "critical")
        categories = {i.category for i in report.issues}
        # Should have both stability and data quality issues
        assert IssueCategory.STABILITY in categories
        assert IssueCategory.DATA_QUALITY in categories

    def test_diagnostics_error_raised(self) -> None:
        """DiagnosticsError is a proper Exception subclass."""
        with pytest.raises(DiagnosticsError):
            raise DiagnosticsError("test error")

    def test_report_serialization_roundtrip(self) -> None:
        """Full report can be serialized to dict for API responses."""
        metrics = make_overfitting_metrics(epochs=10)
        diag = TrainingDiagnostics()
        report = diag.analyze_training(metrics)
        d = report.to_dict()

        assert isinstance(d, dict)
        assert isinstance(d["issues"], list)
        assert isinstance(d["trends"], dict)
        assert isinstance(d["forge_recommendations"], list)
        assert isinstance(d["overall_health"], str)
        assert isinstance(d["plain_language_summary"], str)
