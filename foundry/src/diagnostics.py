"""Failure detection and training diagnostics for Foundry.

Monitors training metric logs, detects issues (loss not converging,
overfitting, instability), and provides **plain-language** guidance
for fixes. All user-facing text avoids ML jargon and links back to
Forge when curriculum changes could help.

Example::

    from foundry.src.diagnostics import TrainingDiagnostics, MetricSnapshot

    metrics = [MetricSnapshot(epoch=i, step=i*100, train_loss=...) for i in range(20)]
    diag = TrainingDiagnostics()
    report = diag.analyze_training(metrics)
    print(report.plain_language_summary)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class DiagnosticsError(Exception):
    """Raised for diagnostics workflow errors."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class IssueSeverity(str, Enum):
    """Severity level for a diagnostic issue.

    Values are ordered so string comparison reflects importance:
    INFO < WARNING < CRITICAL.
    """

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class IssueCategory(str, Enum):
    """Category of a detected training issue.

    Attributes:
        CONVERGENCE: Training is not converging toward a solution.
        OVERFITTING: Model is memorizing rather than learning patterns.
        UNDERFITTING: Model is not learning enough from the data.
        DATA_QUALITY: Curriculum data has quality or quantity problems.
        CONFIGURATION: Training configuration may need adjustment.
        STABILITY: Training process is numerically unstable.
    """

    CONVERGENCE = "convergence"
    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"
    DATA_QUALITY = "data_quality"
    CONFIGURATION = "configuration"
    STABILITY = "stability"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticIssue:
    """A single diagnostic issue detected during training analysis.

    All user-facing fields (title, description, suggestion) use plain
    language understandable by domain experts, not ML practitioners.

    Attributes:
        category: Type of issue.
        severity: How serious the issue is.
        title: Short plain-language title.
        description: Detailed plain-language explanation.
        suggestion: Actionable fix suggestion.
        evidence: Supporting data for technical review.
        detected_at_epoch: Epoch where the issue was detected.
    """

    category: IssueCategory
    severity: IssueSeverity
    title: str
    description: str
    suggestion: str
    evidence: dict[str, Any]
    detected_at_epoch: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "suggestion": self.suggestion,
            "evidence": self.evidence,
            "detected_at_epoch": self.detected_at_epoch,
        }


@dataclass
class MetricSnapshot:
    """A snapshot of training metrics at a single point in time.

    Attributes:
        epoch: Current training epoch.
        step: Current training step.
        train_loss: Training loss value.
        val_loss: Validation loss value (None if not available).
        learning_rate: Current learning rate.
        gradient_norm: Gradient norm (None if not tracked).
        timestamp: When this snapshot was recorded.
    """

    epoch: int
    step: int
    train_loss: float
    val_loss: float | None = None
    learning_rate: float = 0.0
    gradient_norm: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "epoch": self.epoch,
            "step": self.step,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "learning_rate": self.learning_rate,
            "gradient_norm": self.gradient_norm,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricSnapshot:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with metric snapshot fields.

        Returns:
            Restored MetricSnapshot.
        """
        return cls(
            epoch=data["epoch"],
            step=data["step"],
            train_loss=data["train_loss"],
            val_loss=data.get("val_loss"),
            learning_rate=data.get("learning_rate", 0.0),
            gradient_norm=data.get("gradient_norm"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class TrainingTrend:
    """Computed trend for a single metric over time.

    Attributes:
        metric_name: Name of the metric being tracked.
        values: Raw metric values in epoch order.
        slope: Linear regression slope (positive = increasing).
        is_decreasing: Whether the overall trend is downward.
        is_plateaued: Whether the metric has stopped changing.
        plateau_start_epoch: Epoch where plateau began (None if no plateau).
    """

    metric_name: str
    values: list[float]
    slope: float
    is_decreasing: bool
    is_plateaued: bool
    plateau_start_epoch: int | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "metric_name": self.metric_name,
            "values": self.values,
            "slope": self.slope,
            "is_decreasing": self.is_decreasing,
            "is_plateaued": self.is_plateaued,
            "plateau_start_epoch": self.plateau_start_epoch,
        }


@dataclass
class DiagnosticConfig:
    """Configuration for diagnostic thresholds.

    Attributes:
        plateau_patience: Epochs without improvement before flagging.
        plateau_threshold: Minimum change to count as improvement.
        overfit_gap_threshold: val_loss - train_loss gap to flag overfitting.
        loss_spike_threshold: Multiplier vs moving average to flag instability.
        min_epochs_for_analysis: Minimum epochs before analysis is meaningful.
        target_train_loss: Below this is considered healthy.
    """

    plateau_patience: int = 5
    plateau_threshold: float = 0.001
    overfit_gap_threshold: float = 0.3
    loss_spike_threshold: float = 2.0
    min_epochs_for_analysis: int = 3
    target_train_loss: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "plateau_patience": self.plateau_patience,
            "plateau_threshold": self.plateau_threshold,
            "overfit_gap_threshold": self.overfit_gap_threshold,
            "loss_spike_threshold": self.loss_spike_threshold,
            "min_epochs_for_analysis": self.min_epochs_for_analysis,
            "target_train_loss": self.target_train_loss,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagnosticConfig:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with config fields.

        Returns:
            Restored DiagnosticConfig.
        """
        return cls(
            plateau_patience=data.get("plateau_patience", 5),
            plateau_threshold=data.get("plateau_threshold", 0.001),
            overfit_gap_threshold=data.get("overfit_gap_threshold", 0.3),
            loss_spike_threshold=data.get("loss_spike_threshold", 2.0),
            min_epochs_for_analysis=data.get("min_epochs_for_analysis", 3),
            target_train_loss=data.get("target_train_loss", 0.5),
        )


@dataclass
class DiagnosticReport:
    """Complete diagnostic report for a training run.

    Attributes:
        issues: All detected issues.
        trends: Computed trends by metric name.
        overall_health: One of "healthy", "warning", "critical".
        plain_language_summary: Human-readable summary.
        forge_recommendations: Suggestions linking to Forge actions.
    """

    issues: list[DiagnosticIssue]
    trends: dict[str, TrainingTrend]
    overall_health: str
    plain_language_summary: str
    forge_recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "issues": [i.to_dict() for i in self.issues],
            "trends": {k: v.to_dict() for k, v in self.trends.items()},
            "overall_health": self.overall_health,
            "plain_language_summary": self.plain_language_summary,
            "forge_recommendations": self.forge_recommendations,
        }


# ---------------------------------------------------------------------------
# TrendAnalyzer
# ---------------------------------------------------------------------------


class TrendAnalyzer:
    """Analyzes metric value sequences to compute trends.

    Computes linear regression slope, detects plateaus, and
    determines overall direction of a metric over time.
    """

    def analyze(self, values: list[float], metric_name: str) -> TrainingTrend:
        """Analyze a sequence of metric values.

        Args:
            values: Metric values in epoch order.
            metric_name: Name for the resulting trend.

        Returns:
            TrainingTrend with computed slope and plateau info.
        """
        slope = self._compute_slope(values)
        is_plateaued, plateau_start = self._detect_plateau(values)
        return TrainingTrend(
            metric_name=metric_name,
            values=values,
            slope=slope,
            is_decreasing=slope < 0,
            is_plateaued=is_plateaued,
            plateau_start_epoch=plateau_start,
        )

    @staticmethod
    def _compute_slope(values: list[float]) -> float:
        """Compute linear regression slope using least squares.

        Args:
            values: Sequence of values.

        Returns:
            Slope of the best-fit line. Zero for fewer than 2 values.
        """
        n = len(values)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _detect_plateau(
        values: list[float],
        threshold: float = 0.001,
        patience: int = 5,
    ) -> tuple[bool, int | None]:
        """Detect whether a metric has plateaued.

        A plateau is detected when the absolute change between
        consecutive values stays below the threshold for at least
        ``patience`` consecutive epochs.

        Args:
            values: Sequence of values.
            threshold: Minimum change to count as improvement.
            patience: Consecutive flat epochs to trigger plateau.

        Returns:
            Tuple of (is_plateaued, plateau_start_epoch).
        """
        if len(values) < 2:
            return False, None
        run_start = 0
        run_length = 0
        for i in range(1, len(values)):
            if abs(values[i] - values[i - 1]) < threshold:
                if run_length == 0:
                    run_start = i - 1
                run_length += 1
                if run_length >= patience:
                    return True, run_start
            else:
                run_length = 0
        return False, None


# ---------------------------------------------------------------------------
# ConvergenceChecker
# ---------------------------------------------------------------------------


class ConvergenceChecker:
    """Detects convergence problems in training metrics.

    Checks for loss plateaus, lack of improvement, and divergence.

    Args:
        config: Diagnostic configuration thresholds.
    """

    def __init__(self, config: DiagnosticConfig) -> None:
        self._config = config

    def check(self, metrics: list[MetricSnapshot]) -> list[DiagnosticIssue]:
        """Run all convergence checks.

        Args:
            metrics: Training metric snapshots in epoch order.

        Returns:
            List of detected convergence issues.
        """
        if len(metrics) < self._config.min_epochs_for_analysis:
            return []
        train_losses = [m.train_loss for m in metrics]
        issues: list[DiagnosticIssue] = []
        self._append_if_found(issues, self._check_loss_plateau(train_losses))
        self._append_if_found(issues, self._check_no_improvement(train_losses))
        self._append_if_found(issues, self._check_divergence(train_losses))
        return issues

    def _check_loss_plateau(self, train_losses: list[float]) -> DiagnosticIssue | None:
        """Check if training loss has plateaued.

        Args:
            train_losses: Training loss values.

        Returns:
            DiagnosticIssue if plateau detected, else None.
        """
        _, plateau_info = TrendAnalyzer._detect_plateau(
            train_losses,
            threshold=self._config.plateau_threshold,
            patience=self._config.plateau_patience,
        )
        if plateau_info is None:
            return None
        return DiagnosticIssue(
            category=IssueCategory.CONVERGENCE,
            severity=IssueSeverity.WARNING,
            title="Training seems stuck",
            description=(
                "The model has stopped making progress. "
                "It has been training for several rounds without "
                "meaningful improvement, which suggests it may need "
                "different or more diverse examples to learn from."
            ),
            suggestion=(
                "Try adding more diverse examples to your curriculum in Forge. "
                "Focus on areas where the model is weakest."
            ),
            evidence={
                "plateau_start_epoch": plateau_info,
                "patience": self._config.plateau_patience,
            },
            detected_at_epoch=plateau_info + self._config.plateau_patience,
        )

    def _check_no_improvement(self, train_losses: list[float]) -> DiagnosticIssue | None:
        """Check if there is no meaningful improvement from start to end.

        Args:
            train_losses: Training loss values.

        Returns:
            DiagnosticIssue if no improvement, else None.
        """
        if len(train_losses) < 2:
            return None
        improvement = train_losses[0] - train_losses[-1]
        relative_change = improvement / max(abs(train_losses[0]), 1e-8)
        if relative_change < 0.05:
            return DiagnosticIssue(
                category=IssueCategory.CONVERGENCE,
                severity=IssueSeverity.WARNING,
                title="Training is not making progress",
                description=(
                    "After multiple training rounds, the model has not "
                    "shown meaningful improvement. This may indicate the "
                    "examples are too similar or the model needs different "
                    "types of training data."
                ),
                suggestion=(
                    "Review your curriculum in Forge. Consider adding "
                    "examples that cover different aspects of the discipline, "
                    "or check that your examples are clear and consistent."
                ),
                evidence={
                    "start_loss": train_losses[0],
                    "end_loss": train_losses[-1],
                    "relative_improvement": relative_change,
                },
                detected_at_epoch=len(train_losses) - 1,
            )
        return None

    def _check_divergence(self, train_losses: list[float]) -> DiagnosticIssue | None:
        """Check if loss is increasing overall (divergence).

        Args:
            train_losses: Training loss values.

        Returns:
            DiagnosticIssue if divergence detected, else None.
        """
        if len(train_losses) < 3:
            return None
        # Check the second half of training
        mid = len(train_losses) // 2
        second_half = train_losses[mid:]
        slope = TrendAnalyzer._compute_slope(second_half)
        if slope > 0.05:
            return DiagnosticIssue(
                category=IssueCategory.CONVERGENCE,
                severity=IssueSeverity.CRITICAL,
                title="Training is going in the wrong direction",
                description=(
                    "The model's performance is getting worse instead of "
                    "better. This usually means the training settings need "
                    "adjustment, or there may be conflicting examples in "
                    "the curriculum."
                ),
                suggestion=(
                    "Try reducing the learning speed in your training "
                    "configuration. Also review your curriculum in Forge "
                    "for contradictory or confusing examples."
                ),
                evidence={
                    "second_half_slope": slope,
                    "mid_epoch": mid,
                },
                detected_at_epoch=len(train_losses) - 1,
            )
        return None

    @staticmethod
    def _append_if_found(
        issues: list[DiagnosticIssue],
        issue: DiagnosticIssue | None,
    ) -> None:
        """Append issue to list if it is not None.

        Args:
            issues: Target list.
            issue: Issue to append (or None).
        """
        if issue is not None:
            issues.append(issue)


# ---------------------------------------------------------------------------
# OverfitDetector
# ---------------------------------------------------------------------------


class OverfitDetector:
    """Detects overfitting patterns by comparing train and validation loss.

    Args:
        config: Diagnostic configuration thresholds.
    """

    def __init__(self, config: DiagnosticConfig) -> None:
        self._config = config

    def check(self, metrics: list[MetricSnapshot]) -> list[DiagnosticIssue]:
        """Run all overfitting checks.

        Args:
            metrics: Training metric snapshots in epoch order.

        Returns:
            List of detected overfitting issues.
        """
        if len(metrics) < self._config.min_epochs_for_analysis:
            return []
        # Skip if no validation loss available
        if not any(m.val_loss is not None for m in metrics):
            return []
        issues: list[DiagnosticIssue] = []
        gap_issue = self._check_val_train_gap(metrics)
        if gap_issue is not None:
            issues.append(gap_issue)
        val_losses = [m.val_loss for m in metrics if m.val_loss is not None]
        rising_issue = self._check_val_loss_increasing(val_losses)
        if rising_issue is not None:
            issues.append(rising_issue)
        return issues

    def _check_val_train_gap(self, metrics: list[MetricSnapshot]) -> DiagnosticIssue | None:
        """Check if val_loss - train_loss exceeds the overfit threshold.

        Args:
            metrics: Training metric snapshots.

        Returns:
            DiagnosticIssue if gap too large, else None.
        """
        paired = [(m.train_loss, m.val_loss) for m in metrics if m.val_loss is not None]
        if not paired:
            return None
        # Check the last few epochs for gap
        recent = paired[-max(3, len(paired) // 3) :]
        max_gap = max(v - t for t, v in recent)
        if max_gap >= self._config.overfit_gap_threshold:
            return DiagnosticIssue(
                category=IssueCategory.OVERFITTING,
                severity=IssueSeverity.CRITICAL,
                title="The model is memorizing instead of learning",
                description=(
                    "The model performs well on training examples but "
                    "poorly on new ones it has not seen before. This means "
                    "it is memorizing specific answers rather than learning "
                    "the underlying patterns of the discipline."
                ),
                suggestion=(
                    "Add more varied examples to your curriculum in Forge. "
                    "Include different phrasings of similar questions, and "
                    "ensure each competency area has enough diverse examples."
                ),
                evidence={
                    "max_gap": max_gap,
                    "threshold": self._config.overfit_gap_threshold,
                },
                detected_at_epoch=metrics[-1].epoch,
            )
        return None

    def _check_val_loss_increasing(self, val_losses: list[float]) -> DiagnosticIssue | None:
        """Check if validation loss is trending upward.

        Args:
            val_losses: Validation loss values.

        Returns:
            DiagnosticIssue if val loss rising, else None.
        """
        if len(val_losses) < 4:
            return None
        # Check second half trend
        mid = len(val_losses) // 2
        second_half = val_losses[mid:]
        slope = TrendAnalyzer._compute_slope(second_half)
        if slope > 0.01:
            return DiagnosticIssue(
                category=IssueCategory.OVERFITTING,
                severity=IssueSeverity.WARNING,
                title="Performance on new examples is declining",
                description=(
                    "While the model keeps improving on its training "
                    "examples, it is getting worse at handling new ones. "
                    "This is a sign that it is starting to memorize rather "
                    "than learn general patterns."
                ),
                suggestion=(
                    "Consider stopping training earlier, or add more "
                    "examples to your Forge curriculum to give the model "
                    "more variety to learn from."
                ),
                evidence={
                    "val_loss_slope_second_half": slope,
                },
                detected_at_epoch=len(val_losses) - 1,
            )
        return None


# ---------------------------------------------------------------------------
# StabilityChecker
# ---------------------------------------------------------------------------


class StabilityChecker:
    """Detects training instability from loss spikes and gradient issues.

    Args:
        config: Diagnostic configuration thresholds.
    """

    def __init__(self, config: DiagnosticConfig) -> None:
        self._config = config

    def check(self, metrics: list[MetricSnapshot]) -> list[DiagnosticIssue]:
        """Run all stability checks.

        Args:
            metrics: Training metric snapshots in epoch order.

        Returns:
            List of detected stability issues.
        """
        if len(metrics) < self._config.min_epochs_for_analysis:
            return []
        train_losses = [m.train_loss for m in metrics]
        issues = self._check_loss_spikes(train_losses)
        grad_norms = [m.gradient_norm for m in metrics if m.gradient_norm is not None]
        grad_issue = self._check_gradient_explosion(grad_norms)
        if grad_issue is not None:
            issues.append(grad_issue)
        return issues

    def _check_loss_spikes(self, train_losses: list[float]) -> list[DiagnosticIssue]:
        """Check for sudden spikes in training loss.

        A spike is detected when a value exceeds the moving average
        by more than ``loss_spike_threshold`` times.

        Args:
            train_losses: Training loss values.

        Returns:
            List of spike issues (one per spike).
        """
        issues: list[DiagnosticIssue] = []
        if len(train_losses) < 3:
            return issues
        window = 3
        for i in range(window, len(train_losses)):
            avg = sum(train_losses[i - window : i]) / window
            if avg > 0 and train_losses[i] > avg * self._config.loss_spike_threshold:
                issues.append(
                    DiagnosticIssue(
                        category=IssueCategory.STABILITY,
                        severity=IssueSeverity.WARNING,
                        title="Training had an unstable moment",
                        description=(
                            "The model's learning suddenly jumped at one "
                            "point, which can disrupt progress. This may be "
                            "caused by unusual examples or training settings "
                            "that are too aggressive."
                        ),
                        suggestion=(
                            "Check your curriculum for any unusual or "
                            "outlier examples. You may also want to reduce "
                            "the learning rate in your training configuration."
                        ),
                        evidence={
                            "spike_epoch": i,
                            "spike_value": train_losses[i],
                            "moving_avg": avg,
                        },
                        detected_at_epoch=i,
                    )
                )
        return issues

    def _check_gradient_explosion(self, gradient_norms: list[float]) -> DiagnosticIssue | None:
        """Check for gradient explosion from extremely large norms.

        Args:
            gradient_norms: Gradient norm values.

        Returns:
            DiagnosticIssue if explosion detected, else None.
        """
        if len(gradient_norms) < 2:
            return None
        median = sorted(gradient_norms)[len(gradient_norms) // 2]
        if median == 0:
            return None
        max_norm = max(gradient_norms)
        if max_norm > median * 10:
            return DiagnosticIssue(
                category=IssueCategory.STABILITY,
                severity=IssueSeverity.CRITICAL,
                title="Training became erratic",
                description=(
                    "The model experienced sudden, extreme changes "
                    "during training that could cause it to lose "
                    "everything it has learned so far. This is usually "
                    "caused by training settings that are too aggressive."
                ),
                suggestion=(
                    "Try reducing the learning rate or enabling gradient "
                    "clipping in your training configuration. If the "
                    "problem persists, review your curriculum in Forge "
                    "for any unusual examples."
                ),
                evidence={
                    "max_gradient_norm": max_norm,
                    "median_gradient_norm": median,
                    "ratio": max_norm / median,
                },
                detected_at_epoch=gradient_norms.index(max_norm),
            )
        return None


# ---------------------------------------------------------------------------
# DataQualityChecker
# ---------------------------------------------------------------------------


class DataQualityChecker:
    """Evaluates curriculum statistics for data quality issues.

    Works on aggregate statistics from the Forge curriculum,
    not on individual examples.
    """

    def check_curriculum_stats(self, stats: dict[str, Any]) -> list[DiagnosticIssue]:
        """Check curriculum statistics for quality issues.

        Args:
            stats: Dictionary with keys:
                - total_examples: int
                - competency_counts: dict[str, int]
                - answer_lengths: list[int]

        Returns:
            List of detected data quality issues.
        """
        issues: list[DiagnosticIssue] = []
        total = stats.get("total_examples", 0)
        size_issue = self._check_size(total)
        if size_issue is not None:
            issues.append(size_issue)
        counts = stats.get("competency_counts", {})
        balance_issue = self._check_competency_balance(counts)
        if balance_issue is not None:
            issues.append(balance_issue)
        lengths = stats.get("answer_lengths", [])
        length_issue = self._check_answer_length_variance(lengths)
        if length_issue is not None:
            issues.append(length_issue)
        return issues

    @staticmethod
    def _check_size(total: int) -> DiagnosticIssue | None:
        """Check if curriculum has enough examples.

        Args:
            total: Total number of examples.

        Returns:
            DiagnosticIssue if too few, else None.
        """
        if total < 20:
            return DiagnosticIssue(
                category=IssueCategory.DATA_QUALITY,
                severity=IssueSeverity.CRITICAL,
                title="Not enough training examples",
                description=(
                    "The curriculum has very few examples. The model needs "
                    "substantially more examples to learn the discipline "
                    "effectively. With so few examples, results will be "
                    "unreliable."
                ),
                suggestion=(
                    "Use Forge to add more examples. Aim for at least "
                    "50 examples to start, and 300-500 for a well-trained "
                    "model."
                ),
                evidence={"total_examples": total, "minimum_recommended": 50},
                detected_at_epoch=0,
            )
        if total < 50:
            return DiagnosticIssue(
                category=IssueCategory.DATA_QUALITY,
                severity=IssueSeverity.WARNING,
                title="Curriculum could use more examples",
                description=(
                    "The curriculum has a limited number of examples. "
                    "While training can proceed, results may improve "
                    "significantly with more data. Aim for 300-500 "
                    "examples for best results."
                ),
                suggestion=(
                    "Use Forge to add more examples, focusing on "
                    "competency areas that have the fewest examples."
                ),
                evidence={"total_examples": total, "recommended": 300},
                detected_at_epoch=0,
            )
        return None

    @staticmethod
    def _check_competency_balance(
        counts: dict[str, int],
    ) -> DiagnosticIssue | None:
        """Check if competency distribution is balanced.

        Args:
            counts: Mapping of competency name to example count.

        Returns:
            DiagnosticIssue if heavily imbalanced, else None.
        """
        if len(counts) < 2:
            return None
        values = list(counts.values())
        total = sum(values)
        if total == 0:
            return None
        max_ratio = max(values) / total
        if max_ratio > 0.7:
            dominant = max(counts, key=counts.get)  # type: ignore[arg-type]
            return DiagnosticIssue(
                category=IssueCategory.DATA_QUALITY,
                severity=IssueSeverity.WARNING,
                title="Uneven coverage across competency areas",
                description=(
                    f"Most examples are concentrated in '{dominant}'. "
                    "The model will learn that area well but may struggle "
                    "with others. Balanced coverage across all competency "
                    "areas leads to better overall performance."
                ),
                suggestion=(
                    "Use Forge to add more examples for underrepresented "
                    "competency areas. Check the coverage report to see "
                    "which areas need attention."
                ),
                evidence={
                    "dominant_competency": dominant,
                    "dominant_ratio": max_ratio,
                    "counts": counts,
                },
                detected_at_epoch=0,
            )
        return None

    @staticmethod
    def _check_answer_length_variance(
        lengths: list[int],
    ) -> DiagnosticIssue | None:
        """Check for extreme variance in answer lengths.

        Args:
            lengths: List of answer character counts.

        Returns:
            DiagnosticIssue if variance is extreme, else None.
        """
        if len(lengths) < 5:
            return None
        mean_len = sum(lengths) / len(lengths)
        if mean_len == 0:
            return None
        variance = sum((x - mean_len) ** 2 for x in lengths) / len(lengths)
        std_dev = variance**0.5
        cv = std_dev / mean_len  # coefficient of variation
        if cv > 0.8:
            return DiagnosticIssue(
                category=IssueCategory.DATA_QUALITY,
                severity=IssueSeverity.WARNING,
                title="Inconsistent answer lengths in curriculum",
                description=(
                    "The answers in your curriculum vary greatly in length. "
                    "Some are very short while others are very long. "
                    "Consistent answer lengths help the model learn the "
                    "expected level of detail for responses."
                ),
                suggestion=(
                    "Review examples in Forge and aim for consistent "
                    "answer lengths. Either expand short answers or "
                    "summarize overly long ones to find a consistent "
                    "level of detail."
                ),
                evidence={
                    "mean_length": mean_len,
                    "std_dev": std_dev,
                    "coefficient_of_variation": cv,
                },
                detected_at_epoch=0,
            )
        return None


# ---------------------------------------------------------------------------
# TrainingDiagnostics — main orchestrator
# ---------------------------------------------------------------------------


class TrainingDiagnostics:
    """Main orchestrator for training diagnostics.

    Combines all checkers and analyzers to produce a comprehensive
    diagnostic report with plain-language summaries.

    Args:
        config: Diagnostic configuration (uses defaults if not provided).

    Example::

        diag = TrainingDiagnostics()
        report = diag.analyze_training(metrics)
        print(report.plain_language_summary)
    """

    def __init__(self, config: DiagnosticConfig | None = None) -> None:
        self._config = config or DiagnosticConfig()
        self._trend_analyzer = TrendAnalyzer()
        self._convergence = ConvergenceChecker(self._config)
        self._overfit = OverfitDetector(self._config)
        self._stability = StabilityChecker(self._config)
        self._data_quality = DataQualityChecker()

    def analyze_training(self, metrics: list[MetricSnapshot]) -> DiagnosticReport:
        """Analyze training metrics and produce a diagnostic report.

        Args:
            metrics: Training metric snapshots in epoch order.

        Returns:
            DiagnosticReport with issues, trends, and recommendations.

        Raises:
            DiagnosticsError: If metrics list is empty.
        """
        if not metrics:
            raise DiagnosticsError("No metrics provided for analysis.")
        issues: list[DiagnosticIssue] = []
        issues.extend(self._convergence.check(metrics))
        issues.extend(self._overfit.check(metrics))
        issues.extend(self._stability.check(metrics))
        trends = self._compute_trends(metrics)
        health = self._determine_health(issues)
        summary = self._generate_summary(issues, trends)
        recs = self._generate_forge_recommendations(issues)
        return DiagnosticReport(
            issues=issues,
            trends=trends,
            overall_health=health,
            plain_language_summary=summary,
            forge_recommendations=recs,
        )

    def analyze_curriculum(self, curriculum_stats: dict[str, Any]) -> DiagnosticReport:
        """Analyze curriculum statistics without training metrics.

        Args:
            curriculum_stats: Curriculum statistics dict.

        Returns:
            DiagnosticReport focused on data quality.
        """
        issues = self._data_quality.check_curriculum_stats(curriculum_stats)
        health = self._determine_health(issues)
        summary = self._generate_summary(issues, {})
        recs = self._generate_forge_recommendations(issues)
        return DiagnosticReport(
            issues=issues,
            trends={},
            overall_health=health,
            plain_language_summary=summary,
            forge_recommendations=recs,
        )

    def analyze_full(
        self,
        metrics: list[MetricSnapshot],
        curriculum_stats: dict[str, Any] | None = None,
    ) -> DiagnosticReport:
        """Run full analysis combining training metrics and curriculum stats.

        Args:
            metrics: Training metric snapshots.
            curriculum_stats: Optional curriculum statistics.

        Returns:
            Combined DiagnosticReport.

        Raises:
            DiagnosticsError: If metrics list is empty.
        """
        if not metrics:
            raise DiagnosticsError("No metrics provided for analysis.")
        issues: list[DiagnosticIssue] = []
        issues.extend(self._convergence.check(metrics))
        issues.extend(self._overfit.check(metrics))
        issues.extend(self._stability.check(metrics))
        if curriculum_stats is not None:
            issues.extend(self._data_quality.check_curriculum_stats(curriculum_stats))
        trends = self._compute_trends(metrics)
        health = self._determine_health(issues)
        summary = self._generate_summary(issues, trends)
        recs = self._generate_forge_recommendations(issues)
        return DiagnosticReport(
            issues=issues,
            trends=trends,
            overall_health=health,
            plain_language_summary=summary,
            forge_recommendations=recs,
        )

    def _compute_trends(self, metrics: list[MetricSnapshot]) -> dict[str, TrainingTrend]:
        """Compute trends for all available metrics.

        Args:
            metrics: Training metric snapshots.

        Returns:
            Dictionary mapping metric name to TrainingTrend.
        """
        trends: dict[str, TrainingTrend] = {}
        train_losses = [m.train_loss for m in metrics]
        trends["train_loss"] = self._trend_analyzer.analyze(train_losses, "train_loss")
        val_losses = [m.val_loss for m in metrics if m.val_loss is not None]
        if val_losses:
            trends["val_loss"] = self._trend_analyzer.analyze(val_losses, "val_loss")
        return trends

    @staticmethod
    def _determine_health(issues: list[DiagnosticIssue]) -> str:
        """Determine overall health from issue severities.

        Args:
            issues: All detected issues.

        Returns:
            One of "healthy", "warning", "critical".
        """
        if not issues:
            return "healthy"
        severities = {i.severity for i in issues}
        if IssueSeverity.CRITICAL in severities:
            return "critical"
        if IssueSeverity.WARNING in severities:
            return "warning"
        return "healthy"

    @staticmethod
    def _generate_summary(
        issues: list[DiagnosticIssue],
        trends: dict[str, TrainingTrend],
    ) -> str:
        """Generate a plain-language summary of the diagnostic results.

        Args:
            issues: All detected issues.
            trends: Computed trends.

        Returns:
            Human-readable summary string.
        """
        if not issues:
            return _healthy_summary(trends)
        return _problem_summary(issues)

    @staticmethod
    def _generate_forge_recommendations(
        issues: list[DiagnosticIssue],
    ) -> list[str]:
        """Generate Forge-specific recommendations from issues.

        Args:
            issues: All detected issues.

        Returns:
            List of recommendation strings.
        """
        recs: list[str] = []
        seen: set[str] = set()
        for issue in issues:
            rec = _forge_rec_for_issue(issue)
            if rec and rec not in seen:
                recs.append(rec)
                seen.add(rec)
        return recs


# ---------------------------------------------------------------------------
# Summary helper functions (kept short, extracted from class)
# ---------------------------------------------------------------------------


def _healthy_summary(trends: dict[str, TrainingTrend]) -> str:
    """Build summary text for a healthy training run.

    Args:
        trends: Computed training trends.

    Returns:
        Plain-language summary string.
    """
    parts = ["Training is progressing well with no issues detected."]
    train_trend = trends.get("train_loss")
    if train_trend and train_trend.is_decreasing:
        parts.append("The model is steadily improving and learning from the " "curriculum.")
    return " ".join(parts)


def _problem_summary(issues: list[DiagnosticIssue]) -> str:
    """Build summary text when problems are detected.

    Args:
        issues: All detected issues.

    Returns:
        Plain-language summary string.
    """
    critical = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
    warnings = [i for i in issues if i.severity == IssueSeverity.WARNING]
    parts: list[str] = []
    if critical:
        parts.append(
            f"There are {len(critical)} serious "
            f"issue{'s' if len(critical) != 1 else ''} "
            "that need attention."
        )
    if warnings:
        parts.append(
            f"There are {len(warnings)} "
            f"concern{'s' if len(warnings) != 1 else ''} "
            "worth reviewing."
        )
    titles = [i.title for i in issues[:3]]
    parts.append("Key findings: " + "; ".join(titles) + ".")
    return " ".join(parts)


def _forge_rec_for_issue(issue: DiagnosticIssue) -> str | None:
    """Map a diagnostic issue to a Forge recommendation.

    Args:
        issue: The diagnostic issue.

    Returns:
        Forge recommendation string, or None if not applicable.
    """
    rec_map: dict[IssueCategory, str] = {
        IssueCategory.CONVERGENCE: (
            "Review your curriculum in Forge and add more diverse "
            "examples, especially for competency areas where the "
            "model is struggling."
        ),
        IssueCategory.OVERFITTING: (
            "Add more varied examples to your Forge curriculum. "
            "Include different phrasings and scenarios for each "
            "competency area."
        ),
        IssueCategory.UNDERFITTING: (
            "Ensure your Forge curriculum covers all competency "
            "areas with enough examples (target 300-500 total)."
        ),
        IssueCategory.DATA_QUALITY: (
            "Review and improve your Forge curriculum quality. "
            "Check coverage balance, answer consistency, and "
            "total example count."
        ),
        IssueCategory.STABILITY: (
            "Check your Forge curriculum for unusual or outlier "
            "examples that may be causing training instability."
        ),
    }
    return rec_map.get(issue.category)
