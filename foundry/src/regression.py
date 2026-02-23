"""Regression testing and version management for Foundry.

Compares evaluation reports across model versions to detect competency
regressions. Provides plain-language summaries suitable for domain
experts (SMEs), not ML practitioners. Supports version tracking with
rollback capability when regressions are detected.

Key workflow:
    1. Run evaluation on new model version (via evaluation.py)
    2. Compare against baseline with RegressionChecker
    3. If regression detected, rollback via VersionManager
    4. If pass, promote new version with set_active

Example::

    checker = RegressionChecker()
    report = checker.compare(baseline_eval, current_eval, ChangeType.RETRAIN)
    if report.overall_verdict == "fail":
        version_manager.rollback("disc_maint")
    else:
        version_manager.set_active(new_version_id)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from foundry.src.evaluation import CompetencyScore, EvaluationReport


class RegressionError(Exception):
    """Raised for regression testing or version management errors."""


# ===================================================================
# Enums
# ===================================================================


class ChangeType(str, Enum):
    """Type of change that triggered a regression check.

    Attributes:
        RETRAIN: Model was retrained with updated curriculum.
        MERGE: LoRA adapters were merged.
        BASE_MODEL_SWAP: Underlying base model was changed.
        QUARRY_REPROCESS: Knowledge base was reprocessed.
        CURRICULUM_UPDATE: Training curriculum was modified.
    """

    RETRAIN = "retrain"
    MERGE = "merge"
    BASE_MODEL_SWAP = "base_model_swap"
    QUARRY_REPROCESS = "quarry_reprocess"
    CURRICULUM_UPDATE = "curriculum_update"


class RegressionSeverity(str, Enum):
    """Severity of a detected regression.

    Attributes:
        NONE: No regression (used for improvements).
        MINOR: Small performance drop (>10%).
        MAJOR: Significant performance drop (>20%).
        CRITICAL: Severe performance drop (>30%).
    """

    NONE = "none"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


# ===================================================================
# Data classes
# ===================================================================


@dataclass
class CompetencyRegression:
    """A detected regression (or improvement) in a single competency.

    Attributes:
        competency_id: Unique competency identifier.
        competency_name: Human-readable competency name.
        previous_rating: Rating string from baseline evaluation.
        current_rating: Rating string from current evaluation.
        previous_correct: Correct count in baseline.
        current_correct: Correct count in current.
        total_cases: Total test cases for this competency.
        severity: How severe the regression is.
    """

    competency_id: str
    competency_name: str
    previous_rating: str
    current_rating: str
    previous_correct: int
    current_correct: int
    total_cases: int
    severity: RegressionSeverity

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with all fields, severity as string value.
        """
        return {
            "competency_id": self.competency_id,
            "competency_name": self.competency_name,
            "previous_rating": self.previous_rating,
            "current_rating": self.current_rating,
            "previous_correct": self.previous_correct,
            "current_correct": self.current_correct,
            "total_cases": self.total_cases,
            "severity": self.severity.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompetencyRegression:
        """Deserialize from dictionary.

        Args:
            data: Dict with regression fields.

        Returns:
            CompetencyRegression instance.
        """
        return cls(
            competency_id=data["competency_id"],
            competency_name=data["competency_name"],
            previous_rating=data["previous_rating"],
            current_rating=data["current_rating"],
            previous_correct=data["previous_correct"],
            current_correct=data["current_correct"],
            total_cases=data["total_cases"],
            severity=RegressionSeverity(data["severity"]),
        )


@dataclass
class RegressionConfig:
    """Configuration for regression detection thresholds.

    Attributes:
        minor_threshold: Drop percentage triggering MINOR (default 0.1).
        major_threshold: Drop percentage triggering MAJOR (default 0.2).
        critical_threshold: Drop percentage triggering CRITICAL (default 0.3).
        fail_on_major: Whether MAJOR regression causes fail verdict.
        fail_on_critical: Whether CRITICAL regression causes fail verdict.
    """

    minor_threshold: float = 0.1
    major_threshold: float = 0.2
    critical_threshold: float = 0.3
    fail_on_major: bool = True
    fail_on_critical: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with all config fields.
        """
        return {
            "minor_threshold": self.minor_threshold,
            "major_threshold": self.major_threshold,
            "critical_threshold": self.critical_threshold,
            "fail_on_major": self.fail_on_major,
            "fail_on_critical": self.fail_on_critical,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegressionConfig:
        """Deserialize from dictionary.

        Args:
            data: Dict with config fields. Missing keys use defaults.

        Returns:
            RegressionConfig instance.
        """
        return cls(
            minor_threshold=data.get("minor_threshold", 0.1),
            major_threshold=data.get("major_threshold", 0.2),
            critical_threshold=data.get("critical_threshold", 0.3),
            fail_on_major=data.get("fail_on_major", True),
            fail_on_critical=data.get("fail_on_critical", True),
        )


@dataclass
class RegressionReport:
    """Complete regression comparison report between two evaluation runs.

    Attributes:
        report_id: Unique identifier for this regression report.
        baseline_run_id: Evaluation run ID of the baseline.
        current_run_id: Evaluation run ID of the current version.
        change_type: What triggered this regression check.
        regressions: Competencies that got worse.
        improvements: Competencies that got better.
        unchanged: Competency IDs with no significant change.
        overall_verdict: One of "pass", "warn", "fail".
        plain_language_summary: SME-friendly summary text.
        created_at: When this report was generated.
        discipline_id: Discipline being compared (from evaluation reports).
    """

    report_id: str
    baseline_run_id: str
    current_run_id: str
    change_type: ChangeType
    regressions: list[CompetencyRegression]
    improvements: list[CompetencyRegression]
    unchanged: list[str]
    overall_verdict: str
    plain_language_summary: str
    created_at: datetime
    discipline_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with all fields, nested objects serialized.
        """
        return {
            "report_id": self.report_id,
            "baseline_run_id": self.baseline_run_id,
            "current_run_id": self.current_run_id,
            "change_type": self.change_type.value,
            "regressions": [r.to_dict() for r in self.regressions],
            "improvements": [i.to_dict() for i in self.improvements],
            "unchanged": list(self.unchanged),
            "overall_verdict": self.overall_verdict,
            "plain_language_summary": self.plain_language_summary,
            "created_at": self.created_at.isoformat(),
            "discipline_id": self.discipline_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegressionReport:
        """Deserialize from dictionary.

        Args:
            data: Dict with report fields.

        Returns:
            RegressionReport instance.
        """
        regressions = [CompetencyRegression.from_dict(r) for r in data.get("regressions", [])]
        improvements = [CompetencyRegression.from_dict(i) for i in data.get("improvements", [])]
        return cls(
            report_id=data["report_id"],
            baseline_run_id=data["baseline_run_id"],
            current_run_id=data["current_run_id"],
            change_type=ChangeType(data["change_type"]),
            regressions=regressions,
            improvements=improvements,
            unchanged=data.get("unchanged", []),
            overall_verdict=data["overall_verdict"],
            plain_language_summary=data["plain_language_summary"],
            created_at=datetime.fromisoformat(data["created_at"]),
            discipline_id=data.get("discipline_id", ""),
        )


@dataclass
class VersionEntry:
    """A tracked model version in the version registry.

    Attributes:
        version_id: Unique version identifier.
        model_name: Human-readable model name.
        discipline_id: Which discipline this version serves.
        training_run_id: Associated training run (None for base models).
        evaluation_run_id: Associated evaluation run.
        adapter_path: Path to LoRA adapter (None for base models).
        change_type: What change produced this version.
        change_description: Human-readable description of the change.
        created_at: When this version was registered.
        is_active: Whether this is the currently active version.
    """

    version_id: str
    model_name: str
    discipline_id: str
    training_run_id: str | None
    evaluation_run_id: str
    adapter_path: str | None
    change_type: ChangeType
    change_description: str
    created_at: datetime
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with all fields, enums as string values.
        """
        return {
            "version_id": self.version_id,
            "model_name": self.model_name,
            "discipline_id": self.discipline_id,
            "training_run_id": self.training_run_id,
            "evaluation_run_id": self.evaluation_run_id,
            "adapter_path": self.adapter_path,
            "change_type": self.change_type.value,
            "change_description": self.change_description,
            "created_at": self.created_at.isoformat(),
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VersionEntry:
        """Deserialize from dictionary.

        Args:
            data: Dict with version entry fields.

        Returns:
            VersionEntry instance.
        """
        return cls(
            version_id=data["version_id"],
            model_name=data["model_name"],
            discipline_id=data["discipline_id"],
            training_run_id=data.get("training_run_id"),
            evaluation_run_id=data["evaluation_run_id"],
            adapter_path=data.get("adapter_path"),
            change_type=ChangeType(data["change_type"]),
            change_description=data["change_description"],
            created_at=datetime.fromisoformat(data["created_at"]),
            is_active=data.get("is_active", True),
        )


# ===================================================================
# RegressionChecker
# ===================================================================


class RegressionChecker:
    """Compares evaluation reports to detect competency regressions.

    Uses configurable thresholds to classify regressions by severity
    and generates plain-language summaries for domain experts.

    Args:
        config: Thresholds and verdict rules.

    Example::

        checker = RegressionChecker()
        report = checker.compare(baseline, current, ChangeType.RETRAIN)
        print(report.plain_language_summary)
    """

    def __init__(self, config: RegressionConfig | None = None) -> None:
        self._config = config or RegressionConfig()

    def compare(
        self,
        baseline: EvaluationReport,
        current: EvaluationReport,
        change_type: ChangeType,
    ) -> RegressionReport:
        """Compare two evaluation reports and produce a regression report.

        Args:
            baseline: The reference evaluation report.
            current: The new evaluation report to check.
            change_type: What triggered this comparison.

        Returns:
            RegressionReport with regressions, improvements, and verdict.
        """
        regressions = self._detect_regressions(
            baseline.competency_scores,
            current.competency_scores,
        )
        improvements = self._detect_improvements(
            baseline.competency_scores,
            current.competency_scores,
        )
        unchanged = self._find_unchanged(
            baseline.competency_scores,
            current.competency_scores,
            regressions,
            improvements,
        )
        verdict = self._determine_verdict(regressions)
        report = self._build_report(
            baseline,
            current,
            change_type,
            regressions,
            improvements,
            unchanged,
            verdict,
        )
        report.plain_language_summary = self._generate_summary(report)
        return report

    def _detect_regressions(
        self,
        baseline_scores: dict[str, CompetencyScore],
        current_scores: dict[str, CompetencyScore],
    ) -> list[CompetencyRegression]:
        """Find competencies where performance dropped.

        Args:
            baseline_scores: Competency scores from baseline eval.
            current_scores: Competency scores from current eval.

        Returns:
            List of CompetencyRegression for each detected drop.
        """
        regressions: list[CompetencyRegression] = []
        all_ids = set(baseline_scores.keys()) | set(current_scores.keys())
        for comp_id in sorted(all_ids):
            reg = self._check_single_regression(
                comp_id, baseline_scores.get(comp_id), current_scores.get(comp_id)
            )
            if reg is not None:
                regressions.append(reg)
        return regressions

    def _check_single_regression(
        self,
        comp_id: str,
        baseline: CompetencyScore | None,
        current: CompetencyScore | None,
    ) -> CompetencyRegression | None:
        """Check whether a single competency regressed.

        Args:
            comp_id: The competency identifier.
            baseline: Baseline score (None if new competency).
            current: Current score (None if removed competency).

        Returns:
            CompetencyRegression if drop detected, else None.
        """
        base_pct = _correct_pct(baseline)
        curr_pct = _correct_pct(current)
        drop = base_pct - curr_pct
        if drop <= 0:
            return None
        severity = self._compute_severity(base_pct, curr_pct)
        if severity == RegressionSeverity.NONE:
            return None
        return CompetencyRegression(
            competency_id=comp_id,
            competency_name=_comp_name(baseline, current, comp_id),
            previous_rating=_rating_str(baseline),
            current_rating=_rating_str(current),
            previous_correct=baseline.correct if baseline else 0,
            current_correct=current.correct if current else 0,
            total_cases=_total_cases(baseline, current),
            severity=severity,
        )

    def _detect_improvements(
        self,
        baseline_scores: dict[str, CompetencyScore],
        current_scores: dict[str, CompetencyScore],
    ) -> list[CompetencyRegression]:
        """Find competencies where performance improved.

        Args:
            baseline_scores: Competency scores from baseline eval.
            current_scores: Competency scores from current eval.

        Returns:
            List of CompetencyRegression (with NONE severity) for improvements.
        """
        improvements: list[CompetencyRegression] = []
        all_ids = set(baseline_scores.keys()) | set(current_scores.keys())
        for comp_id in sorted(all_ids):
            imp = self._check_single_improvement(
                comp_id, baseline_scores.get(comp_id), current_scores.get(comp_id)
            )
            if imp is not None:
                improvements.append(imp)
        return improvements

    def _check_single_improvement(
        self,
        comp_id: str,
        baseline: CompetencyScore | None,
        current: CompetencyScore | None,
    ) -> CompetencyRegression | None:
        """Check whether a single competency improved.

        Args:
            comp_id: The competency identifier.
            baseline: Baseline score (None if new competency).
            current: Current score (None if removed competency).

        Returns:
            CompetencyRegression with NONE severity if improved, else None.
        """
        base_pct = _correct_pct(baseline)
        curr_pct = _correct_pct(current)
        if curr_pct <= base_pct:
            return None
        return CompetencyRegression(
            competency_id=comp_id,
            competency_name=_comp_name(baseline, current, comp_id),
            previous_rating=_rating_str(baseline),
            current_rating=_rating_str(current),
            previous_correct=baseline.correct if baseline else 0,
            current_correct=current.correct if current else 0,
            total_cases=_total_cases(baseline, current),
            severity=RegressionSeverity.NONE,
        )

    def _compute_severity(
        self,
        baseline_pct: float,
        current_pct: float,
    ) -> RegressionSeverity:
        """Classify the severity of a performance drop.

        Args:
            baseline_pct: Baseline correct percentage (0-1).
            current_pct: Current correct percentage (0-1).

        Returns:
            RegressionSeverity based on configured thresholds.
        """
        drop = baseline_pct - current_pct
        if drop >= self._config.critical_threshold:
            return RegressionSeverity.CRITICAL
        if drop >= self._config.major_threshold:
            return RegressionSeverity.MAJOR
        if drop >= self._config.minor_threshold:
            return RegressionSeverity.MINOR
        return RegressionSeverity.NONE

    def _determine_verdict(
        self,
        regressions: list[CompetencyRegression],
    ) -> str:
        """Determine overall verdict from detected regressions.

        Args:
            regressions: List of detected regressions.

        Returns:
            One of "pass", "warn", or "fail".
        """
        if not regressions:
            return "pass"
        has_critical = any(r.severity == RegressionSeverity.CRITICAL for r in regressions)
        has_major = any(r.severity == RegressionSeverity.MAJOR for r in regressions)
        if has_critical and self._config.fail_on_critical:
            return "fail"
        if has_major and self._config.fail_on_major:
            return "fail"
        return "warn"

    def _generate_summary(self, report: RegressionReport) -> str:
        """Generate a plain-language summary for domain experts.

        Uses competency names and X/Y correct format. Avoids all
        ML jargon (loss, perplexity, F1, precision, recall, etc.).

        Args:
            report: The completed regression report.

        Returns:
            Multi-line SME-friendly summary string.
        """
        if report.overall_verdict == "pass" and not report.regressions:
            return self._generate_pass_summary(report)
        return self._generate_detail_summary(report)

    @staticmethod
    def _generate_pass_summary(report: RegressionReport) -> str:
        """Generate summary for a passing regression check.

        Args:
            report: Report with no regressions.

        Returns:
            Positive summary string.
        """
        lines: list[str] = ["No regressions detected. All competencies maintained or improved."]
        for imp in report.improvements:
            lines.append(
                f"  Improved: {imp.competency_name} "
                f"({imp.previous_correct}/{imp.total_cases} -> "
                f"{imp.current_correct}/{imp.total_cases} correct)"
            )
        return "\n".join(lines)

    @staticmethod
    def _generate_detail_summary(report: RegressionReport) -> str:
        """Generate detailed summary with regressions and improvements.

        Args:
            report: Report containing regressions.

        Returns:
            Multi-line summary with competency-level detail.
        """
        lines: list[str] = []
        lines.append(f"Verdict: {report.overall_verdict}")
        if report.regressions:
            lines.append("Regressions:")
            for reg in report.regressions:
                lines.append(
                    f"  {reg.competency_name}: "
                    f"{reg.previous_correct}/{reg.total_cases} -> "
                    f"{reg.current_correct}/{reg.total_cases} correct "
                    f"({reg.severity.value})"
                )
        if report.improvements:
            lines.append("Improvements:")
            for imp in report.improvements:
                lines.append(
                    f"  {imp.competency_name}: "
                    f"{imp.previous_correct}/{imp.total_cases} -> "
                    f"{imp.current_correct}/{imp.total_cases} correct"
                )
        return "\n".join(lines)

    @staticmethod
    def _find_unchanged(
        baseline_scores: dict[str, CompetencyScore],
        current_scores: dict[str, CompetencyScore],
        regressions: list[CompetencyRegression],
        improvements: list[CompetencyRegression],
    ) -> list[str]:
        """Identify competencies with no significant change.

        Args:
            baseline_scores: Baseline competency scores.
            current_scores: Current competency scores.
            regressions: Detected regressions.
            improvements: Detected improvements.

        Returns:
            List of competency IDs that did not change.
        """
        changed_ids = {r.competency_id for r in regressions} | {
            i.competency_id for i in improvements
        }
        all_ids = set(baseline_scores.keys()) | set(current_scores.keys())
        return sorted(comp_id for comp_id in all_ids if comp_id not in changed_ids)

    @staticmethod
    def _build_report(
        baseline: EvaluationReport,
        current: EvaluationReport,
        change_type: ChangeType,
        regressions: list[CompetencyRegression],
        improvements: list[CompetencyRegression],
        unchanged: list[str],
        verdict: str,
    ) -> RegressionReport:
        """Assemble a RegressionReport from computed components.

        Args:
            baseline: Baseline evaluation report.
            current: Current evaluation report.
            change_type: What triggered this check.
            regressions: Detected regressions.
            improvements: Detected improvements.
            unchanged: Unchanged competency IDs.
            verdict: Overall verdict string.

        Returns:
            RegressionReport (summary not yet populated).
        """
        return RegressionReport(
            report_id=f"regr_{uuid.uuid4().hex[:12]}",
            baseline_run_id=baseline.run_id,
            current_run_id=current.run_id,
            change_type=change_type,
            regressions=regressions,
            improvements=improvements,
            unchanged=unchanged,
            overall_verdict=verdict,
            plain_language_summary="",
            created_at=datetime.now(),
            discipline_id=baseline.discipline_id,
        )


# ===================================================================
# VersionManager
# ===================================================================


class VersionManager:
    """Manages model version history with rollback support.

    Persists version entries as individual JSON files in a directory.
    Supports listing, filtering, activation, and rollback.

    Args:
        versions_dir: Directory for storing version JSON files.

    Example::

        vm = VersionManager(Path("./versions"))
        vm.register_version(entry)
        vm.set_active("ver_002")
        vm.rollback("disc_maint")
    """

    def __init__(self, versions_dir: Path) -> None:
        self._dir = versions_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def register_version(self, entry: VersionEntry) -> None:
        """Register a new model version.

        Args:
            entry: The version entry to persist.
        """
        path = self._dir / f"{entry.version_id}.json"
        path.write_text(
            json.dumps(entry.to_dict(), indent=2),
            encoding="utf-8",
        )

    def get_version(self, version_id: str) -> VersionEntry | None:
        """Retrieve a version by ID.

        Args:
            version_id: The unique version identifier.

        Returns:
            VersionEntry if found, None otherwise.
        """
        path = self._dir / f"{version_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return VersionEntry.from_dict(data)

    def list_versions(
        self,
        discipline_id: str | None = None,
    ) -> list[VersionEntry]:
        """List all versions, optionally filtered by discipline.

        Args:
            discipline_id: If provided, only return versions for this discipline.

        Returns:
            List of VersionEntry sorted by creation time.
        """
        entries = self._load_all()
        if discipline_id is not None:
            entries = [e for e in entries if e.discipline_id == discipline_id]
        return sorted(entries, key=lambda e: e.created_at)

    def get_active_version(self, discipline_id: str) -> VersionEntry | None:
        """Get the currently active version for a discipline.

        Args:
            discipline_id: The discipline to query.

        Returns:
            Active VersionEntry, or None if none is active.
        """
        versions = self.list_versions(discipline_id=discipline_id)
        active = [v for v in versions if v.is_active]
        if not active:
            return None
        return active[-1]

    def set_active(self, version_id: str) -> None:
        """Set a specific version as active, deactivating others.

        Deactivates all versions for the same discipline, then
        activates the specified version.

        Args:
            version_id: The version to activate.

        Raises:
            RegressionError: If version_id is not found.
        """
        target = self.get_version(version_id)
        if target is None:
            raise RegressionError(f"Version not found: {version_id}")
        self._deactivate_discipline(target.discipline_id)
        target.is_active = True
        self.register_version(target)

    def rollback(self, discipline_id: str) -> VersionEntry | None:
        """Rollback to the previous version for a discipline.

        Deactivates the current active version and activates the
        most recent previous version.

        Args:
            discipline_id: The discipline to rollback.

        Returns:
            The newly activated VersionEntry, or None if no previous exists.
        """
        history = self.get_version_history(discipline_id)
        if len(history) < 2:
            return None
        return self._perform_rollback(history)

    def get_version_history(self, discipline_id: str) -> list[VersionEntry]:
        """Get chronological version history for a discipline.

        Args:
            discipline_id: The discipline to query.

        Returns:
            List of VersionEntry sorted oldest to newest.
        """
        return self.list_versions(discipline_id=discipline_id)

    def _load_all(self) -> list[VersionEntry]:
        """Load all version entries from disk.

        Returns:
            List of all stored VersionEntry instances.
        """
        entries: list[VersionEntry] = []
        for path in self._dir.glob("ver_*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            entries.append(VersionEntry.from_dict(data))
        return entries

    def _deactivate_discipline(self, discipline_id: str) -> None:
        """Deactivate all versions for a discipline.

        Args:
            discipline_id: The discipline whose versions to deactivate.
        """
        for version in self.list_versions(discipline_id=discipline_id):
            if version.is_active:
                version.is_active = False
                self.register_version(version)

    def _perform_rollback(
        self,
        history: list[VersionEntry],
    ) -> VersionEntry:
        """Execute the rollback operation on a version history.

        Args:
            history: Chronological list of versions (at least 2).

        Returns:
            The newly activated previous version.
        """
        current = history[-1]
        previous = history[-2]
        current.is_active = False
        self.register_version(current)
        previous.is_active = True
        self.register_version(previous)
        return previous


# ===================================================================
# RegressionRunner
# ===================================================================


class RegressionRunner:
    """Orchestrates regression checks with persistence.

    Combines RegressionChecker for comparison logic with file
    persistence for reports and trend tracking.

    Args:
        checker: RegressionChecker for comparison.
        version_manager: VersionManager for version tracking.
        reports_dir: Directory for saving regression report JSON files.

    Example::

        runner = RegressionRunner(checker, vm, Path("./reports"))
        report = runner.run_regression_check(baseline, current, ChangeType.RETRAIN)
        runner.save_report(report)
    """

    def __init__(
        self,
        checker: RegressionChecker,
        version_manager: VersionManager,
        reports_dir: Path,
    ) -> None:
        self._checker = checker
        self._version_manager = version_manager
        self._reports_dir = reports_dir
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    def run_regression_check(
        self,
        baseline_report: EvaluationReport,
        current_report: EvaluationReport,
        change_type: ChangeType,
    ) -> RegressionReport:
        """Run a regression comparison between two evaluation reports.

        Args:
            baseline_report: The reference evaluation.
            current_report: The new evaluation to check.
            change_type: What triggered this check.

        Returns:
            RegressionReport with comparison results and verdict.
        """
        return self._checker.compare(baseline_report, current_report, change_type)

    def save_report(self, report: RegressionReport) -> Path:
        """Save a regression report to disk.

        Args:
            report: The report to persist.

        Returns:
            Path to the saved JSON file.
        """
        path = self._reports_dir / f"{report.report_id}.json"
        path.write_text(
            json.dumps(report.to_dict(), indent=2),
            encoding="utf-8",
        )
        return path

    def load_report(self, report_id: str) -> RegressionReport:
        """Load a regression report by ID.

        Args:
            report_id: The unique report identifier.

        Returns:
            RegressionReport instance.

        Raises:
            RegressionError: If report file not found.
        """
        path = self._reports_dir / f"{report_id}.json"
        if not path.exists():
            raise RegressionError(f"Regression report not found: {report_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return RegressionReport.from_dict(data)

    def list_reports(
        self,
        discipline_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List saved regression report summaries.

        Args:
            discipline_id: Optional filter by discipline.

        Returns:
            List of summary dicts with report_id, verdict, etc.
        """
        summaries: list[dict[str, Any]] = []
        for path in sorted(self._reports_dir.glob("regr_*.json")):
            summary = self._read_summary(path)
            if summary is None:
                continue
            if discipline_id and summary.get("discipline_id") != discipline_id:
                continue
            summaries.append(summary)
        return summaries

    def get_trend(
        self,
        discipline_id: str,
        competency_id: str,
    ) -> list[dict[str, Any]]:
        """Get accuracy trend for a competency across versions.

        Args:
            discipline_id: The discipline to query.
            competency_id: The competency to track.

        Returns:
            List of dicts with run_id and accuracy, ordered by time.
        """
        trend: list[dict[str, Any]] = []
        reports = self._load_all_reports()
        for report in reports:
            points = self._extract_trend_points(report, discipline_id, competency_id)
            trend.extend(points)
        return sorted(trend, key=lambda p: p.get("created_at", ""))

    def _load_all_reports(self) -> list[RegressionReport]:
        """Load all regression reports from disk.

        Returns:
            List of RegressionReport instances.
        """
        reports: list[RegressionReport] = []
        for path in sorted(self._reports_dir.glob("regr_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            reports.append(RegressionReport.from_dict(data))
        return reports

    @staticmethod
    def _extract_trend_points(
        report: RegressionReport,
        discipline_id: str,
        competency_id: str,
    ) -> list[dict[str, Any]]:
        """Extract trend data points from a regression report.

        Args:
            report: A regression report to extract from.
            discipline_id: Target discipline.
            competency_id: Target competency.

        Returns:
            List of trend point dicts (may be empty).
        """
        points: list[dict[str, Any]] = []
        all_changes = list(report.regressions) + list(report.improvements)
        for change in all_changes:
            if change.competency_id != competency_id:
                continue
            total = change.total_cases if change.total_cases > 0 else 1
            points.append(
                {
                    "run_id": report.current_run_id,
                    "accuracy": change.current_correct / total,
                    "correct": change.current_correct,
                    "total": change.total_cases,
                    "created_at": report.created_at.isoformat(),
                }
            )
        return points

    @staticmethod
    def _read_summary(path: Path) -> dict[str, Any] | None:
        """Read a report summary from a JSON file.

        Args:
            path: Path to the regression report JSON file.

        Returns:
            Summary dict, or None on read error.
        """
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {
                "report_id": data["report_id"],
                "baseline_run_id": data["baseline_run_id"],
                "current_run_id": data["current_run_id"],
                "change_type": data["change_type"],
                "overall_verdict": data["overall_verdict"],
                "discipline_id": data.get("discipline_id", ""),
                "created_at": data.get("created_at", ""),
            }
        except (json.JSONDecodeError, KeyError):
            return None


# ===================================================================
# Module-level helpers
# ===================================================================


def _correct_pct(score: CompetencyScore | None) -> float:
    """Compute correct percentage from a CompetencyScore.

    Args:
        score: CompetencyScore or None.

    Returns:
        Fraction of correct answers (0-1), or 0.0 if None/empty.
    """
    if score is None or score.total_cases == 0:
        return 0.0
    return score.correct / score.total_cases


def _comp_name(
    baseline: CompetencyScore | None,
    current: CompetencyScore | None,
    fallback: str,
) -> str:
    """Get the competency name from either score, with fallback.

    Args:
        baseline: Baseline score (may be None).
        current: Current score (may be None).
        fallback: Fallback string if both are None.

    Returns:
        Human-readable competency name.
    """
    if current is not None:
        return current.competency_name
    if baseline is not None:
        return baseline.competency_name
    return fallback


def _rating_str(score: CompetencyScore | None) -> str:
    """Get rating value string from a score.

    Args:
        score: CompetencyScore or None.

    Returns:
        Rating string value, or "untested" if None.
    """
    if score is None:
        return "untested"
    return score.rating.value


def _total_cases(
    baseline: CompetencyScore | None,
    current: CompetencyScore | None,
) -> int:
    """Get total cases from whichever score is available.

    Args:
        baseline: Baseline score (may be None).
        current: Current score (may be None).

    Returns:
        Total test cases count.
    """
    if current is not None:
        return current.total_cases
    if baseline is not None:
        return baseline.total_cases
    return 0
