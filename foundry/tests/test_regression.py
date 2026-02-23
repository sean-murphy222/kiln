"""Tests for Foundry regression testing and version management system.

Covers: RegressionChecker, VersionManager, RegressionRunner,
dataclass construction/serialization, plain-language summaries,
and integration scenarios (retrain -> evaluate -> regress -> rollback).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from foundry.src.evaluation import (
    CompetencyRating,
    CompetencyScore,
    EvaluationReport,
    EvaluationStatus,
)
from foundry.src.regression import (
    ChangeType,
    CompetencyRegression,
    RegressionChecker,
    RegressionConfig,
    RegressionError,
    RegressionReport,
    RegressionRunner,
    RegressionSeverity,
    VersionEntry,
    VersionManager,
)

# ===================================================================
# Helpers
# ===================================================================


def _make_competency_score(
    comp_id: str,
    comp_name: str,
    total: int,
    correct: int,
    rating: CompetencyRating,
) -> CompetencyScore:
    """Build a CompetencyScore with sensible defaults for non-key fields."""
    return CompetencyScore(
        competency_id=comp_id,
        competency_name=comp_name,
        total_cases=total,
        correct=correct,
        partially_correct=0,
        incorrect=total - correct,
        no_response=0,
        rating=rating,
        summary=f"{correct}/{total} correct",
    )


def _make_eval_report(
    run_id: str,
    model_name: str,
    discipline_id: str,
    scores: dict[str, CompetencyScore],
    overall_accuracy: float | None = None,
) -> EvaluationReport:
    """Build an EvaluationReport with sensible defaults."""
    total = sum(s.total_cases for s in scores.values())
    correct = sum(s.correct for s in scores.values())
    if overall_accuracy is not None:
        accuracy = overall_accuracy
    else:
        accuracy = correct / total if total else 0.0
    return EvaluationReport(
        run_id=run_id,
        model_name=model_name,
        discipline_id=discipline_id,
        status=EvaluationStatus.COMPLETED,
        competency_scores=scores,
        test_results=[],
        total_cases=total,
        overall_correct=correct,
        overall_accuracy=accuracy,
        overall_rating=CompetencyRating.ADEQUATE,
        plain_language_summary="Test summary",
        weak_areas=[],
        strong_areas=[],
        started_at=datetime(2026, 2, 20, 10, 0, 0),
        completed_at=datetime(2026, 2, 20, 10, 5, 0),
    )


def _make_version_entry(
    version_id: str = "ver_001",
    model_name: str = "maint-lora-v1",
    discipline_id: str = "disc_maint",
    evaluation_run_id: str = "eval_001",
    change_type: ChangeType = ChangeType.RETRAIN,
    is_active: bool = True,
    created_at: datetime | None = None,
) -> VersionEntry:
    """Build a VersionEntry with sensible defaults."""
    return VersionEntry(
        version_id=version_id,
        model_name=model_name,
        discipline_id=discipline_id,
        training_run_id="run_001",
        evaluation_run_id=evaluation_run_id,
        adapter_path="/adapters/v1",
        change_type=change_type,
        change_description="Retrained with updated curriculum",
        created_at=created_at or datetime(2026, 2, 20, 10, 0, 0),
        is_active=is_active,
    )


# ===================================================================
# TestCompetencyRegression
# ===================================================================


class TestCompetencyRegression:
    """Tests for CompetencyRegression dataclass."""

    def test_construction(self) -> None:
        """CompetencyRegression stores all fields correctly."""
        reg = CompetencyRegression(
            competency_id="comp_proc",
            competency_name="Procedural Comprehension",
            previous_rating="strong",
            current_rating="needs_improvement",
            previous_correct=9,
            current_correct=5,
            total_cases=10,
            severity=RegressionSeverity.MAJOR,
        )
        assert reg.competency_id == "comp_proc"
        assert reg.competency_name == "Procedural Comprehension"
        assert reg.previous_rating == "strong"
        assert reg.current_rating == "needs_improvement"
        assert reg.previous_correct == 9
        assert reg.current_correct == 5
        assert reg.total_cases == 10
        assert reg.severity == RegressionSeverity.MAJOR

    def test_severity_none(self) -> None:
        """NONE severity represents no regression (used for improvements)."""
        reg = CompetencyRegression(
            competency_id="comp_fault",
            competency_name="Fault Isolation",
            previous_rating="adequate",
            current_rating="strong",
            previous_correct=6,
            current_correct=9,
            total_cases=10,
            severity=RegressionSeverity.NONE,
        )
        assert reg.severity == RegressionSeverity.NONE

    def test_to_dict(self) -> None:
        """to_dict serializes all fields including enum value."""
        reg = CompetencyRegression(
            competency_id="comp_proc",
            competency_name="Procedural Comprehension",
            previous_rating="strong",
            current_rating="weak",
            previous_correct=9,
            current_correct=2,
            total_cases=10,
            severity=RegressionSeverity.CRITICAL,
        )
        data = reg.to_dict()
        assert data["competency_id"] == "comp_proc"
        assert data["severity"] == "critical"
        assert data["previous_correct"] == 9
        assert data["current_correct"] == 2

    def test_severity_levels(self) -> None:
        """All severity enum values are accessible."""
        assert RegressionSeverity.NONE == "none"
        assert RegressionSeverity.MINOR == "minor"
        assert RegressionSeverity.MAJOR == "major"
        assert RegressionSeverity.CRITICAL == "critical"


# ===================================================================
# TestRegressionConfig
# ===================================================================


class TestRegressionConfig:
    """Tests for RegressionConfig dataclass."""

    def test_defaults(self) -> None:
        """Default thresholds are sensible."""
        cfg = RegressionConfig()
        assert cfg.minor_threshold == 0.1
        assert cfg.major_threshold == 0.2
        assert cfg.critical_threshold == 0.3
        assert cfg.fail_on_major is True
        assert cfg.fail_on_critical is True

    def test_custom_thresholds(self) -> None:
        """Custom thresholds override defaults."""
        cfg = RegressionConfig(
            minor_threshold=0.05,
            major_threshold=0.15,
            critical_threshold=0.25,
            fail_on_major=False,
        )
        assert cfg.minor_threshold == 0.05
        assert cfg.major_threshold == 0.15
        assert cfg.critical_threshold == 0.25
        assert cfg.fail_on_major is False

    def test_to_dict(self) -> None:
        """Config serializes to dict."""
        cfg = RegressionConfig()
        data = cfg.to_dict()
        assert data["minor_threshold"] == 0.1
        assert data["fail_on_critical"] is True

    def test_from_dict(self) -> None:
        """Config deserializes from dict."""
        data = {
            "minor_threshold": 0.05,
            "major_threshold": 0.15,
            "critical_threshold": 0.25,
            "fail_on_major": False,
            "fail_on_critical": True,
        }
        cfg = RegressionConfig.from_dict(data)
        assert cfg.minor_threshold == 0.05
        assert cfg.fail_on_major is False

    def test_from_dict_defaults(self) -> None:
        """from_dict uses defaults for missing keys."""
        cfg = RegressionConfig.from_dict({})
        assert cfg.minor_threshold == 0.1
        assert cfg.fail_on_major is True


# ===================================================================
# TestRegressionReport
# ===================================================================


class TestRegressionReport:
    """Tests for RegressionReport dataclass."""

    def test_construction(self) -> None:
        """RegressionReport stores all fields."""
        report = RegressionReport(
            report_id="regr_001",
            baseline_run_id="eval_001",
            current_run_id="eval_002",
            change_type=ChangeType.RETRAIN,
            regressions=[],
            improvements=[],
            unchanged=["comp_proc"],
            overall_verdict="pass",
            plain_language_summary="No regressions detected.",
            created_at=datetime(2026, 2, 20, 10, 0, 0),
        )
        assert report.report_id == "regr_001"
        assert report.overall_verdict == "pass"
        assert len(report.unchanged) == 1

    def test_to_dict(self) -> None:
        """to_dict serializes including nested regressions."""
        regression = CompetencyRegression(
            competency_id="comp_proc",
            competency_name="Procedural Comprehension",
            previous_rating="strong",
            current_rating="adequate",
            previous_correct=9,
            current_correct=7,
            total_cases=10,
            severity=RegressionSeverity.MINOR,
        )
        report = RegressionReport(
            report_id="regr_002",
            baseline_run_id="eval_001",
            current_run_id="eval_002",
            change_type=ChangeType.MERGE,
            regressions=[regression],
            improvements=[],
            unchanged=[],
            overall_verdict="warn",
            plain_language_summary="Minor regression in Procedural Comprehension.",
            created_at=datetime(2026, 2, 20, 10, 0, 0),
        )
        data = report.to_dict()
        assert data["report_id"] == "regr_002"
        assert data["change_type"] == "merge"
        assert len(data["regressions"]) == 1
        assert data["regressions"][0]["severity"] == "minor"
        assert data["overall_verdict"] == "warn"

    def test_from_dict(self) -> None:
        """from_dict round-trips correctly."""
        report = RegressionReport(
            report_id="regr_003",
            baseline_run_id="eval_001",
            current_run_id="eval_002",
            change_type=ChangeType.CURRICULUM_UPDATE,
            regressions=[],
            improvements=[],
            unchanged=["comp_proc", "comp_fault"],
            overall_verdict="pass",
            plain_language_summary="All competencies unchanged.",
            created_at=datetime(2026, 2, 20, 12, 0, 0),
        )
        data = report.to_dict()
        restored = RegressionReport.from_dict(data)
        assert restored.report_id == report.report_id
        assert restored.change_type == ChangeType.CURRICULUM_UPDATE
        assert restored.unchanged == ["comp_proc", "comp_fault"]

    def test_verdicts(self) -> None:
        """Verdict values are limited to pass/warn/fail."""
        for verdict in ("pass", "warn", "fail"):
            report = RegressionReport(
                report_id="regr_v",
                baseline_run_id="eval_001",
                current_run_id="eval_002",
                change_type=ChangeType.RETRAIN,
                regressions=[],
                improvements=[],
                unchanged=[],
                overall_verdict=verdict,
                plain_language_summary="Test",
                created_at=datetime(2026, 2, 20, 10, 0, 0),
            )
            assert report.overall_verdict == verdict


# ===================================================================
# TestVersionEntry
# ===================================================================


class TestVersionEntry:
    """Tests for VersionEntry dataclass."""

    def test_construction(self) -> None:
        """VersionEntry stores all fields correctly."""
        entry = _make_version_entry()
        assert entry.version_id == "ver_001"
        assert entry.model_name == "maint-lora-v1"
        assert entry.discipline_id == "disc_maint"
        assert entry.is_active is True

    def test_active_flag_default(self) -> None:
        """is_active defaults to True."""
        entry = VersionEntry(
            version_id="ver_002",
            model_name="maint-lora-v2",
            discipline_id="disc_maint",
            training_run_id=None,
            evaluation_run_id="eval_002",
            adapter_path=None,
            change_type=ChangeType.BASE_MODEL_SWAP,
            change_description="Swapped base model",
            created_at=datetime(2026, 2, 20, 10, 0, 0),
        )
        assert entry.is_active is True

    def test_to_dict(self) -> None:
        """to_dict serializes all fields including optional None."""
        entry = VersionEntry(
            version_id="ver_003",
            model_name="maint-lora-v3",
            discipline_id="disc_maint",
            training_run_id=None,
            evaluation_run_id="eval_003",
            adapter_path=None,
            change_type=ChangeType.QUARRY_REPROCESS,
            change_description="Reprocessed knowledge base",
            created_at=datetime(2026, 2, 20, 10, 0, 0),
            is_active=False,
        )
        data = entry.to_dict()
        assert data["version_id"] == "ver_003"
        assert data["training_run_id"] is None
        assert data["adapter_path"] is None
        assert data["change_type"] == "quarry_reprocess"
        assert data["is_active"] is False

    def test_from_dict_roundtrip(self) -> None:
        """from_dict round-trips correctly."""
        entry = _make_version_entry()
        data = entry.to_dict()
        restored = VersionEntry.from_dict(data)
        assert restored.version_id == entry.version_id
        assert restored.model_name == entry.model_name
        assert restored.change_type == ChangeType.RETRAIN
        assert restored.is_active is True

    def test_change_types(self) -> None:
        """All ChangeType enum values are accessible."""
        assert ChangeType.RETRAIN == "retrain"
        assert ChangeType.MERGE == "merge"
        assert ChangeType.BASE_MODEL_SWAP == "base_model_swap"
        assert ChangeType.QUARRY_REPROCESS == "quarry_reprocess"
        assert ChangeType.CURRICULUM_UPDATE == "curriculum_update"


# ===================================================================
# TestRegressionChecker
# ===================================================================


class TestRegressionChecker:
    """Tests for RegressionChecker comparison logic."""

    def test_no_regressions_identical(self, regression_checker: RegressionChecker) -> None:
        """Identical reports produce no regressions and verdict=pass."""
        scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 8, CompetencyRating.STRONG
            ),
        }
        baseline = _make_eval_report("eval_001", "model-v1", "disc_maint", scores)
        current = _make_eval_report("eval_002", "model-v1", "disc_maint", scores)

        report = regression_checker.compare(baseline, current, ChangeType.RETRAIN)
        assert report.overall_verdict == "pass"
        assert len(report.regressions) == 0
        assert len(report.improvements) == 0
        assert "comp_proc" in report.unchanged

    def test_minor_regression(self, regression_checker: RegressionChecker) -> None:
        """A 10-15% drop is classified as MINOR severity."""
        baseline_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 8, CompetencyRating.STRONG
            ),
        }
        current_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 7, CompetencyRating.ADEQUATE
            ),
        }
        baseline = _make_eval_report("eval_001", "model-v1", "disc_maint", baseline_scores)
        current = _make_eval_report("eval_002", "model-v2", "disc_maint", current_scores)

        report = regression_checker.compare(baseline, current, ChangeType.RETRAIN)
        assert len(report.regressions) == 1
        assert report.regressions[0].severity == RegressionSeverity.MINOR
        assert report.overall_verdict == "warn"

    def test_major_regression(self) -> None:
        """A 20-29% drop is classified as MAJOR severity, verdict=fail."""
        checker = RegressionChecker(RegressionConfig())
        baseline_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 20, 10, CompetencyRating.ADEQUATE
            ),
        }
        current_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 20, 5, CompetencyRating.WEAK
            ),
        }
        baseline = _make_eval_report("eval_001", "model-v1", "disc_maint", baseline_scores)
        current = _make_eval_report("eval_002", "model-v2", "disc_maint", current_scores)

        report = checker.compare(baseline, current, ChangeType.RETRAIN)
        assert len(report.regressions) == 1
        assert report.regressions[0].severity == RegressionSeverity.MAJOR
        assert report.overall_verdict == "fail"

    def test_critical_regression(self) -> None:
        """A >=30% drop is classified as CRITICAL severity."""
        checker = RegressionChecker(RegressionConfig())
        baseline_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 9, CompetencyRating.STRONG
            ),
        }
        current_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 3, CompetencyRating.WEAK
            ),
        }
        baseline = _make_eval_report("eval_001", "model-v1", "disc_maint", baseline_scores)
        current = _make_eval_report("eval_002", "model-v2", "disc_maint", current_scores)

        report = checker.compare(baseline, current, ChangeType.RETRAIN)
        assert len(report.regressions) == 1
        assert report.regressions[0].severity == RegressionSeverity.CRITICAL
        assert report.overall_verdict == "fail"

    def test_improvement_detected(self, regression_checker: RegressionChecker) -> None:
        """A score increase is detected as an improvement with NONE severity."""
        baseline_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 5, CompetencyRating.ADEQUATE
            ),
        }
        current_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 9, CompetencyRating.STRONG
            ),
        }
        baseline = _make_eval_report("eval_001", "model-v1", "disc_maint", baseline_scores)
        current = _make_eval_report("eval_002", "model-v2", "disc_maint", current_scores)

        report = regression_checker.compare(baseline, current, ChangeType.RETRAIN)
        assert len(report.improvements) == 1
        assert report.improvements[0].severity == RegressionSeverity.NONE
        assert report.improvements[0].current_correct == 9
        assert report.overall_verdict == "pass"

    def test_mixed_results(self, regression_checker: RegressionChecker) -> None:
        """Report tracks regressions and improvements in separate lists."""
        baseline_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 8, CompetencyRating.STRONG
            ),
            "comp_fault": _make_competency_score(
                "comp_fault", "Fault Isolation", 10, 5, CompetencyRating.ADEQUATE
            ),
            "comp_safety": _make_competency_score(
                "comp_safety", "Safety Protocol", 10, 7, CompetencyRating.ADEQUATE
            ),
        }
        current_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 5, CompetencyRating.ADEQUATE
            ),
            "comp_fault": _make_competency_score(
                "comp_fault", "Fault Isolation", 10, 9, CompetencyRating.STRONG
            ),
            "comp_safety": _make_competency_score(
                "comp_safety", "Safety Protocol", 10, 7, CompetencyRating.ADEQUATE
            ),
        }
        baseline = _make_eval_report("eval_001", "model-v1", "disc_maint", baseline_scores)
        current = _make_eval_report("eval_002", "model-v2", "disc_maint", current_scores)

        report = regression_checker.compare(baseline, current, ChangeType.RETRAIN)
        assert len(report.regressions) == 1
        assert report.regressions[0].competency_id == "comp_proc"
        assert len(report.improvements) == 1
        assert report.improvements[0].competency_id == "comp_fault"
        assert "comp_safety" in report.unchanged

    def test_verdict_pass_no_regressions(
        self,
        regression_checker: RegressionChecker,
    ) -> None:
        """Verdict is pass when only improvements exist."""
        baseline_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 5, CompetencyRating.ADEQUATE
            ),
        }
        current_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 8, CompetencyRating.STRONG
            ),
        }
        baseline = _make_eval_report("eval_001", "model-v1", "disc_maint", baseline_scores)
        current = _make_eval_report("eval_002", "model-v2", "disc_maint", current_scores)

        report = regression_checker.compare(baseline, current, ChangeType.RETRAIN)
        assert report.overall_verdict == "pass"

    def test_verdict_fail_configurable(self) -> None:
        """fail_on_major=False downgrades major regression to warn."""
        cfg = RegressionConfig(fail_on_major=False, fail_on_critical=False)
        checker = RegressionChecker(cfg)
        baseline_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 10, CompetencyRating.STRONG
            ),
        }
        current_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 8, CompetencyRating.STRONG
            ),
        }
        baseline = _make_eval_report("eval_001", "model-v1", "disc_maint", baseline_scores)
        current = _make_eval_report("eval_002", "model-v2", "disc_maint", current_scores)

        report = checker.compare(baseline, current, ChangeType.RETRAIN)
        assert report.overall_verdict == "warn"

    def test_zero_total_cases_no_crash(self) -> None:
        """Competency with zero total cases is treated as unchanged."""
        checker = RegressionChecker(RegressionConfig())
        baseline_scores = {
            "comp_proc": CompetencyScore(
                competency_id="comp_proc",
                competency_name="Procedural Comprehension",
                total_cases=0,
                correct=0,
                partially_correct=0,
                incorrect=0,
                no_response=0,
                rating=CompetencyRating.UNTESTED,
                summary="0/0 correct",
            ),
        }
        current_scores = {
            "comp_proc": CompetencyScore(
                competency_id="comp_proc",
                competency_name="Procedural Comprehension",
                total_cases=0,
                correct=0,
                partially_correct=0,
                incorrect=0,
                no_response=0,
                rating=CompetencyRating.UNTESTED,
                summary="0/0 correct",
            ),
        }
        baseline = _make_eval_report("eval_001", "model-v1", "disc_maint", baseline_scores)
        current = _make_eval_report("eval_002", "model-v2", "disc_maint", current_scores)

        report = checker.compare(baseline, current, ChangeType.RETRAIN)
        assert report.overall_verdict == "pass"


# ===================================================================
# TestVersionManager
# ===================================================================


class TestVersionManager:
    """Tests for VersionManager persistence and version tracking."""

    def test_register_and_get(self, tmp_path: Path) -> None:
        """Registered version can be retrieved by ID."""
        mgr = VersionManager(tmp_path / "versions")
        entry = _make_version_entry()
        mgr.register_version(entry)
        result = mgr.get_version("ver_001")
        assert result is not None
        assert result.version_id == "ver_001"
        assert result.model_name == "maint-lora-v1"

    def test_get_nonexistent(self, tmp_path: Path) -> None:
        """get_version returns None for unknown ID."""
        mgr = VersionManager(tmp_path / "versions")
        assert mgr.get_version("nonexistent") is None

    def test_list_all(self, tmp_path: Path) -> None:
        """list_versions returns all registered versions."""
        mgr = VersionManager(tmp_path / "versions")
        mgr.register_version(
            _make_version_entry("ver_001", created_at=datetime(2026, 2, 20, 10, 0, 0))
        )
        mgr.register_version(
            _make_version_entry(
                "ver_002",
                model_name="maint-lora-v2",
                created_at=datetime(2026, 2, 20, 11, 0, 0),
            )
        )
        versions = mgr.list_versions()
        assert len(versions) == 2

    def test_list_by_discipline(self, tmp_path: Path) -> None:
        """list_versions filters by discipline_id."""
        mgr = VersionManager(tmp_path / "versions")
        mgr.register_version(_make_version_entry("ver_001", discipline_id="disc_maint"))
        mgr.register_version(_make_version_entry("ver_002", discipline_id="disc_other"))
        maint_versions = mgr.list_versions(discipline_id="disc_maint")
        assert len(maint_versions) == 1
        assert maint_versions[0].discipline_id == "disc_maint"

    def test_get_active_version(self, tmp_path: Path) -> None:
        """get_active_version returns the active version for a discipline."""
        mgr = VersionManager(tmp_path / "versions")
        mgr.register_version(_make_version_entry("ver_001", is_active=False))
        mgr.register_version(
            _make_version_entry(
                "ver_002",
                model_name="maint-lora-v2",
                is_active=True,
                created_at=datetime(2026, 2, 20, 11, 0, 0),
            )
        )
        active = mgr.get_active_version("disc_maint")
        assert active is not None
        assert active.version_id == "ver_002"

    def test_get_active_version_none(self, tmp_path: Path) -> None:
        """get_active_version returns None when no active version exists."""
        mgr = VersionManager(tmp_path / "versions")
        assert mgr.get_active_version("disc_maint") is None

    def test_set_active(self, tmp_path: Path) -> None:
        """set_active deactivates previous and activates target."""
        mgr = VersionManager(tmp_path / "versions")
        mgr.register_version(_make_version_entry("ver_001", is_active=True))
        mgr.register_version(
            _make_version_entry(
                "ver_002",
                model_name="maint-lora-v2",
                is_active=False,
                created_at=datetime(2026, 2, 20, 11, 0, 0),
            )
        )

        mgr.set_active("ver_002")

        v1 = mgr.get_version("ver_001")
        v2 = mgr.get_version("ver_002")
        assert v1 is not None and v1.is_active is False
        assert v2 is not None and v2.is_active is True

    def test_set_active_unknown_raises(self, tmp_path: Path) -> None:
        """set_active raises RegressionError for unknown version_id."""
        mgr = VersionManager(tmp_path / "versions")
        with pytest.raises(RegressionError, match="not found"):
            mgr.set_active("nonexistent")

    def test_rollback(self, tmp_path: Path) -> None:
        """rollback activates the previous version for a discipline."""
        mgr = VersionManager(tmp_path / "versions")
        mgr.register_version(
            _make_version_entry(
                "ver_001",
                is_active=False,
                created_at=datetime(2026, 2, 20, 10, 0, 0),
            )
        )
        mgr.register_version(
            _make_version_entry(
                "ver_002",
                model_name="maint-lora-v2",
                is_active=True,
                created_at=datetime(2026, 2, 20, 11, 0, 0),
            )
        )

        rolled = mgr.rollback("disc_maint")
        assert rolled is not None
        assert rolled.version_id == "ver_001"

        v1 = mgr.get_version("ver_001")
        v2 = mgr.get_version("ver_002")
        assert v1 is not None and v1.is_active is True
        assert v2 is not None and v2.is_active is False

    def test_rollback_no_previous(self, tmp_path: Path) -> None:
        """rollback returns None when only one version exists."""
        mgr = VersionManager(tmp_path / "versions")
        mgr.register_version(_make_version_entry("ver_001", is_active=True))
        result = mgr.rollback("disc_maint")
        assert result is None

    def test_rollback_empty(self, tmp_path: Path) -> None:
        """rollback returns None when no versions exist."""
        mgr = VersionManager(tmp_path / "versions")
        result = mgr.rollback("disc_maint")
        assert result is None

    def test_get_version_history(self, tmp_path: Path) -> None:
        """get_version_history returns chronological list."""
        mgr = VersionManager(tmp_path / "versions")
        mgr.register_version(
            _make_version_entry(
                "ver_002",
                created_at=datetime(2026, 2, 20, 11, 0, 0),
            )
        )
        mgr.register_version(
            _make_version_entry(
                "ver_001",
                created_at=datetime(2026, 2, 20, 10, 0, 0),
            )
        )
        history = mgr.get_version_history("disc_maint")
        assert len(history) == 2
        assert history[0].version_id == "ver_001"
        assert history[1].version_id == "ver_002"

    def test_empty_history(self, tmp_path: Path) -> None:
        """get_version_history returns empty list for unknown discipline."""
        mgr = VersionManager(tmp_path / "versions")
        history = mgr.get_version_history("disc_unknown")
        assert history == []


# ===================================================================
# TestRegressionRunner
# ===================================================================


class TestRegressionRunner:
    """Tests for RegressionRunner orchestration."""

    def test_run_regression_check(self, tmp_path: Path) -> None:
        """Full regression check returns a report with verdicts."""
        checker = RegressionChecker(RegressionConfig())
        vm = VersionManager(tmp_path / "versions")
        runner = RegressionRunner(checker, vm, tmp_path / "reports")

        baseline_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 8, CompetencyRating.STRONG
            ),
        }
        current_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 5, CompetencyRating.ADEQUATE
            ),
        }
        baseline = _make_eval_report("eval_001", "model-v1", "disc_maint", baseline_scores)
        current = _make_eval_report("eval_002", "model-v2", "disc_maint", current_scores)

        report = runner.run_regression_check(baseline, current, ChangeType.RETRAIN)
        assert report.baseline_run_id == "eval_001"
        assert report.current_run_id == "eval_002"
        assert len(report.regressions) == 1
        assert report.overall_verdict == "fail"

    def test_save_and_load_report(self, tmp_path: Path) -> None:
        """Reports can be saved and loaded by ID."""
        checker = RegressionChecker(RegressionConfig())
        vm = VersionManager(tmp_path / "versions")
        runner = RegressionRunner(checker, vm, tmp_path / "reports")

        scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 8, CompetencyRating.STRONG
            ),
        }
        baseline = _make_eval_report("eval_001", "model-v1", "disc_maint", scores)
        current = _make_eval_report("eval_002", "model-v1", "disc_maint", scores)

        report = runner.run_regression_check(baseline, current, ChangeType.RETRAIN)
        saved_path = runner.save_report(report)
        assert saved_path.exists()

        loaded = runner.load_report(report.report_id)
        assert loaded.report_id == report.report_id
        assert loaded.overall_verdict == report.overall_verdict

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        """load_report raises RegressionError for unknown report."""
        checker = RegressionChecker(RegressionConfig())
        vm = VersionManager(tmp_path / "versions")
        runner = RegressionRunner(checker, vm, tmp_path / "reports")

        with pytest.raises(RegressionError, match="not found"):
            runner.load_report("nonexistent")

    def test_list_reports(self, tmp_path: Path) -> None:
        """list_reports returns summary dicts of saved reports."""
        checker = RegressionChecker(RegressionConfig())
        vm = VersionManager(tmp_path / "versions")
        runner = RegressionRunner(checker, vm, tmp_path / "reports")

        scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 8, CompetencyRating.STRONG
            ),
        }
        baseline = _make_eval_report("eval_001", "model-v1", "disc_maint", scores)
        current = _make_eval_report("eval_002", "model-v1", "disc_maint", scores)

        report = runner.run_regression_check(baseline, current, ChangeType.RETRAIN)
        runner.save_report(report)

        summaries = runner.list_reports()
        assert len(summaries) == 1
        assert summaries[0]["report_id"] == report.report_id

    def test_list_reports_filter_discipline(self, tmp_path: Path) -> None:
        """list_reports filters by discipline_id."""
        checker = RegressionChecker(RegressionConfig())
        vm = VersionManager(tmp_path / "versions")
        runner = RegressionRunner(checker, vm, tmp_path / "reports")

        scores_maint = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 8, CompetencyRating.STRONG
            ),
        }
        scores_other = {
            "comp_other": _make_competency_score(
                "comp_other", "Other Skill", 10, 7, CompetencyRating.ADEQUATE
            ),
        }

        baseline_m = _make_eval_report("eval_001", "model-v1", "disc_maint", scores_maint)
        current_m = _make_eval_report("eval_002", "model-v1", "disc_maint", scores_maint)
        report_m = runner.run_regression_check(baseline_m, current_m, ChangeType.RETRAIN)
        runner.save_report(report_m)

        baseline_o = _make_eval_report("eval_003", "model-v1", "disc_other", scores_other)
        current_o = _make_eval_report("eval_004", "model-v1", "disc_other", scores_other)
        report_o = runner.run_regression_check(baseline_o, current_o, ChangeType.RETRAIN)
        runner.save_report(report_o)

        maint_reports = runner.list_reports(discipline_id="disc_maint")
        assert len(maint_reports) == 1
        assert maint_reports[0]["discipline_id"] == "disc_maint"

    def test_get_trend(self, tmp_path: Path) -> None:
        """get_trend returns accuracy over versions for a competency."""
        checker = RegressionChecker(RegressionConfig())
        vm = VersionManager(tmp_path / "versions")
        runner = RegressionRunner(checker, vm, tmp_path / "reports")

        # Register two versions with evaluation reports
        vm.register_version(
            _make_version_entry(
                "ver_001",
                evaluation_run_id="eval_001",
                created_at=datetime(2026, 2, 20, 10, 0, 0),
            )
        )
        vm.register_version(
            _make_version_entry(
                "ver_002",
                evaluation_run_id="eval_002",
                created_at=datetime(2026, 2, 20, 11, 0, 0),
            )
        )

        # Save corresponding regression reports
        scores_v1 = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 7, CompetencyRating.ADEQUATE
            ),
        }
        scores_v2 = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 9, CompetencyRating.STRONG
            ),
        }
        baseline = _make_eval_report("eval_001", "model-v1", "disc_maint", scores_v1)
        current = _make_eval_report("eval_002", "model-v2", "disc_maint", scores_v2)
        report = runner.run_regression_check(baseline, current, ChangeType.RETRAIN)
        runner.save_report(report)

        trend = runner.get_trend("disc_maint", "comp_proc")
        assert len(trend) >= 1
        # The trend should contain accuracy data points
        for point in trend:
            assert "accuracy" in point
            assert "run_id" in point


# ===================================================================
# TestPlainLanguage
# ===================================================================


class TestPlainLanguage:
    """Tests that summaries use SME-friendly language without ML jargon."""

    def test_summary_no_ml_jargon(self) -> None:
        """Plain language summary does not contain ML terminology."""
        checker = RegressionChecker(RegressionConfig())
        baseline_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 8, CompetencyRating.STRONG
            ),
            "comp_fault": _make_competency_score(
                "comp_fault", "Fault Isolation", 10, 4, CompetencyRating.NEEDS_IMPROVEMENT
            ),
        }
        current_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 5, CompetencyRating.ADEQUATE
            ),
            "comp_fault": _make_competency_score(
                "comp_fault", "Fault Isolation", 10, 8, CompetencyRating.STRONG
            ),
        }
        baseline = _make_eval_report("eval_001", "model-v1", "disc_maint", baseline_scores)
        current = _make_eval_report("eval_002", "model-v2", "disc_maint", current_scores)

        report = checker.compare(baseline, current, ChangeType.RETRAIN)
        summary = report.plain_language_summary.lower()

        # Must not contain ML jargon
        jargon = ["loss", "perplexity", "f1", "precision", "recall", "epoch", "gradient"]
        for term in jargon:
            assert term not in summary, f"Summary contains ML jargon: {term}"

    def test_summary_mentions_competency_names(self) -> None:
        """Summary references competencies by human-readable name."""
        checker = RegressionChecker(RegressionConfig())
        baseline_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 8, CompetencyRating.STRONG
            ),
        }
        current_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 5, CompetencyRating.ADEQUATE
            ),
        }
        baseline = _make_eval_report("eval_001", "model-v1", "disc_maint", baseline_scores)
        current = _make_eval_report("eval_002", "model-v2", "disc_maint", current_scores)

        report = checker.compare(baseline, current, ChangeType.RETRAIN)
        assert "Procedural Comprehension" in report.plain_language_summary

    def test_summary_mentions_correct_counts(self) -> None:
        """Summary includes X/Y correct style language."""
        checker = RegressionChecker(RegressionConfig())
        baseline_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 8, CompetencyRating.STRONG
            ),
        }
        current_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 5, CompetencyRating.ADEQUATE
            ),
        }
        baseline = _make_eval_report("eval_001", "model-v1", "disc_maint", baseline_scores)
        current = _make_eval_report("eval_002", "model-v2", "disc_maint", current_scores)

        report = checker.compare(baseline, current, ChangeType.RETRAIN)
        # Should contain something like "8/10" or "5/10"
        assert "/10" in report.plain_language_summary

    def test_pass_summary(self) -> None:
        """Pass verdict produces a positive summary."""
        checker = RegressionChecker(RegressionConfig())
        scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 8, CompetencyRating.STRONG
            ),
        }
        baseline = _make_eval_report("eval_001", "model-v1", "disc_maint", scores)
        current = _make_eval_report("eval_002", "model-v2", "disc_maint", scores)

        report = checker.compare(baseline, current, ChangeType.RETRAIN)
        summary_lower = report.plain_language_summary.lower()
        assert "no regression" in summary_lower or "pass" in summary_lower


# ===================================================================
# TestIntegration
# ===================================================================


class TestIntegration:
    """End-to-end integration: version -> evaluate -> regress -> rollback."""

    def test_full_lifecycle(self, tmp_path: Path) -> None:
        """Simulate: v1 eval -> retrain -> v2 eval -> regress -> rollback."""
        checker = RegressionChecker(RegressionConfig())
        vm = VersionManager(tmp_path / "versions")
        runner = RegressionRunner(checker, vm, tmp_path / "reports")

        # Step 1: Register version 1 with good scores
        vm.register_version(
            _make_version_entry(
                "ver_001",
                model_name="maint-lora-v1",
                evaluation_run_id="eval_001",
                is_active=True,
                created_at=datetime(2026, 2, 20, 10, 0, 0),
            )
        )
        active = vm.get_active_version("disc_maint")
        assert active is not None
        assert active.version_id == "ver_001"

        # Step 2: Simulate evaluation v1 (baseline)
        baseline_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 9, CompetencyRating.STRONG
            ),
            "comp_fault": _make_competency_score(
                "comp_fault", "Fault Isolation", 10, 8, CompetencyRating.STRONG
            ),
        }
        baseline_report = _make_eval_report(
            "eval_001", "maint-lora-v1", "disc_maint", baseline_scores
        )

        # Step 3: Retrain produces v2 with worse scores
        vm.register_version(
            _make_version_entry(
                "ver_002",
                model_name="maint-lora-v2",
                evaluation_run_id="eval_002",
                is_active=False,
                created_at=datetime(2026, 2, 20, 11, 0, 0),
            )
        )

        current_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 5, CompetencyRating.ADEQUATE
            ),
            "comp_fault": _make_competency_score(
                "comp_fault", "Fault Isolation", 10, 8, CompetencyRating.STRONG
            ),
        }
        current_report = _make_eval_report(
            "eval_002", "maint-lora-v2", "disc_maint", current_scores
        )

        # Step 4: Run regression check
        regr_report = runner.run_regression_check(
            baseline_report, current_report, ChangeType.RETRAIN
        )
        runner.save_report(regr_report)

        assert regr_report.overall_verdict == "fail"
        assert len(regr_report.regressions) == 1
        assert regr_report.regressions[0].competency_id == "comp_proc"
        assert "comp_fault" in regr_report.unchanged

        # Step 5: Regression detected -- rollback
        rolled = vm.rollback("disc_maint")
        assert rolled is not None
        assert rolled.version_id == "ver_001"

        final_active = vm.get_active_version("disc_maint")
        assert final_active is not None
        assert final_active.version_id == "ver_001"

    def test_successful_upgrade(self, tmp_path: Path) -> None:
        """Successful upgrade: v2 is better, no rollback needed."""
        checker = RegressionChecker(RegressionConfig())
        vm = VersionManager(tmp_path / "versions")
        runner = RegressionRunner(checker, vm, tmp_path / "reports")

        vm.register_version(
            _make_version_entry(
                "ver_001",
                is_active=True,
                created_at=datetime(2026, 2, 20, 10, 0, 0),
            )
        )

        baseline_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 6, CompetencyRating.ADEQUATE
            ),
        }
        current_scores = {
            "comp_proc": _make_competency_score(
                "comp_proc", "Procedural Comprehension", 10, 9, CompetencyRating.STRONG
            ),
        }
        baseline = _make_eval_report("eval_001", "model-v1", "disc_maint", baseline_scores)
        current = _make_eval_report("eval_002", "model-v2", "disc_maint", current_scores)

        report = runner.run_regression_check(baseline, current, ChangeType.RETRAIN)
        assert report.overall_verdict == "pass"
        assert len(report.improvements) == 1

        # Promote v2
        vm.register_version(
            _make_version_entry(
                "ver_002",
                model_name="maint-lora-v2",
                is_active=False,
                created_at=datetime(2026, 2, 20, 11, 0, 0),
            )
        )
        vm.set_active("ver_002")
        active = vm.get_active_version("disc_maint")
        assert active is not None
        assert active.version_id == "ver_002"
