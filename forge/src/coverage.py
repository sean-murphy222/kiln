"""Real-time coverage analysis for competency-based training curricula.

Tracks which competency areas have sufficient examples, identifies
gaps, and produces prioritized recommendations for where to focus
example elicitation efforts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from forge.src.storage import ForgeStorage


class CoverageAnalysisError(Exception):
    """Raised for coverage analysis failures."""


@dataclass
class CompetencyCoverage:
    """Coverage status for a single competency.

    Attributes:
        competency_id: ID of the competency.
        competency_name: Human-readable name.
        parent_id: Parent competency ID (None for root).
        example_count: Current number of training examples.
        coverage_target: Required number of examples.
        coverage_ratio: Fraction of target met (0.0-1.0+).
        is_met: True when example_count >= coverage_target.
        gap: Number of additional examples needed (0 if met).
    """

    competency_id: str
    competency_name: str
    parent_id: str | None
    example_count: int
    coverage_target: int
    coverage_ratio: float
    is_met: bool
    gap: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "competency_id": self.competency_id,
            "competency_name": self.competency_name,
            "parent_id": self.parent_id,
            "example_count": self.example_count,
            "coverage_target": self.coverage_target,
            "coverage_ratio": self.coverage_ratio,
            "is_met": self.is_met,
            "gap": self.gap,
        }


@dataclass
class CoverageRecommendation:
    """A prioritized recommendation for improving coverage.

    Attributes:
        competency_id: ID of the competency needing attention.
        competency_name: Human-readable name.
        priority: Priority level (1=highest, 3=lowest).
        reason: Why this competency needs attention.
        examples_needed: How many more examples to add.
    """

    competency_id: str
    competency_name: str
    priority: int
    reason: str
    examples_needed: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "competency_id": self.competency_id,
            "competency_name": self.competency_name,
            "priority": self.priority,
            "reason": self.reason,
            "examples_needed": self.examples_needed,
        }


@dataclass
class CoverageSnapshot:
    """Complete coverage analysis for a discipline at a point in time.

    Attributes:
        discipline_id: ID of the analyzed discipline.
        total_competencies: Total number of competencies.
        met_count: Competencies that meet their target.
        gap_count: Competencies below their target.
        overall_ratio: Weighted coverage ratio across all competencies.
        total_examples: Total training examples.
        total_test_examples: Total held-out test examples.
        total_gap: Total additional examples needed.
        coverage_complete: True when all competencies meet targets.
        competencies: Per-competency coverage details.
        recommendations: Prioritized improvement suggestions.
    """

    discipline_id: str
    total_competencies: int
    met_count: int
    gap_count: int
    overall_ratio: float
    total_examples: int
    total_test_examples: int
    total_gap: int
    coverage_complete: bool
    competencies: list[CompetencyCoverage] = field(default_factory=list)
    recommendations: list[CoverageRecommendation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "discipline_id": self.discipline_id,
            "total_competencies": self.total_competencies,
            "met_count": self.met_count,
            "gap_count": self.gap_count,
            "overall_ratio": self.overall_ratio,
            "total_examples": self.total_examples,
            "total_test_examples": self.total_test_examples,
            "total_gap": self.total_gap,
            "coverage_complete": self.coverage_complete,
            "competencies": [c.to_dict() for c in self.competencies],
            "recommendations": [r.to_dict() for r in self.recommendations],
        }


class CoverageAnalyzer:
    """Real-time coverage analysis engine.

    Wraps ForgeStorage to produce coverage snapshots with
    per-competency metrics and prioritized recommendations.

    Args:
        storage: ForgeStorage instance.

    Example::

        analyzer = CoverageAnalyzer(storage)
        snapshot = analyzer.analyze(discipline_id)
        print(f"{snapshot.overall_ratio:.0%} coverage")
        for rec in snapshot.recommendations:
            print(f"  [{rec.priority}] {rec.competency_name}: {rec.reason}")
    """

    def __init__(self, storage: ForgeStorage) -> None:
        self._storage = storage

    def analyze(self, discipline_id: str) -> CoverageSnapshot:
        """Produce a full coverage snapshot for a discipline.

        Args:
            discipline_id: ID of the discipline to analyze.

        Returns:
            CoverageSnapshot with metrics and recommendations.

        Raises:
            CoverageAnalysisError: If the discipline has no competencies.
        """
        report = self._storage.get_coverage_report(discipline_id)
        competencies = self._storage.get_competencies_for_discipline(discipline_id)

        if not competencies:
            raise CoverageAnalysisError(f"No competencies found for discipline '{discipline_id}'")

        parent_map = {c.id: c.parent_id for c in competencies}

        coverage_items = self._build_coverage_items(report, parent_map)
        recommendations = self._build_recommendations(coverage_items)
        overall_ratio = self._compute_overall_ratio(coverage_items)

        met = sum(1 for c in coverage_items if c.is_met)
        total_gap = sum(c.gap for c in coverage_items)

        return CoverageSnapshot(
            discipline_id=discipline_id,
            total_competencies=len(coverage_items),
            met_count=met,
            gap_count=len(coverage_items) - met,
            overall_ratio=overall_ratio,
            total_examples=report["total_examples"],
            total_test_examples=report["total_test_examples"],
            total_gap=total_gap,
            coverage_complete=report["coverage_complete"],
            competencies=coverage_items,
            recommendations=recommendations,
        )

    def get_competency_coverage(
        self, discipline_id: str, competency_id: str
    ) -> CompetencyCoverage | None:
        """Get coverage for a single competency.

        Args:
            discipline_id: ID of the discipline.
            competency_id: ID of the competency.

        Returns:
            CompetencyCoverage or None if not found.
        """
        snapshot = self.analyze(discipline_id)
        for cc in snapshot.competencies:
            if cc.competency_id == competency_id:
                return cc
        return None

    def is_discipline_ready(self, discipline_id: str) -> bool:
        """Check if a discipline has full coverage.

        Args:
            discipline_id: ID of the discipline.

        Returns:
            True if all competencies meet their coverage targets.
        """
        try:
            snapshot = self.analyze(discipline_id)
            return snapshot.coverage_complete
        except CoverageAnalysisError:
            return False

    @staticmethod
    def _build_coverage_items(
        report: dict[str, Any],
        parent_map: dict[str, str | None],
    ) -> list[CompetencyCoverage]:
        """Build per-competency coverage from the storage report.

        Args:
            report: Raw coverage report from storage.
            parent_map: Mapping of competency_id to parent_id.

        Returns:
            List of CompetencyCoverage items.
        """
        items: list[CompetencyCoverage] = []
        for entry in report["competency_coverage"]:
            cid = entry["competency_id"]
            target = entry["coverage_target"]
            count = entry["example_count"]
            ratio = count / target if target > 0 else 1.0
            gap = max(0, target - count)
            items.append(
                CompetencyCoverage(
                    competency_id=cid,
                    competency_name=entry["competency_name"],
                    parent_id=parent_map.get(cid),
                    example_count=count,
                    coverage_target=target,
                    coverage_ratio=round(ratio, 3),
                    is_met=entry["met"],
                    gap=gap,
                )
            )
        return items

    @staticmethod
    def _build_recommendations(
        items: list[CompetencyCoverage],
    ) -> list[CoverageRecommendation]:
        """Generate prioritized recommendations from coverage items.

        Priority levels:
        - 1 (critical): 0 examples (empty competency)
        - 2 (high): < 50% of target
        - 3 (medium): 50-99% of target

        Args:
            items: Per-competency coverage items.

        Returns:
            Recommendations sorted by priority then gap size.
        """
        recs: list[CoverageRecommendation] = []
        for item in items:
            if item.is_met:
                continue
            if item.example_count == 0:
                priority = 1
                reason = "No examples yet — start here"
            elif item.coverage_ratio < 0.5:
                priority = 2
                reason = (
                    f"Only {item.coverage_ratio:.0%} coverage "
                    f"({item.example_count}/{item.coverage_target})"
                )
            else:
                priority = 3
                reason = (
                    f"Nearly there: {item.coverage_ratio:.0%} coverage "
                    f"({item.example_count}/{item.coverage_target})"
                )
            recs.append(
                CoverageRecommendation(
                    competency_id=item.competency_id,
                    competency_name=item.competency_name,
                    priority=priority,
                    reason=reason,
                    examples_needed=item.gap,
                )
            )
        recs.sort(key=lambda r: (r.priority, -r.examples_needed))
        return recs

    @staticmethod
    def _compute_overall_ratio(items: list[CompetencyCoverage]) -> float:
        """Compute weighted average coverage ratio.

        Weights by coverage_target so larger competencies
        count proportionally more.

        Args:
            items: Per-competency coverage items.

        Returns:
            Overall ratio between 0.0 and 1.0.
        """
        total_target = sum(i.coverage_target for i in items)
        if total_target == 0:
            return 1.0
        total_achieved = sum(min(i.example_count, i.coverage_target) for i in items)
        return round(total_achieved / total_target, 3)
