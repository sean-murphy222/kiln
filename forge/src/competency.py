"""Competency mapping system for structured competency map curation.

Translates discovery session seeds into a validated, hierarchical
competency map ready for example elicitation. Experts can refine,
reorganize, and validate competencies surfaced during discovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from forge.src.models import Competency, DiscoverySession
from forge.src.storage import ForgeStorage


class CompetencyMapError(Exception):
    """Raised for invalid competency map operations."""


@dataclass
class CompetencyMapSummary:
    """Summary of a discipline's competency map status.

    Attributes:
        discipline_id: ID of the discipline.
        total_competencies: Total number of competencies.
        root_competencies: Number of top-level competencies.
        child_competencies: Number of child competencies.
        total_coverage_target: Sum of all coverage targets.
        coverage_complete: True when all competencies meet targets.
        gap_competency_names: Names of competencies below target.
        estimated_examples_needed: Total examples needed to fill gaps.
    """

    discipline_id: str
    total_competencies: int
    root_competencies: int
    child_competencies: int
    total_coverage_target: int
    coverage_complete: bool
    gap_competency_names: list[str] = field(default_factory=list)
    estimated_examples_needed: int = 0


class CompetencyMapper:
    """Engine for refining and organizing competency maps.

    Wraps ForgeStorage to provide a focused API for competency map
    curation: loading seeds from discovery, adding/updating/removing
    competencies, building hierarchy, and validating readiness.

    Args:
        storage: ForgeStorage instance for persistence.

    Example::

        mapper = CompetencyMapper(storage)
        comps = mapper.load_from_discovery(disc_id, session)
        mapper.add_competency(disc_id, "New Skill", "Description")
        summary = mapper.finalize_map(disc_id)
    """

    def __init__(self, storage: ForgeStorage) -> None:
        self._storage = storage

    def load_from_discovery(
        self,
        discipline_id: str,
        session: DiscoverySession,
    ) -> list[Competency]:
        """Load competencies seeded by discovery session completion.

        Confirms the discovery session produced this discipline and
        that seed competencies exist in storage.

        Args:
            discipline_id: ID of the discipline to load.
            session: The completed discovery session.

        Returns:
            List of seed competencies.

        Raises:
            CompetencyMapError: If session doesn't match or no competencies.
        """
        if session.generated_discipline_id != discipline_id:
            raise CompetencyMapError(
                f"Session {session.id} did not generate " f"discipline {discipline_id}"
            )
        comps = self._storage.get_competencies_for_discipline(discipline_id)
        if not comps:
            raise CompetencyMapError(f"No competencies found for discipline {discipline_id}")
        return comps

    def add_competency(
        self,
        discipline_id: str,
        name: str,
        description: str,
        parent_id: str | None = None,
        coverage_target: int = 25,
    ) -> Competency:
        """Add a new competency to a discipline.

        Args:
            discipline_id: Parent discipline ID.
            name: Human-readable competency name.
            description: What this competency covers.
            parent_id: Parent competency ID for hierarchy.
            coverage_target: Number of examples needed.

        Returns:
            The created Competency.

        Raises:
            CompetencyMapError: If discipline missing or parent invalid.
        """
        disc = self._storage.get_discipline(discipline_id)
        if disc is None:
            raise CompetencyMapError(f"Discipline not found: {discipline_id}")
        if parent_id is not None:
            self._validate_parent(parent_id, discipline_id)

        comp = Competency(
            id=Competency.generate_id(),
            name=name,
            description=description,
            discipline_id=discipline_id,
            parent_id=parent_id,
            coverage_target=coverage_target,
        )
        self._storage.create_competency(comp)
        return comp

    def update_competency(
        self,
        competency_id: str,
        name: str | None = None,
        description: str | None = None,
        coverage_target: int | None = None,
    ) -> Competency:
        """Update an existing competency's fields.

        Only provided (non-None) fields are changed.

        Args:
            competency_id: ID of competency to update.
            name: New name (if provided).
            description: New description (if provided).
            coverage_target: New target (if provided).

        Returns:
            The updated Competency.

        Raises:
            CompetencyMapError: If competency not found.
        """
        comp = self._storage.get_competency(competency_id)
        if comp is None:
            raise CompetencyMapError(f"Competency not found: {competency_id}")
        if name is not None:
            comp.name = name
        if description is not None:
            comp.description = description
        if coverage_target is not None:
            comp.coverage_target = coverage_target
        self._storage.update_competency(comp)
        return comp

    def set_parent(
        self,
        competency_id: str,
        parent_id: str | None,
    ) -> Competency:
        """Set or clear a competency's parent.

        Args:
            competency_id: ID of competency to reparent.
            parent_id: New parent ID, or None to make root.

        Returns:
            The updated Competency.

        Raises:
            CompetencyMapError: If not found, parent invalid, or cycle.
        """
        comp = self._storage.get_competency(competency_id)
        if comp is None:
            raise CompetencyMapError(f"Competency not found: {competency_id}")
        if parent_id is not None:
            self._validate_parent(parent_id, comp.discipline_id)
            if parent_id == competency_id:
                raise CompetencyMapError("Cannot set competency as its own parent")
            if self._is_descendant(parent_id, competency_id):
                raise CompetencyMapError(
                    f"Circular reference: {parent_id} is a " f"descendant of {competency_id}"
                )
        comp.parent_id = parent_id
        self._storage.update_competency(comp)
        return comp

    def remove_competency(
        self,
        competency_id: str,
        reassign_children_to: str | None = None,
    ) -> None:
        """Remove a competency from the map.

        Args:
            competency_id: ID of competency to remove.
            reassign_children_to: Reparent children here first.

        Raises:
            CompetencyMapError: If not found, has children without
                reassignment, or reassignment target invalid.
        """
        comp = self._storage.get_competency(competency_id)
        if comp is None:
            raise CompetencyMapError(f"Competency not found: {competency_id}")
        children = self._get_children(competency_id, comp.discipline_id)
        if children:
            if reassign_children_to is None:
                raise CompetencyMapError(
                    f"Competency {competency_id} has "
                    f"{len(children)} children. "
                    "Provide reassign_children_to or "
                    "remove children first."
                )
            self._validate_parent(reassign_children_to, comp.discipline_id)
            for child in children:
                child.parent_id = reassign_children_to
                self._storage.update_competency(child)
        self._storage.delete_competency(competency_id)

    def get_coverage_summary(self, discipline_id: str) -> CompetencyMapSummary:
        """Build a coverage summary for the competency map.

        Args:
            discipline_id: Discipline to report on.

        Returns:
            CompetencyMapSummary with coverage statistics.
        """
        comps = self._storage.get_competencies_for_discipline(discipline_id)
        report = self._storage.get_coverage_report(discipline_id)

        roots = [c for c in comps if c.parent_id is None]
        children = [c for c in comps if c.parent_id is not None]
        total_target = sum(c.coverage_target for c in comps)

        gap_names = [g["competency_name"] for g in report["gaps"]]
        needed = sum(max(0, g["coverage_target"] - g["example_count"]) for g in report["gaps"])

        return CompetencyMapSummary(
            discipline_id=discipline_id,
            total_competencies=len(comps),
            root_competencies=len(roots),
            child_competencies=len(children),
            total_coverage_target=total_target,
            coverage_complete=report["coverage_complete"],
            gap_competency_names=gap_names,
            estimated_examples_needed=needed,
        )

    def get_competency_tree(self, discipline_id: str) -> list[dict[str, Any]]:
        """Build a nested tree of competencies for display.

        Args:
            discipline_id: Discipline to build tree for.

        Returns:
            List of root nodes with 'competency' and 'children' keys.
        """
        comps = self._storage.get_competencies_for_discipline(discipline_id)
        return self._build_tree(comps)

    def finalize_map(self, discipline_id: str) -> CompetencyMapSummary:
        """Validate and finalize the competency map.

        Checks:
        - At least 3 root competencies.
        - No competency with name == description (unrefined seed).
        - All coverage targets > 0.

        Args:
            discipline_id: Discipline to finalize.

        Returns:
            CompetencyMapSummary on success.

        Raises:
            CompetencyMapError: With all violations listed.
        """
        comps = self._storage.get_competencies_for_discipline(discipline_id)
        violations = self._collect_violations(comps)

        if violations:
            raise CompetencyMapError("Map validation failed:\n- " + "\n- ".join(violations))

        return self.get_coverage_summary(discipline_id)

    # --- Private helpers ---

    @staticmethod
    def _collect_violations(
        comps: list[Competency],
    ) -> list[str]:
        """Collect all finalization violations.

        Args:
            comps: All competencies for the discipline.

        Returns:
            List of violation messages.
        """
        violations: list[str] = []

        roots = [c for c in comps if c.parent_id is None]
        if len(roots) < 3:
            violations.append(f"Need at least 3 root competencies, " f"found {len(roots)}")

        for c in comps:
            if c.name.strip() == c.description.strip():
                violations.append(
                    f"Competency '{c.name}' has " f"name == description (needs refinement)"
                )
            if c.coverage_target <= 0:
                violations.append(f"Competency '{c.name}' has " f"coverage_target <= 0")

        return violations

    def _validate_parent(self, parent_id: str, discipline_id: str) -> None:
        """Validate parent exists and belongs to same discipline.

        Args:
            parent_id: Parent competency ID.
            discipline_id: Expected discipline.

        Raises:
            CompetencyMapError: If parent missing or wrong discipline.
        """
        parent = self._storage.get_competency(parent_id)
        if parent is None:
            raise CompetencyMapError(f"Parent competency not found: {parent_id}")
        if parent.discipline_id != discipline_id:
            raise CompetencyMapError(
                f"Parent {parent_id} belongs to discipline "
                f"{parent.discipline_id}, not {discipline_id}"
            )

    def _get_children(self, competency_id: str, discipline_id: str) -> list[Competency]:
        """Get direct children of a competency.

        Args:
            competency_id: Parent competency ID.
            discipline_id: Discipline to search within.

        Returns:
            List of child competencies.
        """
        all_comps = self._storage.get_competencies_for_discipline(discipline_id)
        return [c for c in all_comps if c.parent_id == competency_id]

    def _is_descendant(self, candidate_id: str, ancestor_id: str) -> bool:
        """Check if candidate is a descendant of ancestor.

        Walks up the parent chain from candidate looking for ancestor.

        Args:
            candidate_id: Potential descendant ID.
            ancestor_id: Potential ancestor ID.

        Returns:
            True if candidate is a descendant of ancestor.
        """
        current = self._storage.get_competency(candidate_id)
        visited: set[str] = set()
        while current is not None and current.parent_id is not None:
            if current.parent_id in visited:
                break
            visited.add(current.id)
            if current.parent_id == ancestor_id:
                return True
            current = self._storage.get_competency(current.parent_id)
        return False

    @staticmethod
    def _build_tree(
        competencies: list[Competency],
    ) -> list[dict[str, Any]]:
        """Build nested tree from flat competency list.

        Args:
            competencies: All competencies for a discipline.

        Returns:
            List of root nodes with nested children.
        """
        by_id: dict[str, dict[str, Any]] = {}
        for c in competencies:
            by_id[c.id] = {"competency": c, "children": []}

        roots: list[dict[str, Any]] = []
        for c in competencies:
            node = by_id[c.id]
            if c.parent_id is None or c.parent_id not in by_id:
                roots.append(node)
            else:
                by_id[c.parent_id]["children"].append(node)

        return roots
