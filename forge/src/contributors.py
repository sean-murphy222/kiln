"""Multi-contributor workflow management for Forge.

Provides role-based access control, review queues, conflict
resolution, and contribution attribution tracking within
discipline teams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from forge.src.models import (
    ContributorRole,
    DisciplineContributor,
    Example,
    ReviewStatus,
)
from forge.src.storage import ForgeStorage, ForgeStorageError


class ContributorError(Exception):
    """Raised for contributor workflow errors."""


class ReviewDecision(str, Enum):
    """Decision a reviewer makes on a pending example.

    Attributes:
        APPROVE: Accept the example for training.
        REJECT: Reject the example entirely.
        NEEDS_REVISION: Send back to the contributor for changes.
    """

    APPROVE = "approve"
    REJECT = "reject"
    NEEDS_REVISION = "needs_revision"


_DECISION_TO_STATUS: dict[ReviewDecision, ReviewStatus] = {
    ReviewDecision.APPROVE: ReviewStatus.APPROVED,
    ReviewDecision.REJECT: ReviewStatus.REJECTED,
    ReviewDecision.NEEDS_REVISION: ReviewStatus.NEEDS_REVISION,
}


@dataclass
class ReviewItem:
    """A single item in the review queue.

    Attributes:
        example_id: ID of the example under review.
        example: The full Example object.
        contributor_name: Display name of the contributor who submitted it.
        submitted_at: When the example was created.
        competency_name: Name of the target competency.
    """

    example_id: str
    example: Example
    contributor_name: str
    submitted_at: datetime
    competency_name: str


@dataclass
class ReviewQueue:
    """A queue of examples awaiting review for a discipline.

    Attributes:
        discipline_id: The discipline these examples belong to.
        items: List of ReviewItem objects.
        total_count: Total number of items in the queue.
    """

    discipline_id: str
    items: list[ReviewItem] = field(default_factory=list)
    total_count: int = 0

    @property
    def pending_count(self) -> int:
        """Return the number of pending review items."""
        return len(self.items)


@dataclass
class ContributionStats:
    """Attribution statistics for a contributor within a discipline.

    Attributes:
        contributor_id: The contributor's unique ID.
        contributor_name: Display name.
        total_examples: Total examples submitted.
        approved: Number approved.
        rejected: Number rejected.
        pending: Number still pending review.
        by_competency: Example count per competency ID.
    """

    contributor_id: str
    contributor_name: str
    total_examples: int = 0
    approved: int = 0
    rejected: int = 0
    pending: int = 0
    by_competency: dict[str, int] = field(default_factory=dict)


class ContributorManager:
    """Manages multi-contributor workflows within disciplines.

    Handles role assignment, review queues, conflict resolution,
    and contribution attribution tracking.

    Args:
        storage: ForgeStorage instance for persistence.

    Example::

        manager = ContributorManager(storage)
        manager.assign_to_discipline("contrib_001", "disc_001", ContributorRole.LEAD)
        queue = manager.get_review_queue("disc_001")
    """

    def __init__(self, storage: ForgeStorage) -> None:
        self._storage = storage

    def assign_to_discipline(
        self,
        contributor_id: str,
        discipline_id: str,
        role: ContributorRole,
        competency_area_ids: list[str] | None = None,
    ) -> DisciplineContributor:
        """Add a contributor to a discipline with a role.

        Args:
            contributor_id: The contributor to assign.
            discipline_id: The target discipline.
            role: Role within the discipline.
            competency_area_ids: Competencies this contributor owns.
                Empty list means access to all competencies.

        Returns:
            The created DisciplineContributor association.

        Raises:
            ContributorError: If the contributor is already assigned.
        """
        dc = DisciplineContributor(
            discipline_id=discipline_id,
            contributor_id=contributor_id,
            role=role,
            competency_area_ids=competency_area_ids or [],
        )
        try:
            self._storage.add_contributor_to_discipline(dc)
        except ForgeStorageError as exc:
            raise ContributorError(
                f"Contributor {contributor_id} already assigned to {discipline_id}"
            ) from exc
        return dc

    def update_role(
        self,
        contributor_id: str,
        discipline_id: str,
        new_role: ContributorRole,
    ) -> DisciplineContributor:
        """Change a contributor's role within a discipline.

        Args:
            contributor_id: The contributor to update.
            discipline_id: The discipline.
            new_role: The new role to assign.

        Returns:
            Updated DisciplineContributor.

        Raises:
            ContributorError: If the contributor is not a member.
        """
        dc = self._get_membership(contributor_id, discipline_id)
        dc.role = new_role
        try:
            self._storage.update_contributor_in_discipline(dc)
        except ForgeStorageError as exc:
            raise ContributorError(
                f"Contributor {contributor_id} not a member of {discipline_id}"
            ) from exc
        return dc

    def assign_competency_areas(
        self,
        contributor_id: str,
        discipline_id: str,
        competency_ids: list[str],
    ) -> DisciplineContributor:
        """Set which competencies a contributor owns in a discipline.

        Args:
            contributor_id: The contributor to update.
            discipline_id: The discipline.
            competency_ids: Competency IDs to assign. Empty list
                means discipline-wide access to all competencies.

        Returns:
            Updated DisciplineContributor.

        Raises:
            ContributorError: If the contributor is not a member.
        """
        dc = self._get_membership(contributor_id, discipline_id)
        dc.competency_area_ids = competency_ids
        try:
            self._storage.update_contributor_in_discipline(dc)
        except ForgeStorageError as exc:
            raise ContributorError(
                f"Contributor {contributor_id} not a member of {discipline_id}"
            ) from exc
        return dc

    def get_review_queue(
        self,
        discipline_id: str,
        reviewer_id: str | None = None,
    ) -> ReviewQueue:
        """Get pending examples awaiting review for a discipline.

        Args:
            discipline_id: The discipline to get the queue for.
            reviewer_id: If provided, filter to the reviewer's
                competency areas only.

        Returns:
            ReviewQueue with pending ReviewItem entries.
        """
        allowed_comp_ids = self._get_allowed_competencies(discipline_id, reviewer_id)
        pending = self._collect_pending_examples(discipline_id, allowed_comp_ids)
        items = self._build_review_items(discipline_id, pending)
        return ReviewQueue(
            discipline_id=discipline_id,
            items=items,
            total_count=len(items),
        )

    def submit_review(
        self,
        reviewer_id: str,
        example_id: str,
        decision: ReviewDecision,
        notes: str = "",
    ) -> Example:
        """Submit a review decision for an example.

        Args:
            reviewer_id: The reviewer's contributor ID.
            example_id: The example being reviewed.
            decision: The review decision (approve/reject/needs_revision).
            notes: Optional review notes appended to context.

        Returns:
            Updated Example with new review status.

        Raises:
            ContributorError: If reviewer lacks permission or
                example not found.
        """
        example = self._get_example_or_raise(example_id)
        self._require_reviewer_permission(reviewer_id, example.discipline_id)
        return self._apply_review(example, reviewer_id, decision, notes)

    def get_contribution_stats(
        self,
        contributor_id: str,
        discipline_id: str,
    ) -> ContributionStats:
        """Get attribution statistics for a contributor in a discipline.

        Args:
            contributor_id: The contributor to report on.
            discipline_id: The discipline scope.

        Returns:
            ContributionStats with counts by status and competency.
        """
        contributor = self._storage.get_contributor(contributor_id)
        name = contributor.name if contributor else contributor_id
        examples = self._get_contributor_examples(contributor_id, discipline_id)
        return self._compute_stats(contributor_id, name, examples)

    def get_discipline_stats(
        self,
        discipline_id: str,
    ) -> list[ContributionStats]:
        """Get per-contributor statistics for a discipline.

        Args:
            discipline_id: The discipline to report on.

        Returns:
            List of ContributionStats, one per contributor.
        """
        members = self._storage.get_discipline_contributors(discipline_id)
        stats_list: list[ContributionStats] = []
        for dc in members:
            stats = self.get_contribution_stats(dc.contributor_id, discipline_id)
            stats_list.append(stats)
        return stats_list

    def check_ownership(
        self,
        contributor_id: str,
        discipline_id: str,
        competency_id: str,
    ) -> bool:
        """Check if a contributor owns a competency area.

        Empty competency_area_ids means discipline-wide ownership
        (access to all competencies).

        Args:
            contributor_id: The contributor to check.
            discipline_id: The discipline scope.
            competency_id: The competency to check ownership of.

        Returns:
            True if the contributor owns the competency area.
        """
        members = self._storage.get_discipline_contributors(discipline_id)
        for dc in members:
            if dc.contributor_id == contributor_id:
                if not dc.competency_area_ids:
                    return True
                return competency_id in dc.competency_area_ids
        return False

    def resolve_conflict(
        self,
        reviewer_id: str,
        example_id_keep: str,
        example_id_reject: str,
    ) -> tuple[Example, Example]:
        """Resolve a conflict between two examples.

        Approves one and rejects the other. Both get reviewer
        attribution.

        Args:
            reviewer_id: The reviewer making the decision.
            example_id_keep: Example to approve.
            example_id_reject: Example to reject.

        Returns:
            Tuple of (kept_example, rejected_example).

        Raises:
            ContributorError: If reviewer lacks permission or
                examples not found.
        """
        kept = self._get_example_or_raise(example_id_keep)
        rejected = self._get_example_or_raise(example_id_reject)
        self._require_reviewer_permission(reviewer_id, kept.discipline_id)

        now = datetime.now()
        kept = self._stamp_review(kept, reviewer_id, ReviewStatus.APPROVED, now)
        rejected = self._stamp_review(rejected, reviewer_id, ReviewStatus.REJECTED, now)
        return kept, rejected

    # ---------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------

    def _get_membership(
        self,
        contributor_id: str,
        discipline_id: str,
    ) -> DisciplineContributor:
        """Find a contributor's membership in a discipline.

        Args:
            contributor_id: The contributor ID.
            discipline_id: The discipline ID.

        Returns:
            DisciplineContributor for the pair.

        Raises:
            ContributorError: If not a member.
        """
        members = self._storage.get_discipline_contributors(discipline_id)
        for dc in members:
            if dc.contributor_id == contributor_id:
                return dc
        raise ContributorError(f"Contributor {contributor_id} is not a member of {discipline_id}")

    def _get_allowed_competencies(
        self,
        discipline_id: str,
        reviewer_id: str | None,
    ) -> list[str] | None:
        """Get competency IDs the reviewer is allowed to review.

        Args:
            discipline_id: The discipline scope.
            reviewer_id: The reviewer, or None for all.

        Returns:
            List of competency IDs, or None for unrestricted.
        """
        if reviewer_id is None:
            return None
        members = self._storage.get_discipline_contributors(discipline_id)
        for dc in members:
            if dc.contributor_id == reviewer_id:
                return dc.competency_area_ids or None
        return None

    def _collect_pending_examples(
        self,
        discipline_id: str,
        allowed_comp_ids: list[str] | None,
    ) -> list[Example]:
        """Collect pending examples for a discipline.

        Args:
            discipline_id: The discipline to query.
            allowed_comp_ids: If set, restrict to these competencies.

        Returns:
            List of pending Example objects.
        """
        competencies = self._storage.get_competencies_for_discipline(discipline_id)
        comp_ids = [c.id for c in competencies]

        if allowed_comp_ids is not None:
            comp_ids = [cid for cid in comp_ids if cid in allowed_comp_ids]

        pending: list[Example] = []
        for cid in comp_ids:
            examples = self._storage.get_examples_for_competency(cid)
            for ex in examples:
                if ex.review_status == ReviewStatus.PENDING:
                    pending.append(ex)
        return pending

    def _build_review_items(
        self,
        discipline_id: str,
        pending: list[Example],
    ) -> list[ReviewItem]:
        """Convert pending examples into ReviewItem entries.

        Args:
            discipline_id: The discipline context.
            pending: Pending examples to convert.

        Returns:
            List of ReviewItem with contributor and competency names.
        """
        contributor_cache: dict[str, str] = {}
        competency_cache: dict[str, str] = {}
        items: list[ReviewItem] = []

        for ex in pending:
            contrib_name = self._resolve_contributor_name(ex.contributor_id, contributor_cache)
            comp_name = self._resolve_competency_name(ex.competency_id, competency_cache)
            items.append(
                ReviewItem(
                    example_id=ex.id,
                    example=ex,
                    contributor_name=contrib_name,
                    submitted_at=ex.created_at,
                    competency_name=comp_name,
                )
            )
        return items

    def _resolve_contributor_name(
        self,
        contributor_id: str,
        cache: dict[str, str],
    ) -> str:
        """Look up a contributor name with caching.

        Args:
            contributor_id: The contributor ID.
            cache: Name cache to avoid repeated lookups.

        Returns:
            Contributor display name or the ID as fallback.
        """
        if contributor_id not in cache:
            contrib = self._storage.get_contributor(contributor_id)
            cache[contributor_id] = contrib.name if contrib else contributor_id
        return cache[contributor_id]

    def _resolve_competency_name(
        self,
        competency_id: str,
        cache: dict[str, str],
    ) -> str:
        """Look up a competency name with caching.

        Args:
            competency_id: The competency ID.
            cache: Name cache to avoid repeated lookups.

        Returns:
            Competency display name or the ID as fallback.
        """
        if competency_id not in cache:
            comp = self._storage.get_competency(competency_id)
            cache[competency_id] = comp.name if comp else competency_id
        return cache[competency_id]

    def _get_example_or_raise(self, example_id: str) -> Example:
        """Fetch an example or raise ContributorError.

        Args:
            example_id: The example ID.

        Returns:
            The Example object.

        Raises:
            ContributorError: If example not found.
        """
        example = self._storage.get_example(example_id)
        if example is None:
            raise ContributorError(f"Example not found: {example_id}")
        return example

    def _require_reviewer_permission(
        self,
        reviewer_id: str,
        discipline_id: str,
    ) -> None:
        """Validate that the reviewer has lead or admin role.

        Args:
            reviewer_id: The reviewer's contributor ID.
            discipline_id: The discipline scope.

        Raises:
            ContributorError: If reviewer is not lead or admin.
        """
        members = self._storage.get_discipline_contributors(discipline_id)
        for dc in members:
            if dc.contributor_id == reviewer_id:
                if dc.role in (ContributorRole.LEAD, ContributorRole.ADMIN):
                    return
                raise ContributorError(
                    f"Contributor {reviewer_id} does not have permission "
                    f"to review in {discipline_id}"
                )
        raise ContributorError(
            f"Contributor {reviewer_id} does not have permission " f"to review in {discipline_id}"
        )

    def _apply_review(
        self,
        example: Example,
        reviewer_id: str,
        decision: ReviewDecision,
        notes: str,
    ) -> Example:
        """Apply a review decision to an example.

        Args:
            example: The example to review.
            reviewer_id: The reviewer's contributor ID.
            decision: The review decision.
            notes: Optional notes to append.

        Returns:
            Updated Example persisted to storage.
        """
        now = datetime.now()
        example.review_status = _DECISION_TO_STATUS[decision]
        example.reviewed_by = reviewer_id
        example.reviewed_at = now
        if notes:
            separator = " | " if example.context else ""
            example.context = f"{example.context}{separator}review: {notes}"
        self._storage.update_example(example)
        return example

    def _stamp_review(
        self,
        example: Example,
        reviewer_id: str,
        status: ReviewStatus,
        now: datetime,
    ) -> Example:
        """Stamp review metadata on an example and persist.

        Args:
            example: The example to stamp.
            reviewer_id: The reviewer's contributor ID.
            status: The review status to set.
            now: The timestamp for the review.

        Returns:
            Updated Example persisted to storage.
        """
        example.review_status = status
        example.reviewed_by = reviewer_id
        example.reviewed_at = now
        self._storage.update_example(example)
        return example

    def _get_contributor_examples(
        self,
        contributor_id: str,
        discipline_id: str,
    ) -> list[Example]:
        """Get all examples by a contributor in a discipline.

        Args:
            contributor_id: The contributor ID.
            discipline_id: The discipline scope.

        Returns:
            List of examples by the contributor.
        """
        competencies = self._storage.get_competencies_for_discipline(discipline_id)
        examples: list[Example] = []
        for comp in competencies:
            comp_examples = self._storage.get_examples_for_competency(comp.id)
            for ex in comp_examples:
                if ex.contributor_id == contributor_id:
                    examples.append(ex)
        return examples

    @staticmethod
    def _compute_stats(
        contributor_id: str,
        contributor_name: str,
        examples: list[Example],
    ) -> ContributionStats:
        """Compute contribution stats from a list of examples.

        Args:
            contributor_id: The contributor ID.
            contributor_name: Display name.
            examples: Examples to aggregate.

        Returns:
            ContributionStats with counts by status and competency.
        """
        approved = sum(1 for e in examples if e.review_status == ReviewStatus.APPROVED)
        rejected = sum(1 for e in examples if e.review_status == ReviewStatus.REJECTED)
        pending = sum(1 for e in examples if e.review_status == ReviewStatus.PENDING)
        by_competency: dict[str, int] = {}
        for ex in examples:
            by_competency[ex.competency_id] = by_competency.get(ex.competency_id, 0) + 1
        return ContributionStats(
            contributor_id=contributor_id,
            contributor_name=contributor_name,
            total_examples=len(examples),
            approved=approved,
            rejected=rejected,
            pending=pending,
            by_competency=by_competency,
        )
