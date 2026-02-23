"""Tests for multi-contributor workflow management."""

from __future__ import annotations

from datetime import datetime

import pytest

from forge.src.contributors import (
    ContributionStats,
    ContributorError,
    ContributorManager,
    ReviewDecision,
    ReviewItem,
    ReviewQueue,
)
from forge.src.models import (
    Competency,
    Contributor,
    ContributorRole,
    Example,
    ReviewStatus,
)
from forge.src.storage import ForgeStorage

# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def manager(populated_store: ForgeStorage) -> ContributorManager:
    """ContributorManager backed by a populated store."""
    return ContributorManager(populated_store)


@pytest.fixture
def lead_contributor(populated_store: ForgeStorage) -> Contributor:
    """A lead contributor added to the store."""
    lead = Contributor(id="contrib_lead001", name="Bob Lead", email="bob@example.com")
    populated_store.create_contributor(lead)
    return lead


@pytest.fixture
def second_competency(populated_store: ForgeStorage) -> Competency:
    """A second competency in the same discipline."""
    comp = Competency(
        id="comp_test002",
        name="Parts Interpretation",
        description="Read and interpret parts diagrams",
        discipline_id="disc_test001",
        coverage_target=20,
    )
    populated_store.create_competency(comp)
    return comp


@pytest.fixture
def manager_with_lead(
    manager: ContributorManager,
    lead_contributor: Contributor,
) -> ContributorManager:
    """Manager with a lead contributor assigned to the discipline."""
    manager.assign_to_discipline(
        contributor_id=lead_contributor.id,
        discipline_id="disc_test001",
        role=ContributorRole.LEAD,
    )
    return manager


@pytest.fixture
def pending_examples(
    populated_store: ForgeStorage,
    second_competency: Competency,
) -> list[Example]:
    """Create multiple pending examples across competencies."""
    examples = []
    for i in range(3):
        ex = Example(
            id=f"ex_pending_{i:03d}",
            question=f"Pending question {i} about fault isolation?",
            ideal_answer=f"Pending answer {i} with detailed steps.",
            competency_id="comp_test001",
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            review_status=ReviewStatus.PENDING,
        )
        populated_store.create_example(ex)
        examples.append(ex)
    # Add one for the second competency
    ex_comp2 = Example(
        id="ex_pending_comp2",
        question="How do you read a parts diagram for this system?",
        ideal_answer="Start with the index, locate the assembly number.",
        competency_id=second_competency.id,
        contributor_id="contrib_test001",
        discipline_id="disc_test001",
        review_status=ReviewStatus.PENDING,
    )
    populated_store.create_example(ex_comp2)
    examples.append(ex_comp2)
    return examples


# ---------------------------------------------------------------
# TestReviewItem
# ---------------------------------------------------------------


class TestReviewItem:
    """Tests for ReviewItem dataclass."""

    def test_construction(self) -> None:
        """ReviewItem stores all fields."""
        item = ReviewItem(
            example_id="ex_001",
            example=Example(
                id="ex_001",
                question="Q?",
                ideal_answer="A.",
                competency_id="comp_001",
                contributor_id="contrib_001",
                discipline_id="disc_001",
            ),
            contributor_name="Alice",
            submitted_at=datetime(2026, 1, 1),
            competency_name="Fault Isolation",
        )
        assert item.example_id == "ex_001"
        assert item.contributor_name == "Alice"
        assert item.competency_name == "Fault Isolation"

    def test_fields_match_example(self) -> None:
        """ReviewItem example_id matches the contained example."""
        ex = Example(
            id="ex_abc",
            question="Q?",
            ideal_answer="A.",
            competency_id="comp_x",
            contributor_id="contrib_x",
            discipline_id="disc_x",
        )
        item = ReviewItem(
            example_id=ex.id,
            example=ex,
            contributor_name="Test",
            submitted_at=ex.created_at,
            competency_name="Test Comp",
        )
        assert item.example_id == item.example.id


# ---------------------------------------------------------------
# TestReviewQueue
# ---------------------------------------------------------------


class TestReviewQueue:
    """Tests for ReviewQueue dataclass."""

    def test_construction(self) -> None:
        """ReviewQueue stores discipline_id and items."""
        queue = ReviewQueue(
            discipline_id="disc_001",
            items=[],
            total_count=0,
        )
        assert queue.discipline_id == "disc_001"
        assert queue.items == []

    def test_pending_count(self) -> None:
        """pending_count returns the number of items."""
        ex = Example(
            id="ex_1",
            question="Q?",
            ideal_answer="A.",
            competency_id="c",
            contributor_id="x",
            discipline_id="d",
        )
        item = ReviewItem(
            example_id="ex_1",
            example=ex,
            contributor_name="A",
            submitted_at=datetime.now(),
            competency_name="C",
        )
        queue = ReviewQueue(discipline_id="d", items=[item], total_count=1)
        assert queue.pending_count == 1

    def test_empty_queue(self) -> None:
        """Empty queue has pending_count of 0."""
        queue = ReviewQueue(discipline_id="d", items=[], total_count=0)
        assert queue.pending_count == 0


# ---------------------------------------------------------------
# TestContributionStats
# ---------------------------------------------------------------


class TestContributionStats:
    """Tests for ContributionStats dataclass."""

    def test_construction(self) -> None:
        """ContributionStats stores counts correctly."""
        stats = ContributionStats(
            contributor_id="contrib_001",
            contributor_name="Alice",
            total_examples=10,
            approved=5,
            rejected=2,
            pending=3,
            by_competency={"comp_a": 6, "comp_b": 4},
        )
        assert stats.total_examples == 10
        assert stats.approved == 5
        assert stats.rejected == 2

    def test_by_competency(self) -> None:
        """by_competency breakdown sums correctly."""
        stats = ContributionStats(
            contributor_id="contrib_001",
            contributor_name="Alice",
            total_examples=10,
            approved=5,
            rejected=2,
            pending=3,
            by_competency={"comp_a": 6, "comp_b": 4},
        )
        assert sum(stats.by_competency.values()) == stats.total_examples


# ---------------------------------------------------------------
# TestAssignToDiscipline
# ---------------------------------------------------------------


class TestAssignToDiscipline:
    """Tests for ContributorManager.assign_to_discipline."""

    def test_basic_assignment(self, manager: ContributorManager) -> None:
        """Assign a contributor with default role."""
        dc = manager.assign_to_discipline(
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            role=ContributorRole.CONTRIBUTOR,
        )
        assert dc.contributor_id == "contrib_test001"
        assert dc.discipline_id == "disc_test001"
        assert dc.role == ContributorRole.CONTRIBUTOR

    def test_with_lead_role(
        self,
        manager: ContributorManager,
        lead_contributor: Contributor,
    ) -> None:
        """Assign a contributor as lead."""
        dc = manager.assign_to_discipline(
            contributor_id=lead_contributor.id,
            discipline_id="disc_test001",
            role=ContributorRole.LEAD,
        )
        assert dc.role == ContributorRole.LEAD

    def test_with_competency_areas(
        self,
        manager: ContributorManager,
    ) -> None:
        """Assign with specific competency area ownership."""
        dc = manager.assign_to_discipline(
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            role=ContributorRole.CONTRIBUTOR,
            competency_area_ids=["comp_test001"],
        )
        assert dc.competency_area_ids == ["comp_test001"]

    def test_duplicate_raises(self, manager: ContributorManager) -> None:
        """Assigning same contributor twice raises ContributorError."""
        manager.assign_to_discipline(
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            role=ContributorRole.CONTRIBUTOR,
        )
        with pytest.raises(ContributorError, match="already assigned"):
            manager.assign_to_discipline(
                contributor_id="contrib_test001",
                discipline_id="disc_test001",
                role=ContributorRole.CONTRIBUTOR,
            )


# ---------------------------------------------------------------
# TestUpdateRole
# ---------------------------------------------------------------


class TestUpdateRole:
    """Tests for ContributorManager.update_role."""

    def test_role_change(self, manager: ContributorManager) -> None:
        """Update a contributor's role."""
        manager.assign_to_discipline(
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            role=ContributorRole.CONTRIBUTOR,
        )
        dc = manager.update_role(
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            new_role=ContributorRole.LEAD,
        )
        assert dc.role == ContributorRole.LEAD

    def test_non_member_raises(self, manager: ContributorManager) -> None:
        """Updating role for non-member raises ContributorError."""
        with pytest.raises(ContributorError, match="not a member"):
            manager.update_role(
                contributor_id="contrib_test001",
                discipline_id="disc_test001",
                new_role=ContributorRole.LEAD,
            )

    def test_validates_role_type(self, manager: ContributorManager) -> None:
        """Role must be a valid ContributorRole."""
        manager.assign_to_discipline(
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            role=ContributorRole.CONTRIBUTOR,
        )
        dc = manager.update_role(
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            new_role=ContributorRole.ADMIN,
        )
        assert dc.role == ContributorRole.ADMIN


# ---------------------------------------------------------------
# TestAssignCompetencyAreas
# ---------------------------------------------------------------


class TestAssignCompetencyAreas:
    """Tests for ContributorManager.assign_competency_areas."""

    def test_set_areas(
        self,
        manager: ContributorManager,
        second_competency: Competency,
    ) -> None:
        """Set specific competency areas."""
        manager.assign_to_discipline(
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            role=ContributorRole.CONTRIBUTOR,
        )
        dc = manager.assign_competency_areas(
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            competency_ids=["comp_test001", second_competency.id],
        )
        assert set(dc.competency_area_ids) == {"comp_test001", second_competency.id}

    def test_clear_areas_means_all(self, manager: ContributorManager) -> None:
        """Empty competency_ids means access to all competencies."""
        manager.assign_to_discipline(
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            role=ContributorRole.CONTRIBUTOR,
            competency_area_ids=["comp_test001"],
        )
        dc = manager.assign_competency_areas(
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            competency_ids=[],
        )
        assert dc.competency_area_ids == []

    def test_non_member_raises(self, manager: ContributorManager) -> None:
        """Assigning areas for non-member raises ContributorError."""
        with pytest.raises(ContributorError, match="not a member"):
            manager.assign_competency_areas(
                contributor_id="contrib_test001",
                discipline_id="disc_test001",
                competency_ids=["comp_test001"],
            )


# ---------------------------------------------------------------
# TestGetReviewQueue
# ---------------------------------------------------------------


class TestGetReviewQueue:
    """Tests for ContributorManager.get_review_queue."""

    def test_pending_examples_returned(
        self,
        manager_with_lead: ContributorManager,
        pending_examples: list[Example],
    ) -> None:
        """Review queue returns pending examples for the discipline."""
        queue = manager_with_lead.get_review_queue("disc_test001")
        # 4 from pending_examples + 1 from populated_store (ex_test001 is PENDING)
        assert queue.pending_count >= 4

    def test_filtered_by_reviewer_competencies(
        self,
        manager_with_lead: ContributorManager,
        pending_examples: list[Example],
    ) -> None:
        """Queue filtered when reviewer has specific competency areas."""
        # Assign lead to only comp_test001
        manager_with_lead.assign_competency_areas(
            contributor_id="contrib_lead001",
            discipline_id="disc_test001",
            competency_ids=["comp_test001"],
        )
        queue = manager_with_lead.get_review_queue("disc_test001", reviewer_id="contrib_lead001")
        # Only comp_test001 examples, not comp_test002 ones
        for item in queue.items:
            assert item.example.competency_id == "comp_test001"

    def test_empty_when_none_pending(
        self,
        manager_with_lead: ContributorManager,
        populated_store: ForgeStorage,
    ) -> None:
        """Queue is empty when all examples are approved."""
        # Approve the one existing example
        ex = populated_store.get_example("ex_test001")
        assert ex is not None
        ex.review_status = ReviewStatus.APPROVED
        ex.reviewed_by = "contrib_lead001"
        ex.reviewed_at = datetime.now()
        populated_store.update_example(ex)

        queue = manager_with_lead.get_review_queue("disc_test001")
        assert queue.pending_count == 0

    def test_excludes_approved(
        self,
        manager_with_lead: ContributorManager,
        pending_examples: list[Example],
        populated_store: ForgeStorage,
    ) -> None:
        """Queue excludes approved examples."""
        # Approve one pending example
        ex = populated_store.get_example("ex_pending_000")
        assert ex is not None
        ex.review_status = ReviewStatus.APPROVED
        ex.reviewed_by = "contrib_lead001"
        ex.reviewed_at = datetime.now()
        populated_store.update_example(ex)

        queue = manager_with_lead.get_review_queue("disc_test001")
        example_ids = [item.example_id for item in queue.items]
        assert "ex_pending_000" not in example_ids

    def test_includes_reviewer_info(
        self,
        manager_with_lead: ContributorManager,
        pending_examples: list[Example],
    ) -> None:
        """Review items include contributor name and competency name."""
        queue = manager_with_lead.get_review_queue("disc_test001")
        assert len(queue.items) > 0
        first_item = queue.items[0]
        assert first_item.contributor_name != ""
        assert first_item.competency_name != ""


# ---------------------------------------------------------------
# TestSubmitReview
# ---------------------------------------------------------------


class TestSubmitReview:
    """Tests for ContributorManager.submit_review."""

    def test_approve_sets_status(
        self,
        manager_with_lead: ContributorManager,
    ) -> None:
        """Approving sets review_status to APPROVED."""
        result = manager_with_lead.submit_review(
            reviewer_id="contrib_lead001",
            example_id="ex_test001",
            decision=ReviewDecision.APPROVE,
        )
        assert result.review_status == ReviewStatus.APPROVED
        assert result.reviewed_by == "contrib_lead001"
        assert result.reviewed_at is not None

    def test_reject_sets_status(
        self,
        manager_with_lead: ContributorManager,
    ) -> None:
        """Rejecting sets review_status to REJECTED."""
        result = manager_with_lead.submit_review(
            reviewer_id="contrib_lead001",
            example_id="ex_test001",
            decision=ReviewDecision.REJECT,
        )
        assert result.review_status == ReviewStatus.REJECTED

    def test_needs_revision_sets_status(
        self,
        manager_with_lead: ContributorManager,
    ) -> None:
        """Needs revision sets review_status to NEEDS_REVISION."""
        result = manager_with_lead.submit_review(
            reviewer_id="contrib_lead001",
            example_id="ex_test001",
            decision=ReviewDecision.NEEDS_REVISION,
        )
        assert result.review_status == ReviewStatus.NEEDS_REVISION

    def test_non_lead_raises(
        self,
        manager: ContributorManager,
    ) -> None:
        """Non-lead/admin reviewer raises ContributorError."""
        manager.assign_to_discipline(
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            role=ContributorRole.CONTRIBUTOR,
        )
        with pytest.raises(ContributorError, match="permission"):
            manager.submit_review(
                reviewer_id="contrib_test001",
                example_id="ex_test001",
                decision=ReviewDecision.APPROVE,
            )

    def test_notes_appended_to_context(
        self,
        manager_with_lead: ContributorManager,
        populated_store: ForgeStorage,
    ) -> None:
        """Review notes are appended to example context."""
        manager_with_lead.submit_review(
            reviewer_id="contrib_lead001",
            example_id="ex_test001",
            decision=ReviewDecision.NEEDS_REVISION,
            notes="Please add more detail to the answer.",
        )
        updated = populated_store.get_example("ex_test001")
        assert updated is not None
        assert "Please add more detail to the answer." in updated.context


# ---------------------------------------------------------------
# TestCheckOwnership
# ---------------------------------------------------------------


class TestCheckOwnership:
    """Tests for ContributorManager.check_ownership."""

    def test_empty_areas_owns_all(self, manager: ContributorManager) -> None:
        """Contributor with empty areas owns all competencies."""
        manager.assign_to_discipline(
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            role=ContributorRole.CONTRIBUTOR,
            competency_area_ids=[],
        )
        assert manager.check_ownership("contrib_test001", "disc_test001", "comp_test001")
        assert manager.check_ownership("contrib_test001", "disc_test001", "comp_any_other")

    def test_owns_specific_area(self, manager: ContributorManager) -> None:
        """Contributor owns their assigned competency."""
        manager.assign_to_discipline(
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            role=ContributorRole.CONTRIBUTOR,
            competency_area_ids=["comp_test001"],
        )
        assert manager.check_ownership("contrib_test001", "disc_test001", "comp_test001")

    def test_does_not_own_other_area(self, manager: ContributorManager) -> None:
        """Contributor does not own unassigned competencies."""
        manager.assign_to_discipline(
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            role=ContributorRole.CONTRIBUTOR,
            competency_area_ids=["comp_test001"],
        )
        assert not manager.check_ownership("contrib_test001", "disc_test001", "comp_other")


# ---------------------------------------------------------------
# TestGetContributionStats
# ---------------------------------------------------------------


class TestGetContributionStats:
    """Tests for ContributorManager.get_contribution_stats."""

    def test_counts_correct(
        self,
        manager: ContributorManager,
        populated_store: ForgeStorage,
    ) -> None:
        """Stats count examples by review status."""
        manager.assign_to_discipline(
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            role=ContributorRole.CONTRIBUTOR,
        )
        # The populated_store has 1 pending example (ex_test001)
        stats = manager.get_contribution_stats("contrib_test001", "disc_test001")
        assert stats.total_examples == 1
        assert stats.pending == 1
        assert stats.approved == 0
        assert stats.rejected == 0

    def test_by_competency_breakdown(
        self,
        manager: ContributorManager,
        populated_store: ForgeStorage,
        second_competency: Competency,
    ) -> None:
        """by_competency shows counts per competency."""
        manager.assign_to_discipline(
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            role=ContributorRole.CONTRIBUTOR,
        )
        # Add example in second competency
        populated_store.create_example(
            Example(
                id="ex_comp2_stat",
                question="Parts diagram interpretation question?",
                ideal_answer="Detailed parts answer here with steps.",
                competency_id=second_competency.id,
                contributor_id="contrib_test001",
                discipline_id="disc_test001",
            )
        )
        stats = manager.get_contribution_stats("contrib_test001", "disc_test001")
        assert stats.total_examples == 2
        assert stats.by_competency.get("comp_test001", 0) == 1
        assert stats.by_competency.get(second_competency.id, 0) == 1

    def test_empty_discipline(
        self,
        manager: ContributorManager,
        lead_contributor: Contributor,
    ) -> None:
        """Stats for contributor with no examples in discipline."""
        manager.assign_to_discipline(
            contributor_id=lead_contributor.id,
            discipline_id="disc_test001",
            role=ContributorRole.LEAD,
        )
        stats = manager.get_contribution_stats(lead_contributor.id, "disc_test001")
        assert stats.total_examples == 0
        assert stats.approved == 0
        assert stats.pending == 0


# ---------------------------------------------------------------
# TestGetDisciplineStats
# ---------------------------------------------------------------


class TestGetDisciplineStats:
    """Tests for ContributorManager.get_discipline_stats."""

    def test_returns_per_contributor(
        self,
        manager: ContributorManager,
        lead_contributor: Contributor,
    ) -> None:
        """Discipline stats return one entry per contributor."""
        manager.assign_to_discipline(
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            role=ContributorRole.CONTRIBUTOR,
        )
        manager.assign_to_discipline(
            contributor_id=lead_contributor.id,
            discipline_id="disc_test001",
            role=ContributorRole.LEAD,
        )
        stats_list = manager.get_discipline_stats("disc_test001")
        assert len(stats_list) == 2
        ids = {s.contributor_id for s in stats_list}
        assert "contrib_test001" in ids
        assert lead_contributor.id in ids


# ---------------------------------------------------------------
# TestConflictResolution
# ---------------------------------------------------------------


class TestConflictResolution:
    """Tests for ContributorManager.resolve_conflict."""

    def test_keep_reject_works(
        self,
        manager_with_lead: ContributorManager,
        populated_store: ForgeStorage,
    ) -> None:
        """Keeping one example and rejecting the other works."""
        # Create two conflicting examples
        ex_a = Example(
            id="ex_conflict_a",
            question="Conflicting question about hydraulics?",
            ideal_answer="Answer A approach.",
            competency_id="comp_test001",
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
        )
        ex_b = Example(
            id="ex_conflict_b",
            question="Conflicting question about hydraulics?",
            ideal_answer="Answer B approach.",
            competency_id="comp_test001",
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
        )
        populated_store.create_example(ex_a)
        populated_store.create_example(ex_b)

        kept, rejected = manager_with_lead.resolve_conflict(
            reviewer_id="contrib_lead001",
            example_id_keep="ex_conflict_a",
            example_id_reject="ex_conflict_b",
        )
        assert kept.review_status == ReviewStatus.APPROVED
        assert rejected.review_status == ReviewStatus.REJECTED

    def test_non_lead_raises(
        self,
        manager: ContributorManager,
        populated_store: ForgeStorage,
    ) -> None:
        """Non-lead/admin cannot resolve conflicts."""
        manager.assign_to_discipline(
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            role=ContributorRole.CONTRIBUTOR,
        )
        ex_a = Example(
            id="ex_conf_a2",
            question="Conflict question about system checks?",
            ideal_answer="Answer A for system checks.",
            competency_id="comp_test001",
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
        )
        ex_b = Example(
            id="ex_conf_b2",
            question="Conflict question about system checks?",
            ideal_answer="Answer B for system checks.",
            competency_id="comp_test001",
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
        )
        populated_store.create_example(ex_a)
        populated_store.create_example(ex_b)

        with pytest.raises(ContributorError, match="permission"):
            manager.resolve_conflict(
                reviewer_id="contrib_test001",
                example_id_keep="ex_conf_a2",
                example_id_reject="ex_conf_b2",
            )

    def test_both_get_reviewed_by(
        self,
        manager_with_lead: ContributorManager,
        populated_store: ForgeStorage,
    ) -> None:
        """Both examples get reviewed_by and reviewed_at set."""
        ex_a = Example(
            id="ex_conf_c",
            question="Another conflict about torque specs?",
            ideal_answer="Torque answer C.",
            competency_id="comp_test001",
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
        )
        ex_b = Example(
            id="ex_conf_d",
            question="Another conflict about torque specs?",
            ideal_answer="Torque answer D.",
            competency_id="comp_test001",
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
        )
        populated_store.create_example(ex_a)
        populated_store.create_example(ex_b)

        kept, rejected = manager_with_lead.resolve_conflict(
            reviewer_id="contrib_lead001",
            example_id_keep="ex_conf_c",
            example_id_reject="ex_conf_d",
        )
        assert kept.reviewed_by == "contrib_lead001"
        assert kept.reviewed_at is not None
        assert rejected.reviewed_by == "contrib_lead001"
        assert rejected.reviewed_at is not None
