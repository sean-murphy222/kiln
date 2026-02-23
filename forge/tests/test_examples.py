"""Tests for example elicitation engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.src.examples import (
    ElicitationError,
    ElicitationSession,
    ExampleDraft,
    ExampleElicitor,
    ExampleMetadata,
    ReasoningPattern,
    SessionStatus,
)
from forge.src.models import Competency, Example
from forge.src.storage import ForgeStorage

# --- Fixtures ---


@pytest.fixture
def elicitor(populated_store: ForgeStorage) -> ExampleElicitor:
    """ExampleElicitor with a populated store (no disk persistence)."""
    return ExampleElicitor(populated_store)


@pytest.fixture
def elicitor_with_disk(populated_store: ForgeStorage, temp_dir: Path) -> ExampleElicitor:
    """ExampleElicitor with disk-backed session persistence."""
    return ExampleElicitor(populated_store, sessions_dir=temp_dir / "sessions")


@pytest.fixture
def active_session(elicitor: ExampleElicitor) -> ElicitationSession:
    """An active elicitation session."""
    return elicitor.start_session("disc_test001", "contrib_test001")


@pytest.fixture
def multi_comp_store(populated_store: ForgeStorage) -> ForgeStorage:
    """Store with multiple competencies at varying coverage levels."""
    populated_store.create_competency(
        Competency(
            id="comp_safety",
            name="Safety Protocols",
            description="Safety procedures",
            discipline_id="disc_test001",
            coverage_target=10,
        )
    )
    populated_store.create_competency(
        Competency(
            id="comp_tools",
            name="Tool Identification",
            description="Identifying tools",
            discipline_id="disc_test001",
            coverage_target=15,
        )
    )
    # Add examples to comp_safety (5 of 10 = 50%)
    for i in range(5):
        populated_store.create_example(
            Example(
                id=f"ex_safe_{i:03d}",
                question=f"Safety question {i} about procedures?",
                ideal_answer=f"Safety answer {i} with detailed response.",
                competency_id="comp_safety",
                contributor_id="contrib_test001",
                discipline_id="disc_test001",
            )
        )
    # comp_tools: 0 examples (empty)
    # comp_test001: 1 example (from populated_store fixture, 4% of 25)
    return populated_store


@pytest.fixture
def multi_elicitor(multi_comp_store: ForgeStorage) -> ExampleElicitor:
    """Elicitor with multi-competency store."""
    return ExampleElicitor(multi_comp_store)


# --- ExampleMetadata Tests ---


class TestExampleMetadata:
    """Tests for ExampleMetadata dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        meta = ExampleMetadata()
        assert meta.reasoning_pattern is None
        assert meta.equipment == ""
        assert meta.difficulty == 3
        assert meta.tags == []

    def test_full_construction(self) -> None:
        """Test full construction."""
        meta = ExampleMetadata(
            reasoning_pattern=ReasoningPattern.DIAGNOSTIC,
            equipment="M998 HMMWV",
            procedure_ref="WP 0045 00",
            difficulty=4,
            tags=["engine", "troubleshooting"],
        )
        assert meta.reasoning_pattern == ReasoningPattern.DIAGNOSTIC
        assert meta.equipment == "M998 HMMWV"

    def test_to_dict(self) -> None:
        """Test serialization."""
        meta = ExampleMetadata(
            reasoning_pattern=ReasoningPattern.PROCEDURAL,
            tags=["tag1"],
        )
        d = meta.to_dict()
        assert d["reasoning_pattern"] == "procedural"
        assert d["tags"] == ["tag1"]

    def test_from_dict(self) -> None:
        """Test deserialization round-trip."""
        meta = ExampleMetadata(
            reasoning_pattern=ReasoningPattern.SAFETY,
            equipment="test",
            difficulty=5,
        )
        restored = ExampleMetadata.from_dict(meta.to_dict())
        assert restored.reasoning_pattern == ReasoningPattern.SAFETY
        assert restored.equipment == "test"
        assert restored.difficulty == 5

    def test_from_dict_empty(self) -> None:
        """Test deserialization from empty dict."""
        meta = ExampleMetadata.from_dict({})
        assert meta.reasoning_pattern is None
        assert meta.difficulty == 3


# --- ExampleDraft Tests ---


class TestExampleDraft:
    """Tests for ExampleDraft dataclass."""

    def test_construction(self) -> None:
        """Test draft construction."""
        draft = ExampleDraft(
            id="draft_001",
            question="How do you test?",
            ideal_answer="You test by...",
            competency_id="comp_001",
        )
        assert draft.id == "draft_001"
        assert draft.variants == []

    def test_generate_id(self) -> None:
        """Test ID generation."""
        id1 = ExampleDraft.generate_id()
        id2 = ExampleDraft.generate_id()
        assert id1.startswith("draft_")
        assert id1 != id2

    def test_serialization_round_trip(self) -> None:
        """Test full serialization round-trip."""
        draft = ExampleDraft(
            id="draft_rt",
            question="Test question here?",
            ideal_answer="Test answer here.",
            variants=["Variant 1?", "Variant 2?"],
            context="Some context",
            competency_id="comp_001",
            metadata=ExampleMetadata(
                reasoning_pattern=ReasoningPattern.ANALYTICAL,
                equipment="M1A1",
                difficulty=4,
            ),
        )
        d = draft.to_dict()
        restored = ExampleDraft.from_dict(d)
        assert restored.question == draft.question
        assert restored.variants == draft.variants
        assert restored.metadata.reasoning_pattern == ReasoningPattern.ANALYTICAL


# --- ElicitationSession Tests ---


class TestElicitationSession:
    """Tests for ElicitationSession dataclass."""

    def test_construction(self) -> None:
        """Test session construction."""
        session = ElicitationSession(
            id="esess_001",
            discipline_id="disc_001",
            contributor_id="contrib_001",
        )
        assert session.status == SessionStatus.ACTIVE
        assert session.drafts == []
        assert session.finalized_count == 0

    def test_get_draft(self) -> None:
        """Test getting a draft by ID."""
        session = ElicitationSession(
            id="esess_002",
            discipline_id="disc_001",
            contributor_id="contrib_001",
            drafts=[
                ExampleDraft(id="d1", question="Q1"),
                ExampleDraft(id="d2", question="Q2"),
            ],
        )
        assert session.get_draft("d1") is not None
        assert session.get_draft("d1").question == "Q1"
        assert session.get_draft("nonexistent") is None

    def test_serialization_round_trip(self) -> None:
        """Test session serialization."""
        session = ElicitationSession(
            id="esess_rt",
            discipline_id="disc_001",
            contributor_id="contrib_001",
            focus_competency_id="comp_001",
            drafts=[ExampleDraft(id="d1")],
            finalized_count=3,
            status=SessionStatus.PAUSED,
        )
        d = session.to_dict()
        restored = ElicitationSession.from_dict(d)
        assert restored.id == "esess_rt"
        assert restored.focus_competency_id == "comp_001"
        assert restored.finalized_count == 3
        assert restored.status == SessionStatus.PAUSED
        assert len(restored.drafts) == 1


# --- ExampleElicitor Session Management Tests ---


class TestElicitorSessionManagement:
    """Tests for session lifecycle."""

    def test_start_session(self, elicitor: ExampleElicitor) -> None:
        """Test starting a new session."""
        session = elicitor.start_session("disc_test001", "contrib_test001")
        assert session.discipline_id == "disc_test001"
        assert session.contributor_id == "contrib_test001"
        assert session.status == SessionStatus.ACTIVE

    def test_start_session_nonexistent_discipline(self, elicitor: ExampleElicitor) -> None:
        """Test starting session for missing discipline raises."""
        with pytest.raises(ElicitationError, match="Discipline not found"):
            elicitor.start_session("disc_nonexistent", "contrib_test001")

    def test_get_session(self, elicitor: ExampleElicitor) -> None:
        """Test getting a session by ID."""
        session = elicitor.start_session("disc_test001", "contrib_test001")
        retrieved = elicitor.get_session(session.id)
        assert retrieved.id == session.id

    def test_get_nonexistent_session(self, elicitor: ExampleElicitor) -> None:
        """Test getting a missing session raises."""
        with pytest.raises(ElicitationError, match="Session not found"):
            elicitor.get_session("esess_nonexistent")

    def test_list_sessions(self, elicitor: ExampleElicitor) -> None:
        """Test listing sessions."""
        elicitor.start_session("disc_test001", "contrib_test001")
        elicitor.start_session("disc_test001", "contrib_test001")
        sessions = elicitor.list_sessions()
        assert len(sessions) == 2

    def test_list_sessions_filter_status(self, elicitor: ExampleElicitor) -> None:
        """Test filtering sessions by status."""
        s1 = elicitor.start_session("disc_test001", "contrib_test001")
        elicitor.start_session("disc_test001", "contrib_test001")
        elicitor.pause_session(s1.id)
        active = elicitor.list_sessions(status=SessionStatus.ACTIVE)
        paused = elicitor.list_sessions(status=SessionStatus.PAUSED)
        assert len(active) == 1
        assert len(paused) == 1

    def test_pause_and_resume(
        self, elicitor: ExampleElicitor, active_session: ElicitationSession
    ) -> None:
        """Test pausing and resuming a session."""
        paused = elicitor.pause_session(active_session.id)
        assert paused.status == SessionStatus.PAUSED

        resumed = elicitor.resume_session(active_session.id)
        assert resumed.status == SessionStatus.ACTIVE

    def test_resume_non_paused_raises(
        self, elicitor: ExampleElicitor, active_session: ElicitationSession
    ) -> None:
        """Test resuming a non-paused session raises."""
        with pytest.raises(ElicitationError, match="not paused"):
            elicitor.resume_session(active_session.id)

    def test_complete_session(
        self, elicitor: ExampleElicitor, active_session: ElicitationSession
    ) -> None:
        """Test completing a session."""
        completed = elicitor.complete_session(active_session.id)
        assert completed.status == SessionStatus.COMPLETED

    def test_set_focus_competency(
        self, elicitor: ExampleElicitor, active_session: ElicitationSession
    ) -> None:
        """Test setting focus competency."""
        updated = elicitor.set_focus_competency(active_session.id, "comp_test001")
        assert updated.focus_competency_id == "comp_test001"

    def test_set_focus_nonexistent_competency(
        self, elicitor: ExampleElicitor, active_session: ElicitationSession
    ) -> None:
        """Test setting missing competency raises."""
        with pytest.raises(ElicitationError, match="Competency not found"):
            elicitor.set_focus_competency(active_session.id, "comp_nonexistent")


# --- Draft Management Tests ---


class TestDraftManagement:
    """Tests for draft creation, update, and deletion."""

    def test_create_draft(
        self, elicitor: ExampleElicitor, active_session: ElicitationSession
    ) -> None:
        """Test creating a draft."""
        draft = elicitor.create_draft(
            active_session.id,
            question="How do you replace a filter?",
            ideal_answer="Remove old filter, install new one.",
            competency_id="comp_test001",
        )
        assert draft.question == "How do you replace a filter?"
        assert draft.competency_id == "comp_test001"

    def test_create_draft_uses_session_focus(
        self, elicitor: ExampleElicitor, active_session: ElicitationSession
    ) -> None:
        """Test draft inherits session focus competency."""
        elicitor.set_focus_competency(active_session.id, "comp_test001")
        draft = elicitor.create_draft(
            active_session.id,
            question="Focus question?",
            ideal_answer="Focus answer.",
        )
        assert draft.competency_id == "comp_test001"

    def test_create_draft_with_metadata(
        self, elicitor: ExampleElicitor, active_session: ElicitationSession
    ) -> None:
        """Test creating a draft with metadata."""
        meta = ExampleMetadata(
            reasoning_pattern=ReasoningPattern.DIAGNOSTIC,
            equipment="M998 HMMWV",
            difficulty=4,
        )
        draft = elicitor.create_draft(
            active_session.id,
            question="Test question here?",
            ideal_answer="Test answer here.",
            competency_id="comp_test001",
            metadata=meta,
        )
        assert draft.metadata.reasoning_pattern == ReasoningPattern.DIAGNOSTIC
        assert draft.metadata.equipment == "M998 HMMWV"

    def test_create_draft_on_completed_session_raises(
        self, elicitor: ExampleElicitor, active_session: ElicitationSession
    ) -> None:
        """Test creating draft on completed session raises."""
        elicitor.complete_session(active_session.id)
        with pytest.raises(ElicitationError, match="not active"):
            elicitor.create_draft(active_session.id, question="Q?")

    def test_update_draft(
        self, elicitor: ExampleElicitor, active_session: ElicitationSession
    ) -> None:
        """Test updating a draft."""
        draft = elicitor.create_draft(
            active_session.id,
            question="Original question?",
            competency_id="comp_test001",
        )
        updated = elicitor.update_draft(
            active_session.id,
            draft.id,
            question="Updated question?",
            ideal_answer="New answer.",
        )
        assert updated.question == "Updated question?"
        assert updated.ideal_answer == "New answer."

    def test_update_nonexistent_draft_raises(
        self, elicitor: ExampleElicitor, active_session: ElicitationSession
    ) -> None:
        """Test updating missing draft raises."""
        with pytest.raises(ElicitationError, match="Draft not found"):
            elicitor.update_draft(active_session.id, "draft_nonexistent", question="Q?")

    def test_delete_draft(
        self, elicitor: ExampleElicitor, active_session: ElicitationSession
    ) -> None:
        """Test deleting a draft."""
        draft = elicitor.create_draft(active_session.id, competency_id="comp_test001")
        assert elicitor.delete_draft(active_session.id, draft.id)
        session = elicitor.get_session(active_session.id)
        assert len(session.drafts) == 0

    def test_delete_nonexistent_draft(
        self, elicitor: ExampleElicitor, active_session: ElicitationSession
    ) -> None:
        """Test deleting missing draft returns False."""
        assert not elicitor.delete_draft(active_session.id, "draft_nope")


# --- Validation Tests ---


class TestValidation:
    """Tests for draft validation."""

    def test_valid_draft(self, elicitor: ExampleElicitor) -> None:
        """Test a valid draft passes validation."""
        draft = ExampleDraft(
            id="d_valid",
            question="How do you troubleshoot a hydraulic leak?",
            ideal_answer="Check pressure at each section to isolate the leak.",
            competency_id="comp_test001",
        )
        errors = elicitor.validate_draft(draft)
        assert errors == []

    def test_empty_question(self, elicitor: ExampleElicitor) -> None:
        """Test empty question fails validation."""
        draft = ExampleDraft(
            id="d_empty_q",
            question="",
            ideal_answer="A sufficient answer here.",
            competency_id="comp_test001",
        )
        errors = elicitor.validate_draft(draft)
        assert any("Question is required" in e for e in errors)

    def test_short_question(self, elicitor: ExampleElicitor) -> None:
        """Test short question fails validation."""
        draft = ExampleDraft(
            id="d_short_q",
            question="Short?",
            ideal_answer="A sufficient answer here.",
            competency_id="comp_test001",
        )
        errors = elicitor.validate_draft(draft)
        assert any("at least 10 characters" in e for e in errors)

    def test_empty_answer(self, elicitor: ExampleElicitor) -> None:
        """Test empty answer fails validation."""
        draft = ExampleDraft(
            id="d_empty_a",
            question="A valid question here?",
            ideal_answer="",
            competency_id="comp_test001",
        )
        errors = elicitor.validate_draft(draft)
        assert any("Ideal answer is required" in e for e in errors)

    def test_missing_competency(self, elicitor: ExampleElicitor) -> None:
        """Test missing competency fails validation."""
        draft = ExampleDraft(
            id="d_no_comp",
            question="A valid question here?",
            ideal_answer="A valid answer here.",
            competency_id="",
        )
        errors = elicitor.validate_draft(draft)
        assert any("Competency ID is required" in e for e in errors)

    def test_invalid_difficulty(self, elicitor: ExampleElicitor) -> None:
        """Test invalid difficulty fails validation."""
        draft = ExampleDraft(
            id="d_bad_diff",
            question="A valid question here?",
            ideal_answer="A valid answer here.",
            competency_id="comp_001",
            metadata=ExampleMetadata(difficulty=0),
        )
        errors = elicitor.validate_draft(draft)
        assert any("Difficulty" in e for e in errors)


# --- Finalization Tests ---


class TestFinalization:
    """Tests for draft finalization into Examples."""

    def test_finalize_creates_example(
        self, elicitor: ExampleElicitor, active_session: ElicitationSession
    ) -> None:
        """Test finalization creates a real Example."""
        draft = elicitor.create_draft(
            active_session.id,
            question="How do you replace an air filter?",
            ideal_answer="Open hood, release latches, remove old filter, install new.",
            competency_id="comp_test001",
        )
        example = elicitor.finalize_draft(active_session.id, draft.id)
        assert isinstance(example, Example)
        assert example.question == "How do you replace an air filter?"
        assert example.discipline_id == "disc_test001"
        assert example.contributor_id == "contrib_test001"

    def test_finalize_removes_draft(
        self, elicitor: ExampleElicitor, active_session: ElicitationSession
    ) -> None:
        """Test finalization removes the draft from session."""
        draft = elicitor.create_draft(
            active_session.id,
            question="How do you check tire pressure?",
            ideal_answer="Use a calibrated pressure gauge at the valve stem.",
            competency_id="comp_test001",
        )
        elicitor.finalize_draft(active_session.id, draft.id)
        session = elicitor.get_session(active_session.id)
        assert len(session.drafts) == 0
        assert session.finalized_count == 1

    def test_finalize_increments_counter(
        self, elicitor: ExampleElicitor, active_session: ElicitationSession
    ) -> None:
        """Test finalization increments finalized_count."""
        for i in range(3):
            draft = elicitor.create_draft(
                active_session.id,
                question=f"Question number {i} about maintenance?",
                ideal_answer=f"Answer number {i} with full details.",
                competency_id="comp_test001",
            )
            elicitor.finalize_draft(active_session.id, draft.id)
        session = elicitor.get_session(active_session.id)
        assert session.finalized_count == 3

    def test_finalize_invalid_draft_raises(
        self, elicitor: ExampleElicitor, active_session: ElicitationSession
    ) -> None:
        """Test finalizing an invalid draft raises."""
        draft = elicitor.create_draft(
            active_session.id,
            question="",
            competency_id="comp_test001",
        )
        with pytest.raises(ElicitationError, match="validation failed"):
            elicitor.finalize_draft(active_session.id, draft.id)

    def test_finalize_builds_context(
        self, elicitor: ExampleElicitor, active_session: ElicitationSession
    ) -> None:
        """Test finalization builds context from metadata."""
        meta = ExampleMetadata(
            reasoning_pattern=ReasoningPattern.PROCEDURAL,
            equipment="M998 HMMWV",
            procedure_ref="WP 0045 00",
        )
        draft = elicitor.create_draft(
            active_session.id,
            question="How do you perform the engine oil change?",
            ideal_answer="Drain oil, replace filter, refill to specified level.",
            competency_id="comp_test001",
            metadata=meta,
        )
        example = elicitor.finalize_draft(active_session.id, draft.id)
        assert "reasoning_pattern: procedural" in example.context
        assert "equipment: M998 HMMWV" in example.context
        assert "procedure_ref: WP 0045 00" in example.context


# --- Competency Suggestion Tests ---


class TestCompetencySuggestions:
    """Tests for competency suggestions."""

    def test_suggestions_generated(self, multi_elicitor: ExampleElicitor) -> None:
        """Test suggestions are generated for gaps."""
        session = multi_elicitor.start_session("disc_test001", "contrib_test001")
        suggestions = multi_elicitor.suggest_competencies(session.id)
        assert len(suggestions) > 0

    def test_empty_competency_highest_priority(self, multi_elicitor: ExampleElicitor) -> None:
        """Test empty competencies get priority 1."""
        session = multi_elicitor.start_session("disc_test001", "contrib_test001")
        suggestions = multi_elicitor.suggest_competencies(session.id)
        tools = next(s for s in suggestions if s.competency.id == "comp_tools")
        assert tools.priority == 1
        assert tools.current_count == 0

    def test_suggestions_sorted_by_priority(self, multi_elicitor: ExampleElicitor) -> None:
        """Test suggestions sorted by priority then gap."""
        session = multi_elicitor.start_session("disc_test001", "contrib_test001")
        suggestions = multi_elicitor.suggest_competencies(session.id)
        priorities = [s.priority for s in suggestions]
        assert priorities == sorted(priorities)

    def test_met_competencies_excluded(self, multi_elicitor: ExampleElicitor) -> None:
        """Test met competencies are not suggested."""
        session = multi_elicitor.start_session("disc_test001", "contrib_test001")
        suggestions = multi_elicitor.suggest_competencies(session.id)
        ids = [s.competency.id for s in suggestions]
        # All competencies have gaps in multi_comp_store
        # comp_test001: 1/25, comp_safety: 5/10, comp_tools: 0/15
        assert "comp_test001" in ids
        assert "comp_tools" in ids
        assert "comp_safety" in ids


# --- Session Stats Tests ---


class TestSessionStats:
    """Tests for session statistics."""

    def test_stats_structure(
        self, elicitor: ExampleElicitor, active_session: ElicitationSession
    ) -> None:
        """Test stats dict has expected keys."""
        stats = elicitor.get_session_stats(active_session.id)
        assert "session_id" in stats
        assert "discipline_id" in stats
        assert "status" in stats
        assert "active_drafts" in stats
        assert "finalized_count" in stats

    def test_stats_counts(
        self, elicitor: ExampleElicitor, active_session: ElicitationSession
    ) -> None:
        """Test stats reflect actual state."""
        elicitor.create_draft(
            active_session.id,
            question="A draft question here?",
            competency_id="comp_test001",
        )
        stats = elicitor.get_session_stats(active_session.id)
        assert stats["active_drafts"] == 1
        assert stats["finalized_count"] == 0


# --- Disk Persistence Tests ---


class TestDiskPersistence:
    """Tests for session save/load from disk."""

    def test_session_persisted(self, elicitor_with_disk: ExampleElicitor) -> None:
        """Test session is saved to disk."""
        elicitor_with_disk.start_session("disc_test001", "contrib_test001")
        sessions_dir = elicitor_with_disk._sessions_dir
        assert sessions_dir is not None
        files = list(sessions_dir.glob("esess_*.json"))
        assert len(files) == 1

    def test_session_reloaded(self, populated_store: ForgeStorage, temp_dir: Path) -> None:
        """Test sessions are reloaded on init."""
        sessions_dir = temp_dir / "reload_sessions"
        e1 = ExampleElicitor(populated_store, sessions_dir=sessions_dir)
        session = e1.start_session("disc_test001", "contrib_test001")
        e1.create_draft(
            session.id,
            question="Persisted question?",
            competency_id="comp_test001",
        )

        # Create new elicitor that should reload sessions
        e2 = ExampleElicitor(populated_store, sessions_dir=sessions_dir)
        reloaded = e2.get_session(session.id)
        assert reloaded.discipline_id == "disc_test001"
        assert len(reloaded.drafts) == 1
        assert reloaded.drafts[0].question == "Persisted question?"

    def test_draft_updates_persisted(self, elicitor_with_disk: ExampleElicitor) -> None:
        """Test draft updates are saved to disk."""
        session = elicitor_with_disk.start_session("disc_test001", "contrib_test001")
        draft = elicitor_with_disk.create_draft(
            session.id,
            question="Original draft?",
            competency_id="comp_test001",
        )
        elicitor_with_disk.update_draft(session.id, draft.id, question="Updated draft?")

        # Verify by reading file directly
        sessions_dir = elicitor_with_disk._sessions_dir
        assert sessions_dir is not None
        import json

        path = sessions_dir / f"{session.id}.json"
        with open(path) as f:
            data = json.load(f)
        assert data["drafts"][0]["question"] == "Updated draft?"


# --- Context Building Tests ---


class TestContextBuilding:
    """Tests for context string construction from metadata."""

    def test_empty_metadata(self) -> None:
        """Test empty metadata produces empty context."""
        draft = ExampleDraft(id="d1")
        context = ExampleElicitor._build_context(draft)
        assert context == ""

    def test_user_context_included(self) -> None:
        """Test user context is included."""
        draft = ExampleDraft(id="d1", context="User provided context")
        context = ExampleElicitor._build_context(draft)
        assert "User provided context" in context

    def test_reasoning_pattern_included(self) -> None:
        """Test reasoning pattern is in context."""
        draft = ExampleDraft(
            id="d1",
            metadata=ExampleMetadata(reasoning_pattern=ReasoningPattern.DIAGNOSTIC),
        )
        context = ExampleElicitor._build_context(draft)
        assert "reasoning_pattern: diagnostic" in context

    def test_equipment_included(self) -> None:
        """Test equipment is in context."""
        draft = ExampleDraft(
            id="d1",
            metadata=ExampleMetadata(equipment="M1A1 Abrams"),
        )
        context = ExampleElicitor._build_context(draft)
        assert "equipment: M1A1 Abrams" in context

    def test_default_difficulty_excluded(self) -> None:
        """Test default difficulty (3) is not in context."""
        draft = ExampleDraft(
            id="d1",
            metadata=ExampleMetadata(difficulty=3),
        )
        context = ExampleElicitor._build_context(draft)
        assert "difficulty" not in context

    def test_non_default_difficulty_included(self) -> None:
        """Test non-default difficulty is in context."""
        draft = ExampleDraft(
            id="d1",
            metadata=ExampleMetadata(difficulty=5),
        )
        context = ExampleElicitor._build_context(draft)
        assert "difficulty: 5" in context

    def test_tags_included(self) -> None:
        """Test tags are joined in context."""
        draft = ExampleDraft(
            id="d1",
            metadata=ExampleMetadata(tags=["engine", "oil"]),
        )
        context = ExampleElicitor._build_context(draft)
        assert "tags: engine, oil" in context

    def test_multiple_fields_joined(self) -> None:
        """Test multiple fields are pipe-separated."""
        draft = ExampleDraft(
            id="d1",
            context="Base context",
            metadata=ExampleMetadata(
                reasoning_pattern=ReasoningPattern.PROCEDURAL,
                equipment="HMMWV",
            ),
        )
        context = ExampleElicitor._build_context(draft)
        assert " | " in context
        parts = context.split(" | ")
        assert len(parts) == 3
