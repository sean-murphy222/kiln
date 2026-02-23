"""Tests for discipline discovery interview framework.

Covers: QuestionCatalog, DisciplineModelBuilder, DiscoveryEngine,
and storage round-trip for discovery sessions.
"""

from __future__ import annotations

import pytest

from forge.src.discovery import (
    CATALOG,
    DisciplineModelBuilder,
    DiscoveryEngine,
    DiscoveryError,
    QuestionCatalog,
    QuestionTemplate,
    SessionProgress,
    _next_phase,
)
from forge.src.models import (
    Contributor,
    DisciplineStatus,
    DiscoveryPhase,
    DiscoverySession,
    QuestionResponse,
    ResponseType,
    SessionStatus,
)
from forge.src.storage import ForgeStorage

# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def store() -> ForgeStorage:
    """In-memory storage with schema initialized."""
    s = ForgeStorage(":memory:")
    s.initialize_schema()
    return s


@pytest.fixture
def contributor(store: ForgeStorage) -> Contributor:
    """Contributor persisted in the store."""
    c = Contributor(id="contrib_disc01", name="Test User", email="t@t.com")
    store.create_contributor(c)
    return c


@pytest.fixture
def engine(store: ForgeStorage) -> DiscoveryEngine:
    """DiscoveryEngine wired to in-memory store."""
    return DiscoveryEngine(store)


@pytest.fixture
def in_progress_session(engine: DiscoveryEngine, contributor: Contributor) -> DiscoverySession:
    """A fresh IN_PROGRESS session."""
    return engine.start_session("Test Discipline", contributor.id)


def _answer_all_required(engine: DiscoveryEngine, session: DiscoverySession) -> DiscoverySession:
    """Helper: answer all required questions with dummy data."""
    catalog = CATALOG
    for qid in catalog.get_required_question_ids():
        q = catalog.get_question(qid)
        assert q is not None
        if q.response_type == ResponseType.LIST_ITEMS:
            session = engine.record_response(session, qid, "item1, item2", items=["item1", "item2"])
        else:
            session = engine.record_response(session, qid, "Some answer text")
    return session


# ===================================================================
# QuestionTemplate / SessionProgress dataclass tests
# ===================================================================


class TestQuestionTemplate:
    """Test QuestionTemplate construction."""

    def test_defaults(self) -> None:
        """Test default field values."""
        qt = QuestionTemplate(
            question_id="q_test",
            phase=DiscoveryPhase.ORIENTATION,
            text="Hello?",
        )
        assert qt.response_type == ResponseType.FREE_TEXT
        assert qt.required is True
        assert qt.hint == ""

    def test_custom_values(self) -> None:
        """Test non-default construction."""
        qt = QuestionTemplate(
            question_id="q_test",
            phase=DiscoveryPhase.VOCABULARY,
            text="List terms",
            hint="Include acronyms",
            response_type=ResponseType.LIST_ITEMS,
            required=False,
        )
        assert qt.phase == DiscoveryPhase.VOCABULARY
        assert qt.response_type == ResponseType.LIST_ITEMS
        assert qt.required is False


class TestSessionProgress:
    """Test SessionProgress construction."""

    def test_defaults(self) -> None:
        """Test default field values."""
        sp = SessionProgress(
            session_id="s1",
            current_phase=DiscoveryPhase.ORIENTATION,
        )
        assert sp.completion_percentage == 0.0
        assert sp.unanswered_required == []
        assert sp.estimated_minutes_remaining == 0

    def test_custom_values(self) -> None:
        """Test non-default values."""
        sp = SessionProgress(
            session_id="s1",
            current_phase=DiscoveryPhase.COMPETENCIES,
            phases_complete=[
                DiscoveryPhase.ORIENTATION,
                DiscoveryPhase.DOCUMENTS,
            ],
            completion_percentage=0.6,
            unanswered_required=["q_comp_01"],
            estimated_minutes_remaining=10,
        )
        assert len(sp.phases_complete) == 2
        assert sp.completion_percentage == 0.6


# ===================================================================
# QuestionResponse / DiscoverySession model tests
# ===================================================================


class TestQuestionResponseModel:
    """Test QuestionResponse dataclass and serialization."""

    def test_construction(self) -> None:
        """Test basic construction with defaults."""
        r = QuestionResponse(question_id="q_orient_01", raw_text="Hello")
        assert r.items == []
        assert r.scale_value is None

    def test_roundtrip_serialization(self) -> None:
        """Test to_dict / from_dict roundtrip."""
        r = QuestionResponse(
            question_id="q_orient_01",
            raw_text="text",
            items=["a", "b"],
            scale_value=3,
        )
        d = r.to_dict()
        r2 = QuestionResponse.from_dict(d)
        assert r2.question_id == r.question_id
        assert r2.items == ["a", "b"]
        assert r2.scale_value == 3

    def test_list_items_response(self) -> None:
        """Test LIST_ITEMS response construction."""
        r = QuestionResponse(
            question_id="q_docs_01",
            raw_text="manuals, checklists",
            items=["manuals", "checklists"],
        )
        assert len(r.items) == 2


class TestDiscoverySessionModel:
    """Test DiscoverySession dataclass and serialization."""

    def test_generate_id_prefix(self) -> None:
        """Test ID generation has correct prefix."""
        sid = DiscoverySession.generate_id()
        assert sid.startswith("dsess_")
        assert len(sid) > 6

    def test_default_construction(self) -> None:
        """Test defaults on construction."""
        s = DiscoverySession(
            id="dsess_test",
            discipline_name="Test",
            contributor_id="contrib_01",
        )
        assert s.current_phase == DiscoveryPhase.ORIENTATION
        assert s.status == SessionStatus.IN_PROGRESS
        assert s.responses == {}
        assert s.generated_discipline_id is None
        assert s.completed_at is None

    def test_get_answered_question_ids(self) -> None:
        """Test answered IDs list."""
        s = DiscoverySession(
            id="dsess_test",
            discipline_name="Test",
            contributor_id="contrib_01",
            responses={
                "q1": QuestionResponse(question_id="q1", raw_text="a"),
                "q2": QuestionResponse(question_id="q2", raw_text="b"),
            },
        )
        assert set(s.get_answered_question_ids()) == {"q1", "q2"}

    def test_roundtrip_serialization(self) -> None:
        """Test to_dict / from_dict roundtrip preserves data."""
        s = DiscoverySession(
            id="dsess_test",
            discipline_name="Test",
            contributor_id="contrib_01",
            current_phase=DiscoveryPhase.DOCUMENTS,
            status=SessionStatus.IN_PROGRESS,
            responses={
                "q_orient_01": QuestionResponse(
                    question_id="q_orient_01",
                    raw_text="description",
                ),
            },
        )
        d = s.to_dict()
        s2 = DiscoverySession.from_dict(d)
        assert s2.id == s.id
        assert s2.discipline_name == "Test"
        assert s2.current_phase == DiscoveryPhase.DOCUMENTS
        assert "q_orient_01" in s2.responses


# ===================================================================
# QuestionCatalog tests
# ===================================================================


class TestQuestionCatalog:
    """Test the question catalog queries."""

    def test_default_catalog_has_15_questions(self) -> None:
        """Verify built-in catalog has 15 questions."""
        all_phases = CATALOG.get_all_phases_ordered()
        total = sum(len(CATALOG.get_phase_questions(p)) for p in all_phases)
        assert total == 15

    def test_orientation_has_4_questions(self) -> None:
        """Verify orientation phase has 4 questions."""
        qs = CATALOG.get_phase_questions(DiscoveryPhase.ORIENTATION)
        assert len(qs) == 4

    def test_documents_has_4_questions(self) -> None:
        """Verify documents phase has 4 questions."""
        qs = CATALOG.get_phase_questions(DiscoveryPhase.DOCUMENTS)
        assert len(qs) == 4

    def test_competencies_has_4_questions(self) -> None:
        """Verify competencies phase has 4 questions."""
        qs = CATALOG.get_phase_questions(DiscoveryPhase.COMPETENCIES)
        assert len(qs) == 4

    def test_vocabulary_has_3_questions(self) -> None:
        """Verify vocabulary phase has 3 questions."""
        qs = CATALOG.get_phase_questions(DiscoveryPhase.VOCABULARY)
        assert len(qs) == 3

    def test_get_question_by_id(self) -> None:
        """Test lookup by ID."""
        q = CATALOG.get_question("q_orient_01")
        assert q is not None
        assert q.phase == DiscoveryPhase.ORIENTATION
        assert q.required is True

    def test_get_question_not_found(self) -> None:
        """Test lookup returns None for missing ID."""
        assert CATALOG.get_question("q_nonexistent") is None

    def test_required_question_ids(self) -> None:
        """Test required questions are a subset of all questions."""
        required = CATALOG.get_required_question_ids()
        assert len(required) > 0
        for qid in required:
            q = CATALOG.get_question(qid)
            assert q is not None
            assert q.required is True

    def test_required_ids_for_phase(self) -> None:
        """Test required IDs for orientation phase."""
        required = CATALOG.get_required_ids_for_phase(DiscoveryPhase.ORIENTATION)
        # q_orient_01, 02, 03 are required; 04 is not
        assert len(required) == 3
        assert "q_orient_04" not in required

    def test_phases_order(self) -> None:
        """Test phase ordering is correct."""
        phases = CATALOG.get_all_phases_ordered()
        assert phases == [
            DiscoveryPhase.ORIENTATION,
            DiscoveryPhase.DOCUMENTS,
            DiscoveryPhase.COMPETENCIES,
            DiscoveryPhase.VOCABULARY,
        ]

    def test_custom_catalog(self) -> None:
        """Test catalog with custom templates."""
        custom = [
            QuestionTemplate(
                question_id="custom_01",
                phase=DiscoveryPhase.ORIENTATION,
                text="Custom question?",
            )
        ]
        cat = QuestionCatalog(custom)
        assert len(cat.get_phase_questions(DiscoveryPhase.ORIENTATION)) == 1
        assert cat.get_question("custom_01") is not None
        assert cat.get_question("q_orient_01") is None


# ===================================================================
# _next_phase helper tests
# ===================================================================


class TestNextPhase:
    """Test the _next_phase helper."""

    def test_orientation_to_documents(self) -> None:
        """Test ORIENTATION -> DOCUMENTS."""
        assert _next_phase(DiscoveryPhase.ORIENTATION) == DiscoveryPhase.DOCUMENTS

    def test_documents_to_competencies(self) -> None:
        """Test DOCUMENTS -> COMPETENCIES."""
        assert _next_phase(DiscoveryPhase.DOCUMENTS) == DiscoveryPhase.COMPETENCIES

    def test_competencies_to_vocabulary(self) -> None:
        """Test COMPETENCIES -> VOCABULARY."""
        assert _next_phase(DiscoveryPhase.COMPETENCIES) == DiscoveryPhase.VOCABULARY

    def test_vocabulary_returns_none(self) -> None:
        """Test VOCABULARY returns None (last phase)."""
        assert _next_phase(DiscoveryPhase.VOCABULARY) is None


# ===================================================================
# DisciplineModelBuilder tests
# ===================================================================


class TestDisciplineModelBuilder:
    """Test discipline/competency generation from session data."""

    @pytest.fixture
    def completed_session(self) -> DiscoverySession:
        """Session with all required questions answered."""
        responses = {
            "q_orient_01": QuestionResponse(
                question_id="q_orient_01",
                raw_text="Aircraft maintenance discipline.",
            ),
            "q_orient_02": QuestionResponse(
                question_id="q_orient_02",
                raw_text="mechanics, supervisors",
                items=["Mechanics", "Supervisors"],
            ),
            "q_orient_03": QuestionResponse(
                question_id="q_orient_03",
                raw_text="Careful vs. careless",
            ),
            "q_docs_01": QuestionResponse(
                question_id="q_docs_01",
                raw_text="TOs, checklists",
                items=["Technical Orders", "Checklists"],
            ),
            "q_docs_02": QuestionResponse(
                question_id="q_docs_02",
                raw_text="manuals, forms",
                items=["Manuals", "Forms"],
            ),
            "q_comp_01": QuestionResponse(
                question_id="q_comp_01",
                raw_text="fault isolation, torque application",
                items=["Fault Isolation", "Torque Application"],
            ),
            "q_comp_02": QuestionResponse(
                question_id="q_comp_02",
                raw_text="fault isolation",
                items=["Fault Isolation"],
            ),
            "q_comp_03": QuestionResponse(
                question_id="q_comp_03",
                raw_text="skip steps",
                items=["Skipping steps"],
            ),
            "q_comp_04": QuestionResponse(
                question_id="q_comp_04",
                raw_text="troubleshooting",
                items=["Troubleshooting"],
            ),
            "q_vocab_01": QuestionResponse(
                question_id="q_vocab_01",
                raw_text="TO, WUC",
                items=["TO", "WUC"],
            ),
            "q_vocab_02": QuestionResponse(
                question_id="q_vocab_02",
                raw_text="safety, clearance",
                items=["safety", "clearance"],
            ),
        }
        return DiscoverySession(
            id="dsess_test_complete",
            discipline_name="Aircraft Maintenance",
            contributor_id="contrib_disc01",
            current_phase=DiscoveryPhase.VOCABULARY,
            status=SessionStatus.IN_PROGRESS,
            responses=responses,
        )

    def test_build_discipline(self, completed_session: DiscoverySession) -> None:
        """Test discipline generation from session."""
        builder = DisciplineModelBuilder()
        disc = builder.build_discipline(completed_session, CATALOG)
        assert disc.name == "Aircraft Maintenance"
        assert disc.status == DisciplineStatus.DRAFT
        assert disc.description == "Aircraft maintenance discipline."
        assert "Technical Orders" in disc.document_types
        assert "Manuals" in disc.document_types
        assert "TO" in disc.vocabulary
        assert "safety" in disc.vocabulary
        assert disc.id.startswith("disc_")

    def test_build_discipline_deduplicates(self, completed_session: DiscoverySession) -> None:
        """Test that duplicate vocab/doc_types are removed."""
        # Add duplicate items
        completed_session.responses["q_docs_02"] = QuestionResponse(
            question_id="q_docs_02",
            raw_text="same",
            items=["Technical Orders"],  # duplicate of q_docs_01
        )
        builder = DisciplineModelBuilder()
        disc = builder.build_discipline(completed_session, CATALOG)
        assert disc.document_types.count("Technical Orders") == 1

    def test_build_seed_competencies(self, completed_session: DiscoverySession) -> None:
        """Test competency generation from session."""
        builder = DisciplineModelBuilder()
        comps = builder.build_seed_competencies(completed_session, "disc_test", CATALOG)
        assert len(comps) == 2
        names = [c.name for c in comps]
        assert "Fault Isolation" in names
        assert "Torque Application" in names

    def test_safety_critical_annotation(self, completed_session: DiscoverySession) -> None:
        """Test safety-critical competencies get annotated."""
        builder = DisciplineModelBuilder()
        comps = builder.build_seed_competencies(completed_session, "disc_test", CATALOG)
        fault_iso = next(c for c in comps if c.name == "Fault Isolation")
        assert "SAFETY CRITICAL" in fault_iso.description

        torque = next(c for c in comps if c.name == "Torque Application")
        assert "SAFETY CRITICAL" not in torque.description

    def test_competency_ids_are_unique(self, completed_session: DiscoverySession) -> None:
        """Test all generated competencies have unique IDs."""
        builder = DisciplineModelBuilder()
        comps = builder.build_seed_competencies(completed_session, "disc_test", CATALOG)
        ids = [c.id for c in comps]
        assert len(ids) == len(set(ids))

    def test_competency_default_target(self, completed_session: DiscoverySession) -> None:
        """Test competencies get default coverage target of 25."""
        builder = DisciplineModelBuilder()
        comps = builder.build_seed_competencies(completed_session, "disc_test", CATALOG)
        for c in comps:
            assert c.coverage_target == 25

    def test_missing_response_returns_empty(self) -> None:
        """Test builder handles missing responses gracefully."""
        empty_session = DiscoverySession(
            id="dsess_empty",
            discipline_name="Empty",
            contributor_id="contrib_01",
        )
        builder = DisciplineModelBuilder()
        disc = builder.build_discipline(empty_session, CATALOG)
        assert disc.description == ""
        assert disc.vocabulary == []
        assert disc.document_types == []


# ===================================================================
# DiscoveryEngine Session Lifecycle tests
# ===================================================================


class TestDiscoveryEngineLifecycle:
    """Test session creation, loading, listing, and abandonment."""

    def test_start_session(self, engine: DiscoveryEngine, contributor: Contributor) -> None:
        """Test creating a new session."""
        s = engine.start_session("My Discipline", contributor.id)
        assert s.id.startswith("dsess_")
        assert s.discipline_name == "My Discipline"
        assert s.contributor_id == contributor.id
        assert s.current_phase == DiscoveryPhase.ORIENTATION
        assert s.status == SessionStatus.IN_PROGRESS

    def test_load_session(
        self, engine: DiscoveryEngine, in_progress_session: DiscoverySession
    ) -> None:
        """Test loading an existing session."""
        loaded = engine.load_session(in_progress_session.id)
        assert loaded is not None
        assert loaded.id == in_progress_session.id
        assert loaded.discipline_name == "Test Discipline"

    def test_load_nonexistent_session(self, engine: DiscoveryEngine) -> None:
        """Test loading a missing session returns None."""
        assert engine.load_session("dsess_nonexistent") is None

    def test_list_sessions_all(self, engine: DiscoveryEngine, contributor: Contributor) -> None:
        """Test listing all sessions."""
        engine.start_session("Disc A", contributor.id)
        engine.start_session("Disc B", contributor.id)
        sessions = engine.list_sessions()
        assert len(sessions) == 2

    def test_list_sessions_by_contributor(
        self,
        engine: DiscoveryEngine,
        store: ForgeStorage,
        contributor: Contributor,
    ) -> None:
        """Test filtering sessions by contributor."""
        c2 = Contributor(id="contrib_disc02", name="Other")
        store.create_contributor(c2)
        engine.start_session("Disc A", contributor.id)
        engine.start_session("Disc B", c2.id)
        sessions = engine.list_sessions(contributor_id=contributor.id)
        assert len(sessions) == 1
        assert sessions[0].contributor_id == contributor.id

    def test_list_sessions_by_status(
        self, engine: DiscoveryEngine, contributor: Contributor
    ) -> None:
        """Test filtering sessions by status."""
        s1 = engine.start_session("Disc A", contributor.id)
        engine.start_session("Disc B", contributor.id)
        engine.abandon_session(s1)
        active = engine.list_sessions(status=SessionStatus.IN_PROGRESS)
        assert len(active) == 1
        abandoned = engine.list_sessions(status=SessionStatus.ABANDONED)
        assert len(abandoned) == 1

    def test_abandon_session(
        self, engine: DiscoveryEngine, in_progress_session: DiscoverySession
    ) -> None:
        """Test abandoning a session."""
        result = engine.abandon_session(in_progress_session)
        assert result.status == SessionStatus.ABANDONED
        loaded = engine.load_session(in_progress_session.id)
        assert loaded is not None
        assert loaded.status == SessionStatus.ABANDONED


# ===================================================================
# DiscoveryEngine Interview Flow tests
# ===================================================================


class TestDiscoveryEngineInterviewFlow:
    """Test question retrieval, responses, and phase advancement."""

    def test_get_current_questions_initial(
        self, engine: DiscoveryEngine, in_progress_session: DiscoverySession
    ) -> None:
        """Test initial questions are from ORIENTATION."""
        qs = engine.get_current_questions(in_progress_session)
        assert len(qs) == 4
        assert all(q.phase == DiscoveryPhase.ORIENTATION for q in qs)

    def test_unanswered_first(
        self, engine: DiscoveryEngine, in_progress_session: DiscoverySession
    ) -> None:
        """Test unanswered questions come before answered."""
        engine.record_response(in_progress_session, "q_orient_01", "My answer")
        qs = engine.get_current_questions(in_progress_session)
        # q_orient_01 should be last (already answered)
        assert qs[-1].question_id == "q_orient_01"
        assert qs[0].question_id != "q_orient_01"

    def test_record_response(
        self, engine: DiscoveryEngine, in_progress_session: DiscoverySession
    ) -> None:
        """Test recording a free-text response."""
        updated = engine.record_response(
            in_progress_session, "q_orient_01", "Discipline description"
        )
        assert "q_orient_01" in updated.responses
        r = updated.responses["q_orient_01"]
        assert r.raw_text == "Discipline description"

    def test_record_list_response(
        self, engine: DiscoveryEngine, in_progress_session: DiscoverySession
    ) -> None:
        """Test recording a list-items response."""
        updated = engine.record_response(
            in_progress_session,
            "q_orient_02",
            "mechanics, supervisors",
            items=["mechanics", "supervisors"],
        )
        r = updated.responses["q_orient_02"]
        assert r.items == ["mechanics", "supervisors"]

    def test_record_response_persists(
        self, engine: DiscoveryEngine, in_progress_session: DiscoverySession
    ) -> None:
        """Test response is persisted to storage."""
        engine.record_response(in_progress_session, "q_orient_01", "persisted")
        loaded = engine.load_session(in_progress_session.id)
        assert loaded is not None
        assert "q_orient_01" in loaded.responses
        assert loaded.responses["q_orient_01"].raw_text == "persisted"

    def test_record_invalid_question_raises(
        self, engine: DiscoveryEngine, in_progress_session: DiscoverySession
    ) -> None:
        """Test recording a response with unknown question ID."""
        with pytest.raises(DiscoveryError, match="not found"):
            engine.record_response(in_progress_session, "q_nonexistent", "text")

    def test_record_on_completed_session_raises(
        self, engine: DiscoveryEngine, in_progress_session: DiscoverySession
    ) -> None:
        """Test recording on completed session raises error."""
        in_progress_session.status = SessionStatus.COMPLETED
        with pytest.raises(DiscoveryError, match="already completed"):
            engine.record_response(in_progress_session, "q_orient_01", "text")

    def test_record_on_abandoned_session_raises(
        self, engine: DiscoveryEngine, in_progress_session: DiscoverySession
    ) -> None:
        """Test recording on abandoned session raises error."""
        engine.abandon_session(in_progress_session)
        with pytest.raises(DiscoveryError, match="abandoned"):
            engine.record_response(in_progress_session, "q_orient_01", "text")

    def test_auto_advance_phase(
        self, engine: DiscoveryEngine, in_progress_session: DiscoverySession
    ) -> None:
        """Test auto-advance when all required questions answered."""
        assert in_progress_session.current_phase == DiscoveryPhase.ORIENTATION
        # Answer all required orientation questions
        required = CATALOG.get_required_ids_for_phase(DiscoveryPhase.ORIENTATION)
        for qid in required:
            in_progress_session = engine.record_response(in_progress_session, qid, "answer")
        # Should have advanced to DOCUMENTS
        assert in_progress_session.current_phase == DiscoveryPhase.DOCUMENTS

    def test_no_advance_if_required_missing(
        self, engine: DiscoveryEngine, in_progress_session: DiscoverySession
    ) -> None:
        """Test no advance if required questions unanswered."""
        # Answer only one required question
        engine.record_response(in_progress_session, "q_orient_01", "answer")
        assert in_progress_session.current_phase == DiscoveryPhase.ORIENTATION

    def test_manual_advance_phase(
        self, engine: DiscoveryEngine, in_progress_session: DiscoverySession
    ) -> None:
        """Test manual advance_phase()."""
        # Answer all required orientation questions
        required = CATALOG.get_required_ids_for_phase(DiscoveryPhase.ORIENTATION)
        for qid in required:
            in_progress_session = engine.record_response(in_progress_session, qid, "answer")
        # Auto-advance already moved to DOCUMENTS; now answer docs required
        required_docs = CATALOG.get_required_ids_for_phase(DiscoveryPhase.DOCUMENTS)
        for qid in required_docs:
            in_progress_session = engine.record_response(
                in_progress_session, qid, "doc answer", items=["item"]
            )
        # Should have auto-advanced to COMPETENCIES
        assert in_progress_session.current_phase == DiscoveryPhase.COMPETENCIES

    def test_manual_advance_fails_with_missing(
        self, engine: DiscoveryEngine, in_progress_session: DiscoverySession
    ) -> None:
        """Test manual advance fails if required questions unanswered."""
        with pytest.raises(DiscoveryError, match="unanswered"):
            engine.advance_phase(in_progress_session)

    def test_advance_from_last_phase_fails(
        self, engine: DiscoveryEngine, in_progress_session: DiscoverySession
    ) -> None:
        """Test advance from VOCABULARY (last) fails."""
        in_progress_session = _answer_all_required(engine, in_progress_session)
        # Should now be on VOCABULARY after auto-advance
        assert in_progress_session.current_phase == DiscoveryPhase.VOCABULARY
        # Answer vocab required questions already done by _answer_all_required
        with pytest.raises(DiscoveryError, match="last phase"):
            engine.advance_phase(in_progress_session)


# ===================================================================
# DiscoveryEngine Progress tests
# ===================================================================


class TestDiscoveryEngineProgress:
    """Test progress tracking."""

    def test_progress_initial(
        self, engine: DiscoveryEngine, in_progress_session: DiscoverySession
    ) -> None:
        """Test initial progress is 0%."""
        progress = engine.get_progress(in_progress_session)
        assert progress.completion_percentage == 0.0
        assert progress.phases_complete == []
        assert len(progress.unanswered_required) > 0

    def test_progress_after_answering(
        self, engine: DiscoveryEngine, in_progress_session: DiscoverySession
    ) -> None:
        """Test progress increases after answering questions."""
        engine.record_response(in_progress_session, "q_orient_01", "answer")
        progress = engine.get_progress(in_progress_session)
        assert progress.completion_percentage > 0.0
        assert "q_orient_01" not in progress.unanswered_required

    def test_progress_complete(
        self, engine: DiscoveryEngine, in_progress_session: DiscoverySession
    ) -> None:
        """Test 100% progress when all required answered."""
        in_progress_session = _answer_all_required(engine, in_progress_session)
        progress = engine.get_progress(in_progress_session)
        assert progress.completion_percentage == 1.0
        assert progress.unanswered_required == []
        assert len(progress.phases_complete) == 4

    def test_estimated_minutes(
        self, engine: DiscoveryEngine, in_progress_session: DiscoverySession
    ) -> None:
        """Test estimated minutes decreases as questions answered."""
        initial = engine.get_progress(in_progress_session)
        engine.record_response(in_progress_session, "q_orient_01", "answer")
        after = engine.get_progress(in_progress_session)
        assert after.estimated_minutes_remaining < initial.estimated_minutes_remaining


# ===================================================================
# DiscoveryEngine Completion tests
# ===================================================================


class TestDiscoveryEngineCompletion:
    """Test session completion and discipline generation."""

    def test_complete_session(
        self, engine: DiscoveryEngine, in_progress_session: DiscoverySession
    ) -> None:
        """Test successful session completion."""
        in_progress_session = _answer_all_required(engine, in_progress_session)
        discipline, competencies = engine.complete_session(in_progress_session)
        assert discipline.name == "Test Discipline"
        assert discipline.status == DisciplineStatus.DRAFT
        assert discipline.id.startswith("disc_")
        assert len(competencies) > 0

    def test_complete_session_persists_discipline(
        self,
        engine: DiscoveryEngine,
        store: ForgeStorage,
        in_progress_session: DiscoverySession,
    ) -> None:
        """Test that discipline is persisted to storage."""
        in_progress_session = _answer_all_required(engine, in_progress_session)
        discipline, _ = engine.complete_session(in_progress_session)
        loaded = store.get_discipline(discipline.id)
        assert loaded is not None
        assert loaded.name == discipline.name

    def test_complete_session_persists_competencies(
        self,
        engine: DiscoveryEngine,
        store: ForgeStorage,
        in_progress_session: DiscoverySession,
    ) -> None:
        """Test that competencies are persisted to storage."""
        in_progress_session = _answer_all_required(engine, in_progress_session)
        discipline, competencies = engine.complete_session(in_progress_session)
        stored = store.get_competencies_for_discipline(discipline.id)
        assert len(stored) == len(competencies)

    def test_complete_session_updates_status(
        self, engine: DiscoveryEngine, in_progress_session: DiscoverySession
    ) -> None:
        """Test that session status changes to COMPLETED."""
        in_progress_session = _answer_all_required(engine, in_progress_session)
        engine.complete_session(in_progress_session)
        loaded = engine.load_session(in_progress_session.id)
        assert loaded is not None
        assert loaded.status == SessionStatus.COMPLETED
        assert loaded.generated_discipline_id is not None
        assert loaded.completed_at is not None

    def test_complete_incomplete_session_raises(
        self, engine: DiscoveryEngine, in_progress_session: DiscoverySession
    ) -> None:
        """Test completing without all required answers raises error."""
        with pytest.raises(DiscoveryError, match="unanswered"):
            engine.complete_session(in_progress_session)

    def test_complete_already_completed_raises(
        self, engine: DiscoveryEngine, in_progress_session: DiscoverySession
    ) -> None:
        """Test completing an already-completed session raises error."""
        in_progress_session = _answer_all_required(engine, in_progress_session)
        engine.complete_session(in_progress_session)
        with pytest.raises(DiscoveryError, match="already completed"):
            engine.complete_session(in_progress_session)


# ===================================================================
# Storage round-trip tests
# ===================================================================


class TestStorageDiscoverySessions:
    """Test discovery session storage operations."""

    def test_save_and_load(self, store: ForgeStorage, contributor: Contributor) -> None:
        """Test basic save and reload."""
        session = DiscoverySession(
            id="dsess_storage01",
            discipline_name="Storage Test",
            contributor_id=contributor.id,
        )
        store.save_discovery_session(session)
        loaded = store.get_discovery_session("dsess_storage01")
        assert loaded is not None
        assert loaded.discipline_name == "Storage Test"

    def test_upsert_updates(self, store: ForgeStorage, contributor: Contributor) -> None:
        """Test upsert updates existing session."""
        session = DiscoverySession(
            id="dsess_storage02",
            discipline_name="Before",
            contributor_id=contributor.id,
        )
        store.save_discovery_session(session)
        session.discipline_name = "After"
        store.save_discovery_session(session)
        loaded = store.get_discovery_session("dsess_storage02")
        assert loaded is not None
        assert loaded.discipline_name == "After"

    def test_responses_roundtrip(self, store: ForgeStorage, contributor: Contributor) -> None:
        """Test responses JSON roundtrip through storage."""
        session = DiscoverySession(
            id="dsess_storage03",
            discipline_name="Responses",
            contributor_id=contributor.id,
            responses={
                "q_orient_01": QuestionResponse(
                    question_id="q_orient_01",
                    raw_text="response text",
                    items=["a", "b"],
                    scale_value=4,
                ),
            },
        )
        store.save_discovery_session(session)
        loaded = store.get_discovery_session("dsess_storage03")
        assert loaded is not None
        r = loaded.responses["q_orient_01"]
        assert r.raw_text == "response text"
        assert r.items == ["a", "b"]
        assert r.scale_value == 4

    def test_list_with_filters(self, store: ForgeStorage, contributor: Contributor) -> None:
        """Test list with contributor and status filters."""
        s1 = DiscoverySession(
            id="dsess_f1",
            discipline_name="A",
            contributor_id=contributor.id,
            status=SessionStatus.IN_PROGRESS,
        )
        s2 = DiscoverySession(
            id="dsess_f2",
            discipline_name="B",
            contributor_id=contributor.id,
            status=SessionStatus.ABANDONED,
        )
        store.save_discovery_session(s1)
        store.save_discovery_session(s2)
        active = store.list_discovery_sessions(status=SessionStatus.IN_PROGRESS)
        assert len(active) == 1
        assert active[0].id == "dsess_f1"

    def test_not_found(self, store: ForgeStorage) -> None:
        """Test loading nonexistent session returns None."""
        assert store.get_discovery_session("dsess_missing") is None
