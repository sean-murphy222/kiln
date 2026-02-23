"""Discipline discovery interview framework.

Provides a structured questionnaire system for Step 1 (Discipline Discovery).
Framework-only, no LLM. Templates and forms guide experts through surfacing
discipline characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from forge.src.models import (
    Competency,
    Discipline,
    DisciplineStatus,
    DiscoveryPhase,
    DiscoverySession,
    QuestionResponse,
    ResponseType,
    SessionStatus,
)
from forge.src.storage import ForgeStorage


class DiscoveryError(Exception):
    """Raised for invalid discovery session operations."""


# ===================================================================
# Question Templates
# ===================================================================


@dataclass
class QuestionTemplate:
    """A question in the discovery interview.

    Attributes:
        question_id: Unique identifier (e.g., 'q_orient_01').
        phase: Which interview phase this belongs to.
        text: The question text shown to the user.
        hint: Optional guidance for answering.
        response_type: Expected response format.
        required: Whether the question must be answered.
    """

    question_id: str
    phase: DiscoveryPhase
    text: str
    hint: str = ""
    response_type: ResponseType = ResponseType.FREE_TEXT
    required: bool = True


@dataclass
class SessionProgress:
    """Progress summary for a discovery session.

    Attributes:
        session_id: The session being tracked.
        current_phase: Current interview phase.
        phases_complete: Phases with all required questions answered.
        completion_percentage: Overall progress (0.0 to 1.0).
        unanswered_required: Question IDs still needing answers.
        estimated_minutes_remaining: Rough estimate based on unanswered.
    """

    session_id: str
    current_phase: DiscoveryPhase
    phases_complete: list[DiscoveryPhase] = field(default_factory=list)
    completion_percentage: float = 0.0
    unanswered_required: list[str] = field(default_factory=list)
    estimated_minutes_remaining: int = 0


# ===================================================================
# Question Catalog
# ===================================================================

_PHASE_ORDER = [
    DiscoveryPhase.ORIENTATION,
    DiscoveryPhase.DOCUMENTS,
    DiscoveryPhase.COMPETENCIES,
    DiscoveryPhase.VOCABULARY,
]

_QUESTION_TEMPLATES: list[QuestionTemplate] = [
    # ORIENTATION (~20 min)
    QuestionTemplate(
        question_id="q_orient_01",
        phase=DiscoveryPhase.ORIENTATION,
        text=(
            "Describe this discipline in plain language, as you "
            "would to a new hire on their first day."
        ),
        hint="Focus on what the discipline IS, not technical details.",
        response_type=ResponseType.FREE_TEXT,
        required=True,
    ),
    QuestionTemplate(
        question_id="q_orient_02",
        phase=DiscoveryPhase.ORIENTATION,
        text=("Who performs work in this discipline? " "What roles or job titles?"),
        hint="List all relevant roles, even adjacent ones.",
        response_type=ResponseType.LIST_ITEMS,
        required=True,
    ),
    QuestionTemplate(
        question_id="q_orient_03",
        phase=DiscoveryPhase.ORIENTATION,
        text=("What does a successful practitioner look like " "vs. an unsuccessful one?"),
        response_type=ResponseType.FREE_TEXT,
        required=True,
    ),
    QuestionTemplate(
        question_id="q_orient_04",
        phase=DiscoveryPhase.ORIENTATION,
        text="What are the biggest risks when this discipline is done poorly?",
        response_type=ResponseType.FREE_TEXT,
        required=False,
    ),
    # DOCUMENTS (~15 min)
    QuestionTemplate(
        question_id="q_docs_01",
        phase=DiscoveryPhase.DOCUMENTS,
        text="What documents do practitioners use daily?",
        hint="Include manuals, checklists, schematics, etc.",
        response_type=ResponseType.LIST_ITEMS,
        required=True,
    ),
    QuestionTemplate(
        question_id="q_docs_02",
        phase=DiscoveryPhase.DOCUMENTS,
        text=(
            "What document formats appear most often " "(manuals, checklists, schematics, forms)?"
        ),
        response_type=ResponseType.LIST_ITEMS,
        required=True,
    ),
    QuestionTemplate(
        question_id="q_docs_03",
        phase=DiscoveryPhase.DOCUMENTS,
        text=("Are there documents that are commonly " "misread or misapplied?"),
        response_type=ResponseType.FREE_TEXT,
        required=False,
    ),
    QuestionTemplate(
        question_id="q_docs_04",
        phase=DiscoveryPhase.DOCUMENTS,
        text="How often are documents updated or revised?",
        response_type=ResponseType.FREE_TEXT,
        required=False,
    ),
    # COMPETENCIES (~20 min)
    QuestionTemplate(
        question_id="q_comp_01",
        phase=DiscoveryPhase.COMPETENCIES,
        text=("List the top 5-8 things a practitioner " "must be able to do."),
        hint="Think about core skills, not peripheral tasks.",
        response_type=ResponseType.LIST_ITEMS,
        required=True,
    ),
    QuestionTemplate(
        question_id="q_comp_02",
        phase=DiscoveryPhase.COMPETENCIES,
        text=("Which of those competencies are most " "critical to safety?"),
        response_type=ResponseType.LIST_ITEMS,
        required=True,
    ),
    QuestionTemplate(
        question_id="q_comp_03",
        phase=DiscoveryPhase.COMPETENCIES,
        text=("What are the common failure modes — " "where do new people go wrong?"),
        response_type=ResponseType.LIST_ITEMS,
        required=True,
    ),
    QuestionTemplate(
        question_id="q_comp_04",
        phase=DiscoveryPhase.COMPETENCIES,
        text=(
            "What reasoning patterns are most important "
            "(fault isolation, procedure following, safety checking)?"
        ),
        response_type=ResponseType.LIST_ITEMS,
        required=True,
    ),
    # VOCABULARY (~10 min)
    QuestionTemplate(
        question_id="q_vocab_01",
        phase=DiscoveryPhase.VOCABULARY,
        text="List domain terms that outsiders would not know.",
        hint="Include abbreviations and acronyms.",
        response_type=ResponseType.LIST_ITEMS,
        required=True,
    ),
    QuestionTemplate(
        question_id="q_vocab_02",
        phase=DiscoveryPhase.VOCABULARY,
        text=("List terms that look like common English but " "mean something specific here."),
        response_type=ResponseType.LIST_ITEMS,
        required=True,
    ),
    QuestionTemplate(
        question_id="q_vocab_03",
        phase=DiscoveryPhase.VOCABULARY,
        text=("Are there any terms that are frequently " "confused with each other?"),
        response_type=ResponseType.LIST_ITEMS,
        required=False,
    ),
]


class QuestionCatalog:
    """Catalog of all discovery interview questions.

    Provides query methods for accessing questions by phase, ID,
    and required status.

    Args:
        templates: List of question templates. Defaults to built-in set.
    """

    def __init__(self, templates: list[QuestionTemplate] | None = None) -> None:
        self._templates = templates or list(_QUESTION_TEMPLATES)
        self._by_id = {t.question_id: t for t in self._templates}

    def get_phase_questions(self, phase: DiscoveryPhase) -> list[QuestionTemplate]:
        """Get all questions for a given phase.

        Args:
            phase: The interview phase.

        Returns:
            List of QuestionTemplate for that phase.
        """
        return [t for t in self._templates if t.phase == phase]

    def get_question(self, question_id: str) -> QuestionTemplate | None:
        """Look up a question by ID.

        Args:
            question_id: The question's unique ID.

        Returns:
            QuestionTemplate or None if not found.
        """
        return self._by_id.get(question_id)

    def get_required_question_ids(self) -> list[str]:
        """Get IDs of all required questions across all phases.

        Returns:
            List of question IDs.
        """
        return [t.question_id for t in self._templates if t.required]

    def get_required_ids_for_phase(self, phase: DiscoveryPhase) -> list[str]:
        """Get required question IDs for a specific phase.

        Args:
            phase: The interview phase.

        Returns:
            List of required question IDs.
        """
        return [t.question_id for t in self._templates if t.phase == phase and t.required]

    def get_all_phases_ordered(self) -> list[DiscoveryPhase]:
        """Get phases in interview order.

        Returns:
            Ordered list of DiscoveryPhase values.
        """
        return list(_PHASE_ORDER)


CATALOG = QuestionCatalog()


def _next_phase(current: DiscoveryPhase) -> DiscoveryPhase | None:
    """Get the next phase after the current one.

    Args:
        current: The current phase.

    Returns:
        Next phase, or None if on the last phase.
    """
    idx = _PHASE_ORDER.index(current)
    if idx + 1 >= len(_PHASE_ORDER):
        return None
    return _PHASE_ORDER[idx + 1]


# ===================================================================
# Discipline Model Builder
# ===================================================================


class DisciplineModelBuilder:
    """Generates Discipline and seed Competencies from a completed session."""

    def build_discipline(
        self,
        session: DiscoverySession,
        catalog: QuestionCatalog,
    ) -> Discipline:
        """Extract discipline metadata from session responses.

        Args:
            session: Completed discovery session.
            catalog: Question catalog for reference.

        Returns:
            A new Discipline in DRAFT status.
        """
        description = self._extract_free_text(session, "q_orient_01")
        doc_types = self._extract_list_items(session, "q_docs_01") + self._extract_list_items(
            session, "q_docs_02"
        )
        vocab = self._extract_list_items(session, "q_vocab_01") + self._extract_list_items(
            session, "q_vocab_02"
        )

        return Discipline(
            id=Discipline.generate_id(),
            name=session.discipline_name,
            description=description,
            status=DisciplineStatus.DRAFT,
            created_by=session.contributor_id,
            vocabulary=list(dict.fromkeys(vocab)),
            document_types=list(dict.fromkeys(doc_types)),
        )

    def build_seed_competencies(
        self,
        session: DiscoverySession,
        discipline_id: str,
        catalog: QuestionCatalog,
    ) -> list[Competency]:
        """Extract seed competency list from session responses.

        Args:
            session: Completed discovery session.
            discipline_id: ID of the generated discipline.
            catalog: Question catalog for reference.

        Returns:
            List of seed Competency objects.
        """
        items = self._extract_list_items(session, "q_comp_01")
        safety_items = self._extract_list_items(session, "q_comp_02")
        safety_set = {s.lower().strip() for s in safety_items}

        competencies = []
        for item in items:
            is_safety = item.lower().strip() in safety_set
            description = f"SAFETY CRITICAL: {item}" if is_safety else item
            competencies.append(
                Competency(
                    id=Competency.generate_id(),
                    name=item,
                    description=description,
                    discipline_id=discipline_id,
                    coverage_target=25,
                )
            )
        return competencies

    @staticmethod
    def _extract_list_items(session: DiscoverySession, question_id: str) -> list[str]:
        """Get items from a list-type response.

        Args:
            session: The session containing responses.
            question_id: The question to extract from.

        Returns:
            List of items, or empty list if not answered.
        """
        resp = session.responses.get(question_id)
        if resp is None:
            return []
        return resp.items

    @staticmethod
    def _extract_free_text(session: DiscoverySession, question_id: str) -> str:
        """Get raw text from a free_text response.

        Args:
            session: The session containing responses.
            question_id: The question to extract from.

        Returns:
            Raw text, or empty string if not answered.
        """
        resp = session.responses.get(question_id)
        if resp is None:
            return ""
        return resp.raw_text


# ===================================================================
# Discovery Engine
# ===================================================================


class DiscoveryEngine:
    """Manages discipline discovery interview sessions.

    The main API class for the discovery interview framework.
    Consumers interact only with this class.

    Args:
        storage: ForgeStorage instance with initialized schema.
        catalog: QuestionCatalog to use (defaults to CATALOG).
    """

    def __init__(
        self,
        storage: ForgeStorage,
        catalog: QuestionCatalog | None = None,
    ) -> None:
        self._storage = storage
        self._catalog = catalog or CATALOG
        self._builder = DisciplineModelBuilder()

    # --- Session Lifecycle ---

    def start_session(
        self,
        discipline_name: str,
        contributor_id: str,
    ) -> DiscoverySession:
        """Create and persist a new discovery session.

        Args:
            discipline_name: Working title for the discipline.
            contributor_id: Who is conducting the interview.

        Returns:
            New DiscoverySession in IN_PROGRESS state.
        """
        session = DiscoverySession(
            id=DiscoverySession.generate_id(),
            discipline_name=discipline_name,
            contributor_id=contributor_id,
            current_phase=DiscoveryPhase.ORIENTATION,
            status=SessionStatus.IN_PROGRESS,
        )
        self._storage.save_discovery_session(session)
        return session

    def load_session(self, session_id: str) -> DiscoverySession | None:
        """Load an existing session from storage.

        Args:
            session_id: The session's unique ID.

        Returns:
            DiscoverySession or None if not found.
        """
        return self._storage.get_discovery_session(session_id)

    def list_sessions(
        self,
        contributor_id: str | None = None,
        status: SessionStatus | None = None,
    ) -> list[DiscoverySession]:
        """List sessions, optionally filtered.

        Args:
            contributor_id: Filter by contributor.
            status: Filter by status.

        Returns:
            List of matching sessions.
        """
        return self._storage.list_discovery_sessions(contributor_id=contributor_id, status=status)

    # --- Interview Flow ---

    def get_current_questions(self, session: DiscoverySession) -> list[QuestionTemplate]:
        """Return questions for the session's current phase.

        Unanswered questions appear first.

        Args:
            session: The active session.

        Returns:
            List of QuestionTemplate for the current phase.
        """
        all_q = self._catalog.get_phase_questions(session.current_phase)
        answered = set(session.get_answered_question_ids())
        unanswered = [q for q in all_q if q.question_id not in answered]
        answered_q = [q for q in all_q if q.question_id in answered]
        return unanswered + answered_q

    def record_response(
        self,
        session: DiscoverySession,
        question_id: str,
        raw_text: str,
        items: list[str] | None = None,
        scale_value: int | None = None,
    ) -> DiscoverySession:
        """Record an answer to a question.

        Automatically advances phase if current phase is complete.

        Args:
            session: The active session.
            question_id: ID of the question being answered.
            raw_text: Free-form text response.
            items: Parsed list items (for LIST_ITEMS).
            scale_value: 1-5 value (for SCALE_1_5).

        Returns:
            Updated session.

        Raises:
            DiscoveryError: If question not found or session completed.
        """
        if session.status == SessionStatus.COMPLETED:
            raise DiscoveryError("Session is already completed")
        if session.status == SessionStatus.ABANDONED:
            raise DiscoveryError("Session is abandoned")
        if self._catalog.get_question(question_id) is None:
            raise DiscoveryError(f"Question not found in catalog: {question_id}")

        response = QuestionResponse(
            question_id=question_id,
            raw_text=raw_text,
            items=items or [],
            scale_value=scale_value,
        )
        session.responses[question_id] = response
        session.updated_at = datetime.now()

        # Auto-advance if current phase is complete
        self._auto_advance_phase(session)

        self._storage.save_discovery_session(session)
        return session

    def advance_phase(self, session: DiscoverySession) -> DiscoverySession:
        """Manually advance to the next phase.

        Args:
            session: The active session.

        Returns:
            Updated session.

        Raises:
            DiscoveryError: If required questions unanswered or last phase.
        """
        if not self._is_phase_complete(session, session.current_phase):
            missing = self._get_unanswered_required(session, session.current_phase)
            raise DiscoveryError(
                f"Cannot advance: unanswered required questions "
                f"in {session.current_phase.value}: {missing}"
            )

        nxt = _next_phase(session.current_phase)
        if nxt is None:
            raise DiscoveryError("Already on the last phase (VOCABULARY)")

        session.current_phase = nxt
        session.updated_at = datetime.now()
        self._storage.save_discovery_session(session)
        return session

    def get_progress(self, session: DiscoverySession) -> SessionProgress:
        """Return a progress summary.

        Args:
            session: The session to summarize.

        Returns:
            SessionProgress with computed fields.
        """
        all_required = self._catalog.get_required_question_ids()
        answered = set(session.get_answered_question_ids())
        unanswered = [q for q in all_required if q not in answered]
        answered_count = len([q for q in all_required if q in answered])
        total = len(all_required)
        pct = answered_count / total if total > 0 else 0.0

        phases_done = [p for p in _PHASE_ORDER if self._is_phase_complete(session, p)]

        return SessionProgress(
            session_id=session.id,
            current_phase=session.current_phase,
            phases_complete=phases_done,
            completion_percentage=pct,
            unanswered_required=unanswered,
            estimated_minutes_remaining=len(unanswered) * 2,
        )

    # --- Completion ---

    def complete_session(self, session: DiscoverySession) -> tuple[Discipline, list[Competency]]:
        """Finalize session, generate and persist Discipline + Competencies.

        Args:
            session: The session to complete.

        Returns:
            Tuple of (Discipline, list[Competency]).

        Raises:
            DiscoveryError: If required questions unanswered or already done.
        """
        if session.status == SessionStatus.COMPLETED:
            raise DiscoveryError("Session is already completed")

        # Validate all phases
        all_missing: list[str] = []
        for phase in _PHASE_ORDER:
            all_missing.extend(self._get_unanswered_required(session, phase))
        if all_missing:
            raise DiscoveryError(
                f"Cannot complete: unanswered required questions: " f"{all_missing}"
            )

        # Build discipline and competencies
        discipline = self._builder.build_discipline(session, self._catalog)
        self._storage.create_discipline(discipline)

        competencies = self._builder.build_seed_competencies(session, discipline.id, self._catalog)
        for comp in competencies:
            self._storage.create_competency(comp)

        # Finalize session
        session.status = SessionStatus.COMPLETED
        session.generated_discipline_id = discipline.id
        session.completed_at = datetime.now()
        session.updated_at = datetime.now()
        self._storage.save_discovery_session(session)

        return discipline, competencies

    def abandon_session(self, session: DiscoverySession) -> DiscoverySession:
        """Mark session as abandoned. Preserves data for audit.

        Args:
            session: The session to abandon.

        Returns:
            Updated session with ABANDONED status.
        """
        session.status = SessionStatus.ABANDONED
        session.updated_at = datetime.now()
        self._storage.save_discovery_session(session)
        return session

    # --- Private helpers ---

    def _is_phase_complete(self, session: DiscoverySession, phase: DiscoveryPhase) -> bool:
        """Check if all required questions in a phase are answered."""
        required = self._catalog.get_required_ids_for_phase(phase)
        return all(qid in session.responses for qid in required)

    def _get_unanswered_required(
        self, session: DiscoverySession, phase: DiscoveryPhase
    ) -> list[str]:
        """Get required question IDs not yet answered for a phase."""
        required = self._catalog.get_required_ids_for_phase(phase)
        return [qid for qid in required if qid not in session.responses]

    def _auto_advance_phase(self, session: DiscoverySession) -> None:
        """Advance phase if current phase is complete and not last."""
        if not self._is_phase_complete(session, session.current_phase):
            return
        nxt = _next_phase(session.current_phase)
        if nxt is not None:
            session.current_phase = nxt
