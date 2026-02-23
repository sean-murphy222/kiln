"""Example elicitation engine for Step 3 of Forge curriculum building.

Guides domain experts through creating training examples with
competency tagging, reasoning pattern classification, domain
metadata, and draft management. Sessions are resumable via
JSON serialization.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from forge.src.models import Competency, Example, ReviewStatus
from forge.src.storage import ForgeStorage


class ElicitationError(Exception):
    """Raised for elicitation workflow errors."""


class ReasoningPattern(str, Enum):
    """Types of reasoning exhibited in training examples.

    Attributes:
        PROCEDURAL: Step-by-step procedure execution.
        DIAGNOSTIC: Fault isolation and troubleshooting.
        FACTUAL: Direct knowledge recall.
        ANALYTICAL: Analysis and interpretation of data.
        COMPARATIVE: Comparing options or approaches.
        SAFETY: Safety-critical reasoning and warnings.
    """

    PROCEDURAL = "procedural"
    DIAGNOSTIC = "diagnostic"
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    SAFETY = "safety"


class SessionStatus(str, Enum):
    """Status of an elicitation session.

    Attributes:
        ACTIVE: Session is in progress.
        PAUSED: Session paused for later resumption.
        COMPLETED: All work finished.
    """

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class ExampleMetadata:
    """Domain-specific metadata attached to an example draft.

    Attributes:
        reasoning_pattern: Type of reasoning the example tests.
        equipment: Equipment type or model referenced.
        procedure_ref: Procedure reference (e.g., WP number).
        difficulty: Difficulty level (1-5).
        tags: Free-form tags for categorization.
    """

    reasoning_pattern: ReasoningPattern | None = None
    equipment: str = ""
    procedure_ref: str = ""
    difficulty: int = 3
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "reasoning_pattern": self.reasoning_pattern.value if self.reasoning_pattern else None,
            "equipment": self.equipment,
            "procedure_ref": self.procedure_ref,
            "difficulty": self.difficulty,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExampleMetadata:
        """Deserialize from dictionary."""
        rp = data.get("reasoning_pattern")
        return cls(
            reasoning_pattern=ReasoningPattern(rp) if rp else None,
            equipment=data.get("equipment", ""),
            procedure_ref=data.get("procedure_ref", ""),
            difficulty=data.get("difficulty", 3),
            tags=data.get("tags", []),
        )


@dataclass
class ExampleDraft:
    """An in-progress example before finalization.

    Drafts can be saved and resumed. Once finalized, they become
    full Example objects in ForgeStorage.

    Attributes:
        id: Unique draft identifier.
        question: The question or prompt text.
        ideal_answer: The ideal response text.
        variants: Alternative phrasings of the question.
        context: Additional context for the question.
        competency_id: Target competency area.
        metadata: Domain-specific metadata.
        created_at: When the draft was created.
        updated_at: When the draft was last modified.
    """

    id: str
    question: str = ""
    ideal_answer: str = ""
    variants: list[str] = field(default_factory=list)
    context: str = ""
    competency_id: str = ""
    metadata: ExampleMetadata = field(default_factory=ExampleMetadata)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @staticmethod
    def generate_id() -> str:
        """Generate a unique draft ID."""
        return f"draft_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "ideal_answer": self.ideal_answer,
            "variants": self.variants,
            "context": self.context,
            "competency_id": self.competency_id,
            "metadata": self.metadata.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExampleDraft:
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            question=data.get("question", ""),
            ideal_answer=data.get("ideal_answer", ""),
            variants=data.get("variants", []),
            context=data.get("context", ""),
            competency_id=data.get("competency_id", ""),
            metadata=ExampleMetadata.from_dict(data.get("metadata", {})),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


@dataclass
class CompetencySuggestion:
    """A suggested competency to focus on next.

    Attributes:
        competency: The competency object.
        current_count: Current number of examples.
        target: Coverage target.
        gap: Examples still needed.
        priority: Priority rank (1 = most urgent).
    """

    competency: Competency
    current_count: int
    target: int
    gap: int
    priority: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "competency_id": self.competency.id,
            "competency_name": self.competency.name,
            "current_count": self.current_count,
            "target": self.target,
            "gap": self.gap,
            "priority": self.priority,
        }


@dataclass
class ElicitationSession:
    """Tracks an example elicitation session.

    Sessions are the unit of work for Step 3. A contributor
    starts a session, creates drafts, and finalizes them into
    examples. Sessions can be paused and resumed.

    Attributes:
        id: Unique session identifier.
        discipline_id: The discipline being worked on.
        contributor_id: The contributor creating examples.
        focus_competency_id: Currently focused competency (optional).
        drafts: In-progress example drafts.
        finalized_count: Number of examples finalized this session.
        status: Current session status.
        created_at: When the session started.
        updated_at: When the session was last modified.
    """

    id: str
    discipline_id: str
    contributor_id: str
    focus_competency_id: str | None = None
    drafts: list[ExampleDraft] = field(default_factory=list)
    finalized_count: int = 0
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @staticmethod
    def generate_id() -> str:
        """Generate a unique session ID."""
        return f"esess_{uuid.uuid4().hex[:12]}"

    def get_draft(self, draft_id: str) -> ExampleDraft | None:
        """Get a draft by ID.

        Args:
            draft_id: The draft's unique ID.

        Returns:
            ExampleDraft or None if not found.
        """
        for draft in self.drafts:
            if draft.id == draft_id:
                return draft
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "discipline_id": self.discipline_id,
            "contributor_id": self.contributor_id,
            "focus_competency_id": self.focus_competency_id,
            "drafts": [d.to_dict() for d in self.drafts],
            "finalized_count": self.finalized_count,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ElicitationSession:
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            discipline_id=data["discipline_id"],
            contributor_id=data["contributor_id"],
            focus_competency_id=data.get("focus_competency_id"),
            drafts=[ExampleDraft.from_dict(d) for d in data.get("drafts", [])],
            finalized_count=data.get("finalized_count", 0),
            status=SessionStatus(data.get("status", "active")),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


class ExampleElicitor:
    """Orchestrates example elicitation sessions.

    Wraps ForgeStorage to provide a workflow for creating training
    examples with draft management, competency suggestions, and
    validation before finalization.

    Args:
        storage: ForgeStorage instance for persistence.
        sessions_dir: Directory for saving session state files.
            If None, sessions are not persisted to disk.

    Example::

        elicitor = ExampleElicitor(storage, sessions_dir=Path("./sessions"))
        session = elicitor.start_session("disc_001", "contrib_001")
        draft = elicitor.create_draft(session.id, question="...", ideal_answer="...")
        example = elicitor.finalize_draft(session.id, draft.id)
    """

    def __init__(
        self,
        storage: ForgeStorage,
        sessions_dir: Path | None = None,
    ) -> None:
        self._storage = storage
        self._sessions_dir = sessions_dir
        self._sessions: dict[str, ElicitationSession] = {}

        if sessions_dir:
            sessions_dir.mkdir(parents=True, exist_ok=True)
            self._load_sessions()

    def start_session(
        self,
        discipline_id: str,
        contributor_id: str,
    ) -> ElicitationSession:
        """Start a new elicitation session.

        Args:
            discipline_id: Discipline to create examples for.
            contributor_id: Contributor doing the work.

        Returns:
            New ElicitationSession.

        Raises:
            ElicitationError: If discipline does not exist.
        """
        discipline = self._storage.get_discipline(discipline_id)
        if discipline is None:
            raise ElicitationError(f"Discipline not found: {discipline_id}")

        session = ElicitationSession(
            id=ElicitationSession.generate_id(),
            discipline_id=discipline_id,
            contributor_id=contributor_id,
        )
        self._sessions[session.id] = session
        self._save_session(session)
        return session

    def get_session(self, session_id: str) -> ElicitationSession:
        """Get a session by ID.

        Args:
            session_id: The session's unique ID.

        Returns:
            ElicitationSession.

        Raises:
            ElicitationError: If session not found.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ElicitationError(f"Session not found: {session_id}")
        return session

    def list_sessions(
        self,
        discipline_id: str | None = None,
        status: SessionStatus | None = None,
    ) -> list[ElicitationSession]:
        """List sessions with optional filters.

        Args:
            discipline_id: Filter by discipline.
            status: Filter by status.

        Returns:
            List of matching sessions.
        """
        sessions = list(self._sessions.values())
        if discipline_id is not None:
            sessions = [s for s in sessions if s.discipline_id == discipline_id]
        if status is not None:
            sessions = [s for s in sessions if s.status == status]
        return sessions

    def set_focus_competency(
        self,
        session_id: str,
        competency_id: str,
    ) -> ElicitationSession:
        """Set the competency focus for a session.

        Args:
            session_id: Session to update.
            competency_id: Competency to focus on.

        Returns:
            Updated session.

        Raises:
            ElicitationError: If session or competency not found.
        """
        session = self.get_session(session_id)
        comp = self._storage.get_competency(competency_id)
        if comp is None:
            raise ElicitationError(f"Competency not found: {competency_id}")
        session.focus_competency_id = competency_id
        session.updated_at = datetime.now()
        self._save_session(session)
        return session

    def create_draft(
        self,
        session_id: str,
        question: str = "",
        ideal_answer: str = "",
        competency_id: str | None = None,
        variants: list[str] | None = None,
        context: str = "",
        metadata: ExampleMetadata | None = None,
    ) -> ExampleDraft:
        """Create a new example draft in a session.

        Args:
            session_id: Session to add the draft to.
            question: Question or prompt text.
            ideal_answer: Ideal response text.
            competency_id: Target competency (defaults to session focus).
            variants: Alternative phrasings.
            context: Additional context.
            metadata: Domain-specific metadata.

        Returns:
            New ExampleDraft.

        Raises:
            ElicitationError: If session not found or not active.
        """
        session = self.get_session(session_id)
        self._require_active(session)

        draft = ExampleDraft(
            id=ExampleDraft.generate_id(),
            question=question,
            ideal_answer=ideal_answer,
            variants=variants or [],
            context=context,
            competency_id=competency_id or session.focus_competency_id or "",
            metadata=metadata or ExampleMetadata(),
        )
        session.drafts.append(draft)
        session.updated_at = datetime.now()
        self._save_session(session)
        return draft

    def update_draft(
        self,
        session_id: str,
        draft_id: str,
        **updates: Any,
    ) -> ExampleDraft:
        """Update fields on an existing draft.

        Args:
            session_id: Session containing the draft.
            draft_id: Draft to update.
            **updates: Fields to update (question, ideal_answer,
                variants, context, competency_id, metadata).

        Returns:
            Updated ExampleDraft.

        Raises:
            ElicitationError: If session/draft not found.
        """
        session = self.get_session(session_id)
        self._require_active(session)
        draft = session.get_draft(draft_id)
        if draft is None:
            raise ElicitationError(f"Draft not found: {draft_id}")

        for key, value in updates.items():
            if hasattr(draft, key) and key != "id":
                setattr(draft, key, value)
        draft.updated_at = datetime.now()
        session.updated_at = datetime.now()
        self._save_session(session)
        return draft

    def delete_draft(
        self,
        session_id: str,
        draft_id: str,
    ) -> bool:
        """Delete a draft from a session.

        Args:
            session_id: Session containing the draft.
            draft_id: Draft to delete.

        Returns:
            True if deleted, False if not found.
        """
        session = self.get_session(session_id)
        original_count = len(session.drafts)
        session.drafts = [d for d in session.drafts if d.id != draft_id]
        if len(session.drafts) < original_count:
            session.updated_at = datetime.now()
            self._save_session(session)
            return True
        return False

    def validate_draft(self, draft: ExampleDraft) -> list[str]:
        """Validate a draft before finalization.

        Args:
            draft: Draft to validate.

        Returns:
            List of validation error messages (empty = valid).
        """
        errors: list[str] = []
        if not draft.question.strip():
            errors.append("Question is required")
        if not draft.ideal_answer.strip():
            errors.append("Ideal answer is required")
        if not draft.competency_id:
            errors.append("Competency ID is required")
        if len(draft.question) < 10:
            errors.append("Question must be at least 10 characters")
        if len(draft.ideal_answer) < 10:
            errors.append("Ideal answer must be at least 10 characters")
        if draft.metadata.difficulty < 1 or draft.metadata.difficulty > 5:
            errors.append("Difficulty must be between 1 and 5")
        return errors

    def finalize_draft(
        self,
        session_id: str,
        draft_id: str,
    ) -> Example:
        """Finalize a draft into a persisted Example.

        Validates the draft, creates an Example in ForgeStorage,
        and removes the draft from the session.

        Args:
            session_id: Session containing the draft.
            draft_id: Draft to finalize.

        Returns:
            The created Example.

        Raises:
            ElicitationError: If validation fails or draft not found.
        """
        session = self.get_session(session_id)
        self._require_active(session)
        draft = session.get_draft(draft_id)
        if draft is None:
            raise ElicitationError(f"Draft not found: {draft_id}")

        errors = self.validate_draft(draft)
        if errors:
            raise ElicitationError(f"Draft validation failed: {'; '.join(errors)}")

        # Build context string with metadata
        context = self._build_context(draft)

        example = Example(
            id=Example.generate_id(),
            question=draft.question,
            ideal_answer=draft.ideal_answer,
            competency_id=draft.competency_id,
            contributor_id=session.contributor_id,
            discipline_id=session.discipline_id,
            variants=draft.variants,
            context=context,
            review_status=ReviewStatus.PENDING,
        )

        self._storage.create_example(example)

        # Remove draft and update counters
        session.drafts = [d for d in session.drafts if d.id != draft_id]
        session.finalized_count += 1
        session.updated_at = datetime.now()
        self._save_session(session)
        return example

    def suggest_competencies(
        self,
        session_id: str,
    ) -> list[CompetencySuggestion]:
        """Suggest competencies to focus on based on coverage gaps.

        Competencies with fewer examples relative to their target
        are suggested first. Empty competencies get highest priority.

        Args:
            session_id: Session to generate suggestions for.

        Returns:
            List of CompetencySuggestion sorted by priority.
        """
        session = self.get_session(session_id)
        competencies = self._storage.get_competencies_for_discipline(session.discipline_id)

        suggestions: list[CompetencySuggestion] = []
        for comp in competencies:
            examples = self._storage.get_examples_for_competency(comp.id, include_test_set=False)
            count = len(examples)
            gap = max(0, comp.coverage_target - count)
            if gap == 0:
                continue

            if count == 0:
                priority = 1
            elif count / comp.coverage_target < 0.5:
                priority = 2
            else:
                priority = 3

            suggestions.append(
                CompetencySuggestion(
                    competency=comp,
                    current_count=count,
                    target=comp.coverage_target,
                    gap=gap,
                    priority=priority,
                )
            )

        suggestions.sort(key=lambda s: (s.priority, -s.gap))
        return suggestions

    def get_session_stats(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """Get statistics for a session.

        Args:
            session_id: Session to report on.

        Returns:
            Dict with session statistics.
        """
        session = self.get_session(session_id)
        return {
            "session_id": session.id,
            "discipline_id": session.discipline_id,
            "status": session.status.value,
            "active_drafts": len(session.drafts),
            "finalized_count": session.finalized_count,
            "focus_competency_id": session.focus_competency_id,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
        }

    def pause_session(self, session_id: str) -> ElicitationSession:
        """Pause a session for later resumption.

        Args:
            session_id: Session to pause.

        Returns:
            Updated session.

        Raises:
            ElicitationError: If session not active.
        """
        session = self.get_session(session_id)
        self._require_active(session)
        session.status = SessionStatus.PAUSED
        session.updated_at = datetime.now()
        self._save_session(session)
        return session

    def resume_session(self, session_id: str) -> ElicitationSession:
        """Resume a paused session.

        Args:
            session_id: Session to resume.

        Returns:
            Updated session.

        Raises:
            ElicitationError: If session not paused.
        """
        session = self.get_session(session_id)
        if session.status != SessionStatus.PAUSED:
            raise ElicitationError(
                f"Session {session_id} is not paused " f"(status: {session.status.value})"
            )
        session.status = SessionStatus.ACTIVE
        session.updated_at = datetime.now()
        self._save_session(session)
        return session

    def complete_session(self, session_id: str) -> ElicitationSession:
        """Mark a session as completed.

        Remaining drafts are preserved but the session cannot
        accept new drafts or finalize existing ones.

        Args:
            session_id: Session to complete.

        Returns:
            Updated session.
        """
        session = self.get_session(session_id)
        session.status = SessionStatus.COMPLETED
        session.updated_at = datetime.now()
        self._save_session(session)
        return session

    @staticmethod
    def _build_context(draft: ExampleDraft) -> str:
        """Build context string from draft metadata.

        Args:
            draft: Draft to extract context from.

        Returns:
            Context string combining user context and metadata.
        """
        parts: list[str] = []
        if draft.context:
            parts.append(draft.context)
        if draft.metadata.reasoning_pattern:
            parts.append(f"reasoning_pattern: {draft.metadata.reasoning_pattern.value}")
        if draft.metadata.equipment:
            parts.append(f"equipment: {draft.metadata.equipment}")
        if draft.metadata.procedure_ref:
            parts.append(f"procedure_ref: {draft.metadata.procedure_ref}")
        if draft.metadata.difficulty != 3:
            parts.append(f"difficulty: {draft.metadata.difficulty}")
        if draft.metadata.tags:
            parts.append(f"tags: {', '.join(draft.metadata.tags)}")
        return " | ".join(parts)

    @staticmethod
    def _require_active(session: ElicitationSession) -> None:
        """Raise if session is not active.

        Args:
            session: Session to check.

        Raises:
            ElicitationError: If session is not active.
        """
        if session.status != SessionStatus.ACTIVE:
            raise ElicitationError(
                f"Session {session.id} is not active " f"(status: {session.status.value})"
            )

    def _save_session(self, session: ElicitationSession) -> None:
        """Persist session state to disk if sessions_dir is configured.

        Args:
            session: Session to save.
        """
        if self._sessions_dir is None:
            return
        path = self._sessions_dir / f"{session.id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)

    def _load_sessions(self) -> None:
        """Load all sessions from the sessions directory."""
        if self._sessions_dir is None:
            return
        for path in self._sessions_dir.glob("esess_*.json"):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            session = ElicitationSession.from_dict(data)
            self._sessions[session.id] = session
