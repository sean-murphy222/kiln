"""FastAPI router for Forge curriculum builder.

Exposes REST endpoints for contributors, disciplines, competencies,
examples, discovery sessions, consistency checking, coverage reports,
and curriculum export. Designed to be mounted at /api/forge/ by the
parent application.

All endpoint functions are synchronous (not async) because the
underlying ForgeStorage uses synchronous SQLite calls. FastAPI runs
sync handlers in a thread pool automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from forge.src.competency import CompetencyMapper
from forge.src.consistency import ConsistencyChecker
from forge.src.contributors import ContributorManager
from forge.src.discovery import DiscoveryEngine, DiscoveryError
from forge.src.examples import ExampleElicitor
from forge.src.models import (
    Competency,
    Contributor,
    Discipline,
    DisciplineStatus,
    Example,
    ReviewStatus,
)
from forge.src.storage import ForgeStorage, ForgeStorageError
from forge.src.test_split import TestSetManager

router = APIRouter()

# ---------------------------------------------------------------------------
# Module-level storage instance (initialized by init_forge_storage)
# ---------------------------------------------------------------------------

_storage: ForgeStorage | None = None
_discovery_engine: DiscoveryEngine | None = None
_competency_mapper: CompetencyMapper | None = None
_consistency_checker: ConsistencyChecker | None = None
_contributor_manager: ContributorManager | None = None
_example_elicitor: ExampleElicitor | None = None
_test_set_manager: TestSetManager | None = None


def init_forge_storage(db_path: str | Path = ":memory:") -> ForgeStorage:
    """Initialize the Forge storage backend and all service objects.

    Call this once at application startup before any requests are served.

    Args:
        db_path: Path to SQLite database file, or ':memory:'.

    Returns:
        The initialized ForgeStorage instance.
    """
    import sqlite3

    global _storage, _discovery_engine, _competency_mapper
    global _consistency_checker, _contributor_manager
    global _example_elicitor, _test_set_manager

    _storage = ForgeStorage(db_path)
    # Re-create connection with check_same_thread=False so that
    # sync FastAPI handlers (which run in a threadpool) can use
    # the same connection created during init.
    _storage._conn.close()
    _storage._conn = sqlite3.connect(str(db_path), check_same_thread=False)
    _storage._conn.execute("PRAGMA foreign_keys = ON")
    _storage._conn.row_factory = sqlite3.Row
    _storage.initialize_schema()

    _discovery_engine = DiscoveryEngine(_storage)
    _competency_mapper = CompetencyMapper(_storage)
    _consistency_checker = ConsistencyChecker(_storage)
    _contributor_manager = ContributorManager(_storage)
    _example_elicitor = ExampleElicitor(_storage)
    _test_set_manager = TestSetManager(_storage)

    return _storage


def get_storage() -> ForgeStorage:
    """Return the initialized ForgeStorage or raise.

    Returns:
        The active ForgeStorage instance.

    Raises:
        HTTPException: If storage has not been initialized.
    """
    if _storage is None:
        raise HTTPException(
            status_code=500,
            detail="Forge storage not initialized",
        )
    return _storage


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class ContributorCreate(BaseModel):
    """Request body for creating a contributor."""

    name: str = Field(..., min_length=1, max_length=200)
    email: str = Field(default="", max_length=300)


class ContributorUpdate(BaseModel):
    """Request body for updating a contributor."""

    name: str | None = Field(default=None, min_length=1, max_length=200)
    email: str | None = Field(default=None, max_length=300)


class DisciplineCreate(BaseModel):
    """Request body for creating a discipline."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1)
    created_by: str = Field(..., min_length=1)
    vocabulary: list[str] = Field(default_factory=list)
    document_types: list[str] = Field(default_factory=list)


class DisciplineUpdate(BaseModel):
    """Request body for updating a discipline."""

    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None
    status: str | None = None
    vocabulary: list[str] | None = None
    document_types: list[str] | None = None


class CompetencyCreate(BaseModel):
    """Request body for creating a competency."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1)
    discipline_id: str = Field(..., min_length=1)
    parent_id: str | None = None
    coverage_target: int = Field(default=25, ge=1)


class CompetencyUpdate(BaseModel):
    """Request body for updating a competency."""

    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None
    coverage_target: int | None = Field(default=None, ge=1)
    parent_id: str | None = None


class ExampleCreate(BaseModel):
    """Request body for creating an example."""

    question: str = Field(..., min_length=1)
    ideal_answer: str = Field(..., min_length=1)
    competency_id: str = Field(..., min_length=1)
    contributor_id: str = Field(..., min_length=1)
    discipline_id: str = Field(..., min_length=1)
    variants: list[str] = Field(default_factory=list)
    context: str = ""


class ExampleUpdate(BaseModel):
    """Request body for updating an example."""

    question: str | None = None
    ideal_answer: str | None = None
    variants: list[str] | None = None
    context: str | None = None
    review_status: str | None = None


class DiscoveryStartRequest(BaseModel):
    """Request body for starting a discovery session."""

    discipline_name: str = Field(..., min_length=1, max_length=200)
    contributor_id: str = Field(..., min_length=1)


class DiscoveryAnswerRequest(BaseModel):
    """Request body for recording a discovery answer."""

    session_id: str = Field(..., min_length=1)
    question_id: str = Field(..., min_length=1)
    raw_text: str = Field(default="")
    items: list[str] = Field(default_factory=list)
    scale_value: int | None = None


class CurriculumExportRequest(BaseModel):
    """Request body for curriculum export."""

    created_by: str = Field(..., min_length=1)
    include_test_set: bool = False


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health")
def health_check() -> dict[str, Any]:
    """Return Forge service health status.

    Returns:
        Dict with status, version, and storage availability.
    """
    return {
        "status": "ok",
        "service": "forge",
        "version": "0.1.0",
        "storage_initialized": _storage is not None,
    }


# ---------------------------------------------------------------------------
# Contributors
# ---------------------------------------------------------------------------


@router.get("/contributors")
def list_contributors() -> dict[str, Any]:
    """List all contributors.

    Returns:
        Dict containing list of contributor dicts.
    """
    try:
        storage = get_storage()
        rows = storage._conn.execute("SELECT * FROM contributors").fetchall()
        contributors = [storage._row_to_contributor(r).to_dict() for r in rows]
        return {"contributors": contributors}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to list contributors") from exc


@router.get("/contributors/{contributor_id}")
def get_contributor(contributor_id: str) -> dict[str, Any]:
    """Get a contributor by ID.

    Args:
        contributor_id: The contributor's unique ID.

    Returns:
        Contributor dict.
    """
    try:
        storage = get_storage()
        contributor = storage.get_contributor(contributor_id)
        if contributor is None:
            raise HTTPException(status_code=404, detail="Contributor not found")
        return contributor.to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to get contributor") from exc


@router.post("/contributors", status_code=201)
def create_contributor(body: ContributorCreate) -> dict[str, Any]:
    """Create a new contributor.

    Args:
        body: Contributor creation request.

    Returns:
        Created contributor dict.
    """
    try:
        storage = get_storage()
        contributor = Contributor(
            id=Contributor.generate_id(),
            name=body.name,
            email=body.email,
        )
        storage.create_contributor(contributor)
        return contributor.to_dict()
    except ForgeStorageError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to create contributor") from exc


@router.put("/contributors/{contributor_id}")
def update_contributor(
    contributor_id: str,
    body: ContributorUpdate,
) -> dict[str, Any]:
    """Update an existing contributor.

    Args:
        contributor_id: The contributor's unique ID.
        body: Fields to update.

    Returns:
        Updated contributor dict.
    """
    try:
        storage = get_storage()
        contributor = storage.get_contributor(contributor_id)
        if contributor is None:
            raise HTTPException(status_code=404, detail="Contributor not found")
        if body.name is not None:
            contributor.name = body.name
        if body.email is not None:
            contributor.email = body.email
        storage.update_contributor(contributor)
        return contributor.to_dict()
    except HTTPException:
        raise
    except ForgeStorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to update contributor") from exc


@router.delete("/contributors/{contributor_id}")
def delete_contributor(contributor_id: str) -> dict[str, Any]:
    """Delete a contributor by ID.

    Args:
        contributor_id: The contributor's unique ID.

    Returns:
        Confirmation dict.
    """
    try:
        storage = get_storage()
        deleted = storage.delete_contributor(contributor_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Contributor not found")
        return {"deleted": contributor_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to delete contributor") from exc


# ---------------------------------------------------------------------------
# Disciplines
# ---------------------------------------------------------------------------


@router.get("/disciplines")
def list_disciplines(status: str | None = None) -> dict[str, Any]:
    """List all disciplines, optionally filtered by status.

    Args:
        status: Optional status filter (draft, active, archived).

    Returns:
        Dict containing list of discipline dicts.
    """
    try:
        storage = get_storage()
        ds = None
        if status is not None:
            ds = DisciplineStatus(status)
        disciplines = storage.get_all_disciplines(status=ds)
        return {"disciplines": [d.to_dict() for d in disciplines]}
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status value: {status}",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to list disciplines") from exc


@router.get("/disciplines/{discipline_id}")
def get_discipline(discipline_id: str) -> dict[str, Any]:
    """Get a discipline by ID.

    Args:
        discipline_id: The discipline's unique ID.

    Returns:
        Discipline dict.
    """
    try:
        storage = get_storage()
        discipline = storage.get_discipline(discipline_id)
        if discipline is None:
            raise HTTPException(status_code=404, detail="Discipline not found")
        return discipline.to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to get discipline") from exc


@router.post("/disciplines", status_code=201)
def create_discipline(body: DisciplineCreate) -> dict[str, Any]:
    """Create a new discipline.

    Args:
        body: Discipline creation request.

    Returns:
        Created discipline dict.
    """
    try:
        storage = get_storage()
        discipline = Discipline(
            id=Discipline.generate_id(),
            name=body.name,
            description=body.description,
            created_by=body.created_by,
            vocabulary=body.vocabulary,
            document_types=body.document_types,
        )
        storage.create_discipline(discipline)
        return discipline.to_dict()
    except ForgeStorageError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to create discipline") from exc


@router.put("/disciplines/{discipline_id}")
def update_discipline(
    discipline_id: str,
    body: DisciplineUpdate,
) -> dict[str, Any]:
    """Update an existing discipline.

    Args:
        discipline_id: The discipline's unique ID.
        body: Fields to update.

    Returns:
        Updated discipline dict.
    """
    try:
        storage = get_storage()
        discipline = storage.get_discipline(discipline_id)
        if discipline is None:
            raise HTTPException(status_code=404, detail="Discipline not found")
        if body.name is not None:
            discipline.name = body.name
        if body.description is not None:
            discipline.description = body.description
        if body.status is not None:
            discipline.status = DisciplineStatus(body.status)
        if body.vocabulary is not None:
            discipline.vocabulary = body.vocabulary
        if body.document_types is not None:
            discipline.document_types = body.document_types
        storage.update_discipline(discipline)
        return discipline.to_dict()
    except HTTPException:
        raise
    except ForgeStorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to update discipline") from exc


# ---------------------------------------------------------------------------
# Competencies
# ---------------------------------------------------------------------------


@router.get("/competencies/{competency_id}")
def get_competency(competency_id: str) -> dict[str, Any]:
    """Get a competency by ID.

    Args:
        competency_id: The competency's unique ID.

    Returns:
        Competency dict.
    """
    try:
        storage = get_storage()
        competency = storage.get_competency(competency_id)
        if competency is None:
            raise HTTPException(status_code=404, detail="Competency not found")
        return competency.to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to get competency") from exc


@router.get("/disciplines/{discipline_id}/competencies")
def list_competencies_for_discipline(
    discipline_id: str,
) -> dict[str, Any]:
    """List all competencies for a discipline.

    Args:
        discipline_id: The discipline's unique ID.

    Returns:
        Dict containing list of competency dicts.
    """
    try:
        storage = get_storage()
        competencies = storage.get_competencies_for_discipline(discipline_id)
        return {"competencies": [c.to_dict() for c in competencies]}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to list competencies") from exc


@router.post("/competencies", status_code=201)
def create_competency(body: CompetencyCreate) -> dict[str, Any]:
    """Create a new competency.

    Args:
        body: Competency creation request.

    Returns:
        Created competency dict.
    """
    try:
        storage = get_storage()
        competency = Competency(
            id=Competency.generate_id(),
            name=body.name,
            description=body.description,
            discipline_id=body.discipline_id,
            parent_id=body.parent_id,
            coverage_target=body.coverage_target,
        )
        storage.create_competency(competency)
        return competency.to_dict()
    except ForgeStorageError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to create competency") from exc


@router.put("/competencies/{competency_id}")
def update_competency(
    competency_id: str,
    body: CompetencyUpdate,
) -> dict[str, Any]:
    """Update an existing competency.

    Args:
        competency_id: The competency's unique ID.
        body: Fields to update.

    Returns:
        Updated competency dict.
    """
    try:
        storage = get_storage()
        competency = storage.get_competency(competency_id)
        if competency is None:
            raise HTTPException(status_code=404, detail="Competency not found")
        if body.name is not None:
            competency.name = body.name
        if body.description is not None:
            competency.description = body.description
        if body.coverage_target is not None:
            competency.coverage_target = body.coverage_target
        if body.parent_id is not None:
            competency.parent_id = body.parent_id
        storage.update_competency(competency)
        return competency.to_dict()
    except HTTPException:
        raise
    except ForgeStorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to update competency") from exc


@router.delete("/competencies/{competency_id}")
def delete_competency(competency_id: str) -> dict[str, Any]:
    """Delete a competency by ID.

    Args:
        competency_id: The competency's unique ID.

    Returns:
        Confirmation dict.
    """
    try:
        storage = get_storage()
        deleted = storage.delete_competency(competency_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Competency not found")
        return {"deleted": competency_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to delete competency") from exc


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


@router.get("/examples/{example_id}")
def get_example(example_id: str) -> dict[str, Any]:
    """Get an example by ID.

    Args:
        example_id: The example's unique ID.

    Returns:
        Example dict.
    """
    try:
        storage = get_storage()
        example = storage.get_example(example_id)
        if example is None:
            raise HTTPException(status_code=404, detail="Example not found")
        return example.to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to get example") from exc


@router.get("/competencies/{competency_id}/examples")
def list_examples_for_competency(
    competency_id: str,
    include_test_set: bool = True,
) -> dict[str, Any]:
    """List all examples for a competency.

    Args:
        competency_id: The competency's unique ID.
        include_test_set: Whether to include test-set examples.

    Returns:
        Dict containing list of example dicts.
    """
    try:
        storage = get_storage()
        examples = storage.get_examples_for_competency(
            competency_id,
            include_test_set=include_test_set,
        )
        return {"examples": [e.to_dict() for e in examples]}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to list examples") from exc


@router.post("/examples", status_code=201)
def create_example(body: ExampleCreate) -> dict[str, Any]:
    """Create a new example.

    Args:
        body: Example creation request.

    Returns:
        Created example dict.
    """
    try:
        storage = get_storage()
        example = Example(
            id=Example.generate_id(),
            question=body.question,
            ideal_answer=body.ideal_answer,
            competency_id=body.competency_id,
            contributor_id=body.contributor_id,
            discipline_id=body.discipline_id,
            variants=body.variants,
            context=body.context,
        )
        storage.create_example(example)
        return example.to_dict()
    except ForgeStorageError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to create example") from exc


@router.put("/examples/{example_id}")
def update_example(
    example_id: str,
    body: ExampleUpdate,
) -> dict[str, Any]:
    """Update an existing example.

    Args:
        example_id: The example's unique ID.
        body: Fields to update.

    Returns:
        Updated example dict.
    """
    try:
        storage = get_storage()
        example = storage.get_example(example_id)
        if example is None:
            raise HTTPException(status_code=404, detail="Example not found")
        if body.question is not None:
            example.question = body.question
        if body.ideal_answer is not None:
            example.ideal_answer = body.ideal_answer
        if body.variants is not None:
            example.variants = body.variants
        if body.context is not None:
            example.context = body.context
        if body.review_status is not None:
            example.review_status = ReviewStatus(body.review_status)
        storage.update_example(example)
        return example.to_dict()
    except HTTPException:
        raise
    except ForgeStorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to update example") from exc


@router.delete("/examples/{example_id}")
def delete_example(example_id: str) -> dict[str, Any]:
    """Delete an example by ID.

    Args:
        example_id: The example's unique ID.

    Returns:
        Confirmation dict.
    """
    try:
        storage = get_storage()
        deleted = storage.delete_example(example_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Example not found")
        return {"deleted": example_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to delete example") from exc


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@router.post("/discovery/start", status_code=201)
def start_discovery(body: DiscoveryStartRequest) -> dict[str, Any]:
    """Start a new discipline discovery session.

    Args:
        body: Discovery session start request.

    Returns:
        New session dict with current questions.
    """
    try:
        if _discovery_engine is None:
            raise HTTPException(status_code=500, detail="Discovery engine not initialized")
        session = _discovery_engine.start_session(
            discipline_name=body.discipline_name,
            contributor_id=body.contributor_id,
        )
        questions = _discovery_engine.get_current_questions(session)
        return {
            "session": session.to_dict(),
            "current_questions": [
                {
                    "question_id": q.question_id,
                    "phase": q.phase.value,
                    "text": q.text,
                    "hint": q.hint,
                    "response_type": q.response_type.value,
                    "required": q.required,
                }
                for q in questions
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to start discovery session") from exc


@router.post("/discovery/answer")
def record_discovery_answer(body: DiscoveryAnswerRequest) -> dict[str, Any]:
    """Record an answer to a discovery question.

    Args:
        body: Answer recording request.

    Returns:
        Updated session dict with current questions.
    """
    try:
        if _discovery_engine is None:
            raise HTTPException(status_code=500, detail="Discovery engine not initialized")
        session = _discovery_engine.load_session(body.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        session = _discovery_engine.record_response(
            session=session,
            question_id=body.question_id,
            raw_text=body.raw_text,
            items=body.items,
            scale_value=body.scale_value,
        )
        questions = _discovery_engine.get_current_questions(session)
        return {
            "session": session.to_dict(),
            "current_questions": [
                {
                    "question_id": q.question_id,
                    "phase": q.phase.value,
                    "text": q.text,
                    "hint": q.hint,
                    "response_type": q.response_type.value,
                    "required": q.required,
                }
                for q in questions
            ],
        }
    except HTTPException:
        raise
    except DiscoveryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to record answer") from exc


@router.get("/discovery/{session_id}/progress")
def get_discovery_progress(session_id: str) -> dict[str, Any]:
    """Get progress summary for a discovery session.

    Args:
        session_id: The session's unique ID.

    Returns:
        Progress summary dict.
    """
    try:
        if _discovery_engine is None:
            raise HTTPException(status_code=500, detail="Discovery engine not initialized")
        session = _discovery_engine.load_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        progress = _discovery_engine.get_progress(session)
        return {
            "session_id": progress.session_id,
            "current_phase": progress.current_phase.value,
            "phases_complete": [p.value for p in progress.phases_complete],
            "completion_percentage": progress.completion_percentage,
            "unanswered_required": progress.unanswered_required,
            "estimated_minutes_remaining": progress.estimated_minutes_remaining,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to get progress") from exc


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------


@router.post("/consistency/check/{discipline_id}")
def check_consistency(discipline_id: str) -> dict[str, Any]:
    """Run consistency checks on a discipline's examples.

    Args:
        discipline_id: The discipline to check.

    Returns:
        Consistency report dict.
    """
    try:
        if _consistency_checker is None:
            raise HTTPException(status_code=500, detail="Consistency checker not initialized")
        storage = get_storage()
        discipline = storage.get_discipline(discipline_id)
        if discipline is None:
            raise HTTPException(status_code=404, detail="Discipline not found")
        report = _consistency_checker.check_discipline(discipline_id)
        return {
            "discipline_id": report.discipline_id,
            "example_count": report.example_count,
            "has_errors": report.has_errors,
            "has_warnings": report.has_warnings,
            "issues": [issue.to_dict() for issue in report.issues],
            "checked_at": report.checked_at.isoformat(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to run consistency check") from exc


@router.get("/consistency/report/{discipline_id}")
def get_consistency_report(discipline_id: str) -> dict[str, Any]:
    """Get the latest consistency report for a discipline.

    This re-runs the check to provide a fresh report.

    Args:
        discipline_id: The discipline to report on.

    Returns:
        Consistency report dict.
    """
    return check_consistency(discipline_id)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


@router.get("/coverage/{discipline_id}")
def get_coverage(discipline_id: str) -> dict[str, Any]:
    """Get competency coverage report for a discipline.

    Args:
        discipline_id: The discipline to report on.

    Returns:
        Coverage report dict with per-competency breakdown.
    """
    try:
        storage = get_storage()
        discipline = storage.get_discipline(discipline_id)
        if discipline is None:
            raise HTTPException(status_code=404, detail="Discipline not found")
        report = storage.get_coverage_report(discipline_id)
        return report
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to get coverage report") from exc


# ---------------------------------------------------------------------------
# Curriculum Export
# ---------------------------------------------------------------------------


@router.post("/curriculum/export/{discipline_id}")
def export_curriculum(
    discipline_id: str,
    body: CurriculumExportRequest,
) -> dict[str, Any]:
    """Create a curriculum version snapshot and return export metadata.

    Creates a new curriculum version in the database. The caller can
    use the version information to trigger JSONL export separately.

    Args:
        discipline_id: The discipline to export.
        body: Export configuration.

    Returns:
        Curriculum version dict with example count and version info.
    """
    try:
        storage = get_storage()
        discipline = storage.get_discipline(discipline_id)
        if discipline is None:
            raise HTTPException(status_code=404, detail="Discipline not found")
        version = storage.create_curriculum_version(
            discipline_id=discipline_id,
            created_by=body.created_by,
        )
        return {
            "version_id": version.id,
            "discipline_id": version.discipline_id,
            "version_number": version.version_number,
            "example_count": version.example_count,
            "status": version.status.value,
            "created_at": version.created_at.isoformat(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to export curriculum") from exc
