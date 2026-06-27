"""FastAPI router for the Hearth interaction layer.

Exposes REST endpoints for model management, query handling,
conversation tracking, document browsing, and feedback capture.
Designed to be mounted at ``/api/hearth/`` by the parent application.

Example::

    from fastapi import FastAPI
    from hearth.src.server import router

    app = FastAPI()
    app.include_router(router, prefix="/api/hearth")
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from hearth.src.feedback import (
    FeedbackManager,
    SignalType,
)
from hearth.src.inference import (
    DocumentBrowser,
    HearthEngine,
    InferenceError,
    ModelStatus,
)

logger = logging.getLogger(__name__)

# ===================================================================
# Pydantic request / response models
# ===================================================================


class RegisterModelRequest(BaseModel):
    """Request body for registering a new model slot."""

    slot_id: str = Field(..., min_length=1, max_length=200)
    display_name: str = Field(..., min_length=1, max_length=500)
    base_model_family: str = Field(..., min_length=1, max_length=100)
    discipline_id: str | None = None
    model_path: str | None = None
    lora_path: str | None = None


class QueryRequestBody(BaseModel):
    """Request body for a single-discipline query."""

    query: str = Field(..., min_length=1, max_length=10_000)
    slot_id: str = Field(..., min_length=1)
    conversation_id: str | None = None
    max_context_chunks: int = Field(default=5, ge=1, le=50)
    include_citations: bool = True


class MultiDisciplineQueryRequest(BaseModel):
    """Request body for querying multiple model slots at once."""

    query: str = Field(..., min_length=1, max_length=10_000)
    slot_ids: list[str] = Field(..., min_length=1)


class DocumentSearchRequest(BaseModel):
    """Request body for searching documents by title."""

    query: str = Field(..., min_length=1, max_length=1_000)
    limit: int = Field(default=10, ge=1, le=100)


class SubmitFeedbackRequest(BaseModel):
    """Request body for submitting a feedback signal."""

    signal_type: str = Field(..., min_length=1)
    conversation_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1, max_length=10_000)
    response: str | None = None
    user_comment: str | None = Field(default=None, max_length=5_000)
    discipline_id: str | None = None
    metadata: dict[str, Any] | None = None


class FeedbackRoutingQuery(BaseModel):
    """Query parameters for feedback routing lookup."""

    signal_id: str = Field(..., min_length=1)


class DashboardQuery(BaseModel):
    """Query parameters for the feedback dashboard."""

    discipline_id: str = Field(..., min_length=1)
    days: int = Field(default=30, ge=1, le=365)


class PatternsQuery(BaseModel):
    """Query parameters for pattern analysis."""

    discipline_id: str | None = None


# ===================================================================
# Shared state and factory
# ===================================================================

_state: dict[str, Any] = {
    "engine": None,
    "browser": None,
    "feedback_manager": None,
}


def get_engine() -> HearthEngine:
    """Return the HearthEngine singleton, raising 503 if not initialised.

    Returns:
        The current HearthEngine instance.

    Raises:
        HTTPException: 503 if the engine has not been initialised.
    """
    engine = _state.get("engine")
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Hearth engine not initialised. Call configure() first.",
        )
    return engine


def get_browser() -> DocumentBrowser:
    """Return the DocumentBrowser singleton, raising 503 if not initialised.

    Returns:
        The current DocumentBrowser instance.

    Raises:
        HTTPException: 503 if the browser has not been initialised.
    """
    browser = _state.get("browser")
    if browser is None:
        raise HTTPException(
            status_code=503,
            detail="Document browser not initialised.",
        )
    return browser


def get_feedback_manager() -> FeedbackManager:
    """Return the FeedbackManager singleton, raising 503 if not initialised.

    Returns:
        The current FeedbackManager instance.

    Raises:
        HTTPException: 503 if the feedback manager has not been initialised.
    """
    manager = _state.get("feedback_manager")
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="Feedback manager not initialised.",
        )
    return manager


def configure(
    engine: HearthEngine,
    browser: DocumentBrowser | None = None,
    feedback_manager: FeedbackManager | None = None,
) -> None:
    """Inject dependencies into the module-level state.

    Must be called before the router handles any requests.

    Args:
        engine: A fully-constructed HearthEngine.
        browser: Optional DocumentBrowser (defaults to empty browser).
        feedback_manager: Optional FeedbackManager (defaults to new instance).
    """
    _state["engine"] = engine
    _state["browser"] = browser or DocumentBrowser()
    _state["feedback_manager"] = feedback_manager or FeedbackManager()


# ===================================================================
# Router
# ===================================================================

router = APIRouter()


# -------------------------------------------------------------------
# Health
# -------------------------------------------------------------------


@router.get("/health")
async def health() -> dict[str, Any]:
    """Return Hearth service health status.

    Returns:
        Dictionary with status, version, and component readiness.
    """
    engine_ready = _state.get("engine") is not None
    browser_ready = _state.get("browser") is not None
    feedback_ready = _state.get("feedback_manager") is not None
    return {
        "status": "ok" if engine_ready else "not_configured",
        "version": "0.1.0",
        "components": {
            "engine": engine_ready,
            "browser": browser_ready,
            "feedback": feedback_ready,
        },
    }


# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """List all registered model slots.

    Returns:
        Dictionary with a list of model slot dicts.
    """
    try:
        engine = get_engine()
        slots = engine._manager.list_slots()
        return {"models": [s.to_dict() for s in slots]}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to list models")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.post("/models/register", status_code=201)
async def register_model(request: RegisterModelRequest) -> dict[str, Any]:
    """Register a new model slot.

    Args:
        request: Registration details including slot_id and display_name.

    Returns:
        The created model slot as a dictionary.
    """
    try:
        engine = get_engine()
        slot = engine._manager.register_slot(
            slot_id=request.slot_id,
            display_name=request.display_name,
            base_model_family=request.base_model_family,
            discipline_id=request.discipline_id,
            model_path=request.model_path,
            lora_path=request.lora_path,
        )
        return slot.to_dict()
    except InferenceError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to register model")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.post("/models/{model_id}/load")
async def load_model(model_id: str) -> dict[str, Any]:
    """Load a model into a slot, setting its status to READY.

    Args:
        model_id: The slot_id to load.

    Returns:
        The updated model slot dictionary.
    """
    try:
        engine = get_engine()
        slot = engine._manager.load_model(model_id)
        return slot.to_dict()
    except InferenceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to load model %s", model_id)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.post("/models/{model_id}/unload")
async def unload_model(model_id: str) -> dict[str, Any]:
    """Unload a model from a slot, setting its status to UNLOADED.

    Args:
        model_id: The slot_id to unload.

    Returns:
        Confirmation with the model_id and new status.
    """
    try:
        engine = get_engine()
        engine._manager.unload_model(model_id)
        return {"model_id": model_id, "status": ModelStatus.UNLOADED.value}
    except InferenceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to unload model %s", model_id)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.get("/models/{model_id}/status")
async def model_status(model_id: str) -> dict[str, Any]:
    """Return the current status of a model slot.

    Args:
        model_id: The slot_id to query.

    Returns:
        Dictionary with model_id and current status string.
    """
    try:
        engine = get_engine()
        slot = engine._manager.get_slot(model_id)
        return slot.to_dict()
    except InferenceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get status for model %s", model_id)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


# -------------------------------------------------------------------
# Query
# -------------------------------------------------------------------


@router.post("/query")
async def query(request: QueryRequestBody) -> dict[str, Any]:
    """Execute a single-discipline query through the RAG pipeline.

    Args:
        request: Query text, target slot_id, and optional parameters.

    Returns:
        QueryResponse dictionary with answer, citations, and metadata.
    """
    try:
        engine = get_engine()
        from hearth.src.inference import QueryRequest

        qr = QueryRequest(
            query=request.query,
            slot_id=request.slot_id,
            conversation_id=request.conversation_id,
            max_context_chunks=request.max_context_chunks,
            include_citations=request.include_citations,
        )
        response = engine.query(qr)
        return response.to_dict()
    except InferenceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.post("/query/multi-discipline")
async def multi_discipline_query(
    request: MultiDisciplineQueryRequest,
) -> dict[str, Any]:
    """Query multiple discipline models with the same question.

    Args:
        request: Query text and list of slot_ids.

    Returns:
        Dictionary with a list of QueryResponse dicts, one per slot.
    """
    try:
        engine = get_engine()
        responses = engine.multi_discipline_query(
            query=request.query,
            slot_ids=request.slot_ids,
        )
        return {"responses": [r.to_dict() for r in responses]}
    except InferenceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Multi-discipline query failed")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


# -------------------------------------------------------------------
# Conversations
# -------------------------------------------------------------------


@router.get("/conversations")
async def list_conversations() -> dict[str, Any]:
    """List all conversations, newest first.

    Returns:
        Dictionary with a list of conversation summary dicts.
    """
    try:
        engine = get_engine()
        convos = engine.list_conversations()
        return {
            "conversations": [
                {
                    "conversation_id": c.conversation_id,
                    "title": c.title,
                    "created_at": c.created_at.isoformat(),
                    "model_slot_id": c.model_slot_id,
                    "turn_count": len(c.turns),
                }
                for c in convos
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to list conversations")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str) -> dict[str, Any]:
    """Retrieve a full conversation with all turns.

    Args:
        conversation_id: The conversation to retrieve.

    Returns:
        Full Conversation dictionary.
    """
    try:
        engine = get_engine()
        conv = engine.get_conversation(conversation_id)
        return conv.to_dict()
    except InferenceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get conversation %s", conversation_id)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str) -> dict[str, Any]:
    """Delete a conversation and its associated feedback.

    Args:
        conversation_id: The conversation to delete.

    Returns:
        Confirmation with the deleted conversation_id.
    """
    try:
        engine = get_engine()
        engine.delete_conversation(conversation_id)
        return {"deleted": conversation_id}
    except InferenceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to delete conversation %s", conversation_id)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


# -------------------------------------------------------------------
# Documents
# -------------------------------------------------------------------


@router.get("/documents")
async def list_documents() -> dict[str, Any]:
    """List all documents in the knowledge base.

    Returns:
        Dictionary with a list of document summary dicts.
    """
    try:
        browser = get_browser()
        docs = browser.list_documents()
        return {"documents": [d.to_dict() for d in docs]}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to list documents")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.get("/documents/{document_id}")
async def get_document(document_id: str) -> dict[str, Any]:
    """Retrieve detailed information about a document.

    Args:
        document_id: The document to retrieve.

    Returns:
        DocumentDetail dictionary.
    """
    try:
        browser = get_browser()
        detail = browser.get_document(document_id)
        return detail.to_dict()
    except InferenceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get document %s", document_id)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.post("/documents/search")
async def search_documents(request: DocumentSearchRequest) -> dict[str, Any]:
    """Search documents by title (case-insensitive substring match).

    Args:
        request: Search query and result limit.

    Returns:
        Dictionary with matching document summaries.
    """
    try:
        browser = get_browser()
        results = browser.search_documents(
            query=request.query,
            limit=request.limit,
        )
        return {"results": [d.to_dict() for d in results]}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Document search failed")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


# -------------------------------------------------------------------
# Feedback
# -------------------------------------------------------------------


@router.post("/feedback", status_code=201)
async def submit_feedback(request: SubmitFeedbackRequest) -> dict[str, Any]:
    """Submit a feedback signal for a user interaction.

    Feedback is captured for routing to discipline owners.
    It is NEVER used to auto-generate training data.

    Args:
        request: Feedback details including signal type, conversation, and query.

    Returns:
        The recorded feedback signal as a dictionary.
    """
    try:
        manager = get_feedback_manager()
        signal_type = SignalType(request.signal_type)
        signal = manager.capture(
            signal_type=signal_type,
            conversation_id=request.conversation_id,
            query=request.query,
            response=request.response,
            user_comment=request.user_comment,
            discipline_id=request.discipline_id,
            metadata=request.metadata,
        )
        return signal.to_dict()
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid signal_type: {request.signal_type}",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to submit feedback")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.get("/feedback/routing")
async def get_feedback_routing(signal_id: str) -> dict[str, Any]:
    """Get the routing decision for a specific feedback signal.

    Args:
        signal_id: The signal ID to route.

    Returns:
        RoutingDecision dictionary.
    """
    try:
        manager = get_feedback_manager()
        decision = manager.get_routing(signal_id)
        return decision.to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        error_msg = str(exc)
        if "not found" in error_msg.lower():
            raise HTTPException(status_code=404, detail=error_msg) from exc
        logger.exception("Failed to get routing for signal %s", signal_id)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.get("/feedback/dashboard")
async def get_feedback_dashboard(
    discipline_id: str,
    days: int = 30,
) -> dict[str, Any]:
    """Generate a feedback dashboard for a discipline.

    All dashboard data is informational. Actions listed are
    SUGGESTIONS for human discipline owners, never automated.

    Args:
        discipline_id: Which discipline to summarise.
        days: Number of days to look back (default 30).

    Returns:
        DashboardSummary dictionary.
    """
    try:
        manager = get_feedback_manager()
        dashboard = manager.get_dashboard(
            discipline_id=discipline_id,
            days=days,
        )
        return dashboard.to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to generate dashboard for %s", discipline_id)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@router.get("/feedback/patterns")
async def get_feedback_patterns(
    discipline_id: str | None = None,
) -> dict[str, Any]:
    """Detect feedback patterns, optionally filtered by discipline.

    Patterns are informational -- they describe issues for human
    review, never auto-fix anything.

    Args:
        discipline_id: Optional discipline filter.

    Returns:
        Dictionary with a list of detected pattern dicts.
    """
    try:
        manager = get_feedback_manager()
        patterns = manager.get_patterns(discipline_id=discipline_id)
        return {"patterns": [p.to_dict() for p in patterns]}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get feedback patterns")
        raise HTTPException(status_code=500, detail="Internal server error") from exc
