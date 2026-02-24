"""Hearth inference engine -- model loading, query handling, and model switching.

Manages loaded models, routes queries through the RAG pipeline,
and provides a unified interface for the chat UI. Conversations are
tracked in-memory for the MVP; persistence can be added later.

Example::

    from foundry.src.evaluation import MockInference
    from foundry.src.rag_integration import (
        MockRetrievalAdapter, RAGConfig, RAGPipeline,
    )
    from hearth.src.inference import HearthEngine, ModelManager, QueryRequest

    model = MockInference(default_response="Answer based on context.")
    retrieval = MockRetrievalAdapter(chunks=[...])
    pipeline = RAGPipeline(model=model, retrieval=retrieval)

    manager = ModelManager()
    manager.register_slot("slot_1", "My Model", "phi")
    manager.load_model("slot_1")

    engine = HearthEngine(model_manager=manager, rag_pipeline=pipeline)
    response = engine.query(QueryRequest(query="How?", slot_id="slot_1"))
    print(response.answer)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from foundry.src.rag_integration import (
    CitationSource,
    RAGPipeline,
    RAGResponse,
)

# ===================================================================
# Exceptions
# ===================================================================


class InferenceError(Exception):
    """Raised for Hearth inference engine errors."""


# ===================================================================
# Enums
# ===================================================================


class ModelStatus(str, Enum):
    """Status of a model slot.

    Attributes:
        UNLOADED: Model is registered but not loaded.
        LOADING: Model is currently being loaded.
        READY: Model is loaded and ready for inference.
        ERROR: Model failed to load.
    """

    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class FeedbackType(str, Enum):
    """Types of feedback signals a user can submit.

    Attributes:
        THUMBS_UP: Positive quality signal.
        THUMBS_DOWN: Negative quality signal.
        FLAG_INCORRECT: Factually incorrect response.
        FLAG_INCOMPLETE: Missing important information.
    """

    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    FLAG_INCORRECT = "flag_incorrect"
    FLAG_INCOMPLETE = "flag_incomplete"


# ===================================================================
# Data classes
# ===================================================================


@dataclass
class ModelSlot:
    """A loaded or loadable model configuration.

    Attributes:
        slot_id: Unique identifier for this slot.
        display_name: Human-readable name for the UI.
        base_model_family: Model architecture family (phi, llama, etc.).
        status: Current loading status.
        model_path: Path to the base model weights (None if not loaded).
        discipline_id: Associated discipline (None if general).
        lora_path: Path to LoRA adapter weights (None if base model only).
        loaded_at: Timestamp when the model was loaded (None if unloaded).
    """

    slot_id: str
    display_name: str
    base_model_family: str
    status: ModelStatus
    model_path: str | None = None
    discipline_id: str | None = None
    lora_path: str | None = None
    loaded_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with all slot fields.
        """
        return {
            "slot_id": self.slot_id,
            "display_name": self.display_name,
            "base_model_family": self.base_model_family,
            "status": self.status.value,
            "model_path": self.model_path,
            "discipline_id": self.discipline_id,
            "lora_path": self.lora_path,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelSlot:
        """Deserialize from dictionary.

        Args:
            data: Dict with slot fields.

        Returns:
            ModelSlot instance.
        """
        loaded_at = data.get("loaded_at")
        return cls(
            slot_id=data["slot_id"],
            display_name=data["display_name"],
            base_model_family=data["base_model_family"],
            status=ModelStatus(data["status"]),
            model_path=data.get("model_path"),
            discipline_id=data.get("discipline_id"),
            lora_path=data.get("lora_path"),
            loaded_at=datetime.fromisoformat(loaded_at) if loaded_at else None,
        )


@dataclass
class QueryRequest:
    """A user query to the Hearth inference engine.

    Attributes:
        query: The user question text.
        slot_id: Which model slot to use for inference.
        conversation_id: Existing conversation to append to (None = new).
        max_context_chunks: Maximum retrieval chunks to include.
        include_citations: Whether to include citations in the response.
    """

    query: str
    slot_id: str
    conversation_id: str | None = None
    max_context_chunks: int = 5
    include_citations: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with all request fields.
        """
        return {
            "query": self.query,
            "slot_id": self.slot_id,
            "conversation_id": self.conversation_id,
            "max_context_chunks": self.max_context_chunks,
            "include_citations": self.include_citations,
        }


@dataclass
class CitationInfo:
    """Simplified citation for UI display.

    Attributes:
        document_title: Title of the source document.
        section: Section heading or path within the document.
        page: Page number, or None if unknown.
        relevance_score: How relevant this chunk was (0-1).
        snippet: Short excerpt from the source text.
    """

    document_title: str
    section: str
    page: int | None
    relevance_score: float
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with all citation fields.
        """
        return {
            "document_title": self.document_title,
            "section": self.section,
            "page": self.page,
            "relevance_score": self.relevance_score,
            "snippet": self.snippet,
        }

    def truncate_snippet(self, max_length: int = 200) -> str:
        """Return snippet truncated to max_length with ellipsis if needed.

        Args:
            max_length: Maximum character length for the snippet.

        Returns:
            Truncated snippet string.
        """
        if len(self.snippet) <= max_length:
            return self.snippet
        return self.snippet[:max_length] + "..."


@dataclass
class QueryResponse:
    """Response from the Hearth inference engine.

    Attributes:
        answer: The model-generated answer text.
        citations: Simplified citations for UI display.
        conversation_id: ID of the conversation this belongs to.
        model_used: Identifier for the model that generated the answer.
        latency_ms: End-to-end query latency in milliseconds.
        chunk_count: Number of retrieval chunks used.
    """

    answer: str
    citations: list[CitationInfo]
    conversation_id: str
    model_used: str
    latency_ms: float
    chunk_count: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with all response fields.
        """
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "conversation_id": self.conversation_id,
            "model_used": self.model_used,
            "latency_ms": self.latency_ms,
            "chunk_count": self.chunk_count,
        }


@dataclass
class ConversationTurn:
    """A single turn in a conversation.

    Attributes:
        turn_id: Unique identifier for this turn.
        query: The user question.
        response: The model-generated answer.
        citations: Citations for this turn.
        timestamp: When this turn occurred.
        model_slot_id: Which model slot was used.
    """

    turn_id: str
    query: str
    response: str
    citations: list[CitationInfo]
    timestamp: datetime
    model_slot_id: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with all turn fields.
        """
        return {
            "turn_id": self.turn_id,
            "query": self.query,
            "response": self.response,
            "citations": [c.to_dict() for c in self.citations],
            "timestamp": self.timestamp.isoformat(),
            "model_slot_id": self.model_slot_id,
        }


@dataclass
class Conversation:
    """A conversation session with history.

    Attributes:
        conversation_id: Unique identifier for the conversation.
        title: Auto-generated from the first query.
        turns: List of conversation turns in chronological order.
        created_at: When the conversation was started.
        model_slot_id: Primary model slot used in this conversation.
    """

    conversation_id: str
    title: str
    turns: list[ConversationTurn]
    created_at: datetime
    model_slot_id: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with all conversation fields.
        """
        return {
            "conversation_id": self.conversation_id,
            "title": self.title,
            "turns": [t.to_dict() for t in self.turns],
            "created_at": self.created_at.isoformat(),
            "model_slot_id": self.model_slot_id,
        }


@dataclass
class FeedbackSignal:
    """A feedback signal submitted by the user for a turn.

    Feedback is captured for routing to discipline owners.
    It is NOT used for automatic training data generation.

    Attributes:
        signal_id: Unique identifier for this feedback.
        conversation_id: Which conversation this feedback belongs to.
        turn_id: Which turn this feedback is about.
        feedback_type: Type of feedback signal.
        comment: Optional free-text comment.
        timestamp: When the feedback was submitted.
    """

    signal_id: str
    conversation_id: str
    turn_id: str
    feedback_type: FeedbackType
    comment: str | None
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with all feedback fields.
        """
        return {
            "signal_id": self.signal_id,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "feedback_type": self.feedback_type.value,
            "comment": self.comment,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DocumentSummary:
    """Summary of a document in the knowledge base.

    Attributes:
        document_id: Unique identifier for the document.
        title: Document title.
        document_type: Classification of the document type.
        chunk_count: Number of chunks extracted from this document.
        page_count: Number of pages, or None if unknown.
    """

    document_id: str
    title: str
    document_type: str
    chunk_count: int
    page_count: int | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with all summary fields.
        """
        return {
            "document_id": self.document_id,
            "title": self.title,
            "document_type": self.document_type,
            "chunk_count": self.chunk_count,
            "page_count": self.page_count,
        }


@dataclass
class DocumentDetail:
    """Detailed view of a knowledge base document.

    Attributes:
        document_id: Unique identifier for the document.
        title: Document title.
        document_type: Classification of the document type.
        chunks: List of chunk dicts with text and metadata.
        metadata: Additional document-level metadata.
    """

    document_id: str
    title: str
    document_type: str
    chunks: list[dict[str, Any]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with all detail fields.
        """
        return {
            "document_id": self.document_id,
            "title": self.title,
            "document_type": self.document_type,
            "chunks": list(self.chunks),
            "metadata": dict(self.metadata),
        }


# ===================================================================
# ModelManager
# ===================================================================


class ModelManager:
    """Manages model slots -- loading, unloading, switching.

    For MVP, load_model() simply marks the slot as READY without
    actually loading GPU weights. Production implementations would
    delegate to real model loading backends.

    Example::

        manager = ModelManager()
        manager.register_slot("s1", "Maintenance LoRA", "phi")
        manager.load_model("s1")
        slot = manager.get_slot("s1")
        assert slot.status == ModelStatus.READY
    """

    def __init__(self) -> None:
        """Initialize an empty model manager."""
        self._slots: dict[str, ModelSlot] = {}

    def register_slot(
        self,
        slot_id: str,
        display_name: str,
        base_model_family: str,
        discipline_id: str | None = None,
        model_path: str | None = None,
        lora_path: str | None = None,
    ) -> ModelSlot:
        """Register a new model slot.

        Args:
            slot_id: Unique identifier for the slot.
            display_name: Human-readable name.
            base_model_family: Model architecture family.
            discipline_id: Associated discipline (optional).
            model_path: Path to base model (optional).
            lora_path: Path to LoRA adapter (optional).

        Returns:
            The created ModelSlot.

        Raises:
            InferenceError: If slot_id is already registered.
        """
        if slot_id in self._slots:
            raise InferenceError(f"Slot '{slot_id}' already registered")
        slot = ModelSlot(
            slot_id=slot_id,
            display_name=display_name,
            base_model_family=base_model_family,
            status=ModelStatus.UNLOADED,
            discipline_id=discipline_id,
            model_path=model_path,
            lora_path=lora_path,
        )
        self._slots[slot_id] = slot
        return slot

    def load_model(self, slot_id: str) -> ModelSlot:
        """Load a model into a slot, setting status to READY.

        For MVP, this is a mock load that just updates status.
        Production would load actual model weights.

        Args:
            slot_id: The slot to load.

        Returns:
            The updated ModelSlot with READY status.

        Raises:
            InferenceError: If slot_id is not found.
        """
        slot = self._get_or_raise(slot_id)
        slot.status = ModelStatus.READY
        slot.loaded_at = datetime.now()
        return slot

    def unload_model(self, slot_id: str) -> None:
        """Unload a model from a slot, setting status to UNLOADED.

        Args:
            slot_id: The slot to unload.

        Raises:
            InferenceError: If slot_id is not found.
        """
        slot = self._get_or_raise(slot_id)
        slot.status = ModelStatus.UNLOADED
        slot.loaded_at = None

    def get_slot(self, slot_id: str) -> ModelSlot:
        """Retrieve a slot by ID.

        Args:
            slot_id: The slot identifier.

        Returns:
            The ModelSlot.

        Raises:
            InferenceError: If slot_id is not found.
        """
        return self._get_or_raise(slot_id)

    def list_slots(self) -> list[ModelSlot]:
        """Return all registered slots.

        Returns:
            List of ModelSlot instances.
        """
        return list(self._slots.values())

    def get_ready_slots(self) -> list[ModelSlot]:
        """Return only slots with READY status.

        Returns:
            List of ModelSlot instances that are ready for inference.
        """
        return [s for s in self._slots.values() if s.status == ModelStatus.READY]

    def remove_slot(self, slot_id: str) -> None:
        """Remove a slot from the manager entirely.

        Args:
            slot_id: The slot to remove.

        Raises:
            InferenceError: If slot_id is not found.
        """
        self._get_or_raise(slot_id)
        del self._slots[slot_id]

    def _get_or_raise(self, slot_id: str) -> ModelSlot:
        """Retrieve a slot or raise InferenceError.

        Args:
            slot_id: The slot identifier.

        Returns:
            The ModelSlot.

        Raises:
            InferenceError: If slot_id is not found.
        """
        slot = self._slots.get(slot_id)
        if slot is None:
            raise InferenceError(f"Slot '{slot_id}' not found")
        return slot


# ===================================================================
# HearthEngine
# ===================================================================


class HearthEngine:
    """Main Hearth inference engine.

    Wraps the RAG pipeline and model manager to provide a unified
    query interface with conversation tracking, citation building,
    and feedback capture.

    Args:
        model_manager: Manages model slot loading and switching.
        rag_pipeline: The RAG pipeline for retrieval-augmented generation.

    Example::

        engine = HearthEngine(model_manager=mgr, rag_pipeline=pipeline)
        response = engine.query(QueryRequest(query="How?", slot_id="s1"))
    """

    def __init__(
        self,
        model_manager: ModelManager,
        rag_pipeline: RAGPipeline,
    ) -> None:
        """Initialize the engine.

        Args:
            model_manager: The model manager instance.
            rag_pipeline: The RAG pipeline for query processing.
        """
        self._manager = model_manager
        self._pipeline = rag_pipeline
        self._conversations: dict[str, Conversation] = {}
        self._feedback: dict[str, list[FeedbackSignal]] = {}

    def query(self, request: QueryRequest) -> QueryResponse:
        """Execute a query through the RAG pipeline.

        Args:
            request: The query request with slot_id and query text.

        Returns:
            QueryResponse with answer, citations, and metadata.

        Raises:
            InferenceError: If the slot is not found or not ready.
        """
        self._validate_slot_ready(request.slot_id)
        start = time.monotonic()
        rag_response = self._pipeline.query(request.query)
        latency_ms = (time.monotonic() - start) * 1000

        citations = self._build_citations_for_request(rag_response, request)
        conversation_id = self._ensure_conversation(request)
        self._record_turn(conversation_id, request, rag_response, citations)

        return QueryResponse(
            answer=rag_response.answer,
            citations=citations,
            conversation_id=conversation_id,
            model_used=rag_response.model_name,
            latency_ms=latency_ms,
            chunk_count=len(rag_response.context_used),
        )

    def multi_discipline_query(
        self,
        query: str,
        slot_ids: list[str],
    ) -> list[QueryResponse]:
        """Query multiple discipline models with the same question.

        Runs the query sequentially through each slot. MVP does not
        support parallel execution.

        Args:
            query: The user question.
            slot_ids: List of slot IDs to query.

        Returns:
            List of QueryResponse instances, one per slot.

        Raises:
            InferenceError: If any slot is not ready.
        """
        responses: list[QueryResponse] = []
        for slot_id in slot_ids:
            request = QueryRequest(query=query, slot_id=slot_id)
            responses.append(self.query(request))
        return responses

    def get_conversation(self, conversation_id: str) -> Conversation:
        """Retrieve a conversation by ID.

        Args:
            conversation_id: The conversation identifier.

        Returns:
            The Conversation instance.

        Raises:
            InferenceError: If conversation is not found.
        """
        conv = self._conversations.get(conversation_id)
        if conv is None:
            raise InferenceError(f"Conversation '{conversation_id}' not found")
        return conv

    def list_conversations(self) -> list[Conversation]:
        """Return all conversations, newest first.

        Returns:
            List of Conversation instances.
        """
        convos = list(self._conversations.values())
        return sorted(convos, key=lambda c: c.created_at, reverse=True)

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and its associated feedback.

        Args:
            conversation_id: The conversation to delete.

        Raises:
            InferenceError: If conversation is not found.
        """
        if conversation_id not in self._conversations:
            raise InferenceError(f"Conversation '{conversation_id}' not found")
        del self._conversations[conversation_id]
        self._feedback.pop(conversation_id, None)

    def submit_feedback(
        self,
        conversation_id: str,
        turn_id: str,
        feedback_type: FeedbackType,
        comment: str | None = None,
    ) -> FeedbackSignal:
        """Submit a feedback signal for a conversation turn.

        Feedback is captured for routing to discipline owners.
        It is never used to auto-generate training data.

        Args:
            conversation_id: The conversation containing the turn.
            turn_id: The specific turn to provide feedback on.
            feedback_type: Type of feedback signal.
            comment: Optional free-text comment.

        Returns:
            The created FeedbackSignal.

        Raises:
            InferenceError: If conversation or turn is not found.
        """
        conv = self.get_conversation(conversation_id)
        self._validate_turn_exists(conv, turn_id)
        signal = FeedbackSignal(
            signal_id=f"fb_{uuid.uuid4().hex[:12]}",
            conversation_id=conversation_id,
            turn_id=turn_id,
            feedback_type=feedback_type,
            comment=comment,
            timestamp=datetime.now(),
        )
        self._feedback.setdefault(conversation_id, []).append(signal)
        return signal

    def get_feedback(self, conversation_id: str) -> list[FeedbackSignal]:
        """Retrieve all feedback signals for a conversation.

        Args:
            conversation_id: The conversation to get feedback for.

        Returns:
            List of FeedbackSignal instances.
        """
        return list(self._feedback.get(conversation_id, []))

    def _build_citations(self, rag_response: RAGResponse) -> list[CitationInfo]:
        """Convert RAG CitationSources to simplified CitationInfo for UI.

        Args:
            rag_response: The RAG pipeline response.

        Returns:
            List of CitationInfo instances.
        """
        return [
            _citation_source_to_info(cs, ctx)
            for cs, ctx in _zip_citations_context(rag_response.citations, rag_response.context_used)
        ]

    def _generate_title(self, first_query: str) -> str:
        """Generate a short conversation title from the first query.

        Truncates to a maximum of 80 characters with ellipsis.

        Args:
            first_query: The first query in the conversation.

        Returns:
            A short title string.
        """
        max_title_length = 80
        cleaned = first_query.strip()
        if not cleaned:
            return "New conversation"
        if len(cleaned) <= max_title_length:
            return cleaned
        return cleaned[: max_title_length - 3] + "..."

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _validate_slot_ready(self, slot_id: str) -> None:
        """Verify a slot exists and is READY.

        Args:
            slot_id: The slot to validate.

        Raises:
            InferenceError: If slot is not found or not ready.
        """
        slot = self._manager.get_slot(slot_id)
        if slot.status != ModelStatus.READY:
            raise InferenceError(f"Slot '{slot_id}' is not ready (status: {slot.status.value})")

    def _build_citations_for_request(
        self,
        rag_response: RAGResponse,
        request: QueryRequest,
    ) -> list[CitationInfo]:
        """Build citations if requested.

        Args:
            rag_response: The RAG pipeline response.
            request: The original query request.

        Returns:
            List of CitationInfo, or empty list if citations disabled.
        """
        if not request.include_citations:
            return []
        return self._build_citations(rag_response)

    def _ensure_conversation(self, request: QueryRequest) -> str:
        """Get existing or create new conversation ID.

        Args:
            request: The query request.

        Returns:
            The conversation_id to use.
        """
        if request.conversation_id and request.conversation_id in self._conversations:
            return request.conversation_id
        return self._create_conversation(request)

    def _create_conversation(self, request: QueryRequest) -> str:
        """Create a new conversation.

        Args:
            request: The query request that starts the conversation.

        Returns:
            The new conversation_id.
        """
        conv_id = f"conv_{uuid.uuid4().hex[:12]}"
        title = self._generate_title(request.query)
        conv = Conversation(
            conversation_id=conv_id,
            title=title,
            turns=[],
            created_at=datetime.now(),
            model_slot_id=request.slot_id,
        )
        self._conversations[conv_id] = conv
        return conv_id

    def _record_turn(
        self,
        conversation_id: str,
        request: QueryRequest,
        rag_response: RAGResponse,
        citations: list[CitationInfo],
    ) -> None:
        """Record a turn in the conversation history.

        Args:
            conversation_id: The conversation to append to.
            request: The original query request.
            rag_response: The RAG pipeline response.
            citations: The built citation list.
        """
        turn = ConversationTurn(
            turn_id=f"turn_{uuid.uuid4().hex[:12]}",
            query=request.query,
            response=rag_response.answer,
            citations=citations,
            timestamp=datetime.now(),
            model_slot_id=request.slot_id,
        )
        self._conversations[conversation_id].turns.append(turn)

    @staticmethod
    def _validate_turn_exists(conv: Conversation, turn_id: str) -> None:
        """Check that a turn exists within a conversation.

        Args:
            conv: The conversation to search.
            turn_id: The turn ID to find.

        Raises:
            InferenceError: If the turn is not found.
        """
        for turn in conv.turns:
            if turn.turn_id == turn_id:
                return
        raise InferenceError(f"Turn '{turn_id}' not found in conversation '{conv.conversation_id}'")


# ===================================================================
# DocumentBrowser
# ===================================================================


class DocumentBrowser:
    """Browse available knowledge base documents.

    Provides listing, retrieval, and search over configured documents.
    For MVP, documents are passed at construction time. Production would
    integrate with Quarry's document index.

    Args:
        documents: List of DocumentSummary for browsing.

    Example::

        browser = DocumentBrowser(documents=[...])
        docs = browser.list_documents()
        detail = browser.get_document("doc_001")
    """

    def __init__(self, documents: list[DocumentSummary] | None = None) -> None:
        """Initialize the browser with a list of documents.

        Args:
            documents: List of DocumentSummary instances.
        """
        self._documents: dict[str, DocumentSummary] = {}
        for doc in documents or []:
            self._documents[doc.document_id] = doc

    def list_documents(self) -> list[DocumentSummary]:
        """Return all available documents.

        Returns:
            List of DocumentSummary instances.
        """
        return list(self._documents.values())

    def get_document(self, document_id: str) -> DocumentDetail:
        """Retrieve detailed information about a document.

        For MVP, returns a DocumentDetail with empty chunks.
        Production would load actual chunk data from Quarry.

        Args:
            document_id: The document to retrieve.

        Returns:
            DocumentDetail with document information.

        Raises:
            InferenceError: If document is not found.
        """
        summary = self._documents.get(document_id)
        if summary is None:
            raise InferenceError(f"Document '{document_id}' not found")
        return DocumentDetail(
            document_id=summary.document_id,
            title=summary.title,
            document_type=summary.document_type,
            chunks=[],
            metadata={"chunk_count": summary.chunk_count, "page_count": summary.page_count},
        )

    def search_documents(
        self,
        query: str,
        limit: int = 10,
    ) -> list[DocumentSummary]:
        """Search documents by title (case-insensitive substring match).

        Args:
            query: Search query to match against document titles.
            limit: Maximum number of results to return.

        Returns:
            List of matching DocumentSummary instances.
        """
        query_lower = query.lower()
        matches = [doc for doc in self._documents.values() if query_lower in doc.title.lower()]
        return matches[:limit]


# ===================================================================
# Module-level helpers
# ===================================================================


def _citation_source_to_info(
    source: CitationSource,
    context_text: str,
) -> CitationInfo:
    """Convert a Foundry CitationSource to a Hearth CitationInfo.

    Args:
        source: The citation source from the RAG pipeline.
        context_text: The chunk text used for the snippet.

    Returns:
        CitationInfo with snippet extracted from context.
    """
    snippet = _extract_snippet(context_text)
    return CitationInfo(
        document_title=source.document_title,
        section=source.section,
        page=source.page,
        relevance_score=source.relevance_score,
        snippet=snippet,
    )


def _extract_snippet(text: str, max_length: int = 200) -> str:
    """Extract a snippet from chunk text, truncating if needed.

    Args:
        text: The full chunk text.
        max_length: Maximum snippet length.

    Returns:
        Truncated snippet string.
    """
    cleaned = text.strip()
    if not cleaned:
        return ""
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[:max_length] + "..."


def _zip_citations_context(
    citations: list[CitationSource],
    context_used: list[str],
) -> list[tuple[CitationSource, str]]:
    """Pair citations with their context text, padding if needed.

    If there are more citations than context entries, empty strings
    are used for the missing context.

    Args:
        citations: List of citation sources.
        context_used: List of context text strings.

    Returns:
        List of (CitationSource, context_text) tuples.
    """
    pairs: list[tuple[CitationSource, str]] = []
    for i, cs in enumerate(citations):
        ctx = context_used[i] if i < len(context_used) else ""
        pairs.append((cs, ctx))
    return pairs
