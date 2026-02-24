"""Tests for the Hearth inference engine.

Covers model management, query handling, conversation tracking,
citation building, multi-discipline queries, document browsing,
and feedback signal capture.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from foundry.src.evaluation import MockInference
from foundry.src.rag_integration import (
    CitationSource,
    MockRetrievalAdapter,
    RAGPipeline,
    RAGResponse,
)
from hearth.src.inference import (
    CitationInfo,
    Conversation,
    ConversationTurn,
    DocumentBrowser,
    DocumentDetail,
    DocumentSummary,
    FeedbackSignal,
    FeedbackType,
    HearthEngine,
    InferenceError,
    ModelManager,
    ModelSlot,
    ModelStatus,
    QueryRequest,
    QueryResponse,
)

# ===================================================================
# Helpers
# ===================================================================


def _make_chunks(count: int = 3) -> list[dict]:
    """Build mock retrieval chunks for testing.

    Args:
        count: Number of chunks to generate.

    Returns:
        List of chunk dicts with text, metadata, and score.
    """
    chunks = []
    for i in range(count):
        chunks.append(
            {
                "text": f"Chunk {i} content about filter replacement.",
                "metadata": {
                    "chunk_id": f"chunk_{i:03d}",
                    "document_title": f"TM-{i:03d} Maintenance Manual",
                    "section": f"Section {i + 1}.{i + 2}",
                    "page": (i + 1) * 10,
                },
                "score": round(0.95 - i * 0.1, 2),
            }
        )
    return chunks


def _make_retrieval_adapter(count: int = 3) -> MockRetrievalAdapter:
    """Create a MockRetrievalAdapter with test chunks.

    Args:
        count: Number of chunks to configure.

    Returns:
        A configured MockRetrievalAdapter.
    """
    return MockRetrievalAdapter(chunks=_make_chunks(count))


def _make_rag_pipeline(
    response_text: str = "The filter should be replaced every 500 hours.",
    chunk_count: int = 3,
) -> RAGPipeline:
    """Create a RAGPipeline with mock inference and retrieval.

    Args:
        response_text: Default response from the mock model.
        chunk_count: Number of chunks the retrieval adapter returns.

    Returns:
        A configured RAGPipeline.
    """
    model = MockInference(default_response=response_text)
    retrieval = _make_retrieval_adapter(chunk_count)
    return RAGPipeline(
        model=model,
        retrieval=retrieval,
        model_name="test-model",
    )


def _make_engine(
    response_text: str = "The filter should be replaced every 500 hours.",
    chunk_count: int = 3,
) -> HearthEngine:
    """Create a HearthEngine with a pre-configured model slot.

    Args:
        response_text: Default response for the mock model.
        chunk_count: Number of chunks from retrieval.

    Returns:
        A HearthEngine with one ready slot named 'slot_1'.
    """
    pipeline = _make_rag_pipeline(response_text, chunk_count)
    manager = ModelManager()
    manager.register_slot(
        slot_id="slot_1",
        display_name="Test Model",
        base_model_family="phi",
        discipline_id="disc_maint",
    )
    manager.load_model("slot_1")
    engine = HearthEngine(model_manager=manager, rag_pipeline=pipeline)
    return engine


# ===================================================================
# TestModelStatus
# ===================================================================


class TestModelStatus:
    """Tests for the ModelStatus enum."""

    def test_values_exist(self) -> None:
        """All expected status values are defined."""
        assert ModelStatus.UNLOADED == "unloaded"
        assert ModelStatus.LOADING == "loading"
        assert ModelStatus.READY == "ready"
        assert ModelStatus.ERROR == "error"

    def test_from_string(self) -> None:
        """ModelStatus can be constructed from a string value."""
        assert ModelStatus("ready") == ModelStatus.READY
        assert ModelStatus("unloaded") == ModelStatus.UNLOADED


# ===================================================================
# TestModelSlot
# ===================================================================


class TestModelSlot:
    """Tests for the ModelSlot dataclass."""

    def test_construction_defaults(self) -> None:
        """ModelSlot initializes with expected defaults."""
        slot = ModelSlot(
            slot_id="slot_1",
            display_name="Test Model",
            base_model_family="phi",
            status=ModelStatus.UNLOADED,
        )
        assert slot.slot_id == "slot_1"
        assert slot.display_name == "Test Model"
        assert slot.model_path is None
        assert slot.base_model_family == "phi"
        assert slot.discipline_id is None
        assert slot.status == ModelStatus.UNLOADED
        assert slot.lora_path is None
        assert slot.loaded_at is None

    def test_construction_full(self) -> None:
        """ModelSlot accepts all fields explicitly."""
        now = datetime.now()
        slot = ModelSlot(
            slot_id="slot_2",
            display_name="Maint LoRA",
            model_path="/models/phi-3",
            base_model_family="phi",
            discipline_id="disc_maint",
            status=ModelStatus.READY,
            lora_path="/loras/maint-v1",
            loaded_at=now,
        )
        assert slot.model_path == "/models/phi-3"
        assert slot.discipline_id == "disc_maint"
        assert slot.lora_path == "/loras/maint-v1"
        assert slot.loaded_at == now

    def test_to_dict(self) -> None:
        """ModelSlot serializes to a dictionary."""
        slot = ModelSlot(
            slot_id="slot_1",
            display_name="Test",
            base_model_family="phi",
            status=ModelStatus.UNLOADED,
        )
        data = slot.to_dict()
        assert data["slot_id"] == "slot_1"
        assert data["status"] == "unloaded"
        assert data["model_path"] is None

    def test_to_dict_roundtrip(self) -> None:
        """ModelSlot can be reconstructed from its dict representation."""
        now = datetime.now()
        slot = ModelSlot(
            slot_id="s1",
            display_name="Test",
            model_path="/m",
            base_model_family="llama",
            discipline_id="d1",
            status=ModelStatus.READY,
            lora_path="/l",
            loaded_at=now,
        )
        data = slot.to_dict()
        restored = ModelSlot.from_dict(data)
        assert restored.slot_id == slot.slot_id
        assert restored.display_name == slot.display_name
        assert restored.status == slot.status


# ===================================================================
# TestModelManager
# ===================================================================


class TestModelManager:
    """Tests for the ModelManager class."""

    def test_register_slot(self) -> None:
        """register_slot creates an unloaded slot."""
        mgr = ModelManager()
        slot = mgr.register_slot(
            slot_id="slot_1",
            display_name="Test Model",
            base_model_family="phi",
        )
        assert slot.slot_id == "slot_1"
        assert slot.status == ModelStatus.UNLOADED

    def test_register_duplicate_raises(self) -> None:
        """Registering a slot with an existing ID raises InferenceError."""
        mgr = ModelManager()
        mgr.register_slot(slot_id="s1", display_name="A", base_model_family="phi")
        with pytest.raises(InferenceError, match="already registered"):
            mgr.register_slot(slot_id="s1", display_name="B", base_model_family="phi")

    def test_load_model(self) -> None:
        """load_model transitions slot to READY."""
        mgr = ModelManager()
        mgr.register_slot(slot_id="s1", display_name="A", base_model_family="phi")
        slot = mgr.load_model("s1")
        assert slot.status == ModelStatus.READY
        assert slot.loaded_at is not None

    def test_load_unknown_slot_raises(self) -> None:
        """Loading a non-existent slot raises InferenceError."""
        mgr = ModelManager()
        with pytest.raises(InferenceError, match="not found"):
            mgr.load_model("nonexistent")

    def test_unload_model(self) -> None:
        """unload_model transitions slot back to UNLOADED."""
        mgr = ModelManager()
        mgr.register_slot(slot_id="s1", display_name="A", base_model_family="phi")
        mgr.load_model("s1")
        mgr.unload_model("s1")
        slot = mgr.get_slot("s1")
        assert slot.status == ModelStatus.UNLOADED
        assert slot.loaded_at is None

    def test_unload_unknown_raises(self) -> None:
        """Unloading a non-existent slot raises InferenceError."""
        mgr = ModelManager()
        with pytest.raises(InferenceError, match="not found"):
            mgr.unload_model("ghost")

    def test_get_slot(self) -> None:
        """get_slot returns the correct slot."""
        mgr = ModelManager()
        mgr.register_slot(slot_id="s1", display_name="A", base_model_family="phi")
        slot = mgr.get_slot("s1")
        assert slot.slot_id == "s1"

    def test_get_slot_missing_raises(self) -> None:
        """get_slot raises InferenceError for unknown slot_id."""
        mgr = ModelManager()
        with pytest.raises(InferenceError, match="not found"):
            mgr.get_slot("missing")

    def test_list_slots(self) -> None:
        """list_slots returns all registered slots."""
        mgr = ModelManager()
        mgr.register_slot(slot_id="s1", display_name="A", base_model_family="phi")
        mgr.register_slot(slot_id="s2", display_name="B", base_model_family="llama")
        slots = mgr.list_slots()
        assert len(slots) == 2
        ids = {s.slot_id for s in slots}
        assert ids == {"s1", "s2"}

    def test_list_slots_empty(self) -> None:
        """list_slots returns empty list when no slots registered."""
        mgr = ModelManager()
        assert mgr.list_slots() == []

    def test_get_ready_slots(self) -> None:
        """get_ready_slots filters to only READY slots."""
        mgr = ModelManager()
        mgr.register_slot(slot_id="s1", display_name="A", base_model_family="phi")
        mgr.register_slot(slot_id="s2", display_name="B", base_model_family="llama")
        mgr.load_model("s1")
        ready = mgr.get_ready_slots()
        assert len(ready) == 1
        assert ready[0].slot_id == "s1"

    def test_remove_slot(self) -> None:
        """remove_slot removes a slot from the manager."""
        mgr = ModelManager()
        mgr.register_slot(slot_id="s1", display_name="A", base_model_family="phi")
        mgr.remove_slot("s1")
        assert mgr.list_slots() == []

    def test_remove_slot_missing_raises(self) -> None:
        """remove_slot raises InferenceError for unknown slot_id."""
        mgr = ModelManager()
        with pytest.raises(InferenceError, match="not found"):
            mgr.remove_slot("ghost")


# ===================================================================
# TestQueryRequest
# ===================================================================


class TestQueryRequest:
    """Tests for the QueryRequest dataclass."""

    def test_construction_defaults(self) -> None:
        """QueryRequest initializes with expected defaults."""
        req = QueryRequest(query="How to replace filter?", slot_id="s1")
        assert req.query == "How to replace filter?"
        assert req.slot_id == "s1"
        assert req.conversation_id is None
        assert req.max_context_chunks == 5
        assert req.include_citations is True

    def test_construction_overrides(self) -> None:
        """QueryRequest accepts overridden values."""
        req = QueryRequest(
            query="test",
            slot_id="s2",
            conversation_id="conv_1",
            max_context_chunks=3,
            include_citations=False,
        )
        assert req.conversation_id == "conv_1"
        assert req.max_context_chunks == 3
        assert req.include_citations is False

    def test_to_dict(self) -> None:
        """QueryRequest serializes to dictionary."""
        req = QueryRequest(query="q", slot_id="s1")
        data = req.to_dict()
        assert data["query"] == "q"
        assert data["slot_id"] == "s1"


# ===================================================================
# TestQueryResponse
# ===================================================================


class TestQueryResponse:
    """Tests for the QueryResponse dataclass."""

    def test_construction(self) -> None:
        """QueryResponse initializes with all fields."""
        citation = CitationInfo(
            document_title="TM-001",
            section="Section 3.1",
            page=42,
            relevance_score=0.9,
            snippet="Replace the filter...",
        )
        resp = QueryResponse(
            answer="Replace the filter every 500 hours.",
            citations=[citation],
            conversation_id="conv_1",
            model_used="test-model",
            latency_ms=150.0,
            chunk_count=3,
        )
        assert resp.answer == "Replace the filter every 500 hours."
        assert len(resp.citations) == 1
        assert resp.latency_ms == 150.0

    def test_to_dict(self) -> None:
        """QueryResponse serializes to dictionary including citations."""
        resp = QueryResponse(
            answer="answer",
            citations=[],
            conversation_id="c1",
            model_used="m1",
            latency_ms=100.0,
            chunk_count=0,
        )
        data = resp.to_dict()
        assert data["answer"] == "answer"
        assert data["citations"] == []
        assert data["model_used"] == "m1"


# ===================================================================
# TestCitationInfo
# ===================================================================


class TestCitationInfo:
    """Tests for the CitationInfo dataclass."""

    def test_construction(self) -> None:
        """CitationInfo initializes with all fields."""
        cite = CitationInfo(
            document_title="TM-001",
            section="Section 1.2",
            page=10,
            relevance_score=0.88,
            snippet="The oil filter assembly...",
        )
        assert cite.document_title == "TM-001"
        assert cite.page == 10
        assert cite.relevance_score == 0.88

    def test_page_none(self) -> None:
        """CitationInfo supports None for page."""
        cite = CitationInfo(
            document_title="Manual",
            section="Intro",
            page=None,
            relevance_score=0.5,
            snippet="Overview of...",
        )
        assert cite.page is None

    def test_to_dict(self) -> None:
        """CitationInfo serializes to dictionary."""
        cite = CitationInfo(
            document_title="TM-001",
            section="S1",
            page=5,
            relevance_score=0.7,
            snippet="text",
        )
        data = cite.to_dict()
        assert data["document_title"] == "TM-001"
        assert data["page"] == 5
        assert data["snippet"] == "text"

    def test_snippet_truncation(self) -> None:
        """Long snippets are truncated by truncate_snippet."""
        cite = CitationInfo(
            document_title="TM",
            section="S",
            page=1,
            relevance_score=0.5,
            snippet="x" * 500,
        )
        truncated = cite.truncate_snippet(max_length=200)
        assert len(truncated) <= 203  # 200 + '...'
        assert truncated.endswith("...")


# ===================================================================
# TestConversation
# ===================================================================


class TestConversation:
    """Tests for the Conversation and ConversationTurn dataclasses."""

    def test_conversation_construction(self) -> None:
        """Conversation initializes with expected fields."""
        conv = Conversation(
            conversation_id="conv_1",
            title="Filter replacement",
            turns=[],
            created_at=datetime.now(),
            model_slot_id="s1",
        )
        assert conv.conversation_id == "conv_1"
        assert conv.title == "Filter replacement"
        assert conv.turns == []

    def test_conversation_turn_construction(self) -> None:
        """ConversationTurn initializes with all fields."""
        turn = ConversationTurn(
            turn_id="t1",
            query="How?",
            response="Like this.",
            citations=[],
            timestamp=datetime.now(),
            model_slot_id="s1",
        )
        assert turn.turn_id == "t1"
        assert turn.query == "How?"

    def test_conversation_to_dict(self) -> None:
        """Conversation serializes to dictionary."""
        conv = Conversation(
            conversation_id="c1",
            title="Test",
            turns=[],
            created_at=datetime(2026, 1, 1),
            model_slot_id="s1",
        )
        data = conv.to_dict()
        assert data["conversation_id"] == "c1"
        assert data["model_slot_id"] == "s1"

    def test_turn_to_dict(self) -> None:
        """ConversationTurn serializes to dictionary."""
        turn = ConversationTurn(
            turn_id="t1",
            query="q",
            response="r",
            citations=[],
            timestamp=datetime(2026, 1, 1),
            model_slot_id="s1",
        )
        data = turn.to_dict()
        assert data["turn_id"] == "t1"
        assert data["query"] == "q"


# ===================================================================
# TestHearthEngine
# ===================================================================


class TestHearthEngine:
    """Tests for the HearthEngine class."""

    def test_query_returns_response(self) -> None:
        """query() returns a QueryResponse with answer and citations."""
        engine = _make_engine()
        request = QueryRequest(query="How to replace filter?", slot_id="slot_1")
        response = engine.query(request)
        assert isinstance(response, QueryResponse)
        assert response.answer != ""
        assert response.conversation_id != ""

    def test_query_creates_conversation(self) -> None:
        """query() with no conversation_id creates a new conversation."""
        engine = _make_engine()
        request = QueryRequest(query="Test question", slot_id="slot_1")
        response = engine.query(request)
        conv = engine.get_conversation(response.conversation_id)
        assert conv is not None
        assert len(conv.turns) == 1

    def test_query_appends_to_conversation(self) -> None:
        """query() with existing conversation_id appends a turn."""
        engine = _make_engine()
        req1 = QueryRequest(query="First question", slot_id="slot_1")
        resp1 = engine.query(req1)

        req2 = QueryRequest(
            query="Follow up",
            slot_id="slot_1",
            conversation_id=resp1.conversation_id,
        )
        engine.query(req2)

        conv = engine.get_conversation(resp1.conversation_id)
        assert len(conv.turns) == 2

    def test_query_with_unloaded_slot_raises(self) -> None:
        """query() with an unloaded model raises InferenceError."""
        pipeline = _make_rag_pipeline()
        manager = ModelManager()
        manager.register_slot(
            slot_id="s1",
            display_name="Not loaded",
            base_model_family="phi",
        )
        engine = HearthEngine(model_manager=manager, rag_pipeline=pipeline)
        request = QueryRequest(query="test", slot_id="s1")
        with pytest.raises(InferenceError, match="not ready"):
            engine.query(request)

    def test_query_with_unknown_slot_raises(self) -> None:
        """query() with a non-existent slot raises InferenceError."""
        engine = _make_engine()
        request = QueryRequest(query="test", slot_id="nonexistent")
        with pytest.raises(InferenceError):
            engine.query(request)

    def test_query_includes_citations(self) -> None:
        """query() returns CitationInfo objects from RAG pipeline."""
        engine = _make_engine(chunk_count=2)
        request = QueryRequest(query="test", slot_id="slot_1")
        response = engine.query(request)
        assert len(response.citations) >= 1
        assert all(isinstance(c, CitationInfo) for c in response.citations)

    def test_query_no_citations_when_disabled(self) -> None:
        """query() returns empty citations when include_citations is False."""
        engine = _make_engine()
        request = QueryRequest(
            query="test",
            slot_id="slot_1",
            include_citations=False,
        )
        response = engine.query(request)
        assert response.citations == []

    def test_query_latency_positive(self) -> None:
        """query() reports positive latency."""
        engine = _make_engine()
        request = QueryRequest(query="test", slot_id="slot_1")
        response = engine.query(request)
        assert response.latency_ms >= 0.0

    def test_get_conversation_missing_raises(self) -> None:
        """get_conversation with unknown ID raises InferenceError."""
        engine = _make_engine()
        with pytest.raises(InferenceError, match="not found"):
            engine.get_conversation("nonexistent")

    def test_list_conversations_empty(self) -> None:
        """list_conversations returns empty list initially."""
        pipeline = _make_rag_pipeline()
        manager = ModelManager()
        engine = HearthEngine(model_manager=manager, rag_pipeline=pipeline)
        assert engine.list_conversations() == []

    def test_list_conversations_after_queries(self) -> None:
        """list_conversations returns conversations after queries."""
        engine = _make_engine()
        req1 = QueryRequest(query="q1", slot_id="slot_1")
        req2 = QueryRequest(query="q2", slot_id="slot_1")
        engine.query(req1)
        engine.query(req2)
        convos = engine.list_conversations()
        assert len(convos) == 2

    def test_delete_conversation(self) -> None:
        """delete_conversation removes a conversation."""
        engine = _make_engine()
        request = QueryRequest(query="test", slot_id="slot_1")
        response = engine.query(request)
        engine.delete_conversation(response.conversation_id)
        assert len(engine.list_conversations()) == 0

    def test_delete_conversation_missing_raises(self) -> None:
        """delete_conversation with unknown ID raises InferenceError."""
        engine = _make_engine()
        with pytest.raises(InferenceError, match="not found"):
            engine.delete_conversation("ghost")

    def test_generate_title(self) -> None:
        """_generate_title produces a short title from the query."""
        engine = _make_engine()
        title = engine._generate_title("How do I replace the oil filter on the M1A2 Abrams?")
        assert len(title) > 0
        assert len(title) <= 80

    def test_generate_title_short_query(self) -> None:
        """_generate_title works with very short queries."""
        engine = _make_engine()
        title = engine._generate_title("Hi")
        assert len(title) > 0

    def test_build_citations(self) -> None:
        """_build_citations converts RAG CitationSources to CitationInfo."""
        engine = _make_engine()
        rag_response = RAGResponse(
            query="test",
            answer="answer",
            citations=[
                CitationSource(
                    chunk_id="c1",
                    document_title="TM-001",
                    section="S1",
                    page=10,
                    relevance_score=0.9,
                ),
            ],
            context_used=["chunk text"],
            retrieval_time_ms=10.0,
            generation_time_ms=20.0,
            total_time_ms=30.0,
            model_name="test",
        )
        citations = engine._build_citations(rag_response)
        assert len(citations) == 1
        assert citations[0].document_title == "TM-001"
        assert citations[0].page == 10


# ===================================================================
# TestMultiDisciplineQuery
# ===================================================================


class TestMultiDisciplineQuery:
    """Tests for multi-discipline dual-query mode."""

    def test_multi_discipline_query_returns_list(self) -> None:
        """multi_discipline_query returns a list of QueryResponse."""
        pipeline = _make_rag_pipeline()
        manager = ModelManager()
        manager.register_slot(
            slot_id="s1",
            display_name="Model A",
            base_model_family="phi",
            discipline_id="disc_a",
        )
        manager.register_slot(
            slot_id="s2",
            display_name="Model B",
            base_model_family="llama",
            discipline_id="disc_b",
        )
        manager.load_model("s1")
        manager.load_model("s2")
        engine = HearthEngine(model_manager=manager, rag_pipeline=pipeline)

        responses = engine.multi_discipline_query(
            query="How to maintain equipment?",
            slot_ids=["s1", "s2"],
        )
        assert len(responses) == 2
        assert all(isinstance(r, QueryResponse) for r in responses)

    def test_multi_discipline_empty_slot_ids(self) -> None:
        """multi_discipline_query with empty slot_ids returns empty list."""
        engine = _make_engine()
        responses = engine.multi_discipline_query(
            query="test",
            slot_ids=[],
        )
        assert responses == []

    def test_multi_discipline_unloaded_slot_raises(self) -> None:
        """multi_discipline_query with unloaded slot raises InferenceError."""
        pipeline = _make_rag_pipeline()
        manager = ModelManager()
        manager.register_slot(
            slot_id="s1",
            display_name="A",
            base_model_family="phi",
        )
        engine = HearthEngine(model_manager=manager, rag_pipeline=pipeline)
        with pytest.raises(InferenceError, match="not ready"):
            engine.multi_discipline_query("test", slot_ids=["s1"])


# ===================================================================
# TestDocumentBrowser
# ===================================================================


class TestDocumentBrowser:
    """Tests for the DocumentBrowser class."""

    def _make_browser(self) -> DocumentBrowser:
        """Create a DocumentBrowser with mock documents.

        Returns:
            A DocumentBrowser with 3 pre-loaded documents.
        """
        docs = [
            DocumentSummary(
                document_id="doc_001",
                title="TM-001 Maintenance Manual",
                document_type="technical_manual",
                chunk_count=50,
                page_count=120,
            ),
            DocumentSummary(
                document_id="doc_002",
                title="TM-002 Parts Catalog",
                document_type="parts_catalog",
                chunk_count=30,
                page_count=80,
            ),
            DocumentSummary(
                document_id="doc_003",
                title="TM-003 Reference Card",
                document_type="reference_card",
                chunk_count=5,
                page_count=2,
            ),
        ]
        return DocumentBrowser(documents=docs)

    def test_list_documents(self) -> None:
        """list_documents returns all configured documents."""
        browser = self._make_browser()
        docs = browser.list_documents()
        assert len(docs) == 3

    def test_list_documents_empty(self) -> None:
        """list_documents returns empty list when no docs configured."""
        browser = DocumentBrowser(documents=[])
        assert browser.list_documents() == []

    def test_get_document(self) -> None:
        """get_document returns detail for a known document."""
        browser = self._make_browser()
        detail = browser.get_document("doc_001")
        assert isinstance(detail, DocumentDetail)
        assert detail.document_id == "doc_001"
        assert detail.title == "TM-001 Maintenance Manual"

    def test_get_document_missing_raises(self) -> None:
        """get_document with unknown ID raises InferenceError."""
        browser = self._make_browser()
        with pytest.raises(InferenceError, match="not found"):
            browser.get_document("nonexistent")

    def test_search_documents(self) -> None:
        """search_documents filters by query in title."""
        browser = self._make_browser()
        results = browser.search_documents("Parts")
        assert len(results) == 1
        assert results[0].document_id == "doc_002"

    def test_search_documents_case_insensitive(self) -> None:
        """search_documents is case-insensitive."""
        browser = self._make_browser()
        results = browser.search_documents("maintenance")
        assert len(results) == 1
        assert results[0].document_id == "doc_001"

    def test_search_documents_no_match(self) -> None:
        """search_documents returns empty when no title matches."""
        browser = self._make_browser()
        results = browser.search_documents("nonexistent query")
        assert results == []

    def test_search_documents_limit(self) -> None:
        """search_documents respects the limit parameter."""
        browser = self._make_browser()
        results = browser.search_documents("TM", limit=2)
        assert len(results) <= 2

    def test_document_summary_to_dict(self) -> None:
        """DocumentSummary serializes to dictionary."""
        summary = DocumentSummary(
            document_id="d1",
            title="Test",
            document_type="manual",
            chunk_count=10,
            page_count=20,
        )
        data = summary.to_dict()
        assert data["document_id"] == "d1"
        assert data["chunk_count"] == 10

    def test_document_detail_to_dict(self) -> None:
        """DocumentDetail serializes to dictionary."""
        detail = DocumentDetail(
            document_id="d1",
            title="Test",
            document_type="manual",
            chunks=[{"id": "c1", "text": "content"}],
            metadata={"source": "test"},
        )
        data = detail.to_dict()
        assert data["document_id"] == "d1"
        assert len(data["chunks"]) == 1


# ===================================================================
# TestFeedbackSignal
# ===================================================================


class TestFeedbackSignal:
    """Tests for feedback signal capture."""

    def test_feedback_type_values(self) -> None:
        """FeedbackType enum has expected values."""
        assert FeedbackType.THUMBS_UP == "thumbs_up"
        assert FeedbackType.THUMBS_DOWN == "thumbs_down"
        assert FeedbackType.FLAG_INCORRECT == "flag_incorrect"
        assert FeedbackType.FLAG_INCOMPLETE == "flag_incomplete"

    def test_feedback_signal_construction(self) -> None:
        """FeedbackSignal initializes with all fields."""
        signal = FeedbackSignal(
            signal_id="fb_001",
            conversation_id="conv_1",
            turn_id="t1",
            feedback_type=FeedbackType.THUMBS_UP,
            comment=None,
            timestamp=datetime.now(),
        )
        assert signal.signal_id == "fb_001"
        assert signal.feedback_type == FeedbackType.THUMBS_UP
        assert signal.comment is None

    def test_feedback_signal_with_comment(self) -> None:
        """FeedbackSignal supports optional comment."""
        signal = FeedbackSignal(
            signal_id="fb_002",
            conversation_id="conv_1",
            turn_id="t1",
            feedback_type=FeedbackType.FLAG_INCORRECT,
            comment="The torque value is wrong",
            timestamp=datetime.now(),
        )
        assert signal.comment == "The torque value is wrong"

    def test_feedback_signal_to_dict(self) -> None:
        """FeedbackSignal serializes to dictionary."""
        signal = FeedbackSignal(
            signal_id="fb_001",
            conversation_id="c1",
            turn_id="t1",
            feedback_type=FeedbackType.THUMBS_DOWN,
            comment=None,
            timestamp=datetime(2026, 1, 1),
        )
        data = signal.to_dict()
        assert data["signal_id"] == "fb_001"
        assert data["feedback_type"] == "thumbs_down"


# ===================================================================
# TestHearthEngineWithFeedback
# ===================================================================


class TestHearthEngineWithFeedback:
    """Tests for feedback capture via HearthEngine."""

    def test_submit_feedback(self) -> None:
        """submit_feedback stores a feedback signal."""
        engine = _make_engine()
        req = QueryRequest(query="test", slot_id="slot_1")
        resp = engine.query(req)
        conv = engine.get_conversation(resp.conversation_id)
        turn_id = conv.turns[0].turn_id

        signal = engine.submit_feedback(
            conversation_id=resp.conversation_id,
            turn_id=turn_id,
            feedback_type=FeedbackType.THUMBS_UP,
        )
        assert isinstance(signal, FeedbackSignal)
        assert signal.feedback_type == FeedbackType.THUMBS_UP

    def test_submit_feedback_unknown_conversation_raises(self) -> None:
        """submit_feedback with unknown conversation raises InferenceError."""
        engine = _make_engine()
        with pytest.raises(InferenceError, match="not found"):
            engine.submit_feedback(
                conversation_id="ghost",
                turn_id="t1",
                feedback_type=FeedbackType.THUMBS_DOWN,
            )

    def test_submit_feedback_unknown_turn_raises(self) -> None:
        """submit_feedback with unknown turn_id raises InferenceError."""
        engine = _make_engine()
        req = QueryRequest(query="test", slot_id="slot_1")
        resp = engine.query(req)

        with pytest.raises(InferenceError, match="Turn.*not found"):
            engine.submit_feedback(
                conversation_id=resp.conversation_id,
                turn_id="ghost_turn",
                feedback_type=FeedbackType.THUMBS_DOWN,
            )

    def test_get_feedback_for_conversation(self) -> None:
        """get_feedback returns all signals for a conversation."""
        engine = _make_engine()
        req = QueryRequest(query="test", slot_id="slot_1")
        resp = engine.query(req)
        conv = engine.get_conversation(resp.conversation_id)
        turn_id = conv.turns[0].turn_id

        engine.submit_feedback(
            conversation_id=resp.conversation_id,
            turn_id=turn_id,
            feedback_type=FeedbackType.THUMBS_UP,
        )
        engine.submit_feedback(
            conversation_id=resp.conversation_id,
            turn_id=turn_id,
            feedback_type=FeedbackType.THUMBS_DOWN,
            comment="Not helpful",
        )

        signals = engine.get_feedback(resp.conversation_id)
        assert len(signals) == 2

    def test_get_feedback_empty(self) -> None:
        """get_feedback returns empty list when no feedback exists."""
        engine = _make_engine()
        req = QueryRequest(query="test", slot_id="slot_1")
        resp = engine.query(req)
        signals = engine.get_feedback(resp.conversation_id)
        assert signals == []


# ===================================================================
# TestIntegration
# ===================================================================


class TestIntegration:
    """End-to-end integration tests for HearthEngine + ModelManager + RAG."""

    def test_full_query_flow(self) -> None:
        """Complete flow: register, load, query, get conversation, citations."""
        manager = ModelManager()
        manager.register_slot(
            slot_id="maint_model",
            display_name="Maintenance LoRA v1",
            base_model_family="phi",
            discipline_id="disc_maint",
        )
        manager.load_model("maint_model")

        model = MockInference(default_response="Remove the old filter and install the new one.")
        retrieval = MockRetrievalAdapter(chunks=_make_chunks(3))
        pipeline = RAGPipeline(model=model, retrieval=retrieval, model_name="maint-lora-v1")

        engine = HearthEngine(model_manager=manager, rag_pipeline=pipeline)

        request = QueryRequest(
            query="How do I replace the oil filter?",
            slot_id="maint_model",
        )
        response = engine.query(request)

        assert "filter" in response.answer.lower()
        assert response.conversation_id is not None
        assert response.model_used == "maint-lora-v1"
        assert len(response.citations) > 0

        conv = engine.get_conversation(response.conversation_id)
        assert len(conv.turns) == 1
        assert conv.model_slot_id == "maint_model"

    def test_model_switch_between_queries(self) -> None:
        """Queries can use different model slots."""
        manager = ModelManager()
        manager.register_slot(
            slot_id="s1",
            display_name="Model A",
            base_model_family="phi",
            discipline_id="disc_a",
        )
        manager.register_slot(
            slot_id="s2",
            display_name="Model B",
            base_model_family="llama",
            discipline_id="disc_b",
        )
        manager.load_model("s1")
        manager.load_model("s2")

        pipeline = _make_rag_pipeline("Switched model response.")
        engine = HearthEngine(model_manager=manager, rag_pipeline=pipeline)

        resp1 = engine.query(QueryRequest(query="q1", slot_id="s1"))
        resp2 = engine.query(QueryRequest(query="q2", slot_id="s2"))

        assert resp1.conversation_id != resp2.conversation_id

    def test_feedback_after_query(self) -> None:
        """Feedback can be submitted after a successful query."""
        engine = _make_engine()
        req = QueryRequest(query="Test question", slot_id="slot_1")
        resp = engine.query(req)
        conv = engine.get_conversation(resp.conversation_id)
        turn_id = conv.turns[0].turn_id

        signal = engine.submit_feedback(
            conversation_id=resp.conversation_id,
            turn_id=turn_id,
            feedback_type=FeedbackType.FLAG_INCOMPLETE,
            comment="Missing torque specs",
        )
        assert signal.comment == "Missing torque specs"
        assert signal.feedback_type == FeedbackType.FLAG_INCOMPLETE

    def test_conversation_continuity(self) -> None:
        """Multiple queries in same conversation maintain history."""
        engine = _make_engine()
        req1 = QueryRequest(query="First question", slot_id="slot_1")
        resp1 = engine.query(req1)

        req2 = QueryRequest(
            query="Second question",
            slot_id="slot_1",
            conversation_id=resp1.conversation_id,
        )
        resp2 = engine.query(req2)

        assert resp1.conversation_id == resp2.conversation_id
        conv = engine.get_conversation(resp1.conversation_id)
        assert len(conv.turns) == 2
        assert conv.turns[0].query == "First question"
        assert conv.turns[1].query == "Second question"

    def test_multi_discipline_integration(self) -> None:
        """Multi-discipline query returns responses from multiple slots."""
        manager = ModelManager()
        for i in range(3):
            sid = f"slot_{i}"
            manager.register_slot(
                slot_id=sid,
                display_name=f"Model {i}",
                base_model_family="phi",
                discipline_id=f"disc_{i}",
            )
            manager.load_model(sid)

        pipeline = _make_rag_pipeline("Multi-discipline answer.")
        engine = HearthEngine(model_manager=manager, rag_pipeline=pipeline)

        responses = engine.multi_discipline_query(
            query="Cross-discipline question",
            slot_ids=["slot_0", "slot_1", "slot_2"],
        )
        assert len(responses) == 3
