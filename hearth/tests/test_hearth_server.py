"""Tests for the Hearth FastAPI router.

Covers health, model management, query handling, conversation CRUD,
document browsing, and feedback capture endpoints using httpx TestClient.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from foundry.src.evaluation import MockInference
from foundry.src.rag_integration import MockRetrievalAdapter, RAGPipeline
from hearth.src.feedback import FeedbackManager
from hearth.src.inference import (
    DocumentBrowser,
    DocumentSummary,
    HearthEngine,
    ModelManager,
)
from hearth.src.server import _state, configure, router

# ===================================================================
# Fixtures
# ===================================================================


def _make_chunks(count: int = 3) -> list[dict[str, Any]]:
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


def _make_documents() -> list[DocumentSummary]:
    """Build a list of test document summaries.

    Returns:
        List of DocumentSummary instances.
    """
    return [
        DocumentSummary(
            document_id="doc_001",
            title="TM-001 Maintenance Manual",
            document_type="technical_manual",
            chunk_count=42,
            page_count=120,
        ),
        DocumentSummary(
            document_id="doc_002",
            title="TM-002 Parts Catalog",
            document_type="parts_catalog",
            chunk_count=18,
            page_count=60,
        ),
    ]


@pytest.fixture()
def client() -> TestClient:
    """Create a TestClient with Hearth router mounted and configured.

    Returns:
        httpx TestClient ready for requests.
    """
    app = FastAPI()
    app.include_router(router, prefix="/api/hearth")

    manager = ModelManager()
    retrieval = MockRetrievalAdapter(chunks=_make_chunks())
    model = MockInference(default_response="Answer based on context.")
    pipeline = RAGPipeline(model=model, retrieval=retrieval)
    engine = HearthEngine(model_manager=manager, rag_pipeline=pipeline)

    browser = DocumentBrowser(documents=_make_documents())
    feedback_manager = FeedbackManager()

    configure(engine=engine, browser=browser, feedback_manager=feedback_manager)

    yield TestClient(app)

    # Reset state after each test
    _state["engine"] = None
    _state["browser"] = None
    _state["feedback_manager"] = None


@pytest.fixture()
def unconfigured_client() -> TestClient:
    """Create a TestClient where Hearth has NOT been configured.

    Returns:
        httpx TestClient that should return 503 for most endpoints.
    """
    app = FastAPI()
    app.include_router(router, prefix="/api/hearth")

    # Ensure state is cleared
    _state["engine"] = None
    _state["browser"] = None
    _state["feedback_manager"] = None

    return TestClient(app)


# ===================================================================
# Health
# ===================================================================


class TestHealth:
    """Tests for GET /health."""

    def test_health_configured(self, client: TestClient) -> None:
        """Health returns ok when engine is configured."""
        resp = client.get("/api/hearth/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert data["components"]["engine"] is True
        assert data["components"]["browser"] is True
        assert data["components"]["feedback"] is True

    def test_health_unconfigured(self, unconfigured_client: TestClient) -> None:
        """Health returns not_configured when engine is not set."""
        resp = unconfigured_client.get("/api/hearth/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "not_configured"
        assert data["components"]["engine"] is False


# ===================================================================
# Models
# ===================================================================


class TestModels:
    """Tests for model management endpoints."""

    def test_list_models_empty(self, client: TestClient) -> None:
        """Listing models on fresh engine returns empty list."""
        resp = client.get("/api/hearth/models")
        assert resp.status_code == 200
        assert resp.json()["models"] == []

    def test_register_model(self, client: TestClient) -> None:
        """Registering a model returns 201 with slot data."""
        payload = {
            "slot_id": "slot_1",
            "display_name": "Maintenance LoRA",
            "base_model_family": "phi",
        }
        resp = client.post("/api/hearth/models/register", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["slot_id"] == "slot_1"
        assert data["display_name"] == "Maintenance LoRA"
        assert data["status"] == "unloaded"

    def test_register_duplicate_returns_409(self, client: TestClient) -> None:
        """Registering a slot_id that already exists returns 409."""
        payload = {
            "slot_id": "slot_dup",
            "display_name": "Model A",
            "base_model_family": "llama",
        }
        client.post("/api/hearth/models/register", json=payload)
        resp = client.post("/api/hearth/models/register", json=payload)
        assert resp.status_code == 409

    def test_load_and_unload_model(self, client: TestClient) -> None:
        """Loading then unloading a model transitions statuses correctly."""
        payload = {
            "slot_id": "slot_lu",
            "display_name": "LU Model",
            "base_model_family": "phi",
        }
        client.post("/api/hearth/models/register", json=payload)

        # Load
        resp = client.post("/api/hearth/models/slot_lu/load")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

        # Unload
        resp = client.post("/api/hearth/models/slot_lu/unload")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unloaded"

    def test_load_nonexistent_returns_404(self, client: TestClient) -> None:
        """Loading a model that does not exist returns 404."""
        resp = client.post("/api/hearth/models/no_such_slot/load")
        assert resp.status_code == 404

    def test_get_model_status(self, client: TestClient) -> None:
        """Getting model status returns current slot state."""
        payload = {
            "slot_id": "slot_status",
            "display_name": "Status Model",
            "base_model_family": "phi",
        }
        client.post("/api/hearth/models/register", json=payload)
        resp = client.get("/api/hearth/models/slot_status/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unloaded"

    def test_get_model_status_nonexistent_returns_404(self, client: TestClient) -> None:
        """Getting status for a nonexistent model returns 404."""
        resp = client.get("/api/hearth/models/ghost/status")
        assert resp.status_code == 404

    def test_list_models_after_register(self, client: TestClient) -> None:
        """Listing models after registration returns the registered slot."""
        payload = {
            "slot_id": "slot_list",
            "display_name": "Listed Model",
            "base_model_family": "phi",
        }
        client.post("/api/hearth/models/register", json=payload)
        resp = client.get("/api/hearth/models")
        assert resp.status_code == 200
        models = resp.json()["models"]
        assert len(models) == 1
        assert models[0]["slot_id"] == "slot_list"

    def test_models_503_when_unconfigured(self, unconfigured_client: TestClient) -> None:
        """Model endpoints return 503 when engine is not configured."""
        resp = unconfigured_client.get("/api/hearth/models")
        assert resp.status_code == 503


# ===================================================================
# Query
# ===================================================================


class TestQuery:
    """Tests for query endpoints."""

    def _register_and_load(self, client: TestClient, slot_id: str = "q_slot") -> None:
        """Helper to register and load a model slot.

        Args:
            client: The test client.
            slot_id: Slot identifier to use.
        """
        client.post(
            "/api/hearth/models/register",
            json={
                "slot_id": slot_id,
                "display_name": "Query Model",
                "base_model_family": "phi",
            },
        )
        client.post(f"/api/hearth/models/{slot_id}/load")

    def test_query_success(self, client: TestClient) -> None:
        """A valid query returns an answer with conversation_id."""
        self._register_and_load(client)
        resp = client.post(
            "/api/hearth/query",
            json={"query": "How do I replace a filter?", "slot_id": "q_slot"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "conversation_id" in data
        assert data["model_used"] is not None

    def test_query_unloaded_model_returns_422(self, client: TestClient) -> None:
        """Querying an unloaded model returns 422."""
        client.post(
            "/api/hearth/models/register",
            json={
                "slot_id": "unloaded_slot",
                "display_name": "Unloaded",
                "base_model_family": "phi",
            },
        )
        resp = client.post(
            "/api/hearth/query",
            json={"query": "test", "slot_id": "unloaded_slot"},
        )
        assert resp.status_code == 422

    def test_query_nonexistent_slot_returns_422(self, client: TestClient) -> None:
        """Querying a nonexistent slot returns 422."""
        resp = client.post(
            "/api/hearth/query",
            json={"query": "test", "slot_id": "nonexistent"},
        )
        assert resp.status_code == 422

    def test_query_creates_conversation(self, client: TestClient) -> None:
        """A query creates a new conversation accessible via list."""
        self._register_and_load(client)
        resp = client.post(
            "/api/hearth/query",
            json={"query": "What is maintenance?", "slot_id": "q_slot"},
        )
        conv_id = resp.json()["conversation_id"]

        convos_resp = client.get("/api/hearth/conversations")
        assert convos_resp.status_code == 200
        ids = [c["conversation_id"] for c in convos_resp.json()["conversations"]]
        assert conv_id in ids

    def test_multi_discipline_query(self, client: TestClient) -> None:
        """Multi-discipline query returns one response per slot."""
        self._register_and_load(client, "md_slot_a")
        self._register_and_load(client, "md_slot_b")
        resp = client.post(
            "/api/hearth/query/multi-discipline",
            json={
                "query": "How do I replace a filter?",
                "slot_ids": ["md_slot_a", "md_slot_b"],
            },
        )
        assert resp.status_code == 200
        responses = resp.json()["responses"]
        assert len(responses) == 2

    def test_multi_discipline_query_bad_slot_returns_422(self, client: TestClient) -> None:
        """Multi-discipline query with a bad slot returns 422."""
        self._register_and_load(client, "good_slot")
        resp = client.post(
            "/api/hearth/query/multi-discipline",
            json={
                "query": "test",
                "slot_ids": ["good_slot", "bad_slot"],
            },
        )
        assert resp.status_code == 422

    def test_query_empty_body_returns_422(self, client: TestClient) -> None:
        """Query with empty body returns 422 from validation."""
        resp = client.post("/api/hearth/query", json={})
        assert resp.status_code == 422


# ===================================================================
# Conversations
# ===================================================================


class TestConversations:
    """Tests for conversation endpoints."""

    def _create_conversation(self, client: TestClient) -> str:
        """Helper to register a model, load it, and run a query.

        Args:
            client: The test client.

        Returns:
            The conversation_id from the query response.
        """
        client.post(
            "/api/hearth/models/register",
            json={
                "slot_id": "conv_slot",
                "display_name": "Conv Model",
                "base_model_family": "phi",
            },
        )
        client.post("/api/hearth/models/conv_slot/load")
        resp = client.post(
            "/api/hearth/query",
            json={"query": "Hello", "slot_id": "conv_slot"},
        )
        return resp.json()["conversation_id"]

    def test_list_conversations_empty(self, client: TestClient) -> None:
        """Listing conversations when none exist returns empty list."""
        resp = client.get("/api/hearth/conversations")
        assert resp.status_code == 200
        assert resp.json()["conversations"] == []

    def test_get_conversation(self, client: TestClient) -> None:
        """Retrieving a conversation returns full turn data."""
        conv_id = self._create_conversation(client)
        resp = client.get(f"/api/hearth/conversations/{conv_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["conversation_id"] == conv_id
        assert len(data["turns"]) == 1

    def test_get_nonexistent_conversation_returns_404(self, client: TestClient) -> None:
        """Getting a nonexistent conversation returns 404."""
        resp = client.get("/api/hearth/conversations/no_such_conv")
        assert resp.status_code == 404

    def test_delete_conversation(self, client: TestClient) -> None:
        """Deleting a conversation removes it from the list."""
        conv_id = self._create_conversation(client)
        resp = client.delete(f"/api/hearth/conversations/{conv_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == conv_id

        # Verify it is gone
        resp = client.get(f"/api/hearth/conversations/{conv_id}")
        assert resp.status_code == 404

    def test_delete_nonexistent_conversation_returns_404(self, client: TestClient) -> None:
        """Deleting a nonexistent conversation returns 404."""
        resp = client.delete("/api/hearth/conversations/ghost")
        assert resp.status_code == 404

    def test_conversation_summary_includes_turn_count(self, client: TestClient) -> None:
        """Conversation list entries include turn_count."""
        conv_id = self._create_conversation(client)
        resp = client.get("/api/hearth/conversations")
        convos = resp.json()["conversations"]
        matching = [c for c in convos if c["conversation_id"] == conv_id]
        assert len(matching) == 1
        assert matching[0]["turn_count"] == 1


# ===================================================================
# Documents
# ===================================================================


class TestDocuments:
    """Tests for document browsing endpoints."""

    def test_list_documents(self, client: TestClient) -> None:
        """Listing documents returns the configured summaries."""
        resp = client.get("/api/hearth/documents")
        assert resp.status_code == 200
        docs = resp.json()["documents"]
        assert len(docs) == 2
        ids = {d["document_id"] for d in docs}
        assert "doc_001" in ids
        assert "doc_002" in ids

    def test_get_document(self, client: TestClient) -> None:
        """Getting a specific document returns detail data."""
        resp = client.get("/api/hearth/documents/doc_001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_id"] == "doc_001"
        assert data["title"] == "TM-001 Maintenance Manual"

    def test_get_nonexistent_document_returns_404(self, client: TestClient) -> None:
        """Getting a nonexistent document returns 404."""
        resp = client.get("/api/hearth/documents/no_doc")
        assert resp.status_code == 404

    def test_search_documents(self, client: TestClient) -> None:
        """Searching documents by title returns matching results."""
        resp = client.post(
            "/api/hearth/documents/search",
            json={"query": "Parts", "limit": 5},
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 1
        assert results[0]["document_id"] == "doc_002"

    def test_search_documents_no_match(self, client: TestClient) -> None:
        """Searching with no matching query returns empty results."""
        resp = client.post(
            "/api/hearth/documents/search",
            json={"query": "nonexistent-xyz", "limit": 5},
        )
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_documents_503_when_unconfigured(self, unconfigured_client: TestClient) -> None:
        """Document endpoints return 503 when browser is not configured."""
        resp = unconfigured_client.get("/api/hearth/documents")
        assert resp.status_code == 503


# ===================================================================
# Feedback
# ===================================================================


class TestFeedback:
    """Tests for feedback capture and analysis endpoints."""

    def test_submit_feedback(self, client: TestClient) -> None:
        """Submitting feedback returns 201 with the recorded signal."""
        payload = {
            "signal_type": "accepted",
            "conversation_id": "conv_123",
            "query": "How do I replace a filter?",
            "response": "Remove the old filter and install a new one.",
        }
        resp = client.post("/api/hearth/feedback", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["signal_type"] == "accepted"
        assert data["conversation_id"] == "conv_123"
        assert "signal_id" in data

    def test_submit_feedback_invalid_type_returns_422(self, client: TestClient) -> None:
        """Submitting feedback with an invalid signal type returns 422."""
        payload = {
            "signal_type": "invalid_type",
            "conversation_id": "conv_123",
            "query": "test",
        }
        resp = client.post("/api/hearth/feedback", json=payload)
        assert resp.status_code == 422

    def test_feedback_routing(self, client: TestClient) -> None:
        """Getting routing for a submitted signal returns a decision."""
        payload = {
            "signal_type": "flagged_error",
            "conversation_id": "conv_rt",
            "query": "What is the torque spec?",
            "response": "Incorrect answer.",
        }
        fb_resp = client.post("/api/hearth/feedback", json=payload)
        signal_id = fb_resp.json()["signal_id"]

        resp = client.get(
            "/api/hearth/feedback/routing",
            params={"signal_id": signal_id},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["target"] == "forge"
        assert data["priority"] == "high"

    def test_feedback_routing_nonexistent_signal(self, client: TestClient) -> None:
        """Getting routing for a nonexistent signal returns an error."""
        resp = client.get(
            "/api/hearth/feedback/routing",
            params={"signal_id": "sig_nonexistent"},
        )
        assert resp.status_code == 404

    def test_feedback_dashboard(self, client: TestClient) -> None:
        """Dashboard returns summary data for a discipline."""
        # Submit a few signals first
        for signal_type in ["accepted", "rejected", "flagged_error"]:
            client.post(
                "/api/hearth/feedback",
                json={
                    "signal_type": signal_type,
                    "conversation_id": "conv_dash",
                    "query": "Dashboard test query",
                    "discipline_id": "maintenance",
                },
            )

        resp = client.get(
            "/api/hearth/feedback/dashboard",
            params={"discipline_id": "maintenance", "days": 30},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["discipline_id"] == "maintenance"
        assert data["total_queries"] == 3
        assert "acceptance_rate" in data
        assert "rejection_rate" in data

    def test_feedback_patterns_empty(self, client: TestClient) -> None:
        """Patterns endpoint with no signals returns empty list."""
        resp = client.get("/api/hearth/feedback/patterns")
        assert resp.status_code == 200
        assert resp.json()["patterns"] == []

    def test_feedback_patterns_with_data(self, client: TestClient) -> None:
        """Patterns endpoint detects patterns from repeated signals."""
        # Submit enough signals to trigger a pattern (min_signals=3 default)
        for _ in range(4):
            client.post(
                "/api/hearth/feedback",
                json={
                    "signal_type": "no_result",
                    "conversation_id": "conv_pat",
                    "query": "Same repeated query",
                    "discipline_id": "maintenance",
                },
            )

        resp = client.get(
            "/api/hearth/feedback/patterns",
            params={"discipline_id": "maintenance"},
        )
        assert resp.status_code == 200
        patterns = resp.json()["patterns"]
        assert len(patterns) >= 1

    def test_feedback_503_when_unconfigured(self, unconfigured_client: TestClient) -> None:
        """Feedback endpoints return 503 when not configured."""
        resp = unconfigured_client.post(
            "/api/hearth/feedback",
            json={
                "signal_type": "accepted",
                "conversation_id": "c",
                "query": "q",
            },
        )
        assert resp.status_code == 503

    def test_submit_feedback_with_metadata(self, client: TestClient) -> None:
        """Submitting feedback with metadata stores it correctly."""
        payload = {
            "signal_type": "rejected",
            "conversation_id": "conv_meta",
            "query": "test query",
            "metadata": {"citation_score": 0.3},
        }
        resp = client.post("/api/hearth/feedback", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["metadata"]["citation_score"] == 0.3


# ===================================================================
# Validation edge cases
# ===================================================================


class TestValidation:
    """Tests for Pydantic validation on request bodies."""

    def test_register_model_missing_fields(self, client: TestClient) -> None:
        """Registering a model with missing required fields returns 422."""
        resp = client.post(
            "/api/hearth/models/register",
            json={"slot_id": "x"},
        )
        assert resp.status_code == 422

    def test_query_missing_query_field(self, client: TestClient) -> None:
        """Query with missing query field returns 422."""
        resp = client.post(
            "/api/hearth/query",
            json={"slot_id": "s"},
        )
        assert resp.status_code == 422

    def test_document_search_empty_query(self, client: TestClient) -> None:
        """Document search with empty query string returns 422."""
        resp = client.post(
            "/api/hearth/documents/search",
            json={"query": "", "limit": 5},
        )
        assert resp.status_code == 422

    def test_feedback_missing_conversation_id(self, client: TestClient) -> None:
        """Feedback with missing conversation_id returns 422."""
        resp = client.post(
            "/api/hearth/feedback",
            json={"signal_type": "accepted", "query": "q"},
        )
        assert resp.status_code == 422

    def test_multi_discipline_empty_slot_ids(self, client: TestClient) -> None:
        """Multi-discipline query with empty slot_ids returns 422."""
        resp = client.post(
            "/api/hearth/query/multi-discipline",
            json={"query": "test", "slot_ids": []},
        )
        assert resp.status_code == 422
