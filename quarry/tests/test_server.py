"""
Tests for CHONK FastAPI server.
"""

import pytest
from fastapi.testclient import TestClient

from chonk.server import app, _state


@pytest.fixture(autouse=True)
def reset_state():
    """Reset server state before each test."""
    _state["project"] = None
    _state["tester"] = None
    yield
    _state["project"] = None
    _state["tester"] = None


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health check returns ok."""
        response = client.get("/api/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestProjectEndpoints:
    """Tests for project management endpoints."""

    def test_create_project(self, client):
        """Test creating a new project."""
        response = client.post(
            "/api/project/new",
            json={"name": "Test Project"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "Test Project"
        assert "id" in data
        assert "created_at" in data

    def test_get_project(self, client):
        """Test getting current project."""
        # Create project first
        client.post("/api/project/new", json={"name": "Test"})

        response = client.get("/api/project")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "Test"
        assert "documents" in data
        assert "test_suites" in data

    def test_get_project_without_creating(self, client):
        """Test getting project when none exists."""
        response = client.get("/api/project")
        assert response.status_code == 400
        assert "No project loaded" in response.json()["detail"]


class TestUtilityEndpoints:
    """Tests for utility endpoints."""

    def test_get_loaders(self, client):
        """Test getting available loaders."""
        response = client.get("/api/loaders")
        assert response.status_code == 200

        data = response.json()
        assert "extensions" in data
        assert ".pdf" in data["extensions"]
        assert ".md" in data["extensions"]

    def test_get_chunkers(self, client):
        """Test getting available chunkers."""
        response = client.get("/api/chunkers")
        assert response.status_code == 200

        data = response.json()
        assert "chunkers" in data
        assert "hierarchy" in data["chunkers"]
        assert "recursive" in data["chunkers"]

    def test_get_export_formats(self, client):
        """Test getting export formats."""
        response = client.get("/api/export/formats")
        assert response.status_code == 200

        data = response.json()
        assert "formats" in data
        assert "jsonl" in data["formats"]
        assert "json" in data["formats"]


class TestSettingsEndpoints:
    """Tests for settings endpoints."""

    def test_get_settings(self, client):
        """Test getting settings."""
        response = client.get("/api/settings")
        assert response.status_code == 200

        data = response.json()
        assert "default_chunker" in data
        assert data["default_chunker"] == "hierarchy"

    def test_save_settings(self, client):
        """Test saving settings."""
        response = client.post(
            "/api/settings",
            json={"default_target_tokens": 500},
        )
        assert response.status_code == 200
        assert response.json()["saved"] is True

        # Verify saved
        response = client.get("/api/settings")
        assert response.json()["default_target_tokens"] == 500


class TestTestSuiteEndpoints:
    """Tests for test suite endpoints."""

    def test_create_test_suite(self, client):
        """Test creating a test suite."""
        # Create project first
        client.post("/api/project/new", json={"name": "Test"})

        response = client.post("/api/test-suites?name=My%20Suite")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "My Suite"
        assert "id" in data

    def test_add_query_to_suite(self, client):
        """Test adding query to test suite."""
        # Setup
        client.post("/api/project/new", json={"name": "Test"})
        suite_response = client.post("/api/test-suites?name=Suite")
        suite_id = suite_response.json()["id"]

        # Add query
        response = client.post(
            f"/api/test-suites/{suite_id}/queries",
            json={
                "query": "What is the introduction about?",
                "expected_chunk_ids": ["chunk_1"],
                "notes": "Test query",
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["query"] == "What is the introduction about?"
        assert data["notes"] == "Test query"


class TestSearchEndpoints:
    """Tests for search endpoints."""

    def test_search_without_index(self, client):
        """Test search when no chunks indexed."""
        client.post("/api/project/new", json={"name": "Test"})

        response = client.post(
            "/api/test/search",
            json={"query": "test query"},
        )
        # Should fail because no chunks are indexed
        assert response.status_code == 400

    def test_get_test_status(self, client):
        """Test getting test/index status."""
        client.post("/api/project/new", json={"name": "Test"})

        response = client.get("/api/test/status")
        assert response.status_code == 200

        data = response.json()
        assert "indexed" in data
        assert "chunk_count" in data


class TestChunkEndpoints:
    """Tests for chunk manipulation endpoints."""

    def test_merge_requires_multiple_chunks(self, client):
        """Test merge validation."""
        client.post("/api/project/new", json={"name": "Test"})

        response = client.post(
            "/api/chunks/merge",
            json={"chunk_ids": ["single_chunk"]},
        )
        assert response.status_code == 400
        assert "at least 2" in response.json()["detail"].lower()

    def test_split_invalid_position(self, client):
        """Test split with invalid position."""
        client.post("/api/project/new", json={"name": "Test"})

        response = client.post(
            "/api/chunks/split",
            json={"chunk_id": "nonexistent", "split_position": 10},
        )
        assert response.status_code == 404


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_json(self, client):
        """Test handling of invalid JSON."""
        client.post("/api/project/new", json={"name": "Test"})

        response = client.post(
            "/api/chunks/merge",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_missing_required_fields(self, client):
        """Test handling of missing required fields."""
        response = client.post(
            "/api/project/new",
            json={},  # Missing 'name'
        )
        assert response.status_code == 422
