"""Tests for the Forge FastAPI server endpoints.

Uses httpx TestClient with an in-memory ForgeStorage backend to
exercise all major REST endpoints: contributors, disciplines,
competencies, examples, discovery, consistency, coverage, and
curriculum export.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from forge.src.models import (
    Competency,
    Contributor,
    Discipline,
    DisciplineStatus,
    Example,
)
from forge.src.server import init_forge_storage, router
from forge.src.storage import ForgeStorage


@pytest.fixture
def app() -> FastAPI:
    """Create a FastAPI app with the forge router mounted."""
    test_app = FastAPI()
    test_app.include_router(router, prefix="/api/forge")
    return test_app


@pytest.fixture
def storage() -> ForgeStorage:
    """Initialize in-memory storage for the router."""
    return init_forge_storage(":memory:")


@pytest.fixture
def client(app: FastAPI, storage: ForgeStorage) -> TestClient:
    """TestClient wired to the Forge router with initialized storage."""
    return TestClient(app)


@pytest.fixture
def seeded_storage(storage: ForgeStorage) -> ForgeStorage:
    """Storage pre-populated with a contributor, discipline, competency, and example."""
    contrib = Contributor(
        id="contrib_seed01",
        name="Test User",
        email="test@example.com",
    )
    storage.create_contributor(contrib)

    disc = Discipline(
        id="disc_seed01",
        name="Military Maintenance",
        description="Aircraft maintenance procedures",
        status=DisciplineStatus.DRAFT,
        created_by=contrib.id,
        vocabulary=["torque", "clearance"],
        document_types=["manual"],
    )
    storage.create_discipline(disc)

    comp = Competency(
        id="comp_seed01",
        name="Fault Isolation",
        description="Identify and isolate equipment faults",
        discipline_id=disc.id,
        coverage_target=25,
    )
    storage.create_competency(comp)

    ex = Example(
        id="ex_seed01",
        question="How do you isolate a hydraulic leak?",
        ideal_answer="Check pressure readings, isolate sections, inspect fittings.",
        competency_id=comp.id,
        contributor_id=contrib.id,
        discipline_id=disc.id,
    )
    storage.create_example(ex)

    return storage


@pytest.fixture
def seeded_client(app: FastAPI, seeded_storage: ForgeStorage) -> TestClient:
    """TestClient with pre-populated storage."""
    return TestClient(app)


# ===================================================================
# Health
# ===================================================================


class TestHealth:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self, client: TestClient) -> None:
        """Health endpoint returns status ok."""
        resp = client.get("/api/forge/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "forge"
        assert data["storage_initialized"] is True

    def test_health_includes_version(self, client: TestClient) -> None:
        """Health endpoint includes version info."""
        resp = client.get("/api/forge/health")
        assert "version" in resp.json()


# ===================================================================
# Contributors
# ===================================================================


class TestContributors:
    """Tests for contributor CRUD endpoints."""

    def test_create_contributor(self, client: TestClient) -> None:
        """POST /contributors creates a new contributor."""
        resp = client.post(
            "/api/forge/contributors",
            json={"name": "Bob Jones", "email": "bob@example.com"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Bob Jones"
        assert data["email"] == "bob@example.com"
        assert data["id"].startswith("contrib_")

    def test_create_contributor_minimal(self, client: TestClient) -> None:
        """POST /contributors works with name only."""
        resp = client.post(
            "/api/forge/contributors",
            json={"name": "Minimal User"},
        )
        assert resp.status_code == 201
        assert resp.json()["email"] == ""

    def test_create_contributor_empty_name_rejected(self, client: TestClient) -> None:
        """POST /contributors rejects empty name."""
        resp = client.post(
            "/api/forge/contributors",
            json={"name": ""},
        )
        assert resp.status_code == 422

    def test_get_contributor(self, seeded_client: TestClient) -> None:
        """GET /contributors/{id} returns an existing contributor."""
        resp = seeded_client.get("/api/forge/contributors/contrib_seed01")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Test User"

    def test_get_contributor_not_found(self, client: TestClient) -> None:
        """GET /contributors/{id} returns 404 for missing contributor."""
        resp = client.get("/api/forge/contributors/contrib_missing")
        assert resp.status_code == 404

    def test_list_contributors(self, seeded_client: TestClient) -> None:
        """GET /contributors returns all contributors."""
        resp = seeded_client.get("/api/forge/contributors")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["contributors"]) >= 1
        names = [c["name"] for c in data["contributors"]]
        assert "Test User" in names

    def test_update_contributor(self, seeded_client: TestClient) -> None:
        """PUT /contributors/{id} updates fields."""
        resp = seeded_client.put(
            "/api/forge/contributors/contrib_seed01",
            json={"name": "Updated Name"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated Name"

    def test_update_contributor_not_found(self, client: TestClient) -> None:
        """PUT /contributors/{id} returns 404 for missing contributor."""
        resp = client.put(
            "/api/forge/contributors/contrib_missing",
            json={"name": "Nope"},
        )
        assert resp.status_code == 404

    def test_delete_contributor(self, seeded_client: TestClient) -> None:
        """DELETE /contributors/{id} removes a contributor with no FK refs."""
        # Create a standalone contributor with no FK references
        create_resp = seeded_client.post(
            "/api/forge/contributors",
            json={"name": "Deletable User"},
        )
        deletable_id = create_resp.json()["id"]
        resp = seeded_client.delete(f"/api/forge/contributors/{deletable_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == deletable_id

    def test_delete_contributor_not_found(self, client: TestClient) -> None:
        """DELETE /contributors/{id} returns 404 for missing contributor."""
        resp = client.delete("/api/forge/contributors/contrib_missing")
        assert resp.status_code == 404


# ===================================================================
# Disciplines
# ===================================================================


class TestDisciplines:
    """Tests for discipline CRUD endpoints."""

    def test_create_discipline(self, seeded_client: TestClient) -> None:
        """POST /disciplines creates a new discipline."""
        resp = seeded_client.post(
            "/api/forge/disciplines",
            json={
                "name": "Electrical Systems",
                "description": "Power distribution and wiring",
                "created_by": "contrib_seed01",
                "vocabulary": ["voltage", "amperage"],
                "document_types": ["schematic"],
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Electrical Systems"
        assert data["id"].startswith("disc_")

    def test_get_discipline(self, seeded_client: TestClient) -> None:
        """GET /disciplines/{id} returns an existing discipline."""
        resp = seeded_client.get("/api/forge/disciplines/disc_seed01")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Military Maintenance"

    def test_get_discipline_not_found(self, client: TestClient) -> None:
        """GET /disciplines/{id} returns 404 for missing discipline."""
        resp = client.get("/api/forge/disciplines/disc_missing")
        assert resp.status_code == 404

    def test_list_disciplines(self, seeded_client: TestClient) -> None:
        """GET /disciplines returns all disciplines."""
        resp = seeded_client.get("/api/forge/disciplines")
        assert resp.status_code == 200
        assert len(resp.json()["disciplines"]) >= 1

    def test_list_disciplines_filter_status(self, seeded_client: TestClient) -> None:
        """GET /disciplines?status=draft filters by status."""
        resp = seeded_client.get("/api/forge/disciplines?status=draft")
        assert resp.status_code == 200
        for d in resp.json()["disciplines"]:
            assert d["status"] == "draft"

    def test_list_disciplines_invalid_status(self, client: TestClient) -> None:
        """GET /disciplines?status=invalid returns 400."""
        resp = client.get("/api/forge/disciplines?status=invalid")
        assert resp.status_code == 400

    def test_update_discipline(self, seeded_client: TestClient) -> None:
        """PUT /disciplines/{id} updates fields."""
        resp = seeded_client.put(
            "/api/forge/disciplines/disc_seed01",
            json={"name": "Updated Discipline", "status": "active"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Updated Discipline"
        assert data["status"] == "active"

    def test_update_discipline_not_found(self, client: TestClient) -> None:
        """PUT /disciplines/{id} returns 404 for missing discipline."""
        resp = client.put(
            "/api/forge/disciplines/disc_missing",
            json={"name": "Nope"},
        )
        assert resp.status_code == 404


# ===================================================================
# Competencies
# ===================================================================


class TestCompetencies:
    """Tests for competency CRUD endpoints."""

    def test_create_competency(self, seeded_client: TestClient) -> None:
        """POST /competencies creates a new competency."""
        resp = seeded_client.post(
            "/api/forge/competencies",
            json={
                "name": "Parts Interpretation",
                "description": "Read and interpret parts lists",
                "discipline_id": "disc_seed01",
                "coverage_target": 30,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Parts Interpretation"
        assert data["id"].startswith("comp_")
        assert data["coverage_target"] == 30

    def test_get_competency(self, seeded_client: TestClient) -> None:
        """GET /competencies/{id} returns an existing competency."""
        resp = seeded_client.get("/api/forge/competencies/comp_seed01")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Fault Isolation"

    def test_get_competency_not_found(self, client: TestClient) -> None:
        """GET /competencies/{id} returns 404 for missing competency."""
        resp = client.get("/api/forge/competencies/comp_missing")
        assert resp.status_code == 404

    def test_list_competencies_for_discipline(self, seeded_client: TestClient) -> None:
        """GET /disciplines/{id}/competencies returns competencies."""
        resp = seeded_client.get("/api/forge/disciplines/disc_seed01/competencies")
        assert resp.status_code == 200
        comps = resp.json()["competencies"]
        assert len(comps) >= 1
        assert comps[0]["discipline_id"] == "disc_seed01"

    def test_update_competency(self, seeded_client: TestClient) -> None:
        """PUT /competencies/{id} updates fields."""
        resp = seeded_client.put(
            "/api/forge/competencies/comp_seed01",
            json={"name": "Updated Competency", "coverage_target": 50},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Updated Competency"
        assert data["coverage_target"] == 50

    def test_update_competency_not_found(self, client: TestClient) -> None:
        """PUT /competencies/{id} returns 404 for missing competency."""
        resp = client.put(
            "/api/forge/competencies/comp_missing",
            json={"name": "Nope"},
        )
        assert resp.status_code == 404

    def test_delete_competency(self, seeded_client: TestClient) -> None:
        """DELETE /competencies/{id} removes a competency."""
        resp = seeded_client.delete("/api/forge/competencies/comp_seed01")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == "comp_seed01"

    def test_delete_competency_not_found(self, client: TestClient) -> None:
        """DELETE /competencies/{id} returns 404 for missing competency."""
        resp = client.delete("/api/forge/competencies/comp_missing")
        assert resp.status_code == 404


# ===================================================================
# Examples
# ===================================================================


class TestExamples:
    """Tests for example CRUD endpoints."""

    def test_create_example(self, seeded_client: TestClient) -> None:
        """POST /examples creates a new example."""
        resp = seeded_client.post(
            "/api/forge/examples",
            json={
                "question": "What is the torque spec for the main rotor bolt?",
                "ideal_answer": "The main rotor bolt requires 150 ft-lbs of torque.",
                "competency_id": "comp_seed01",
                "contributor_id": "contrib_seed01",
                "discipline_id": "disc_seed01",
                "variants": ["Torque value for rotor bolt?"],
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["id"].startswith("ex_")
        assert data["question"].startswith("What is the torque")

    def test_get_example(self, seeded_client: TestClient) -> None:
        """GET /examples/{id} returns an existing example."""
        resp = seeded_client.get("/api/forge/examples/ex_seed01")
        assert resp.status_code == 200
        assert "hydraulic" in resp.json()["question"].lower()

    def test_get_example_not_found(self, client: TestClient) -> None:
        """GET /examples/{id} returns 404 for missing example."""
        resp = client.get("/api/forge/examples/ex_missing")
        assert resp.status_code == 404

    def test_list_examples_for_competency(self, seeded_client: TestClient) -> None:
        """GET /competencies/{id}/examples returns examples."""
        resp = seeded_client.get("/api/forge/competencies/comp_seed01/examples")
        assert resp.status_code == 200
        examples = resp.json()["examples"]
        assert len(examples) >= 1

    def test_update_example(self, seeded_client: TestClient) -> None:
        """PUT /examples/{id} updates fields."""
        resp = seeded_client.put(
            "/api/forge/examples/ex_seed01",
            json={"question": "Updated question about hydraulic leaks?"},
        )
        assert resp.status_code == 200
        assert resp.json()["question"] == "Updated question about hydraulic leaks?"

    def test_update_example_review_status(self, seeded_client: TestClient) -> None:
        """PUT /examples/{id} can update review_status."""
        resp = seeded_client.put(
            "/api/forge/examples/ex_seed01",
            json={"review_status": "approved"},
        )
        assert resp.status_code == 200
        assert resp.json()["review_status"] == "approved"

    def test_update_example_invalid_review_status(self, seeded_client: TestClient) -> None:
        """PUT /examples/{id} rejects invalid review_status."""
        resp = seeded_client.put(
            "/api/forge/examples/ex_seed01",
            json={"review_status": "bogus"},
        )
        assert resp.status_code == 400

    def test_update_example_not_found(self, client: TestClient) -> None:
        """PUT /examples/{id} returns 404 for missing example."""
        resp = client.put(
            "/api/forge/examples/ex_missing",
            json={"question": "Nope"},
        )
        assert resp.status_code == 404

    def test_delete_example(self, seeded_client: TestClient) -> None:
        """DELETE /examples/{id} removes an example."""
        resp = seeded_client.delete("/api/forge/examples/ex_seed01")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == "ex_seed01"

    def test_delete_example_not_found(self, client: TestClient) -> None:
        """DELETE /examples/{id} returns 404 for missing example."""
        resp = client.delete("/api/forge/examples/ex_missing")
        assert resp.status_code == 404


# ===================================================================
# Discovery
# ===================================================================


class TestDiscovery:
    """Tests for discovery session endpoints."""

    def test_start_discovery_session(self, seeded_client: TestClient) -> None:
        """POST /discovery/start creates a new session."""
        resp = seeded_client.post(
            "/api/forge/discovery/start",
            json={
                "discipline_name": "Engine Repair",
                "contributor_id": "contrib_seed01",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "session" in data
        assert data["session"]["discipline_name"] == "Engine Repair"
        assert data["session"]["status"] == "in_progress"
        assert "current_questions" in data
        assert len(data["current_questions"]) > 0

    def test_record_discovery_answer(self, seeded_client: TestClient) -> None:
        """POST /discovery/answer records an answer and returns updated session."""
        # Start session first
        start_resp = seeded_client.post(
            "/api/forge/discovery/start",
            json={
                "discipline_name": "Engine Repair",
                "contributor_id": "contrib_seed01",
            },
        )
        session_id = start_resp.json()["session"]["id"]

        # Record an answer
        resp = seeded_client.post(
            "/api/forge/discovery/answer",
            json={
                "session_id": session_id,
                "question_id": "q_orient_01",
                "raw_text": "Engine repair involves maintaining and fixing aircraft engines.",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "q_orient_01" in data["session"]["responses"]

    def test_record_answer_invalid_session(self, client: TestClient) -> None:
        """POST /discovery/answer returns 404 for missing session."""
        resp = client.post(
            "/api/forge/discovery/answer",
            json={
                "session_id": "dsess_missing",
                "question_id": "q_orient_01",
                "raw_text": "Answer text",
            },
        )
        assert resp.status_code == 404

    def test_record_answer_invalid_question(self, seeded_client: TestClient) -> None:
        """POST /discovery/answer returns 400 for invalid question ID."""
        start_resp = seeded_client.post(
            "/api/forge/discovery/start",
            json={
                "discipline_name": "Test",
                "contributor_id": "contrib_seed01",
            },
        )
        session_id = start_resp.json()["session"]["id"]

        resp = seeded_client.post(
            "/api/forge/discovery/answer",
            json={
                "session_id": session_id,
                "question_id": "q_bogus_99",
                "raw_text": "Answer text",
            },
        )
        assert resp.status_code == 400

    def test_get_discovery_progress(self, seeded_client: TestClient) -> None:
        """GET /discovery/{session_id}/progress returns progress."""
        start_resp = seeded_client.post(
            "/api/forge/discovery/start",
            json={
                "discipline_name": "Test Discipline",
                "contributor_id": "contrib_seed01",
            },
        )
        session_id = start_resp.json()["session"]["id"]

        resp = seeded_client.get(f"/api/forge/discovery/{session_id}/progress")
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_phase"] == "orientation"
        assert data["completion_percentage"] == 0.0
        assert len(data["unanswered_required"]) > 0

    def test_get_discovery_progress_not_found(self, client: TestClient) -> None:
        """GET /discovery/{session_id}/progress returns 404."""
        resp = client.get("/api/forge/discovery/dsess_missing/progress")
        assert resp.status_code == 404


# ===================================================================
# Consistency
# ===================================================================


class TestConsistency:
    """Tests for consistency check endpoints."""

    def test_check_consistency(self, seeded_client: TestClient) -> None:
        """POST /consistency/check/{id} runs consistency checks."""
        resp = seeded_client.post("/api/forge/consistency/check/disc_seed01")
        assert resp.status_code == 200
        data = resp.json()
        assert data["discipline_id"] == "disc_seed01"
        assert "issues" in data
        assert "example_count" in data
        assert isinstance(data["has_errors"], bool)
        assert isinstance(data["has_warnings"], bool)

    def test_check_consistency_not_found(self, client: TestClient) -> None:
        """POST /consistency/check/{id} returns 404 for missing discipline."""
        resp = client.post("/api/forge/consistency/check/disc_missing")
        assert resp.status_code == 404

    def test_get_consistency_report(self, seeded_client: TestClient) -> None:
        """GET /consistency/report/{id} returns a fresh report."""
        resp = seeded_client.get("/api/forge/consistency/report/disc_seed01")
        assert resp.status_code == 200
        assert resp.json()["discipline_id"] == "disc_seed01"


# ===================================================================
# Coverage
# ===================================================================


class TestCoverage:
    """Tests for coverage report endpoint."""

    def test_get_coverage(self, seeded_client: TestClient) -> None:
        """GET /coverage/{id} returns coverage data."""
        resp = seeded_client.get("/api/forge/coverage/disc_seed01")
        assert resp.status_code == 200
        data = resp.json()
        assert data["discipline_id"] == "disc_seed01"
        assert "total_examples" in data
        assert "competency_coverage" in data
        assert "gaps" in data
        assert isinstance(data["coverage_complete"], bool)

    def test_get_coverage_not_found(self, client: TestClient) -> None:
        """GET /coverage/{id} returns 404 for missing discipline."""
        resp = client.get("/api/forge/coverage/disc_missing")
        assert resp.status_code == 404

    def test_coverage_shows_gap(self, seeded_client: TestClient) -> None:
        """Coverage report shows gap when examples < target."""
        resp = seeded_client.get("/api/forge/coverage/disc_seed01")
        data = resp.json()
        # We have 1 example but target is 25, so there should be a gap
        assert data["coverage_complete"] is False
        assert len(data["gaps"]) > 0
        gap = data["gaps"][0]
        assert gap["example_count"] < gap["coverage_target"]


# ===================================================================
# Curriculum Export
# ===================================================================


class TestCurriculumExport:
    """Tests for curriculum export endpoint."""

    def test_export_curriculum(self, seeded_client: TestClient) -> None:
        """POST /curriculum/export/{id} creates a version snapshot."""
        resp = seeded_client.post(
            "/api/forge/curriculum/export/disc_seed01",
            json={"created_by": "contrib_seed01"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["discipline_id"] == "disc_seed01"
        assert data["version_number"] == 1
        assert data["example_count"] >= 1
        assert data["status"] == "draft"
        assert data["version_id"].startswith("curv_")

    def test_export_curriculum_increments_version(self, seeded_client: TestClient) -> None:
        """Successive exports increment version_number."""
        resp1 = seeded_client.post(
            "/api/forge/curriculum/export/disc_seed01",
            json={"created_by": "contrib_seed01"},
        )
        resp2 = seeded_client.post(
            "/api/forge/curriculum/export/disc_seed01",
            json={"created_by": "contrib_seed01"},
        )
        assert resp1.json()["version_number"] == 1
        assert resp2.json()["version_number"] == 2

    def test_export_curriculum_not_found(self, client: TestClient) -> None:
        """POST /curriculum/export/{id} returns 404 for missing discipline."""
        resp = client.post(
            "/api/forge/curriculum/export/disc_missing",
            json={"created_by": "contrib_test"},
        )
        assert resp.status_code == 404


# ===================================================================
# Integration: End-to-end create flow
# ===================================================================


class TestEndToEndFlow:
    """Integration tests exercising the full create-read-update cycle."""

    def test_contributor_discipline_competency_example_flow(self, client: TestClient) -> None:
        """Full CRUD flow: contributor -> discipline -> competency -> example."""
        # Create contributor
        resp = client.post(
            "/api/forge/contributors",
            json={"name": "Flow Tester", "email": "flow@test.com"},
        )
        assert resp.status_code == 201
        contrib_id = resp.json()["id"]

        # Create discipline
        resp = client.post(
            "/api/forge/disciplines",
            json={
                "name": "Test Flow Discipline",
                "description": "E2E test discipline",
                "created_by": contrib_id,
            },
        )
        assert resp.status_code == 201
        disc_id = resp.json()["id"]

        # Create competency
        resp = client.post(
            "/api/forge/competencies",
            json={
                "name": "Flow Competency",
                "description": "A test competency for the flow",
                "discipline_id": disc_id,
            },
        )
        assert resp.status_code == 201
        comp_id = resp.json()["id"]

        # Create example
        resp = client.post(
            "/api/forge/examples",
            json={
                "question": "What is the flow test procedure?",
                "ideal_answer": "Follow the checklist step by step.",
                "competency_id": comp_id,
                "contributor_id": contrib_id,
                "discipline_id": disc_id,
            },
        )
        assert resp.status_code == 201
        ex_id = resp.json()["id"]

        # Verify coverage shows the example
        resp = client.get(f"/api/forge/coverage/{disc_id}")
        assert resp.status_code == 200
        assert resp.json()["total_examples"] == 1

        # Verify consistency check runs
        resp = client.post(f"/api/forge/consistency/check/{disc_id}")
        assert resp.status_code == 200
        assert resp.json()["example_count"] == 1

        # Export curriculum
        resp = client.post(
            f"/api/forge/curriculum/export/{disc_id}",
            json={"created_by": contrib_id},
        )
        assert resp.status_code == 200
        assert resp.json()["example_count"] == 1

        # Clean up: delete example, competency
        assert client.delete(f"/api/forge/examples/{ex_id}").status_code == 200
        assert client.delete(f"/api/forge/competencies/{comp_id}").status_code == 200

    def test_discovery_full_flow(self, client: TestClient) -> None:
        """Discovery session: start, answer questions, check progress."""
        # Create contributor first
        resp = client.post(
            "/api/forge/contributors",
            json={"name": "Discovery Tester"},
        )
        contrib_id = resp.json()["id"]

        # Start session
        resp = client.post(
            "/api/forge/discovery/start",
            json={
                "discipline_name": "Avionics",
                "contributor_id": contrib_id,
            },
        )
        assert resp.status_code == 201
        session_id = resp.json()["session"]["id"]
        assert resp.json()["session"]["current_phase"] == "orientation"

        # Answer first question
        resp = client.post(
            "/api/forge/discovery/answer",
            json={
                "session_id": session_id,
                "question_id": "q_orient_01",
                "raw_text": "Avionics is the electronic systems on aircraft.",
            },
        )
        assert resp.status_code == 200

        # Check progress (should have some completion)
        resp = client.get(f"/api/forge/discovery/{session_id}/progress")
        assert resp.status_code == 200
        progress = resp.json()
        assert progress["completion_percentage"] > 0.0
        assert "q_orient_01" not in progress["unanswered_required"]
