"""Tests for the Foundry FastAPI server endpoints.

Uses httpx TestClient against the Foundry APIRouter mounted on
a minimal FastAPI app. Each test operates against in-memory state
and temporary directories to avoid side effects.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from foundry.src.server import _state, router


@pytest.fixture(autouse=True)
def _reset_server_state(tmp_path: Path) -> None:
    """Reset in-memory server state before each test.

    Ensures tests are fully isolated. Sets data_dir to a
    temporary directory so file-based registries do not persist.
    """
    _state["pipelines"] = {}
    _state["training_registry"] = None
    _state["evaluation_history"] = None
    _state["version_manager"] = None
    _state["regression_runner"] = None
    _state["merge_registry"] = None
    _state["data_dir"] = tmp_path / "foundry_data"
    _state["data_dir"].mkdir(parents=True, exist_ok=True)


@pytest.fixture
def app() -> FastAPI:
    """Create a minimal FastAPI app with the Foundry router mounted."""
    test_app = FastAPI()
    test_app.include_router(router, prefix="/api/foundry")
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Return a TestClient for the test app."""
    return TestClient(app)


@pytest.fixture
def curriculum_file(tmp_path: Path) -> Path:
    """Write a valid 20-record JSONL curriculum file.

    Returns:
        Path to the curriculum JSONL file.
    """
    path = tmp_path / "curriculum.jsonl"
    records = []
    for i in range(20):
        comp_id = "comp_proc" if i < 10 else "comp_fault"
        records.append(
            {
                "instruction": f"How do you perform procedure {i}?",
                "input": "",
                "output": (
                    f"Step 1: Inspect component {i}. "
                    f"Step 2: Apply torque. "
                    f"Step 3: Verify clearance."
                ),
                "metadata": {
                    "example_id": f"ex_{i:04d}",
                    "discipline_id": "disc_maint",
                    "competency_id": comp_id,
                },
            }
        )
    lines = [json.dumps(r) for r in records]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture
def test_set_file(tmp_path: Path) -> Path:
    """Write a valid 5-record test set JSONL file.

    Returns:
        Path to the test set JSONL file.
    """
    path = tmp_path / "test_set.jsonl"
    records = [
        {
            "instruction": f"Test question {i}?",
            "input": "",
            "output": f"Answer {i}.",
            "metadata": {
                "example_id": f"test_ex_{i:03d}",
                "discipline_id": "disc_maint",
                "competency_id": "comp_proc" if i < 3 else "comp_fault",
            },
        }
        for i in range(5)
    ]
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return path


# ===================================================================
# Health endpoint tests
# ===================================================================


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_ok(self, client: TestClient) -> None:
        """Health check returns status ok."""
        resp = client.get("/api/foundry/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "foundry"
        assert "version" in data


# ===================================================================
# Training endpoint tests
# ===================================================================


class TestTrainingConfigure:
    """Tests for POST /training/configure."""

    def test_manual_configure(
        self, client: TestClient, curriculum_file: Path, tmp_path: Path
    ) -> None:
        """Manual configuration returns run_id and config."""
        resp = client.post(
            "/api/foundry/training/configure",
            json={
                "base_model_family": "phi",
                "curriculum_path": str(curriculum_file),
                "output_dir": str(tmp_path / "output"),
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert data["status"] == "pending"
        assert data["config"]["base_model_family"] == "phi"

    def test_auto_configure(
        self, client: TestClient, curriculum_file: Path, tmp_path: Path
    ) -> None:
        """Auto-configuration tunes hyperparameters from curriculum size."""
        resp = client.post(
            "/api/foundry/training/configure",
            json={
                "base_model_family": "phi",
                "curriculum_path": str(curriculum_file),
                "output_dir": str(tmp_path / "output"),
                "auto_configure": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert data["config"]["epochs"] == 5  # <100 examples -> 5 epochs

    def test_configure_invalid_family(
        self, client: TestClient, curriculum_file: Path, tmp_path: Path
    ) -> None:
        """Invalid model family returns 400."""
        resp = client.post(
            "/api/foundry/training/configure",
            json={
                "base_model_family": "nonexistent",
                "curriculum_path": str(curriculum_file),
                "output_dir": str(tmp_path / "output"),
            },
        )
        assert resp.status_code == 400


class TestTrainingStart:
    """Tests for POST /training/start."""

    def test_start_training(
        self, client: TestClient, curriculum_file: Path, tmp_path: Path
    ) -> None:
        """Starting a configured run returns completed status."""
        configure_resp = client.post(
            "/api/foundry/training/configure",
            json={
                "base_model_family": "phi",
                "curriculum_path": str(curriculum_file),
                "output_dir": str(tmp_path / "output"),
            },
        )
        run_id = configure_resp.json()["run_id"]

        start_resp = client.post(
            "/api/foundry/training/start",
            params={"run_id": run_id},
        )
        assert start_resp.status_code == 200
        data = start_resp.json()
        assert data["status"] == "completed"
        assert data["total_examples"] == 20

    def test_start_unknown_run(self, client: TestClient) -> None:
        """Starting a non-existent run returns 404."""
        resp = client.post(
            "/api/foundry/training/start",
            params={"run_id": "run_nonexistent"},
        )
        assert resp.status_code == 404


class TestTrainingStatus:
    """Tests for GET /training/{run_id}/status."""

    def test_status_pending(
        self, client: TestClient, curriculum_file: Path, tmp_path: Path
    ) -> None:
        """Uncompleted run shows pending status."""
        configure_resp = client.post(
            "/api/foundry/training/configure",
            json={
                "base_model_family": "phi",
                "curriculum_path": str(curriculum_file),
                "output_dir": str(tmp_path / "output"),
            },
        )
        run_id = configure_resp.json()["run_id"]

        status_resp = client.get(f"/api/foundry/training/{run_id}/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "pending"

    def test_status_after_training(
        self, client: TestClient, curriculum_file: Path, tmp_path: Path
    ) -> None:
        """Completed run includes result dict."""
        configure_resp = client.post(
            "/api/foundry/training/configure",
            json={
                "base_model_family": "phi",
                "curriculum_path": str(curriculum_file),
                "output_dir": str(tmp_path / "output"),
            },
        )
        run_id = configure_resp.json()["run_id"]
        client.post("/api/foundry/training/start", params={"run_id": run_id})

        status_resp = client.get(f"/api/foundry/training/{run_id}/status")
        data = status_resp.json()
        assert data["status"] == "completed"
        assert "result" in data
        assert data["result"]["status"] == "completed"

    def test_status_unknown_run(self, client: TestClient) -> None:
        """Unknown run_id returns 404."""
        resp = client.get("/api/foundry/training/run_fake/status")
        assert resp.status_code == 404


class TestTrainingCancel:
    """Tests for POST /training/{run_id}/cancel."""

    def test_cancel_pending_run(
        self, client: TestClient, curriculum_file: Path, tmp_path: Path
    ) -> None:
        """Cancelling a pending run sets status to cancelled."""
        configure_resp = client.post(
            "/api/foundry/training/configure",
            json={
                "base_model_family": "phi",
                "curriculum_path": str(curriculum_file),
                "output_dir": str(tmp_path / "output"),
            },
        )
        run_id = configure_resp.json()["run_id"]

        cancel_resp = client.post(f"/api/foundry/training/{run_id}/cancel")
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["status"] == "cancelled"

    def test_cancel_unknown_run(self, client: TestClient) -> None:
        """Cancelling unknown run returns 404."""
        resp = client.post("/api/foundry/training/run_fake/cancel")
        assert resp.status_code == 404


class TestTrainingRuns:
    """Tests for GET /training/runs."""

    def test_list_empty(self, client: TestClient) -> None:
        """Empty registry returns zero runs."""
        resp = client.get("/api/foundry/training/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["runs"] == []

    def test_list_after_training(
        self, client: TestClient, curriculum_file: Path, tmp_path: Path
    ) -> None:
        """After training completes, the run appears in the registry."""
        configure_resp = client.post(
            "/api/foundry/training/configure",
            json={
                "base_model_family": "phi",
                "curriculum_path": str(curriculum_file),
                "output_dir": str(tmp_path / "output"),
            },
        )
        run_id = configure_resp.json()["run_id"]
        client.post("/api/foundry/training/start", params={"run_id": run_id})

        list_resp = client.get("/api/foundry/training/runs")
        data = list_resp.json()
        assert data["total"] >= 1


# ===================================================================
# Evaluation endpoint tests
# ===================================================================


class TestEvaluationRun:
    """Tests for POST /evaluation/run."""

    def test_run_evaluation(self, client: TestClient, test_set_file: Path) -> None:
        """Running evaluation returns a report with competency scores."""
        resp = client.post(
            "/api/foundry/evaluation/run",
            json={
                "test_set_path": str(test_set_file),
                "competency_names": {
                    "comp_proc": "Procedural Comprehension",
                    "comp_fault": "Fault Isolation",
                },
                "model_name": "test-model",
                "discipline_id": "disc_maint",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert data["status"] == "completed"
        assert data["total_cases"] == 5
        assert "competency_scores" in data

    def test_run_evaluation_missing_file(self, client: TestClient) -> None:
        """Non-existent test set returns 400."""
        resp = client.post(
            "/api/foundry/evaluation/run",
            json={
                "test_set_path": "/nonexistent/path.jsonl",
                "competency_names": {},
                "model_name": "test-model",
                "discipline_id": "disc_maint",
            },
        )
        assert resp.status_code == 400


class TestEvaluationGet:
    """Tests for GET /evaluation/{eval_id}."""

    def test_get_saved_evaluation(self, client: TestClient, test_set_file: Path) -> None:
        """A saved evaluation can be retrieved by ID."""
        run_resp = client.post(
            "/api/foundry/evaluation/run",
            json={
                "test_set_path": str(test_set_file),
                "competency_names": {"comp_proc": "Proc", "comp_fault": "Fault"},
                "model_name": "test-model",
                "discipline_id": "disc_maint",
            },
        )
        eval_id = run_resp.json()["run_id"]

        get_resp = client.get(f"/api/foundry/evaluation/{eval_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["run_id"] == eval_id

    def test_get_unknown_evaluation(self, client: TestClient) -> None:
        """Unknown eval_id returns 404."""
        resp = client.get("/api/foundry/evaluation/eval_nonexistent")
        assert resp.status_code == 404


class TestEvaluationCompare:
    """Tests for GET /evaluation/compare."""

    def test_compare_two_evaluations(self, client: TestClient, test_set_file: Path) -> None:
        """Comparing two runs returns accuracy delta."""
        # Run two evaluations
        resp_a = client.post(
            "/api/foundry/evaluation/run",
            json={
                "test_set_path": str(test_set_file),
                "competency_names": {"comp_proc": "Proc", "comp_fault": "Fault"},
                "model_name": "model-v1",
                "discipline_id": "disc_maint",
            },
        )
        resp_b = client.post(
            "/api/foundry/evaluation/run",
            json={
                "test_set_path": str(test_set_file),
                "competency_names": {"comp_proc": "Proc", "comp_fault": "Fault"},
                "model_name": "model-v2",
                "discipline_id": "disc_maint",
            },
        )
        eval_id_a = resp_a.json()["run_id"]
        eval_id_b = resp_b.json()["run_id"]

        compare_resp = client.get(
            "/api/foundry/evaluation/compare",
            params={"eval_id_a": eval_id_a, "eval_id_b": eval_id_b},
        )
        assert compare_resp.status_code == 200
        data = compare_resp.json()
        assert "eval_a" in data
        assert "eval_b" in data
        assert "accuracy_delta" in data


# ===================================================================
# Diagnostics endpoint tests
# ===================================================================


class TestDiagnosticsAnalyze:
    """Tests for POST /diagnostics/analyze/{run_id}."""

    def test_analyze_with_metrics(self, client: TestClient) -> None:
        """Providing metrics returns a diagnostic report."""
        metrics = [
            {
                "epoch": i,
                "step": i * 10,
                "train_loss": 2.0 / (1 + i * 0.1),
                "val_loss": 2.2 / (1 + i * 0.1),
                "learning_rate": 2e-4,
                "timestamp": datetime.now().isoformat(),
            }
            for i in range(10)
        ]

        resp = client.post(
            "/api/foundry/diagnostics/analyze/run_test123",
            json={"metrics": metrics},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == "run_test123"
        assert "overall_health" in data
        assert "issues" in data
        assert "trends" in data

    def test_analyze_with_curriculum_stats(self, client: TestClient) -> None:
        """Including curriculum stats adds data quality checks."""
        metrics = [
            {
                "epoch": i,
                "step": i * 10,
                "train_loss": 2.0 / (1 + i * 0.1),
                "timestamp": datetime.now().isoformat(),
            }
            for i in range(5)
        ]

        resp = client.post(
            "/api/foundry/diagnostics/analyze/run_test456",
            json={
                "metrics": metrics,
                "curriculum_stats": {
                    "total_examples": 10,
                    "competency_counts": {"comp_a": 9, "comp_b": 1},
                },
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall_health"] in ("healthy", "warning", "critical")

    def test_analyze_empty_metrics(self, client: TestClient) -> None:
        """Empty metrics list returns 400."""
        resp = client.post(
            "/api/foundry/diagnostics/analyze/run_empty",
            json={"metrics": []},
        )
        assert resp.status_code == 400


class TestDiagnosticsGet:
    """Tests for GET /diagnostics/{run_id}."""

    def test_get_diagnostics_after_training(
        self, client: TestClient, curriculum_file: Path, tmp_path: Path
    ) -> None:
        """Diagnostics for a completed run returns a report."""
        configure_resp = client.post(
            "/api/foundry/training/configure",
            json={
                "base_model_family": "phi",
                "curriculum_path": str(curriculum_file),
                "output_dir": str(tmp_path / "output"),
            },
        )
        run_id = configure_resp.json()["run_id"]
        client.post("/api/foundry/training/start", params={"run_id": run_id})

        diag_resp = client.get(f"/api/foundry/diagnostics/{run_id}")
        assert diag_resp.status_code == 200
        data = diag_resp.json()
        assert data["run_id"] == run_id
        assert "overall_health" in data

    def test_get_diagnostics_unknown_run(self, client: TestClient) -> None:
        """Unknown run returns 404."""
        resp = client.get("/api/foundry/diagnostics/run_fake")
        assert resp.status_code == 404

    def test_get_diagnostics_no_result(
        self, client: TestClient, curriculum_file: Path, tmp_path: Path
    ) -> None:
        """Run without results returns 400."""
        configure_resp = client.post(
            "/api/foundry/training/configure",
            json={
                "base_model_family": "phi",
                "curriculum_path": str(curriculum_file),
                "output_dir": str(tmp_path / "output"),
            },
        )
        run_id = configure_resp.json()["run_id"]

        diag_resp = client.get(f"/api/foundry/diagnostics/{run_id}")
        assert diag_resp.status_code == 400


# ===================================================================
# Regression endpoint tests
# ===================================================================


class TestRegressionCheck:
    """Tests for POST /regression/check."""

    def test_regression_check(self, client: TestClient, test_set_file: Path) -> None:
        """Regression check between two evals returns a report."""
        # Create two evaluations
        resp_a = client.post(
            "/api/foundry/evaluation/run",
            json={
                "test_set_path": str(test_set_file),
                "competency_names": {"comp_proc": "Proc", "comp_fault": "Fault"},
                "model_name": "baseline",
                "discipline_id": "disc_maint",
            },
        )
        resp_b = client.post(
            "/api/foundry/evaluation/run",
            json={
                "test_set_path": str(test_set_file),
                "competency_names": {"comp_proc": "Proc", "comp_fault": "Fault"},
                "model_name": "current",
                "discipline_id": "disc_maint",
            },
        )

        eval_id_a = resp_a.json()["run_id"]
        eval_id_b = resp_b.json()["run_id"]

        check_resp = client.post(
            "/api/foundry/regression/check",
            json={
                "baseline_eval_id": eval_id_a,
                "current_eval_id": eval_id_b,
                "change_type": "retrain",
            },
        )
        assert check_resp.status_code == 200
        data = check_resp.json()
        assert data["overall_verdict"] in ("pass", "warn", "fail")
        assert "plain_language_summary" in data


class TestRegressionVersions:
    """Tests for GET /regression/versions and POST /regression/register."""

    def test_list_empty_versions(self, client: TestClient) -> None:
        """Empty version registry returns zero entries."""
        resp = client.get("/api/foundry/regression/versions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    def test_register_and_list_version(self, client: TestClient) -> None:
        """Registering a version makes it appear in the list."""
        register_resp = client.post(
            "/api/foundry/regression/register",
            json={
                "model_name": "maint-lora-v1",
                "discipline_id": "disc_maint",
                "evaluation_run_id": "eval_test001",
                "change_type": "retrain",
                "change_description": "Initial training run.",
            },
        )
        assert register_resp.status_code == 200
        version_data = register_resp.json()
        assert version_data["version_id"].startswith("ver_")
        assert version_data["is_active"] is True

        list_resp = client.get("/api/foundry/regression/versions")
        data = list_resp.json()
        assert data["total"] == 1

    def test_filter_versions_by_discipline(self, client: TestClient) -> None:
        """Discipline filter narrows version list."""
        # Register two versions for different disciplines
        client.post(
            "/api/foundry/regression/register",
            json={
                "model_name": "maint-v1",
                "discipline_id": "disc_maint",
                "evaluation_run_id": "eval_001",
                "change_type": "retrain",
            },
        )
        client.post(
            "/api/foundry/regression/register",
            json={
                "model_name": "weapons-v1",
                "discipline_id": "disc_weapons",
                "evaluation_run_id": "eval_002",
                "change_type": "retrain",
            },
        )

        # Filter by discipline
        resp = client.get(
            "/api/foundry/regression/versions",
            params={"discipline_id": "disc_maint"},
        )
        data = resp.json()
        assert data["total"] == 1
        assert data["versions"][0]["discipline_id"] == "disc_maint"


# ===================================================================
# Merging endpoint tests
# ===================================================================


class TestMergingMerge:
    """Tests for POST /merging/merge."""

    def test_linear_merge(self, client: TestClient, tmp_path: Path) -> None:
        """Linear merge of two compatible adapters succeeds."""
        adapter_a = tmp_path / "adapter_a"
        adapter_b = tmp_path / "adapter_b"
        adapter_a.mkdir()
        adapter_b.mkdir()

        resp = client.post(
            "/api/foundry/merging/merge",
            json={
                "adapters": [
                    {
                        "adapter_path": str(adapter_a),
                        "discipline_id": "disc_hydraulics",
                        "discipline_name": "Hydraulic Systems",
                        "base_model": "microsoft/phi-3-mini-4k-instruct",
                        "base_model_family": "phi",
                        "lora_rank": 16,
                    },
                    {
                        "adapter_path": str(adapter_b),
                        "discipline_id": "disc_electrical",
                        "discipline_name": "Electrical Systems",
                        "base_model": "microsoft/phi-3-mini-4k-instruct",
                        "base_model_family": "phi",
                        "lora_rank": 16,
                    },
                ],
                "method": "linear",
                "output_dir": str(tmp_path / "merged"),
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["method"] == "linear"
        assert len(data["adapters"]) == 2

    def test_incompatible_merge_returns_400(self, client: TestClient, tmp_path: Path) -> None:
        """Merging adapters with different base models returns 400."""
        adapter_a = tmp_path / "adapter_a"
        adapter_b = tmp_path / "adapter_b"
        adapter_a.mkdir()
        adapter_b.mkdir()

        resp = client.post(
            "/api/foundry/merging/merge",
            json={
                "adapters": [
                    {
                        "adapter_path": str(adapter_a),
                        "discipline_id": "disc_a",
                        "discipline_name": "Disc A",
                        "base_model": "model-alpha",
                        "base_model_family": "phi",
                        "lora_rank": 16,
                    },
                    {
                        "adapter_path": str(adapter_b),
                        "discipline_id": "disc_b",
                        "discipline_name": "Disc B",
                        "base_model": "model-beta",
                        "base_model_family": "phi",
                        "lora_rank": 16,
                    },
                ],
                "method": "linear",
            },
        )
        assert resp.status_code == 400


class TestMergingRegistry:
    """Tests for GET /merging/registry."""

    def test_list_empty_registry(self, client: TestClient) -> None:
        """Empty registry returns zero merges."""
        resp = client.get("/api/foundry/merging/registry")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    def test_list_after_merge(self, client: TestClient, tmp_path: Path) -> None:
        """After a merge, it appears in the registry."""
        adapter_a = tmp_path / "adapter_a"
        adapter_b = tmp_path / "adapter_b"
        adapter_a.mkdir()
        adapter_b.mkdir()

        client.post(
            "/api/foundry/merging/merge",
            json={
                "adapters": [
                    {
                        "adapter_path": str(adapter_a),
                        "discipline_id": "disc_a",
                        "discipline_name": "Disc A",
                        "base_model": "microsoft/phi-3-mini-4k-instruct",
                        "base_model_family": "phi",
                        "lora_rank": 16,
                    },
                    {
                        "adapter_path": str(adapter_b),
                        "discipline_id": "disc_b",
                        "discipline_name": "Disc B",
                        "base_model": "microsoft/phi-3-mini-4k-instruct",
                        "base_model_family": "phi",
                        "lora_rank": 16,
                    },
                ],
                "method": "linear",
                "output_dir": str(tmp_path / "merged"),
            },
        )

        resp = client.get("/api/foundry/merging/registry")
        data = resp.json()
        assert data["total"] == 1


class TestMergingMethods:
    """Tests for GET /merging/methods."""

    def test_list_methods(self, client: TestClient) -> None:
        """Returns both LINEAR and TIES methods."""
        resp = client.get("/api/foundry/merging/methods")
        assert resp.status_code == 200
        data = resp.json()
        method_names = [m["name"] for m in data["methods"]]
        assert "linear" in method_names
        assert "ties" in method_names
