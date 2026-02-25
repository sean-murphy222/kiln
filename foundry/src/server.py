"""FastAPI router for the Foundry backend.

Exposes REST endpoints for LoRA training, competency-based evaluation,
training diagnostics, regression testing, version management, and model
merging. Mounted by the parent application at ``/api/foundry/``.

All request and response bodies use Pydantic models for validation.
Endpoints wrap calls to the underlying Foundry modules and translate
domain exceptions into proper HTTP error responses.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from foundry.src.diagnostics import (
    DiagnosticsError,
    MetricSnapshot,
    TrainingDiagnostics,
)
from foundry.src.evaluation import (
    EvaluationError,
    EvaluationHistory,
    EvaluationRunner,
    MockInference,
)
from foundry.src.merging import (
    AdapterInfo,
    MergeConfig,
    MergeMethod,
    MergePipeline,
    MergeRegistry,
    MergingError,
)
from foundry.src.regression import (
    ChangeType,
    RegressionChecker,
    RegressionError,
    RegressionRunner,
    VersionEntry,
    VersionManager,
)
from foundry.src.training import (
    BaseModelFamily,
    HyperparameterAutoConfig,
    TrainingConfig,
    TrainingError,
    TrainingPipeline,
    TrainingRegistry,
    TrainingStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ===================================================================
# In-memory state
# ===================================================================

_state: dict[str, Any] = {
    "pipelines": {},
    "training_registry": None,
    "evaluation_history": None,
    "version_manager": None,
    "regression_runner": None,
    "merge_registry": None,
    "data_dir": None,
}


def _get_data_dir() -> Path:
    """Return the Foundry data directory, creating it if needed.

    Returns:
        Path to the data directory.
    """
    if _state["data_dir"] is not None:
        return _state["data_dir"]
    data_dir = Path("data/foundry")
    data_dir.mkdir(parents=True, exist_ok=True)
    _state["data_dir"] = data_dir
    return data_dir


def _get_training_registry() -> TrainingRegistry:
    """Return the shared TrainingRegistry, creating it if needed.

    Returns:
        TrainingRegistry instance.
    """
    if _state["training_registry"] is None:
        registry_dir = _get_data_dir() / "training_registry"
        _state["training_registry"] = TrainingRegistry(registry_dir)
    return _state["training_registry"]


def _get_evaluation_history() -> EvaluationHistory:
    """Return the shared EvaluationHistory, creating it if needed.

    Returns:
        EvaluationHistory instance.
    """
    if _state["evaluation_history"] is None:
        history_dir = _get_data_dir() / "evaluation_history"
        _state["evaluation_history"] = EvaluationHistory(history_dir)
    return _state["evaluation_history"]


def _get_version_manager() -> VersionManager:
    """Return the shared VersionManager, creating it if needed.

    Returns:
        VersionManager instance.
    """
    if _state["version_manager"] is None:
        versions_dir = _get_data_dir() / "versions"
        _state["version_manager"] = VersionManager(versions_dir)
    return _state["version_manager"]


def _get_regression_runner() -> RegressionRunner:
    """Return the shared RegressionRunner, creating it if needed.

    Returns:
        RegressionRunner instance.
    """
    if _state["regression_runner"] is None:
        checker = RegressionChecker()
        vm = _get_version_manager()
        reports_dir = _get_data_dir() / "regression_reports"
        _state["regression_runner"] = RegressionRunner(checker, vm, reports_dir)
    return _state["regression_runner"]


def _get_merge_registry() -> MergeRegistry:
    """Return the shared MergeRegistry, creating it if needed.

    Returns:
        MergeRegistry instance.
    """
    if _state["merge_registry"] is None:
        registry_dir = _get_data_dir() / "merge_registry"
        _state["merge_registry"] = MergeRegistry(registry_dir)
    return _state["merge_registry"]


# ===================================================================
# Pydantic request/response models
# ===================================================================


class HealthResponse(BaseModel):
    """Response for the health check endpoint."""

    status: str
    service: str
    version: str


class LoRAConfigRequest(BaseModel):
    """LoRA adapter configuration in a training request."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])


class TrainingConfigureRequest(BaseModel):
    """Request body for POST /training/configure."""

    base_model_family: str
    curriculum_path: str
    output_dir: str
    base_model: str | None = None
    lora: LoRAConfigRequest | None = None
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    seed: int = 42
    validation_split: float = 0.1
    auto_configure: bool = False


class TrainingStartResponse(BaseModel):
    """Response for POST /training/start."""

    run_id: str
    status: str
    message: str


class TrainingStatusResponse(BaseModel):
    """Response for GET /training/{run_id}/status."""

    run_id: str
    status: str
    result: dict[str, Any] | None = None


class TrainingCancelResponse(BaseModel):
    """Response for POST /training/{run_id}/cancel."""

    run_id: str
    status: str
    message: str


class EvaluationRunRequest(BaseModel):
    """Request body for POST /evaluation/run."""

    test_set_path: str
    competency_names: dict[str, str]
    model_name: str
    discipline_id: str


class EvaluationCompareRequest(BaseModel):
    """Query parameters for GET /evaluation/compare."""

    eval_id_a: str
    eval_id_b: str


class DiagnosticsAnalyzeRequest(BaseModel):
    """Request body for POST /diagnostics/analyze/{run_id}."""

    metrics: list[dict[str, Any]]
    curriculum_stats: dict[str, Any] | None = None


class RegressionCheckRequest(BaseModel):
    """Request body for POST /regression/check."""

    baseline_eval_id: str
    current_eval_id: str
    change_type: str


class VersionRegisterRequest(BaseModel):
    """Request body for POST /regression/register."""

    model_name: str
    discipline_id: str
    evaluation_run_id: str
    training_run_id: str | None = None
    adapter_path: str | None = None
    change_type: str = "retrain"
    change_description: str = ""


class MergeRequest(BaseModel):
    """Request body for POST /merging/merge."""

    adapters: list[dict[str, Any]]
    method: str = "linear"
    weights: list[float] | None = None
    ties_density: float = 0.5
    ties_majority_sign: bool = True
    output_dir: str | None = None
    validate_compatibility: bool = True


# ===================================================================
# Health endpoint
# ===================================================================


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Return service health status.

    Returns:
        Dict with status, service name, and version.
    """
    return {
        "status": "ok",
        "service": "foundry",
        "version": "0.1.0",
    }


# ===================================================================
# Training endpoints
# ===================================================================


@router.post("/training/configure")
async def configure_training(
    request: TrainingConfigureRequest,
) -> dict[str, Any]:
    """Configure a training run and return a run_id.

    If ``auto_configure`` is true, hyperparameters are automatically
    tuned based on the curriculum size and model family.

    Args:
        request: Training configuration parameters.

    Returns:
        Dict with run_id and configuration details.
    """
    try:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        curriculum_path = Path(request.curriculum_path)
        output_dir = Path(request.output_dir)

        if request.auto_configure:
            config = _auto_configure_training(request, curriculum_path, output_dir)
        else:
            config = _manual_configure_training(request, curriculum_path, output_dir)

        pipeline = TrainingPipeline(config)
        _state["pipelines"][run_id] = {
            "pipeline": pipeline,
            "config": config,
            "status": TrainingStatus.PENDING.value,
            "result": None,
        }

        return {
            "run_id": run_id,
            "config": config.to_dict(),
            "status": TrainingStatus.PENDING.value,
        }
    except (TrainingError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _auto_configure_training(
    request: TrainingConfigureRequest,
    curriculum_path: Path,
    output_dir: Path,
) -> TrainingConfig:
    """Build a TrainingConfig using HyperparameterAutoConfig.

    Args:
        request: The incoming request.
        curriculum_path: Path to the JSONL curriculum.
        output_dir: Output directory for artifacts.

    Returns:
        Auto-tuned TrainingConfig.
    """
    from foundry.src.training import CurriculumLoader

    loader = CurriculumLoader()
    records = loader.load(curriculum_path)
    auto = HyperparameterAutoConfig()
    family = BaseModelFamily(request.base_model_family)
    return auto.configure(
        curriculum_size=len(records),
        base_family=family,
        curriculum_path=curriculum_path,
        output_dir=output_dir,
    )


def _manual_configure_training(
    request: TrainingConfigureRequest,
    curriculum_path: Path,
    output_dir: Path,
) -> TrainingConfig:
    """Build a TrainingConfig from explicit request parameters.

    Args:
        request: The incoming request.
        curriculum_path: Path to the JSONL curriculum.
        output_dir: Output directory for artifacts.

    Returns:
        Manually configured TrainingConfig.
    """
    from foundry.src.training import LoRAConfig

    family = BaseModelFamily(request.base_model_family)
    base_model = request.base_model
    if base_model is None:
        from foundry.src.training import _DEFAULT_BASE_MODELS

        base_model = _DEFAULT_BASE_MODELS[family]

    lora_data = request.lora
    if lora_data is not None:
        lora = LoRAConfig(
            rank=lora_data.rank,
            alpha=lora_data.alpha,
            dropout=lora_data.dropout,
            target_modules=lora_data.target_modules,
        )
    else:
        lora = LoRAConfig()

    return TrainingConfig(
        base_model=base_model,
        base_model_family=family,
        curriculum_path=curriculum_path,
        output_dir=output_dir,
        lora=lora,
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        max_seq_length=request.max_seq_length,
        warmup_ratio=request.warmup_ratio,
        weight_decay=request.weight_decay,
        gradient_accumulation_steps=request.gradient_accumulation_steps,
        seed=request.seed,
        validation_split=request.validation_split,
    )


@router.post("/training/start")
async def start_training(run_id: str) -> dict[str, Any]:
    """Start a previously configured training run.

    Prepares and runs the pipeline (dry-run in MVP), then
    registers the result in the training registry.

    Args:
        run_id: The run identifier from /training/configure.

    Returns:
        Dict with run_id, status, and result summary.
    """
    run_data = _state["pipelines"].get(run_id)
    if run_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Training run not found: {run_id}",
        )

    pipeline: TrainingPipeline = run_data["pipeline"]

    try:
        pipeline.prepare()
        run_data["status"] = TrainingStatus.PREPARING.value
        result = pipeline.run()
        run_data["status"] = result.status.value
        run_data["result"] = result

        registry = _get_training_registry()
        training_run = registry.register_run(result)

        return {
            "run_id": run_id,
            "registry_run_id": training_run.run_id,
            "status": result.status.value,
            "total_examples": result.total_examples,
            "training_examples": result.training_examples,
            "validation_examples": result.validation_examples,
            "adapter_path": (str(result.adapter_path) if result.adapter_path else None),
        }
    except TrainingError as exc:
        run_data["status"] = TrainingStatus.FAILED.value
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/training/{run_id}/status")
async def get_training_status(run_id: str) -> dict[str, Any]:
    """Get the current status of a training run.

    Args:
        run_id: The run identifier.

    Returns:
        Dict with run_id, status, and optional result.
    """
    run_data = _state["pipelines"].get(run_id)
    if run_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Training run not found: {run_id}",
        )

    response: dict[str, Any] = {
        "run_id": run_id,
        "status": run_data["status"],
    }

    result = run_data.get("result")
    if result is not None:
        response["result"] = result.to_dict()

    return response


@router.post("/training/{run_id}/cancel")
async def cancel_training(run_id: str) -> dict[str, Any]:
    """Cancel a training run.

    Args:
        run_id: The run identifier.

    Returns:
        Dict with run_id, status, and confirmation message.
    """
    run_data = _state["pipelines"].get(run_id)
    if run_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Training run not found: {run_id}",
        )

    pipeline: TrainingPipeline = run_data["pipeline"]
    pipeline.cancel()
    run_data["status"] = TrainingStatus.CANCELLED.value

    return {
        "run_id": run_id,
        "status": TrainingStatus.CANCELLED.value,
        "message": "Training run cancelled.",
    }


@router.get("/training/runs")
async def list_training_runs(
    discipline_id: str | None = None,
) -> dict[str, Any]:
    """List registered training runs.

    Args:
        discipline_id: Optional discipline filter.

    Returns:
        Dict with a list of training run summaries.
    """
    try:
        registry = _get_training_registry()
        runs = registry.list_runs(discipline_id=discipline_id)
        return {
            "runs": [r.to_dict() for r in runs],
            "total": len(runs),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ===================================================================
# Evaluation endpoints
# ===================================================================


@router.post("/evaluation/run")
async def run_evaluation(request: EvaluationRunRequest) -> dict[str, Any]:
    """Run a competency-based evaluation.

    Uses MockInference in MVP mode. Loads test cases from a
    Forge-exported JSONL file and produces an SME-friendly report.

    Args:
        request: Evaluation parameters.

    Returns:
        Dict with the full evaluation report.
    """
    try:
        runner = EvaluationRunner()
        test_set_path = Path(request.test_set_path)
        test_cases = runner.load_test_cases(test_set_path)

        model = MockInference(default_response="I don't know.")
        report = runner.run_evaluation(
            model=model,
            test_cases=test_cases,
            competency_names=request.competency_names,
            model_name=request.model_name,
            discipline_id=request.discipline_id,
        )

        history = _get_evaluation_history()
        history.save_report(report)

        return report.to_dict()
    except EvaluationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"File not found: {exc}") from exc


@router.get("/evaluation/compare")
async def compare_evaluations(
    eval_id_a: str,
    eval_id_b: str,
) -> dict[str, Any]:
    """Compare two evaluation reports side by side.

    Returns accuracy and competency differences between two
    evaluation runs. This route is registered before the
    ``/evaluation/{eval_id}`` wildcard to avoid path conflicts.

    Args:
        eval_id_a: First evaluation run identifier.
        eval_id_b: Second evaluation run identifier.

    Returns:
        Dict with side-by-side comparison data.
    """
    try:
        history = _get_evaluation_history()
        report_a = history.load_report(eval_id_a)
        report_b = history.load_report(eval_id_b)

        return {
            "eval_a": {
                "run_id": report_a.run_id,
                "model_name": report_a.model_name,
                "overall_accuracy": report_a.overall_accuracy,
                "overall_rating": report_a.overall_rating.value,
                "competency_scores": {
                    k: v.to_dict() for k, v in report_a.competency_scores.items()
                },
            },
            "eval_b": {
                "run_id": report_b.run_id,
                "model_name": report_b.model_name,
                "overall_accuracy": report_b.overall_accuracy,
                "overall_rating": report_b.overall_rating.value,
                "competency_scores": {
                    k: v.to_dict() for k, v in report_b.competency_scores.items()
                },
            },
            "accuracy_delta": (report_b.overall_accuracy - report_a.overall_accuracy),
        }
    except EvaluationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/evaluation/{eval_id}")
async def get_evaluation(eval_id: str) -> dict[str, Any]:
    """Retrieve a saved evaluation report by ID.

    Args:
        eval_id: The evaluation run identifier.

    Returns:
        Dict with the full evaluation report.
    """
    try:
        history = _get_evaluation_history()
        report = history.load_report(eval_id)
        return report.to_dict()
    except EvaluationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ===================================================================
# Diagnostics endpoints
# ===================================================================


@router.post("/diagnostics/analyze/{run_id}")
async def analyze_diagnostics(
    run_id: str,
    request: DiagnosticsAnalyzeRequest,
) -> dict[str, Any]:
    """Analyze training metrics for a run and produce diagnostics.

    Accepts metric snapshots as JSON and returns a plain-language
    diagnostic report with detected issues, trends, and Forge
    recommendations.

    Args:
        run_id: The training run identifier (for labeling).
        request: Metrics and optional curriculum stats.

    Returns:
        Dict with the full diagnostic report.
    """
    try:
        snapshots = _parse_metric_snapshots(request.metrics)
        diag = TrainingDiagnostics()

        if request.curriculum_stats is not None:
            report = diag.analyze_full(snapshots, request.curriculum_stats)
        else:
            report = diag.analyze_training(snapshots)

        report_dict = report.to_dict()
        report_dict["run_id"] = run_id
        return report_dict
    except DiagnosticsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metrics data: {exc}",
        ) from exc


def _parse_metric_snapshots(
    raw_metrics: list[dict[str, Any]],
) -> list[MetricSnapshot]:
    """Parse raw metric dicts into MetricSnapshot objects.

    Args:
        raw_metrics: List of metric dictionaries.

    Returns:
        List of MetricSnapshot instances.
    """
    snapshots: list[MetricSnapshot] = []
    for item in raw_metrics:
        snapshots.append(
            MetricSnapshot(
                epoch=item["epoch"],
                step=item["step"],
                train_loss=item["train_loss"],
                val_loss=item.get("val_loss"),
                learning_rate=item.get("learning_rate", 0.0),
                gradient_norm=item.get("gradient_norm"),
                timestamp=(
                    datetime.fromisoformat(item["timestamp"])
                    if "timestamp" in item
                    else datetime.now()
                ),
            )
        )
    return snapshots


@router.get("/diagnostics/{run_id}")
async def get_diagnostics(run_id: str) -> dict[str, Any]:
    """Retrieve diagnostics for a training run.

    Loads the training result from the registry, converts its
    metrics history to diagnostic snapshots, and runs analysis.

    Args:
        run_id: The training run identifier.

    Returns:
        Dict with the diagnostic report.
    """
    run_data = _state["pipelines"].get(run_id)
    if run_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Training run not found: {run_id}",
        )

    result = run_data.get("result")
    if result is None:
        raise HTTPException(
            status_code=400,
            detail="Training run has no results yet.",
        )

    try:
        snapshots = [
            MetricSnapshot(
                epoch=m.epoch,
                step=m.step,
                train_loss=m.train_loss,
                val_loss=m.val_loss,
                learning_rate=m.learning_rate,
            )
            for m in result.metrics_history
        ]

        diag = TrainingDiagnostics()
        report = diag.analyze_training(snapshots)
        report_dict = report.to_dict()
        report_dict["run_id"] = run_id
        return report_dict
    except DiagnosticsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ===================================================================
# Regression endpoints
# ===================================================================


@router.post("/regression/check")
async def run_regression_check(
    request: RegressionCheckRequest,
) -> dict[str, Any]:
    """Compare two evaluation reports for regressions.

    Args:
        request: Baseline and current evaluation IDs plus change type.

    Returns:
        Dict with the regression report including verdict.
    """
    try:
        history = _get_evaluation_history()
        baseline = history.load_report(request.baseline_eval_id)
        current = history.load_report(request.current_eval_id)
        change_type = ChangeType(request.change_type)

        runner = _get_regression_runner()
        report = runner.run_regression_check(baseline, current, change_type)
        runner.save_report(report)

        return report.to_dict()
    except EvaluationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (RegressionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/regression/versions")
async def list_versions(
    discipline_id: str | None = None,
) -> dict[str, Any]:
    """List registered model versions.

    Args:
        discipline_id: Optional discipline filter.

    Returns:
        Dict with list of version entries.
    """
    try:
        vm = _get_version_manager()
        versions = vm.list_versions(discipline_id=discipline_id)
        return {
            "versions": [v.to_dict() for v in versions],
            "total": len(versions),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/regression/register")
async def register_version(
    request: VersionRegisterRequest,
) -> dict[str, Any]:
    """Register a new model version.

    Args:
        request: Version metadata.

    Returns:
        Dict with the registered version entry.
    """
    try:
        version_id = f"ver_{uuid.uuid4().hex[:12]}"
        change_type = ChangeType(request.change_type)

        entry = VersionEntry(
            version_id=version_id,
            model_name=request.model_name,
            discipline_id=request.discipline_id,
            training_run_id=request.training_run_id,
            evaluation_run_id=request.evaluation_run_id,
            adapter_path=request.adapter_path,
            change_type=change_type,
            change_description=request.change_description,
            created_at=datetime.now(),
            is_active=True,
        )

        vm = _get_version_manager()
        vm.register_version(entry)

        return entry.to_dict()
    except (RegressionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ===================================================================
# Merging endpoints
# ===================================================================


@router.post("/merging/merge")
async def merge_adapters(request: MergeRequest) -> dict[str, Any]:
    """Merge multiple LoRA adapters into a single adapter.

    Args:
        request: Adapter information, method, and configuration.

    Returns:
        Dict with the merge result.
    """
    try:
        adapters = [AdapterInfo.from_dict(a) for a in request.adapters]
        method = MergeMethod(request.method)

        output_dir = Path(request.output_dir) if request.output_dir is not None else None

        config = MergeConfig(
            method=method,
            weights=request.weights,
            ties_density=request.ties_density,
            ties_majority_sign=request.ties_majority_sign,
            output_dir=output_dir,
            validate_compatibility=request.validate_compatibility,
        )

        pipeline = MergePipeline(config=config)
        result = pipeline.merge(adapters)

        registry = _get_merge_registry()
        registry.register(result)

        return result.to_dict()
    except MergingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid merge request: {exc}",
        ) from exc


@router.get("/merging/registry")
async def list_merges() -> dict[str, Any]:
    """List all registered merge operations.

    Returns:
        Dict with list of merge results.
    """
    try:
        registry = _get_merge_registry()
        merges = registry.list_merges()
        return {
            "merges": [m.to_dict() for m in merges],
            "total": len(merges),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/merging/methods")
async def list_merge_methods() -> dict[str, Any]:
    """List available merge methods with descriptions.

    Returns:
        Dict with method names and descriptions.
    """
    return {
        "methods": [
            {
                "name": MergeMethod.LINEAR.value,
                "description": (
                    "Weighted linear interpolation of adapter weights. "
                    "Requires matching LoRA ranks across all adapters."
                ),
            },
            {
                "name": MergeMethod.TIES.value,
                "description": (
                    "TrIm, Elect Sign & merge. Sparse combination that "
                    "handles rank mismatches and resolves parameter "
                    "sign conflicts via majority vote."
                ),
            },
        ],
    }
