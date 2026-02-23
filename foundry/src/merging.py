"""Model merging pipeline for combining LoRA adapters.

Supports linear interpolation and TIES (TrIm, Elect Sign, merge)
methods for combining multiple discipline-specific LoRA adapters
into a single merged adapter. The actual weight manipulation is
abstracted for MVP; the pipeline, configuration, validation, and
registry are fully functional.

Example::

    from foundry.src.merging import (
        AdapterInfo, MergeConfig, MergePipeline, MergeRegistry,
    )

    adapters = [adapter_a, adapter_b]
    config = MergeConfig(method=MergeMethod.LINEAR)
    pipeline = MergePipeline(config=config)
    result = pipeline.merge(adapters)
    print(result.plain_language_summary)
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ===================================================================
# Exceptions
# ===================================================================


class MergingError(Exception):
    """Raised when a merging operation fails."""


# ===================================================================
# Enums
# ===================================================================


class MergeMethod(str, Enum):
    """Supported model merging methods.

    Attributes:
        LINEAR: Weighted linear interpolation of adapter weights.
        TIES: TrIm, Elect Sign & merge for sparse combination.
    """

    LINEAR = "linear"
    TIES = "ties"


class MergeStatus(str, Enum):
    """Status of a merge operation.

    Attributes:
        PENDING: Not yet started.
        VALIDATING: Checking adapter compatibility.
        MERGING: Executing the merge.
        EVALUATING: Running post-merge evaluation.
        COMPLETED: Merge finished successfully.
        FAILED: Merge encountered an error.
    """

    PENDING = "pending"
    VALIDATING = "validating"
    MERGING = "merging"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"


# ===================================================================
# AdapterInfo
# ===================================================================


@dataclass
class AdapterInfo:
    """Information about a trained LoRA adapter.

    Attributes:
        adapter_path: Path to the adapter weights directory.
        discipline_id: Unique discipline identifier.
        discipline_name: Human-readable discipline name.
        base_model: HuggingFace model identifier the adapter was trained on.
        base_model_family: Model architecture family (phi, llama, etc.).
        lora_rank: LoRA rank used during training.
        training_run_id: Optional reference to the training run.
    """

    adapter_path: Path
    discipline_id: str
    discipline_name: str
    base_model: str
    base_model_family: str
    lora_rank: int
    training_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with all adapter info fields.
        """
        return {
            "adapter_path": str(self.adapter_path),
            "discipline_id": self.discipline_id,
            "discipline_name": self.discipline_name,
            "base_model": self.base_model,
            "base_model_family": self.base_model_family,
            "lora_rank": self.lora_rank,
            "training_run_id": self.training_run_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdapterInfo:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with adapter info fields.

        Returns:
            AdapterInfo instance.
        """
        return cls(
            adapter_path=Path(data["adapter_path"]),
            discipline_id=data["discipline_id"],
            discipline_name=data["discipline_name"],
            base_model=data["base_model"],
            base_model_family=data["base_model_family"],
            lora_rank=data["lora_rank"],
            training_run_id=data.get("training_run_id"),
        )


# ===================================================================
# MergeConfig
# ===================================================================


@dataclass
class MergeConfig:
    """Configuration for a model merge operation.

    Attributes:
        method: Merging method to use (LINEAR or TIES).
        weights: Per-adapter weights; None means equal weighting.
        ties_density: Fraction of parameters to keep (TIES only).
        ties_majority_sign: Whether to use majority sign election (TIES only).
        output_dir: Directory for merge output; auto-generated if None.
        validate_compatibility: Whether to check adapter compatibility.
    """

    method: MergeMethod = MergeMethod.LINEAR
    weights: list[float] | None = None
    ties_density: float = 0.5
    ties_majority_sign: bool = True
    output_dir: Path | None = None
    validate_compatibility: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with all config fields.
        """
        return {
            "method": self.method.value,
            "weights": self.weights,
            "ties_density": self.ties_density,
            "ties_majority_sign": self.ties_majority_sign,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "validate_compatibility": self.validate_compatibility,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MergeConfig:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with merge config fields.

        Returns:
            MergeConfig instance.
        """
        output_dir = Path(data["output_dir"]) if data.get("output_dir") else None
        return cls(
            method=MergeMethod(data.get("method", "linear")),
            weights=data.get("weights"),
            ties_density=data.get("ties_density", 0.5),
            ties_majority_sign=data.get("ties_majority_sign", True),
            output_dir=output_dir,
            validate_compatibility=data.get("validate_compatibility", True),
        )


# ===================================================================
# CompatibilityResult
# ===================================================================


@dataclass
class CompatibilityResult:
    """Result of adapter compatibility checking.

    Attributes:
        is_compatible: Whether all adapters can be merged.
        issues: Blocking problems that prevent merging.
        warnings: Non-blocking concerns to note.
    """

    is_compatible: bool
    issues: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with compatibility result fields.
        """
        return {
            "is_compatible": self.is_compatible,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
        }


# ===================================================================
# MergeResult
# ===================================================================


@dataclass
class MergeResult:
    """Result of a completed (or failed) merge operation.

    Attributes:
        merge_id: Unique identifier for this merge.
        method: Merging method used.
        status: Final status of the merge.
        adapters: List of source adapters.
        weights_used: Actual weights applied during merge.
        merged_adapter_path: Path to merged output (None if failed).
        evaluation_results: Per-discipline accuracy (None if not evaluated).
        started_at: When the merge started.
        completed_at: When the merge finished (None if incomplete).
        error_message: Error description if merge failed.
        plain_language_summary: SME-friendly merge description.
    """

    merge_id: str
    method: MergeMethod
    status: MergeStatus
    adapters: list[AdapterInfo]
    weights_used: list[float]
    merged_adapter_path: Path | None
    evaluation_results: dict[str, float] | None
    started_at: datetime
    completed_at: datetime | None
    error_message: str | None
    plain_language_summary: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with all merge result fields.
        """
        return {
            "merge_id": self.merge_id,
            "method": self.method.value,
            "status": self.status.value,
            "adapters": [a.to_dict() for a in self.adapters],
            "weights_used": list(self.weights_used),
            "merged_adapter_path": (
                str(self.merged_adapter_path) if self.merged_adapter_path else None
            ),
            "evaluation_results": self.evaluation_results,
            "started_at": self.started_at.isoformat(),
            "completed_at": (self.completed_at.isoformat() if self.completed_at else None),
            "error_message": self.error_message,
            "plain_language_summary": self.plain_language_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MergeResult:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with merge result fields.

        Returns:
            MergeResult instance.
        """
        adapters = [AdapterInfo.from_dict(a) for a in data.get("adapters", [])]
        merged_path = Path(data["merged_adapter_path"]) if data.get("merged_adapter_path") else None
        completed_at = _parse_optional_datetime(data.get("completed_at"))
        return cls(
            merge_id=data["merge_id"],
            method=MergeMethod(data["method"]),
            status=MergeStatus(data["status"]),
            adapters=adapters,
            weights_used=data.get("weights_used", []),
            merged_adapter_path=merged_path,
            evaluation_results=data.get("evaluation_results"),
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=completed_at,
            error_message=data.get("error_message"),
            plain_language_summary=data.get("plain_language_summary", ""),
        )


def _parse_optional_datetime(value: str | None) -> datetime | None:
    """Parse an ISO datetime string, returning None if value is None.

    Args:
        value: ISO format datetime string or None.

    Returns:
        Parsed datetime or None.
    """
    if value is None:
        return None
    return datetime.fromisoformat(value)


# ===================================================================
# CompatibilityChecker
# ===================================================================


class CompatibilityChecker:
    """Checks whether a set of adapters can be merged together.

    Validates base model match, LoRA rank compatibility (method-dependent),
    and adapter path existence.

    Example::

        checker = CompatibilityChecker()
        result = checker.check([adapter_a, adapter_b])
        if not result.is_compatible:
            print(result.issues)
    """

    def check(
        self,
        adapters: list[AdapterInfo],
        method: MergeMethod = MergeMethod.LINEAR,
    ) -> CompatibilityResult:
        """Check compatibility of adapters for merging.

        Args:
            adapters: List of adapters to check.
            method: Merge method that will be used.

        Returns:
            CompatibilityResult with issues and warnings.
        """
        issues: list[str] = []
        warnings: list[str] = []

        issues.extend(self._check_base_model(adapters))
        issues.extend(self._check_lora_rank(adapters, method))
        warnings.extend(self._check_lora_rank_warnings(adapters, method))
        warnings.extend(self._check_adapter_paths(adapters))

        return CompatibilityResult(
            is_compatible=len(issues) == 0,
            issues=issues,
            warnings=warnings,
        )

    @staticmethod
    def _check_base_model(adapters: list[AdapterInfo]) -> list[str]:
        """Verify all adapters share the same base model.

        Args:
            adapters: List of adapters to check.

        Returns:
            List of issue strings (empty if compatible).
        """
        if len(adapters) <= 1:
            return []
        base_models = {a.base_model for a in adapters}
        if len(base_models) > 1:
            models = ", ".join(sorted(base_models))
            return [f"Mismatched base models: {models}"]
        return []

    @staticmethod
    def _check_lora_rank(
        adapters: list[AdapterInfo],
        method: MergeMethod,
    ) -> list[str]:
        """Check LoRA rank compatibility (blocking issues only).

        For LINEAR: all ranks must match.
        For TIES: rank mismatch is a warning, not an issue.

        Args:
            adapters: List of adapters to check.
            method: Merge method.

        Returns:
            List of issue strings.
        """
        if len(adapters) <= 1:
            return []
        ranks = {a.lora_rank for a in adapters}
        if len(ranks) > 1 and method == MergeMethod.LINEAR:
            rank_list = ", ".join(str(r) for r in sorted(ranks))
            return [f"Mismatched LoRA ranks for linear merge: {rank_list}"]
        return []

    @staticmethod
    def _check_lora_rank_warnings(
        adapters: list[AdapterInfo],
        method: MergeMethod,
    ) -> list[str]:
        """Generate warnings for LoRA rank differences (TIES only).

        Args:
            adapters: List of adapters to check.
            method: Merge method.

        Returns:
            List of warning strings.
        """
        if len(adapters) <= 1:
            return []
        ranks = {a.lora_rank for a in adapters}
        if len(ranks) > 1 and method == MergeMethod.TIES:
            return ["Different LoRA ranks detected; TIES will handle this but quality may vary"]
        return []

    @staticmethod
    def _check_adapter_paths(adapters: list[AdapterInfo]) -> list[str]:
        """Warn about adapter paths that do not exist.

        Args:
            adapters: List of adapters to check.

        Returns:
            List of warning strings for missing paths.
        """
        warnings: list[str] = []
        for adapter in adapters:
            if not adapter.adapter_path.exists():
                warnings.append(f"Adapter path does not exist: {adapter.adapter_path}")
        return warnings


# ===================================================================
# LinearMerger
# ===================================================================


class LinearMerger:
    """Merges LoRA adapters using weighted linear interpolation.

    In MVP: creates output directory, writes metadata JSON, and
    simulates the merge. Production implementation would load
    actual tensor weights and compute weighted averages.

    Example::

        merger = LinearMerger()
        output = merger.merge(adapters, weights=[0.5, 0.5], output_path=path)
    """

    def merge(
        self,
        adapters: list[AdapterInfo],
        weights: list[float],
        output_path: Path,
    ) -> Path:
        """Execute a linear merge of adapters.

        Args:
            adapters: Source adapters to merge.
            weights: Per-adapter weights (will be normalized).
            output_path: Directory for merged output.

        Returns:
            Path to the merged adapter directory.
        """
        normalized = self._normalize_weights(weights)
        output_path.mkdir(parents=True, exist_ok=True)
        self._write_metadata(adapters, normalized, output_path)
        logger.info(
            "Linear merge completed: %d adapters -> %s",
            len(adapters),
            output_path,
        )
        return output_path

    @staticmethod
    def _normalize_weights(weights: list[float]) -> list[float]:
        """Normalize weights to sum to 1.0.

        Args:
            weights: Raw weight values.

        Returns:
            Normalized weights summing to 1.0.
        """
        total = sum(weights)
        if total == 0:
            count = len(weights)
            return [1.0 / count] * count
        return [w / total for w in weights]

    @staticmethod
    def _write_metadata(
        adapters: list[AdapterInfo],
        weights: list[float],
        output_path: Path,
    ) -> None:
        """Write merge metadata to a JSON file.

        Args:
            adapters: Source adapters.
            weights: Normalized weights.
            output_path: Output directory.
        """
        metadata = {
            "method": "linear",
            "adapters": [a.to_dict() for a in adapters],
            "weights": weights,
            "merged_at": datetime.now().isoformat(),
        }
        path = output_path / "merge_metadata.json"
        path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )


# ===================================================================
# TIESMerger
# ===================================================================


class TIESMerger:
    """Merges LoRA adapters using TIES (TrIm, Elect Sign & merge).

    TIES merging trims small-magnitude parameters, resolves sign
    conflicts via majority vote, then merges. In MVP: simulated
    with metadata recording.

    Example::

        merger = TIESMerger()
        config = MergeConfig(method=MergeMethod.TIES, ties_density=0.3)
        output = merger.merge(adapters, config=config, output_path=path)
    """

    def merge(
        self,
        adapters: list[AdapterInfo],
        config: MergeConfig,
        output_path: Path,
    ) -> Path:
        """Execute a TIES merge of adapters.

        Args:
            adapters: Source adapters to merge.
            config: Merge configuration with TIES parameters.
            output_path: Directory for merged output.

        Returns:
            Path to the merged adapter directory.
        """
        output_path.mkdir(parents=True, exist_ok=True)
        self._apply_density_mask(config.ties_density)
        self._majority_sign_election(config.ties_majority_sign)
        self._write_metadata(adapters, config, output_path)
        logger.info(
            "TIES merge completed: %d adapters, density=%.2f -> %s",
            len(adapters),
            config.ties_density,
            output_path,
        )
        return output_path

    @staticmethod
    def _apply_density_mask(density: float) -> None:
        """Apply density-based parameter trimming (conceptual).

        In production, this would zero out parameters below the
        density threshold. For MVP, this is a no-op placeholder.

        Args:
            density: Fraction of parameters to keep (0-1).
        """
        logger.debug("TIES density mask: keeping %.0f%% of parameters", density * 100)

    @staticmethod
    def _majority_sign_election(use_majority: bool) -> None:
        """Resolve sign conflicts via majority vote (conceptual).

        In production, this would examine each parameter position
        across all adapters and keep the majority sign direction.
        For MVP, this is a no-op placeholder.

        Args:
            use_majority: Whether to apply majority sign election.
        """
        if use_majority:
            logger.debug("TIES majority sign election enabled")

    @staticmethod
    def _write_metadata(
        adapters: list[AdapterInfo],
        config: MergeConfig,
        output_path: Path,
    ) -> None:
        """Write TIES merge metadata to a JSON file.

        Args:
            adapters: Source adapters.
            config: TIES merge configuration.
            output_path: Output directory.
        """
        metadata = {
            "method": "ties",
            "adapters": [a.to_dict() for a in adapters],
            "ties_density": config.ties_density,
            "ties_majority_sign": config.ties_majority_sign,
            "merged_at": datetime.now().isoformat(),
        }
        path = output_path / "merge_metadata.json"
        path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )


# ===================================================================
# MergePipeline
# ===================================================================


class MergePipeline:
    """Orchestrates the full merge lifecycle.

    Validates adapter compatibility, executes the merge using the
    configured method, and generates a plain-language summary.

    Args:
        config: Merge configuration.

    Example::

        pipeline = MergePipeline(config=MergeConfig())
        result = pipeline.merge([adapter_a, adapter_b])
        print(result.plain_language_summary)
    """

    def __init__(self, config: MergeConfig | None = None) -> None:
        """Initialize the merge pipeline.

        Args:
            config: Merge configuration (defaults to MergeConfig()).
        """
        self._config = config or MergeConfig()
        self._status = MergeStatus.PENDING

    def merge(self, adapters: list[AdapterInfo]) -> MergeResult:
        """Execute the full merge pipeline.

        Args:
            adapters: List of adapters to merge.

        Returns:
            MergeResult with status, path, and summary.

        Raises:
            MergingError: If adapters are incompatible.
        """
        started_at = datetime.now()
        merge_id = f"merge_{uuid.uuid4().hex[:12]}"

        if self._config.validate_compatibility:
            self._validate(adapters)

        weights = self._resolve_weights(adapters)
        self._status = MergeStatus.MERGING
        merged_path = self._execute_merge(adapters, weights)

        self._status = MergeStatus.COMPLETED
        result = self._build_result(
            merge_id,
            adapters,
            weights,
            merged_path,
            started_at,
        )
        result.plain_language_summary = self._generate_summary(result)
        return result

    def get_status(self) -> MergeStatus:
        """Return the current pipeline status.

        Returns:
            Current MergeStatus.
        """
        return self._status

    def _validate(self, adapters: list[AdapterInfo]) -> None:
        """Validate adapter compatibility.

        Args:
            adapters: Adapters to check.

        Raises:
            MergingError: If adapters are not compatible.
        """
        self._status = MergeStatus.VALIDATING
        checker = CompatibilityChecker()
        result = checker.check(adapters, method=self._config.method)
        if not result.is_compatible:
            issues = "; ".join(result.issues)
            raise MergingError(f"Compatibility check failed: {issues}")
        for warning in result.warnings:
            logger.warning("Merge warning: %s", warning)

    def _resolve_weights(self, adapters: list[AdapterInfo]) -> list[float]:
        """Determine weights for the merge.

        Uses configured weights or defaults to equal weighting.

        Args:
            adapters: List of adapters.

        Returns:
            List of per-adapter weights.
        """
        if self._config.weights is not None:
            return list(self._config.weights)
        count = len(adapters)
        return [1.0 / count] * count

    def _execute_merge(
        self,
        adapters: list[AdapterInfo],
        weights: list[float],
    ) -> Path:
        """Execute the merge using the configured method.

        Args:
            adapters: Source adapters.
            weights: Per-adapter weights.

        Returns:
            Path to the merged adapter directory.
        """
        output_path = self._get_output_path()
        if self._config.method == MergeMethod.TIES:
            merger = TIESMerger()
            return merger.merge(adapters, self._config, output_path)
        merger_linear = LinearMerger()
        return merger_linear.merge(adapters, weights, output_path)

    def _get_output_path(self) -> Path:
        """Determine the output directory for the merge.

        Returns:
            Path to the output directory.
        """
        if self._config.output_dir is not None:
            return self._config.output_dir
        return Path(f"merged_{uuid.uuid4().hex[:8]}")

    def _build_result(
        self,
        merge_id: str,
        adapters: list[AdapterInfo],
        weights: list[float],
        merged_path: Path,
        started_at: datetime,
    ) -> MergeResult:
        """Assemble a MergeResult from completed merge.

        Args:
            merge_id: Unique merge identifier.
            adapters: Source adapters.
            weights: Weights used.
            merged_path: Path to merged output.
            started_at: When the merge started.

        Returns:
            A completed MergeResult.
        """
        return MergeResult(
            merge_id=merge_id,
            method=self._config.method,
            status=MergeStatus.COMPLETED,
            adapters=list(adapters),
            weights_used=list(weights),
            merged_adapter_path=merged_path,
            evaluation_results=None,
            started_at=started_at,
            completed_at=datetime.now(),
            error_message=None,
            plain_language_summary="",
        )

    @staticmethod
    def _generate_summary(result: MergeResult) -> str:
        """Generate a plain-language summary of the merge.

        Uses SME-friendly language: describes what was combined,
        the method used, and the discipline names involved.

        Args:
            result: The completed merge result.

        Returns:
            Plain-language summary string.
        """
        count = len(result.adapters)
        names = [a.discipline_name for a in result.adapters]
        method_name = result.method.value.upper()
        name_list = ", ".join(names)

        lines = [
            f"Combined {count} discipline adapters using {method_name} merging.",
            f"Disciplines: {name_list}.",
        ]
        weights_str = ", ".join(f"{w:.1%}" for w in result.weights_used)
        lines.append(f"Weights: {weights_str}.")

        return " ".join(lines)


# ===================================================================
# MergeRegistry
# ===================================================================


class MergeRegistry:
    """Persistent registry of merge operations.

    Stores merge metadata as individual JSON files in a registry
    directory. Supports listing, retrieval, and finding the latest.

    Args:
        registry_dir: Directory for storing merge metadata files.

    Example::

        registry = MergeRegistry(Path("./merge_registry"))
        registry.register(result)
        loaded = registry.get(result.merge_id)
    """

    def __init__(self, registry_dir: Path) -> None:
        """Initialize the registry.

        Args:
            registry_dir: Directory for storing merge metadata.
        """
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def register(self, result: MergeResult) -> None:
        """Register a completed merge result.

        Args:
            result: The merge result to persist.
        """
        path = self.registry_dir / f"{result.merge_id}.json"
        path.write_text(
            json.dumps(result.to_dict(), indent=2),
            encoding="utf-8",
        )
        logger.info("Registered merge: %s", result.merge_id)

    def get(self, merge_id: str) -> MergeResult | None:
        """Retrieve a merge result by ID.

        Args:
            merge_id: The unique merge identifier.

        Returns:
            MergeResult if found, None otherwise.
        """
        path = self.registry_dir / f"{merge_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return MergeResult.from_dict(data)

    def list_merges(self) -> list[MergeResult]:
        """List all registered merge results.

        Returns:
            List of MergeResult sorted by start time.
        """
        results = self._load_all()
        return sorted(results, key=lambda r: r.started_at)

    def get_latest(self) -> MergeResult | None:
        """Return the most recently started merge.

        Returns:
            The latest MergeResult, or None if registry is empty.
        """
        results = self.list_merges()
        if not results:
            return None
        return results[-1]

    def _load_all(self) -> list[MergeResult]:
        """Load all merge entries from the registry directory.

        Returns:
            List of all stored MergeResult entries.
        """
        results: list[MergeResult] = []
        for path in self.registry_dir.glob("merge_*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            results.append(MergeResult.from_dict(data))
        return results
