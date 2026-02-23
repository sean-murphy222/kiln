"""Foundry LoRA training pipeline.

Handles all non-GPU aspects of LoRA fine-tuning: configuration,
curriculum loading, validation, hyperparameter auto-configuration,
progress tracking, result management, and a training registry.

The actual GPU training step is abstracted behind a dry-run backend
for MVP testing. Production backends (Unsloth, Axolotl) can be
swapped in without changing the pipeline interface.
"""

from __future__ import annotations

import json
import logging
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ===================================================================
# Exceptions
# ===================================================================


class TrainingError(Exception):
    """Raised when a training pipeline operation fails."""


# ===================================================================
# Enums
# ===================================================================


class BaseModelFamily(str, Enum):
    """Supported base model families for LoRA training."""

    PHI = "phi"
    LLAMA = "llama"
    MISTRAL = "mistral"
    QWEN = "qwen"


class TrainingStatus(str, Enum):
    """Status of a training pipeline run."""

    PENDING = "pending"
    PREPARING = "preparing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ===================================================================
# Default base models per family
# ===================================================================

_DEFAULT_BASE_MODELS: dict[BaseModelFamily, str] = {
    BaseModelFamily.PHI: "microsoft/phi-3-mini-4k-instruct",
    BaseModelFamily.LLAMA: "meta-llama/Llama-3-8B-Instruct",
    BaseModelFamily.MISTRAL: "mistralai/Mistral-7B-Instruct-v0.3",
    BaseModelFamily.QWEN: "Qwen/Qwen2-7B-Instruct",
}


# ===================================================================
# LoRAConfig
# ===================================================================


@dataclass
class LoRAConfig:
    """Configuration for Low-Rank Adaptation (LoRA) parameters.

    Attributes:
        rank: Rank of the low-rank decomposition.
        alpha: Scaling factor for LoRA weights.
        dropout: Dropout probability for LoRA layers.
        target_modules: Transformer modules to apply LoRA to.
    """

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoRAConfig:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with LoRA configuration values.
                  Missing keys use defaults.

        Returns:
            LoRAConfig instance.
        """
        return cls(
            rank=data.get("rank", 16),
            alpha=data.get("alpha", 32),
            dropout=data.get("dropout", 0.05),
            target_modules=data.get("target_modules", ["q_proj", "v_proj"]),
        )


# ===================================================================
# TrainingConfig
# ===================================================================


@dataclass
class TrainingConfig:
    """Full training configuration for a LoRA fine-tuning run.

    Attributes:
        base_model: HuggingFace model identifier.
        base_model_family: Model architecture family.
        curriculum_path: Path to the JSONL curriculum file.
        output_dir: Directory for training outputs and adapter weights.
        lora: LoRA-specific configuration.
        epochs: Number of training epochs.
        batch_size: Per-device training batch size.
        learning_rate: Peak learning rate for the optimizer.
        max_seq_length: Maximum sequence length for tokenization.
        warmup_ratio: Fraction of steps for learning rate warmup.
        weight_decay: L2 regularization weight decay.
        gradient_accumulation_steps: Steps to accumulate before update.
        seed: Random seed for reproducibility.
        validation_split: Fraction of data held out for validation.
    """

    base_model: str
    base_model_family: BaseModelFamily | str
    curriculum_path: Path
    output_dir: Path
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    seed: int = 42
    validation_split: float = 0.1

    def __post_init__(self) -> None:
        """Normalize types after construction."""
        if isinstance(self.base_model_family, str):
            self.base_model_family = BaseModelFamily(self.base_model_family)
        if isinstance(self.curriculum_path, str):
            self.curriculum_path = Path(self.curriculum_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "base_model": self.base_model,
            "base_model_family": self.base_model_family.value,
            "curriculum_path": str(self.curriculum_path),
            "output_dir": str(self.output_dir),
            "lora": self.lora.to_dict(),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_seq_length": self.max_seq_length,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "seed": self.seed,
            "validation_split": self.validation_split,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with training configuration values.

        Returns:
            TrainingConfig instance.
        """
        lora_data = data.get("lora", {})
        lora = LoRAConfig.from_dict(lora_data)
        return cls(
            base_model=data["base_model"],
            base_model_family=data["base_model_family"],
            curriculum_path=Path(data["curriculum_path"]),
            output_dir=Path(data["output_dir"]),
            lora=lora,
            epochs=data.get("epochs", 3),
            batch_size=data.get("batch_size", 4),
            learning_rate=data.get("learning_rate", 2e-4),
            max_seq_length=data.get("max_seq_length", 2048),
            warmup_ratio=data.get("warmup_ratio", 0.03),
            weight_decay=data.get("weight_decay", 0.01),
            gradient_accumulation_steps=data.get("gradient_accumulation_steps", 4),
            seed=data.get("seed", 42),
            validation_split=data.get("validation_split", 0.1),
        )


# ===================================================================
# TrainingMetrics
# ===================================================================


@dataclass
class TrainingMetrics:
    """Metrics captured at a single training step or epoch boundary.

    Attributes:
        epoch: Current epoch number.
        step: Global training step number.
        train_loss: Training loss at this step.
        val_loss: Validation loss (None if not evaluated).
        learning_rate: Current learning rate.
        timestamp: When the metrics were recorded.
    """

    epoch: int
    step: int
    train_loss: float
    val_loss: float | None
    learning_rate: float
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "epoch": self.epoch,
            "step": self.step,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "learning_rate": self.learning_rate,
            "timestamp": self.timestamp.isoformat(),
        }


# ===================================================================
# TrainingResult
# ===================================================================


@dataclass
class TrainingResult:
    """Result of a completed (or failed) training run.

    Attributes:
        config: The training configuration used.
        status: Final status of the training run.
        adapter_path: Path to the saved LoRA adapter (None if failed).
        metrics_history: List of metrics recorded during training.
        total_examples: Total number of curriculum examples.
        training_examples: Number of examples in the training set.
        validation_examples: Number of examples in the validation set.
        started_at: When the training started.
        completed_at: When the training completed.
        error_message: Error description if training failed.
    """

    config: TrainingConfig
    status: TrainingStatus
    total_examples: int
    training_examples: int
    validation_examples: int
    adapter_path: Path | None = None
    metrics_history: list[TrainingMetrics] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "config": self.config.to_dict(),
            "status": self.status.value,
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "metrics_history": [m.to_dict() for m in self.metrics_history],
            "total_examples": self.total_examples,
            "training_examples": self.training_examples,
            "validation_examples": self.validation_examples,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (self.completed_at.isoformat() if self.completed_at else None),
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingResult:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with training result values.

        Returns:
            TrainingResult instance.
        """
        config = TrainingConfig.from_dict(data["config"])
        adapter_path = Path(data["adapter_path"]) if data.get("adapter_path") else None
        started_at = _parse_optional_datetime(data.get("started_at"))
        completed_at = _parse_optional_datetime(data.get("completed_at"))
        return cls(
            config=config,
            status=TrainingStatus(data["status"]),
            adapter_path=adapter_path,
            metrics_history=[],
            total_examples=data["total_examples"],
            training_examples=data["training_examples"],
            validation_examples=data["validation_examples"],
            started_at=started_at,
            completed_at=completed_at,
            error_message=data.get("error_message"),
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
# TrainingProgressCallback
# ===================================================================


@runtime_checkable
class TrainingProgressCallback(Protocol):
    """Protocol for receiving training progress notifications."""

    def on_step(self, metrics: TrainingMetrics) -> None:
        """Called after each training step.

        Args:
            metrics: Metrics for the completed step.
        """
        ...

    def on_epoch_end(self, epoch: int, metrics: TrainingMetrics) -> None:
        """Called at the end of each epoch.

        Args:
            epoch: The epoch number that just completed.
            metrics: Final metrics for the epoch.
        """
        ...

    def on_complete(self, result: TrainingResult) -> None:
        """Called when training completes successfully.

        Args:
            result: The final training result.
        """
        ...

    def on_error(self, error: str) -> None:
        """Called when training encounters an error.

        Args:
            error: Description of the error.
        """
        ...


# ===================================================================
# CurriculumLoader
# ===================================================================


class CurriculumLoader:
    """Loads and validates Forge JSONL curriculum files.

    Validates that records follow the Alpaca format with required
    fields: instruction, input, output, metadata.
    """

    def load(self, path: Path) -> list[dict[str, Any]]:
        """Load and validate a JSONL curriculum file.

        Args:
            path: Path to the JSONL file.

        Returns:
            List of validated Alpaca-format records.

        Raises:
            TrainingError: If file is missing, empty, or unparseable.
        """
        self._check_file_exists(path)
        raw_lines = self._read_lines(path)
        records = self._parse_lines(raw_lines, path)
        valid = [r for r in records if self.validate_record(r)]
        if not valid:
            msg = f"Curriculum file contains no valid records: {path}"
            raise TrainingError(msg)
        if len(valid) < len(records):
            skipped = len(records) - len(valid)
            logger.warning("Filtered %d invalid records from %s", skipped, path)
        return valid

    def validate_record(self, record: dict[str, Any]) -> bool:
        """Check whether a record follows Alpaca format.

        Args:
            record: Dictionary to validate.

        Returns:
            True if record has non-empty instruction and output.
        """
        instruction = record.get("instruction", "")
        output = record.get("output", "")
        return bool(instruction) and bool(output)

    def split_train_val(
        self,
        records: list[dict[str, Any]],
        val_ratio: float,
        seed: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split records into training and validation sets.

        Args:
            records: Full list of curriculum records.
            val_ratio: Fraction to hold out for validation (0.0 to 1.0).
            seed: Random seed for reproducible splitting.

        Returns:
            Tuple of (training_records, validation_records).
        """
        if val_ratio <= 0.0:
            return list(records), []
        rng = random.Random(seed)
        indices = list(range(len(records)))
        rng.shuffle(indices)
        val_count = int(len(records) * val_ratio)
        val_indices = set(indices[:val_count])
        train = [r for i, r in enumerate(records) if i not in val_indices]
        val = [r for i, r in enumerate(records) if i in val_indices]
        return train, val

    def get_statistics(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute statistics about the curriculum records.

        Args:
            records: List of Alpaca-format curriculum records.

        Returns:
            Dictionary with total_records, competency_counts,
            discipline_counts, avg_instruction_length, avg_output_length.
        """
        competency_counts = _count_metadata_field(records, "competency_id")
        discipline_counts = _count_metadata_field(records, "discipline_id")
        avg_instr = _average_length(records, "instruction")
        avg_out = _average_length(records, "output")
        return {
            "total_records": len(records),
            "competency_counts": competency_counts,
            "discipline_counts": discipline_counts,
            "avg_instruction_length": avg_instr,
            "avg_output_length": avg_out,
        }

    @staticmethod
    def _check_file_exists(path: Path) -> None:
        """Raise TrainingError if the file does not exist.

        Args:
            path: Path to check.

        Raises:
            TrainingError: If file is not found.
        """
        if not path.exists():
            raise TrainingError(f"Curriculum file not found: {path}")

    @staticmethod
    def _read_lines(path: Path) -> list[str]:
        """Read non-empty lines from a file.

        Args:
            path: Path to the file.

        Returns:
            List of non-empty stripped lines.

        Raises:
            TrainingError: If file is empty.
        """
        text = path.read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise TrainingError(f"Curriculum file is empty: {path}")
        return lines

    @staticmethod
    def _parse_lines(lines: list[str], path: Path) -> list[dict[str, Any]]:
        """Parse JSONL lines into dictionaries.

        Args:
            lines: Raw JSONL lines.
            path: Original file path (for error messages).

        Returns:
            List of parsed dictionaries.

        Raises:
            TrainingError: If any line fails to parse.
        """
        records: list[dict[str, Any]] = []
        for i, line in enumerate(lines):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                msg = f"Failed to parse line {i + 1} in {path}: {exc}"
                raise TrainingError(msg) from exc
        return records


def _count_metadata_field(records: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    """Count occurrences of a metadata field value across records.

    Args:
        records: List of Alpaca-format records.
        field_name: Key within the metadata dict to count.

    Returns:
        Dictionary mapping field values to their counts.
    """
    counts: dict[str, int] = {}
    for record in records:
        metadata = record.get("metadata", {})
        value = metadata.get(field_name, "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _average_length(records: list[dict[str, Any]], key: str) -> float:
    """Compute average string length of a field across records.

    Args:
        records: List of records.
        key: Top-level key whose string value length to average.

    Returns:
        Average character length, or 0.0 if no records.
    """
    if not records:
        return 0.0
    total = sum(len(str(r.get(key, ""))) for r in records)
    return total / len(records)


# ===================================================================
# HyperparameterAutoConfig
# ===================================================================


class HyperparameterAutoConfig:
    """Auto-configures training hyperparameters based on curriculum size.

    Rules:
        - Small (<100 examples): more epochs, lower LR, smaller rank
        - Medium (100-300): default values
        - Large (>300): fewer epochs, larger batch, larger rank
    """

    def configure(
        self,
        curriculum_size: int,
        base_family: BaseModelFamily,
        curriculum_path: Path,
        output_dir: Path,
    ) -> TrainingConfig:
        """Generate a TrainingConfig tuned for the curriculum size.

        Args:
            curriculum_size: Number of training examples.
            base_family: Target model architecture family.
            curriculum_path: Path to the JSONL curriculum file.
            output_dir: Output directory for training artifacts.

        Returns:
            A TrainingConfig with auto-tuned hyperparameters.

        Raises:
            TrainingError: If curriculum_size is less than 1.
        """
        if curriculum_size < 1:
            raise TrainingError("Curriculum must contain at least 1 example")
        base_model = _DEFAULT_BASE_MODELS[base_family]
        params = self._compute_params(curriculum_size)
        lora = LoRAConfig(rank=params["lora_rank"], alpha=params["lora_rank"] * 2)
        return TrainingConfig(
            base_model=base_model,
            base_model_family=base_family,
            curriculum_path=curriculum_path,
            output_dir=output_dir,
            lora=lora,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
        )

    @staticmethod
    def _compute_params(curriculum_size: int) -> dict[str, Any]:
        """Compute hyperparameters based on curriculum size.

        Args:
            curriculum_size: Number of training examples.

        Returns:
            Dictionary of parameter name to value.
        """
        if curriculum_size < 100:
            return {
                "epochs": 5,
                "batch_size": 4,
                "learning_rate": 1e-4,
                "lora_rank": 16,
            }
        if curriculum_size <= 300:
            return {
                "epochs": 3,
                "batch_size": 4,
                "learning_rate": 2e-4,
                "lora_rank": 16,
            }
        return {
            "epochs": 2,
            "batch_size": 8,
            "learning_rate": 2e-4,
            "lora_rank": 32,
        }


# ===================================================================
# TrainingPipeline
# ===================================================================


class TrainingPipeline:
    """Orchestrates a single LoRA training run.

    Manages the lifecycle from configuration validation through
    curriculum loading, training execution, and result persistence.
    The actual GPU training is abstracted behind _execute_training,
    which defaults to a dry-run simulator for MVP testing.

    Attributes:
        config: The training configuration.
    """

    def __init__(self, config: TrainingConfig) -> None:
        """Initialize the pipeline.

        Args:
            config: Training configuration to use.
        """
        self.config = config
        self._status = TrainingStatus.PENDING
        self._train_data: list[dict[str, Any]] = []
        self._val_data: list[dict[str, Any]] = []
        self._total_examples = 0

    def get_status(self) -> TrainingStatus:
        """Return the current pipeline status.

        Returns:
            Current TrainingStatus.
        """
        return self._status

    def prepare(self) -> None:
        """Validate config, load curriculum, create output directory.

        Raises:
            TrainingError: If validation or loading fails.
        """
        self._validate_config()
        self._load_curriculum()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._status = TrainingStatus.PREPARING

    def run(
        self,
        callback: TrainingProgressCallback | None = None,
    ) -> TrainingResult:
        """Execute the training pipeline.

        Args:
            callback: Optional progress callback.

        Returns:
            TrainingResult with status, metrics, and adapter path.

        Raises:
            TrainingError: If pipeline is not prepared or was cancelled.
        """
        self._check_runnable()
        self._status = TrainingStatus.TRAINING
        return self._execute_training(self._train_data, self._val_data, callback)

    def cancel(self) -> None:
        """Cancel the training pipeline."""
        self._status = TrainingStatus.CANCELLED

    def save_result(self, result: TrainingResult) -> Path:
        """Save training result to a JSON file in the output directory.

        Args:
            result: The training result to save.

        Returns:
            Path to the saved JSON file.
        """
        path = self.config.output_dir / "training_result.json"
        path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return path

    def _validate_config(self) -> None:
        """Validate the training configuration.

        Raises:
            TrainingError: If any config value is invalid.
        """
        if not self.config.base_model:
            raise TrainingError("base_model must not be empty")
        if self.config.epochs < 1:
            raise TrainingError("epochs must be at least 1")
        if self.config.learning_rate <= 0:
            raise TrainingError("learning_rate must be positive")
        if self.config.batch_size < 1:
            raise TrainingError("batch_size must be at least 1")

    def _load_curriculum(self) -> None:
        """Load and split curriculum data.

        Raises:
            TrainingError: If curriculum cannot be loaded.
        """
        loader = CurriculumLoader()
        records = loader.load(self.config.curriculum_path)
        self._total_examples = len(records)
        self._train_data, self._val_data = loader.split_train_val(
            records, self.config.validation_split, self.config.seed
        )

    def _create_training_args(self) -> dict[str, Any]:
        """Generate a training arguments dictionary.

        Returns:
            Dictionary compatible with HuggingFace TrainingArguments.
        """
        return {
            "output_dir": str(self.config.output_dir),
            "num_train_epochs": self.config.epochs,
            "per_device_train_batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "warmup_ratio": self.config.warmup_ratio,
            "weight_decay": self.config.weight_decay,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "max_seq_length": self.config.max_seq_length,
            "seed": self.config.seed,
            "lora_r": self.config.lora.rank,
            "lora_alpha": self.config.lora.alpha,
            "lora_dropout": self.config.lora.dropout,
            "lora_target_modules": self.config.lora.target_modules,
        }

    def _execute_training(
        self,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
        callback: TrainingProgressCallback | None,
    ) -> TrainingResult:
        """Run the dry-run training simulator.

        Simulates training by generating decreasing loss metrics
        across epochs and steps. In production, this method would
        delegate to Unsloth or Axolotl.

        Args:
            train_data: Training split records.
            val_data: Validation split records.
            callback: Optional progress callback.

        Returns:
            A completed TrainingResult with simulated metrics.
        """
        started_at = datetime.now()
        adapter_path = self.config.output_dir / "adapter"
        adapter_path.mkdir(parents=True, exist_ok=True)
        metrics_history = self._simulate_metrics(train_data, val_data, callback)
        completed_at = datetime.now()
        result = self._build_result(
            adapter_path,
            metrics_history,
            started_at,
            completed_at,
            train_data,
            val_data,
        )
        if callback is not None:
            callback.on_complete(result)
        self._status = TrainingStatus.COMPLETED
        return result

    def _simulate_metrics(
        self,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
        callback: TrainingProgressCallback | None,
    ) -> list[TrainingMetrics]:
        """Generate simulated training metrics.

        Args:
            train_data: Training records.
            val_data: Validation records.
            callback: Optional callback to notify.

        Returns:
            List of simulated TrainingMetrics.
        """
        metrics_history: list[TrainingMetrics] = []
        steps_per_epoch = max(
            1,
            math.ceil(len(train_data) / self.config.batch_size),
        )
        global_step = 0
        for epoch in range(1, self.config.epochs + 1):
            for step in range(steps_per_epoch):
                global_step += 1
                metrics = self._make_step_metrics(epoch, global_step, val_data)
                metrics_history.append(metrics)
                if callback is not None:
                    callback.on_step(metrics)
            if callback is not None:
                callback.on_epoch_end(epoch, metrics_history[-1])
        return metrics_history

    def _make_step_metrics(
        self,
        epoch: int,
        global_step: int,
        val_data: list[dict[str, Any]],
    ) -> TrainingMetrics:
        """Create a single simulated metrics entry.

        Args:
            epoch: Current epoch.
            global_step: Global step counter.
            val_data: Validation data (for determining val_loss).

        Returns:
            TrainingMetrics with simulated loss.
        """
        base_loss = 2.0 / (1.0 + global_step * 0.1)
        val_loss = base_loss * 1.1 if val_data else None
        return TrainingMetrics(
            epoch=epoch,
            step=global_step,
            train_loss=base_loss,
            val_loss=val_loss,
            learning_rate=self.config.learning_rate,
            timestamp=datetime.now(),
        )

    def _build_result(
        self,
        adapter_path: Path,
        metrics_history: list[TrainingMetrics],
        started_at: datetime,
        completed_at: datetime,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
    ) -> TrainingResult:
        """Assemble a TrainingResult from completed training.

        Args:
            adapter_path: Path to saved adapter.
            metrics_history: Collected metrics.
            started_at: Training start time.
            completed_at: Training end time.
            train_data: Training split.
            val_data: Validation split.

        Returns:
            A completed TrainingResult.
        """
        return TrainingResult(
            config=self.config,
            status=TrainingStatus.COMPLETED,
            adapter_path=adapter_path,
            metrics_history=metrics_history,
            total_examples=self._total_examples,
            training_examples=len(train_data),
            validation_examples=len(val_data),
            started_at=started_at,
            completed_at=completed_at,
        )

    def _check_runnable(self) -> None:
        """Verify pipeline is in a runnable state.

        Raises:
            TrainingError: If not prepared or cancelled.
        """
        if self._status == TrainingStatus.CANCELLED:
            raise TrainingError("Pipeline has been cancelled")
        if self._status == TrainingStatus.PENDING:
            raise TrainingError("Pipeline must be prepared before running — call prepare() first")


# ===================================================================
# TrainingRun
# ===================================================================


@dataclass
class TrainingRun:
    """Lightweight reference to a completed training run.

    Attributes:
        run_id: Unique identifier for this run.
        config_path: Path to the saved config JSON.
        result_path: Path to the saved result JSON.
        adapter_path: Path to the saved adapter (None if failed).
        discipline_id: Discipline from curriculum metadata.
        created_at: When the run was registered.
    """

    run_id: str
    config_path: Path
    result_path: Path
    adapter_path: Path | None
    discipline_id: str
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "run_id": self.run_id,
            "config_path": str(self.config_path),
            "result_path": str(self.result_path),
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "discipline_id": self.discipline_id,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingRun:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with run reference values.

        Returns:
            TrainingRun instance.
        """
        adapter_path = Path(data["adapter_path"]) if data.get("adapter_path") else None
        return cls(
            run_id=data["run_id"],
            config_path=Path(data["config_path"]),
            result_path=Path(data["result_path"]),
            adapter_path=adapter_path,
            discipline_id=data["discipline_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
        )


# ===================================================================
# TrainingRegistry
# ===================================================================


class TrainingRegistry:
    """Persistent registry of training runs.

    Stores run metadata as individual JSON files in a registry
    directory. Supports listing, filtering, and retrieval.

    Attributes:
        registry_dir: Directory where run metadata files are stored.
    """

    def __init__(self, registry_dir: Path) -> None:
        """Initialize the registry.

        Args:
            registry_dir: Directory for storing run metadata.
        """
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def register_run(self, result: TrainingResult) -> TrainingRun:
        """Register a completed training run.

        Args:
            result: The training result to register.

        Returns:
            A TrainingRun reference for retrieval.
        """
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        discipline_id = self._extract_discipline_id(result)
        run = self._create_run_entry(run_id, result, discipline_id)
        self._save_run(run)
        return run

    def get_run(self, run_id: str) -> TrainingRun | None:
        """Retrieve a training run by ID.

        Args:
            run_id: The unique run identifier.

        Returns:
            TrainingRun if found, None otherwise.
        """
        path = self.registry_dir / f"{run_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return TrainingRun.from_dict(data)

    def list_runs(self, discipline_id: str | None = None) -> list[TrainingRun]:
        """List all registered runs, optionally filtered by discipline.

        Args:
            discipline_id: If provided, only return runs for this discipline.

        Returns:
            List of TrainingRun references sorted by creation time.
        """
        runs = self._load_all_runs()
        if discipline_id is not None:
            runs = [r for r in runs if r.discipline_id == discipline_id]
        return sorted(runs, key=lambda r: r.created_at)

    def get_latest_run(self, discipline_id: str | None = None) -> TrainingRun | None:
        """Return the most recently created run.

        Args:
            discipline_id: If provided, filter by this discipline.

        Returns:
            The latest TrainingRun, or None if no runs exist.
        """
        runs = self.list_runs(discipline_id=discipline_id)
        if not runs:
            return None
        return runs[-1]

    def _load_all_runs(self) -> list[TrainingRun]:
        """Load all run entries from the registry directory.

        Returns:
            List of all stored TrainingRun entries.
        """
        runs: list[TrainingRun] = []
        for path in self.registry_dir.glob("run_*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            runs.append(TrainingRun.from_dict(data))
        return runs

    def _save_run(self, run: TrainingRun) -> None:
        """Save a run entry to disk.

        Args:
            run: The TrainingRun to persist.
        """
        path = self.registry_dir / f"{run.run_id}.json"
        path.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def _extract_discipline_id(result: TrainingResult) -> str:
        """Extract discipline_id from the curriculum path.

        Reads the first record of the curriculum file to find
        the discipline_id in its metadata.

        Args:
            result: Training result with config pointing to curriculum.

        Returns:
            Discipline ID string, or 'unknown' if not found.
        """
        try:
            path = result.config.curriculum_path
            if not path.exists():
                return "unknown"
            first_line = path.read_text(encoding="utf-8").split("\n")[0]
            record = json.loads(first_line)
            return record.get("metadata", {}).get("discipline_id", "unknown")
        except (json.JSONDecodeError, IndexError, OSError):
            return "unknown"

    @staticmethod
    def _create_run_entry(
        run_id: str,
        result: TrainingResult,
        discipline_id: str,
    ) -> TrainingRun:
        """Build a TrainingRun entry from a result.

        Args:
            run_id: Unique run identifier.
            result: The completed training result.
            discipline_id: Extracted discipline identifier.

        Returns:
            A TrainingRun reference.
        """
        config_path = result.config.output_dir / "config.json"
        result_path = result.config.output_dir / "training_result.json"
        return TrainingRun(
            run_id=run_id,
            config_path=config_path,
            result_path=result_path,
            adapter_path=result.adapter_path,
            discipline_id=discipline_id,
            created_at=datetime.now(),
        )
