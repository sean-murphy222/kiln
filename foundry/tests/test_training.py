"""Tests for the Foundry LoRA training pipeline.

TDD test suite covering configuration, curriculum loading,
hyperparameter auto-configuration, training pipeline execution,
and the training registry.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pytest

from foundry.src.training import (
    BaseModelFamily,
    CurriculumLoader,
    HyperparameterAutoConfig,
    LoRAConfig,
    TrainingConfig,
    TrainingError,
    TrainingMetrics,
    TrainingPipeline,
    TrainingRegistry,
    TrainingResult,
    TrainingRun,
    TrainingStatus,
)

# ===================================================================
# TestLoRAConfig
# ===================================================================


class TestLoRAConfig:
    """Tests for LoRAConfig dataclass."""

    def test_default_construction(self) -> None:
        """LoRAConfig has sensible defaults."""
        config = LoRAConfig()
        assert config.rank == 16
        assert config.alpha == 32
        assert config.dropout == 0.05
        assert config.target_modules == ["q_proj", "v_proj"]

    def test_custom_construction(self) -> None:
        """LoRAConfig accepts custom parameters."""
        config = LoRAConfig(
            rank=64,
            alpha=128,
            dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        assert config.rank == 64
        assert config.alpha == 128
        assert config.dropout == 0.1
        assert len(config.target_modules) == 4

    def test_to_dict(self) -> None:
        """LoRAConfig serializes to dictionary."""
        config = LoRAConfig(rank=32, alpha=64)
        data = config.to_dict()
        assert data["rank"] == 32
        assert data["alpha"] == 64
        assert data["dropout"] == 0.05
        assert data["target_modules"] == ["q_proj", "v_proj"]

    def test_from_dict(self) -> None:
        """LoRAConfig deserializes from dictionary."""
        data = {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.02,
            "target_modules": ["q_proj"],
        }
        config = LoRAConfig.from_dict(data)
        assert config.rank == 8
        assert config.alpha == 16
        assert config.dropout == 0.02
        assert config.target_modules == ["q_proj"]

    def test_roundtrip_serialization(self) -> None:
        """LoRAConfig survives to_dict/from_dict roundtrip."""
        original = LoRAConfig(rank=64, alpha=128, dropout=0.1)
        restored = LoRAConfig.from_dict(original.to_dict())
        assert restored.rank == original.rank
        assert restored.alpha == original.alpha
        assert restored.dropout == original.dropout
        assert restored.target_modules == original.target_modules

    def test_from_dict_with_defaults(self) -> None:
        """LoRAConfig from_dict fills missing fields with defaults."""
        data: dict = {}
        config = LoRAConfig.from_dict(data)
        assert config.rank == 16
        assert config.alpha == 32


# ===================================================================
# TestTrainingConfig
# ===================================================================


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_construction(self, tmp_path: Path) -> None:
        """TrainingConfig has sensible defaults."""
        config = TrainingConfig(
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family=BaseModelFamily.PHI,
            curriculum_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "output",
        )
        assert config.epochs == 3
        assert config.batch_size == 4
        assert config.learning_rate == 2e-4
        assert config.max_seq_length == 2048
        assert config.warmup_ratio == 0.03
        assert config.weight_decay == 0.01
        assert config.gradient_accumulation_steps == 4
        assert config.seed == 42
        assert config.validation_split == 0.1
        assert isinstance(config.lora, LoRAConfig)

    def test_custom_lora_config(self, tmp_path: Path) -> None:
        """TrainingConfig accepts custom LoRA configuration."""
        lora = LoRAConfig(rank=64, alpha=128)
        config = TrainingConfig(
            base_model="meta-llama/Llama-3-8B-Instruct",
            base_model_family=BaseModelFamily.LLAMA,
            curriculum_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "output",
            lora=lora,
        )
        assert config.lora.rank == 64
        assert config.lora.alpha == 128

    def test_to_dict(self, tmp_path: Path) -> None:
        """TrainingConfig serializes to dictionary."""
        config = TrainingConfig(
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family=BaseModelFamily.PHI,
            curriculum_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "output",
        )
        data = config.to_dict()
        assert data["base_model"] == "microsoft/phi-3-mini-4k-instruct"
        assert data["base_model_family"] == "phi"
        assert data["epochs"] == 3
        assert "lora" in data
        assert data["lora"]["rank"] == 16

    def test_from_dict(self, tmp_path: Path) -> None:
        """TrainingConfig deserializes from dictionary."""
        data = {
            "base_model": "microsoft/phi-3-mini-4k-instruct",
            "base_model_family": "phi",
            "curriculum_path": str(tmp_path / "data.jsonl"),
            "output_dir": str(tmp_path / "output"),
            "epochs": 5,
            "batch_size": 8,
            "learning_rate": 1e-4,
            "lora": {"rank": 32, "alpha": 64},
        }
        config = TrainingConfig.from_dict(data)
        assert config.base_model == "microsoft/phi-3-mini-4k-instruct"
        assert config.base_model_family == BaseModelFamily.PHI
        assert config.epochs == 5
        assert config.batch_size == 8
        assert config.lora.rank == 32

    def test_roundtrip_serialization(self, tmp_path: Path) -> None:
        """TrainingConfig survives to_dict/from_dict roundtrip."""
        original = TrainingConfig(
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family=BaseModelFamily.PHI,
            curriculum_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "output",
            epochs=5,
            learning_rate=1e-4,
        )
        restored = TrainingConfig.from_dict(original.to_dict())
        assert restored.base_model == original.base_model
        assert restored.epochs == original.epochs
        assert restored.learning_rate == original.learning_rate
        assert restored.lora.rank == original.lora.rank

    def test_base_model_family_enum(self) -> None:
        """BaseModelFamily enum contains expected model families."""
        assert BaseModelFamily.PHI == "phi"
        assert BaseModelFamily.LLAMA == "llama"
        assert BaseModelFamily.MISTRAL == "mistral"
        assert BaseModelFamily.QWEN == "qwen"


# ===================================================================
# TestTrainingMetrics
# ===================================================================


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_construction(self) -> None:
        """TrainingMetrics can be constructed with required fields."""
        now = datetime.now()
        metrics = TrainingMetrics(
            epoch=1,
            step=100,
            train_loss=0.5,
            val_loss=0.6,
            learning_rate=2e-4,
            timestamp=now,
        )
        assert metrics.epoch == 1
        assert metrics.step == 100
        assert metrics.train_loss == 0.5
        assert metrics.val_loss == 0.6
        assert metrics.learning_rate == 2e-4
        assert metrics.timestamp == now

    def test_construction_without_val_loss(self) -> None:
        """TrainingMetrics allows None val_loss."""
        metrics = TrainingMetrics(
            epoch=1,
            step=50,
            train_loss=0.8,
            val_loss=None,
            learning_rate=1e-4,
            timestamp=datetime.now(),
        )
        assert metrics.val_loss is None

    def test_to_dict(self) -> None:
        """TrainingMetrics serializes to dictionary."""
        now = datetime.now()
        metrics = TrainingMetrics(
            epoch=2,
            step=200,
            train_loss=0.3,
            val_loss=0.4,
            learning_rate=1e-4,
            timestamp=now,
        )
        data = metrics.to_dict()
        assert data["epoch"] == 2
        assert data["step"] == 200
        assert data["train_loss"] == 0.3
        assert data["val_loss"] == 0.4
        assert data["learning_rate"] == 1e-4
        assert data["timestamp"] == now.isoformat()

    def test_to_dict_none_val_loss(self) -> None:
        """TrainingMetrics serialization preserves None val_loss."""
        metrics = TrainingMetrics(
            epoch=1,
            step=1,
            train_loss=1.0,
            val_loss=None,
            learning_rate=2e-4,
            timestamp=datetime.now(),
        )
        data = metrics.to_dict()
        assert data["val_loss"] is None


# ===================================================================
# TestTrainingResult
# ===================================================================


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_construction_pending(self, tmp_path: Path) -> None:
        """TrainingResult starts in PENDING status."""
        config = TrainingConfig(
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family=BaseModelFamily.PHI,
            curriculum_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "output",
        )
        result = TrainingResult(
            config=config,
            status=TrainingStatus.PENDING,
            total_examples=100,
            training_examples=90,
            validation_examples=10,
        )
        assert result.status == TrainingStatus.PENDING
        assert result.adapter_path is None
        assert result.metrics_history == []
        assert result.started_at is None
        assert result.completed_at is None
        assert result.error_message is None

    def test_to_dict(self, tmp_path: Path) -> None:
        """TrainingResult serializes to dictionary."""
        config = TrainingConfig(
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family=BaseModelFamily.PHI,
            curriculum_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "output",
        )
        now = datetime.now()
        result = TrainingResult(
            config=config,
            status=TrainingStatus.COMPLETED,
            adapter_path=tmp_path / "adapter",
            total_examples=100,
            training_examples=90,
            validation_examples=10,
            started_at=now,
            completed_at=now,
        )
        data = result.to_dict()
        assert data["status"] == "completed"
        assert data["total_examples"] == 100
        assert data["training_examples"] == 90
        assert data["validation_examples"] == 10
        assert "config" in data

    def test_from_dict(self, tmp_path: Path) -> None:
        """TrainingResult deserializes from dictionary."""
        now = datetime.now()
        data = {
            "config": {
                "base_model": "microsoft/phi-3-mini-4k-instruct",
                "base_model_family": "phi",
                "curriculum_path": str(tmp_path / "data.jsonl"),
                "output_dir": str(tmp_path / "output"),
                "lora": {},
            },
            "status": "completed",
            "adapter_path": str(tmp_path / "adapter"),
            "metrics_history": [],
            "total_examples": 100,
            "training_examples": 90,
            "validation_examples": 10,
            "started_at": now.isoformat(),
            "completed_at": now.isoformat(),
            "error_message": None,
        }
        result = TrainingResult.from_dict(data)
        assert result.status == TrainingStatus.COMPLETED
        assert result.total_examples == 100

    def test_roundtrip_serialization(self, tmp_path: Path) -> None:
        """TrainingResult survives to_dict/from_dict roundtrip."""
        config = TrainingConfig(
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family=BaseModelFamily.PHI,
            curriculum_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "output",
        )
        now = datetime.now()
        original = TrainingResult(
            config=config,
            status=TrainingStatus.COMPLETED,
            adapter_path=tmp_path / "adapter",
            total_examples=50,
            training_examples=45,
            validation_examples=5,
            started_at=now,
            completed_at=now,
        )
        restored = TrainingResult.from_dict(original.to_dict())
        assert restored.status == original.status
        assert restored.total_examples == original.total_examples
        assert restored.training_examples == original.training_examples

    def test_status_enum_values(self) -> None:
        """TrainingStatus enum contains expected values."""
        assert TrainingStatus.PENDING == "pending"
        assert TrainingStatus.PREPARING == "preparing"
        assert TrainingStatus.TRAINING == "training"
        assert TrainingStatus.COMPLETED == "completed"
        assert TrainingStatus.FAILED == "failed"
        assert TrainingStatus.CANCELLED == "cancelled"

    def test_result_with_error(self, tmp_path: Path) -> None:
        """TrainingResult can record an error state."""
        config = TrainingConfig(
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family=BaseModelFamily.PHI,
            curriculum_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "output",
        )
        result = TrainingResult(
            config=config,
            status=TrainingStatus.FAILED,
            total_examples=50,
            training_examples=45,
            validation_examples=5,
            error_message="CUDA out of memory",
        )
        assert result.status == TrainingStatus.FAILED
        assert result.error_message == "CUDA out of memory"
        data = result.to_dict()
        assert data["error_message"] == "CUDA out of memory"

    def test_result_with_metrics_history(self, tmp_path: Path) -> None:
        """TrainingResult can store metrics history."""
        config = TrainingConfig(
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family=BaseModelFamily.PHI,
            curriculum_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "output",
        )
        now = datetime.now()
        metrics = [
            TrainingMetrics(
                epoch=1,
                step=i * 10,
                train_loss=1.0 - i * 0.1,
                val_loss=None,
                learning_rate=2e-4,
                timestamp=now,
            )
            for i in range(5)
        ]
        result = TrainingResult(
            config=config,
            status=TrainingStatus.COMPLETED,
            metrics_history=metrics,
            total_examples=50,
            training_examples=45,
            validation_examples=5,
        )
        data = result.to_dict()
        assert len(data["metrics_history"]) == 5
        assert data["metrics_history"][0]["step"] == 0
        assert data["metrics_history"][4]["step"] == 40


# ===================================================================
# TestCurriculumLoader
# ===================================================================


class TestCurriculumLoader:
    """Tests for CurriculumLoader."""

    def test_load_valid_jsonl(self, sample_curriculum_path: Path) -> None:
        """CurriculumLoader loads valid JSONL file."""
        loader = CurriculumLoader()
        records = loader.load(sample_curriculum_path)
        assert len(records) == 20

    def test_load_returns_alpaca_records(self, sample_curriculum_path: Path) -> None:
        """Loaded records have Alpaca format fields."""
        loader = CurriculumLoader()
        records = loader.load(sample_curriculum_path)
        for record in records:
            assert "instruction" in record
            assert "input" in record
            assert "output" in record
            assert "metadata" in record

    def test_load_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """CurriculumLoader raises TrainingError for missing file."""
        loader = CurriculumLoader()
        with pytest.raises(TrainingError, match="not found"):
            loader.load(tmp_path / "nonexistent.jsonl")

    def test_load_empty_file_raises(self, tmp_path: Path) -> None:
        """CurriculumLoader raises TrainingError for empty file."""
        empty_path = tmp_path / "empty.jsonl"
        empty_path.write_text("", encoding="utf-8")
        loader = CurriculumLoader()
        with pytest.raises(TrainingError, match="empty"):
            loader.load(empty_path)

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        """CurriculumLoader raises TrainingError for malformed JSON."""
        bad_path = tmp_path / "bad.jsonl"
        bad_path.write_text("not valid json\n", encoding="utf-8")
        loader = CurriculumLoader()
        with pytest.raises(TrainingError, match="parse"):
            loader.load(bad_path)

    def test_validate_record_valid(self) -> None:
        """validate_record returns True for valid Alpaca record."""
        loader = CurriculumLoader()
        record = {
            "instruction": "How?",
            "input": "",
            "output": "Like this.",
            "metadata": {"example_id": "ex_001"},
        }
        assert loader.validate_record(record) is True

    def test_validate_record_missing_instruction(self) -> None:
        """validate_record returns False when instruction is missing."""
        loader = CurriculumLoader()
        record = {"input": "", "output": "answer", "metadata": {}}
        assert loader.validate_record(record) is False

    def test_validate_record_missing_output(self) -> None:
        """validate_record returns False when output is missing."""
        loader = CurriculumLoader()
        record = {"instruction": "q", "input": "", "metadata": {}}
        assert loader.validate_record(record) is False

    def test_validate_record_empty_instruction(self) -> None:
        """validate_record returns False when instruction is empty string."""
        loader = CurriculumLoader()
        record = {"instruction": "", "input": "", "output": "a", "metadata": {}}
        assert loader.validate_record(record) is False

    def test_validate_record_empty_output(self) -> None:
        """validate_record returns False when output is empty string."""
        loader = CurriculumLoader()
        record = {"instruction": "q", "input": "", "output": "", "metadata": {}}
        assert loader.validate_record(record) is False

    def test_split_train_val(self, sample_curriculum_path: Path) -> None:
        """split_train_val divides records into train and validation sets."""
        loader = CurriculumLoader()
        records = loader.load(sample_curriculum_path)
        train, val = loader.split_train_val(records, val_ratio=0.2, seed=42)
        assert len(train) + len(val) == len(records)
        assert len(val) == 4  # 20 * 0.2

    def test_split_train_val_deterministic(self, sample_curriculum_path: Path) -> None:
        """split_train_val is deterministic with same seed."""
        loader = CurriculumLoader()
        records = loader.load(sample_curriculum_path)
        train1, val1 = loader.split_train_val(records, val_ratio=0.1, seed=42)
        train2, val2 = loader.split_train_val(records, val_ratio=0.1, seed=42)
        assert train1 == train2
        assert val1 == val2

    def test_split_train_val_different_seeds(self, sample_curriculum_path: Path) -> None:
        """split_train_val produces different splits with different seeds."""
        loader = CurriculumLoader()
        records = loader.load(sample_curriculum_path)
        train1, val1 = loader.split_train_val(records, val_ratio=0.2, seed=42)
        train2, val2 = loader.split_train_val(records, val_ratio=0.2, seed=99)
        # With different seeds, the splits should differ
        assert val1 != val2

    def test_split_train_val_zero_ratio(self, sample_curriculum_path: Path) -> None:
        """split_train_val with 0.0 ratio puts everything in train."""
        loader = CurriculumLoader()
        records = loader.load(sample_curriculum_path)
        train, val = loader.split_train_val(records, val_ratio=0.0, seed=42)
        assert len(train) == 20
        assert len(val) == 0

    def test_get_statistics(self, sample_curriculum_path: Path) -> None:
        """get_statistics returns counts by competency and lengths."""
        loader = CurriculumLoader()
        records = loader.load(sample_curriculum_path)
        stats = loader.get_statistics(records)
        assert stats["total_records"] == 20
        assert "competency_counts" in stats
        assert "comp_test001" in stats["competency_counts"]
        assert "comp_test002" in stats["competency_counts"]
        assert stats["competency_counts"]["comp_test001"] == 10
        assert stats["competency_counts"]["comp_test002"] == 10
        assert "avg_instruction_length" in stats
        assert "avg_output_length" in stats
        assert stats["avg_instruction_length"] > 0
        assert stats["avg_output_length"] > 0

    def test_get_statistics_with_discipline_counts(self, sample_curriculum_path: Path) -> None:
        """get_statistics includes discipline distribution."""
        loader = CurriculumLoader()
        records = loader.load(sample_curriculum_path)
        stats = loader.get_statistics(records)
        assert "discipline_counts" in stats
        assert stats["discipline_counts"]["disc_test001"] == 20

    def test_load_filters_invalid_records(self, tmp_path: Path) -> None:
        """CurriculumLoader filters out invalid records with warning."""
        path = tmp_path / "mixed.jsonl"
        valid = {
            "instruction": "How?",
            "input": "",
            "output": "Like this.",
            "metadata": {"example_id": "ex_001"},
        }
        invalid = {"instruction": "", "input": "", "output": "", "metadata": {}}
        lines = [json.dumps(valid), json.dumps(invalid), json.dumps(valid)]
        path.write_text("\n".join(lines), encoding="utf-8")
        loader = CurriculumLoader()
        records = loader.load(path)
        assert len(records) == 2


# ===================================================================
# TestHyperparameterAutoConfig
# ===================================================================


class TestHyperparameterAutoConfig:
    """Tests for HyperparameterAutoConfig."""

    def test_small_curriculum(self, tmp_path: Path) -> None:
        """Small curriculum (<100) gets more epochs and lower learning rate."""
        auto = HyperparameterAutoConfig()
        config = auto.configure(
            curriculum_size=50,
            base_family=BaseModelFamily.PHI,
            curriculum_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "output",
        )
        assert config.epochs > 3  # More epochs for small dataset
        assert config.learning_rate < 2e-4  # Lower LR for small dataset

    def test_medium_curriculum(self, tmp_path: Path) -> None:
        """Medium curriculum (100-300) gets default hyperparameters."""
        auto = HyperparameterAutoConfig()
        config = auto.configure(
            curriculum_size=200,
            base_family=BaseModelFamily.PHI,
            curriculum_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "output",
        )
        assert config.epochs == 3
        assert config.learning_rate == 2e-4

    def test_large_curriculum(self, tmp_path: Path) -> None:
        """Large curriculum (>300) gets fewer epochs and larger batch."""
        auto = HyperparameterAutoConfig()
        config = auto.configure(
            curriculum_size=500,
            base_family=BaseModelFamily.PHI,
            curriculum_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "output",
        )
        assert config.epochs < 3  # Fewer epochs for large dataset
        assert config.batch_size > 4  # Larger batch for large dataset

    def test_phi_family_config(self, tmp_path: Path) -> None:
        """PHI family gets appropriate base model and modules."""
        auto = HyperparameterAutoConfig()
        config = auto.configure(
            curriculum_size=200,
            base_family=BaseModelFamily.PHI,
            curriculum_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "output",
        )
        assert config.base_model_family == BaseModelFamily.PHI
        assert "phi" in config.base_model.lower()

    def test_llama_family_config(self, tmp_path: Path) -> None:
        """LLAMA family gets appropriate base model."""
        auto = HyperparameterAutoConfig()
        config = auto.configure(
            curriculum_size=200,
            base_family=BaseModelFamily.LLAMA,
            curriculum_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "output",
        )
        assert config.base_model_family == BaseModelFamily.LLAMA
        assert "llama" in config.base_model.lower()

    def test_mistral_family_config(self, tmp_path: Path) -> None:
        """MISTRAL family gets appropriate base model."""
        auto = HyperparameterAutoConfig()
        config = auto.configure(
            curriculum_size=200,
            base_family=BaseModelFamily.MISTRAL,
            curriculum_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "output",
        )
        assert config.base_model_family == BaseModelFamily.MISTRAL
        assert "mistral" in config.base_model.lower()

    def test_qwen_family_config(self, tmp_path: Path) -> None:
        """QWEN family gets appropriate base model."""
        auto = HyperparameterAutoConfig()
        config = auto.configure(
            curriculum_size=200,
            base_family=BaseModelFamily.QWEN,
            curriculum_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "output",
        )
        assert config.base_model_family == BaseModelFamily.QWEN
        assert "qwen" in config.base_model.lower()

    def test_lora_rank_scales_with_size(self, tmp_path: Path) -> None:
        """LoRA rank increases for larger curricula."""
        auto = HyperparameterAutoConfig()
        small = auto.configure(
            curriculum_size=50,
            base_family=BaseModelFamily.PHI,
            curriculum_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "output",
        )
        large = auto.configure(
            curriculum_size=500,
            base_family=BaseModelFamily.PHI,
            curriculum_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "output",
        )
        assert large.lora.rank >= small.lora.rank

    def test_zero_curriculum_raises(self, tmp_path: Path) -> None:
        """configure raises TrainingError for zero-size curriculum."""
        auto = HyperparameterAutoConfig()
        with pytest.raises(TrainingError, match="at least 1"):
            auto.configure(
                curriculum_size=0,
                base_family=BaseModelFamily.PHI,
                curriculum_path=tmp_path / "data.jsonl",
                output_dir=tmp_path / "output",
            )


# ===================================================================
# TestTrainingPipeline
# ===================================================================


class TestTrainingPipeline:
    """Tests for TrainingPipeline."""

    def test_initial_status_pending(self, sample_training_config: TrainingConfig) -> None:
        """Pipeline starts in PENDING status."""
        pipeline = TrainingPipeline(config=sample_training_config)
        assert pipeline.get_status() == TrainingStatus.PENDING

    def test_prepare_validates_config(self, sample_training_config: TrainingConfig) -> None:
        """prepare() validates config and loads curriculum."""
        pipeline = TrainingPipeline(config=sample_training_config)
        pipeline.prepare()
        assert pipeline.get_status() == TrainingStatus.PREPARING

    def test_prepare_creates_output_dir(self, sample_training_config: TrainingConfig) -> None:
        """prepare() creates the output directory."""
        pipeline = TrainingPipeline(config=sample_training_config)
        pipeline.prepare()
        assert sample_training_config.output_dir.exists()

    def test_prepare_invalid_curriculum_path(self, tmp_path: Path) -> None:
        """prepare() raises TrainingError for missing curriculum file."""
        config = TrainingConfig(
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family=BaseModelFamily.PHI,
            curriculum_path=tmp_path / "nonexistent.jsonl",
            output_dir=tmp_path / "output",
        )
        pipeline = TrainingPipeline(config=config)
        with pytest.raises(TrainingError):
            pipeline.prepare()

    def test_run_dry_run(self, training_pipeline: TrainingPipeline) -> None:
        """run() in dry-run mode completes without GPU."""
        result = training_pipeline.run()
        assert result.status == TrainingStatus.COMPLETED
        assert result.total_examples == 20
        assert result.training_examples + result.validation_examples == 20

    def test_run_produces_metrics(self, training_pipeline: TrainingPipeline) -> None:
        """run() produces metrics history."""
        result = training_pipeline.run()
        assert len(result.metrics_history) > 0
        for m in result.metrics_history:
            assert m.train_loss >= 0
            assert m.learning_rate > 0

    def test_run_sets_timestamps(self, training_pipeline: TrainingPipeline) -> None:
        """run() sets started_at and completed_at."""
        result = training_pipeline.run()
        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.completed_at >= result.started_at

    def test_run_creates_adapter_dir(self, training_pipeline: TrainingPipeline) -> None:
        """run() creates adapter output path."""
        result = training_pipeline.run()
        assert result.adapter_path is not None
        assert result.adapter_path.exists()

    def test_run_with_callback(self, training_pipeline: TrainingPipeline) -> None:
        """run() invokes callback methods."""
        steps: list[TrainingMetrics] = []
        epochs: list[int] = []
        completed: list[TrainingResult] = []

        class TestCallback:
            """Test callback that records invocations."""

            def on_step(self, metrics: TrainingMetrics) -> None:
                steps.append(metrics)

            def on_epoch_end(self, epoch: int, metrics: TrainingMetrics) -> None:
                epochs.append(epoch)

            def on_complete(self, result: TrainingResult) -> None:
                completed.append(result)

            def on_error(self, error: str) -> None:
                pass

        callback = TestCallback()
        training_pipeline.run(callback=callback)
        assert len(steps) > 0
        assert len(epochs) > 0
        assert len(completed) == 1
        assert completed[0].status == TrainingStatus.COMPLETED

    def test_cancel(self, sample_training_config: TrainingConfig) -> None:
        """cancel() sets status to CANCELLED."""
        pipeline = TrainingPipeline(config=sample_training_config)
        pipeline.prepare()
        pipeline.cancel()
        assert pipeline.get_status() == TrainingStatus.CANCELLED

    def test_run_after_cancel_raises(self, sample_training_config: TrainingConfig) -> None:
        """run() after cancel raises TrainingError."""
        pipeline = TrainingPipeline(config=sample_training_config)
        pipeline.prepare()
        pipeline.cancel()
        with pytest.raises(TrainingError, match="cancelled"):
            pipeline.run()

    def test_run_without_prepare_raises(self, sample_training_config: TrainingConfig) -> None:
        """run() without prepare raises TrainingError."""
        pipeline = TrainingPipeline(config=sample_training_config)
        with pytest.raises(TrainingError, match="prepare"):
            pipeline.run()

    def test_save_result(self, training_pipeline: TrainingPipeline) -> None:
        """save_result writes result JSON to output directory."""
        result = training_pipeline.run()
        path = training_pipeline.save_result(result)
        assert path.exists()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["status"] == "completed"
        assert data["total_examples"] == 20

    def test_create_training_args(self, training_pipeline: TrainingPipeline) -> None:
        """_create_training_args produces a valid config dict."""
        args = training_pipeline._create_training_args()
        assert "learning_rate" in args
        assert "num_train_epochs" in args
        assert "per_device_train_batch_size" in args
        assert "output_dir" in args
        assert "seed" in args

    def test_validate_config_bad_model(self, tmp_path: Path) -> None:
        """_validate_config rejects empty model name."""
        config = TrainingConfig(
            base_model="",
            base_model_family=BaseModelFamily.PHI,
            curriculum_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "output",
        )
        pipeline = TrainingPipeline(config=config)
        with pytest.raises(TrainingError, match="base_model"):
            pipeline._validate_config()

    def test_validate_config_bad_epochs(self, sample_curriculum_path: Path, tmp_path: Path) -> None:
        """_validate_config rejects zero epochs."""
        config = TrainingConfig(
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family=BaseModelFamily.PHI,
            curriculum_path=sample_curriculum_path,
            output_dir=tmp_path / "output",
            epochs=0,
        )
        pipeline = TrainingPipeline(config=config)
        with pytest.raises(TrainingError, match="epochs"):
            pipeline._validate_config()

    def test_validate_config_bad_learning_rate(
        self, sample_curriculum_path: Path, tmp_path: Path
    ) -> None:
        """_validate_config rejects non-positive learning rate."""
        config = TrainingConfig(
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family=BaseModelFamily.PHI,
            curriculum_path=sample_curriculum_path,
            output_dir=tmp_path / "output",
            learning_rate=0.0,
        )
        pipeline = TrainingPipeline(config=config)
        with pytest.raises(TrainingError, match="learning_rate"):
            pipeline._validate_config()

    def test_validate_config_bad_batch_size(
        self, sample_curriculum_path: Path, tmp_path: Path
    ) -> None:
        """_validate_config rejects zero batch size."""
        config = TrainingConfig(
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family=BaseModelFamily.PHI,
            curriculum_path=sample_curriculum_path,
            output_dir=tmp_path / "output",
            batch_size=0,
        )
        pipeline = TrainingPipeline(config=config)
        with pytest.raises(TrainingError, match="batch_size"):
            pipeline._validate_config()


# ===================================================================
# TestTrainingRegistry
# ===================================================================


class TestTrainingRegistry:
    """Tests for TrainingRegistry."""

    def _make_result(
        self,
        tmp_path: Path,
        discipline_id: str = "disc_test001",
    ) -> TrainingResult:
        """Create a minimal completed TrainingResult for registry tests."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        curriculum_path = tmp_path / "data.jsonl"
        curriculum_path.write_text(
            json.dumps(
                {
                    "instruction": "q",
                    "input": "",
                    "output": "a",
                    "metadata": {"discipline_id": discipline_id},
                }
            ),
            encoding="utf-8",
        )
        config = TrainingConfig(
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family=BaseModelFamily.PHI,
            curriculum_path=curriculum_path,
            output_dir=tmp_path / "output",
        )
        adapter_path = tmp_path / "adapter"
        adapter_path.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        return TrainingResult(
            config=config,
            status=TrainingStatus.COMPLETED,
            adapter_path=adapter_path,
            total_examples=50,
            training_examples=45,
            validation_examples=5,
            started_at=now,
            completed_at=now,
        )

    def test_register_run(self, tmp_path: Path) -> None:
        """register_run creates a TrainingRun entry."""
        registry = TrainingRegistry(registry_dir=tmp_path / "registry")
        result = self._make_result(tmp_path)
        run = registry.register_run(result)
        assert isinstance(run, TrainingRun)
        assert run.run_id is not None
        assert run.adapter_path is not None

    def test_get_run(self, tmp_path: Path) -> None:
        """get_run returns a previously registered run."""
        registry = TrainingRegistry(registry_dir=tmp_path / "registry")
        result = self._make_result(tmp_path)
        run = registry.register_run(result)
        retrieved = registry.get_run(run.run_id)
        assert retrieved is not None
        assert retrieved.run_id == run.run_id

    def test_get_run_nonexistent(self, tmp_path: Path) -> None:
        """get_run returns None for unknown run_id."""
        registry = TrainingRegistry(registry_dir=tmp_path / "registry")
        assert registry.get_run("nonexistent_id") is None

    def test_list_runs(self, tmp_path: Path) -> None:
        """list_runs returns all registered runs."""
        registry = TrainingRegistry(registry_dir=tmp_path / "registry")
        result1 = self._make_result(tmp_path / "run1")
        result2 = self._make_result(tmp_path / "run2")
        registry.register_run(result1)
        registry.register_run(result2)
        runs = registry.list_runs()
        assert len(runs) == 2

    def test_list_runs_filter_by_discipline(self, tmp_path: Path) -> None:
        """list_runs filters by discipline_id."""
        registry = TrainingRegistry(registry_dir=tmp_path / "registry")
        result1 = self._make_result(tmp_path / "run1", discipline_id="disc_aaa")
        result2 = self._make_result(tmp_path / "run2", discipline_id="disc_bbb")
        registry.register_run(result1)
        registry.register_run(result2)
        runs_aaa = registry.list_runs(discipline_id="disc_aaa")
        runs_bbb = registry.list_runs(discipline_id="disc_bbb")
        assert len(runs_aaa) == 1
        assert len(runs_bbb) == 1

    def test_get_latest_run(self, tmp_path: Path) -> None:
        """get_latest_run returns the most recently registered run."""
        registry = TrainingRegistry(registry_dir=tmp_path / "registry")
        result1 = self._make_result(tmp_path / "run1")
        result2 = self._make_result(tmp_path / "run2")
        registry.register_run(result1)
        time.sleep(0.01)  # Ensure different timestamps
        run2 = registry.register_run(result2)
        latest = registry.get_latest_run()
        assert latest is not None
        assert latest.run_id == run2.run_id

    def test_get_latest_run_empty_registry(self, tmp_path: Path) -> None:
        """get_latest_run returns None when registry is empty."""
        registry = TrainingRegistry(registry_dir=tmp_path / "registry")
        assert registry.get_latest_run() is None

    def test_get_latest_run_by_discipline(self, tmp_path: Path) -> None:
        """get_latest_run filters by discipline."""
        registry = TrainingRegistry(registry_dir=tmp_path / "registry")
        result1 = self._make_result(tmp_path / "run1", discipline_id="disc_aaa")
        result2 = self._make_result(tmp_path / "run2", discipline_id="disc_bbb")
        run1 = registry.register_run(result1)
        registry.register_run(result2)
        latest_aaa = registry.get_latest_run(discipline_id="disc_aaa")
        assert latest_aaa is not None
        assert latest_aaa.run_id == run1.run_id

    def test_registry_persists_to_disk(self, tmp_path: Path) -> None:
        """Registry data persists across instances."""
        registry_dir = tmp_path / "registry"
        registry1 = TrainingRegistry(registry_dir=registry_dir)
        result = self._make_result(tmp_path)
        run = registry1.register_run(result)

        # Create a new registry instance pointing at same directory
        registry2 = TrainingRegistry(registry_dir=registry_dir)
        retrieved = registry2.get_run(run.run_id)
        assert retrieved is not None
        assert retrieved.run_id == run.run_id


# ===================================================================
# TestIntegration
# ===================================================================


class TestIntegration:
    """End-to-end integration tests for the training pipeline."""

    def test_full_pipeline_flow(self, sample_curriculum_path: Path, tmp_path: Path) -> None:
        """Full pipeline: configure -> prepare -> run -> save -> register."""
        # Step 1: Auto-configure from curriculum
        loader = CurriculumLoader()
        records = loader.load(sample_curriculum_path)
        stats = loader.get_statistics(records)

        auto = HyperparameterAutoConfig()
        config = auto.configure(
            curriculum_size=stats["total_records"],
            base_family=BaseModelFamily.PHI,
            curriculum_path=sample_curriculum_path,
            output_dir=tmp_path / "training_output",
        )

        # Step 2: Create and prepare pipeline
        pipeline = TrainingPipeline(config=config)
        pipeline.prepare()
        assert pipeline.get_status() == TrainingStatus.PREPARING

        # Step 3: Run training (dry-run)
        result = pipeline.run()
        assert result.status == TrainingStatus.COMPLETED
        assert result.adapter_path is not None
        assert result.total_examples == 20

        # Step 4: Save result
        result_path = pipeline.save_result(result)
        assert result_path.exists()

        # Step 5: Register in training registry
        registry = TrainingRegistry(registry_dir=tmp_path / "registry")
        run = registry.register_run(result)
        assert run.run_id is not None

        # Step 6: Retrieve from registry
        retrieved = registry.get_run(run.run_id)
        assert retrieved is not None

    def test_pipeline_with_auto_configured_hyperparameters(self, tmp_path: Path) -> None:
        """Pipeline uses auto-configured hyperparameters correctly."""
        # Create a small curriculum (50 records)
        path = tmp_path / "small_curriculum.jsonl"
        records = []
        for i in range(50):
            records.append(
                {
                    "instruction": f"Question {i}?",
                    "input": "",
                    "output": f"Answer {i}.",
                    "metadata": {
                        "example_id": f"ex_{i:04d}",
                        "discipline_id": "disc_test001",
                        "competency_id": "comp_test001",
                        "contributor_id": "contrib_test001",
                        "review_status": "approved",
                        "created_at": "2026-02-20T10:00:00",
                    },
                }
            )
        lines = [json.dumps(r) for r in records]
        path.write_text("\n".join(lines), encoding="utf-8")

        auto = HyperparameterAutoConfig()
        config = auto.configure(
            curriculum_size=50,
            base_family=BaseModelFamily.LLAMA,
            curriculum_path=path,
            output_dir=tmp_path / "output",
        )

        # Small curriculum should get more epochs
        assert config.epochs > 3

        pipeline = TrainingPipeline(config=config)
        pipeline.prepare()
        result = pipeline.run()
        assert result.status == TrainingStatus.COMPLETED
        assert result.total_examples == 50

    def test_curriculum_statistics_match_pipeline(
        self, sample_curriculum_path: Path, tmp_path: Path
    ) -> None:
        """Curriculum statistics are consistent with pipeline counts."""
        loader = CurriculumLoader()
        records = loader.load(sample_curriculum_path)
        stats = loader.get_statistics(records)

        config = TrainingConfig(
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family=BaseModelFamily.PHI,
            curriculum_path=sample_curriculum_path,
            output_dir=tmp_path / "output",
        )
        pipeline = TrainingPipeline(config=config)
        pipeline.prepare()
        result = pipeline.run()

        assert result.total_examples == stats["total_records"]
        assert result.training_examples + result.validation_examples == stats["total_records"]
