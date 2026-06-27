"""Tests for foundry.src.training_backends (no GPU).

Covers the pure helpers (target-module resolution, chat formatting, config
kwarg shaping), the HF->callback bridge mapping, DryRunBackend behavior, and
the environment-gated backend selection. RealLoRABackend.execute itself is
exercised by the GPU tests.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from foundry.src import backend_config as bc
from foundry.src.training import (
    BaseModelFamily,
    TrainingConfig,
    TrainingMetrics,
    TrainingStatus,
    _select_training_backend,
)
from foundry.src.training_backends import (
    DryRunBackend,
    RealLoRABackend,
    TrainingBackend,
    _BridgeLogic,
    format_chat_text,
    lora_config_kwargs,
    resolve_target_modules,
    sft_config_kwargs,
)


def _make_config(tmp_path: Path, family: str = "qwen") -> TrainingConfig:
    return TrainingConfig(
        base_model="test/model",
        base_model_family=family,
        curriculum_path=tmp_path / "curriculum.jsonl",
        output_dir=tmp_path / "out",
        epochs=2,
        batch_size=4,
    )


class _RecordingCallback:
    def __init__(self) -> None:
        self.steps: list[TrainingMetrics] = []
        self.epochs: list[int] = []
        self.completed = 0
        self.errors: list[str] = []

    def on_step(self, metrics: TrainingMetrics) -> None:
        self.steps.append(metrics)

    def on_epoch_end(self, epoch: int, metrics: TrainingMetrics) -> None:
        self.epochs.append(epoch)

    def on_complete(self, result) -> None:
        self.completed += 1

    def on_error(self, error: str) -> None:
        self.errors.append(error)


class _FakeTokenizer:
    chat_template = "TEMPLATE"

    def apply_chat_template(self, messages, tokenize=False):
        return "::".join(f"{m['role']}={m['content']}" for m in messages)


# --- resolve_target_modules ------------------------------------------------


class TestResolveTargetModules:
    def test_phi_uses_fused_qkv(self) -> None:
        assert resolve_target_modules(BaseModelFamily.PHI) == ["qkv_proj", "o_proj"]

    def test_qwen_uses_split_projections(self) -> None:
        assert resolve_target_modules(BaseModelFamily.QWEN) == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]

    def test_string_family_accepted(self) -> None:
        assert resolve_target_modules("phi") == ["qkv_proj", "o_proj"]

    def test_default_config_is_overridden_per_family(self) -> None:
        # The generic default must be replaced by the family-specific set.
        assert resolve_target_modules(BaseModelFamily.PHI, ["q_proj", "v_proj"]) == [
            "qkv_proj",
            "o_proj",
        ]

    def test_explicit_custom_modules_honored(self) -> None:
        custom = ["q_proj", "k_proj", "gate_proj"]
        assert resolve_target_modules(BaseModelFamily.PHI, custom) == custom


# --- format_chat_text ------------------------------------------------------


class TestFormatChatText:
    def test_uses_chat_template_when_available(self) -> None:
        rec = {"instruction": "Q?", "input": "", "output": "A."}
        text = format_chat_text(rec, _FakeTokenizer())
        assert text == "user=Q?::assistant=A."

    def test_includes_input_when_present(self) -> None:
        rec = {"instruction": "Q?", "input": "ctx", "output": "A."}
        text = format_chat_text(rec, _FakeTokenizer())
        assert "Q?\n\nctx" in text

    def test_fallback_without_template(self) -> None:
        class NoTemplate:
            chat_template = None

        rec = {"instruction": "Q?", "input": "", "output": "A."}
        text = format_chat_text(rec, NoTemplate())
        assert "### Instruction:" in text and "### Response:" in text


# --- config kwargs ---------------------------------------------------------


class TestConfigKwargs:
    def test_lora_kwargs(self, tmp_path: Path) -> None:
        kwargs = lora_config_kwargs(_make_config(tmp_path, family="qwen"))
        assert kwargs["r"] == 16
        assert kwargs["lora_alpha"] == 32
        assert kwargs["task_type"] == "CAUSAL_LM"
        assert kwargs["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]

    def test_sft_kwargs(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        kwargs = sft_config_kwargs(config)
        assert kwargs["num_train_epochs"] == 2
        assert kwargs["per_device_train_batch_size"] == 4
        assert kwargs["dataset_text_field"] == "text"
        assert kwargs["report_to"] == "none"
        assert kwargs["output_dir"] == str(config.output_dir)


# --- _BridgeLogic ----------------------------------------------------------


class TestBridgeLogic:
    def test_handle_log_records_step(self, tmp_path: Path) -> None:
        cb = _RecordingCallback()
        logic = _BridgeLogic(cb, _make_config(tmp_path))
        state = types.SimpleNamespace(global_step=3, epoch=1.0)
        logic.handle_log({"loss": 1.5, "learning_rate": 1e-4}, state)
        assert len(logic.metrics_history) == 1
        assert logic.metrics_history[0].train_loss == 1.5
        assert logic.metrics_history[0].step == 3
        assert len(cb.steps) == 1

    def test_handle_log_ignores_logs_without_loss(self, tmp_path: Path) -> None:
        cb = _RecordingCallback()
        logic = _BridgeLogic(cb, _make_config(tmp_path))
        logic.handle_log({"eval_runtime": 0.1}, types.SimpleNamespace(global_step=1, epoch=1.0))
        assert logic.metrics_history == []
        assert cb.steps == []

    def test_handle_log_captures_eval_loss(self, tmp_path: Path) -> None:
        logic = _BridgeLogic(None, _make_config(tmp_path))
        logic.handle_log(
            {"loss": 1.0, "eval_loss": 1.3}, types.SimpleNamespace(global_step=1, epoch=1.0)
        )
        assert logic.metrics_history[0].val_loss == 1.3

    def test_handle_epoch_end_notifies(self, tmp_path: Path) -> None:
        cb = _RecordingCallback()
        logic = _BridgeLogic(cb, _make_config(tmp_path))
        logic.handle_log({"loss": 1.0}, types.SimpleNamespace(global_step=1, epoch=1.0))
        logic.handle_epoch_end(types.SimpleNamespace(epoch=1.0))
        assert cb.epochs == [1]


# --- DryRunBackend ---------------------------------------------------------


class TestDryRunBackend:
    def test_protocol_conformance(self) -> None:
        assert isinstance(DryRunBackend(), TrainingBackend)
        assert isinstance(RealLoRABackend(), TrainingBackend)

    def test_execute_completes_with_metrics(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        cb = _RecordingCallback()
        train = [{"instruction": f"q{i}", "input": "", "output": "a"} for i in range(5)]
        val = [{"instruction": "qv", "input": "", "output": "a"}]
        result = DryRunBackend().execute(
            train_data=train, val_data=val, config=config, args={}, callback=cb
        )
        assert result.status == TrainingStatus.COMPLETED
        assert result.total_examples == 6
        assert result.training_examples == 5
        assert result.validation_examples == 1
        assert (config.output_dir / "adapter").exists()
        # decreasing loss
        assert result.metrics_history[0].train_loss > result.metrics_history[-1].train_loss
        # callbacks: 2 steps/epoch * 2 epochs = 4 steps, 2 epoch ends, 1 complete
        assert len(cb.steps) == 4
        assert cb.epochs == [1, 2]
        assert cb.completed == 1


# --- backend selection -----------------------------------------------------


class TestBackendSelection:
    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(bc.FOUNDRY_TRAINING_BACKEND, raising=False)

    def test_default_is_dryrun(self) -> None:
        assert isinstance(_select_training_backend(), DryRunBackend)

    def test_real_selected_by_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FOUNDRY_TRAINING_BACKEND", "real")
        assert isinstance(_select_training_backend(), RealLoRABackend)
