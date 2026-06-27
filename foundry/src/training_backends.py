"""Pluggable training backends for the Foundry LoRA pipeline.

The training execution step sits behind a :class:`TrainingBackend` Protocol so
that:

    * :class:`DryRunBackend` (the default) simulates training with decreasing
      loss — fast, deterministic, no GPU — keeping the suite green, and
    * :class:`RealLoRABackend` performs a genuine ``peft`` + ``trl`` LoRA
      fine-tune that writes a loadable adapter, selected only when
      ``FOUNDRY_TRAINING_BACKEND=real``.

All heavy imports (torch/transformers/peft/trl/datasets) are lazy — they happen
inside ``RealLoRABackend.execute`` — so importing this module never requires
those packages, preserving the dependency-free default path. The dataclasses
and Protocol come from :mod:`foundry.src.training`; importing them here is safe
because ``training`` does not import this module at load time (it resolves the
default backend lazily), so there is no import cycle.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from foundry.src.training import (
    BaseModelFamily,
    TrainingConfig,
    TrainingMetrics,
    TrainingProgressCallback,
    TrainingResult,
    TrainingStatus,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Protocol
# ===================================================================


@runtime_checkable
class TrainingBackend(Protocol):
    """Executes the training step and returns a TrainingResult."""

    def execute(
        self,
        *,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
        config: TrainingConfig,
        args: dict[str, Any],
        callback: TrainingProgressCallback | None,
    ) -> TrainingResult:
        """Run training and return its result.

        Args:
            train_data: Training split records (Alpaca format).
            val_data: Validation split records.
            config: The training configuration.
            args: Pre-built HuggingFace-style training arguments.
            callback: Optional progress callback.

        Returns:
            A TrainingResult (status COMPLETED or FAILED).
        """
        ...


# ===================================================================
# Per-family LoRA target modules
# ===================================================================

_DEFAULT_TARGET_MODULES = ["q_proj", "v_proj"]

_TARGET_MODULES_BY_FAMILY: dict[BaseModelFamily, list[str]] = {
    # Phi-3 fuses QKV into a single projection; q_proj/v_proj attach nothing.
    BaseModelFamily.PHI: ["qkv_proj", "o_proj"],
    BaseModelFamily.LLAMA: ["q_proj", "k_proj", "v_proj", "o_proj"],
    BaseModelFamily.MISTRAL: ["q_proj", "k_proj", "v_proj", "o_proj"],
    BaseModelFamily.QWEN: ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def resolve_target_modules(
    family: BaseModelFamily | str,
    configured: list[str] | None = None,
) -> list[str]:
    """Resolve LoRA target modules for a model family.

    Honors an explicit, non-default ``configured`` list; otherwise picks the
    correct attention projections for the family. This fixes the generic
    ``["q_proj", "v_proj"]`` default, which attaches nothing on Phi-3.

    Args:
        family: Model family (enum or string).
        configured: The configured target modules from ``LoRAConfig``.

    Returns:
        The list of module names LoRA should adapt.
    """
    if configured and list(configured) != _DEFAULT_TARGET_MODULES:
        return list(configured)
    fam = family if isinstance(family, BaseModelFamily) else BaseModelFamily(family)
    return list(_TARGET_MODULES_BY_FAMILY.get(fam, _DEFAULT_TARGET_MODULES))


def format_chat_text(record: dict[str, Any], tokenizer: Any) -> str:
    """Render an Alpaca record into a single SFT training string.

    Uses the tokenizer chat template when available so prompt formatting
    matches inference; falls back to a plain instruction/response layout.

    Args:
        record: Alpaca-format record with ``instruction``/``input``/``output``.
        tokenizer: A tokenizer exposing ``apply_chat_template`` (optional).

    Returns:
        The formatted training text.
    """
    instruction = record.get("instruction", "")
    user_input = record.get("input", "")
    output = record.get("output", "")
    user_content = instruction if not user_input else f"{instruction}\n\n{user_input}"
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]
    if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return f"### Instruction:\n{user_content}\n\n### Response:\n{output}"


def lora_config_kwargs(config: TrainingConfig) -> dict[str, Any]:
    """Build the keyword args for ``peft.LoraConfig`` from a TrainingConfig."""
    return {
        "r": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "target_modules": resolve_target_modules(
            config.base_model_family, config.lora.target_modules
        ),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def sft_config_kwargs(config: TrainingConfig) -> dict[str, Any]:
    """Build the keyword args for ``trl.SFTConfig`` from a TrainingConfig."""
    return {
        "output_dir": str(config.output_dir),
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_length": config.max_seq_length,
        "seed": config.seed,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "dataset_text_field": "text",
    }


# ===================================================================
# Callback bridge (HF TrainerCallback events -> TrainingProgressCallback)
# ===================================================================


class _BridgeLogic:
    """Maps HuggingFace trainer events to a TrainingProgressCallback.

    Kept free of any ``transformers`` import so the mapping is unit-testable
    without the trainer. The thin ``TrainerCallback`` subclass that actually
    receives events (constructed inside ``RealLoRABackend.execute``) delegates
    here.
    """

    def __init__(
        self,
        callback: TrainingProgressCallback | None,
        config: TrainingConfig,
    ) -> None:
        self.callback = callback
        self.config = config
        self.metrics_history: list[TrainingMetrics] = []

    def handle_log(self, logs: dict[str, Any], state: Any) -> None:
        """Convert a training ``on_log`` event into a step metric.

        Args:
            logs: The HF log dict (may contain ``loss``/``learning_rate``).
            state: The HF ``TrainerState`` (provides ``global_step``/``epoch``).
        """
        if "loss" not in logs:
            return  # eval-only or housekeeping logs carry no train loss
        metrics = TrainingMetrics(
            epoch=int(getattr(state, "epoch", 0) or 0),
            step=int(getattr(state, "global_step", len(self.metrics_history) + 1)),
            train_loss=float(logs["loss"]),
            val_loss=(float(logs["eval_loss"]) if "eval_loss" in logs else None),
            learning_rate=float(logs.get("learning_rate", self.config.learning_rate)),
            timestamp=datetime.now(),
        )
        self.metrics_history.append(metrics)
        if self.callback is not None:
            self.callback.on_step(metrics)

    def handle_epoch_end(self, state: Any) -> None:
        """Convert an ``on_epoch_end`` event into a callback notification.

        Args:
            state: The HF ``TrainerState`` (provides ``epoch``).
        """
        if self.callback is not None and self.metrics_history:
            epoch = int(getattr(state, "epoch", len(self.metrics_history)) or 0)
            self.callback.on_epoch_end(epoch, self.metrics_history[-1])


# ===================================================================
# DryRunBackend (default)
# ===================================================================


class DryRunBackend:
    """Simulate training with monotonically decreasing loss (no GPU).

    Reproduces the original in-pipeline simulator byte-for-byte so existing
    training tests pass unchanged.
    """

    def execute(
        self,
        *,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
        config: TrainingConfig,
        args: dict[str, Any],
        callback: TrainingProgressCallback | None,
    ) -> TrainingResult:
        """Produce a completed TrainingResult with simulated metrics."""
        started_at = datetime.now()
        adapter_path = Path(config.output_dir) / "adapter"
        adapter_path.mkdir(parents=True, exist_ok=True)
        metrics_history = self._simulate_metrics(train_data, val_data, config, callback)
        completed_at = datetime.now()
        result = TrainingResult(
            config=config,
            status=TrainingStatus.COMPLETED,
            adapter_path=adapter_path,
            metrics_history=metrics_history,
            total_examples=len(train_data) + len(val_data),
            training_examples=len(train_data),
            validation_examples=len(val_data),
            started_at=started_at,
            completed_at=completed_at,
        )
        if callback is not None:
            callback.on_complete(result)
        return result

    def _simulate_metrics(
        self,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
        config: TrainingConfig,
        callback: TrainingProgressCallback | None,
    ) -> list[TrainingMetrics]:
        """Generate decreasing-loss metrics across epochs and steps."""
        metrics_history: list[TrainingMetrics] = []
        steps_per_epoch = max(1, math.ceil(len(train_data) / config.batch_size))
        global_step = 0
        for epoch in range(1, config.epochs + 1):
            for _step in range(steps_per_epoch):
                global_step += 1
                metrics = self._make_step_metrics(epoch, global_step, val_data, config)
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
        config: TrainingConfig,
    ) -> TrainingMetrics:
        """Create a single simulated metrics entry."""
        base_loss = 2.0 / (1.0 + global_step * 0.1)
        val_loss = base_loss * 1.1 if val_data else None
        return TrainingMetrics(
            epoch=epoch,
            step=global_step,
            train_loss=base_loss,
            val_loss=val_loss,
            learning_rate=config.learning_rate,
            timestamp=datetime.now(),
        )


# ===================================================================
# RealLoRABackend (opt-in)
# ===================================================================


class RealLoRABackend:
    """Run a genuine LoRA fine-tune with ``peft`` + ``trl`` + ``datasets``.

    Loads the base model in bf16, attaches a LoRA adapter, trains with
    ``SFTTrainer`` on the chat-formatted curriculum, and saves a loadable PEFT
    adapter to ``<output_dir>/adapter``. Heavy imports are deferred to
    ``execute`` so this module stays import-light. 4-bit/QLoRA training is out
    of scope for MVP; training uses bf16.
    """

    def execute(
        self,
        *,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
        config: TrainingConfig,
        args: dict[str, Any],
        callback: TrainingProgressCallback | None,
    ) -> TrainingResult:
        """Run a real LoRA fine-tune; return COMPLETED or FAILED result."""
        started_at = datetime.now()
        adapter_path = Path(config.output_dir) / "adapter"
        total = len(train_data) + len(val_data)
        try:
            import torch
            from datasets import Dataset
            from peft import LoraConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
            from trl import SFTConfig, SFTTrainer

            tokenizer = AutoTokenizer.from_pretrained(config.base_model)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

            train_ds = Dataset.from_dict(
                {"text": [format_chat_text(r, tokenizer) for r in train_data]}
            )
            eval_ds = (
                Dataset.from_dict({"text": [format_chat_text(r, tokenizer) for r in val_data]})
                if val_data
                else None
            )

            lora_config = LoraConfig(**lora_config_kwargs(config))
            sft_config = SFTConfig(**sft_config_kwargs(config))
            model = AutoModelForCausalLM.from_pretrained(config.base_model, dtype=torch.bfloat16)

            logic = _BridgeLogic(callback, config)

            class _HFCallbackBridge(TrainerCallback):
                def on_log(self, a, s, c, logs=None, **kw):  # noqa: ANN001, ANN204
                    logic.handle_log(logs or {}, s)

                def on_epoch_end(self, a, s, c, **kw):  # noqa: ANN001, ANN204
                    logic.handle_epoch_end(s)

            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                peft_config=lora_config,
                processing_class=tokenizer,
                callbacks=[_HFCallbackBridge()],
            )
            trainer.train()

            adapter_path.mkdir(parents=True, exist_ok=True)
            trainer.model.save_pretrained(str(adapter_path))
            tokenizer.save_pretrained(str(adapter_path))

            result = TrainingResult(
                config=config,
                status=TrainingStatus.COMPLETED,
                adapter_path=adapter_path,
                metrics_history=logic.metrics_history,
                total_examples=total,
                training_examples=len(train_data),
                validation_examples=len(val_data),
                started_at=started_at,
                completed_at=datetime.now(),
            )
            if callback is not None:
                callback.on_complete(result)
            return result
        except Exception as exc:  # noqa: BLE001 - surface any training failure
            logger.exception("Real LoRA training failed")
            if callback is not None:
                callback.on_error(str(exc))
            return TrainingResult(
                config=config,
                status=TrainingStatus.FAILED,
                adapter_path=None,
                metrics_history=[],
                total_examples=total,
                training_examples=len(train_data),
                validation_examples=len(val_data),
                started_at=started_at,
                completed_at=datetime.now(),
                error_message=str(exc),
            )
