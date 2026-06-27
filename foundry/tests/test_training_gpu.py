"""GPU validation for the real LoRA training backend (peft + trl).

Skipped unless ``KILN_RUN_GPU_TESTS=1`` (see the root ``conftest.py``). Runs a
genuine micro LoRA fine-tune of the small validation model on a tiny
curriculum, verifies a loadable PEFT adapter is produced, then loads it through
TransformersInference and generates. Requires CUDA and the
``[inference,training]`` extras.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundry.src import backend_config as bc
from foundry.src.training import TrainingConfig, TrainingPipeline, TrainingStatus
from foundry.src.training_backends import RealLoRABackend


@pytest.fixture
def tiny_curriculum(tmp_path: Path) -> Path:
    """Write a small Alpaca curriculum for a fast training run."""
    records = [
        {
            "instruction": f"What is maintenance step {i}?",
            "input": "",
            "output": f"Step {i}: inspect the component, torque to spec, verify clearance.",
            "metadata": {
                "example_id": f"ex{i:03d}",
                "discipline_id": "disc_x",
                "competency_id": "comp_y",
                "review_status": "approved",
            },
        }
        for i in range(12)
    ]
    path = tmp_path / "curriculum.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


@pytest.fixture
def real_config(tiny_curriculum: Path, tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(
        base_model=bc.get_validation_model(),
        base_model_family=bc.model_family_for(bc.get_validation_model()),
        curriculum_path=tiny_curriculum,
        output_dir=tmp_path / "run",
        epochs=1,
        batch_size=2,
        gradient_accumulation_steps=1,
        max_seq_length=256,
    )


class _Recorder:
    def __init__(self) -> None:
        self.steps = 0
        self.completed = 0
        self.errors: list[str] = []

    def on_step(self, metrics) -> None:
        self.steps += 1

    def on_epoch_end(self, epoch, metrics) -> None:
        pass

    def on_complete(self, result) -> None:
        self.completed += 1

    def on_error(self, error: str) -> None:
        self.errors.append(error)


@pytest.mark.gpu
def test_real_lora_train_produces_loadable_adapter(real_config: TrainingConfig) -> None:
    recorder = _Recorder()
    pipeline = TrainingPipeline(config=real_config, backend=RealLoRABackend())
    pipeline.prepare()
    result = pipeline.run(callback=recorder)

    assert result.status == TrainingStatus.COMPLETED, result.error_message
    assert recorder.completed == 1
    assert recorder.errors == []

    adapter = result.adapter_path
    assert adapter is not None
    assert (adapter / "adapter_config.json").exists()
    assert (adapter / "adapter_model.safetensors").exists()
    assert len(result.metrics_history) > 0  # real loss logged via the callback bridge

    # The freshly trained adapter loads and generates through the inference seam.
    from foundry.src.inference_backends import TransformersInference

    model = TransformersInference(real_config.base_model, adapter_path=str(adapter))
    try:
        out = model.generate("What is maintenance step 3?", max_tokens=32)
        assert isinstance(out, str)
        assert out.strip() != ""
    finally:
        model.close()


@pytest.mark.gpu
def test_real_training_via_env_gate(
    real_config: TrainingConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With FOUNDRY_TRAINING_BACKEND=real, the pipeline selects the real backend."""
    monkeypatch.setenv("FOUNDRY_TRAINING_BACKEND", "real")
    pipeline = TrainingPipeline(config=real_config)
    assert isinstance(pipeline._backend, RealLoRABackend)
