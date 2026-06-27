"""GPU validation tests for the real transformers inference backend.

Skipped unless ``KILN_RUN_GPU_TESTS=1`` (see the root ``conftest.py``). These
load a small real model on the local GPU and exercise actual generation, so
they require CUDA, network access (first run downloads the model), and the
``[inference]`` extra installed.
"""

from __future__ import annotations

import pytest

from foundry.src import backend_config as bc
from foundry.src.evaluation import ModelInference


@pytest.fixture(scope="module")
def real_model():
    """Load the small validation model once per module, then free VRAM."""
    from foundry.src.inference_backends import TransformersInference

    model = TransformersInference(bc.DEFAULT_VALIDATION_MODEL)
    yield model
    model.close()


@pytest.mark.gpu
def test_cuda_available() -> None:
    """The GPU path requires a working CUDA device."""
    import torch

    assert torch.cuda.is_available()


@pytest.mark.gpu
def test_real_model_conforms_to_protocol(real_model) -> None:
    assert isinstance(real_model, ModelInference)


@pytest.mark.gpu
def test_generate_returns_nonempty_text(real_model) -> None:
    out = real_model.generate("What is 2 + 2? Reply with just the number.", max_tokens=16)
    assert isinstance(out, str)
    assert out.strip() != ""


@pytest.mark.gpu
def test_generate_respects_max_tokens(real_model) -> None:
    short = real_model.generate("List the planets of the solar system.", max_tokens=4)
    assert isinstance(short, str)


@pytest.mark.gpu
def test_vram_within_budget(real_model) -> None:
    """A 0.5B bf16 model should sit well under the 16 GB budget."""
    import torch

    real_model.generate("Hello.", max_tokens=8)
    allocated_gb = torch.cuda.memory_allocated() / (1024**3)
    assert allocated_gb < 4.0, f"VRAM {allocated_gb:.2f} GB exceeds bf16 budget"


@pytest.mark.gpu
def test_factory_builds_real_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The factory wires a real backend end-to-end from env config."""
    from foundry.src.inference_factory import build_inference

    monkeypatch.setenv("KILN_INFERENCE_BACKEND", "transformers")
    monkeypatch.setenv("KILN_BASE_MODEL", bc.DEFAULT_VALIDATION_MODEL)
    model = build_inference()
    try:
        out = model.generate("Say hello in one word.", max_tokens=8)
        assert isinstance(out, str) and out.strip() != ""
    finally:
        if hasattr(model, "close"):
            model.close()
