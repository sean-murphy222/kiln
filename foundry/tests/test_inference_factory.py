"""Tests for foundry.src.inference_factory.build_inference.

Verifies the default is always MockInference (so the suite stays green) and
that the real ``transformers`` backend is lazily wired without loading weights.
"""

from __future__ import annotations

import pytest

from foundry.src import backend_config as bc
from foundry.src import inference_backends
from foundry.src.evaluation import MockInference
from foundry.src.inference_factory import InferenceConfigError, build_inference


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in bc.ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class _StubBackend:
    """Stand-in for TransformersInference that records constructor kwargs."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        return "stub"


def test_default_returns_mock() -> None:
    assert isinstance(build_inference(), MockInference)


def test_mock_default_response_used() -> None:
    model = build_inference(default_response="nope")
    assert model.generate("an unknown prompt") == "nope"


def test_explicit_mock_backend() -> None:
    assert isinstance(build_inference(backend="mock"), MockInference)


def test_transformers_lazy_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inference_backends, "TransformersInference", _StubBackend)
    model = build_inference(base_model="Qwen/Qwen2.5-0.5B-Instruct", backend="transformers")
    assert isinstance(model, _StubBackend)
    assert model.kwargs["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_transformers_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KILN_INFERENCE_BACKEND", "transformers")
    monkeypatch.setenv("KILN_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    monkeypatch.setenv("KILN_ADAPTER_PATH", "/adapters/maint")
    monkeypatch.setattr(inference_backends, "TransformersInference", _StubBackend)
    model = build_inference()
    assert isinstance(model, _StubBackend)
    assert model.kwargs["adapter_path"] == "/adapters/maint"


def test_transformers_requires_base_model() -> None:
    with pytest.raises(InferenceConfigError):
        build_inference(backend="transformers")


def test_unknown_backend_raises() -> None:
    with pytest.raises(InferenceConfigError):
        build_inference(backend="bogus")
