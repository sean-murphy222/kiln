"""Tests for foundry.src.backend_config — env-driven backend selection.

These verify that the defaults keep the mock inference / dry-run training path
active (so the existing suite stays green) and that environment variables
override each setting as expected.
"""

from __future__ import annotations

import pytest

from foundry.src import backend_config as bc


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear all Kiln backend env vars before each test for isolation."""
    for var in bc.ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestDefaults:
    """With no env vars set, every helper returns a mock/dry-run-safe default."""

    def test_inference_backend_defaults_to_mock(self) -> None:
        assert bc.get_inference_backend() == "mock"

    def test_training_backend_defaults_to_dryrun(self) -> None:
        assert bc.get_training_backend() == "dryrun"

    def test_load_4bit_defaults_false(self) -> None:
        assert bc.load_4bit() is False

    def test_dtype_defaults_bfloat16(self) -> None:
        assert bc.get_dtype() == "bfloat16"

    def test_max_new_tokens_defaults_512(self) -> None:
        assert bc.get_max_new_tokens() == 512

    def test_adapter_path_defaults_none(self) -> None:
        assert bc.get_adapter_path() is None

    def test_run_gpu_tests_defaults_false(self) -> None:
        assert bc.run_gpu_tests() is False


class TestOverrides:
    """Environment variables override the defaults."""

    def test_inference_backend_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KILN_INFERENCE_BACKEND", "transformers")
        assert bc.get_inference_backend() == "transformers"

    def test_inference_backend_normalized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KILN_INFERENCE_BACKEND", "  Transformers  ")
        assert bc.get_inference_backend() == "transformers"

    def test_training_backend_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FOUNDRY_TRAINING_BACKEND", "real")
        assert bc.get_training_backend() == "real"

    def test_base_model_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KILN_BASE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
        assert bc.get_base_model() == "meta-llama/Llama-3.2-3B-Instruct"

    def test_blank_base_model_falls_through_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bc, "has_hf_token", lambda: False)
        monkeypatch.setenv("KILN_BASE_MODEL", "   ")
        assert bc.get_base_model() == bc.FALLBACK_BASE_MODEL

    def test_adapter_path_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KILN_ADAPTER_PATH", "/models/adapter")
        assert bc.get_adapter_path() == "/models/adapter"

    def test_dtype_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KILN_INFERENCE_DTYPE", "float16")
        assert bc.get_dtype() == "float16"

    def test_max_new_tokens_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KILN_MAX_NEW_TOKENS", "256")
        assert bc.get_max_new_tokens() == 256

    def test_max_new_tokens_invalid_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KILN_MAX_NEW_TOKENS", "not-a-number")
        assert bc.get_max_new_tokens() == 512

    def test_max_new_tokens_nonpositive_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KILN_MAX_NEW_TOKENS", "0")
        assert bc.get_max_new_tokens() == 512

    def test_validation_model_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KILN_VALIDATION_MODEL", "sshleifer/tiny-gpt2")
        assert bc.get_validation_model() == "sshleifer/tiny-gpt2"


class TestModelResolution:
    """American-model defaults with automatic Llama->Phi fallback by token."""

    @pytest.fixture(autouse=True)
    def _clear_model_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("KILN_BASE_MODEL", raising=False)
        monkeypatch.delenv("KILN_VALIDATION_MODEL", raising=False)

    def test_base_model_default_llama_with_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(bc, "has_hf_token", lambda: True)
        assert bc.get_base_model() == bc.DEFAULT_BASE_MODEL
        assert "llama" in bc.DEFAULT_BASE_MODEL.lower()

    def test_base_model_fallback_phi_without_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(bc, "has_hf_token", lambda: False)
        assert bc.get_base_model() == bc.FALLBACK_BASE_MODEL
        assert "phi" in bc.FALLBACK_BASE_MODEL.lower()

    def test_validation_model_with_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(bc, "has_hf_token", lambda: True)
        assert bc.get_validation_model() == bc.DEFAULT_VALIDATION_MODEL

    def test_validation_model_fallback_without_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bc, "has_hf_token", lambda: False)
        assert bc.get_validation_model() == bc.FALLBACK_VALIDATION_MODEL

    def test_no_chinese_models_in_defaults(self) -> None:
        for model_id in (
            bc.DEFAULT_BASE_MODEL,
            bc.FALLBACK_BASE_MODEL,
            bc.DEFAULT_VALIDATION_MODEL,
            bc.FALLBACK_VALIDATION_MODEL,
        ):
            assert "qwen" not in model_id.lower()

    def test_model_family_for(self) -> None:
        assert bc.model_family_for("meta-llama/Llama-3.2-3B-Instruct") == "llama"
        assert bc.model_family_for("microsoft/Phi-3.5-mini-instruct") == "phi"
        assert bc.model_family_for("mistralai/Mistral-7B-Instruct-v0.3") == "mistral"
        assert bc.model_family_for("some/unknown-model") == "llama"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True),
        ("true", True),
        ("True", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("", False),
        ("no", False),
        ("maybe", False),
    ],
)
def test_load_4bit_parsing(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
) -> None:
    """load_4bit() recognizes common truthy tokens, everything else is False."""
    monkeypatch.setenv("KILN_LOAD_4BIT", value)
    assert bc.load_4bit() is expected
