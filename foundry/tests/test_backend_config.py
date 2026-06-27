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

    def test_base_model_defaults_none(self) -> None:
        assert bc.get_base_model() is None

    def test_adapter_path_defaults_none(self) -> None:
        assert bc.get_adapter_path() is None

    def test_validation_model_default(self) -> None:
        assert bc.get_validation_model() == "Qwen/Qwen2.5-0.5B-Instruct"

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
        monkeypatch.setenv("KILN_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        assert bc.get_base_model() == "Qwen/Qwen2.5-0.5B-Instruct"

    def test_blank_value_treated_as_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KILN_BASE_MODEL", "   ")
        assert bc.get_base_model() is None

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
