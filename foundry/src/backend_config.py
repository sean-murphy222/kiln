"""Environment-driven backend selection for Foundry/Hearth model execution.

This is the single source of truth for choosing between the default mock /
dry-run backends and the real transformers + peft backends. Every value is
read from an environment variable, so flipping to a real backend is a pure
configuration change with no code edits.

The module imports nothing heavy (only the standard library), so it is safe to
import anywhere — including the degraded-mode server mount where torch/peft may
be absent. The defaults keep the mock inference / dry-run training path active,
so the entire existing test suite passes untouched.

Environment variables:
    KILN_INFERENCE_BACKEND: ``mock`` (default) or ``transformers``.
    FOUNDRY_TRAINING_BACKEND: ``dryrun`` (default) or ``real``.
    KILN_BASE_MODEL: HuggingFace model id for the real inference backend.
    KILN_ADAPTER_PATH: Path to a trained LoRA adapter to attach at inference.
    KILN_LOAD_4BIT: ``1`` to enable 4-bit quantization (default off -> bf16).
    KILN_INFERENCE_DTYPE: Torch dtype name (default ``bfloat16``).
    KILN_MAX_NEW_TOKENS: Generation length cap (default 512).
    KILN_VALIDATION_MODEL: Small model id used by GPU validation tests.
    KILN_RUN_GPU_TESTS: ``1`` to run GPU-marked tests.
"""

from __future__ import annotations

import os

# --- Environment variable names (exported so tests can clear them) ----------
KILN_INFERENCE_BACKEND = "KILN_INFERENCE_BACKEND"
FOUNDRY_TRAINING_BACKEND = "FOUNDRY_TRAINING_BACKEND"
KILN_BASE_MODEL = "KILN_BASE_MODEL"
KILN_ADAPTER_PATH = "KILN_ADAPTER_PATH"
KILN_LOAD_4BIT = "KILN_LOAD_4BIT"
KILN_INFERENCE_DTYPE = "KILN_INFERENCE_DTYPE"
KILN_MAX_NEW_TOKENS = "KILN_MAX_NEW_TOKENS"
KILN_VALIDATION_MODEL = "KILN_VALIDATION_MODEL"
KILN_RUN_GPU_TESTS = "KILN_RUN_GPU_TESTS"

ENV_VARS = (
    KILN_INFERENCE_BACKEND,
    FOUNDRY_TRAINING_BACKEND,
    KILN_BASE_MODEL,
    KILN_ADAPTER_PATH,
    KILN_LOAD_4BIT,
    KILN_INFERENCE_DTYPE,
    KILN_MAX_NEW_TOKENS,
    KILN_VALIDATION_MODEL,
    KILN_RUN_GPU_TESTS,
)

# --- Defaults (keep the mock / dry-run path active) -------------------------
DEFAULT_INFERENCE_BACKEND = "mock"
DEFAULT_TRAINING_BACKEND = "dryrun"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_VALIDATION_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

_TRUTHY = {"1", "true", "yes", "on"}


def _get(name: str) -> str | None:
    """Return a stripped environment value, or ``None`` if unset/blank.

    Args:
        name: Environment variable name.

    Returns:
        The trimmed value, or ``None`` when the variable is absent or empty.
    """
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def _bool(name: str) -> bool:
    """Return ``True`` when the environment value is a truthy token.

    Args:
        name: Environment variable name.

    Returns:
        ``True`` for ``1``/``true``/``yes``/``on`` (case-insensitive), else
        ``False``.
    """
    raw = _get(name)
    return raw is not None and raw.lower() in _TRUTHY


def get_inference_backend() -> str:
    """Return the selected inference backend.

    Returns:
        ``mock`` (default) or another lowercased identifier such as
        ``transformers``.
    """
    return (_get(KILN_INFERENCE_BACKEND) or DEFAULT_INFERENCE_BACKEND).lower()


def get_training_backend() -> str:
    """Return the selected training backend.

    Returns:
        ``dryrun`` (default) or ``real`` (lowercased).
    """
    return (_get(FOUNDRY_TRAINING_BACKEND) or DEFAULT_TRAINING_BACKEND).lower()


def get_base_model() -> str | None:
    """Return the configured base model id, or ``None`` if unset."""
    return _get(KILN_BASE_MODEL)


def get_adapter_path() -> str | None:
    """Return the configured LoRA adapter path, or ``None`` if unset."""
    return _get(KILN_ADAPTER_PATH)


def load_4bit() -> bool:
    """Return whether 4-bit quantization is requested (default ``False``)."""
    return _bool(KILN_LOAD_4BIT)


def get_dtype() -> str:
    """Return the inference torch dtype name (default ``bfloat16``)."""
    return _get(KILN_INFERENCE_DTYPE) or DEFAULT_DTYPE


def get_max_new_tokens() -> int:
    """Return the generation length cap.

    Returns:
        The configured positive integer, or 512 when unset or invalid.
    """
    raw = _get(KILN_MAX_NEW_TOKENS)
    if raw is None:
        return DEFAULT_MAX_NEW_TOKENS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_NEW_TOKENS
    return value if value > 0 else DEFAULT_MAX_NEW_TOKENS


def get_validation_model() -> str:
    """Return the small model id used for GPU validation tests."""
    return _get(KILN_VALIDATION_MODEL) or DEFAULT_VALIDATION_MODEL


def run_gpu_tests() -> bool:
    """Return whether GPU-marked tests should run (``KILN_RUN_GPU_TESTS=1``)."""
    return _bool(KILN_RUN_GPU_TESTS)
