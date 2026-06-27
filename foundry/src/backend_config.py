"""Environment-driven backend selection for Foundry/Hearth model execution.

This is the single source of truth for choosing between the default mock /
dry-run backends and the real transformers + peft backends. Every value is
read from an environment variable, so flipping to a real backend is a pure
configuration change with no code edits.

The module imports nothing heavy at import time (only the standard library; the
optional Hugging Face token check imports ``huggingface_hub`` lazily), so it is
safe to import anywhere — including the degraded-mode server mount where
torch/peft may be absent. The defaults keep the mock inference / dry-run training
path active, so the entire existing test suite passes untouched.

Model policy: American models only. The default base model is gated Llama-3.2-3B
(needs a Hugging Face token); when no token is available it falls back
automatically to the ungated, MIT-licensed Phi-3.5-mini.

Environment variables:
    KILN_INFERENCE_BACKEND: ``mock`` (default) or ``transformers``.
    FOUNDRY_TRAINING_BACKEND: ``dryrun`` (default) or ``real``.
    KILN_BASE_MODEL: HuggingFace base model id (default: Llama-3.2-3B with a
        token, else Phi-3.5-mini).
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

# American models only. Llama is gated (needs a Hugging Face token); Phi-3.5-mini
# is MIT-licensed and ungated, used as the automatic fallback when no token.
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
FALLBACK_BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
DEFAULT_VALIDATION_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
FALLBACK_VALIDATION_MODEL = "microsoft/Phi-3.5-mini-instruct"

_TRUTHY = {"1", "true", "yes", "on"}

# Keyword -> family, checked in order against a lowercased model id.
_FAMILY_KEYWORDS = (
    ("phi", "phi"),
    ("llama", "llama"),
    ("mistral", "mistral"),
    ("qwen", "qwen"),
)


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


def has_hf_token() -> bool:
    """Return True if a Hugging Face access token is available.

    Checks the cached CLI login and the standard env vars (via
    ``huggingface_hub.get_token`` when importable, else env directly). Used to
    decide whether the gated default (Llama) is reachable or the ungated
    fallback (Phi) must be used.
    """
    try:
        from huggingface_hub import get_token

        if get_token():
            return True
    except Exception:  # pragma: no cover - huggingface_hub is present in this env
        pass
    return any(
        os.environ.get(name)
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
    )


def get_base_model() -> str:
    """Return the base model id for real inference (American models only).

    Resolution order: ``KILN_BASE_MODEL`` if set; otherwise the gated default
    Llama-3.2-3B when a Hugging Face token is available, else the ungated
    Phi-3.5-mini fallback.
    """
    explicit = _get(KILN_BASE_MODEL)
    if explicit:
        return explicit
    return DEFAULT_BASE_MODEL if has_hf_token() else FALLBACK_BASE_MODEL


def model_family_for(model_id: str) -> str:
    """Infer the LoRA model family from a model id.

    Args:
        model_id: HuggingFace model id.

    Returns:
        One of ``llama``/``phi``/``mistral``/``qwen``; defaults to ``llama`` for
        an unrecognized American instruct model.
    """
    lowered = model_id.lower()
    for keyword, family in _FAMILY_KEYWORDS:
        if keyword in lowered:
            return family
    return "llama"


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
    """Return the small model id used for GPU validation tests.

    ``KILN_VALIDATION_MODEL`` if set; otherwise the gated default Llama-3.2-1B
    when a Hugging Face token is available, else the ungated Phi-3.5-mini.
    """
    explicit = _get(KILN_VALIDATION_MODEL)
    if explicit:
        return explicit
    return DEFAULT_VALIDATION_MODEL if has_hf_token() else FALLBACK_VALIDATION_MODEL


def run_gpu_tests() -> bool:
    """Return whether GPU-marked tests should run (``KILN_RUN_GPU_TESTS=1``)."""
    return _bool(KILN_RUN_GPU_TESTS)
