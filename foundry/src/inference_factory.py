"""Factory that selects the inference backend (mock vs. real).

``build_inference`` is the single decision point for which
:class:`foundry.src.evaluation.ModelInference` implementation is used. It is
called by every production construction site so that:

    * the default is always :class:`~foundry.src.evaluation.MockInference`
      (keeping the test suite and degraded-mode mount green), and
    * switching to the real ``transformers`` backend is a pure environment
      change with no code edits.

The real backend is imported lazily, only when selected, so importing this
module never requires torch/transformers/peft.
"""

from __future__ import annotations

from foundry.src import backend_config
from foundry.src.evaluation import MockInference, ModelInference


class InferenceConfigError(RuntimeError):
    """Raised when a real inference backend is requested but misconfigured."""


def build_inference(
    base_model: str | None = None,
    adapter_path: str | None = None,
    *,
    backend: str | None = None,
    default_response: str = "I don't know.",
    load_4bit: bool | None = None,
    dtype: str | None = None,
    max_new_tokens: int | None = None,
) -> ModelInference:
    """Construct a :class:`ModelInference` per configuration.

    Args:
        base_model: Base model id for the real backend. Falls back to
            ``KILN_BASE_MODEL``.
        adapter_path: LoRA adapter path for the real backend. Falls back to
            ``KILN_ADAPTER_PATH``.
        backend: Explicit backend name (``mock``/``transformers``). Falls back
            to ``KILN_INFERENCE_BACKEND`` (default ``mock``).
        default_response: Response used by the mock backend for unknown prompts.
        load_4bit: Override for 4-bit loading; falls back to ``KILN_LOAD_4BIT``.
        dtype: Override for the torch dtype; falls back to ``KILN_INFERENCE_DTYPE``.
        max_new_tokens: Override for generation length; falls back to
            ``KILN_MAX_NEW_TOKENS``.

    Returns:
        A mock or real backend implementing the ModelInference Protocol.

    Raises:
        InferenceConfigError: If ``transformers`` is selected without a base
            model, or an unknown backend name is given.
    """
    resolved = (backend or backend_config.get_inference_backend()).lower()

    if resolved == "mock":
        return MockInference(default_response=default_response)

    if resolved == "transformers":
        model_id = base_model or backend_config.get_base_model()
        if not model_id:
            raise InferenceConfigError(
                "Inference backend 'transformers' requires a base model: set "
                "KILN_BASE_MODEL or pass base_model=."
            )
        # Lazy import: only pull the heavy backend when actually selected.
        from foundry.src.inference_backends import TransformersInference

        return TransformersInference(
            base_model=model_id,
            adapter_path=adapter_path or backend_config.get_adapter_path(),
            load_4bit=backend_config.load_4bit() if load_4bit is None else load_4bit,
            dtype=dtype or backend_config.get_dtype(),
            max_new_tokens=max_new_tokens or backend_config.get_max_new_tokens(),
        )

    raise InferenceConfigError(f"Unknown inference backend: {resolved!r}")
