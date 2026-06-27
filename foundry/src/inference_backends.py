"""Real model-inference backends for Foundry/Hearth.

This module provides :class:`TransformersInference`, a concrete implementation
of the :class:`foundry.src.evaluation.ModelInference` Protocol backed by
HuggingFace ``transformers`` + ``peft``. It is the single real inference
backend reused by Hearth, Foundry evaluation, and Foundry RAG.

Design constraints:
    * All heavy imports (``torch``, ``transformers``, ``peft``,
      ``bitsandbytes``) happen lazily inside ``__init__``/``generate``/``close``
      so importing this module never requires those packages. This keeps the
      degraded-mode server mount and the dependency-free test run working.
    * ``bfloat16`` is the default; 4-bit quantization is opt-in (it may be
      unavailable on some platforms, e.g. Blackwell/sm_120 + Windows).
    * The class is structurally compatible with the ModelInference Protocol
      (it exposes ``generate(prompt, max_tokens)``), so it can be injected
      anywhere a mock is used today.
"""

from __future__ import annotations

from typing import Any


class TransformersInference:
    """Generate text with a HuggingFace causal LM and an optional LoRA adapter.

    Implements the :class:`foundry.src.evaluation.ModelInference` Protocol.

    Args:
        base_model: HuggingFace model id (or local path) for the base weights.
        adapter_path: Optional path to a trained PEFT LoRA adapter to attach.
        load_4bit: Load the base model in 4-bit (requires ``bitsandbytes``).
            Defaults to ``False`` (bf16/dtype path).
        dtype: Torch dtype name used for bf16/fp16 loading and 4-bit compute.
        device_map: ``transformers`` device placement strategy.
        max_new_tokens: Default generation length when a caller does not pass
            an explicit value.
        chat: Apply the tokenizer chat template when available.

    Example::

        model = TransformersInference("Qwen/Qwen2.5-0.5B-Instruct")
        answer = model.generate("What is torque?")
    """

    def __init__(
        self,
        base_model: str,
        adapter_path: str | None = None,
        *,
        load_4bit: bool = False,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 512,
        chat: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.base_model = base_model
        self.adapter_path = adapter_path
        self.load_4bit = load_4bit
        self.dtype_name = dtype
        self.max_new_tokens = max_new_tokens
        self.chat = chat

        torch_dtype = getattr(torch, dtype, torch.bfloat16)

        model_kwargs: dict[str, Any] = {"device_map": device_map}
        if load_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["dtype"] = torch_dtype

        self._tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)

        model.eval()
        self._model = model

        # Ensure a pad token exists so batched generation does not crash.
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def _build_prompt(self, prompt: str) -> str:
        """Render the user prompt, applying the chat template when available.

        Args:
            prompt: Raw user prompt text.

        Returns:
            The string to tokenize and feed to the model.
        """
        template = getattr(self._tokenizer, "chat_template", None)
        if self.chat and template and hasattr(self._tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        """Generate a completion for ``prompt``.

        Args:
            prompt: Input prompt text.
            max_tokens: Optional max new tokens; falls back to the configured
                ``max_new_tokens`` when ``None`` or non-positive.

        Returns:
            The generated text with the prompt tokens stripped.
        """
        import torch

        n_new = max_tokens if (max_tokens and max_tokens > 0) else self.max_new_tokens
        text = self._build_prompt(prompt)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=n_new,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated = output_ids[0][prompt_len:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def close(self) -> None:
        """Release the model/tokenizer and free CUDA memory if possible."""
        self._model = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass
