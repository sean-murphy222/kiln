"""Tests for foundry.src.inference_backends.TransformersInference (no GPU).

The tokenizer/model/peft layers are faked so we exercise the wiring (prompt
assembly, token slicing, max-new-tokens passthrough, dtype vs. 4-bit, adapter
attachment) without downloading weights or needing a GPU.
"""

from __future__ import annotations

import types

import pytest
import torch

from foundry.src.evaluation import ModelInference
from foundry.src.inference_backends import TransformersInference


class FakeEncoding(dict):
    """Dict-like batch encoding exposing a chainable no-op ``.to()``."""

    def to(self, device: object) -> FakeEncoding:
        self["_device"] = device
        return self


class FakeTokenizer:
    """Minimal tokenizer stand-in returning fixed tensors."""

    def __init__(
        self, chat_template: str | None = "TEMPLATE", pad_token_id: int | None = 0
    ) -> None:
        self.chat_template = chat_template
        self.pad_token_id = pad_token_id
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.last_text: str | None = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "CHAT::" + messages[0]["content"]

    def __call__(self, text, return_tensors="pt"):
        self.last_text = text
        return FakeEncoding(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
        )

    def decode(self, ids, skip_special_tokens=True):
        return "DEC:" + ",".join(str(int(i)) for i in ids)


class FakeModel:
    """Minimal causal-LM stand-in that echoes generate kwargs."""

    def __init__(self) -> None:
        self.device = "cpu"
        self.eval_called = False
        self.last_kwargs: dict | None = None

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        return torch.tensor([[1, 2, 3, 4, 5]])  # 3 prompt + 2 generated


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch):
    """Patch transformers Auto* ``from_pretrained`` to return the fakes.

    Patching the method on the real classes (rather than replacing the classes
    on the lazy ``transformers`` module) prevents any network/model load
    regardless of how the class is imported inside the backend.
    """
    import transformers

    tok = FakeTokenizer()
    model = FakeModel()
    captured: dict = {}

    def fake_tok_fp(name, **kw):
        captured["tok_name"] = name
        return tok

    def fake_model_fp(name, **kw):
        captured["model_name"] = name
        captured["model_kwargs"] = kw
        return model

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_tok_fp)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fake_model_fp)
    return types.SimpleNamespace(tok=tok, model=model, captured=captured)


def test_protocol_conformance(patched) -> None:
    obj = TransformersInference("m")
    assert isinstance(obj, ModelInference)


def test_model_set_to_eval(patched) -> None:
    TransformersInference("m")
    assert patched.model.eval_called is True


def test_bf16_path_passes_dtype(patched) -> None:
    TransformersInference("m", load_4bit=False)
    assert "dtype" in patched.captured["model_kwargs"]
    assert "quantization_config" not in patched.captured["model_kwargs"]


def test_4bit_path_passes_quant_config(patched, monkeypatch) -> None:
    import transformers

    monkeypatch.setattr(transformers, "BitsAndBytesConfig", lambda **kw: {"bnb": kw})
    TransformersInference("m", load_4bit=True)
    assert "quantization_config" in patched.captured["model_kwargs"]
    assert "dtype" not in patched.captured["model_kwargs"]


def test_build_prompt_uses_chat_template(patched) -> None:
    obj = TransformersInference("m", chat=True)
    assert obj._build_prompt("hello") == "CHAT::hello"


def test_build_prompt_raw_when_chat_disabled(patched) -> None:
    obj = TransformersInference("m", chat=False)
    assert obj._build_prompt("hello") == "hello"


def test_build_prompt_raw_when_no_template(patched) -> None:
    obj = TransformersInference("m", chat=True)
    obj._tokenizer.chat_template = None
    assert obj._build_prompt("hello") == "hello"


def test_generate_slices_prompt_tokens(patched) -> None:
    obj = TransformersInference("m")
    # 5 output tokens, 3 prompt tokens -> generated [4, 5]
    assert obj.generate("hello") == "DEC:4,5"


def test_generate_passes_explicit_max_tokens(patched) -> None:
    obj = TransformersInference("m")
    obj.generate("hello", max_tokens=7)
    assert patched.model.last_kwargs["max_new_tokens"] == 7


def test_generate_falls_back_to_configured_max(patched) -> None:
    obj = TransformersInference("m", max_new_tokens=33)
    obj.generate("hello")  # no explicit max_tokens
    assert patched.model.last_kwargs["max_new_tokens"] == 33


def test_pad_token_defaulted_when_missing(monkeypatch) -> None:
    import transformers

    tok = FakeTokenizer(pad_token_id=None)
    model = FakeModel()
    monkeypatch.setattr(
        transformers, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *a, **k: tok)
    )
    monkeypatch.setattr(
        transformers,
        "AutoModelForCausalLM",
        types.SimpleNamespace(from_pretrained=lambda *a, **k: model),
    )
    TransformersInference("m")
    assert tok.pad_token == tok.eos_token


def test_adapter_attached_when_path_given(patched, monkeypatch) -> None:
    import peft

    wrapped = FakeModel()
    calls: dict = {}

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, adapter_path):
            calls["model"] = model
            calls["adapter_path"] = adapter_path
            return wrapped

    monkeypatch.setattr(peft, "PeftModel", FakePeftModel)
    obj = TransformersInference("m", adapter_path="/path/to/adapter")
    assert calls["adapter_path"] == "/path/to/adapter"
    assert obj._model is wrapped
