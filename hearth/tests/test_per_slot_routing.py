"""Tests for Hearth per-slot real-model routing (opt-in).

The default mock path is unchanged (no real backend, no routing). When
KILN_INFERENCE_BACKEND=transformers, ModelManager loads a real backend per slot
(single-resident with eviction) and HearthEngine routes generation to it. The
real loader (build_inference) is mocked so these run without a GPU.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from foundry.src import backend_config as bc
from foundry.src.evaluation import MockInference
from foundry.src.rag_integration import MockRetrievalAdapter, RAGConfig, RAGPipeline
from hearth.src.inference import (
    HearthEngine,
    InferenceError,
    ModelManager,
    ModelStatus,
    QueryRequest,
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in bc.ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class FakeBackend:
    def __init__(self, answer: str = "REAL-ANSWER") -> None:
        self.answer = answer
        self.closed = False

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        return self.answer

    def close(self) -> None:
        self.closed = True


def _enable_real(monkeypatch: pytest.MonkeyPatch, backends: list[FakeBackend]) -> None:
    monkeypatch.setenv("KILN_INFERENCE_BACKEND", "transformers")
    it = iter(backends)
    monkeypatch.setattr(
        "foundry.src.inference_factory.build_inference", lambda **kw: next(it)
    )


# --- default (mock) path unchanged -----------------------------------------


def test_mock_load_does_not_create_backend() -> None:
    mgr = ModelManager()
    mgr.register_slot("s1", "Maint", "llama")
    slot = mgr.load_model("s1")
    assert slot.status == ModelStatus.READY
    assert mgr.get_backend("s1") is None


# --- real path: loading, eviction, errors ----------------------------------


def test_real_load_caches_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeBackend()
    _enable_real(monkeypatch, [fake])
    mgr = ModelManager()
    mgr.register_slot("s1", "Maint", "llama", model_path="base/x")
    slot = mgr.load_model("s1")
    assert slot.status == ModelStatus.READY
    assert mgr.get_backend("s1") is fake


def test_single_resident_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    fake1, fake2 = FakeBackend(), FakeBackend()
    _enable_real(monkeypatch, [fake1, fake2])
    mgr = ModelManager()
    mgr.register_slot("s1", "A", "llama", model_path="base/x")
    mgr.register_slot("s2", "B", "llama", model_path="base/x")

    mgr.load_model("s1")
    assert mgr.get_backend("s1") is fake1

    mgr.load_model("s2")
    assert mgr.get_backend("s2") is fake2
    assert mgr.get_backend("s1") is None  # evicted
    assert fake1.closed is True
    assert mgr.get_slot("s1").status == ModelStatus.UNLOADED


def test_unload_frees_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeBackend()
    _enable_real(monkeypatch, [fake])
    mgr = ModelManager()
    mgr.register_slot("s1", "A", "llama", model_path="base/x")
    mgr.load_model("s1")
    mgr.unload_model("s1")
    assert fake.closed is True
    assert mgr.get_backend("s1") is None
    assert mgr.get_slot("s1").status == ModelStatus.UNLOADED


def test_load_failure_sets_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KILN_INFERENCE_BACKEND", "transformers")

    def boom(**kwargs):
        raise RuntimeError("no weights")

    monkeypatch.setattr("foundry.src.inference_factory.build_inference", boom)
    mgr = ModelManager()
    mgr.register_slot("s1", "A", "llama", model_path="base/x")
    with pytest.raises(InferenceError):
        mgr.load_model("s1")
    assert mgr.get_slot("s1").status == ModelStatus.ERROR


# --- routing through HearthEngine.query ------------------------------------


def _pipeline() -> RAGPipeline:
    chunks = [
        {
            "text": "Depressurize the system before removing the filter.",
            "metadata": {"chunk_id": "c1", "document_title": "TM", "section": "3", "page": 1},
            "score": 0.9,
        }
    ]
    return RAGPipeline(
        model=MockInference(default_response="MOCK"),
        retrieval=MockRetrievalAdapter(chunks=chunks),
        config=RAGConfig(),
    )


def test_query_routes_to_real_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeBackend(answer="REAL-ANSWER")
    _enable_real(monkeypatch, [fake])
    mgr = ModelManager()
    mgr.register_slot("s1", "A", "llama", model_path="base/x")
    mgr.load_model("s1")
    engine = HearthEngine(model_manager=mgr, rag_pipeline=_pipeline())
    resp = engine.query(QueryRequest(query="How do I remove the filter?", slot_id="s1"))
    assert resp.answer == "REAL-ANSWER"


def test_query_uses_mock_when_no_real_backend() -> None:
    mgr = ModelManager()
    mgr.register_slot("s1", "A", "llama")
    mgr.load_model("s1")  # mock path -> no backend cached
    engine = HearthEngine(model_manager=mgr, rag_pipeline=_pipeline())
    resp = engine.query(QueryRequest(query="How?", slot_id="s1"))
    assert resp.answer == "MOCK"


def test_real_query_does_not_bleed_into_mock_slot() -> None:
    """A real-backed query must not leave the real model on the pipeline for a
    later slot that has no real backend (regression for the routing bleed)."""
    mgr = ModelManager()
    mgr.register_slot("real", "R", "llama")
    mgr.register_slot("mock", "M", "llama")
    mgr.load_model("real")
    mgr.load_model("mock")
    # Simulate a mixed deployment: a real backend cached for 'real' only.
    mgr._backends["real"] = FakeBackend(answer="REAL-ANSWER")
    engine = HearthEngine(model_manager=mgr, rag_pipeline=_pipeline())

    assert engine.query(QueryRequest(query="q", slot_id="real")).answer == "REAL-ANSWER"
    # The mock slot must fall back to the default model, not the real one.
    assert engine.query(QueryRequest(query="q", slot_id="mock")).answer == "MOCK"


def test_concurrent_queries_do_not_cross_contaminate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under concurrency, each slot's query is answered by its own model.

    Regression for the shared-pipeline race: without the lock, a mock-slot query
    running while a real-slot query holds the swapped model could be answered by
    the real model. The lock serializes the swap+generate so this can't happen.
    """
    mgr = ModelManager()
    mgr.register_slot("real", "R", "llama")
    mgr.register_slot("mock", "M", "llama")
    mgr.load_model("real")
    mgr.load_model("mock")
    mgr._backends["real"] = FakeBackend(answer="REAL")  # mixed: real cached for one slot
    engine = HearthEngine(model_manager=mgr, rag_pipeline=_pipeline())

    def ask(slot_id: str) -> str:
        return engine.query(QueryRequest(query="q", slot_id=slot_id)).answer

    slots = ["real", "mock"] * 25
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(ask, slots))

    paired = list(zip(slots, answers, strict=True))
    assert all(ans == ("REAL" if slot == "real" else "MOCK") for slot, ans in paired)


@pytest.mark.gpu
def test_real_multi_slot_eviction_and_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two real slots load with single-resident eviction and both answer."""
    monkeypatch.setenv("KILN_INFERENCE_BACKEND", "transformers")
    model_id = bc.get_validation_model()
    mgr = ModelManager()
    mgr.register_slot("s1", "A", "llama", model_path=model_id)
    mgr.register_slot("s2", "B", "llama", model_path=model_id)
    engine = HearthEngine(model_manager=mgr, rag_pipeline=_pipeline())

    mgr.load_model("s1")
    r1 = engine.query(QueryRequest(query="How do I remove the filter?", slot_id="s1"))
    assert r1.answer.strip() != ""

    mgr.load_model("s2")  # evicts s1
    assert mgr.get_backend("s1") is None
    assert mgr.get_slot("s1").status == ModelStatus.UNLOADED

    r2 = engine.query(QueryRequest(query="How do I remove the filter?", slot_id="s2"))
    assert r2.answer.strip() != ""
    mgr.unload_model("s2")
