"""Tests for the Quarry->Hearth retrieval bridge (RealQuarryRetrievalAdapter).

Unit tests use a fake tester (hermetic, no model load). One integration test
uses a real RetrievalTester with real (CPU) all-MiniLM embeddings to prove the
end-to-end path: indexed chunks -> adapter -> RAGPipeline -> citations.
"""

from __future__ import annotations

from pathlib import Path

import chonk.server as chonk_server
import pytest

from foundry.src.evaluation import MockInference
from foundry.src.rag_integration import RAGPipeline, RetrievalAdapter
from kiln_retrieval import RealQuarryRetrievalAdapter

# --------------------------------------------------------------------------
# Fakes for hermetic unit tests
# --------------------------------------------------------------------------


class _FakeChunk:
    def __init__(self, cid: str, content: str, hpath: str = "", page: int | None = None) -> None:
        self.id = cid
        self.content = content
        self.hierarchy_path = hpath
        self.page_range = (page, page) if page is not None else None


class _FakeResult:
    def __init__(self, chunk: _FakeChunk, score: float, doc_name: str) -> None:
        self.chunk = chunk
        self.score = score
        self.document_name = doc_name


class _FakeTester:
    def __init__(self, results: list[_FakeResult], indexed: bool = True) -> None:
        self._results = results
        self.is_indexed = indexed
        self.calls: list[tuple] = []

    def search(self, query: str, top_k: int = 5, document_ids=None) -> list[_FakeResult]:
        self.calls.append((query, top_k, document_ids))
        return self._results[:top_k]


def _patch_tester(monkeypatch: pytest.MonkeyPatch, tester) -> None:
    monkeypatch.setattr(chonk_server, "get_active_tester", lambda: tester)


# --------------------------------------------------------------------------
# Unit tests
# --------------------------------------------------------------------------


def test_protocol_conformance() -> None:
    assert isinstance(RealQuarryRetrievalAdapter(), RetrievalAdapter)


def test_maps_search_result_to_chunk_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _FakeResult(
        _FakeChunk("c1", "filter content", hpath="Ch 3 > Hydraulics", page=42),
        score=0.91,
        doc_name="TM-9.pdf",
    )
    _patch_tester(monkeypatch, _FakeTester([result]))

    out = RealQuarryRetrievalAdapter().retrieve("how to replace filter")

    assert len(out) == 1
    chunk = out[0]
    assert chunk["text"] == "filter content"
    assert chunk["score"] == pytest.approx(0.91)
    assert chunk["metadata"] == {
        "chunk_id": "c1",
        "document_title": "TM-9.pdf",
        "section": "Ch 3 > Hydraulics",
        "page": 42,
    }


def test_no_tester_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_tester(monkeypatch, None)
    assert RealQuarryRetrievalAdapter().retrieve("q") == []


def test_unindexed_tester_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_tester(monkeypatch, _FakeTester([], indexed=False))
    assert RealQuarryRetrievalAdapter().retrieve("q") == []


def test_missing_metadata_uses_safe_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _FakeResult(_FakeChunk("c2", "text", hpath="", page=None), 0.5, doc_name="")
    _patch_tester(monkeypatch, _FakeTester([result]))

    meta = RealQuarryRetrievalAdapter().retrieve("q")[0]["metadata"]
    assert meta["document_title"] == "Unknown Document"
    assert meta["section"] == ""
    assert meta["page"] is None


def test_document_ids_filter_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    tester = _FakeTester([])
    _patch_tester(monkeypatch, tester)
    RealQuarryRetrievalAdapter().retrieve("q", filters={"document_ids": ["d1", "d2"]})
    assert tester.calls[-1][2] == ["d1", "d2"]


def test_top_k_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    results = [_FakeResult(_FakeChunk(f"c{i}", "t"), 0.5, "d") for i in range(5)]
    tester = _FakeTester(results)
    _patch_tester(monkeypatch, tester)
    out = RealQuarryRetrievalAdapter(top_k=2).retrieve("q")
    assert len(out) == 2
    assert tester.calls[-1][1] == 2  # search called with top_k=2


# --------------------------------------------------------------------------
# Integration test — real RetrievalTester (real CPU embeddings) end to end
# --------------------------------------------------------------------------


def _make_real_document():
    from chonk.core.document import ChonkDocument, Chunk, DocumentMetadata

    chunks = [
        Chunk(
            id="chunk_hyd",
            block_ids=["b1"],
            content=(
                "To replace the hydraulic filter, first depressurize the system, "
                "then remove the filter housing cover and extract the element."
            ),
            token_count=24,
            hierarchy_path="Chapter 3 > Hydraulic System",
            system_metadata={"start_page": 42, "end_page": 42},
        ),
        Chunk(
            id="chunk_elec",
            block_ids=["b2"],
            content=(
                "The electrical system runs on a 24-volt bus with fuses located "
                "in the forward distribution panel."
            ),
            token_count=20,
            hierarchy_path="Chapter 4 > Electrical System",
            system_metadata={"start_page": 70, "end_page": 70},
        ),
    ]
    return ChonkDocument(
        id="doc_tm",
        source_path=Path("TM-9-test.pdf"),
        source_type="pdf",
        blocks=[],
        chunks=chunks,
        metadata=DocumentMetadata(title="TM 9-Test"),
    )


def test_real_index_end_to_end_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Index a real doc, run a RAG query, and confirm citations resolve."""
    from chonk.testing import RetrievalTester

    tester = RetrievalTester()
    tester.index_documents([_make_real_document()])
    _patch_tester(monkeypatch, tester)

    pipeline = RAGPipeline(
        model=MockInference(default_response="(grounded answer)"),
        retrieval=RealQuarryRetrievalAdapter(),
    )
    resp = pipeline.query("How do I replace the hydraulic filter?")

    # Generation ran through the pipeline...
    assert resp.answer == "(grounded answer)"
    # ...grounded in retrieved context with resolving citations.
    assert resp.context_used, "expected non-empty retrieved context"
    cited_ids = {c.chunk_id for c in resp.citations}
    assert "chunk_hyd" in cited_ids
    hyd = next(c for c in resp.citations if c.chunk_id == "chunk_hyd")
    assert hyd.document_title == "TM-9-test.pdf"
    assert hyd.page == 42
    assert hyd.section == "Chapter 3 > Hydraulic System"


def test_default_pipeline_uses_live_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pipeline kiln_server actually builds retrieves from the live index.

    Exercises the real production path: kiln_server._create_default_rag_pipeline()
    + the real chonk.server.get_active_tester reading _state['tester'].
    """
    from chonk.testing import RetrievalTester

    import kiln_server

    tester = RetrievalTester()
    tester.index_documents([_make_real_document()])
    monkeypatch.setitem(chonk_server._state, "tester", tester)

    pipeline = kiln_server._create_default_rag_pipeline()
    resp = pipeline.query("How do I replace the hydraulic filter?")

    assert resp.context_used
    assert "chunk_hyd" in {c.chunk_id for c in resp.citations}
