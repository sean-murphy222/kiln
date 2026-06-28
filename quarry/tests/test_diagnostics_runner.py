"""Regression tests for the diagnostics question-testing path.

This path (RetrievalTester.index_chunks + QuestionTestRunner.run_diagnostic_tests)
had no coverage and shipped three latent bugs that 500'd /api/diagnostics/analyze:
1. RetrievalTester had no index_chunks method.
2. test_runner used SearchResult.chunk_id (the field is SearchResult.chunk.id).
3. _analyze_results indexed by status "pass"/"fail" into a dict keyed
   "passed"/"failed" -> KeyError.

The embedder is faked so these run fast without loading a model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from chonk.core.document import ChonkDocument, Chunk
from chonk.diagnostics.test_runner import QuestionTestRunner
from chonk.testing.searcher import SearchResult


def _chunks(n: int = 6) -> list[Chunk]:
    return [
        Chunk(
            id=Chunk.generate_id(),
            block_ids=[f"#/texts/{i}"],
            content=(
                f"Section {i}: The operator shall depressurize the system before "
                f"removing the filter. See figure {i} and table {i}. Verify pressure "
                "is zero before proceeding to step 2."
            ),
            token_count=30,
            hierarchy_path=f"4.{i} Procedures",
            system_metadata={"source": "docling", "start_page": i + 1, "end_page": i + 1},
        )
        for i in range(n)
    ]


def _doc(chunks: list[Chunk]) -> ChonkDocument:
    return ChonkDocument(
        id="doc_test",
        source_path=Path("MIL-STD.pdf"),
        source_type="pdf",
        blocks=[],
        chunks=chunks,
    )


def test_retrieval_tester_index_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """index_chunks indexes a flat chunk list and search returns SearchResults."""

    class FakeEmbedder:
        def __init__(self, *_a, **_k) -> None:
            pass

        def embed_many(self, texts, show_progress: bool = False):
            return np.array([[float(len(t)), 1.0] for t in texts])

        def embed(self, text):
            return np.array([float(len(text)), 1.0])

    monkeypatch.setattr("chonk.testing.searcher.Embedder", FakeEmbedder)
    from chonk.testing.searcher import RetrievalTester

    chunks = _chunks(3)
    tester = RetrievalTester()
    n = tester.index_chunks(chunks, doc_id="d1", doc_name="D.pdf")

    assert n == 3
    assert tester.is_indexed is True
    assert tester.chunk_count == 3
    results = tester.search("depressurize the system", top_k=2)
    assert results
    assert results[0].chunk.id in {c.id for c in chunks}


class _FakeTester:
    """Minimal tester for run_diagnostic_tests (no embedding)."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._is_indexed = False

    @property
    def is_indexed(self) -> bool:
        return self._is_indexed

    def index_chunks(self, chunks, doc_id: str = "", doc_name: str = "") -> int:
        self._is_indexed = True
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return [
            SearchResult(chunk=c, score=0.9 - 0.05 * i, rank=i + 1)
            for i, c in enumerate(self._chunks[:top_k])
        ]


def test_run_diagnostic_tests_end_to_end() -> None:
    """The full runner produces a report without raising (covers bugs 2 and 3)."""
    chunks = _chunks(6)
    runner = QuestionTestRunner(_FakeTester(chunks))

    report = runner.run_diagnostic_tests(_doc(chunks), top_k=5)
    data = report.to_dict()

    assert "summary" in data
    assert "by_test_type" in data
    # Each test-type bucket uses the passed/partial/failed keys without KeyError.
    for bucket in data["by_test_type"].values():
        assert set(bucket) >= {"passed", "partial", "failed"}
