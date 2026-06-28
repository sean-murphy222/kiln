"""Regression tests for the strategy query-testing endpoints.

These endpoints had latent bugs that 500'd (never tested):
- /api/test/query mapped SearchResult.chunk_id/content_preview (fields live on
  result.chunk), and re-used the shared tester (corrupting the index).
- /api/test/compare-strategies built a TestQueryRequest (no strategies field) ->
  AttributeError; it must build a StrategyTestRequest.
- chunker name not found -> get_chunker returns None -> None.chunk() 500;
  should be a clean 400.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from chonk import server as srv
from chonk.core.document import Block, BlockType, ChonkDocument, ChonkProject
from chonk.testing.searcher import SearchResult
from fastapi.testclient import TestClient


class _FakeTester:
    """Avoids loading the embedding model in the happy-path test."""

    def __init__(self, *_a, **_k) -> None:
        self._chunks = []

    @property
    def is_indexed(self) -> bool:
        return bool(self._chunks)

    def index_chunks(self, chunks, doc_id: str = "", doc_name: str = "") -> int:
        self._chunks = list(chunks)
        return len(self._chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return [
            SearchResult(chunk=c, score=0.9, rank=i + 1) for i, c in enumerate(self._chunks[:top_k])
        ]


@pytest.fixture
def client() -> TestClient:
    project = ChonkProject(id="p1", name="P")
    doc = ChonkDocument(
        id="d1",
        source_path=Path("x.pdf"),
        source_type="pdf",
        blocks=[
            Block(id="b0", type=BlockType.HEADING, content="Section 1", page=1, heading_level=1),
            Block(
                id="b1",
                type=BlockType.TEXT,
                content="Depressurize the system before removing the filter assembly.",
                page=1,
            ),
        ],
        chunks=[],
    )
    project.documents.append(doc)
    srv._state["project"] = project
    srv._state["tester"] = _FakeTester()
    try:
        yield TestClient(srv.app)
    finally:
        srv._state["project"] = None
        srv._state["tester"] = None


def test_test_query_unknown_strategy_is_400(client: TestClient) -> None:
    resp = client.post(
        "/api/test/query", json={"query": "filter", "strategies": ["does-not-exist"]}
    )
    assert resp.status_code == 400


def test_compare_strategies_unknown_strategy_is_400_not_500(client: TestClient) -> None:
    # If compare built the wrong request model this would 500 (AttributeError);
    # the None-chunker guard makes it a clean 400.
    resp = client.post(
        "/api/test/compare-strategies",
        json={"queries": ["filter"], "strategies": ["does-not-exist"]},
    )
    assert resp.status_code == 400


def test_test_query_maps_search_result_fields(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("chonk.server.RetrievalTester", _FakeTester)
    resp = client.post("/api/test/query", json={"query": "filter", "strategies": ["hierarchy"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategies"]
    hits = body["strategies"][0]["results"]
    assert hits and "chunk_id" in hits[0] and "content_preview" in hits[0]
