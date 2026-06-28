"""Endpoint test for the iterative apply-fixes convergence loop.

apply-fixes must cycle (diagnose -> plan -> execute) until no fixable problems
remain, not stop after one pass. Uses a fake tester (no embedding) and the
autouse conftest autosave isolation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from chonk import server as srv
from chonk.core.document import ChonkDocument, ChonkProject, Chunk
from fastapi.testclient import TestClient


class _FakeTester:
    def index_documents(self, documents) -> int:
        return sum(len(d.chunks) for d in documents)


@pytest.fixture
def client() -> TestClient:
    project = ChonkProject(id="p1", name="P")
    # Several tiny, mergeable fragments in the same section -> multiple passes.
    chunks = [
        Chunk(
            id=f"c{i}",
            block_ids=[],
            content=f"Fragment number {i}.",
            token_count=4,
            hierarchy_path="1. Scope",
        )
        for i in range(6)
    ]
    doc = ChonkDocument(
        id="d1", source_path=Path("x.pdf"), source_type="pdf", blocks=[], chunks=chunks
    )
    project.documents.append(doc)
    srv._state["project"] = project
    srv._state["tester"] = _FakeTester()
    try:
        yield TestClient(srv.app)
    finally:
        srv._state["project"] = None
        srv._state["tester"] = None


def test_apply_fixes_cycles_and_reduces_chunks(client: TestClient) -> None:
    resp = client.post("/api/diagnostics/apply-fixes", json={"document_id": "d1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"] == "success"
    assert body["iterations"] >= 1
    fr = body["fix_result"]
    assert fr["chunks_after"] < fr["chunks_before"]  # fragments got merged
    # The fixed chunks are persisted on the document.
    assert len(srv._state["project"].documents[0].chunks) == fr["chunks_after"]
