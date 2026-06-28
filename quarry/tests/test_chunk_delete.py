"""Tests for bulk chunk deletion (POST /api/chunks/delete)."""

from __future__ import annotations

from pathlib import Path

import pytest
from chonk import server as srv
from chonk.core.document import ChonkDocument, ChonkProject, Chunk
from fastapi.testclient import TestClient


class _FakeTester:
    """Avoids loading the embedding model during re-index."""

    is_indexed = True

    def index_documents(self, documents) -> int:
        return sum(len(d.chunks) for d in documents)


@pytest.fixture
def client_with_doc() -> TestClient:
    project = ChonkProject(id="p1", name="P")
    chunks = [
        Chunk(id=f"c{i}", block_ids=[], content=f"chunk {i} content", token_count=5)
        for i in range(4)
    ]
    doc = ChonkDocument(
        id="d1",
        source_path=Path("x.pdf"),
        source_type="pdf",
        blocks=[],
        chunks=chunks,
    )
    project.documents.append(doc)
    srv._state["project"] = project
    srv._state["tester"] = _FakeTester()
    try:
        yield TestClient(srv.app)
    finally:
        srv._state["project"] = None
        srv._state["tester"] = None


def test_bulk_delete_removes_selected_chunks(client_with_doc: TestClient) -> None:
    resp = client_with_doc.post("/api/chunks/delete", json={"chunk_ids": ["c0", "c2"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] == 2
    assert body["requested"] == 2
    remaining = [c.id for c in srv._state["project"].documents[0].chunks]
    assert remaining == ["c1", "c3"]


def test_bulk_delete_empty_ids_is_400(client_with_doc: TestClient) -> None:
    resp = client_with_doc.post("/api/chunks/delete", json={"chunk_ids": []})
    assert resp.status_code == 400


def test_bulk_delete_unknown_ids_is_404(client_with_doc: TestClient) -> None:
    resp = client_with_doc.post("/api/chunks/delete", json={"chunk_ids": ["nope"]})
    assert resp.status_code == 404
