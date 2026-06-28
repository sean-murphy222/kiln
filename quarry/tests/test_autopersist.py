"""Tests for Quarry auto-persistence (survive a server restart).

The project lives in memory; without persistence a restart drops it and the
next request fails with "No project loaded" while the UI still shows the doc.
Autosave-on-mutation + autoload-on-first-access fixes that.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from chonk import server as srv
from chonk.core.document import ChonkDocument, ChonkProject, Chunk
from fastapi import HTTPException


class _FakeTester:
    """Avoids loading the embedding model during re-index on autoload."""

    def index_documents(self, documents) -> int:
        return sum(len(d.chunks) for d in documents)


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "_AUTOSAVE_PATH", tmp_path / "auto.chonk")
    monkeypatch.setattr(srv, "RetrievalTester", _FakeTester)
    yield
    srv._state["project"] = None
    srv._state["tester"] = None


def _project_with_chunk() -> ChonkProject:
    proj = ChonkProject(id="p1", name="Persisted")
    doc = ChonkDocument(
        id="d1",
        source_path=Path("x.pdf"),
        source_type="pdf",
        blocks=[],
        chunks=[Chunk(id="c1", block_ids=[], content="hello world", token_count=2)],
    )
    proj.documents.append(doc)
    return proj


def test_autosave_then_autoload_restores_project() -> None:
    srv._state["project"] = _project_with_chunk()
    srv._autosave_project()
    assert srv._AUTOSAVE_PATH.exists()

    # Simulate a restart: in-memory state is gone.
    srv._state["project"] = None
    srv._state["tester"] = None

    restored = srv._get_project()  # autoloads
    assert restored.name == "Persisted"
    assert restored.documents[0].chunks[0].content == "hello world"


def test_autosave_preserves_explicit_project_path() -> None:
    proj = _project_with_chunk()
    proj.project_path = Path("user/chosen/path.chonk")
    srv._state["project"] = proj
    srv._autosave_project()
    # Autosave must not hijack the user's explicit save path.
    assert proj.project_path == Path("user/chosen/path.chonk")


def test_get_project_raises_when_none_and_no_autosave() -> None:
    srv._state["project"] = None
    with pytest.raises(HTTPException):
        srv._get_project()
