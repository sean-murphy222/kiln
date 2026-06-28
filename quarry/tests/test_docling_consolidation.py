"""Tests for section-aware semantic consolidation in DoclingChunker.

After Docling chunking, adjacent chunks in the SAME section that are semantically
similar and fit the token budget are merged into coherent section units. The
embedder is faked for deterministic, model-free tests.
"""

from __future__ import annotations

import numpy as np
from chonk.chunkers.docling_chunker import DoclingChunker
from chonk.core.document import Chunk


def _c(cid: str, content: str, path: str, toks: int = 40) -> Chunk:
    return Chunk(
        id=cid,
        block_ids=[f"#/{cid}"],
        content=content,
        token_count=toks,
        hierarchy_path=path,
        system_metadata={"source": "docling", "start_page": 1, "end_page": 1, "block_count": 1},
    )


class _FakeEmbedder:
    """Maps chunk content -> a fixed vector (no model load)."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._m = mapping

    def embed_many(self, texts, show_progress: bool = False):
        return np.array([self._m[t] for t in texts], dtype="float32")


def _chunker(mapping, **kw) -> DoclingChunker:
    return DoclingChunker(consolidate=True, embedder=_FakeEmbedder(mapping), **kw)


def test_merges_similar_same_section_neighbors() -> None:
    a = _c("a", "The operator shall depressurize the system.", "4.1 General")
    b = _c("b", "Verify the pressure gauge reads zero first.", "4.1 General")
    c = _c("c", "Wear approved eye protection at all times.", "4.2 Safety")
    mapping = {a.content: [1.0, 0.0], b.content: [0.95, 0.05], c.content: [0.0, 1.0]}

    out = _chunker(mapping)._consolidate_chunks([a, b, c])

    assert len(out) == 2  # a+b merged, c separate
    assert out[0].system_metadata["consolidated_from"] == 2
    assert a.content in out[0].content and b.content in out[0].content
    assert out[0].hierarchy_path == "4.1 General"
    assert out[1].id == c.id  # single-member group passes through unchanged


def test_does_not_merge_across_sections() -> None:
    a = _c("a", "First section content here.", "4.1 General")
    b = _c("b", "Second section content here.", "4.2 Safety")
    mapping = {a.content: [1.0, 0.0], b.content: [1.0, 0.0]}  # identical vectors

    out = _chunker(mapping)._consolidate_chunks([a, b])

    assert len(out) == 2  # different sections never merge


def test_does_not_merge_dissimilar_neighbors() -> None:
    a = _c("a", "Hydraulic system maintenance steps.", "4.1 General")
    b = _c("b", "Unrelated administrative boilerplate text.", "4.1 General")
    mapping = {a.content: [1.0, 0.0], b.content: [0.0, 1.0]}  # orthogonal -> sim 0

    out = _chunker(mapping)._consolidate_chunks([a, b])

    assert len(out) == 2  # same section but not similar -> not merged


def test_respects_token_budget() -> None:
    a = _c("a", "Similar content one.", "4.1 General", toks=40)
    b = _c("b", "Similar content two.", "4.1 General", toks=40)
    mapping = {a.content: [1.0, 0.0], b.content: [1.0, 0.0]}

    out = _chunker(mapping, consolidation_max_tokens=50)._consolidate_chunks([a, b])

    assert len(out) == 2  # 40 + 40 > 50 budget -> not merged
