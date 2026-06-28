"""Tests for the Docling-native chunker.

Docling's HybridChunker is stubbed (injected via the private ``_chunker``) so
these run without docling installed and without a GPU. We verify the mapping
from Docling chunks to our Chunk model, heading -> hierarchy_path, page range
from provenance, text cleaning, and the conservative boilerplate filter.
"""

from __future__ import annotations

from types import SimpleNamespace

from chonk.chunkers.docling_chunker import DoclingChunker


def _prov(page: int) -> SimpleNamespace:
    return SimpleNamespace(page_no=page)


def _item(ref: str, pages: list[int]) -> SimpleNamespace:
    return SimpleNamespace(self_ref=ref, prov=[_prov(p) for p in pages])


def _dchunk(text, headings=None, items=None) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        meta=SimpleNamespace(headings=headings, doc_items=items or []),
    )


class _FakeChunker:
    """Stand-in for docling HybridChunker."""

    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self._chunks = chunks

    def chunk(self, dl_doc=None):
        return iter(self._chunks)

    def contextualize(self, chunk=None) -> str:
        heads = " / ".join(chunk.meta.headings or [])
        return f"{heads}\n{chunk.text}" if heads else chunk.text


def _chunker_with(chunks: list[SimpleNamespace]) -> DoclingChunker:
    dc = DoclingChunker()
    dc._chunker = _FakeChunker(chunks)  # inject; skips real HybridChunker build
    return dc


def test_maps_chunk_with_headings_and_pages() -> None:
    dc = _chunker_with(
        [
            _dchunk(
                "A work package is a unit of work in logical sequence.",
                headings=["4 REQUIREMENTS", "4.1 General"],
                items=[_item("#/texts/1", [5, 6])],
            )
        ]
    )

    chunks = dc.chunk_docling_document(object())

    assert len(chunks) == 1
    c = chunks[0]
    assert c.content == "A work package is a unit of work in logical sequence."
    assert c.hierarchy_path == "4 REQUIREMENTS > 4.1 General"
    assert c.token_count > 0
    assert c.block_ids == ["#/texts/1"]
    assert c.system_metadata["start_page"] == 5
    assert c.system_metadata["end_page"] == 6
    assert c.system_metadata["source"] == "docling"
    assert "4 REQUIREMENTS" in c.system_metadata["embed_text"]


def test_page_range_spans_multiple_items() -> None:
    dc = _chunker_with(
        [
            _dchunk(
                "Spans two pages of body text describing the procedure steps.",
                items=[_item("#/t1", [10]), _item("#/t2", [11])],
            )
        ]
    )
    c = dc.chunk_docling_document(object())[0]
    assert c.system_metadata["start_page"] == 10
    assert c.system_metadata["end_page"] == 11


def test_no_headings_gives_empty_hierarchy_path() -> None:
    dc = _chunker_with([_dchunk("Body text.", headings=None, items=[_item("#/t", [1])])])
    c = dc.chunk_docling_document(object())[0]
    assert c.hierarchy_path == ""


def test_letter_spacing_is_cleaned_in_chunk_text() -> None:
    dc = _chunker_with(
        [_dchunk("N O T   M E A S U R E M E N T", items=[_item("#/t", [1])])]
    )
    c = dc.chunk_docling_document(object())[0]
    assert c.content == "NOT MEASUREMENT"


def test_empty_chunk_is_dropped() -> None:
    dc = _chunker_with([_dchunk("   ", items=[])])
    assert dc.chunk_docling_document(object()) == []


def test_small_front_matter_boilerplate_dropped() -> None:
    dc = _chunker_with(
        [
            _dchunk("NOT MEASUREMENT SENSITIVE", items=[_item("#/t1", [1])]),
            _dchunk(
                "Real body content about removing the filter and depressurizing first.",
                headings=["3 PROCEDURES"],
                items=[_item("#/t2", [2])],
            ),
        ]
    )
    chunks = dc.chunk_docling_document(object())
    contents = [c.content for c in chunks]
    assert not any("MEASUREMENT SENSITIVE" in c for c in contents)  # dropped
    assert any("removing the filter" in c for c in contents)  # kept


def test_large_chunk_mentioning_boilerplate_is_kept() -> None:
    big = "This standard is NOT MEASUREMENT SENSITIVE. " + "procedure step detail. " * 60
    dc = _chunker_with([_dchunk(big, items=[_item("#/t", [1])])])
    chunks = dc.chunk_docling_document(object())
    assert len(chunks) == 1  # large body survives despite the phrase
