"""Bridge Quarry's live retrieval index into the Foundry/Hearth RAG layer.

``RealQuarryRetrievalAdapter`` implements Foundry's ``RetrievalAdapter`` Protocol
(``retrieve(query, filters) -> list[dict]``) by wrapping Quarry's in-memory
``RetrievalTester`` — the same index the Quarry server populates on document
upload. In the unified ``kiln_server`` both Quarry and Hearth run in one process,
so the adapter reads the live index directly.

The tester is looked up **lazily at retrieve() time** (via
``chonk.server.get_active_tester``), so the adapter always sees the current corpus
and degrades gracefully (returns no chunks) when nothing has been indexed yet
rather than raising. ``chonk`` is imported lazily inside ``retrieve`` to keep this
module import-light and avoid import-order coupling.
"""

from __future__ import annotations

from typing import Any

DEFAULT_TOP_K = 5


class RealQuarryRetrievalAdapter:
    """Retrieval adapter backed by Quarry's live ``RetrievalTester``.

    Implements the Foundry ``RetrievalAdapter`` Protocol consumed by
    ``RAGPipeline``.

    Args:
        top_k: Maximum number of chunks to return per query.
    """

    def __init__(self, top_k: int = DEFAULT_TOP_K) -> None:
        self.top_k = top_k

    def retrieve(self, query: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Retrieve relevant chunks from the live Quarry index.

        Args:
            query: The search query text.
            filters: Optional filters; an optional ``document_ids`` list restricts
                the search to specific documents.

        Returns:
            A list of chunk dicts (``text``/``score``/``metadata``), or an empty
            list when no corpus has been indexed yet.
        """
        from chonk.server import get_active_tester

        tester = get_active_tester()
        if tester is None or not getattr(tester, "is_indexed", False):
            return []

        document_ids = filters.get("document_ids") if filters else None
        results = tester.search(query, top_k=self.top_k, document_ids=document_ids)
        return [self._to_chunk_dict(result) for result in results]

    @staticmethod
    def _to_chunk_dict(result: Any) -> dict[str, Any]:
        """Map a Quarry ``SearchResult`` to a Foundry RAG chunk dict.

        Args:
            result: A ``SearchResult`` from ``RetrievalTester.search``.

        Returns:
            Dict with ``text``, ``score``, and ``metadata`` (``chunk_id``,
            ``document_title``, ``section``, ``page``) — the keys
            ``ContextBuilder`` uses to build context and citations.
        """
        chunk = result.chunk
        return {
            "text": chunk.content,
            "score": float(result.score),
            "metadata": {
                "chunk_id": chunk.id,
                "document_title": getattr(result, "document_name", None) or "Unknown Document",
                "section": _section_of(chunk),
                "page": _first_page(chunk),
            },
        }


def _section_of(chunk: Any) -> str:
    """Return a human-readable section string from a chunk's hierarchy path."""
    path = getattr(chunk, "hierarchy_path", None)
    if isinstance(path, (list, tuple)):
        return " > ".join(str(part) for part in path)
    if isinstance(path, str):
        return path
    return ""


def _first_page(chunk: Any) -> int | None:
    """Return the chunk's starting page number, or None if unknown."""
    page_range = getattr(chunk, "page_range", None)
    if isinstance(page_range, (list, tuple)) and page_range:
        try:
            return int(page_range[0])
        except (TypeError, ValueError):
            return None
    if isinstance(page_range, int):
        return page_range
    return None
