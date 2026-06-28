"""Docling-native chunker.

Uses Docling's ``HybridChunker`` to produce semantically-coherent, token-aware
chunks directly from a ``DoclingDocument`` — the way Docling is meant to be used
(DocumentConverter -> DoclingDocument -> HybridChunker). This replaces the old
path that flattened Docling output to ``Block`` objects and re-chunked them with
the heuristic ``HierarchyChunker`` (which over-segmented into one micro-chunk per
line).

Docling owns chunk boundaries, structure, reading order, and token sizing. On
top, with taste, we only fill the gaps Docling intentionally leaves:
- collapse letter-spacing artifacts in the text layer (Docling keeps them), and
- drop *small front-matter* boilerplate (Docling faithfully keeps cover pages).

This chunker is intentionally NOT registered in ``ChunkerRegistry``: its input is
a ``DoclingDocument``, not a ``list[Block]``, so it cannot satisfy the
``chunk(blocks)`` contract. It is invoked directly from the upload pipeline when
a ``DoclingDocument`` is available.
"""

from __future__ import annotations

import logging
from typing import Any

from chonk.cleaning.normalizer import normalize_text
from chonk.core.document import Chunk
from chonk.qa.patterns import (
    BOILERPLATE_PATTERNS,
    DISTRIBUTION_PATTERNS,
    TOC_PATTERNS,
    match_patterns,
)
from chonk.utils.tokens import count_tokens

logger = logging.getLogger(__name__)

# Embedding model whose tokenizer/window sizes the chunks. all-MiniLM-L6-v2 has
# a 256-token window, so chunks are sized to fit it for retrieval.
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MAX_TOKENS = 256

# Front-matter boilerplate is dropped only when the chunk is ALSO small, so a
# legitimate body chunk that merely mentions a phrase is never removed.
_FRONT_MATTER_PATTERNS = BOILERPLATE_PATTERNS + DISTRIBUTION_PATTERNS + TOC_PATTERNS
_BOILERPLATE_MAX_TOKENS = 100


class DoclingChunker:
    """Chunk a ``DoclingDocument`` with Docling's HybridChunker into our Chunks.

    Args:
        embed_model: HF id of the embedding model whose tokenizer sizes chunks.
            Defaults to all-MiniLM-L6-v2.
        max_tokens: Per-chunk token budget (against the embed model's window).
        merge_peers: Whether HybridChunker merges undersized sibling chunks.
    """

    def __init__(
        self,
        embed_model: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        merge_peers: bool = True,
    ) -> None:
        self._embed_model = embed_model or DEFAULT_EMBED_MODEL
        self._max_tokens = max_tokens
        self._merge_peers = merge_peers
        self._chunker: Any | None = None

    def _build_chunker(self) -> Any:
        """Construct a HybridChunker, with graceful tokenizer fallback.

        Uses a HuggingFace tokenizer matched to the embedding model (so chunk
        sizes respect the embedder's window). If that tokenizer can't be loaded
        (offline / no cache), falls back to HybridChunker's default tokenizer so
        the path still works locally.
        """
        from docling.chunking import HybridChunker

        try:
            from docling_core.transforms.chunker.tokenizer.huggingface import (
                HuggingFaceTokenizer,
            )
            from transformers import AutoTokenizer

            # AutoTokenizer needs a fully-qualified repo id; the app's
            # embedding_model setting is often the short "all-MiniLM-L6-v2".
            model_id = self._embed_model
            if "/" not in model_id:
                model_id = f"sentence-transformers/{model_id}"

            tokenizer = HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(model_id),
                max_tokens=self._max_tokens,
            )
            return HybridChunker(tokenizer=tokenizer, merge_peers=self._merge_peers)
        except Exception as exc:  # offline / no cache / API drift
            logger.warning(
                "DoclingChunker: falling back to default tokenizer (%s): %s",
                self._embed_model,
                exc,
            )
            return HybridChunker()

    def chunk_docling_document(self, dl_doc: Any) -> list[Chunk]:
        """Convert a DoclingDocument into a list of our ``Chunk`` objects.

        Args:
            dl_doc: A parsed ``DoclingDocument``.

        Returns:
            Chunks with cleaned content, heading-based hierarchy_path, page
            range from provenance, and the contextualized embed text stored in
            system_metadata. Small front-matter boilerplate is dropped.
        """
        if self._chunker is None:
            self._chunker = self._build_chunker()

        chunks: list[Chunk] = []
        for dc in self._chunker.chunk(dl_doc=dl_doc):
            chunk = self._to_chunk(dc)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    def _to_chunk(self, dc: Any) -> Chunk | None:
        """Map one Docling chunk to our ``Chunk`` (or None to drop it)."""
        raw = getattr(dc, "text", "") or ""
        content = normalize_text(raw)
        if not content.strip():
            return None

        token_count = count_tokens(content)
        if self._is_front_matter_boilerplate(content, token_count):
            return None

        meta = getattr(dc, "meta", None)
        headings = list(getattr(meta, "headings", None) or []) if meta else []
        hierarchy_path = " > ".join(h.strip() for h in headings if h and h.strip())

        doc_items = (getattr(meta, "doc_items", None) or []) if meta else []
        block_ids = [
            ref for it in doc_items if (ref := getattr(it, "self_ref", ""))
        ]
        pages = self._pages(doc_items)

        # Heading-prefixed text is what should be embedded downstream.
        embed_text = content
        try:
            ctx = self._chunker.contextualize(chunk=dc)
            if ctx:
                embed_text = normalize_text(ctx)
        except Exception:  # contextualize is best-effort
            pass

        system_metadata: dict[str, Any] = {
            "source": "docling",
            "block_count": len(doc_items),
            "embed_text": embed_text,
        }
        if pages:
            system_metadata["start_page"] = min(pages)
            system_metadata["end_page"] = max(pages)

        return Chunk(
            id=Chunk.generate_id(),
            block_ids=block_ids,
            content=content,
            token_count=token_count,
            hierarchy_path=hierarchy_path,
            system_metadata=system_metadata,
        )

    @staticmethod
    def _pages(doc_items: list[Any]) -> list[int]:
        """Collect page numbers from a chunk's doc_items' provenance."""
        pages: set[int] = set()
        for it in doc_items:
            for prov in getattr(it, "prov", None) or []:
                page_no = getattr(prov, "page_no", None)
                if isinstance(page_no, int):
                    pages.add(page_no)
        return sorted(pages)

    @staticmethod
    def _is_front_matter_boilerplate(content: str, token_count: int) -> bool:
        """True only for SMALL chunks that match a front-matter pattern.

        Gating on size keeps the filter conservative: a large body chunk that
        merely mentions a phrase is never dropped.
        """
        if token_count > _BOILERPLATE_MAX_TOKENS:
            return False
        return match_patterns(content, _FRONT_MATTER_PATTERNS) is not None
