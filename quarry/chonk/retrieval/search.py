"""Stage 2: Semantic search on filtered chunk subset.

Defines the SearchProvider protocol and a keyword-based implementation
for testing. Production deployments plug in embedding-based providers.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Any

from chonk.core.document import Chunk


@dataclass
class ScoredChunk:
    """A chunk with a relevance score from semantic search.

    Args:
        chunk: The matching chunk.
        score: Relevance score (0.0-1.0, higher is better).
        rank: Position in results (1-based).
    """

    chunk: Chunk
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "chunk_id": self.chunk.id,
            "score": self.score,
            "rank": self.rank,
        }


class SearchProvider(ABC):
    """Abstract interface for semantic search providers.

    Implementations must provide a search method that returns
    scored chunks ranked by relevance.
    """

    @abstractmethod
    def search(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 10,
    ) -> list[ScoredChunk]:
        """Search chunks for the most relevant matches.

        Args:
            query: Natural language query.
            chunks: Pre-filtered chunks to search.
            top_k: Maximum results to return.

        Returns:
            List of ScoredChunk ordered by score descending.
        """


class KeywordSearch(SearchProvider):
    """Simple keyword-overlap search for testing.

    Uses TF-IDF-like scoring based on term overlap between
    query and chunk content. Not suitable for production but
    sufficient for pipeline testing.
    """

    def search(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 10,
    ) -> list[ScoredChunk]:
        """Search by keyword overlap scoring.

        Args:
            query: Natural language query.
            chunks: Chunks to search.
            top_k: Maximum results.

        Returns:
            ScoredChunks ranked by keyword overlap.
        """
        if not chunks or not query.strip():
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        doc_freq = self._compute_doc_freq(chunks, query_terms)
        n_docs = len(chunks)

        scored: list[tuple[Chunk, float]] = []
        for chunk in chunks:
            score = self._score_chunk(chunk, query_terms, doc_freq, n_docs)
            if score > 0:
                scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]

        max_score = scored[0][1] if scored else 1.0
        return [
            ScoredChunk(
                chunk=chunk,
                score=round(score / max_score, 3) if max_score > 0 else 0.0,
                rank=i + 1,
            )
            for i, (chunk, score) in enumerate(scored)
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split text into lowercase alpha tokens.

        Args:
            text: Text to tokenize.

        Returns:
            List of lowercase word tokens.
        """
        return [w.lower() for w in re.findall(r"[a-zA-Z]+", text) if len(w) > 1]

    def _compute_doc_freq(self, chunks: list[Chunk], terms: list[str]) -> dict[str, int]:
        """Count how many chunks contain each query term.

        Args:
            chunks: Document chunks.
            terms: Query terms.

        Returns:
            Mapping of term to document frequency.
        """
        freq: dict[str, int] = {t: 0 for t in terms}
        for chunk in chunks:
            chunk_terms = set(self._tokenize(chunk.content))
            for term in terms:
                if term in chunk_terms:
                    freq[term] = freq.get(term, 0) + 1
        return freq

    def _score_chunk(
        self,
        chunk: Chunk,
        query_terms: list[str],
        doc_freq: dict[str, int],
        n_docs: int,
    ) -> float:
        """Compute TF-IDF-like score for a chunk.

        Args:
            chunk: Chunk to score.
            query_terms: Tokenized query.
            doc_freq: Document frequencies.
            n_docs: Total documents.

        Returns:
            Relevance score (higher is better).
        """
        chunk_tokens = self._tokenize(chunk.content)
        chunk_counts = Counter(chunk_tokens)
        chunk_len = len(chunk_tokens) if chunk_tokens else 1

        score = 0.0
        for term in query_terms:
            tf = chunk_counts.get(term, 0) / chunk_len
            df = doc_freq.get(term, 0)
            idf = math.log((n_docs + 1) / (df + 1)) + 1.0
            score += tf * idf

        return score
