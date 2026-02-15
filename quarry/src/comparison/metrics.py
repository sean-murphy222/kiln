"""
Chunking metrics.

Metrics for evaluating chunking quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chonk.core.document import Block, Chunk


@dataclass
class ChunkingMetrics:
    """
    Metrics for evaluating a chunking strategy.

    These metrics help users understand if a chunking strategy is good
    for their document and use case.
    """

    total_chunks: int
    total_blocks: int
    avg_tokens_per_chunk: float
    min_tokens: int
    max_tokens: int
    median_tokens: int

    # Quality metrics
    avg_quality_score: float
    chunks_with_context: int  # Chunks with hierarchy paths
    hierarchy_preservation: float  # 0-1, how well structure is preserved

    # Distribution
    chunks_under_100_tokens: int
    chunks_100_to_500_tokens: int
    chunks_over_500_tokens: int

    @staticmethod
    def from_chunks(chunks: list[Chunk], blocks: list[Block]) -> ChunkingMetrics:
        """Calculate metrics from chunks."""
        if not chunks:
            return ChunkingMetrics(
                total_chunks=0,
                total_blocks=len(blocks),
                avg_tokens_per_chunk=0,
                min_tokens=0,
                max_tokens=0,
                median_tokens=0,
                avg_quality_score=0,
                chunks_with_context=0,
                hierarchy_preservation=0,
                chunks_under_100_tokens=0,
                chunks_100_to_500_tokens=0,
                chunks_over_500_tokens=0,
            )

        token_counts = [c.token_count for c in chunks]
        quality_scores = [c.quality.overall for c in chunks if c.quality]

        # Count chunks with hierarchy context
        chunks_with_hierarchy = sum(1 for c in chunks if c.hierarchy_path)

        # Hierarchy preservation: ratio of chunks with hierarchy paths
        hierarchy_preservation = chunks_with_hierarchy / len(chunks) if chunks else 0

        # Token distribution
        under_100 = sum(1 for t in token_counts if t < 100)
        between_100_500 = sum(1 for t in token_counts if 100 <= t <= 500)
        over_500 = sum(1 for t in token_counts if t > 500)

        # Calculate median
        sorted_tokens = sorted(token_counts)
        median_tokens = sorted_tokens[len(sorted_tokens) // 2]

        return ChunkingMetrics(
            total_chunks=len(chunks),
            total_blocks=len(blocks),
            avg_tokens_per_chunk=sum(token_counts) / len(token_counts),
            min_tokens=min(token_counts),
            max_tokens=max(token_counts),
            median_tokens=median_tokens,
            avg_quality_score=sum(quality_scores) / len(quality_scores)
            if quality_scores
            else 0,
            chunks_with_context=chunks_with_hierarchy,
            hierarchy_preservation=hierarchy_preservation,
            chunks_under_100_tokens=under_100,
            chunks_100_to_500_tokens=between_100_500,
            chunks_over_500_tokens=over_500,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_chunks": self.total_chunks,
            "total_blocks": self.total_blocks,
            "avg_tokens_per_chunk": round(self.avg_tokens_per_chunk, 1),
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "median_tokens": self.median_tokens,
            "avg_quality_score": round(self.avg_quality_score, 3),
            "chunks_with_context": self.chunks_with_context,
            "hierarchy_preservation": round(self.hierarchy_preservation, 3),
            "token_distribution": {
                "under_100": self.chunks_under_100_tokens,
                "100_to_500": self.chunks_100_to_500_tokens,
                "over_500": self.chunks_over_500_tokens,
            },
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"""
ChunkingMetrics:
  Total chunks: {self.total_chunks}
  Avg tokens: {self.avg_tokens_per_chunk:.1f}
  Range: {self.min_tokens}-{self.max_tokens}
  Quality score: {self.avg_quality_score:.3f}
  Hierarchy preservation: {self.hierarchy_preservation:.1%}

  Distribution:
    < 100 tokens: {self.chunks_under_100_tokens}
    100-500 tokens: {self.chunks_100_to_500_tokens}
    > 500 tokens: {self.chunks_over_500_tokens}
"""
