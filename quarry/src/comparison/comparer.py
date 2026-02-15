"""
Strategy comparer.

Compare different chunking strategies side-by-side to help users choose
the best approach for their document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chonk.chunkers.base import BaseChunker
from chonk.core.document import Block, Chunk
from chonk.comparison.metrics import ChunkingMetrics


@dataclass
class StrategyResult:
    """Result from applying one chunking strategy."""

    strategy_name: str
    chunks: list[Chunk]
    metrics: ChunkingMetrics
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """
    Result of comparing multiple chunking strategies.

    Shows side-by-side metrics and recommendations.
    """

    strategies: list[StrategyResult]

    @property
    def best_strategy(self) -> StrategyResult | None:
        """
        Determine the best strategy based on metrics.

        Scoring criteria:
        - Higher hierarchy preservation
        - Better token distribution
        - Higher quality scores
        - Lower chunk count (fewer is often better for cost)
        """
        if not self.strategies:
            return None

        def score_strategy(result: StrategyResult) -> float:
            m = result.metrics
            score = 0.0

            # Hierarchy preservation is KEY
            score += m.hierarchy_preservation * 40

            # Quality scores
            score += m.avg_quality_score * 30

            # Token distribution (prefer closer to target)
            if 300 <= m.avg_tokens_per_chunk <= 600:
                score += 20
            elif 200 <= m.avg_tokens_per_chunk <= 800:
                score += 10

            # Prefer fewer chunks (cost efficiency)
            if m.total_chunks < 1000:
                score += 10

            return score

        return max(self.strategies, key=score_strategy)

    def get_recommendation(self) -> str:
        """Get a human-readable recommendation."""
        best = self.best_strategy

        if not best:
            return "No strategies to compare"

        recommendations = []

        # Main recommendation
        recommendations.append(
            f"✅ RECOMMENDED: {best.strategy_name.upper()} strategy"
        )

        # Why it's best
        m = best.metrics
        reasons = []

        if m.hierarchy_preservation > 0.8:
            reasons.append("preserves document structure")

        if m.avg_quality_score > 0.9:
            reasons.append("high quality chunks")

        if 300 <= m.avg_tokens_per_chunk <= 600:
            reasons.append("optimal token size")

        if m.chunks_with_context > m.total_chunks * 0.9:
            reasons.append("includes hierarchy context")

        if reasons:
            recommendations.append(f"   Reasons: {', '.join(reasons)}")

        # Comparison with others
        other_strategies = [s for s in self.strategies if s != best]
        if other_strategies:
            worst = min(other_strategies, key=lambda s: s.metrics.avg_quality_score)
            quality_diff = (
                best.metrics.avg_quality_score - worst.metrics.avg_quality_score
            )
            if quality_diff > 0.1:
                recommendations.append(
                    f"   {best.strategy_name} is {quality_diff:.1%} better quality than {worst.strategy_name}"
                )

        return "\n".join(recommendations)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "strategies": [
                {
                    "name": result.strategy_name,
                    "metrics": result.metrics.to_dict(),
                    "config": result.config,
                }
                for result in self.strategies
            ],
            "best_strategy": self.best_strategy.strategy_name
            if self.best_strategy
            else None,
            "recommendation": self.get_recommendation(),
        }


class StrategyComparer:
    """
    Compare multiple chunking strategies on the same blocks.

    This is a killer feature - showing users concrete differences between
    strategies before they commit to one.
    """

    @staticmethod
    def compare(
        blocks: list[Block], chunkers: list[tuple[str, BaseChunker]]
    ) -> ComparisonResult:
        """
        Compare multiple chunking strategies.

        Args:
            blocks: List of blocks to chunk
            chunkers: List of (name, chunker) tuples

        Returns:
            ComparisonResult with metrics for each strategy
        """
        results = []

        for name, chunker in chunkers:
            # Apply chunking strategy
            chunks = chunker.chunk(blocks)

            # Calculate metrics
            metrics = ChunkingMetrics.from_chunks(chunks, blocks)

            # Get config
            config = chunker.config.to_dict() if hasattr(chunker, "config") else {}

            results.append(
                StrategyResult(
                    strategy_name=name,
                    chunks=chunks,
                    metrics=metrics,
                    config=config,
                )
            )

        return ComparisonResult(strategies=results)

    @staticmethod
    def compare_with_queries(
        blocks: list[Block],
        chunkers: list[tuple[str, BaseChunker]],
        test_queries: list[str],
    ) -> dict[str, Any]:
        """
        Compare strategies with actual test queries.

        This is the ULTIMATE comparison - showing which strategy
        retrieves better results for real queries.

        Args:
            blocks: List of blocks to chunk
            chunkers: List of (name, chunker) tuples
            test_queries: List of test queries to run

        Returns:
            Dictionary with comparison results and query performance
        """
        from chonk.testing.embedder import Embedder
        from chonk.testing.searcher import RetrievalTester

        comparison = StrategyComparer.compare(blocks, chunkers)

        # Test each strategy with queries
        embedder = Embedder()
        query_results = {}

        for result in comparison.strategies:
            # Create embeddings for chunks
            tester = RetrievalTester()
            for chunk in result.chunks:
                embedding = embedder.embed(chunk.content)
                tester.add_chunk(chunk, embedding)

            # Test each query
            strategy_query_results = []
            for query in test_queries:
                query_embedding = embedder.embed(query)
                retrieved = tester.search(query_embedding, top_k=3)

                strategy_query_results.append(
                    {
                        "query": query,
                        "retrieved_chunks": len(retrieved),
                        "top_chunk_id": retrieved[0].chunk.id if retrieved else None,
                        "top_score": retrieved[0].score if retrieved else 0,
                    }
                )

            query_results[result.strategy_name] = strategy_query_results

        return {
            "comparison": comparison.to_dict(),
            "query_results": query_results,
            "recommendation": comparison.get_recommendation(),
        }
