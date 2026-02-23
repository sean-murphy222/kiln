"""3-stage metadata-filtered retrieval pipeline.

Orchestrates: Stage 1 (metadata filter) -> Stage 2 (semantic search)
-> Stage 3 (structural validation) to produce high-precision results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chonk.core.document import Chunk
from chonk.retrieval.filters import FilterCriteria, MetadataFilter
from chonk.retrieval.search import KeywordSearch, SearchProvider
from chonk.retrieval.validation import ResultValidator, ValidationResult, ValidationRule


@dataclass
class RetrievalConfig:
    """Configuration for the retrieval pipeline.

    Args:
        top_k: Maximum results to return.
        min_score: Minimum relevance score threshold.
        min_validation_score: Minimum validation score to keep.
    """

    top_k: int = 10
    min_score: float = 0.0
    min_validation_score: float = 0.0


@dataclass
class StageMetrics:
    """Performance metrics for a single pipeline stage.

    Args:
        stage: Stage name.
        input_count: Items entering this stage.
        output_count: Items leaving this stage.
        reduction_ratio: Fraction removed (0.0-1.0).
    """

    stage: str
    input_count: int
    output_count: int
    reduction_ratio: float


@dataclass
class RetrievalResult:
    """Result of the full 3-stage retrieval pipeline.

    Args:
        query: Original query string.
        results: Final validated results.
        stage_metrics: Per-stage performance metrics.
        total_reduction: Overall search space reduction.
    """

    query: str
    results: list[ValidationResult] = field(default_factory=list)
    stage_metrics: list[StageMetrics] = field(default_factory=list)
    total_reduction: float = 0.0

    @property
    def top_chunks(self) -> list[Chunk]:
        """Return just the chunks from results, ordered by score."""
        return [vr.chunk.chunk for vr in self.results]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "query": self.query,
            "result_count": len(self.results),
            "results": [
                {
                    "chunk_id": vr.chunk.chunk.id,
                    "search_score": vr.chunk.score,
                    "validation_score": vr.validation_score,
                    "adjusted_score": vr.adjusted_score,
                }
                for vr in self.results
            ],
            "stage_metrics": [
                {
                    "stage": sm.stage,
                    "input_count": sm.input_count,
                    "output_count": sm.output_count,
                    "reduction_ratio": sm.reduction_ratio,
                }
                for sm in self.stage_metrics
            ],
            "total_reduction": self.total_reduction,
        }


class RetrievalPipeline:
    """3-stage metadata-filtered retrieval pipeline.

    Stage 1: Deterministic metadata pre-filter (80-90% reduction).
    Stage 2: Semantic search on filtered subset.
    Stage 3: Structural validation and re-scoring.

    Args:
        search_provider: Semantic search implementation.
        validation_rules: Rules for Stage 3 validation.
        config: Pipeline configuration.

    Example::

        from chonk.retrieval.filters import FilterCriteria, FilterOperator
        from chonk.retrieval.search import KeywordSearch

        pipeline = RetrievalPipeline(
            search_provider=KeywordSearch(),
        )
        criteria = FilterCriteria()
        criteria.add("tm_number", FilterOperator.CONTAINS, "9-2320")
        result = pipeline.retrieve("How to replace air filter?", chunks, criteria)
        for vr in result.results:
            print(vr.chunk.chunk.content[:80])
    """

    def __init__(
        self,
        search_provider: SearchProvider | None = None,
        validation_rules: list[ValidationRule] | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._filter = MetadataFilter()
        self._search = search_provider or KeywordSearch()
        self._validator = ResultValidator(
            rules=validation_rules,
            min_validation_score=(config or RetrievalConfig()).min_validation_score,
        )
        self._config = config or RetrievalConfig()

    def retrieve(
        self,
        query: str,
        chunks: list[Chunk],
        criteria: FilterCriteria | None = None,
    ) -> RetrievalResult:
        """Execute the full 3-stage retrieval pipeline.

        Args:
            query: Natural language query.
            chunks: All available chunks.
            criteria: Stage 1 filter criteria. If None, skips filtering.

        Returns:
            RetrievalResult with validated results and metrics.
        """
        metrics: list[StageMetrics] = []
        total_input = len(chunks)

        # Stage 1: Metadata pre-filter
        if criteria and (criteria.conditions or criteria.document_type):
            filter_result = self._filter.filter(chunks, criteria)
            filtered = filter_result.passed
            metrics.append(
                StageMetrics(
                    stage="metadata_filter",
                    input_count=filter_result.total_input,
                    output_count=len(filtered),
                    reduction_ratio=filter_result.reduction_ratio,
                )
            )
        else:
            filtered = list(chunks)
            metrics.append(
                StageMetrics(
                    stage="metadata_filter",
                    input_count=total_input,
                    output_count=total_input,
                    reduction_ratio=0.0,
                )
            )

        # Stage 2: Semantic search
        search_results = self._search.search(query, filtered, self._config.top_k)
        if self._config.min_score > 0:
            search_results = [sr for sr in search_results if sr.score >= self._config.min_score]
        metrics.append(
            StageMetrics(
                stage="semantic_search",
                input_count=len(filtered),
                output_count=len(search_results),
                reduction_ratio=round(
                    1.0 - len(search_results) / len(filtered) if filtered else 0.0,
                    3,
                ),
            )
        )

        # Stage 3: Validation
        validated = self._validator.validate(search_results)
        metrics.append(
            StageMetrics(
                stage="validation",
                input_count=len(search_results),
                output_count=len(validated),
                reduction_ratio=round(
                    1.0 - len(validated) / len(search_results) if search_results else 0.0,
                    3,
                ),
            )
        )

        total_reduction = round(
            1.0 - len(validated) / total_input if total_input > 0 else 0.0,
            3,
        )

        return RetrievalResult(
            query=query,
            results=validated,
            stage_metrics=metrics,
            total_reduction=total_reduction,
        )
