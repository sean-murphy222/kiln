"""3-stage metadata-filtered retrieval pipeline.

Stage 1: Deterministic metadata pre-filter on enrichment fields.
Stage 2: Semantic search on filtered chunk subset.
Stage 3: Structural validation against expected patterns.
"""

from chonk.retrieval.filters import (
    FilterCondition,
    FilterCriteria,
    FilterOperator,
    FilterResult,
    MetadataFilter,
)
from chonk.retrieval.pipeline import (
    RetrievalConfig,
    RetrievalPipeline,
    RetrievalResult,
    StageMetrics,
)
from chonk.retrieval.search import KeywordSearch, ScoredChunk, SearchProvider
from chonk.retrieval.validation import (
    ResultValidator,
    ValidationCheck,
    ValidationResult,
    ValidationRule,
)

__all__ = [
    "FilterCondition",
    "FilterCriteria",
    "FilterOperator",
    "FilterResult",
    "KeywordSearch",
    "MetadataFilter",
    "ResultValidator",
    "RetrievalConfig",
    "RetrievalPipeline",
    "RetrievalResult",
    "ScoredChunk",
    "SearchProvider",
    "StageMetrics",
    "ValidationCheck",
    "ValidationResult",
    "ValidationRule",
]
