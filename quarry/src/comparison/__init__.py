"""
Comparison module - Compare chunking strategies side-by-side.

This is a core feature of CHONK - helping users choose the right chunking
strategy by showing concrete comparisons.
"""

from chonk.comparison.comparer import StrategyComparer, ComparisonResult
from chonk.comparison.metrics import ChunkingMetrics

__all__ = [
    "StrategyComparer",
    "ComparisonResult",
    "ChunkingMetrics",
]
