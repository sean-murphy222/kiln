"""QA filtering module for zero-value content detection.

Identifies and stamps boilerplate, TOCs, indices, distribution
statements, repetitive headers/footers, and other low-value content.
Filtered blocks are stamped with metadata (not removed) so chunkers
can skip them while preserving auditability.
"""

from chonk.qa.filter_log import FilterCategory, FilterLog, FilterLogEntry
from chonk.qa.filters import BlockFilter, FilterResult
from chonk.qa.patterns import CompiledPattern, match_patterns
from chonk.qa.rules import FilterRule, RuleSet

__all__ = [
    "BlockFilter",
    "CompiledPattern",
    "FilterCategory",
    "FilterLog",
    "FilterLogEntry",
    "FilterResult",
    "FilterRule",
    "RuleSet",
    "match_patterns",
]
