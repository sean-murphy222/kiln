"""Block content cleaning and normalization.

Applies text-level cleaning operations to block content: whitespace
normalization, continuation consolidation, formatting artifact removal,
and page marker stripping. Works on blocks in-place, preserving all
metadata and skipping QA-filtered blocks.
"""

from chonk.cleaning.normalizer import BlockNormalizer, CleaningResult

__all__ = [
    "BlockNormalizer",
    "CleaningResult",
]
