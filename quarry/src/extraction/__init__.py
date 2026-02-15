"""
Document extraction module.

Provides a tiered extraction system:
- Tier 1 (Fast): PyMuPDF + pdfplumber - no GPU, instant
- Tier 2 (Enhanced): Docling - better tables/formulas
- Tier 3 (AI): LayoutParser - complex layouts, scanned docs
"""

from chonk.extraction.strategy import (
    ExtractionResult,
    ExtractionStrategy,
    ExtractionTier,
    get_available_tiers,
    get_extractor,
)

__all__ = [
    "ExtractionResult",
    "ExtractionStrategy",
    "ExtractionTier",
    "get_available_tiers",
    "get_extractor",
]
