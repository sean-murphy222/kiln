"""
Fast extractor (Tier 1).

Uses PyMuPDF + pdfplumber for extraction.
No GPU required, instant processing.
"""

from __future__ import annotations

from pathlib import Path

from chonk.core.document import Block, DocumentMetadata
from chonk.extraction.strategy import ExtractionResult, ExtractionTier
from chonk.loaders import LoaderRegistry


class FastExtractor:
    """
    Tier 1: Fast extraction using PyMuPDF + pdfplumber.

    This is the default extractor that works without GPU
    and processes documents instantly.

    Features:
    - PDF outline/TOC extraction
    - Font-based heading detection
    - Table extraction via pdfplumber
    - Bounding box information
    """

    def __init__(self) -> None:
        self._warnings: list[str] = []

    @property
    def tier(self) -> ExtractionTier:
        return ExtractionTier.FAST

    def is_available(self) -> bool:
        """Always available - uses core dependencies."""
        return True

    def extract(self, path: Path) -> ExtractionResult:
        """
        Extract content using fast (Tier 1) extraction.

        Args:
            path: Path to the document

        Returns:
            ExtractionResult with blocks and metadata
        """
        self._warnings = []

        # Use existing loader infrastructure
        document = LoaderRegistry.load_document(path)

        return ExtractionResult(
            blocks=document.blocks,
            metadata=document.metadata,
            tier_used=ExtractionTier.FAST,
            extraction_info={
                "loader_used": document.loader_used,
                "has_toc": document.metadata.custom.get("has_toc", False),
                "toc_entries": document.metadata.custom.get("toc_entry_count", 0),
            },
            warnings=self._warnings,
        )
