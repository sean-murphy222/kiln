"""
Document analysis module.

Provides structure analysis and hierarchy extraction from documents
before chunking. This stage reconciles multiple signals (TOC, tags,
visual heuristics) into a unified document structure.
"""

from chonk.analysis.structure import (
    DocumentStructure,
    StructureNode,
    StructureAnalyzer,
)
from chonk.analysis.pdf_structure import PDFStructureExtractor

__all__ = [
    "DocumentStructure",
    "StructureNode",
    "StructureAnalyzer",
    "PDFStructureExtractor",
]
