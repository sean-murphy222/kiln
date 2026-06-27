"""
Document analysis module.

Provides structure analysis and hierarchy extraction from documents
before chunking. This stage reconciles multiple signals (TOC, tags,
visual heuristics) into a unified document structure.
"""

from chonk.analysis.pdf_structure import PDFStructureExtractor
from chonk.analysis.structure import (
    DocumentStructure,
    StructureAnalyzer,
    StructureNode,
)

__all__ = [
    "DocumentStructure",
    "StructureNode",
    "StructureAnalyzer",
    "PDFStructureExtractor",
]
