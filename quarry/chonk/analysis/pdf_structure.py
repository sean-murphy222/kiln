"""
PDF-specific structure extraction using PyMuPDF (fitz).

Extracts:
- Document outline (TOC/bookmarks)
- Tagged structure (PDF/UA)
- Font analysis for heading detection
- Reading order
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from chonk.analysis.structure import (
    DocumentStructure,
    StructureNode,
    StructureSource,
)
from chonk.core.document import Block, BlockType


@dataclass
class FontInfo:
    """Information about a font used in the document."""

    name: str
    size: float
    flags: int  # Bold, italic, etc.
    usage_count: int = 0

    @property
    def is_bold(self) -> bool:
        return bool(self.flags & 2**4)  # Bit 4 = bold

    @property
    def is_italic(self) -> bool:
        return bool(self.flags & 2**1)  # Bit 1 = italic


@dataclass
class PDFAnalysisResult:
    """Result of PDF structure analysis."""

    # Table of Contents from PDF outline
    toc_entries: list[dict[str, Any]] = field(default_factory=list)
    has_toc: bool = False

    # Tagged structure
    structure_tree: dict[str, Any] | None = None
    is_tagged: bool = False

    # Font analysis
    fonts: dict[str, FontInfo] = field(default_factory=dict)
    body_font_size: float = 12.0
    heading_fonts: set[str] = field(default_factory=set)

    # Page-level info
    page_count: int = 0
    page_labels: dict[int, str] = field(default_factory=dict)

    # Cross-references and links
    internal_links: list[dict[str, Any]] = field(default_factory=list)


class PDFStructureExtractor:
    """
    Extract document structure from PDF using PyMuPDF.

    This provides higher-fidelity structure information than
    pdfplumber alone, including:
    - PDF outline/bookmarks (Table of Contents)
    - Tagged PDF structure tree
    - Font statistics for heading detection
    """

    def __init__(self) -> None:
        self._warnings: list[str] = []

    def extract(self, path: Path) -> PDFAnalysisResult:
        """
        Extract all structure information from a PDF.

        Args:
            path: Path to the PDF file

        Returns:
            PDFAnalysisResult with all extracted structure info
        """
        self._warnings = []
        result = PDFAnalysisResult()

        try:
            doc = fitz.open(path)

            result.page_count = len(doc)

            # Extract TOC/outline
            self._extract_toc(doc, result)

            # Analyze fonts across document
            self._analyze_fonts(doc, result)

            # Extract tagged structure if available
            self._extract_tagged_structure(doc, result)

            # Extract internal links (cross-references)
            self._extract_links(doc, result)

            # Extract page labels if present
            self._extract_page_labels(doc, result)

            doc.close()

        except Exception as e:
            self._warnings.append(f"PDF structure extraction error: {e}")

        return result

    def _extract_toc(self, doc: fitz.Document, result: PDFAnalysisResult) -> None:
        """Extract table of contents from PDF outline/bookmarks."""
        try:
            toc = doc.get_toc(simple=False)  # Get full TOC with destinations

            if toc:
                result.has_toc = True
                result.toc_entries = []

                for entry in toc:
                    level = entry[0]  # Nesting level (1 = top)
                    title = entry[1]  # Bookmark title
                    page = entry[2]  # Target page (1-indexed in fitz)
                    dest = entry[3] if len(entry) > 3 else None  # Destination details

                    result.toc_entries.append({
                        "level": level,
                        "title": title,
                        "page": page,
                        "destination": dest,
                    })

        except Exception as e:
            self._warnings.append(f"TOC extraction error: {e}")

    def _analyze_fonts(self, doc: fitz.Document, result: PDFAnalysisResult) -> None:
        """
        Analyze fonts used in the document.

        Identifies body text font and potential heading fonts.
        """
        font_usage: dict[str, dict[str, Any]] = {}

        # Sample pages (first 20 or all if less)
        sample_pages = min(20, len(doc))

        for page_num in range(sample_pages):
            page = doc[page_num]

            # Get text blocks with font info
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

            for block in blocks.get("blocks", []):
                if block.get("type") != 0:  # 0 = text block
                    continue

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font_name = span.get("font", "unknown")
                        font_size = round(span.get("size", 12), 1)
                        flags = span.get("flags", 0)
                        text_len = len(span.get("text", ""))

                        key = f"{font_name}_{font_size}"
                        if key not in font_usage:
                            font_usage[key] = {
                                "name": font_name,
                                "size": font_size,
                                "flags": flags,
                                "char_count": 0,
                            }
                        font_usage[key]["char_count"] += text_len

        # Find body font (most used)
        if font_usage:
            body_font_key = max(font_usage, key=lambda k: font_usage[k]["char_count"])
            body_info = font_usage[body_font_key]
            result.body_font_size = body_info["size"]

            # Build font info dict
            for key, info in font_usage.items():
                result.fonts[key] = FontInfo(
                    name=info["name"],
                    size=info["size"],
                    flags=info["flags"],
                    usage_count=info["char_count"],
                )

                # Identify likely heading fonts (larger than body)
                if info["size"] > result.body_font_size * 1.15:
                    result.heading_fonts.add(key)

    def _extract_tagged_structure(
        self, doc: fitz.Document, result: PDFAnalysisResult
    ) -> None:
        """
        Extract tagged structure from PDF/UA documents.

        Tagged PDFs have semantic structure information.
        """
        try:
            # Check if document is tagged
            # PyMuPDF exposes this through metadata
            metadata = doc.metadata
            if metadata:
                # Some PDFs indicate tagging in metadata
                pass

            # Try to access structure tree
            # Note: Full structure tree access in PyMuPDF is limited
            # We check for basic indicators

            # Check for marked content
            for page_num in range(min(5, len(doc))):
                page = doc[page_num]

                # Get structure elements if available
                # This is a simplified check - full implementation
                # would parse the structure tree

                text_dict = page.get_text("dict")
                if text_dict.get("blocks"):
                    for block in text_dict["blocks"]:
                        # Look for structure indicators
                        if block.get("type") == 0:  # Text
                            # Tagged content often has specific markers
                            pass

            # For now, mark as not tagged unless we have clear evidence
            result.is_tagged = False

        except Exception as e:
            self._warnings.append(f"Tagged structure extraction error: {e}")

    def _extract_links(self, doc: fitz.Document, result: PDFAnalysisResult) -> None:
        """Extract internal links (cross-references)."""
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]

                for link in page.get_links():
                    if link.get("kind") == fitz.LINK_GOTO:  # Internal link
                        result.internal_links.append({
                            "from_page": page_num + 1,
                            "to_page": link.get("page", 0) + 1,
                            "rect": link.get("from"),
                        })

        except Exception as e:
            self._warnings.append(f"Link extraction error: {e}")

    def _extract_page_labels(
        self, doc: fitz.Document, result: PDFAnalysisResult
    ) -> None:
        """Extract page labels (e.g., 'i', 'ii', '1', '2')."""
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                label = page.get_label()
                if label:
                    result.page_labels[page_num + 1] = label

        except Exception as e:
            self._warnings.append(f"Page label extraction error: {e}")

    def build_structure_tree(
        self, result: PDFAnalysisResult
    ) -> StructureNode | None:
        """
        Build a StructureNode tree from TOC entries.

        Returns None if no TOC is available.
        """
        if not result.has_toc or not result.toc_entries:
            return None

        # Create root node
        root = StructureNode(
            id="toc_root",
            title="",
            level=0,
            source=StructureSource.TOC,
            confidence=0.95,
        )

        # Stack for tracking current position
        stack: list[StructureNode] = [root]

        for i, entry in enumerate(result.toc_entries):
            level = entry["level"]
            title = entry["title"]
            page = entry["page"]

            # Find parent - pop until we're at the right level
            while len(stack) > level:
                stack.pop()

            # Ensure stack has enough depth
            while len(stack) < level:
                # Create placeholder parent if needed
                placeholder = StructureNode(
                    id=f"toc_placeholder_{len(stack)}_{i}",
                    title="",
                    level=len(stack),
                    source=StructureSource.TOC,
                    confidence=0.5,
                )
                stack[-1].add_child(placeholder)
                stack.append(placeholder)

            # Create node for this entry
            node = StructureNode(
                id=f"toc_{i}",
                title=title,
                level=level,
                source=StructureSource.TOC,
                confidence=0.95,
                page_start=page,
            )

            stack[-1].add_child(node)
            stack.append(node)

        # Compute page_end for each node
        self._compute_page_ends(root, result.page_count)

        return root

    def _compute_page_ends(self, node: StructureNode, total_pages: int) -> int:
        """
        Compute page_end for each node based on next sibling/parent.

        Returns the page where this node's content ends.
        """
        if not node.children:
            # Leaf node - page_end will be set by parent
            return node.page_start or total_pages

        # Process children
        for i, child in enumerate(node.children):
            if i + 1 < len(node.children):
                # Next sibling exists - this node ends where next starts
                next_sibling = node.children[i + 1]
                child.page_end = (next_sibling.page_start or total_pages) - 1
            else:
                # Last child - ends where parent ends
                child.page_end = node.page_end or total_pages

            # Recurse
            self._compute_page_ends(child, total_pages)

        return node.page_end or total_pages

    def enhance_blocks_with_structure(
        self,
        blocks: list[Block],
        result: PDFAnalysisResult,
    ) -> list[Block]:
        """
        Enhance blocks with structure information.

        Updates block metadata with:
        - Heading confidence based on font analysis
        - TOC match information
        """
        toc_titles = {
            self._normalize_title(e["title"]): e
            for e in result.toc_entries
        } if result.has_toc else {}

        for block in blocks:
            # Check if this block matches a TOC entry
            if block.type == BlockType.HEADING:
                normalized = self._normalize_title(block.content)
                if normalized in toc_titles:
                    toc_entry = toc_titles[normalized]
                    block.metadata["toc_match"] = True
                    block.metadata["toc_level"] = toc_entry["level"]
                    block.metadata["toc_page"] = toc_entry["page"]

                    # Update heading level from TOC if not set
                    if not block.heading_level:
                        block.heading_level = toc_entry["level"]

            # Add font analysis info
            font_size = block.metadata.get("avg_font_size", 12)
            if font_size > result.body_font_size * 1.15:
                block.metadata["likely_heading"] = True
                block.metadata["size_ratio"] = font_size / result.body_font_size

        return blocks

    def _normalize_title(self, title: str) -> str:
        """Normalize a title for comparison."""
        import re
        title = title.lower().strip()
        title = re.sub(r"^\d+(\.\d+)*\.?\s*", "", title)  # Remove numbering
        title = re.sub(r"\s+", " ", title)
        return title
