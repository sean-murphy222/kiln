"""
PDF document loader using pdfplumber and PyMuPDF.

Extracts text, tables, and metadata from PDF files with full
bounding box information for visual overlay.

Uses PyMuPDF for:
- Document outline/TOC extraction
- Font analysis
- Tagged structure detection

Uses pdfplumber for:
- Text extraction with bounding boxes
- Table detection and extraction
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import pdfplumber
from pdfplumber.page import Page

from chonk.analysis.pdf_structure import PDFStructureExtractor, PDFAnalysisResult
from chonk.core.document import Block, BlockType, BoundingBox, DocumentMetadata
from chonk.loaders.base import BaseLoader, LoaderError, LoaderRegistry


@LoaderRegistry.register
class PDFLoader(BaseLoader):
    """
    Load PDF documents using pdfplumber and PyMuPDF.

    Extracts:
    - Text blocks with bounding boxes
    - Headings (detected by font size, style, and TOC matching)
    - Tables with structure preserved
    - Document metadata
    - Document outline/TOC for structure
    """

    SUPPORTED_EXTENSIONS: ClassVar[list[str]] = [".pdf"]
    LOADER_NAME: ClassVar[str] = "pdf_pdfplumber"

    # Thresholds for heading detection
    HEADING_SIZE_RATIO = 1.2  # Font size must be 20% larger than body text
    MIN_HEADING_SIZE = 12  # Minimum font size to consider as heading

    def __init__(self) -> None:
        super().__init__()
        self._body_font_size: float = 12.0  # Will be computed per document
        self._structure_extractor = PDFStructureExtractor()
        self._pdf_analysis: PDFAnalysisResult | None = None

    def load(self, path: Path) -> tuple[list[Block], DocumentMetadata]:
        """Load a PDF and extract blocks and metadata."""
        self._reset_messages()

        try:
            # First, extract structure using PyMuPDF
            self._pdf_analysis = self._structure_extractor.extract(path)

            # Use PyMuPDF's font analysis for body font size
            if self._pdf_analysis.body_font_size > 0:
                self._body_font_size = self._pdf_analysis.body_font_size

            # Log TOC info
            if self._pdf_analysis.has_toc:
                self._add_info(
                    f"Found TOC with {len(self._pdf_analysis.toc_entries)} entries"
                )

            # Extract content using pdfplumber
            with pdfplumber.open(path) as pdf:
                metadata = self._extract_metadata(pdf, path)
                blocks = self._extract_blocks(pdf)

                # Enhance blocks with structure information
                if self._pdf_analysis:
                    blocks = self._structure_extractor.enhance_blocks_with_structure(
                        blocks, self._pdf_analysis
                    )

                # Store structure info in metadata for chunker
                metadata.custom["has_toc"] = self._pdf_analysis.has_toc
                metadata.custom["toc_entry_count"] = len(
                    self._pdf_analysis.toc_entries
                )
                metadata.custom["is_tagged_pdf"] = self._pdf_analysis.is_tagged

                return blocks, metadata

        except Exception as e:
            raise LoaderError(
                f"Failed to load PDF: {e}",
                source_path=path,
                details=str(e),
            ) from e

    def get_toc_structure(self):
        """
        Get the TOC-based structure tree if available.

        Call this after load() to get the document outline.
        """
        if self._pdf_analysis and self._pdf_analysis.has_toc:
            return self._structure_extractor.build_structure_tree(self._pdf_analysis)
        return None

    def _extract_metadata(self, pdf: pdfplumber.PDF, path: Path) -> DocumentMetadata:
        """Extract document metadata from PDF."""
        info = pdf.metadata or {}

        # Parse dates (PDF date format: D:YYYYMMDDHHmmSS)
        created = self._parse_pdf_date(info.get("CreationDate"))
        modified = self._parse_pdf_date(info.get("ModDate"))

        # Count total words across all pages
        word_count = 0
        for page in pdf.pages:
            text = page.extract_text() or ""
            word_count += len(text.split())

        return DocumentMetadata(
            title=info.get("Title"),
            author=info.get("Author"),
            subject=info.get("Subject"),
            keywords=self._parse_keywords(info.get("Keywords")),
            created_date=created,
            modified_date=modified,
            page_count=len(pdf.pages),
            word_count=word_count,
            custom={
                "producer": info.get("Producer"),
                "creator": info.get("Creator"),
            },
        )

    def _parse_pdf_date(self, date_str: str | None) -> datetime | None:
        """Parse PDF date format (D:YYYYMMDDHHmmSS) to datetime."""
        if not date_str:
            return None

        # Remove the D: prefix if present
        if date_str.startswith("D:"):
            date_str = date_str[2:]

        # Try common PDF date formats
        formats = [
            "%Y%m%d%H%M%S",
            "%Y%m%d%H%M%S%z",
            "%Y%m%d%H%M",
            "%Y%m%d",
        ]

        # Clean timezone info (often malformed in PDFs)
        date_str = re.sub(r"[+-]\d{2}'\d{2}'?$", "", date_str)
        date_str = re.sub(r"Z$", "", date_str)

        for fmt in formats:
            try:
                return datetime.strptime(date_str[:len(fmt.replace("%", ""))], fmt)
            except ValueError:
                continue

        self._add_warning(f"Could not parse date: {date_str}")
        return None

    def _parse_keywords(self, keywords_str: str | None) -> list[str]:
        """Parse keywords string into list."""
        if not keywords_str:
            return []
        # Keywords can be comma or semicolon separated
        keywords = re.split(r"[,;]", keywords_str)
        return [k.strip() for k in keywords if k.strip()]

    def _extract_blocks(self, pdf: pdfplumber.PDF) -> list[Block]:
        """Extract all blocks from PDF pages."""
        blocks: list[Block] = []

        # First pass: determine typical body font size
        self._body_font_size = self._compute_body_font_size(pdf)

        for page_num, page in enumerate(pdf.pages, start=1):
            page_blocks = self._extract_page_blocks(page, page_num)
            blocks.extend(page_blocks)

        return blocks

    def _compute_body_font_size(self, pdf: pdfplumber.PDF) -> float:
        """
        Compute the most common font size (assumed to be body text).

        This helps us identify headings as text larger than body.
        """
        font_sizes: dict[float, int] = {}

        for page in pdf.pages[:10]:  # Sample first 10 pages
            chars = page.chars or []
            for char in chars:
                size = round(char.get("size", 12), 1)
                font_sizes[size] = font_sizes.get(size, 0) + 1

        if not font_sizes:
            return 12.0

        # Return the most common font size
        return max(font_sizes, key=lambda s: font_sizes[s])

    def _extract_page_blocks(self, page: Page, page_num: int) -> list[Block]:
        """Extract blocks from a single page."""
        blocks: list[Block] = []

        # Extract tables first (so we can exclude table regions from text)
        table_regions = self._extract_tables(page, page_num, blocks)

        # Extract text blocks, excluding table regions
        self._extract_text_blocks(page, page_num, blocks, table_regions)

        # Sort blocks by vertical position (top to bottom)
        blocks.sort(key=lambda b: (b.bbox.y1 if b.bbox else 0, b.bbox.x1 if b.bbox else 0))

        return blocks

    def _extract_tables(
        self, page: Page, page_num: int, blocks: list[Block]
    ) -> list[tuple[float, float, float, float]]:
        """
        Extract tables from a page and return their bounding regions.

        Returns list of (x0, y0, x1, y1) tuples for table regions.
        """
        table_regions = []

        try:
            tables = page.find_tables()
            for table in tables:
                bbox = table.bbox  # (x0, y0, x1, y1)
                table_regions.append(bbox)

                # Extract table data
                table_data = table.extract()
                if table_data:
                    # Convert to markdown-style table
                    content = self._table_to_markdown(table_data)

                    block = Block(
                        id=Block.generate_id(),
                        type=BlockType.TABLE,
                        content=content,
                        bbox=BoundingBox(
                            x1=bbox[0],
                            y1=bbox[1],
                            x2=bbox[2],
                            y2=bbox[3],
                            page=page_num,
                        ),
                        page=page_num,
                        metadata={
                            "rows": len(table_data),
                            "cols": len(table_data[0]) if table_data else 0,
                        },
                    )
                    blocks.append(block)

        except Exception as e:
            self._add_warning(f"Table extraction error on page {page_num}: {e}")

        return table_regions

    def _table_to_markdown(self, table_data: list[list[str | None]]) -> str:
        """Convert table data to markdown format."""
        if not table_data:
            return ""

        lines = []
        for i, row in enumerate(table_data):
            # Clean cells
            cells = [str(cell or "").strip().replace("\n", " ") for cell in row]
            line = "| " + " | ".join(cells) + " |"
            lines.append(line)

            # Add header separator after first row
            if i == 0:
                separator = "| " + " | ".join(["---"] * len(cells)) + " |"
                lines.append(separator)

        return "\n".join(lines)

    def _extract_text_blocks(
        self,
        page: Page,
        page_num: int,
        blocks: list[Block],
        table_regions: list[tuple[float, float, float, float]],
    ) -> None:
        """Extract text blocks from a page, excluding table regions."""
        # Get text with layout information
        words = page.extract_words(
            keep_blank_chars=True,
            x_tolerance=3,
            y_tolerance=3,
            extra_attrs=["fontname", "size"],
        )

        if not words:
            return

        # Filter out words that are inside table regions
        words = [w for w in words if not self._point_in_regions(w["x0"], w["top"], table_regions)]

        # Group words into lines
        lines = self._group_words_into_lines(words)

        # Group lines into paragraphs/blocks
        current_block_lines: list[dict[str, Any]] = []
        current_block_type = BlockType.TEXT

        for line in lines:
            line_type = self._classify_line(line)

            # Check if this line starts a new block
            if self._should_start_new_block(current_block_lines, line, line_type, current_block_type):
                # Save current block if it has content
                if current_block_lines:
                    block = self._create_block_from_lines(
                        current_block_lines, page_num, current_block_type
                    )
                    if block:
                        blocks.append(block)

                current_block_lines = [line]
                current_block_type = line_type
            else:
                current_block_lines.append(line)

        # Save final block
        if current_block_lines:
            block = self._create_block_from_lines(
                current_block_lines, page_num, current_block_type
            )
            if block:
                blocks.append(block)

    def _point_in_regions(
        self, x: float, y: float, regions: list[tuple[float, float, float, float]]
    ) -> bool:
        """Check if a point is inside any of the given regions."""
        for x0, y0, x1, y1 in regions:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return True
        return False

    def _group_words_into_lines(self, words: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Group words into lines based on vertical position."""
        if not words:
            return []

        # Sort by vertical position, then horizontal
        words = sorted(words, key=lambda w: (w["top"], w["x0"]))

        lines = []
        current_line_words: list[dict[str, Any]] = [words[0]]
        current_top = words[0]["top"]

        for word in words[1:]:
            # Same line if vertical position is similar
            if abs(word["top"] - current_top) < 5:
                current_line_words.append(word)
            else:
                # New line
                lines.append(self._merge_line_words(current_line_words))
                current_line_words = [word]
                current_top = word["top"]

        # Don't forget the last line
        if current_line_words:
            lines.append(self._merge_line_words(current_line_words))

        return lines

    def _merge_line_words(self, words: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge words in a line into a single line dict."""
        words = sorted(words, key=lambda w: w["x0"])

        # Compute average font size for the line
        sizes = [w.get("size", 12) for w in words]
        avg_size = sum(sizes) / len(sizes) if sizes else 12

        # Check if line appears to be bold (heuristic)
        fontnames = [w.get("fontname", "") for w in words]
        is_bold = any("bold" in fn.lower() for fn in fontnames)

        return {
            "text": " ".join(w["text"] for w in words),
            "x0": words[0]["x0"],
            "x1": words[-1]["x1"],
            "top": min(w["top"] for w in words),
            "bottom": max(w["bottom"] for w in words),
            "size": avg_size,
            "is_bold": is_bold,
        }

    def _classify_line(self, line: dict[str, Any]) -> BlockType:
        """Classify a line as heading, text, etc."""
        text = line["text"].strip()
        size = line.get("size", 12)
        is_bold = line.get("is_bold", False)

        # Empty or very short lines
        if len(text) < 2:
            return BlockType.TEXT

        # Check for heading based on font size
        if size >= self._body_font_size * self.HEADING_SIZE_RATIO:
            return BlockType.HEADING

        # Check for bold short lines (likely headings)
        if is_bold and len(text) < 100 and not text.endswith((".", ",", ";")):
            return BlockType.HEADING

        # Check for numbered headings (e.g., "1. Introduction", "1.1 Overview")
        if re.match(r"^\d+(\.\d+)*\.?\s+[A-Z]", text):
            return BlockType.HEADING

        # Check for list items
        if re.match(r"^[\u2022\u2023\u25E6\u2043\u2219•\-\*]\s", text):
            return BlockType.LIST_ITEM

        return BlockType.TEXT

    def _should_start_new_block(
        self,
        current_lines: list[dict[str, Any]],
        new_line: dict[str, Any],
        new_type: BlockType,
        current_type: BlockType,
    ) -> bool:
        """Determine if a new line should start a new block."""
        if not current_lines:
            return True

        # Type change always starts new block
        if new_type != current_type:
            return True

        # Headings are always their own block
        if new_type == BlockType.HEADING:
            return True

        # Large vertical gap starts new block
        last_line = current_lines[-1]
        gap = new_line["top"] - last_line["bottom"]
        line_height = last_line["bottom"] - last_line["top"]

        if gap > line_height * 1.5:
            return True

        return False

    def _create_block_from_lines(
        self, lines: list[dict[str, Any]], page_num: int, block_type: BlockType
    ) -> Block | None:
        """Create a Block from a list of lines."""
        if not lines:
            return None

        # Combine line text
        content = "\n".join(line["text"] for line in lines).strip()
        if not content:
            return None

        # Compute bounding box
        x1 = min(line["x0"] for line in lines)
        y1 = min(line["top"] for line in lines)
        x2 = max(line["x1"] for line in lines)
        y2 = max(line["bottom"] for line in lines)

        # Determine heading level if it's a heading
        heading_level = None
        if block_type == BlockType.HEADING:
            avg_size = sum(line.get("size", 12) for line in lines) / len(lines)
            heading_level = self._size_to_heading_level(avg_size)

        return Block(
            id=Block.generate_id(),
            type=block_type,
            content=content,
            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, page=page_num),
            page=page_num,
            heading_level=heading_level,
            metadata={
                "avg_font_size": sum(line.get("size", 12) for line in lines) / len(lines),
            },
        )

    def _size_to_heading_level(self, size: float) -> int:
        """Convert font size to heading level (1-6)."""
        ratio = size / self._body_font_size

        if ratio >= 2.0:
            return 1
        elif ratio >= 1.7:
            return 2
        elif ratio >= 1.4:
            return 3
        elif ratio >= 1.2:
            return 4
        elif ratio >= 1.1:
            return 5
        else:
            return 6
