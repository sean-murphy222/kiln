"""
Docling extractor (Tier 2).

Uses IBM's Docling for enhanced document extraction.
Better tables, formulas, and reading order detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chonk.core.document import (
    Block,
    BlockType,
    BoundingBox,
    DocumentMetadata,
)
from chonk.extraction.strategy import (
    ExtractionResult,
    ExtractionTier,
    _GPU_INCOMPATIBLE,
    _GPU_UPGRADE_MESSAGE,
)


class DoclingExtractor:
    """
    Tier 2: Enhanced extraction using IBM Docling.

    Provides significantly better extraction for:
    - Complex tables
    - Mathematical formulas
    - Reading order in multi-column layouts
    - Code blocks
    - Figures and captions

    Requires: pip install chonk[enhanced]
    """

    def __init__(self) -> None:
        self._warnings: list[str] = []
        self._docling_available = self._check_available()
        # Add warning if GPU was disabled due to incompatibility
        if _GPU_INCOMPATIBLE:
            if _GPU_UPGRADE_MESSAGE:
                self._warnings.append(_GPU_UPGRADE_MESSAGE)
            else:
                self._warnings.append(
                    "GPU architecture not supported by PyTorch. "
                    "Using CPU mode - extraction may be slower."
                )

    @property
    def tier(self) -> ExtractionTier:
        return ExtractionTier.ENHANCED

    def is_available(self) -> bool:
        """Check if Docling is installed."""
        return self._docling_available

    def _check_available(self) -> bool:
        """Check if Docling can be imported."""
        try:
            from docling.document_converter import DocumentConverter
            return True
        except ImportError:
            return False

    def extract(self, path: Path) -> ExtractionResult:
        """
        Extract content using Docling (Tier 2).

        Args:
            path: Path to the document

        Returns:
            ExtractionResult with blocks and metadata
        """
        # Reset warnings but preserve GPU warning if it exists
        gpu_warning = None
        if _GPU_INCOMPATIBLE:
            gpu_warning = (
                "GPU architecture not supported by PyTorch (e.g., RTX 50 series). "
                "Using CPU mode - extraction may be slower."
            )
        self._warnings = [gpu_warning] if gpu_warning else []

        if not self._docling_available:
            raise RuntimeError(
                "Docling is not installed. Install with: pip install chonk[enhanced]"
            )

        try:
            from docling.document_converter import DocumentConverter
        except ImportError as e:
            raise RuntimeError(f"Failed to import Docling: {e}")

        # Create converter with default options
        # Docling 2.64+ uses simpler API - format_options removed in favor of defaults
        converter = DocumentConverter()

        # Convert document
        try:
            result = converter.convert(str(path))
            doc = result.document
        except Exception as e:
            self._warnings.append(f"Docling conversion error: {e}")
            # Fall back to fast extraction
            from chonk.extraction.fast_extractor import FastExtractor
            return FastExtractor().extract(path)

        # Convert Docling output to CHONK blocks
        blocks = self._convert_docling_to_blocks(doc)
        metadata = self._extract_metadata(doc, path)

        return ExtractionResult(
            blocks=blocks,
            metadata=metadata,
            tier_used=ExtractionTier.ENHANCED,
            extraction_info={
                "extractor": "docling",
                "docling_version": self._get_docling_version(),
                "element_count": len(blocks),
            },
            warnings=self._warnings,
        )

    def _convert_docling_to_blocks(self, doc: Any) -> list[Block]:
        """Convert Docling document to CHONK blocks."""
        blocks: list[Block] = []
        block_id = 0

        # Iterate through Docling's document structure
        try:
            # Docling provides elements through iterate_items()
            for item, level in doc.iterate_items():
                block = self._convert_item_to_block(item, level, block_id)
                if block:
                    blocks.append(block)
                    block_id += 1
        except AttributeError:
            # Fallback for different Docling versions
            try:
                # Try accessing through export
                md_content = doc.export_to_markdown()
                blocks = self._parse_markdown_to_blocks(md_content)
            except Exception as e:
                self._warnings.append(f"Failed to extract Docling content: {e}")

        return blocks

    def _convert_item_to_block(
        self, item: Any, level: int, block_id: int
    ) -> Block | None:
        """Convert a single Docling item to a CHONK block."""
        try:
            # Get item type and content
            item_type = type(item).__name__.lower()
            content = ""
            block_type = BlockType.TEXT
            heading_level = None
            bbox = None
            page = 1

            # Handle different Docling item types
            if hasattr(item, "text"):
                content = item.text
            elif hasattr(item, "export_to_markdown"):
                content = item.export_to_markdown()

            # Determine block type
            # Docling uses "sectionheaderitem" for headings
            if "heading" in item_type or "title" in item_type or "header" in item_type:
                block_type = BlockType.HEADING
                heading_level = min(level, 6) if level > 0 else 1
            elif "table" in item_type:
                block_type = BlockType.TABLE
                if hasattr(item, "export_to_markdown"):
                    content = item.export_to_markdown()
            elif "code" in item_type:
                block_type = BlockType.CODE
            elif "list" in item_type:
                block_type = BlockType.LIST
            elif "formula" in item_type or "equation" in item_type:
                block_type = BlockType.CODE  # Treat formulas as code for now
                content = f"$${content}$$" if content else ""
            elif "figure" in item_type or "image" in item_type:
                block_type = BlockType.IMAGE
                content = item.caption if hasattr(item, "caption") else "[Figure]"

            # Extract bounding box if available
            if hasattr(item, "prov") and item.prov:
                prov = item.prov[0] if isinstance(item.prov, list) else item.prov
                if hasattr(prov, "bbox"):
                    b = prov.bbox
                    bbox = BoundingBox(
                        x1=b.l if hasattr(b, "l") else 0,
                        y1=b.t if hasattr(b, "t") else 0,
                        x2=b.r if hasattr(b, "r") else 0,
                        y2=b.b if hasattr(b, "b") else 0,
                        page=prov.page_no if hasattr(prov, "page_no") else 1,
                    )
                    page = bbox.page

            if not content or not content.strip():
                return None

            return Block(
                id=f"docling_blk_{block_id}",
                type=block_type,
                content=content.strip(),
                bbox=bbox,
                page=page,
                heading_level=heading_level,
                metadata={
                    "source": "docling",
                    "docling_type": item_type,
                    "level": level,
                },
            )

        except Exception as e:
            self._warnings.append(f"Failed to convert Docling item: {e}")
            return None

    def _parse_markdown_to_blocks(self, md_content: str) -> list[Block]:
        """Parse markdown content into blocks (fallback method)."""
        import re

        blocks: list[Block] = []
        lines = md_content.split("\n")
        current_block = []
        current_type = BlockType.TEXT
        block_id = 0

        for line in lines:
            # Check for headings
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if heading_match:
                # Save current block
                if current_block:
                    blocks.append(Block(
                        id=f"docling_blk_{block_id}",
                        type=current_type,
                        content="\n".join(current_block).strip(),
                        page=1,
                    ))
                    block_id += 1
                    current_block = []

                # Create heading block
                level = len(heading_match.group(1))
                blocks.append(Block(
                    id=f"docling_blk_{block_id}",
                    type=BlockType.HEADING,
                    content=heading_match.group(2),
                    page=1,
                    heading_level=level,
                ))
                block_id += 1
                current_type = BlockType.TEXT
                continue

            # Check for code blocks
            if line.startswith("```"):
                if current_type == BlockType.CODE:
                    # End code block
                    blocks.append(Block(
                        id=f"docling_blk_{block_id}",
                        type=BlockType.CODE,
                        content="\n".join(current_block).strip(),
                        page=1,
                    ))
                    block_id += 1
                    current_block = []
                    current_type = BlockType.TEXT
                else:
                    # Start code block
                    if current_block:
                        blocks.append(Block(
                            id=f"docling_blk_{block_id}",
                            type=current_type,
                            content="\n".join(current_block).strip(),
                            page=1,
                        ))
                        block_id += 1
                        current_block = []
                    current_type = BlockType.CODE
                continue

            # Check for table rows
            if line.startswith("|") and "|" in line[1:]:
                if current_type != BlockType.TABLE:
                    if current_block:
                        blocks.append(Block(
                            id=f"docling_blk_{block_id}",
                            type=current_type,
                            content="\n".join(current_block).strip(),
                            page=1,
                        ))
                        block_id += 1
                        current_block = []
                    current_type = BlockType.TABLE
                current_block.append(line)
                continue

            # Empty line - potentially end current block
            if not line.strip():
                if current_type == BlockType.TABLE:
                    blocks.append(Block(
                        id=f"docling_blk_{block_id}",
                        type=BlockType.TABLE,
                        content="\n".join(current_block).strip(),
                        page=1,
                    ))
                    block_id += 1
                    current_block = []
                    current_type = BlockType.TEXT
                elif current_block:
                    current_block.append(line)
                continue

            # Regular text
            if current_type == BlockType.TABLE:
                # End table if we get non-table content
                blocks.append(Block(
                    id=f"docling_blk_{block_id}",
                    type=BlockType.TABLE,
                    content="\n".join(current_block).strip(),
                    page=1,
                ))
                block_id += 1
                current_block = []
                current_type = BlockType.TEXT

            current_block.append(line)

        # Save final block
        if current_block:
            content = "\n".join(current_block).strip()
            if content:
                blocks.append(Block(
                    id=f"docling_blk_{block_id}",
                    type=current_type,
                    content=content,
                    page=1,
                ))

        return blocks

    def _extract_metadata(self, doc: Any, path: Path) -> DocumentMetadata:
        """Extract metadata from Docling document."""
        metadata = DocumentMetadata()

        try:
            # Try to get metadata from Docling document
            if hasattr(doc, "metadata"):
                dm = doc.metadata
                metadata.title = getattr(dm, "title", None)
                metadata.author = getattr(dm, "author", None)

            # Count pages
            if hasattr(doc, "pages"):
                metadata.page_count = len(doc.pages)
            elif hasattr(doc, "num_pages"):
                metadata.page_count = doc.num_pages

            # Get file size
            metadata.file_size_bytes = path.stat().st_size

            # Count words from content
            if hasattr(doc, "export_to_markdown"):
                content = doc.export_to_markdown()
                metadata.word_count = len(content.split())

        except Exception as e:
            self._warnings.append(f"Failed to extract Docling metadata: {e}")

        # Add extraction info
        metadata.custom["extractor"] = "docling"
        metadata.custom["tier"] = "enhanced"

        return metadata

    def _get_docling_version(self) -> str:
        """Get installed Docling version."""
        try:
            import docling
            return getattr(docling, "__version__", "unknown")
        except Exception:
            return "unknown"
