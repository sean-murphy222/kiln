"""
Plain text document loader.

Simple loader for .txt files that splits on paragraph boundaries.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

from chonk.core.document import Block, BlockType, DocumentMetadata
from chonk.loaders.base import BaseLoader, LoaderError, LoaderRegistry


@LoaderRegistry.register
class TextLoader(BaseLoader):
    """
    Load plain text documents.

    Splits text into paragraph blocks based on blank lines.
    """

    SUPPORTED_EXTENSIONS: ClassVar[list[str]] = [".txt", ".text"]
    LOADER_NAME: ClassVar[str] = "text"

    def load(self, path: Path) -> tuple[list[Block], DocumentMetadata]:
        """Load a text file and extract blocks and metadata."""
        self._reset_messages()

        try:
            content = path.read_text(encoding="utf-8")
            metadata = self._extract_metadata(content, path)
            blocks = self._extract_blocks(content)
            return blocks, metadata

        except UnicodeDecodeError:
            # Try with different encodings
            for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                try:
                    content = path.read_text(encoding=encoding)
                    self._add_warning(f"Used fallback encoding: {encoding}")
                    metadata = self._extract_metadata(content, path)
                    blocks = self._extract_blocks(content)
                    return blocks, metadata
                except UnicodeDecodeError:
                    continue

            raise LoaderError(
                "Could not decode text file with any supported encoding",
                source_path=path,
                details="Tried: utf-8, latin-1, cp1252, iso-8859-1",
            )

        except Exception as e:
            raise LoaderError(
                f"Failed to load text file: {e}",
                source_path=path,
                details=str(e),
            ) from e

    def _extract_metadata(self, content: str, path: Path) -> DocumentMetadata:
        """Extract basic metadata from text content."""
        lines = content.split("\n")

        # Use first non-empty line as title if it's short
        title = None
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) < 100:
                title = stripped
                break

        return DocumentMetadata(
            title=title,
            word_count=len(content.split()),
            page_count=1,
        )

    def _extract_blocks(self, content: str) -> list[Block]:
        """Split text into paragraph blocks."""
        blocks: list[Block] = []

        # Split on one or more blank lines
        paragraphs = re.split(r"\n\s*\n+", content)

        for para in paragraphs:
            text = para.strip()
            if not text:
                continue

            # Determine if this looks like a heading
            # (short line, no ending punctuation, possibly all caps)
            block_type = BlockType.TEXT
            heading_level = None

            lines = text.split("\n")
            if len(lines) == 1 and len(text) < 80:
                # Single short line
                if not text.endswith((".", ",", ";", ":", "!", "?")):
                    if text.isupper():
                        block_type = BlockType.HEADING
                        heading_level = 1
                    elif text[0].isupper():
                        block_type = BlockType.HEADING
                        heading_level = 2

            blocks.append(
                Block(
                    id=Block.generate_id(),
                    type=block_type,
                    content=text,
                    page=1,
                    heading_level=heading_level,
                )
            )

        return blocks
