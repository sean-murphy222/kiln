"""
Markdown document loader using markdown-it-py.

Parses Markdown into semantic blocks while preserving structure.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar

from markdown_it import MarkdownIt
from markdown_it.token import Token

from chonk.core.document import Block, BlockType, DocumentMetadata
from chonk.loaders.base import BaseLoader, LoaderError, LoaderRegistry


@LoaderRegistry.register
class MarkdownLoader(BaseLoader):
    """
    Load Markdown documents using markdown-it-py.

    Extracts:
    - Headings with levels
    - Paragraphs
    - Code blocks (with language info)
    - Lists
    - Tables
    - Blockquotes
    """

    SUPPORTED_EXTENSIONS: ClassVar[list[str]] = [".md", ".markdown", ".mdown"]
    LOADER_NAME: ClassVar[str] = "markdown"

    def __init__(self) -> None:
        super().__init__()
        self._md = MarkdownIt("commonmark", {"typographer": True})
        # Enable tables
        self._md.enable("table")

    def load(self, path: Path) -> tuple[list[Block], DocumentMetadata]:
        """Load a Markdown file and extract blocks and metadata."""
        self._reset_messages()

        try:
            content = path.read_text(encoding="utf-8")
            metadata = self._extract_metadata(content, path)
            blocks = self._extract_blocks(content)
            return blocks, metadata

        except Exception as e:
            raise LoaderError(
                f"Failed to load Markdown: {e}",
                source_path=path,
                details=str(e),
            ) from e

    def _extract_metadata(self, content: str, path: Path) -> DocumentMetadata:
        """Extract metadata from Markdown front matter and content."""
        title = None

        # Try to extract title from first H1
        h1_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if h1_match:
            title = h1_match.group(1).strip()

        # Try to extract from YAML front matter
        frontmatter = self._extract_frontmatter(content)
        if frontmatter:
            title = frontmatter.get("title", title)

        word_count = len(content.split())

        return DocumentMetadata(
            title=title,
            word_count=word_count,
            custom=frontmatter or {},
        )

    def _extract_frontmatter(self, content: str) -> dict[str, Any] | None:
        """Extract YAML front matter if present."""
        if not content.startswith("---"):
            return None

        # Find the closing ---
        end_match = re.search(r"\n---\s*\n", content[3:])
        if not end_match:
            return None

        frontmatter_text = content[3 : 3 + end_match.start()]

        # Simple YAML parsing (key: value pairs)
        result = {}
        for line in frontmatter_text.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                result[key.strip()] = value.strip().strip('"\'')

        return result

    def _extract_blocks(self, content: str) -> list[Block]:
        """Extract blocks from Markdown content."""
        # Remove front matter if present
        if content.startswith("---"):
            end_match = re.search(r"\n---\s*\n", content[3:])
            if end_match:
                content = content[3 + end_match.end() :]

        tokens = self._md.parse(content)
        blocks: list[Block] = []

        i = 0
        while i < len(tokens):
            token = tokens[i]
            block, skip = self._process_token(token, tokens, i)
            if block:
                blocks.append(block)
            i += skip + 1

        return blocks

    def _process_token(
        self, token: Token, tokens: list[Token], index: int
    ) -> tuple[Block | None, int]:
        """
        Process a token into a Block.

        Returns (block, tokens_to_skip).
        """
        if token.type == "heading_open":
            return self._process_heading(tokens, index)

        elif token.type == "paragraph_open":
            return self._process_paragraph(tokens, index)

        elif token.type == "fence":
            return self._process_code_fence(token)

        elif token.type == "code_block":
            return self._process_code_block(token)

        elif token.type == "bullet_list_open":
            return self._process_list(tokens, index, ordered=False)

        elif token.type == "ordered_list_open":
            return self._process_list(tokens, index, ordered=True)

        elif token.type == "table_open":
            return self._process_table(tokens, index)

        elif token.type == "blockquote_open":
            return self._process_blockquote(tokens, index)

        return None, 0

    def _process_heading(
        self, tokens: list[Token], index: int
    ) -> tuple[Block | None, int]:
        """Process a heading token group."""
        open_token = tokens[index]
        level = int(open_token.tag[1])  # h1 -> 1, h2 -> 2, etc.

        # Get inline content
        inline_token = tokens[index + 1]
        content = inline_token.content

        # Skip: heading_open, inline, heading_close
        return (
            Block(
                id=Block.generate_id(),
                type=BlockType.HEADING,
                content=content,
                heading_level=level,
                page=1,
            ),
            2,
        )

    def _process_paragraph(
        self, tokens: list[Token], index: int
    ) -> tuple[Block | None, int]:
        """Process a paragraph token group."""
        # Get inline content
        inline_token = tokens[index + 1]
        content = inline_token.content

        if not content.strip():
            return None, 2

        # Skip: paragraph_open, inline, paragraph_close
        return (
            Block(
                id=Block.generate_id(),
                type=BlockType.TEXT,
                content=content,
                page=1,
            ),
            2,
        )

    def _process_code_fence(self, token: Token) -> tuple[Block | None, int]:
        """Process a fenced code block."""
        content = token.content
        language = token.info.strip() if token.info else None

        return (
            Block(
                id=Block.generate_id(),
                type=BlockType.CODE,
                content=content,
                page=1,
                metadata={"language": language} if language else {},
            ),
            0,
        )

    def _process_code_block(self, token: Token) -> tuple[Block | None, int]:
        """Process an indented code block."""
        return (
            Block(
                id=Block.generate_id(),
                type=BlockType.CODE,
                content=token.content,
                page=1,
            ),
            0,
        )

    def _process_list(
        self, tokens: list[Token], index: int, ordered: bool
    ) -> tuple[Block | None, int]:
        """Process a list into a single block."""
        close_tag = "ordered_list_close" if ordered else "bullet_list_close"

        # Find the closing token
        depth = 1
        end_index = index + 1
        items = []

        while end_index < len(tokens) and depth > 0:
            t = tokens[end_index]
            if t.type in ("bullet_list_open", "ordered_list_open"):
                depth += 1
            elif t.type in ("bullet_list_close", "ordered_list_close"):
                depth -= 1
            elif t.type == "inline" and depth == 1:
                items.append(t.content)
            end_index += 1

        # Format as list
        if ordered:
            content = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
        else:
            content = "\n".join(f"- {item}" for item in items)

        return (
            Block(
                id=Block.generate_id(),
                type=BlockType.LIST,
                content=content,
                page=1,
                metadata={"ordered": ordered, "item_count": len(items)},
            ),
            end_index - index - 1,
        )

    def _process_table(
        self, tokens: list[Token], index: int
    ) -> tuple[Block | None, int]:
        """Process a table into a single block."""
        # Find the closing token
        end_index = index + 1
        while end_index < len(tokens) and tokens[end_index].type != "table_close":
            end_index += 1

        # Extract table content
        rows: list[list[str]] = []
        current_row: list[str] = []

        for i in range(index, end_index + 1):
            t = tokens[i]
            if t.type == "tr_open":
                current_row = []
            elif t.type == "tr_close":
                if current_row:
                    rows.append(current_row)
            elif t.type == "inline":
                current_row.append(t.content)

        # Format as markdown table
        content = self._table_to_markdown(rows)

        return (
            Block(
                id=Block.generate_id(),
                type=BlockType.TABLE,
                content=content,
                page=1,
                metadata={"rows": len(rows), "cols": len(rows[0]) if rows else 0},
            ),
            end_index - index,
        )

    def _table_to_markdown(self, rows: list[list[str]]) -> str:
        """Convert table rows to markdown format."""
        if not rows:
            return ""

        lines = []
        for i, row in enumerate(rows):
            line = "| " + " | ".join(row) + " |"
            lines.append(line)

            # Add header separator after first row
            if i == 0:
                separator = "| " + " | ".join(["---"] * len(row)) + " |"
                lines.append(separator)

        return "\n".join(lines)

    def _process_blockquote(
        self, tokens: list[Token], index: int
    ) -> tuple[Block | None, int]:
        """Process a blockquote into a block."""
        # Find the closing token
        depth = 1
        end_index = index + 1
        content_parts = []

        while end_index < len(tokens) and depth > 0:
            t = tokens[end_index]
            if t.type == "blockquote_open":
                depth += 1
            elif t.type == "blockquote_close":
                depth -= 1
            elif t.type == "inline" and depth == 1:
                content_parts.append(t.content)
            end_index += 1

        content = "\n".join(f"> {part}" for part in content_parts)

        return (
            Block(
                id=Block.generate_id(),
                type=BlockType.TEXT,
                content=content,
                page=1,
                metadata={"is_blockquote": True},
            ),
            end_index - index - 1,
        )
