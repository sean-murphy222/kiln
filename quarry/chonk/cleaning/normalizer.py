"""Block content normalizer for cleaning extracted text.

Applies a configurable sequence of cleaning operations to block
content. Skips QA-filtered blocks. Stamps cleaned blocks with
metadata for auditability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from chonk.core.document import Block
from chonk.qa.filters import FILTER_FLAG_KEY

# Metadata keys stamped onto cleaned blocks
CLEANING_APPLIED_KEY = "cleaning_applied"
CLEANING_OPS_KEY = "cleaning_operations"
ORIGINAL_CONTENT_KEY = "original_content"

# --- Compiled patterns for cleaning operations ---

# Page markers: "Page 1", "Page 1 of 10", standalone numbers at end
_PAGE_MARKER_PATTERNS = [
    re.compile(r"^\s*page\s+\d+\s*(?:of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*-\s*\d+\s*-\s*$"),
    re.compile(r"^\s*\d+\s*$"),
]

# Continuation markers
_CONTINUATION_RE = re.compile(
    r"\s*\(?continued\.?\)?\.?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_CONTINUED_FROM_RE = re.compile(
    r"^\s*\(?continued\s+(?:from|on)\s+(?:page\s+)?\d+\)?\.?\s*",
    re.IGNORECASE,
)

# Formatting artifacts: lines of repeated punctuation
_ARTIFACT_LINE_RE = re.compile(
    r"^\s*[.\-=_*~#]{3,}\s*$",
    re.MULTILINE,
)

# Multiple whitespace within a line
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

# Multiple blank lines
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# Line-end hyphenation: "word-\n" followed by lowercase continuation
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n([a-z])")

# Smart quotes and special characters
_SMART_QUOTES = {
    "\u2018": "'",  # left single
    "\u2019": "'",  # right single
    "\u201c": '"',  # left double
    "\u201d": '"',  # right double
    "\u2013": "-",  # en dash
    "\u2014": "--",  # em dash
    "\u2026": "...",  # ellipsis
    "\u00a0": " ",  # non-breaking space
}


@dataclass
class CleaningResult:
    """Result of running the normalizer over a block list.

    Attributes:
        document_id: ID of the cleaned document.
        total_blocks: Total blocks evaluated.
        cleaned_count: Blocks where content was modified.
        skipped_filtered: Blocks skipped (QA-filtered).
        skipped_empty: Blocks skipped (empty content).
        operations_applied: Count of each operation applied.
    """

    document_id: str
    total_blocks: int
    cleaned_count: int
    skipped_filtered: int
    skipped_empty: int
    operations_applied: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "document_id": self.document_id,
            "total_blocks": self.total_blocks,
            "cleaned_count": self.cleaned_count,
            "skipped_filtered": self.skipped_filtered,
            "skipped_empty": self.skipped_empty,
            "operations_applied": self.operations_applied,
        }


class BlockNormalizer:
    """Applies text cleaning operations to block content.

    Processes blocks in-place, preserving all metadata and structure.
    Skips QA-filtered blocks. Stamps cleaned blocks with metadata
    tracking which operations were applied.

    Args:
        store_originals: If True, store original content in metadata.

    Example::

        normalizer = BlockNormalizer()
        result = normalizer.normalize(blocks, document_id="doc1")
        print(f"{result.cleaned_count} blocks cleaned")
    """

    def __init__(self, store_originals: bool = False) -> None:
        self._store_originals = store_originals

    def normalize(
        self,
        blocks: list[Block],
        document_id: str = "unknown",
    ) -> CleaningResult:
        """Run all cleaning operations over a block list.

        Modifies block content in-place. Skips blocks that are
        QA-filtered or have empty content.

        Args:
            blocks: Blocks to clean.
            document_id: Document ID for the result.

        Returns:
            CleaningResult with counts and operations.
        """
        skipped_filtered = 0
        skipped_empty = 0
        cleaned = 0
        ops_counts: dict[str, int] = {}

        for block in blocks:
            if block.metadata.get(FILTER_FLAG_KEY, False):
                skipped_filtered += 1
                continue
            if not block.content.strip():
                skipped_empty += 1
                continue

            original = block.content
            applied = self._clean_block(block)

            if block.content != original:
                cleaned += 1
                self._stamp_block(block, applied, original)
                for op in applied:
                    ops_counts[op] = ops_counts.get(op, 0) + 1

        return CleaningResult(
            document_id=document_id,
            total_blocks=len(blocks),
            cleaned_count=cleaned,
            skipped_filtered=skipped_filtered,
            skipped_empty=skipped_empty,
            operations_applied=ops_counts,
        )

    def _clean_block(self, block: Block) -> list[str]:
        """Apply all cleaning operations to a single block.

        Args:
            block: Block to clean (modified in-place).

        Returns:
            List of operation names that changed content.
        """
        applied: list[str] = []

        content = block.content
        content = self._normalize_chars(content, applied)
        content = self._repair_hyphenation(content, applied)
        content = self._remove_continuation_markers(content, applied)
        content = self._remove_formatting_artifacts(content, applied)
        content = self._normalize_whitespace(content, applied)

        block.content = content
        return applied

    @staticmethod
    def _normalize_chars(text: str, applied: list[str]) -> str:
        """Replace smart quotes and special Unicode characters.

        Args:
            text: Input text.
            applied: List to append operation name if changed.

        Returns:
            Cleaned text.
        """
        original = text
        for char, replacement in _SMART_QUOTES.items():
            text = text.replace(char, replacement)
        if text != original:
            applied.append("normalize_chars")
        return text

    @staticmethod
    def _repair_hyphenation(text: str, applied: list[str]) -> str:
        """Rejoin words split across lines with hyphens.

        Handles patterns like "impor-\\ntant" -> "important".

        Args:
            text: Input text.
            applied: List to append operation name if changed.

        Returns:
            Cleaned text.
        """
        result = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
        if result != text:
            applied.append("repair_hyphenation")
        return result

    @staticmethod
    def _remove_continuation_markers(text: str, applied: list[str]) -> str:
        """Strip continuation markers from content.

        Removes "(continued)", "continued from page X", etc.

        Args:
            text: Input text.
            applied: List to append operation name if changed.

        Returns:
            Cleaned text.
        """
        original = text
        text = _CONTINUATION_RE.sub("", text)
        text = _CONTINUED_FROM_RE.sub("", text)
        if text != original:
            applied.append("remove_continuations")
        return text

    @staticmethod
    def _remove_formatting_artifacts(text: str, applied: list[str]) -> str:
        """Remove lines consisting of repeated punctuation.

        Strips lines like "-----", "=====", ".....", etc.

        Args:
            text: Input text.
            applied: List to append operation name if changed.

        Returns:
            Cleaned text.
        """
        result = _ARTIFACT_LINE_RE.sub("", text)
        if result != text:
            applied.append("remove_artifacts")
        return result

    @staticmethod
    def _normalize_whitespace(text: str, applied: list[str]) -> str:
        """Collapse multiple spaces and blank lines.

        Reduces multiple spaces to single space within lines,
        limits consecutive blank lines to one, and strips
        leading/trailing whitespace.

        Args:
            text: Input text.
            applied: List to append operation name if changed.

        Returns:
            Cleaned text.
        """
        original = text
        # Collapse multiple spaces within lines
        lines = text.split("\n")
        lines = [_MULTI_SPACE_RE.sub(" ", line) for line in lines]
        text = "\n".join(lines)
        # Limit consecutive blank lines
        text = _MULTI_NEWLINE_RE.sub("\n\n", text)
        # Strip outer whitespace
        text = text.strip()
        if text != original:
            applied.append("normalize_whitespace")
        return text

    def _stamp_block(
        self,
        block: Block,
        operations: list[str],
        original: str,
    ) -> None:
        """Write cleaning metadata onto a block.

        Args:
            block: Block to stamp.
            operations: List of operation names applied.
            original: Original content before cleaning.
        """
        block.metadata[CLEANING_APPLIED_KEY] = True
        block.metadata[CLEANING_OPS_KEY] = operations
        if self._store_originals:
            block.metadata[ORIGINAL_CONTENT_KEY] = original

    @staticmethod
    def is_page_marker(text: str) -> bool:
        """Check if text is a standalone page marker.

        Useful for pre-filtering page number blocks.

        Args:
            text: Text to check.

        Returns:
            True if text matches a page marker pattern.
        """
        stripped = text.strip()
        return any(p.match(stripped) for p in _PAGE_MARKER_PATTERNS)

    @staticmethod
    def was_cleaned(block: Block) -> bool:
        """Check if a block has been cleaned.

        Args:
            block: Block to check.

        Returns:
            True if the block has the cleaning_applied flag.
        """
        return bool(block.metadata.get(CLEANING_APPLIED_KEY, False))
