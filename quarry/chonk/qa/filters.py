"""Core QA filter for identifying zero-value content blocks.

Stamps filtered blocks with metadata rather than removing them,
preserving auditability. Chunkers skip stamped blocks.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any

from chonk.core.document import Block, BlockType
from chonk.qa.filter_log import FilterCategory, FilterLog, FilterLogEntry
from chonk.qa.patterns import ALL_PATTERNS, match_patterns
from chonk.qa.rules import RuleSet

# Metadata keys stamped onto filtered blocks
FILTER_FLAG_KEY = "qa_filtered"
FILTER_REASON_KEY = "qa_filter_reason"
FILTER_CATEGORY_KEY = "qa_filter_category"
FILTER_RULE_KEY = "qa_filter_rule"


@dataclass
class FilterResult:
    """Result of running the filter pass over a document.

    Attributes:
        document_id: ID of the filtered document.
        total_blocks: Total blocks evaluated.
        filtered_count: Blocks that were filtered.
        passed_count: Blocks that passed.
        log: Detailed log of all filtering decisions.
    """

    document_id: str
    total_blocks: int
    filtered_count: int
    passed_count: int
    log: FilterLog

    @property
    def filter_ratio(self) -> float:
        """Fraction of blocks filtered (0.0-1.0)."""
        if self.total_blocks == 0:
            return 0.0
        return self.filtered_count / self.total_blocks

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "document_id": self.document_id,
            "total_blocks": self.total_blocks,
            "filtered_count": self.filtered_count,
            "passed_count": self.passed_count,
            "filter_ratio": self.filter_ratio,
            "log": self.log.to_dict(),
        }


class BlockFilter:
    """QA pass that identifies and stamps zero-value content blocks.

    Filtered blocks are NOT removed from the block list. They are
    stamped with metadata so downstream chunkers can skip them while
    preserving a full audit trail.

    Args:
        rule_set: Rules to apply. Defaults to RuleSet.default().
        repetition_threshold: Similarity ratio above which blocks
            are considered near-duplicates (0.0-1.0).

    Example::

        f = BlockFilter()
        result = f.filter_blocks(blocks, document_id="doc1")
        print(f"{result.filtered_count} blocks filtered")
    """

    def __init__(
        self,
        rule_set: RuleSet | None = None,
        repetition_threshold: float = 0.85,
    ) -> None:
        self._rule_set = rule_set or RuleSet.default()
        self._repetition_threshold = repetition_threshold

    def filter_blocks(
        self,
        blocks: list[Block],
        document_id: str = "unknown",
        document_type: str | None = None,
    ) -> FilterResult:
        """Run QA filter pass over a block list.

        Stamps filtered blocks in-place. Does not remove them.

        Args:
            blocks: Blocks to evaluate.
            document_id: Document ID for logging.
            document_type: Document type for rule selection.

        Returns:
            FilterResult with counts and log.
        """
        active = self._rule_set.get_active_categories()
        log = FilterLog(
            document_id=document_id,
            document_type=document_type or "unknown",
        )
        seen_content: dict[str, str] = {}
        filtered = 0

        for block in blocks:
            should_filter, category, reason, rule_name = self._check_block(
                block, seen_content, active
            )
            if should_filter and category is not None:
                self._stamp_block(block, category, reason, rule_name)
                log.entries.append(
                    FilterLogEntry(
                        block_id=block.id,
                        block_type=block.type.value,
                        category=category,
                        reason=reason,
                        rule_name=rule_name,
                        page=block.page,
                        content_preview=block.content[:80],
                    )
                )
                filtered += 1
            else:
                norm = self._normalize_text(block.content)
                if norm:
                    seen_content[norm[:100]] = block.id

        return FilterResult(
            document_id=document_id,
            total_blocks=len(blocks),
            filtered_count=filtered,
            passed_count=len(blocks) - filtered,
            log=log,
        )

    def _check_block(
        self,
        block: Block,
        seen_content: dict[str, str],
        active_categories: set[FilterCategory],
    ) -> tuple[bool, FilterCategory | None, str, str]:
        """Evaluate one block against all filter checks.

        Args:
            block: Block to evaluate.
            seen_content: Dict of normalized content seen so far.
            active_categories: Currently active filter categories.

        Returns:
            Tuple of (should_filter, category, reason, rule_name).
        """
        if block.type == BlockType.HEADER:
            if FilterCategory.PAGE_HEADER in active_categories:
                return (
                    True,
                    FilterCategory.PAGE_HEADER,
                    "Block type is HEADER",
                    "block_type_header",
                )

        if block.type == BlockType.FOOTER:
            if FilterCategory.PAGE_FOOTER in active_categories:
                return (
                    True,
                    FilterCategory.PAGE_FOOTER,
                    "Block type is FOOTER",
                    "block_type_footer",
                )

        result = self._check_pattern_match(block, active_categories)
        if result[0]:
            return result

        result = self._check_positional(block, active_categories)
        if result[0]:
            return result

        result = self._check_repetition(block, seen_content, active_categories)
        if result[0]:
            return result

        return (False, None, "", "")

    def _check_pattern_match(
        self,
        block: Block,
        active_categories: set[FilterCategory],
    ) -> tuple[bool, FilterCategory | None, str, str]:
        """Check block content against the pattern library.

        Args:
            block: Block to check.
            active_categories: Active filter categories.

        Returns:
            Tuple of (should_filter, category, reason, rule_name).
        """
        active_patterns = [p for p in ALL_PATTERNS if p.category in active_categories]
        matched = match_patterns(block.content, active_patterns)
        if matched is not None:
            return (
                True,
                matched.category,
                f"Matched pattern: {matched.name}",
                matched.name,
            )
        return (False, None, "", "")

    def _check_positional(
        self,
        block: Block,
        active_categories: set[FilterCategory],
    ) -> tuple[bool, FilterCategory | None, str, str]:
        """Check block position for header/footer detection.

        Uses bounding box to detect blocks in the top or bottom
        10% of the page that are short text.

        Args:
            block: Block to check.
            active_categories: Active filter categories.

        Returns:
            Tuple of (should_filter, category, reason, rule_name).
        """
        if block.bbox is None:
            return (False, None, "", "")

        content_len = len(block.content.split())
        if content_len > 15:
            return (False, None, "", "")

        page_height = block.bbox.y2
        if page_height <= 0:
            return (False, None, "", "")

        relative_top = block.bbox.y1 / page_height

        if relative_top < 0.10 and FilterCategory.PAGE_HEADER in active_categories:
            return (
                True,
                FilterCategory.PAGE_HEADER,
                "Short text in top 10% of page",
                "positional_header",
            )

        relative_bottom = 1.0 - (block.bbox.y2 / page_height)
        if relative_bottom < 0:
            return (False, None, "", "")

        return (False, None, "", "")

    def _check_repetition(
        self,
        block: Block,
        seen_content: dict[str, str],
        active_categories: set[FilterCategory],
    ) -> tuple[bool, FilterCategory | None, str, str]:
        """Detect near-duplicate blocks.

        Args:
            block: Block to check.
            seen_content: Dict mapping normalized text prefix to block_id.
            active_categories: Active filter categories.

        Returns:
            Tuple of (should_filter, category, reason, rule_name).
        """
        if FilterCategory.REPETITION not in active_categories:
            return (False, None, "", "")

        norm = self._normalize_text(block.content)
        if not norm or len(norm) < 5:
            return (False, None, "", "")

        key = norm[:100]
        if key in seen_content:
            sim = self._similarity(norm, key)
            if sim >= self._repetition_threshold:
                return (
                    True,
                    FilterCategory.REPETITION,
                    f"Near-duplicate of block {seen_content[key]} " f"(similarity: {sim:.2f})",
                    "repetition_detect",
                )

        return (False, None, "", "")

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for repetition comparison.

        Lowercases, collapses whitespace, strips leading/trailing space.

        Args:
            text: Raw text to normalize.

        Returns:
            Normalized text string.
        """
        return " ".join(text.lower().split())

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Compute text similarity ratio.

        Args:
            a: First text.
            b: Second text.

        Returns:
            Similarity ratio (0.0-1.0).
        """
        return difflib.SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _stamp_block(
        block: Block,
        category: FilterCategory,
        reason: str,
        rule_name: str,
    ) -> None:
        """Write filter metadata onto a block.

        Args:
            block: Block to stamp.
            category: Why it was filtered.
            reason: Human-readable explanation.
            rule_name: Which rule matched.
        """
        block.metadata[FILTER_FLAG_KEY] = True
        block.metadata[FILTER_REASON_KEY] = reason
        block.metadata[FILTER_CATEGORY_KEY] = category.value
        block.metadata[FILTER_RULE_KEY] = rule_name

    @staticmethod
    def is_filtered(block: Block) -> bool:
        """Check if a block has been marked as filtered.

        Args:
            block: Block to check.

        Returns:
            True if the block has the qa_filtered flag.
        """
        return bool(block.metadata.get(FILTER_FLAG_KEY, False))
