"""Tests for the QA filtering module.

Covers: FilterLog, patterns, rules, BlockFilter, and integration.
"""

from __future__ import annotations

from pathlib import Path

from chonk.core.document import Block, BlockType, BoundingBox
from chonk.qa.filter_log import FilterCategory, FilterLog, FilterLogEntry
from chonk.qa.filters import (
    FILTER_CATEGORY_KEY,
    FILTER_FLAG_KEY,
    FILTER_REASON_KEY,
    FILTER_RULE_KEY,
    BlockFilter,
)
from chonk.qa.patterns import (
    ALL_PATTERNS,
    BOILERPLATE_PATTERNS,
    DISTRIBUTION_PATTERNS,
    INDEX_PATTERNS,
    TOC_PATTERNS,
    match_patterns,
)
from chonk.qa.rules import FilterRule, RuleSet

# ===================================================================
# Helpers
# ===================================================================


def _block(
    content: str,
    block_type: BlockType = BlockType.TEXT,
    page: int = 1,
    bbox: BoundingBox | None = None,
) -> Block:
    """Create a test block."""
    return Block(
        id=Block.generate_id(),
        type=block_type,
        content=content,
        page=page,
        bbox=bbox,
    )


# ===================================================================
# FilterCategory enum tests
# ===================================================================


class TestFilterCategory:
    """Test FilterCategory enum."""

    def test_all_categories_exist(self) -> None:
        """Test all expected categories are defined."""
        names = {c.value for c in FilterCategory}
        assert "toc" in names
        assert "index" in names
        assert "distribution_statement" in names
        assert "boilerplate" in names
        assert "page_header" in names
        assert "page_footer" in names
        assert "repetition" in names

    def test_string_enum(self) -> None:
        """Test categories are string enums."""
        assert FilterCategory.TOC == "toc"
        assert isinstance(FilterCategory.TOC, str)


# ===================================================================
# FilterLogEntry tests
# ===================================================================


class TestFilterLogEntry:
    """Test FilterLogEntry dataclass."""

    def test_construction(self) -> None:
        """Test basic construction."""
        entry = FilterLogEntry(
            block_id="b1",
            block_type="text",
            category=FilterCategory.TOC,
            reason="Matched TOC heading",
            rule_name="toc_heading",
            page=1,
            content_preview="Table of Contents",
        )
        assert entry.block_id == "b1"
        assert entry.category == FilterCategory.TOC

    def test_roundtrip(self) -> None:
        """Test to_dict/from_dict roundtrip."""
        entry = FilterLogEntry(
            block_id="b1",
            block_type="text",
            category=FilterCategory.BOILERPLATE,
            reason="Intentionally blank",
            rule_name="intentionally_blank",
            page=5,
            content_preview="This page intentionally left blank",
        )
        d = entry.to_dict()
        restored = FilterLogEntry.from_dict(d)
        assert restored.block_id == entry.block_id
        assert restored.category == entry.category
        assert restored.reason == entry.reason
        assert restored.page == 5


# ===================================================================
# FilterLog tests
# ===================================================================


class TestFilterLog:
    """Test FilterLog dataclass."""

    def test_empty_log(self) -> None:
        """Test empty log properties."""
        log = FilterLog(document_id="doc1")
        assert log.filtered_count == 0
        assert log.by_category() == {}

    def test_filtered_count(self) -> None:
        """Test count with entries."""
        log = FilterLog(document_id="doc1")
        log.entries.append(
            FilterLogEntry(
                block_id="b1",
                block_type="text",
                category=FilterCategory.TOC,
                reason="toc",
                rule_name="toc_heading",
                page=1,
                content_preview="TOC",
            )
        )
        assert log.filtered_count == 1

    def test_by_category(self) -> None:
        """Test grouping by category."""
        log = FilterLog(document_id="doc1")
        log.entries.append(
            FilterLogEntry(
                block_id="b1",
                block_type="text",
                category=FilterCategory.TOC,
                reason="toc",
                rule_name="toc",
                page=1,
                content_preview="TOC",
            )
        )
        log.entries.append(
            FilterLogEntry(
                block_id="b2",
                block_type="text",
                category=FilterCategory.BOILERPLATE,
                reason="blank",
                rule_name="blank",
                page=2,
                content_preview="blank",
            )
        )
        log.entries.append(
            FilterLogEntry(
                block_id="b3",
                block_type="text",
                category=FilterCategory.TOC,
                reason="toc2",
                rule_name="toc",
                page=3,
                content_preview="TOC 2",
            )
        )
        grouped = log.by_category()
        assert len(grouped[FilterCategory.TOC]) == 2
        assert len(grouped[FilterCategory.BOILERPLATE]) == 1

    def test_roundtrip(self) -> None:
        """Test to_dict/from_dict roundtrip."""
        log = FilterLog(document_id="doc1", document_type="manual")
        log.entries.append(
            FilterLogEntry(
                block_id="b1",
                block_type="text",
                category=FilterCategory.DISTRIBUTION,
                reason="test",
                rule_name="test",
                page=1,
                content_preview="preview",
            )
        )
        d = log.to_dict()
        restored = FilterLog.from_dict(d)
        assert restored.document_id == "doc1"
        assert restored.document_type == "manual"
        assert restored.filtered_count == 1

    def test_save_load(self, tmp_path: Path) -> None:
        """Test save and load from file."""
        log = FilterLog(document_id="doc1")
        log.entries.append(
            FilterLogEntry(
                block_id="b1",
                block_type="text",
                category=FilterCategory.TOC,
                reason="test",
                rule_name="test",
                page=1,
                content_preview="preview",
            )
        )
        path = tmp_path / "log.json"
        log.save(path)
        loaded = FilterLog.load(path)
        assert loaded.document_id == "doc1"
        assert loaded.filtered_count == 1


# ===================================================================
# Pattern tests
# ===================================================================


class TestPatterns:
    """Test compiled pattern library."""

    def test_toc_heading_matches(self) -> None:
        """Test TOC heading pattern."""
        result = match_patterns("Table of Contents", TOC_PATTERNS)
        assert result is not None
        assert result.name == "toc_heading"

    def test_toc_heading_case_insensitive(self) -> None:
        """Test TOC pattern is case-insensitive."""
        result = match_patterns("TABLE OF CONTENTS", TOC_PATTERNS)
        assert result is not None

    def test_contents_short(self) -> None:
        """Test short 'Contents' heading."""
        result = match_patterns("Contents", TOC_PATTERNS)
        assert result is not None
        assert result.name == "toc_heading_short"

    def test_toc_dotleader(self) -> None:
        """Test TOC dot-leader line."""
        result = match_patterns("Chapter 1 ........... 5", TOC_PATTERNS)
        assert result is not None
        assert result.name == "toc_dotleader"

    def test_index_heading(self) -> None:
        """Test index heading pattern."""
        result = match_patterns("Index", INDEX_PATTERNS)
        assert result is not None
        assert result.name == "index_heading"

    def test_alphabetical_index(self) -> None:
        """Test alphabetical index heading."""
        result = match_patterns("Alphabetical Index", INDEX_PATTERNS)
        assert result is not None

    def test_distro_statement(self) -> None:
        """Test distribution statement prefix."""
        result = match_patterns(
            "Distribution Statement A",
            DISTRIBUTION_PATTERNS,
        )
        assert result is not None
        assert result.name == "distro_statement"

    def test_distro_approved(self) -> None:
        """Test approved for public release."""
        result = match_patterns(
            "Approved for Public Release; Distribution Unlimited",
            DISTRIBUTION_PATTERNS,
        )
        assert result is not None

    def test_intentionally_blank(self) -> None:
        """Test intentionally blank page."""
        result = match_patterns(
            "This page intentionally left blank",
            BOILERPLATE_PATTERNS,
        )
        assert result is not None
        assert result.name == "intentionally_blank"

    def test_fouo(self) -> None:
        """Test For Official Use Only."""
        result = match_patterns("For Official Use Only", BOILERPLATE_PATTERNS)
        assert result is not None
        assert result.name == "fouo"

    def test_proprietary(self) -> None:
        """Test proprietary confidential."""
        result = match_patterns(
            "This document is Proprietary and Confidential",
            BOILERPLATE_PATTERNS,
        )
        assert result is not None
        assert result.name == "proprietary"

    def test_normal_text_no_match(self) -> None:
        """Test that normal content does not match."""
        result = match_patterns("The hydraulic system operates at 3000 PSI.")
        assert result is None

    def test_heading_no_match(self) -> None:
        """Test that a section heading does not match."""
        result = match_patterns("3.1 Hydraulic System Overview")
        assert result is None

    def test_all_patterns_populated(self) -> None:
        """Test ALL_PATTERNS includes all categories."""
        categories = {p.category for p in ALL_PATTERNS}
        assert FilterCategory.TOC in categories
        assert FilterCategory.INDEX in categories
        assert FilterCategory.DISTRIBUTION in categories
        assert FilterCategory.BOILERPLATE in categories


# ===================================================================
# Rules tests
# ===================================================================


class TestFilterRule:
    """Test FilterRule dataclass."""

    def test_construction(self) -> None:
        """Test basic construction."""
        rule = FilterRule(
            name="test_rule",
            description="A test rule",
            categories=[FilterCategory.TOC],
        )
        assert rule.enabled is True
        assert len(rule.categories) == 1

    def test_roundtrip(self) -> None:
        """Test to_dict/from_dict roundtrip."""
        rule = FilterRule(
            name="test",
            description="desc",
            categories=[FilterCategory.TOC, FilterCategory.INDEX],
            enabled=False,
        )
        d = rule.to_dict()
        restored = FilterRule.from_dict(d)
        assert restored.name == "test"
        assert restored.enabled is False
        assert len(restored.categories) == 2


class TestRuleSet:
    """Test RuleSet configuration."""

    def test_default_has_all_categories(self) -> None:
        """Test default rule set enables all categories."""
        rs = RuleSet.default()
        active = rs.get_active_categories()
        assert FilterCategory.TOC in active
        assert FilterCategory.INDEX in active
        assert FilterCategory.DISTRIBUTION in active
        assert FilterCategory.BOILERPLATE in active
        assert FilterCategory.PAGE_HEADER in active
        assert FilterCategory.PAGE_FOOTER in active
        assert FilterCategory.REPETITION in active

    def test_parts_catalog_disables_index(self) -> None:
        """Test parts catalog rule set disables INDEX filtering."""
        rs = RuleSet.for_document_type("PARTS_CATALOG")
        active = rs.get_active_categories()
        assert FilterCategory.INDEX not in active
        assert FilterCategory.TOC in active

    def test_technical_manual(self) -> None:
        """Test technical manual rule set keeps all categories."""
        rs = RuleSet.for_document_type("TECHNICAL_MANUAL")
        active = rs.get_active_categories()
        assert FilterCategory.TOC in active
        assert FilterCategory.INDEX in active

    def test_disabled_rule_not_in_active(self) -> None:
        """Test disabled rules are excluded."""
        rs = RuleSet.default()
        for rule in rs.rules:
            if FilterCategory.TOC in rule.categories:
                rule.enabled = False
        active = rs.get_active_categories()
        assert FilterCategory.TOC not in active

    def test_roundtrip(self) -> None:
        """Test to_dict/from_dict roundtrip."""
        rs = RuleSet.default()
        d = rs.to_dict()
        restored = RuleSet.from_dict(d)
        assert restored.name == rs.name
        assert len(restored.rules) == len(rs.rules)


# ===================================================================
# BlockFilter tests
# ===================================================================


class TestBlockFilter:
    """Test BlockFilter core functionality."""

    def test_constructor_defaults(self) -> None:
        """Test default construction."""
        f = BlockFilter()
        assert f._repetition_threshold == 0.85

    def test_empty_blocks(self) -> None:
        """Test filtering empty block list."""
        f = BlockFilter()
        result = f.filter_blocks([], document_id="doc1")
        assert result.total_blocks == 0
        assert result.filtered_count == 0
        assert result.filter_ratio == 0.0


class TestPatternFiltering:
    """Test pattern-based filtering."""

    def test_toc_block_filtered(self) -> None:
        """Test TOC heading block is filtered."""
        f = BlockFilter()
        blocks = [_block("Table of Contents")]
        result = f.filter_blocks(blocks)
        assert result.filtered_count == 1
        assert BlockFilter.is_filtered(blocks[0])

    def test_normal_text_passes(self) -> None:
        """Test normal text block passes."""
        f = BlockFilter()
        blocks = [_block("The engine runs at 3000 RPM.")]
        result = f.filter_blocks(blocks)
        assert result.filtered_count == 0
        assert not BlockFilter.is_filtered(blocks[0])

    def test_toc_dotleader_filtered(self) -> None:
        """Test TOC dot-leader block is filtered."""
        f = BlockFilter()
        blocks = [_block("Chapter 3 ............... 42")]
        result = f.filter_blocks(blocks)
        assert result.filtered_count == 1

    def test_contents_heading_filtered(self) -> None:
        """Test 'Contents' heading is filtered."""
        f = BlockFilter()
        blocks = [_block("Contents")]
        result = f.filter_blocks(blocks)
        assert result.filtered_count == 1


class TestDistributionFiltering:
    """Test distribution statement filtering."""

    def test_distro_statement_filtered(self) -> None:
        """Test distribution statement block is filtered."""
        f = BlockFilter()
        blocks = [_block("Distribution Statement A")]
        result = f.filter_blocks(blocks)
        assert result.filtered_count == 1
        assert result.log.entries[0].category == FilterCategory.DISTRIBUTION

    def test_distro_approved_filtered(self) -> None:
        """Test approved for public release."""
        f = BlockFilter()
        blocks = [_block("Approved for Public Release; " "Distribution is Unlimited")]
        result = f.filter_blocks(blocks)
        assert result.filtered_count == 1


class TestBoilerplateFiltering:
    """Test boilerplate filtering."""

    def test_blank_page_filtered(self) -> None:
        """Test intentionally blank page."""
        f = BlockFilter()
        blocks = [_block("This page intentionally left blank")]
        result = f.filter_blocks(blocks)
        assert result.filtered_count == 1
        assert result.log.entries[0].category == FilterCategory.BOILERPLATE

    def test_fouo_filtered(self) -> None:
        """Test For Official Use Only."""
        f = BlockFilter()
        blocks = [_block("For Official Use Only")]
        result = f.filter_blocks(blocks)
        assert result.filtered_count == 1


class TestHeaderFooterFiltering:
    """Test header/footer block type filtering."""

    def test_header_block_type_filtered(self) -> None:
        """Test HEADER block type is filtered."""
        f = BlockFilter()
        blocks = [_block("TM 1-2345", block_type=BlockType.HEADER)]
        result = f.filter_blocks(blocks)
        assert result.filtered_count == 1
        assert result.log.entries[0].category == FilterCategory.PAGE_HEADER

    def test_footer_block_type_filtered(self) -> None:
        """Test FOOTER block type is filtered."""
        f = BlockFilter()
        blocks = [_block("Page 42", block_type=BlockType.FOOTER)]
        result = f.filter_blocks(blocks)
        assert result.filtered_count == 1
        assert result.log.entries[0].category == FilterCategory.PAGE_FOOTER


class TestPositionalFiltering:
    """Test positional header/footer detection."""

    def test_top_of_page_short_text(self) -> None:
        """Test short text in top 10% is filtered as header."""
        f = BlockFilter()
        bbox = BoundingBox(x1=50, y1=10, x2=200, y2=800, page=1)
        blocks = [_block("TM 1-2345-A", bbox=bbox)]
        result = f.filter_blocks(blocks)
        assert result.filtered_count == 1
        assert result.log.entries[0].category == FilterCategory.PAGE_HEADER

    def test_long_text_at_top_not_filtered(self) -> None:
        """Test that long text at top of page is NOT filtered."""
        f = BlockFilter()
        bbox = BoundingBox(x1=50, y1=10, x2=200, y2=800, page=1)
        long_text = " ".join(["word"] * 20)
        blocks = [_block(long_text, bbox=bbox)]
        result = f.filter_blocks(blocks)
        assert result.filtered_count == 0

    def test_no_bbox_not_filtered(self) -> None:
        """Test block without bbox is not positionally filtered."""
        f = BlockFilter()
        blocks = [_block("TM 1-2345")]
        result = f.filter_blocks(blocks)
        assert result.filtered_count == 0


class TestRepetitionFiltering:
    """Test near-duplicate detection."""

    def test_exact_duplicate_filtered(self) -> None:
        """Test exact duplicate block is filtered."""
        f = BlockFilter()
        blocks = [
            _block("Safety warning: always wear PPE", page=1),
            _block("Safety warning: always wear PPE", page=2),
        ]
        result = f.filter_blocks(blocks)
        assert result.filtered_count == 1
        assert not BlockFilter.is_filtered(blocks[0])
        assert BlockFilter.is_filtered(blocks[1])

    def test_different_text_not_filtered(self) -> None:
        """Test different text blocks are not filtered."""
        f = BlockFilter()
        blocks = [
            _block("The hydraulic system provides power.", page=1),
            _block("The electrical system provides control.", page=2),
        ]
        result = f.filter_blocks(blocks)
        assert result.filtered_count == 0

    def test_very_short_text_not_repetition(self) -> None:
        """Test very short text is not flagged as repetition."""
        f = BlockFilter()
        blocks = [
            _block("OK", page=1),
            _block("OK", page=2),
        ]
        result = f.filter_blocks(blocks)
        # "OK" normalized is too short (< 5 chars)
        assert result.filtered_count == 0

    def test_custom_threshold(self) -> None:
        """Test custom repetition threshold."""
        f = BlockFilter(repetition_threshold=0.5)
        blocks = [
            _block("Check the hydraulic fluid level daily", page=1),
            _block("Check the hydraulic fluid level daily", page=2),
        ]
        result = f.filter_blocks(blocks)
        assert result.filtered_count == 1


# ===================================================================
# Filter stamping and metadata tests
# ===================================================================


class TestFilterStamping:
    """Test that filter metadata is correctly stamped."""

    def test_stamp_sets_metadata(self) -> None:
        """Test filter stamp writes correct metadata keys."""
        f = BlockFilter()
        blocks = [_block("Table of Contents")]
        f.filter_blocks(blocks)
        b = blocks[0]
        assert b.metadata[FILTER_FLAG_KEY] is True
        assert "toc_heading" in b.metadata[FILTER_RULE_KEY]
        assert b.metadata[FILTER_CATEGORY_KEY] == "toc"
        assert FILTER_REASON_KEY in b.metadata

    def test_is_filtered_true(self) -> None:
        """Test is_filtered returns True for stamped block."""
        b = _block("Table of Contents")
        f = BlockFilter()
        f.filter_blocks([b])
        assert BlockFilter.is_filtered(b) is True

    def test_is_filtered_false(self) -> None:
        """Test is_filtered returns False for normal block."""
        b = _block("Normal text content.")
        assert BlockFilter.is_filtered(b) is False

    def test_block_remains_in_list(self) -> None:
        """Test filtered blocks are NOT removed from list."""
        f = BlockFilter()
        blocks = [
            _block("Table of Contents"),
            _block("Normal content paragraph."),
        ]
        result = f.filter_blocks(blocks)
        assert len(blocks) == 2
        assert result.filtered_count == 1


# ===================================================================
# FilterResult tests
# ===================================================================


class TestFilterResult:
    """Test FilterResult dataclass."""

    def test_counts_add_up(self) -> None:
        """Test filtered + passed = total."""
        f = BlockFilter()
        blocks = [
            _block("Table of Contents"),
            _block("Normal content."),
            _block("This page intentionally left blank"),
        ]
        result = f.filter_blocks(blocks)
        assert result.total_blocks == 3
        assert result.filtered_count + result.passed_count == 3
        assert result.filtered_count == 2

    def test_filter_ratio(self) -> None:
        """Test filter ratio calculation."""
        f = BlockFilter()
        blocks = [
            _block("Table of Contents"),
            _block("Normal content."),
        ]
        result = f.filter_blocks(blocks)
        assert abs(result.filter_ratio - 0.5) < 0.01

    def test_to_dict(self) -> None:
        """Test FilterResult serialization."""
        f = BlockFilter()
        blocks = [_block("Normal content.")]
        result = f.filter_blocks(blocks, document_id="doc1")
        d = result.to_dict()
        assert d["document_id"] == "doc1"
        assert d["total_blocks"] == 1


# ===================================================================
# Document type override tests
# ===================================================================


class TestDocumentTypeOverride:
    """Test document-type-specific rule sets."""

    def test_parts_catalog_keeps_index(self) -> None:
        """Test parts catalog does not filter index blocks."""
        rs = RuleSet.for_document_type("PARTS_CATALOG")
        f = BlockFilter(rule_set=rs)
        blocks = [_block("Index")]
        result = f.filter_blocks(blocks)
        assert result.filtered_count == 0

    def test_default_filters_index(self) -> None:
        """Test default rule set DOES filter index blocks."""
        f = BlockFilter()
        blocks = [_block("Index")]
        result = f.filter_blocks(blocks)
        assert result.filtered_count == 1

    def test_disabled_category_not_filtered(self) -> None:
        """Test disabled category skips matching blocks."""
        rs = RuleSet.default()
        for rule in rs.rules:
            if FilterCategory.TOC in rule.categories:
                rule.enabled = False
        f = BlockFilter(rule_set=rs)
        blocks = [_block("Table of Contents")]
        result = f.filter_blocks(blocks)
        assert result.filtered_count == 0


# ===================================================================
# Integration tests
# ===================================================================


class TestFilterIntegration:
    """Integration tests for the full filter pipeline."""

    def test_mixed_document(self) -> None:
        """Test filtering a document with mixed content."""
        f = BlockFilter()
        blocks = [
            _block("Table of Contents", block_type=BlockType.HEADING),
            _block("Chapter 1 ......... 1"),
            _block("Chapter 2 ......... 15"),
            _block("1.1 Introduction\n\n" "This manual covers the hydraulic system."),
            _block("Distribution Statement A: " "Approved for Public Release"),
            _block("This page intentionally left blank"),
            _block("TM 1-2345", block_type=BlockType.HEADER),
            _block(
                "The hydraulic pump provides 3000 PSI.",
                block_type=BlockType.TEXT,
            ),
        ]
        result = f.filter_blocks(blocks, document_id="test_doc")
        # TOC heading, 2 dot-leaders, distro, blank page, header = 6
        assert result.filtered_count == 6
        assert result.passed_count == 2
        assert result.log.filtered_count == 6

    def test_filter_log_populated(self) -> None:
        """Test filter log has correct entries."""
        f = BlockFilter()
        blocks = [
            _block("Table of Contents"),
            _block("Normal content."),
        ]
        result = f.filter_blocks(blocks, document_id="doc1")
        assert result.log.document_id == "doc1"
        assert len(result.log.entries) == 1
        assert result.log.entries[0].category == FilterCategory.TOC

    def test_all_passed_document(self) -> None:
        """Test document with no filterable content."""
        f = BlockFilter()
        blocks = [
            _block("Section 1: Overview"),
            _block("The system operates under normal conditions."),
            _block("Maintenance is required every 500 hours."),
        ]
        result = f.filter_blocks(blocks)
        assert result.filtered_count == 0
        assert result.passed_count == 3
