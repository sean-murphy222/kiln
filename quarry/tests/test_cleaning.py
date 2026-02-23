"""Tests for block content cleaning and normalization."""

from __future__ import annotations

from chonk.cleaning.normalizer import (
    BlockNormalizer,
    CleaningResult,
)
from chonk.core.document import Block, BlockType


def _block(
    content: str,
    block_type: BlockType = BlockType.TEXT,
    **metadata: object,
) -> Block:
    """Create a test block with given content."""
    return Block(
        id=Block.generate_id(),
        type=block_type,
        content=content,
        metadata=dict(metadata),
    )


# --- CleaningResult ---


class TestCleaningResult:
    """Tests for CleaningResult dataclass."""

    def test_construction(self) -> None:
        """Test basic construction."""
        r = CleaningResult(
            document_id="doc1",
            total_blocks=10,
            cleaned_count=5,
            skipped_filtered=2,
            skipped_empty=1,
            operations_applied={"normalize_whitespace": 3},
        )
        assert r.total_blocks == 10
        assert r.cleaned_count == 5

    def test_to_dict(self) -> None:
        """Test serialization."""
        r = CleaningResult(
            document_id="doc1",
            total_blocks=5,
            cleaned_count=2,
            skipped_filtered=1,
            skipped_empty=0,
            operations_applied={"repair_hyphenation": 1},
        )
        d = r.to_dict()
        assert d["document_id"] == "doc1"
        assert d["operations_applied"]["repair_hyphenation"] == 1


# --- Whitespace Normalization ---


class TestWhitespaceNormalization:
    """Tests for whitespace normalization."""

    def test_collapse_multiple_spaces(self) -> None:
        """Test multiple spaces become single space."""
        b = _block("hello    world")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert b.content == "hello world"

    def test_collapse_tabs(self) -> None:
        """Test tabs collapsed to single space."""
        b = _block("hello\t\tworld")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert b.content == "hello world"

    def test_limit_blank_lines(self) -> None:
        """Test multiple blank lines limited to one."""
        b = _block("paragraph one\n\n\n\nparagraph two")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert b.content == "paragraph one\n\nparagraph two"

    def test_strip_outer_whitespace(self) -> None:
        """Test leading/trailing whitespace stripped."""
        b = _block("  hello world  ")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert b.content == "hello world"

    def test_already_clean_unchanged(self) -> None:
        """Test that clean content isn't modified."""
        b = _block("This is clean text.")
        normalizer = BlockNormalizer()
        result = normalizer.normalize([b])
        assert b.content == "This is clean text."
        assert result.cleaned_count == 0


# --- Character Normalization ---


class TestCharacterNormalization:
    """Tests for smart quote and special character replacement."""

    def test_smart_single_quotes(self) -> None:
        """Test smart single quotes become straight."""
        b = _block("it\u2018s a test\u2019s")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert b.content == "it's a test's"

    def test_smart_double_quotes(self) -> None:
        """Test smart double quotes become straight."""
        b = _block("\u201chello\u201d")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert b.content == '"hello"'

    def test_en_dash(self) -> None:
        """Test en dash becomes hyphen."""
        b = _block("pages 1\u20135")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert b.content == "pages 1-5"

    def test_em_dash(self) -> None:
        """Test em dash becomes double hyphen."""
        b = _block("word\u2014another")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert b.content == "word--another"

    def test_ellipsis_char(self) -> None:
        """Test Unicode ellipsis becomes three dots."""
        b = _block("wait\u2026")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert b.content == "wait..."

    def test_non_breaking_space(self) -> None:
        """Test non-breaking space becomes regular space."""
        b = _block("hello\u00a0world")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert b.content == "hello world"


# --- Hyphenation Repair ---


class TestHyphenationRepair:
    """Tests for line-end hyphenation repair."""

    def test_simple_hyphenation(self) -> None:
        """Test word split across lines is rejoined."""
        b = _block("impor-\ntant information")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert "important" in b.content

    def test_preserves_real_hyphens(self) -> None:
        """Test hyphens not at line breaks are preserved."""
        b = _block("well-known fact")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert b.content == "well-known fact"

    def test_multiple_hyphenations(self) -> None:
        """Test multiple hyphenated words in one block."""
        b = _block("mainte-\nnance and calibra-\ntion")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert "maintenance" in b.content
        assert "calibration" in b.content

    def test_uppercase_not_rejoined(self) -> None:
        """Test hyphen before uppercase (likely new sentence) preserved."""
        b = _block("end of line-\nNew paragraph")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        # Uppercase after hyphen-newline is not rejoined
        assert "line-\nNew" in b.content


# --- Continuation Markers ---


class TestContinuationMarkers:
    """Tests for continuation marker removal."""

    def test_parenthesized_continued(self) -> None:
        """Test '(continued)' is removed."""
        b = _block("Table 3.1 (continued)")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert "continued" not in b.content.lower()
        assert "Table 3.1" in b.content

    def test_continued_no_parens(self) -> None:
        """Test 'continued' without parens is removed."""
        b = _block("Section 2 continued.")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert "continued" not in b.content.lower()

    def test_continued_from_page(self) -> None:
        """Test 'continued from page X' prefix removed."""
        b = _block("Continued from page 5. The procedure...")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert "continued" not in b.content.lower()
        assert "procedure" in b.content

    def test_continued_on_page(self) -> None:
        """Test 'continued on page X' prefix removed."""
        b = _block("Continued on page 12 The next step...")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert "continued" not in b.content.lower()


# --- Formatting Artifacts ---


class TestFormattingArtifacts:
    """Tests for formatting artifact removal."""

    def test_dashes_line(self) -> None:
        """Test line of dashes removed."""
        b = _block("Header\n----------\nContent")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert "----------" not in b.content
        assert "Header" in b.content
        assert "Content" in b.content

    def test_equals_line(self) -> None:
        """Test line of equals removed."""
        b = _block("Title\n==========\nBody")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert "==========" not in b.content

    def test_dots_line(self) -> None:
        """Test line of dots removed."""
        b = _block("text\n..........\nmore text")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert ".........." not in b.content

    def test_short_punctuation_preserved(self) -> None:
        """Test that short punctuation (< 3 chars) is preserved."""
        b = _block("item 1. item 2.")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert b.content == "item 1. item 2."

    def test_asterisks_line(self) -> None:
        """Test line of asterisks removed."""
        b = _block("above\n***\nbelow")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert "***" not in b.content


# --- Page Markers ---


class TestPageMarkers:
    """Tests for page marker detection."""

    def test_page_number(self) -> None:
        """Test 'Page 5' detected as page marker."""
        assert BlockNormalizer.is_page_marker("Page 5")

    def test_page_x_of_y(self) -> None:
        """Test 'Page 5 of 10' detected."""
        assert BlockNormalizer.is_page_marker("Page 5 of 10")

    def test_dashed_number(self) -> None:
        """Test '- 5 -' detected as page marker."""
        assert BlockNormalizer.is_page_marker("- 5 -")

    def test_standalone_number(self) -> None:
        """Test standalone number detected."""
        assert BlockNormalizer.is_page_marker("42")

    def test_page_case_insensitive(self) -> None:
        """Test page marker is case insensitive."""
        assert BlockNormalizer.is_page_marker("PAGE 3")
        assert BlockNormalizer.is_page_marker("page 3")

    def test_normal_text_not_marker(self) -> None:
        """Test that regular text is not a page marker."""
        assert not BlockNormalizer.is_page_marker("This is normal text.")

    def test_number_in_text_not_marker(self) -> None:
        """Test that number within text is not a marker."""
        assert not BlockNormalizer.is_page_marker("Section 5 describes the process.")


# --- QA Filter Integration ---


class TestFilteredBlockSkipping:
    """Tests for skipping QA-filtered blocks."""

    def test_filtered_block_skipped(self) -> None:
        """Test QA-filtered blocks are not cleaned."""
        b = _block(
            "  messy  content  ",
            qa_filtered=True,
        )
        normalizer = BlockNormalizer()
        result = normalizer.normalize([b])
        assert b.content == "  messy  content  "
        assert result.skipped_filtered == 1
        assert result.cleaned_count == 0

    def test_empty_block_skipped(self) -> None:
        """Test empty blocks are skipped."""
        b = _block("   ")
        normalizer = BlockNormalizer()
        result = normalizer.normalize([b])
        assert result.skipped_empty == 1

    def test_mixed_blocks(self) -> None:
        """Test mix of filtered, empty, and cleanable blocks."""
        blocks = [
            _block("  needs  cleaning  "),
            _block("  filtered  ", qa_filtered=True),
            _block("   "),
            _block("also  needs  work  "),
        ]
        normalizer = BlockNormalizer()
        result = normalizer.normalize(blocks)
        assert result.cleaned_count == 2
        assert result.skipped_filtered == 1
        assert result.skipped_empty == 1
        assert result.total_blocks == 4


# --- Metadata Stamping ---


class TestMetadataStamping:
    """Tests for cleaning metadata stamps."""

    def test_cleaned_block_stamped(self) -> None:
        """Test cleaned blocks get metadata stamps."""
        b = _block("  messy  content  ")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert b.metadata.get("cleaning_applied") is True
        assert "cleaning_operations" in b.metadata
        assert "normalize_whitespace" in b.metadata["cleaning_operations"]

    def test_unchanged_block_not_stamped(self) -> None:
        """Test unchanged blocks don't get stamps."""
        b = _block("Clean content already.")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert "cleaning_applied" not in b.metadata

    def test_store_originals(self) -> None:
        """Test original content stored when enabled."""
        b = _block("  messy  content  ")
        normalizer = BlockNormalizer(store_originals=True)
        normalizer.normalize([b])
        assert b.metadata.get("original_content") == ("  messy  content  ")

    def test_no_originals_by_default(self) -> None:
        """Test originals not stored by default."""
        b = _block("  messy  content  ")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert "original_content" not in b.metadata

    def test_was_cleaned_helper(self) -> None:
        """Test was_cleaned static method."""
        b = _block("  messy  ")
        normalizer = BlockNormalizer()
        assert not BlockNormalizer.was_cleaned(b)
        normalizer.normalize([b])
        assert BlockNormalizer.was_cleaned(b)


# --- Operations Tracking ---


class TestOperationsTracking:
    """Tests for operation counting in results."""

    def test_whitespace_counted(self) -> None:
        """Test whitespace operations are counted."""
        blocks = [
            _block("  a  "),
            _block("  b  "),
        ]
        normalizer = BlockNormalizer()
        result = normalizer.normalize(blocks)
        assert result.operations_applied.get("normalize_whitespace", 0) == 2

    def test_multiple_operations_per_block(self) -> None:
        """Test multiple operations on one block are all counted."""
        b = _block("impor-\ntant  info  (continued)")
        normalizer = BlockNormalizer()
        result = normalizer.normalize([b])
        ops = result.operations_applied
        assert "repair_hyphenation" in ops
        assert "remove_continuations" in ops

    def test_no_ops_when_clean(self) -> None:
        """Test no operations tracked for clean content."""
        b = _block("Already clean.")
        normalizer = BlockNormalizer()
        result = normalizer.normalize([b])
        assert result.operations_applied == {}


# --- Integration ---


class TestCleaningIntegration:
    """Integration tests combining multiple operations."""

    def test_full_cleaning_pipeline(self) -> None:
        """Test all operations work together."""
        b = _block(
            "  Mainte-\nnance\u00a0procedure  "
            "(continued)\n\n\n\n"
            "Step 1:  Do  the  thing.\n"
            "----------\n"
            "Step 2:  Do  the  next\u2014thing.  "
        )
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert "Maintenance" in b.content
        assert "continued" not in b.content.lower()
        assert "----------" not in b.content
        assert "--thing" in b.content  # em dash → --
        # No excessive whitespace
        assert "  " not in b.content

    def test_preserves_block_type(self) -> None:
        """Test block type is not modified."""
        b = _block("  heading  text  ", block_type=BlockType.HEADING)
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert b.type == BlockType.HEADING

    def test_preserves_existing_metadata(self) -> None:
        """Test existing metadata is preserved."""
        b = _block("  messy  ", custom_field="keep_me")
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert b.metadata["custom_field"] == "keep_me"
        assert b.metadata["cleaning_applied"] is True

    def test_document_id_in_result(self) -> None:
        """Test document ID flows to result."""
        b = _block("  x  ")
        normalizer = BlockNormalizer()
        result = normalizer.normalize([b], document_id="doc42")
        assert result.document_id == "doc42"

    def test_heading_content_cleaned(self) -> None:
        """Test heading blocks also get cleaned."""
        b = _block(
            "  Chapter  1  ",
            block_type=BlockType.HEADING,
        )
        normalizer = BlockNormalizer()
        normalizer.normalize([b])
        assert b.content == "Chapter 1"
