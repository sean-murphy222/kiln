"""Tests for HierarchyBuilder."""

from __future__ import annotations

import pytest

from chonk.core.document import Block, BlockType
from chonk.hierarchy.builder import HierarchyBuilder
from chonk.hierarchy.tree import HierarchyTree


# ===================================================================
# Helpers
# ===================================================================


def _heading(content: str, level: int = 1, page: int = 1) -> Block:
    """Create a heading block."""
    return Block(
        id=Block.generate_id(),
        type=BlockType.HEADING,
        content=content,
        page=page,
        heading_level=level,
    )


def _text(content: str, page: int = 1) -> Block:
    """Create a text block."""
    return Block(
        id=Block.generate_id(),
        type=BlockType.TEXT,
        content=content,
        page=page,
    )


def _table(content: str, page: int = 1) -> Block:
    """Create a table block."""
    return Block(
        id=Block.generate_id(),
        type=BlockType.TABLE,
        content=content,
        page=page,
    )


# ===================================================================
# extract_section_id
# ===================================================================


class TestExtractSectionId:
    """Tests for section ID extraction from heading text."""

    def test_simple_number(self):
        assert HierarchyBuilder.extract_section_id("3 Safety") == "3"

    def test_dotted_number(self):
        assert HierarchyBuilder.extract_section_id("2.1.1 Safety") == "2.1.1"

    def test_deeply_dotted(self):
        assert HierarchyBuilder.extract_section_id("E.5.3.5 Maintenance") == "E.5.3.5"

    def test_decimal_section(self):
        assert HierarchyBuilder.extract_section_id("3.66 Graphics") == "3.66"

    def test_letter_prefix(self):
        assert HierarchyBuilder.extract_section_id("E.5.3.5 Title") == "E.5.3.5"

    def test_plain_text_heading(self):
        result = HierarchyBuilder.extract_section_id("FOREWORD")
        assert result == "FOREWORD"

    def test_plain_title_case(self):
        result = HierarchyBuilder.extract_section_id("Introduction")
        assert result == "Introduction"

    def test_special_chars_stripped(self):
        result = HierarchyBuilder.extract_section_id('Title: "Something"')
        assert '"' not in result

    def test_long_heading_truncated(self):
        long_heading = "A" * 100
        result = HierarchyBuilder.extract_section_id(long_heading)
        assert len(result) <= 50


# ===================================================================
# build_from_blocks - Basic Hierarchy
# ===================================================================


class TestBuildFromBlocksBasic:
    """Tests for basic hierarchy building."""

    def test_empty_blocks(self):
        tree = HierarchyBuilder.build_from_blocks([], document_id="test")
        assert isinstance(tree, HierarchyTree)
        assert tree.total_nodes == 0

    def test_content_only(self):
        """Blocks with no headings go to root."""
        blocks = [_text("paragraph 1"), _text("paragraph 2")]
        tree = HierarchyBuilder.build_from_blocks(blocks)
        # Root with content counts as 1 node
        assert tree.total_nodes == 1
        assert len(tree.root.content_blocks) == 2

    def test_single_heading(self):
        blocks = [_heading("Chapter 1"), _text("Content")]
        tree = HierarchyBuilder.build_from_blocks(blocks)
        assert tree.total_nodes == 1
        node = tree.root.children[0]
        assert node.heading == "Chapter 1"
        assert len(node.content_blocks) == 1

    def test_two_sibling_headings(self):
        blocks = [
            _heading("Chapter 1", level=1),
            _text("Content 1"),
            _heading("Chapter 2", level=1),
            _text("Content 2"),
        ]
        tree = HierarchyBuilder.build_from_blocks(blocks)
        assert tree.total_nodes == 2
        assert len(tree.root.children) == 2

    def test_parent_child_nesting(self):
        blocks = [
            _heading("Chapter 1", level=1),
            _text("Chapter content"),
            _heading("Section 1.1", level=2),
            _text("Section content"),
        ]
        tree = HierarchyBuilder.build_from_blocks(blocks)
        assert tree.total_nodes == 2
        ch1 = tree.root.children[0]
        assert len(ch1.children) == 1
        assert ch1.children[0].heading == "Section 1.1"

    def test_document_id_preserved(self):
        tree = HierarchyBuilder.build_from_blocks([], document_id="my-doc")
        assert tree.document_id == "my-doc"

    def test_metadata_preserved(self):
        tree = HierarchyBuilder.build_from_blocks(
            [], document_id="d", metadata={"source": "test"}
        )
        assert tree.metadata["source"] == "test"


# ===================================================================
# build_from_blocks - 6 Heading Levels
# ===================================================================


class TestSixHeadingLevels:
    """Tests for handling all 6 heading levels properly."""

    def test_six_nested_levels(self):
        """Build a tree with all 6 heading levels."""
        blocks = [
            _heading("Level 1", level=1),
            _text("L1 content"),
            _heading("Level 2", level=2),
            _text("L2 content"),
            _heading("Level 3", level=3),
            _text("L3 content"),
            _heading("Level 4", level=4),
            _text("L4 content"),
            _heading("Level 5", level=5),
            _text("L5 content"),
            _heading("Level 6", level=6),
            _text("L6 content"),
        ]
        tree = HierarchyBuilder.build_from_blocks(blocks)
        assert tree.total_nodes == 6
        assert tree.max_depth == 6

    def test_levels_1_and_3_skipping(self):
        """Handle level jumps (e.g., H1 directly to H3)."""
        blocks = [
            _heading("Level 1", level=1),
            _heading("Level 3 (skipped 2)", level=3),
            _text("Content"),
        ]
        tree = HierarchyBuilder.build_from_blocks(blocks)
        # Level 3 should be child of level 1
        assert len(tree.root.children) == 1
        h1 = tree.root.children[0]
        assert len(h1.children) == 1
        assert h1.children[0].level == 3

    def test_level_backtrack(self):
        """After deep nesting, going back to higher level."""
        blocks = [
            _heading("H1", level=1),
            _heading("H1.1", level=2),
            _heading("H1.1.1", level=3),
            _text("Deep content"),
            _heading("H2", level=1),
            _text("Back to top"),
        ]
        tree = HierarchyBuilder.build_from_blocks(blocks)
        assert len(tree.root.children) == 2
        assert tree.root.children[0].heading == "H1"
        assert tree.root.children[1].heading == "H2"

    def test_mixed_level_siblings(self):
        """Multiple sections at same level under a parent."""
        blocks = [
            _heading("Chapter", level=1),
            _heading("Sec A", level=2),
            _text("A content"),
            _heading("Sec B", level=2),
            _text("B content"),
            _heading("Sec C", level=2),
            _text("C content"),
        ]
        tree = HierarchyBuilder.build_from_blocks(blocks)
        ch = tree.root.children[0]
        assert len(ch.children) == 3

    def test_repeated_same_level(self):
        """Multiple headings at the same level are siblings."""
        blocks = [
            _heading("A", level=2),
            _text("A content"),
            _heading("B", level=2),
            _text("B content"),
        ]
        tree = HierarchyBuilder.build_from_blocks(blocks)
        # Both should be children of root since no level-1 exists
        assert len(tree.root.children) == 2


# ===================================================================
# build_from_blocks - Section Numbering
# ===================================================================


class TestSectionNumbering:
    """Tests for section numbering scheme detection."""

    def test_simple_numbering(self):
        blocks = [
            _heading("1 Introduction", level=1),
            _heading("2 Methods", level=1),
            _heading("3 Results", level=1),
        ]
        tree = HierarchyBuilder.build_from_blocks(blocks)
        ids = [n.section_id for n in tree.root.children]
        assert ids == ["1", "2", "3"]

    def test_dotted_numbering(self):
        blocks = [
            _heading("1.1 Overview", level=2),
            _heading("1.2 Details", level=2),
            _heading("1.2.1 Sub-detail", level=3),
        ]
        tree = HierarchyBuilder.build_from_blocks(blocks)
        all_nodes = tree.get_all_nodes()
        ids = [n.section_id for n in all_nodes]
        assert "1.1" in ids
        assert "1.2" in ids
        assert "1.2.1" in ids

    def test_letter_prefix_numbering(self):
        """Sections like A.1, E.5.3.5."""
        blocks = [
            _heading("A.1 Appendix Section", level=2),
            _heading("A.2 Another Section", level=2),
        ]
        tree = HierarchyBuilder.build_from_blocks(blocks)
        ids = [n.section_id for n in tree.root.children]
        assert "A.1" in ids
        assert "A.2" in ids

    def test_unnumbered_heading_uses_text(self):
        """Headings without numbers use cleaned text as ID."""
        blocks = [_heading("FOREWORD", level=1)]
        tree = HierarchyBuilder.build_from_blocks(blocks)
        assert tree.root.children[0].section_id == "FOREWORD"


# ===================================================================
# build_from_blocks - Content Assignment
# ===================================================================


class TestContentAssignment:
    """Tests for proper content block assignment."""

    def test_content_follows_heading(self):
        blocks = [
            _heading("H1", level=1),
            _text("Para 1"),
            _text("Para 2"),
        ]
        tree = HierarchyBuilder.build_from_blocks(blocks)
        node = tree.root.children[0]
        assert len(node.content_blocks) == 2

    def test_table_assigned_to_section(self):
        blocks = [
            _heading("Data Section", level=1),
            _text("See the table below:"),
            _table("Col1 | Col2\nA | B"),
        ]
        tree = HierarchyBuilder.build_from_blocks(blocks)
        node = tree.root.children[0]
        assert len(node.content_blocks) == 2

    def test_content_before_first_heading_in_root(self):
        """Content before any heading goes to root node."""
        blocks = [
            _text("Preamble text"),
            _heading("Chapter 1", level=1),
            _text("Chapter content"),
        ]
        tree = HierarchyBuilder.build_from_blocks(blocks)
        assert len(tree.root.content_blocks) == 1
        assert tree.root.content_blocks[0].content == "Preamble text"

    def test_multiple_block_types(self):
        """Different block types all assigned correctly."""
        blocks = [
            _heading("Mixed Section", level=1),
            _text("Text paragraph"),
            _table("Table data"),
            Block(
                id=Block.generate_id(),
                type=BlockType.LIST,
                content="- item 1\n- item 2",
            ),
        ]
        tree = HierarchyBuilder.build_from_blocks(blocks)
        node = tree.root.children[0]
        assert len(node.content_blocks) == 3


# ===================================================================
# flatten_to_sections
# ===================================================================


class TestFlattenToSections:
    """Tests for tree flattening."""

    def test_flatten_simple(self):
        blocks = [
            _heading("H1", level=1),
            _text("Content"),
            _heading("H2", level=1),
            _text("More"),
        ]
        tree = HierarchyBuilder.build_from_blocks(blocks)
        sections = HierarchyBuilder.flatten_to_sections(tree)
        assert len(sections) == 2

    def test_flatten_preserves_hierarchy_path(self):
        blocks = [
            _heading("Chapter 1", level=1),
            _heading("Section 1.1", level=2),
            _text("Content"),
        ]
        tree = HierarchyBuilder.build_from_blocks(blocks)
        sections = HierarchyBuilder.flatten_to_sections(tree)
        sec = [s for s in sections if s["heading"] == "Section 1.1"][0]
        assert "Chapter 1" in sec["hierarchy_path"]

    def test_flatten_includes_all_fields(self):
        blocks = [_heading("Test", level=1), _text("Content")]
        tree = HierarchyBuilder.build_from_blocks(blocks)
        sections = HierarchyBuilder.flatten_to_sections(tree)
        sec = sections[0]
        assert "section_id" in sec
        assert "heading" in sec
        assert "content" in sec
        assert "level" in sec
        assert "hierarchy_path" in sec
        assert "has_children" in sec

    def test_flatten_empty_tree(self):
        tree = HierarchyBuilder.build_from_blocks([])
        sections = HierarchyBuilder.flatten_to_sections(tree)
        assert sections == []
