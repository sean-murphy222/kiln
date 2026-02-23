"""Tests for HierarchyNode and HierarchyTree data structures."""

from __future__ import annotations

from chonk.core.document import Block, BlockType
from chonk.hierarchy.tree import HierarchyNode, HierarchyTree


# ===================================================================
# Helpers
# ===================================================================


def _make_heading(content: str, level: int = 1, page: int = 1) -> Block:
    """Create a heading block."""
    return Block(
        id=Block.generate_id(),
        type=BlockType.HEADING,
        content=content,
        page=page,
        heading_level=level,
    )


def _make_text(content: str, page: int = 1) -> Block:
    """Create a text block."""
    return Block(
        id=Block.generate_id(),
        type=BlockType.TEXT,
        content=content,
        page=page,
    )


def _make_simple_tree() -> HierarchyTree:
    """Build a simple 3-level hierarchy tree for testing."""
    root = HierarchyNode(section_id="root", level=0)
    ch1 = HierarchyNode(
        section_id="1",
        heading="Chapter 1",
        heading_block=_make_heading("Chapter 1", level=1),
        level=1,
    )
    sec1_1 = HierarchyNode(
        section_id="1.1",
        heading="Section 1.1",
        heading_block=_make_heading("Section 1.1", level=2),
        level=2,
        content_blocks=[_make_text("Content of section 1.1")],
    )
    sec1_2 = HierarchyNode(
        section_id="1.2",
        heading="Section 1.2",
        heading_block=_make_heading("Section 1.2", level=2),
        level=2,
        content_blocks=[_make_text("Content of section 1.2")],
    )
    ch2 = HierarchyNode(
        section_id="2",
        heading="Chapter 2",
        heading_block=_make_heading("Chapter 2", level=1, page=3),
        level=1,
        content_blocks=[_make_text("Content of chapter 2", page=3)],
    )
    root.add_child(ch1)
    ch1.add_child(sec1_1)
    ch1.add_child(sec1_2)
    root.add_child(ch2)
    return HierarchyTree(root=root, document_id="test-doc")


# ===================================================================
# HierarchyNode
# ===================================================================


class TestHierarchyNode:
    """Tests for HierarchyNode data structure."""

    def test_add_child_sets_parent(self):
        parent = HierarchyNode(section_id="parent", level=0)
        child = HierarchyNode(section_id="child", level=1)
        parent.add_child(child)
        assert child.parent is parent
        assert child in parent.children

    def test_add_content(self):
        node = HierarchyNode(section_id="s1", level=1)
        block = _make_text("Hello world")
        node.add_content(block)
        assert len(node.content_blocks) == 1
        assert node.content_blocks[0] is block

    def test_content_text_joins_blocks(self):
        node = HierarchyNode(section_id="s1", level=1)
        node.add_content(_make_text("First paragraph"))
        node.add_content(_make_text("Second paragraph"))
        assert "First paragraph" in node.content_text
        assert "Second paragraph" in node.content_text
        assert "\n\n" in node.content_text

    def test_full_text_includes_heading(self):
        node = HierarchyNode(
            section_id="s1",
            heading="My Heading",
            level=1,
            content_blocks=[_make_text("Body text")],
        )
        assert "My Heading" in node.full_text
        assert "Body text" in node.full_text

    def test_full_text_without_heading(self):
        node = HierarchyNode(
            section_id="s1",
            level=1,
            content_blocks=[_make_text("Body only")],
        )
        assert node.full_text == "Body only"

    def test_page_range_from_blocks(self):
        node = HierarchyNode(
            section_id="s1",
            heading="H",
            heading_block=_make_heading("H", page=2),
            level=1,
            content_blocks=[_make_text("text", page=2), _make_text("more", page=4)],
        )
        assert node.page_range == [2, 4]

    def test_page_range_empty(self):
        node = HierarchyNode(section_id="s1", level=1)
        assert node.page_range == [0, 0]

    def test_hierarchy_path(self):
        tree = _make_simple_tree()
        sec1_1 = tree.root.children[0].children[0]
        assert sec1_1.hierarchy_path == "Chapter 1 > Section 1.1"

    def test_hierarchy_path_root(self):
        tree = _make_simple_tree()
        assert tree.root.hierarchy_path == ""

    def test_depth(self):
        tree = _make_simple_tree()
        assert tree.root.depth == 0
        assert tree.root.children[0].depth == 1
        assert tree.root.children[0].children[0].depth == 2

    def test_is_leaf(self):
        tree = _make_simple_tree()
        assert not tree.root.is_leaf
        assert not tree.root.children[0].is_leaf
        assert tree.root.children[0].children[0].is_leaf

    def test_descendant_count(self):
        tree = _make_simple_tree()
        assert tree.root.descendant_count == 4  # ch1, sec1.1, sec1.2, ch2

    def test_get_all_descendants(self):
        tree = _make_simple_tree()
        desc = tree.root.get_all_descendants()
        assert len(desc) == 4
        ids = [n.section_id for n in desc]
        assert "1" in ids
        assert "1.1" in ids
        assert "1.2" in ids
        assert "2" in ids

    def test_get_leaves(self):
        tree = _make_simple_tree()
        leaves = tree.root.get_leaves()
        assert len(leaves) == 3  # sec1.1, sec1.2, ch2
        leaf_ids = [n.section_id for n in leaves]
        assert "1.1" in leaf_ids
        assert "1.2" in leaf_ids
        assert "2" in leaf_ids

    def test_to_dict_roundtrip(self):
        tree = _make_simple_tree()
        node = tree.root.children[0]
        d = node.to_dict()
        assert d["section_id"] == "1"
        assert d["heading"] == "Chapter 1"
        assert d["heading_level"] == 1
        assert len(d["children"]) == 2

    def test_repr(self):
        node = HierarchyNode(section_id="s1", heading="Test", level=2)
        r = repr(node)
        assert "s1" in r
        assert "Test" in r
        assert "level=2" in r


# ===================================================================
# HierarchyTree
# ===================================================================


class TestHierarchyTree:
    """Tests for HierarchyTree data structure."""

    def test_total_nodes_excludes_empty_root(self):
        tree = _make_simple_tree()
        assert tree.total_nodes == 4

    def test_max_depth(self):
        tree = _make_simple_tree()
        assert tree.max_depth == 2

    def test_leaf_count(self):
        tree = _make_simple_tree()
        assert tree.leaf_count == 3

    def test_get_all_nodes(self):
        tree = _make_simple_tree()
        nodes = tree.get_all_nodes()
        assert len(nodes) == 4

    def test_get_all_nodes_include_root(self):
        tree = _make_simple_tree()
        nodes = tree.get_all_nodes(include_root=True)
        assert len(nodes) == 5

    def test_get_node_by_id(self):
        tree = _make_simple_tree()
        node = tree.get_node_by_id("1.2")
        assert node is not None
        assert node.heading == "Section 1.2"

    def test_get_node_by_id_not_found(self):
        tree = _make_simple_tree()
        assert tree.get_node_by_id("nonexistent") is None

    def test_get_nodes_at_level(self):
        tree = _make_simple_tree()
        level1 = tree.get_nodes_at_level(1)
        assert len(level1) == 2
        level2 = tree.get_nodes_at_level(2)
        assert len(level2) == 2

    def test_get_statistics(self):
        tree = _make_simple_tree()
        stats = tree.get_statistics()
        assert stats["total_nodes"] == 4
        assert stats["max_depth"] == 2
        assert "level_distribution" in stats

    def test_level_distribution(self):
        tree = _make_simple_tree()
        dist = tree._get_level_distribution()
        assert dist[1] == 2
        assert dist[2] == 2

    def test_to_dict_roundtrip(self):
        tree = _make_simple_tree()
        d = tree.to_dict()
        assert d["document_id"] == "test-doc"
        assert "statistics" in d
        assert "root" in d

    def test_repr(self):
        tree = _make_simple_tree()
        r = repr(tree)
        assert "test-doc" in r
        assert "nodes=" in r

    def test_single_node_tree(self):
        """Tree with only root should have correct stats."""
        root = HierarchyNode(section_id="root", level=0)
        tree = HierarchyTree(root=root, document_id="empty")
        assert tree.total_nodes == 0
        assert tree.leaf_count == 1
