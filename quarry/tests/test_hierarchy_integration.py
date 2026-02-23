"""Tests for orphan prevention and Tier 1 structural profile integration."""

from __future__ import annotations

from chonk.core.document import Block, BlockType
from chonk.hierarchy.builder import HierarchyBuilder
from chonk.tier1.fingerprinter import DocumentFingerprint
from chonk.tier1.taxonomy import DocumentType

# ===================================================================
# Helpers
# ===================================================================


def _heading(content: str, level: int = 1) -> Block:
    """Create a heading block."""
    return Block(
        id=Block.generate_id(),
        type=BlockType.HEADING,
        content=content,
        heading_level=level,
    )


def _text(content: str) -> Block:
    """Create a text block."""
    return Block(id=Block.generate_id(), type=BlockType.TEXT, content=content)


# ===================================================================
# Orphan Prevention
# ===================================================================


class TestOrphanPrevention:
    """Tests for preventing orphaned headings during build."""

    def test_repair_merges_consecutive_orphan_headings(self):
        """Consecutive headings with no content between them are common in TOCs.

        The repair should merge empty parent headings' text into children.
        """
        blocks = [
            _heading("Chapter 1", level=1),
            _heading("Section 1.1", level=2),
            _text("Actual content here"),
        ]
        tree = HierarchyBuilder.build_from_blocks(blocks, repair_orphans=True)
        # Chapter 1 should still exist but is NOT orphan since it has children
        ch1 = tree.root.children[0]
        assert ch1.heading == "Chapter 1"
        assert len(ch1.children) == 1

    def test_repair_trailing_orphan_merged_up(self):
        """A final heading with no content is merged into parent."""
        blocks = [
            _heading("Chapter 1", level=1),
            _text("Chapter content"),
            _heading("Empty Trailing", level=2),
        ]
        tree = HierarchyBuilder.build_from_blocks(blocks, repair_orphans=True)
        ch1 = tree.root.children[0]
        # The orphan heading text should be added as content to the parent
        assert len(ch1.children) == 0
        assert any("Empty Trailing" in b.content for b in ch1.content_blocks)

    def test_no_repair_when_disabled(self):
        """Default build does not repair orphans."""
        blocks = [
            _heading("Chapter 1", level=1),
            _text("Content"),
            _heading("Orphan", level=2),
        ]
        tree = HierarchyBuilder.build_from_blocks(blocks, repair_orphans=False)
        ch1 = tree.root.children[0]
        assert len(ch1.children) == 1  # Orphan still exists as child

    def test_repair_preserves_non_orphan_nodes(self):
        """Nodes with content are not affected by orphan repair."""
        blocks = [
            _heading("H1", level=1),
            _text("H1 content"),
            _heading("H2", level=1),
            _text("H2 content"),
        ]
        tree = HierarchyBuilder.build_from_blocks(blocks, repair_orphans=True)
        assert len(tree.root.children) == 2
        for child in tree.root.children:
            assert len(child.content_blocks) > 0

    def test_repair_multi_level_orphan_chain(self):
        """Chain of headings with no content except at deepest level."""
        blocks = [
            _heading("L1", level=1),
            _heading("L2", level=2),
            _heading("L3", level=3),
            _text("Deep content"),
        ]
        tree = HierarchyBuilder.build_from_blocks(blocks, repair_orphans=True)
        # L1 has child L2, L2 has child L3, L3 has content = all non-orphan
        l1 = tree.root.children[0]
        assert len(l1.children) == 1
        l2 = l1.children[0]
        assert len(l2.children) == 1
        l3 = l2.children[0]
        assert len(l3.content_blocks) == 1


# ===================================================================
# Tier 1 Structural Profile Integration
# ===================================================================


class TestTier1Integration:
    """Tests for using Tier 1 structural profiles with hierarchy building."""

    def test_build_with_structural_hints(self):
        """Builder accepts a structural_hints dict from Tier 1."""
        blocks = [
            _heading("1 Safety", level=1),
            _text("Safety content"),
            _heading("2 Procedures", level=1),
            _text("Procedure content"),
        ]
        hints = {
            "document_type": DocumentType.TECHNICAL_MANUAL.value,
            "numbering_scheme": "decimal",
        }
        tree = HierarchyBuilder.build_from_blocks(
            blocks, document_id="doc", structural_hints=hints
        )
        assert tree.metadata.get("document_type") == "technical_manual"
        assert tree.metadata.get("numbering_scheme") == "decimal"

    def test_build_with_fingerprint_metadata(self):
        """Builder can incorporate fingerprint features into tree metadata."""
        blocks = [_heading("Test", level=1), _text("Content")]
        fp = DocumentFingerprint()
        tree = HierarchyBuilder.build_from_blocks_with_profile(
            blocks=blocks,
            document_id="doc",
            fingerprint=fp,
            document_type=DocumentType.TECHNICAL_MANUAL,
        )
        assert tree.metadata.get("document_type") == "technical_manual"
        assert "fingerprint_summary" in tree.metadata

    def test_fingerprint_summary_has_key_fields(self):
        """Fingerprint summary includes useful structural information."""
        blocks = [_heading("Test", level=1), _text("Content")]
        fp = DocumentFingerprint()
        tree = HierarchyBuilder.build_from_blocks_with_profile(
            blocks=blocks,
            document_id="doc",
            fingerprint=fp,
            document_type=DocumentType.FORM,
        )
        summary = tree.metadata["fingerprint_summary"]
        assert "page_count" in summary
        assert "has_toc" in summary
        assert "font_count" in summary

    def test_numbering_scheme_auto_detected(self):
        """Tree metadata includes detected numbering scheme."""
        blocks = [
            _heading("1 First", level=1),
            _text("Content"),
            _heading("2 Second", level=1),
            _text("Content"),
            _heading("2.1 Sub", level=2),
            _text("Content"),
        ]
        tree = HierarchyBuilder.build_from_blocks_with_profile(
            blocks=blocks,
            document_id="doc",
            fingerprint=DocumentFingerprint(),
            document_type=DocumentType.TECHNICAL_MANUAL,
        )
        assert "numbering_scheme" in tree.metadata
        scheme = tree.metadata["numbering_scheme"]
        assert scheme["scheme_type"] == "decimal"

    def test_build_with_profile_still_builds_correct_tree(self):
        """Profile integration doesn't break basic tree structure."""
        blocks = [
            _heading("Chapter 1", level=1),
            _heading("Section 1.1", level=2),
            _text("Content"),
            _heading("Chapter 2", level=1),
            _text("More content"),
        ]
        tree = HierarchyBuilder.build_from_blocks_with_profile(
            blocks=blocks,
            document_id="doc",
            fingerprint=DocumentFingerprint(),
            document_type=DocumentType.REPORT,
        )
        assert tree.total_nodes == 3
        assert tree.max_depth == 2
