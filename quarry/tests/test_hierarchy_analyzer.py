"""Tests for HierarchyAnalyzer."""

from __future__ import annotations

from chonk.core.document import Block, BlockType
from chonk.hierarchy.analyzer import HierarchyAnalysis, HierarchyAnalyzer, HierarchyIssue
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


def _build_tree(blocks: list[Block]) -> HierarchyTree:
    """Build hierarchy tree from blocks."""
    return HierarchyBuilder.build_from_blocks(blocks)


# ===================================================================
# HierarchyIssue
# ===================================================================


class TestHierarchyIssue:
    """Tests for issue data structure."""

    def test_issue_creation(self):
        issue = HierarchyIssue(
            severity="warning",
            issue_type="orphan_heading",
            message="Test message",
            node_id="s1",
        )
        assert issue.severity == "warning"
        assert issue.issue_type == "orphan_heading"

    def test_issue_with_suggested_fix(self):
        issue = HierarchyIssue(
            severity="error",
            issue_type="test",
            message="msg",
            suggested_fix="Do this instead",
        )
        assert issue.suggested_fix == "Do this instead"


# ===================================================================
# HierarchyAnalysis
# ===================================================================


class TestHierarchyAnalysis:
    """Tests for analysis result."""

    def test_has_errors(self):
        tree = _build_tree([])
        analysis = HierarchyAnalysis(tree=tree)
        assert not analysis.has_errors
        analysis.issues.append(
            HierarchyIssue(severity="error", issue_type="test", message="err")
        )
        assert analysis.has_errors

    def test_has_warnings(self):
        tree = _build_tree([])
        analysis = HierarchyAnalysis(tree=tree)
        assert not analysis.has_warnings
        analysis.issues.append(
            HierarchyIssue(severity="warning", issue_type="test", message="warn")
        )
        assert analysis.has_warnings

    def test_quality_score_starts_at_one(self):
        blocks = [
            _heading("H1", level=1),
            _text("content " * 50),
            _heading("H1.1", level=2),
            _text("more content " * 50),
        ]
        tree = _build_tree(blocks)
        analysis = HierarchyAnalysis(tree=tree)
        assert analysis.quality_score >= 0.9

    def test_quality_penalized_by_errors(self):
        blocks = [_heading("H1", level=1), _text("content " * 50)]
        tree = _build_tree(blocks)
        analysis = HierarchyAnalysis(tree=tree)
        base_score = analysis.quality_score
        analysis.issues.append(
            HierarchyIssue(severity="error", issue_type="test", message="err")
        )
        assert analysis.quality_score < base_score

    def test_quality_score_clamped(self):
        tree = _build_tree([])
        analysis = HierarchyAnalysis(tree=tree)
        for _ in range(20):
            analysis.issues.append(
                HierarchyIssue(severity="error", issue_type="test", message="err")
            )
        assert analysis.quality_score >= 0.0

    def test_to_dict(self):
        tree = _build_tree([_heading("H", level=1), _text("content")])
        analysis = HierarchyAnalysis(tree=tree)
        d = analysis.to_dict()
        assert "quality_score" in d
        assert "has_errors" in d
        assert "issues" in d
        assert "statistics" in d


# ===================================================================
# HierarchyAnalyzer - Orphan Detection
# ===================================================================


class TestOrphanDetection:
    """Tests for orphan heading detection."""

    def test_heading_with_content_not_orphan(self):
        blocks = [_heading("H1", level=1), _text("Content")]
        tree = _build_tree(blocks)
        analysis = HierarchyAnalyzer.analyze(tree)
        orphans = [i for i in analysis.issues if i.issue_type == "orphan_heading"]
        assert len(orphans) == 0

    def test_heading_with_children_not_orphan(self):
        """Parent heading with children but no direct content is NOT orphan."""
        blocks = [
            _heading("Parent", level=1),
            _heading("Child", level=2),
            _text("Child content"),
        ]
        tree = _build_tree(blocks)
        analysis = HierarchyAnalyzer.analyze(tree)
        orphans = [i for i in analysis.issues if i.issue_type == "orphan_heading"]
        assert len(orphans) == 0

    def test_heading_no_content_no_children_is_orphan(self):
        blocks = [_heading("Lonely", level=1)]
        tree = _build_tree(blocks)
        analysis = HierarchyAnalyzer.analyze(tree)
        orphans = [i for i in analysis.issues if i.issue_type == "orphan_heading"]
        assert len(orphans) == 1
        assert orphans[0].node_id is not None


# ===================================================================
# HierarchyAnalyzer - Section Sizes
# ===================================================================


class TestSectionSizes:
    """Tests for section size checks."""

    def test_normal_section_no_issue(self):
        blocks = [_heading("H1", level=1), _text("Normal content " * 20)]
        tree = _build_tree(blocks)
        analysis = HierarchyAnalyzer.analyze(tree)
        size_issues = [
            i
            for i in analysis.issues
            if i.issue_type in ("oversized_section", "undersized_section")
        ]
        assert len(size_issues) == 0

    def test_oversized_section_detected(self):
        """Section with very many tokens triggers warning."""
        blocks = [_heading("Big Section", level=1), _text("word " * 3000)]
        tree = _build_tree(blocks)
        analysis = HierarchyAnalyzer.analyze(tree)
        oversize = [i for i in analysis.issues if i.issue_type == "oversized_section"]
        assert len(oversize) == 1

    def test_undersized_leaf_detected(self):
        """Very small leaf section triggers info."""
        blocks = [_heading("Tiny", level=1), _text("Hi")]
        tree = _build_tree(blocks)
        analysis = HierarchyAnalyzer.analyze(tree)
        undersize = [i for i in analysis.issues if i.issue_type == "undersized_section"]
        assert len(undersize) == 1


# ===================================================================
# HierarchyAnalyzer - Depth Checks
# ===================================================================


class TestDepthChecks:
    """Tests for depth-related checks."""

    def test_moderate_depth_ok(self):
        blocks = [
            _heading("L1", level=1),
            _heading("L2", level=2),
            _heading("L3", level=3),
            _text("Content"),
        ]
        tree = _build_tree(blocks)
        analysis = HierarchyAnalyzer.analyze(tree)
        depth_issues = [
            i for i in analysis.issues if i.issue_type in ("too_deep", "too_flat")
        ]
        assert len(depth_issues) == 0

    def test_flat_structure_flagged(self):
        blocks = [
            _heading("Only", level=1),
            _text("Content"),
        ]
        tree = _build_tree(blocks)
        analysis = HierarchyAnalyzer.analyze(tree)
        flat = [i for i in analysis.issues if i.issue_type == "too_flat"]
        assert len(flat) == 1


# ===================================================================
# HierarchyAnalyzer - Balance
# ===================================================================


class TestBalanceChecks:
    """Tests for tree balance analysis."""

    def test_balanced_tree_no_issue(self):
        blocks = [
            _heading("H1", level=1),
            _heading("H1.1", level=2),
            _text("Content"),
            _heading("H2", level=1),
            _heading("H2.1", level=2),
            _text("Content"),
        ]
        tree = _build_tree(blocks)
        analysis = HierarchyAnalyzer.analyze(tree)
        balance = [i for i in analysis.issues if i.issue_type == "imbalanced"]
        assert len(balance) == 0


# ===================================================================
# HierarchyAnalyzer - Recommendations
# ===================================================================


class TestRecommendations:
    """Tests for recommendation generation."""

    def test_good_hierarchy_gets_recommendation(self):
        blocks = []
        for i in range(12):
            blocks.append(_heading(f"Section {i}", level=1))
            blocks.append(_heading(f"Sub {i}", level=2))
            blocks.append(_text(f"Content for section {i} " * 10))
        tree = _build_tree(blocks)
        analysis = HierarchyAnalyzer.analyze(tree)
        assert any("HIERARCHICAL" in r for r in analysis.recommendations)

    def test_flat_gets_alternative_recommendation(self):
        blocks = [_heading("Only", level=1), _text("Content")]
        tree = _build_tree(blocks)
        analysis = HierarchyAnalyzer.analyze(tree)
        assert any("Flat" in r or "SEMANTIC" in r for r in analysis.recommendations)


# ===================================================================
# HierarchyAnalyzer - compare_trees
# ===================================================================


class TestCompare:
    """Tests for tree comparison."""

    def test_compare_returns_differences(self):
        blocks1 = [_heading("H1", level=1), _text("short")]
        blocks2 = [
            _heading("H1", level=1),
            _heading("H1.1", level=2),
            _text("longer content " * 20),
        ]
        tree1 = _build_tree(blocks1)
        tree2 = _build_tree(blocks2)
        result = HierarchyAnalyzer.compare_trees(tree1, tree2)
        assert "tree1" in result
        assert "tree2" in result
        assert "differences" in result
        assert result["differences"]["node_diff"] == 1

    def test_compare_same_tree(self):
        blocks = [_heading("H1", level=1), _text("content")]
        tree = _build_tree(blocks)
        result = HierarchyAnalyzer.compare_trees(tree, tree)
        assert result["differences"]["node_diff"] == 0
