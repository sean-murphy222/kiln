"""
Hierarchy analyzer.

Analyzes hierarchy trees for quality, issues, and recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chonk.hierarchy.tree import HierarchyTree


@dataclass
class HierarchyIssue:
    """An issue found in the hierarchy tree."""

    severity: str  # "warning", "error", "info"
    issue_type: str
    message: str
    node_id: str | None = None
    suggested_fix: str | None = None


@dataclass
class HierarchyAnalysis:
    """
    Result of hierarchy analysis.

    Contains statistics, issues, and recommendations.
    """

    tree: HierarchyTree
    issues: list[HierarchyIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return any(issue.severity == "error" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(issue.severity == "warning" for issue in self.issues)

    @property
    def quality_score(self) -> float:
        """
        Calculate overall hierarchy quality score (0-1).

        Based on:
        - Structure completeness
        - Balance (not too many orphans)
        - Depth appropriateness
        - Section size distribution
        """
        score = 1.0

        # Penalty for errors
        error_count = sum(1 for i in self.issues if i.severity == "error")
        score -= error_count * 0.2

        # Penalty for warnings
        warning_count = sum(1 for i in self.issues if i.severity == "warning")
        score -= warning_count * 0.05

        # Bonus for good structure
        stats = self.tree.get_statistics()

        # Good depth range (2-5 levels)
        max_depth = stats["max_depth"]
        if 2 <= max_depth <= 5:
            score += 0.1
        elif max_depth > 8:
            score -= 0.1

        # Good leaf ratio (most sections should have content)
        nodes_with_content = stats["nodes_with_content"]
        total_nodes = stats["total_nodes"]
        if total_nodes > 0:
            content_ratio = nodes_with_content / total_nodes
            if content_ratio > 0.7:
                score += 0.1
            elif content_ratio < 0.3:
                score -= 0.1

        return max(0.0, min(1.0, score))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "quality_score": self.quality_score,
            "has_errors": self.has_errors,
            "has_warnings": self.has_warnings,
            "issues": [
                {
                    "severity": issue.severity,
                    "type": issue.issue_type,
                    "message": issue.message,
                    "node_id": issue.node_id,
                    "suggested_fix": issue.suggested_fix,
                }
                for issue in self.issues
            ],
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "statistics": self.tree.get_statistics(),
        }


class HierarchyAnalyzer:
    """
    Analyzes hierarchy trees for quality and issues.

    This helps users understand if their document structure is good
    for chunking and retrieval.
    """

    @staticmethod
    def analyze(tree: HierarchyTree) -> HierarchyAnalysis:
        """
        Perform comprehensive analysis of a hierarchy tree.

        Checks for:
        - Orphan headings (heading with no content)
        - Oversized sections (too many tokens)
        - Undersized sections (too few tokens)
        - Deep nesting (too many levels)
        - Flat structure (not enough hierarchy)
        - Imbalanced tree (some branches much deeper than others)
        """
        analysis = HierarchyAnalysis(tree=tree)

        # Run all checks
        HierarchyAnalyzer._check_orphan_headings(tree, analysis)
        HierarchyAnalyzer._check_section_sizes(tree, analysis)
        HierarchyAnalyzer._check_depth(tree, analysis)
        HierarchyAnalyzer._check_balance(tree, analysis)
        HierarchyAnalyzer._generate_recommendations(tree, analysis)

        return analysis

    @staticmethod
    def _check_orphan_headings(
        tree: HierarchyTree, analysis: HierarchyAnalysis
    ) -> None:
        """Check for headings with no content."""
        all_nodes = tree.get_all_nodes()

        for node in all_nodes:
            if node.heading and not node.content_blocks and not node.children:
                analysis.issues.append(
                    HierarchyIssue(
                        severity="warning",
                        issue_type="orphan_heading",
                        message=f"Heading '{node.heading}' has no content",
                        node_id=node.section_id,
                        suggested_fix="Consider merging with adjacent section or adding content",
                    )
                )

    @staticmethod
    def _check_section_sizes(
        tree: HierarchyTree, analysis: HierarchyAnalysis
    ) -> None:
        """Check for sections that are too large or too small."""
        all_nodes = tree.get_all_nodes()
        nodes_with_content = [n for n in all_nodes if n.content_blocks]

        for node in nodes_with_content:
            token_count = node.token_count

            if token_count > 2000:
                analysis.issues.append(
                    HierarchyIssue(
                        severity="warning",
                        issue_type="oversized_section",
                        message=f"Section '{node.heading}' has {token_count} tokens (very large)",
                        node_id=node.section_id,
                        suggested_fix="Consider splitting into subsections",
                    )
                )
            elif token_count < 10 and node.is_leaf:
                analysis.issues.append(
                    HierarchyIssue(
                        severity="info",
                        issue_type="undersized_section",
                        message=f"Section '{node.heading}' has only {token_count} tokens",
                        node_id=node.section_id,
                        suggested_fix="Consider merging with adjacent section",
                    )
                )

    @staticmethod
    def _check_depth(tree: HierarchyTree, analysis: HierarchyAnalysis) -> None:
        """Check if tree depth is appropriate."""
        max_depth = tree.max_depth

        if max_depth > 8:
            analysis.issues.append(
                HierarchyIssue(
                    severity="warning",
                    issue_type="too_deep",
                    message=f"Document has {max_depth} levels of nesting (very deep)",
                    suggested_fix=(
                        "Deep nesting may indicate over-structured "
                        "document. Consider flattening some levels."
                    ),
                )
            )
        elif max_depth == 1:
            analysis.issues.append(
                HierarchyIssue(
                    severity="info",
                    issue_type="too_flat",
                    message="Document has only 1 level (very flat structure)",
                    suggested_fix="Consider if subsections could be identified",
                )
            )

    @staticmethod
    def _check_balance(tree: HierarchyTree, analysis: HierarchyAnalysis) -> None:
        """Check if tree is reasonably balanced."""
        # Get depth of all leaf nodes
        leaves = tree.root.get_leaves()
        if len(leaves) < 2:
            return  # Can't check balance with < 2 leaves

        depths = [leaf.depth for leaf in leaves]
        min_depth = min(depths)
        max_depth = max(depths)
        depth_range = max_depth - min_depth

        if depth_range > 4:
            analysis.issues.append(
                HierarchyIssue(
                    severity="info",
                    issue_type="imbalanced",
                    message=f"Tree is imbalanced (depth range: {depth_range} levels)",
                    suggested_fix=(
                        "Some sections are much deeper than others. "
                        "This is OK for complex documents."
                    ),
                )
            )

    @staticmethod
    def _generate_recommendations(
        tree: HierarchyTree, analysis: HierarchyAnalysis
    ) -> None:
        """Generate recommendations based on analysis."""
        stats = tree.get_statistics()

        # Recommend chunking strategy based on structure
        if stats["max_depth"] >= 2 and stats["nodes_with_content"] > 10:
            analysis.recommendations.append(
                "✅ This document has good hierarchy - HIERARCHICAL chunking recommended"
            )
        elif stats["max_depth"] == 1:
            analysis.recommendations.append(
                "ℹ️  Flat structure detected - consider FIXED or SEMANTIC chunking"
            )

        # Recommend adjustments based on section sizes
        avg_tokens = stats["avg_tokens_per_node"]
        if avg_tokens > 800:
            analysis.recommendations.append(
                f"⚠️  Average section size ({avg_tokens:.0f} tokens) is large - "
                f"consider using chunker with smaller max_tokens"
            )
        elif avg_tokens < 100:
            analysis.recommendations.append(
                f"ℹ️  Average section size ({avg_tokens:.0f} tokens) is small - "
                f"sections may be too granular for RAG"
            )

        # Recommend based on content coverage
        content_ratio = (
            stats["nodes_with_content"] / stats["total_nodes"]
            if stats["total_nodes"] > 0
            else 0
        )

        if content_ratio < 0.5:
            analysis.recommendations.append(
                f"⚠️  Only {content_ratio*100:.0f}% of sections have content - "
                f"many headings without content"
            )

        # Recommend based on tree balance
        if len(analysis.issues) == 0:
            analysis.recommendations.append(
                "✅ No issues detected - hierarchy looks good!"
            )

    @staticmethod
    def compare_trees(
        tree1: HierarchyTree, tree2: HierarchyTree
    ) -> dict[str, Any]:
        """
        Compare two hierarchy trees.

        Useful for comparing different extraction methods or parameters.
        """
        stats1 = tree1.get_statistics()
        stats2 = tree2.get_statistics()

        return {
            "tree1": {
                "nodes": stats1["total_nodes"],
                "depth": stats1["max_depth"],
                "avg_tokens": stats1["avg_tokens_per_node"],
            },
            "tree2": {
                "nodes": stats2["total_nodes"],
                "depth": stats2["max_depth"],
                "avg_tokens": stats2["avg_tokens_per_node"],
            },
            "differences": {
                "node_diff": stats2["total_nodes"] - stats1["total_nodes"],
                "depth_diff": stats2["max_depth"] - stats1["max_depth"],
                "token_diff": stats2["avg_tokens_per_node"]
                - stats1["avg_tokens_per_node"],
            },
        }
