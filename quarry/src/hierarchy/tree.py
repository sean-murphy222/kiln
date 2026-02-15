"""
Hierarchy tree data structures.

The core data structures for representing document hierarchy in CHONK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chonk.core.document import Block


@dataclass
class HierarchyNode:
    """
    A node in the document hierarchy tree.

    This is the core data structure for CHONK's organization features.
    Each node represents a section in the document with:
    - Separated heading and content
    - Parent-child relationships
    - Traceability to source blocks
    - Metadata for chunking decisions
    """

    section_id: str
    heading: str | None = None
    heading_block: Block | None = None
    level: int = 0
    content_blocks: list[Block] = field(default_factory=list)
    children: list[HierarchyNode] = field(default_factory=list)
    parent: HierarchyNode | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_child(self, child: HierarchyNode) -> None:
        """Add a child node to this section."""
        child.parent = self
        self.children.append(child)

    def add_content(self, block: Block) -> None:
        """Add a content block to this section."""
        self.content_blocks.append(block)

    @property
    def content_text(self) -> str:
        """Get combined content text (without heading)."""
        return "\n\n".join(block.content for block in self.content_blocks)

    @property
    def full_text(self) -> str:
        """Get heading + content together."""
        parts = []
        if self.heading:
            parts.append(self.heading)
        if self.content_text:
            parts.append(self.content_text)
        return "\n\n".join(parts)

    @property
    def token_count(self) -> int:
        """Calculate total tokens including heading and content."""
        from chonk.utils.tokens import TokenCounter

        counter = TokenCounter()
        total = 0

        if self.heading_block:
            total += counter.count(self.heading_block.content)
        for block in self.content_blocks:
            total += counter.count(block.content)

        return total

    @property
    def page_range(self) -> list[int]:
        """Get page range for this section."""
        pages = set()

        if self.heading_block and self.heading_block.page:
            pages.add(self.heading_block.page)

        for block in self.content_blocks:
            if block.page:
                pages.add(block.page)

        if pages:
            return [min(pages), max(pages)]
        return [0, 0]

    @property
    def hierarchy_path(self) -> str:
        """
        Get the full hierarchy path to this node.

        Example: "Section 2 > 2.1 General > 2.1.1 Safety"
        """
        parts = []
        current: HierarchyNode | None = self

        while current and current.heading:
            parts.insert(0, current.heading)
            current = current.parent

        return " > ".join(parts)

    @property
    def depth(self) -> int:
        """Get depth of this node in the tree (root = 0)."""
        depth = 0
        current = self.parent
        while current:
            depth += 1
            current = current.parent
        return depth

    @property
    def is_leaf(self) -> bool:
        """Check if this node is a leaf (no children)."""
        return len(self.children) == 0

    @property
    def descendant_count(self) -> int:
        """Count all descendants (children, grandchildren, etc.)."""
        count = len(self.children)
        for child in self.children:
            count += child.descendant_count
        return count

    def get_all_descendants(self) -> list[HierarchyNode]:
        """Get all descendants as a flat list (DFS order)."""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_all_descendants())
        return descendants

    def get_leaves(self) -> list[HierarchyNode]:
        """Get all leaf nodes under this node."""
        if self.is_leaf:
            return [self]

        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves

    def to_dict(self, include_children: bool = True) -> dict[str, Any]:
        """
        Convert to dictionary for JSON export.

        Args:
            include_children: If True, recursively include children
        """
        result = {
            "section_id": self.section_id,
            "heading": self.heading,
            "heading_block_id": self.heading_block.id if self.heading_block else None,
            "heading_level": self.level,
            "content": self.content_text,
            "content_block_ids": [b.id for b in self.content_blocks],
            "token_count": self.token_count,
            "page_range": self.page_range,
            "block_count": len(self.content_blocks)
            + (1 if self.heading_block else 0),
            "hierarchy_path": self.hierarchy_path,
            "depth": self.depth,
            "is_leaf": self.is_leaf,
            "child_count": len(self.children),
            "metadata": self.metadata,
        }

        if include_children:
            result["children"] = [child.to_dict(True) for child in self.children]

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any], blocks_by_id: dict[str, Block]) -> HierarchyNode:
        """
        Reconstruct from dictionary.

        Args:
            data: Dictionary from to_dict()
            blocks_by_id: Mapping of block IDs to Block objects
        """
        # Get heading block
        heading_block = None
        if data.get("heading_block_id"):
            heading_block = blocks_by_id.get(data["heading_block_id"])

        # Get content blocks
        content_blocks = []
        for block_id in data.get("content_block_ids", []):
            if block_id in blocks_by_id:
                content_blocks.append(blocks_by_id[block_id])

        # Create node
        node = cls(
            section_id=data["section_id"],
            heading=data.get("heading"),
            heading_block=heading_block,
            level=data.get("heading_level", 0),
            content_blocks=content_blocks,
            metadata=data.get("metadata", {}),
        )

        # Recursively add children
        for child_data in data.get("children", []):
            child = cls.from_dict(child_data, blocks_by_id)
            node.add_child(child)

        return node

    def __repr__(self) -> str:
        """String representation for debugging."""
        heading_preview = (self.heading or "ROOT")[:40]
        return f"<HierarchyNode {self.section_id} '{heading_preview}' level={self.level} children={len(self.children)}>"


@dataclass
class HierarchyTree:
    """
    A complete document hierarchy tree.

    Wraps the root node and provides tree-level operations.
    """

    root: HierarchyNode
    document_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_nodes(self) -> int:
        """Count all nodes in the tree (excluding root if it's empty)."""
        if not self.root.heading and not self.root.content_blocks:
            # Root is just a container
            return self.root.descendant_count
        return 1 + self.root.descendant_count

    @property
    def max_depth(self) -> int:
        """Get maximum depth of the tree."""
        def get_max_depth(node: HierarchyNode) -> int:
            if not node.children:
                return node.depth
            return max(get_max_depth(child) for child in node.children)

        return get_max_depth(self.root)

    @property
    def leaf_count(self) -> int:
        """Count leaf nodes (sections with no subsections)."""
        return len(self.root.get_leaves())

    def get_all_nodes(self, include_root: bool = False) -> list[HierarchyNode]:
        """Get all nodes as a flat list (DFS order)."""
        nodes = []

        if include_root:
            nodes.append(self.root)

        nodes.extend(self.root.get_all_descendants())
        return nodes

    def get_node_by_id(self, section_id: str) -> HierarchyNode | None:
        """Find a node by its section_id."""
        def search(node: HierarchyNode) -> HierarchyNode | None:
            if node.section_id == section_id:
                return node
            for child in node.children:
                result = search(child)
                if result:
                    return result
            return None

        return search(self.root)

    def get_nodes_at_level(self, level: int) -> list[HierarchyNode]:
        """Get all nodes at a specific heading level."""
        return [
            node
            for node in self.get_all_nodes()
            if node.level == level
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get tree statistics for analysis."""
        all_nodes = self.get_all_nodes()
        nodes_with_content = [n for n in all_nodes if n.content_blocks]
        nodes_with_children = [n for n in all_nodes if n.children]

        # Token distribution
        token_counts = [n.token_count for n in nodes_with_content]

        return {
            "total_nodes": len(all_nodes),
            "nodes_with_content": len(nodes_with_content),
            "nodes_with_children": len(nodes_with_children),
            "leaf_nodes": self.leaf_count,
            "max_depth": self.max_depth,
            "avg_tokens_per_node": sum(token_counts) / len(token_counts)
            if token_counts
            else 0,
            "min_tokens": min(token_counts) if token_counts else 0,
            "max_tokens": max(token_counts) if token_counts else 0,
            "level_distribution": self._get_level_distribution(),
        }

    def _get_level_distribution(self) -> dict[int, int]:
        """Get count of nodes at each level."""
        from collections import Counter

        all_nodes = self.get_all_nodes()
        levels = [node.level for node in all_nodes if node.heading]
        return dict(Counter(levels))

    def to_dict(self) -> dict[str, Any]:
        """Convert entire tree to dictionary."""
        return {
            "document_id": self.document_id,
            "metadata": self.metadata,
            "statistics": self.get_statistics(),
            "root": self.root.to_dict(include_children=True),
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], blocks_by_id: dict[str, Block]
    ) -> HierarchyTree:
        """Reconstruct tree from dictionary."""
        root = HierarchyNode.from_dict(data["root"], blocks_by_id)

        return cls(
            root=root,
            document_id=data["document_id"],
            metadata=data.get("metadata", {}),
        )

    def print_tree(self, max_depth: int = 3, show_content: bool = False) -> None:
        """Print tree structure for debugging."""
        def print_node(node: HierarchyNode, indent: int = 0) -> None:
            if indent > max_depth:
                return

            prefix = "  " * indent
            heading_preview = (node.heading or "ROOT")[:60]

            print(f"{prefix}[{node.section_id}] {heading_preview}")
            print(
                f"{prefix}  Level: {node.level}, Blocks: {len(node.content_blocks)}, "
                f"Tokens: {node.token_count}, Pages: {node.page_range}"
            )

            if show_content and node.content_text:
                content_preview = node.content_text[:80].replace("\n", " ")
                print(f"{prefix}  Content: {content_preview}...")

            for child in node.children:
                print_node(child, indent + 1)

        print_node(self.root)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<HierarchyTree doc={self.document_id} "
            f"nodes={self.total_nodes} "
            f"depth={self.max_depth}>"
        )
