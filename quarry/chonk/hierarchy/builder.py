"""
Hierarchy tree builder.

Builds hierarchy trees from extracted blocks.
"""

from __future__ import annotations

import re
from typing import Any

from chonk.core.document import Block, BlockType
from chonk.hierarchy.tree import HierarchyNode, HierarchyTree


class HierarchyBuilder:
    """
    Builds hierarchy trees from blocks.

    This is the core of CHONK's organization logic - taking flat blocks
    and building meaningful section structure.
    """

    @staticmethod
    def extract_section_id(heading_text: str) -> str:
        """
        Extract section ID from heading text.

        Examples:
        "E.5.3.5 Maintenance work packages..." -> "E.5.3.5"
        "3.66 Graphic(s)." -> "3.66"
        "2.1.1 Safety" -> "2.1.1"
        "FOREWORD" -> "FOREWORD"
        "Introduction" -> "Introduction"
        """
        # Try numbered section (e.g., "3.66", "2.1.1")
        match = re.match(r"^(\d+(?:\.\d+)*)", heading_text.strip())
        if match:
            return match.group(1)

        # Try letter-prefixed section (e.g., "A.1", "E.5.3.5")
        match = re.match(r"^([A-Z](?:\.\d+)+)", heading_text.strip())
        if match:
            return match.group(1)

        # Otherwise use first 50 chars as ID (cleaned)
        clean_id = heading_text.strip()[:50]
        # Remove special chars that might break JSON/APIs
        clean_id = re.sub(r'[<>:"/\\|?*]', "", clean_id)
        return clean_id

    @staticmethod
    def build_from_blocks(
        blocks: list[Block],
        document_id: str = "doc",
        metadata: dict[str, Any] | None = None,
    ) -> HierarchyTree:
        """
        Build a hierarchy tree from a list of blocks.

        Strategy:
        1. Create root node
        2. For each heading block, create a node
        3. Attach content blocks to the current heading
        4. Build parent-child relationships based on heading levels

        Args:
            blocks: List of blocks from extraction
            document_id: ID for the document
            metadata: Optional metadata for the tree

        Returns:
            HierarchyTree with full structure
        """
        root = HierarchyNode(
            section_id="root",
            heading=None,
            level=0,
        )

        current_node = root
        node_stack = [root]  # Stack to track hierarchy

        for block in blocks:
            if block.type == BlockType.HEADING:
                # Create new section node
                section_id = HierarchyBuilder.extract_section_id(block.content)
                heading_level = block.heading_level or 1

                node = HierarchyNode(
                    section_id=section_id,
                    heading=block.content,
                    heading_block=block,
                    level=heading_level,
                )

                # Find the right parent based on level
                # Pop stack until we find a parent with lower level
                while len(node_stack) > 1 and node_stack[-1].level >= heading_level:
                    node_stack.pop()

                parent = node_stack[-1]
                parent.add_child(node)
                node_stack.append(node)
                current_node = node

            else:
                # Add content to current section
                current_node.add_content(block)

        return HierarchyTree(
            root=root,
            document_id=document_id,
            metadata=metadata or {},
        )

    @staticmethod
    def build_from_docling_result(
        extraction_result: Any, document_id: str = "doc"
    ) -> HierarchyTree:
        """
        Build hierarchy tree from a Docling extraction result.

        Args:
            extraction_result: Result from DoclingExtractor.extract()
            document_id: ID for the document

        Returns:
            HierarchyTree with full structure
        """
        return HierarchyBuilder.build_from_blocks(
            blocks=extraction_result.blocks,
            document_id=document_id,
            metadata={
                "extraction_tier": extraction_result.tier_used.value,
                "extractor": extraction_result.extraction_info.get("extractor", "unknown"),
                "page_count": extraction_result.metadata.page_count,
                "word_count": extraction_result.metadata.word_count,
            },
        )

    @staticmethod
    def rebuild_with_custom_rules(
        tree: HierarchyTree,
        merge_headings: list[str] | None = None,
        split_at: list[str] | None = None,
    ) -> HierarchyTree:
        """
        Rebuild tree with custom rules.

        This allows users to refine the automatically-built tree.

        Args:
            tree: Original tree
            merge_headings: List of section IDs to merge with parent
            split_at: List of section IDs to split into subsections

        Returns:
            New HierarchyTree with modifications applied
        """
        # TODO: Implement custom rebuilding logic
        # This is for future UI-driven refinement
        raise NotImplementedError("Custom tree rebuilding coming soon!")

    @staticmethod
    def flatten_to_sections(tree: HierarchyTree) -> list[dict[str, Any]]:
        """
        Flatten tree to a list of sections for easier processing.

        Each section includes heading, content, and hierarchy path.

        Returns:
            List of section dictionaries
        """
        sections = []

        for node in tree.get_all_nodes():
            if node.heading:  # Skip root
                sections.append(
                    {
                        "section_id": node.section_id,
                        "heading": node.heading,
                        "content": node.content_text,
                        "heading_block_id": node.heading_block.id
                        if node.heading_block
                        else None,
                        "content_block_ids": [b.id for b in node.content_blocks],
                        "level": node.level,
                        "token_count": node.token_count,
                        "page_range": node.page_range,
                        "hierarchy_path": node.hierarchy_path,
                        "has_children": len(node.children) > 0,
                        "child_count": len(node.children),
                    }
                )

        return sections
