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
        repair_orphans: bool = False,
        structural_hints: dict[str, Any] | None = None,
    ) -> HierarchyTree:
        """Build a hierarchy tree from a list of blocks.

        Strategy:
        1. Create root node
        2. For each heading block, create a node
        3. Attach content blocks to the current heading
        4. Build parent-child relationships based on heading levels
        5. Optionally repair orphaned headings

        Args:
            blocks: List of blocks from extraction.
            document_id: ID for the document.
            metadata: Optional metadata for the tree.
            repair_orphans: If True, merge orphan headings into parents.
            structural_hints: Optional dict from Tier 1 analysis.

        Returns:
            HierarchyTree with full structure.
        """
        tree_metadata = dict(metadata or {})
        if structural_hints:
            tree_metadata.update(structural_hints)

        root = HierarchyNode(
            section_id="root",
            heading=None,
            level=0,
        )

        current_node = root
        node_stack = [root]

        for block in blocks:
            if block.type == BlockType.HEADING:
                section_id = HierarchyBuilder.extract_section_id(block.content)
                heading_level = block.heading_level or 1

                node = HierarchyNode(
                    section_id=section_id,
                    heading=block.content,
                    heading_block=block,
                    level=heading_level,
                )

                while len(node_stack) > 1 and node_stack[-1].level >= heading_level:
                    node_stack.pop()

                parent = node_stack[-1]
                parent.add_child(node)
                node_stack.append(node)
                current_node = node

            else:
                current_node.add_content(block)

        tree = HierarchyTree(
            root=root,
            document_id=document_id,
            metadata=tree_metadata,
        )

        if repair_orphans:
            HierarchyBuilder._repair_orphans(root)

        return tree

    @staticmethod
    def _repair_orphans(node: HierarchyNode) -> None:
        """Remove orphan leaf headings by merging them into their parent.

        An orphan is a heading with no content and no children.
        Its heading text is appended as a content block to the parent.

        Args:
            node: The node whose children to repair (recursive).
        """
        # Process children bottom-up so nested orphans resolve first
        for child in list(node.children):
            HierarchyBuilder._repair_orphans(child)

        orphans = [
            c for c in node.children
            if c.heading and not c.content_blocks and not c.children
        ]
        for orphan in orphans:
            node.children.remove(orphan)
            # Convert the orphan heading into a content block on the parent
            fallback_block = Block(
                id=Block.generate_id(),
                type=BlockType.TEXT,
                content=orphan.heading or "",
                page=orphan.heading_block.page if orphan.heading_block else 1,
            )
            node.content_blocks.append(fallback_block)

    @staticmethod
    def build_from_blocks_with_profile(
        blocks: list[Block],
        document_id: str,
        fingerprint: Any,
        document_type: Any,
        repair_orphans: bool = True,
    ) -> HierarchyTree:
        """Build hierarchy enriched with Tier 1 structural profile.

        Uses the fingerprint and document type classification to add
        metadata about the document's structure and numbering scheme.

        Args:
            blocks: List of blocks from extraction.
            document_id: ID for the document.
            fingerprint: DocumentFingerprint from Tier 1.
            document_type: DocumentType enum value.
            repair_orphans: If True, merge orphan headings into parents.

        Returns:
            HierarchyTree with structural metadata.
        """
        from chonk.hierarchy.numbering import NumberingScheme

        # Extract heading texts for numbering detection
        heading_texts = [
            b.content for b in blocks if b.type == BlockType.HEADING
        ]
        scheme = NumberingScheme.detect(heading_texts)

        # Build fingerprint summary from available features
        fp_summary = HierarchyBuilder._build_fingerprint_summary(fingerprint)

        metadata: dict[str, Any] = {
            "document_type": document_type.value
            if hasattr(document_type, "value")
            else str(document_type),
            "numbering_scheme": scheme.to_dict(),
            "fingerprint_summary": fp_summary,
        }

        return HierarchyBuilder.build_from_blocks(
            blocks=blocks,
            document_id=document_id,
            metadata=metadata,
            repair_orphans=repair_orphans,
        )

    @staticmethod
    def _build_fingerprint_summary(fingerprint: Any) -> dict[str, Any]:
        """Extract key structural info from a DocumentFingerprint.

        Args:
            fingerprint: DocumentFingerprint from Tier 1.

        Returns:
            Dictionary with key structural fields.
        """
        summary: dict[str, Any] = {
            "page_count": 0,
            "has_toc": False,
            "font_count": 0,
        }
        if hasattr(fingerprint, "byte_features"):
            bf = fingerprint.byte_features
            summary["page_count"] = getattr(bf, "page_count", 0)
        if hasattr(fingerprint, "structural_rhythm"):
            sr = fingerprint.structural_rhythm
            summary["has_toc"] = getattr(sr, "toc_present", False)
        if hasattr(fingerprint, "font_features"):
            ff = fingerprint.font_features
            summary["font_count"] = getattr(ff, "font_count", 0)
        return summary

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
