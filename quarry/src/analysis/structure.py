"""
Document structure analysis.

Defines the unified structure model and analyzer that reconciles
multiple signals into a coherent document hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from chonk.core.document import Block, BlockType, ChonkDocument


class StructureSource(Enum):
    """Source of structure information."""

    TOC = "toc"  # PDF outline/bookmarks
    TAGGED = "tagged"  # PDF/UA tagged structure
    VISUAL = "visual"  # Font size/style heuristics
    SEMANTIC = "semantic"  # Content-based detection (numbered headings)
    USER = "user"  # User-provided hints


@dataclass
class StructureNode:
    """
    A node in the document structure tree.

    Represents a section of the document with its heading,
    content, and child sections.
    """

    id: str
    title: str
    level: int  # 0 = root, 1 = top-level section, etc.

    # Source information
    source: StructureSource
    confidence: float  # 0.0 - 1.0

    # Location in document
    page_start: int | None = None
    page_end: int | None = None
    block_ids: list[str] = field(default_factory=list)

    # Tree structure
    children: list[StructureNode] = field(default_factory=list)
    parent: StructureNode | None = field(default=None, repr=False)

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> str:
        """Get the full path to this node."""
        parts = []
        node: StructureNode | None = self
        while node and node.title:
            parts.insert(0, node.title)
            node = node.parent
        return " > ".join(parts)

    def add_child(self, child: StructureNode) -> None:
        """Add a child node."""
        child.parent = self
        self.children.append(child)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "level": self.level,
            "source": self.source.value,
            "confidence": self.confidence,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "block_ids": self.block_ids,
            "children": [c.to_dict() for c in self.children],
            "metadata": self.metadata,
        }


@dataclass
class DocumentStructure:
    """
    Complete structure analysis of a document.

    Contains multiple structure signals that can be reconciled
    into a unified hierarchy.
    """

    # The unified/reconciled structure tree
    root: StructureNode

    # Individual structure signals (before reconciliation)
    toc_structure: StructureNode | None = None  # From PDF outline
    tagged_structure: StructureNode | None = None  # From PDF tags
    visual_structure: StructureNode | None = None  # From heuristics

    # Mapping from structure nodes to blocks
    node_to_blocks: dict[str, list[str]] = field(default_factory=dict)

    # Analysis metadata
    has_toc: bool = False
    has_tags: bool = False
    is_tagged_pdf: bool = False

    # Confidence in the overall structure
    structure_confidence: float = 0.0

    def get_section_for_block(self, block_id: str) -> StructureNode | None:
        """Find the section containing a block."""
        return self._find_block_in_tree(self.root, block_id)

    def _find_block_in_tree(
        self, node: StructureNode, block_id: str
    ) -> StructureNode | None:
        """Recursively search for block in tree."""
        if block_id in node.block_ids:
            return node
        for child in node.children:
            result = self._find_block_in_tree(child, block_id)
            if result:
                return result
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "root": self.root.to_dict(),
            "has_toc": self.has_toc,
            "has_tags": self.has_tags,
            "is_tagged_pdf": self.is_tagged_pdf,
            "structure_confidence": self.structure_confidence,
        }


class StructureAnalyzer:
    """
    Analyzes document structure from multiple signals.

    Reconciles TOC, tagged structure, and visual heuristics
    into a unified document hierarchy.
    """

    def __init__(self) -> None:
        self._warnings: list[str] = []

    def analyze(self, document: ChonkDocument) -> DocumentStructure:
        """
        Analyze a document and extract its structure.

        This is the main entry point that coordinates
        extraction from multiple sources.
        """
        self._warnings = []

        # Create root node
        root = StructureNode(
            id="root",
            title="",
            level=0,
            source=StructureSource.VISUAL,
            confidence=1.0,
        )

        structure = DocumentStructure(root=root)

        # Build visual structure from blocks (fallback)
        visual_root = self._build_visual_structure(document.blocks)
        structure.visual_structure = visual_root

        # For PDFs, try to extract TOC and tagged structure
        # (This is handled by PDFStructureExtractor)

        # Reconcile all signals
        self._reconcile_structures(structure, document.blocks)

        return structure

    def _build_visual_structure(self, blocks: list[Block]) -> StructureNode:
        """
        Build structure tree from visual heuristics.

        Uses heading blocks to create hierarchy.
        """
        root = StructureNode(
            id="visual_root",
            title="",
            level=0,
            source=StructureSource.VISUAL,
            confidence=0.6,
        )

        # Stack for tracking current position in hierarchy
        stack: list[StructureNode] = [root]
        current_content_blocks: list[str] = []

        for block in blocks:
            if block.type == BlockType.HEADING:
                level = block.heading_level or 1

                # Assign content blocks to current section
                if stack[-1] != root or current_content_blocks:
                    stack[-1].block_ids.extend(current_content_blocks)
                    current_content_blocks = []

                # Pop stack until we find the right parent
                while len(stack) > 1 and stack[-1].level >= level:
                    stack.pop()

                # Create new section node
                section = StructureNode(
                    id=f"visual_{block.id}",
                    title=block.content[:100].strip(),
                    level=level,
                    source=StructureSource.VISUAL,
                    confidence=self._compute_heading_confidence(block),
                    page_start=block.page,
                    block_ids=[block.id],
                )

                stack[-1].add_child(section)
                stack.append(section)
            else:
                current_content_blocks.append(block.id)

        # Assign remaining content blocks
        if current_content_blocks:
            stack[-1].block_ids.extend(current_content_blocks)

        return root

    def _compute_heading_confidence(self, block: Block) -> float:
        """
        Compute confidence that a block is truly a heading.

        Higher confidence for:
        - Explicit heading level
        - Bold text
        - Numbered format
        - Short text
        """
        confidence = 0.5

        if block.heading_level:
            confidence += 0.2

        if block.metadata.get("is_bold"):
            confidence += 0.1

        # Numbered headings are more reliable
        import re
        if re.match(r"^\d+(\.\d+)*\.?\s", block.content):
            confidence += 0.2

        # Short text is more likely to be a heading
        if len(block.content) < 100:
            confidence += 0.1

        return min(confidence, 1.0)

    def _reconcile_structures(
        self, structure: DocumentStructure, blocks: list[Block]
    ) -> None:
        """
        Reconcile multiple structure signals into unified tree.

        Priority:
        1. TOC (if available) - most authoritative
        2. Tagged structure (if PDF/UA) - semantic
        3. Visual heuristics - fallback
        """
        # Start with visual structure as base
        if structure.visual_structure:
            structure.root = structure.visual_structure
            structure.structure_confidence = 0.6

        # If we have TOC, it takes precedence
        if structure.toc_structure and structure.has_toc:
            structure.root = self._merge_toc_with_visual(
                structure.toc_structure,
                structure.visual_structure,
                blocks,
            )
            structure.structure_confidence = 0.9

        # If we have tagged structure, use it to validate/enhance
        elif structure.tagged_structure and structure.has_tags:
            structure.root = self._merge_tagged_with_visual(
                structure.tagged_structure,
                structure.visual_structure,
                blocks,
            )
            structure.structure_confidence = 0.85

        # Build node_to_blocks mapping
        self._build_block_mapping(structure)

    def _merge_toc_with_visual(
        self,
        toc: StructureNode,
        visual: StructureNode | None,
        blocks: list[Block],
    ) -> StructureNode:
        """
        Merge TOC structure with visual structure.

        TOC provides section titles and page locations.
        Visual provides block boundaries.
        """
        # Map TOC entries to blocks by page
        block_by_page: dict[int, list[Block]] = {}
        for block in blocks:
            if block.page not in block_by_page:
                block_by_page[block.page] = []
            block_by_page[block.page].append(block)

        # Assign blocks to TOC sections
        self._assign_blocks_to_toc(toc, block_by_page, blocks)

        return toc

    def _assign_blocks_to_toc(
        self,
        node: StructureNode,
        block_by_page: dict[int, list[Block]],
        all_blocks: list[Block],
    ) -> None:
        """Recursively assign blocks to TOC sections."""
        if not node.children:
            # Leaf node - assign all blocks from page_start to next section
            if node.page_start:
                # Find blocks on this page after this heading
                page_blocks = block_by_page.get(node.page_start, [])

                # Try to find matching heading block
                for block in page_blocks:
                    if (
                        block.type == BlockType.HEADING
                        and self._titles_match(block.content, node.title)
                    ):
                        node.block_ids.append(block.id)
                        break

                # Assign content blocks (simplified - full implementation would
                # track page ranges more precisely)
                for block in page_blocks:
                    if block.id not in node.block_ids:
                        node.block_ids.append(block.id)
        else:
            # Non-leaf node - assign heading block, then recurse
            if node.page_start:
                page_blocks = block_by_page.get(node.page_start, [])
                for block in page_blocks:
                    if (
                        block.type == BlockType.HEADING
                        and self._titles_match(block.content, node.title)
                    ):
                        node.block_ids.append(block.id)
                        break

            for child in node.children:
                self._assign_blocks_to_toc(child, block_by_page, all_blocks)

    def _titles_match(self, block_title: str, toc_title: str) -> bool:
        """Check if a block title matches a TOC entry."""
        # Normalize and compare
        def normalize(s: str) -> str:
            import re
            s = s.lower().strip()
            s = re.sub(r"^\d+(\.\d+)*\.?\s*", "", s)  # Remove numbering
            s = re.sub(r"\s+", " ", s)
            return s

        return normalize(block_title) == normalize(toc_title)

    def _merge_tagged_with_visual(
        self,
        tagged: StructureNode,
        visual: StructureNode | None,
        blocks: list[Block],
    ) -> StructureNode:
        """
        Merge tagged PDF structure with visual structure.

        Tagged structure provides semantic section info.
        """
        # For now, prefer tagged structure when available
        # Full implementation would do more sophisticated merging
        return tagged

    def _build_block_mapping(self, structure: DocumentStructure) -> None:
        """Build mapping from nodes to blocks."""
        structure.node_to_blocks = {}
        self._collect_block_mapping(structure.root, structure.node_to_blocks)

    def _collect_block_mapping(
        self, node: StructureNode, mapping: dict[str, list[str]]
    ) -> None:
        """Recursively collect block mappings."""
        mapping[node.id] = node.block_ids
        for child in node.children:
            self._collect_block_mapping(child, mapping)
