"""
Hierarchy-aware chunker.

The recommended chunker for CHONK - respects document structure
by keeping content with its headings.

Now enhanced to use:
- PDF outline/TOC when available (most authoritative)
- TOC-matched heading blocks
- Visual heading detection as fallback
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from chonk.core.document import Block, BlockType, Chunk, ChonkDocument
from chonk.chunkers.base import BaseChunker, ChunkerConfig, ChunkerRegistry


@dataclass
class Section:
    """A section of a document with a heading and content."""

    heading: Block | None
    heading_level: int
    content_blocks: list[Block]
    children: list["Section"]
    parent: "Section | None"

    @property
    def all_blocks(self) -> list[Block]:
        """Get all blocks in this section (heading + content)."""
        blocks = []
        if self.heading:
            blocks.append(self.heading)
        blocks.extend(self.content_blocks)
        return blocks

    @property
    def path(self) -> str:
        """Get the hierarchy path to this section."""
        parts = []
        current: Section | None = self
        while current:
            if current.heading:
                parts.insert(0, current.heading.content)
            current = current.parent
        return " > ".join(parts)


@ChunkerRegistry.register
class HierarchyChunker(BaseChunker):
    """
    Chunk documents while respecting heading hierarchy.

    This is the recommended chunker for most documents:
    - Uses TOC structure when available (most reliable)
    - Falls back to TOC-matched heading blocks
    - Uses visual heading detection as final fallback
    - Groups content under its heading
    - Respects section boundaries
    - Builds hierarchy paths for context
    - Falls back to size-based splitting when sections are too large
    """

    CHUNKER_NAME: ClassVar[str] = "hierarchy"

    def chunk(self, blocks: list[Block]) -> list[Chunk]:
        """Chunk blocks while respecting hierarchy."""
        if not blocks:
            return []

        # Check if we have TOC-enhanced blocks
        has_toc_info = any(
            block.metadata.get("toc_match") for block in blocks
            if block.type == BlockType.HEADING
        )

        if has_toc_info:
            # Use TOC-enhanced chunking
            root = self._build_section_tree_from_toc(blocks)
        else:
            # Fall back to visual hierarchy
            root = self._build_section_tree(blocks)

        # Convert sections to chunks
        chunks = self._sections_to_chunks(root)

        return chunks

    def _build_section_tree_from_toc(self, blocks: list[Block]) -> Section:
        """
        Build section tree prioritizing TOC-matched headings.

        TOC-matched blocks have higher confidence for hierarchy.
        """
        root = Section(
            heading=None,
            heading_level=0,
            content_blocks=[],
            children=[],
            parent=None,
        )

        current_section = root
        section_stack = [root]

        for block in blocks:
            is_heading = block.type == BlockType.HEADING
            toc_match = block.metadata.get("toc_match", False)
            toc_level = block.metadata.get("toc_level")

            if is_heading:
                # Prefer TOC level if available, otherwise use detected level
                if toc_match and toc_level:
                    level = toc_level
                else:
                    level = block.heading_level or 1

                # Find the right parent for this heading level
                while len(section_stack) > 1 and section_stack[-1].heading_level >= level:
                    section_stack.pop()

                parent = section_stack[-1]

                # Create new section
                new_section = Section(
                    heading=block,
                    heading_level=level,
                    content_blocks=[],
                    children=[],
                    parent=parent,
                )
                parent.children.append(new_section)
                section_stack.append(new_section)
                current_section = new_section

            else:
                # Add content to current section
                current_section.content_blocks.append(block)

        return root

    def _build_section_tree(self, blocks: list[Block]) -> Section:
        """Build a tree of sections from blocks."""
        # Root section has no heading
        root = Section(
            heading=None,
            heading_level=0,
            content_blocks=[],
            children=[],
            parent=None,
        )

        current_section = root
        section_stack = [root]

        for block in blocks:
            if block.type == BlockType.HEADING:
                level = block.heading_level or 1

                # Find the right parent for this heading level
                while len(section_stack) > 1 and section_stack[-1].heading_level >= level:
                    section_stack.pop()

                parent = section_stack[-1]

                # Create new section
                new_section = Section(
                    heading=block,
                    heading_level=level,
                    content_blocks=[],
                    children=[],
                    parent=parent,
                )
                parent.children.append(new_section)
                section_stack.append(new_section)
                current_section = new_section

            else:
                # Add content to current section
                current_section.content_blocks.append(block)

        return root

    def _sections_to_chunks(self, root: Section) -> list[Chunk]:
        """Convert section tree to chunks."""
        chunks: list[Chunk] = []

        # Process root content first
        if root.content_blocks:
            self._chunk_section_content(
                root.content_blocks,
                "",
                chunks,
            )

        # Process children
        self._process_section_children(root, chunks)

        return chunks

    def _process_section_children(self, section: Section, chunks: list[Chunk]) -> None:
        """Recursively process section children."""
        for child in section.children:
            self._process_section(child, chunks)

    def _process_section(self, section: Section, chunks: list[Chunk]) -> None:
        """Process a single section into chunks."""
        # Get all blocks in this section
        all_blocks = section.all_blocks

        if not all_blocks:
            # Process children if no content in this section
            self._process_section_children(section, chunks)
            return

        # Calculate total tokens
        total_tokens = sum(self._count_tokens(b.content) for b in all_blocks)

        # If section fits in one chunk, create it
        if total_tokens <= self.config.max_tokens:
            chunks.append(self._create_chunk(all_blocks, section.path))
            # Still process children
            self._process_section_children(section, chunks)

        elif self.config.group_under_headings and section.heading:
            # Section is too large - split content but keep heading with first chunk
            self._chunk_large_section(section, chunks)

        else:
            # Split content without special heading handling
            self._chunk_section_content(
                all_blocks,
                section.path,
                chunks,
            )
            self._process_section_children(section, chunks)

    def _chunk_large_section(self, section: Section, chunks: list[Chunk]) -> None:
        """Chunk a section that's too large to fit in one chunk."""
        heading = section.heading
        content_blocks = section.content_blocks
        path = section.path

        if not content_blocks:
            # Just the heading
            if heading:
                chunks.append(self._create_chunk([heading], path))
            self._process_section_children(section, chunks)
            return

        # Split content into groups that fit
        content_groups = self._split_blocks_by_size(content_blocks)

        # First group includes heading
        for i, group in enumerate(content_groups):
            if i == 0 and heading:
                group_blocks = [heading] + group
            else:
                group_blocks = group

            chunks.append(self._create_chunk(group_blocks, path))

        # Process children
        self._process_section_children(section, chunks)

    def _chunk_section_content(
        self, blocks: list[Block], path: str, chunks: list[Chunk]
    ) -> None:
        """Chunk a list of blocks by size."""
        if not blocks:
            return

        groups = self._split_blocks_by_size(blocks)
        for group in groups:
            chunks.append(self._create_chunk(group, path))

    def _split_blocks_by_size(self, blocks: list[Block]) -> list[list[Block]]:
        """Split blocks into groups that fit within token limits."""
        if not blocks:
            return []

        groups: list[list[Block]] = []
        current_group: list[Block] = []
        current_tokens = 0

        for block in blocks:
            block_tokens = self._count_tokens(block.content)

            # Atomic blocks get their own group
            if self._is_atomic_block(block):
                if current_group:
                    groups.append(current_group)
                    current_group = []
                    current_tokens = 0
                groups.append([block])
                continue

            # If block is larger than max, it needs to be split
            if block_tokens > self.config.max_tokens:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                    current_tokens = 0
                # Split large block
                split_groups = self._split_large_block(block)
                groups.extend(split_groups)
                continue

            # Check if adding this block exceeds target
            if current_tokens + block_tokens > self.config.target_tokens:
                if current_group:
                    groups.append(current_group)
                current_group = [block]
                current_tokens = block_tokens
            else:
                current_group.append(block)
                current_tokens += block_tokens

        if current_group:
            groups.append(current_group)

        return groups

    def _split_large_block(self, block: Block) -> list[list[Block]]:
        """Split a block that's larger than max tokens."""
        # Split by sentences
        content = block.content
        sentences = self._split_into_sentences(content)

        groups: list[list[Block]] = []
        current_text = ""
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            if current_tokens + sentence_tokens > self.config.target_tokens:
                if current_text:
                    # Create a synthetic block for this group
                    synthetic = Block(
                        id=Block.generate_id(),
                        type=block.type,
                        content=current_text.strip(),
                        page=block.page,
                        parent_id=block.id,  # Link to original
                        metadata={
                            **block.metadata,
                            "is_split": True,
                            "original_block_id": block.id,
                        },
                    )
                    groups.append([synthetic])

                current_text = sentence
                current_tokens = sentence_tokens
            else:
                current_text += sentence
                current_tokens += sentence_tokens

        if current_text:
            synthetic = Block(
                id=Block.generate_id(),
                type=block.type,
                content=current_text.strip(),
                page=block.page,
                parent_id=block.id,
                metadata={
                    **block.metadata,
                    "is_split": True,
                    "original_block_id": block.id,
                },
            )
            groups.append([synthetic])

        return groups

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re

        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        # Keep the separators
        result = []
        for s in sentences:
            if s:
                result.append(s + " ")
        return result

    def _is_atomic_block(self, block: Block) -> bool:
        """Check if a block should not be split."""
        if self.config.preserve_tables and block.type == BlockType.TABLE:
            return True
        if self.config.preserve_code and block.type == BlockType.CODE:
            return True
        return False
