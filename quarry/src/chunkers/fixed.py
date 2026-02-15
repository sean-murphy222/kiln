"""
Fixed-size chunker with overlap.

Simple chunking strategy that creates chunks of approximately
equal token size with configurable overlap.
"""

from __future__ import annotations

from typing import ClassVar

from chonk.core.document import Block, BlockType, Chunk
from chonk.chunkers.base import BaseChunker, ChunkerConfig, ChunkerRegistry


@ChunkerRegistry.register
class FixedSizeChunker(BaseChunker):
    """
    Create chunks of approximately fixed token size.

    This is the simplest chunking strategy:
    - Groups blocks until target token count is reached
    - Adds overlap by repeating ending content in next chunk
    - Optionally respects block boundaries (doesn't split mid-block)
    """

    CHUNKER_NAME: ClassVar[str] = "fixed"

    def chunk(self, blocks: list[Block]) -> list[Chunk]:
        """Chunk blocks into fixed-size chunks."""
        if not blocks:
            return []

        chunks: list[Chunk] = []
        current_blocks: list[Block] = []
        current_tokens = 0

        for block in blocks:
            block_tokens = self._count_tokens(block.content)

            # Check if this block should be its own chunk (tables, code)
            if self._is_atomic_block(block):
                # Save current chunk if any
                if current_blocks:
                    chunks.append(self._create_chunk(current_blocks))
                    current_blocks = []
                    current_tokens = 0

                # Create chunk for atomic block
                chunks.append(self._create_chunk([block]))
                continue

            # Check if adding this block exceeds max
            if current_tokens + block_tokens > self.config.max_tokens and current_blocks:
                # Create chunk from current blocks
                chunks.append(self._create_chunk(current_blocks))

                # Calculate overlap
                overlap_blocks = self._get_overlap_blocks(current_blocks)
                current_blocks = overlap_blocks
                current_tokens = sum(
                    self._count_tokens(b.content) for b in overlap_blocks
                )

            # Add block to current chunk
            current_blocks.append(block)
            current_tokens += block_tokens

            # Check if we've hit the target
            if current_tokens >= self.config.target_tokens:
                chunks.append(self._create_chunk(current_blocks))

                # Calculate overlap
                overlap_blocks = self._get_overlap_blocks(current_blocks)
                current_blocks = overlap_blocks
                current_tokens = sum(
                    self._count_tokens(b.content) for b in overlap_blocks
                )

        # Don't forget the last chunk
        if current_blocks:
            # Merge with previous chunk if too small
            if (
                current_tokens < self.config.min_tokens
                and chunks
                and not self._is_atomic_block(blocks[-1])
            ):
                # Merge with last chunk
                last_chunk = chunks[-1]
                merged_block_ids = last_chunk.block_ids + [b.id for b in current_blocks if b.id not in last_chunk.block_ids]
                merged_blocks = [b for b in blocks if b.id in merged_block_ids]
                chunks[-1] = self._create_chunk(merged_blocks)
            else:
                chunks.append(self._create_chunk(current_blocks))

        return chunks

    def _is_atomic_block(self, block: Block) -> bool:
        """Check if a block should not be split or merged."""
        if self.config.preserve_tables and block.type == BlockType.TABLE:
            return True
        if self.config.preserve_code and block.type == BlockType.CODE:
            return True
        return False

    def _get_overlap_blocks(self, blocks: list[Block]) -> list[Block]:
        """Get blocks for overlap with next chunk."""
        if self.config.overlap_tokens == 0 or not blocks:
            return []

        overlap_blocks: list[Block] = []
        overlap_tokens = 0

        # Work backwards from the end
        for block in reversed(blocks):
            block_tokens = self._count_tokens(block.content)
            if overlap_tokens + block_tokens <= self.config.overlap_tokens:
                overlap_blocks.insert(0, block)
                overlap_tokens += block_tokens
            else:
                break

        return overlap_blocks
