"""
Recursive character chunker.

Similar to LangChain's RecursiveCharacterTextSplitter - tries
progressively smaller separators to split text.
"""

from __future__ import annotations

import re
from typing import ClassVar

from chonk.core.document import Block, BlockType, Chunk
from chonk.chunkers.base import BaseChunker, ChunkerConfig, ChunkerRegistry


@ChunkerRegistry.register
class RecursiveChunker(BaseChunker):
    """
    Recursively split content using a hierarchy of separators.

    Tries to split on:
    1. Paragraph breaks (double newline)
    2. Single newlines
    3. Sentences (. ! ?)
    4. Spaces
    5. Characters (last resort)

    This produces more natural-feeling chunks than fixed-size.
    """

    CHUNKER_NAME: ClassVar[str] = "recursive"

    # Separators in order of preference (try largest first)
    SEPARATORS = [
        "\n\n",  # Paragraph
        "\n",  # Line
        ". ",  # Sentence
        "! ",
        "? ",
        "; ",  # Clause
        ", ",  # Phrase
        " ",  # Word
        "",  # Character (last resort)
    ]

    def chunk(self, blocks: list[Block]) -> list[Chunk]:
        """Chunk blocks using recursive splitting."""
        if not blocks:
            return []

        chunks: list[Chunk] = []

        for block in blocks:
            # Atomic blocks get their own chunk
            if self._is_atomic_block(block):
                chunks.append(self._create_chunk([block]))
                continue

            # Split block content recursively
            text_chunks = self._split_text(block.content)

            # Create chunks from splits
            for text in text_chunks:
                if text.strip():
                    # Create a synthetic block for each split
                    chunks.append(
                        Chunk(
                            id=Chunk.generate_id(),
                            block_ids=[block.id],
                            content=text.strip(),
                            token_count=self._count_tokens(text),
                            hierarchy_path="",
                            system_metadata={
                                "start_page": block.page,
                                "end_page": block.page,
                                "block_count": 1,
                                "is_split": True,
                            },
                        )
                    )

        # Merge small chunks
        chunks = self._merge_small_chunks(chunks)

        # Add overlap
        chunks = self._add_overlap(chunks)

        return chunks

    def _split_text(self, text: str) -> list[str]:
        """Recursively split text to target size."""
        token_count = self._count_tokens(text)

        # If text is small enough, return it
        if token_count <= self.config.target_tokens:
            return [text]

        # Try each separator
        for separator in self.SEPARATORS:
            if separator and separator in text:
                splits = self._split_by_separator(text, separator)
                if len(splits) > 1:
                    # Recursively process splits and recombine
                    result = []
                    for split in splits:
                        result.extend(self._split_text(split))
                    return self._recombine_splits(result)

        # Last resort: split by character count
        return self._split_by_tokens(text)

    def _split_by_separator(self, text: str, separator: str) -> list[str]:
        """Split text by separator, keeping separator at end of chunks."""
        if not separator:
            return list(text)

        parts = text.split(separator)
        result = []

        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                # Add separator back (except for last part)
                result.append(part + separator)
            else:
                result.append(part)

        return [p for p in result if p.strip()]

    def _recombine_splits(self, splits: list[str]) -> list[str]:
        """Recombine splits that are too small."""
        result = []
        current = ""
        current_tokens = 0

        for split in splits:
            split_tokens = self._count_tokens(split)

            # If adding this would exceed target, save current and start new
            if current_tokens + split_tokens > self.config.target_tokens and current:
                result.append(current)
                current = split
                current_tokens = split_tokens
            else:
                current += split
                current_tokens += split_tokens

        if current:
            result.append(current)

        return result

    def _split_by_tokens(self, text: str) -> list[str]:
        """Split text to approximately target token count."""
        # Estimate chars per token (rough approximation)
        chars_per_token = 4
        target_chars = self.config.target_tokens * chars_per_token

        result = []
        for i in range(0, len(text), target_chars):
            chunk = text[i : i + target_chars]
            # Try to break at word boundary
            if i + target_chars < len(text):
                last_space = chunk.rfind(" ")
                if last_space > target_chars * 0.5:
                    chunk = chunk[:last_space]
            result.append(chunk)

        return result

    def _is_atomic_block(self, block: Block) -> bool:
        """Check if a block should not be split."""
        if self.config.preserve_tables and block.type == BlockType.TABLE:
            return True
        if self.config.preserve_code and block.type == BlockType.CODE:
            return True
        return False

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge chunks that are too small."""
        if not chunks:
            return []

        result: list[Chunk] = []

        for chunk in chunks:
            if not result:
                result.append(chunk)
                continue

            last = result[-1]

            # Merge if last chunk is too small
            if last.token_count < self.config.min_tokens:
                # Merge content
                merged_content = last.content + "\n\n" + chunk.content
                merged_tokens = self._count_tokens(merged_content)

                # Only merge if result isn't too big
                if merged_tokens <= self.config.max_tokens:
                    merged = Chunk(
                        id=last.id,
                        block_ids=list(set(last.block_ids + chunk.block_ids)),
                        content=merged_content,
                        token_count=merged_tokens,
                        hierarchy_path=last.hierarchy_path,
                        system_metadata={
                            **last.system_metadata,
                            "merged": True,
                        },
                    )
                    result[-1] = merged
                    continue

            result.append(chunk)

        return result

    def _add_overlap(self, chunks: list[Chunk]) -> list[Chunk]:
        """Add overlap between chunks."""
        if self.config.overlap_tokens == 0 or len(chunks) < 2:
            return chunks

        result = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk)
                continue

            # Get overlap from previous chunk
            prev = chunks[i - 1]
            overlap_text = self._get_overlap_text(prev.content)

            if overlap_text:
                new_content = overlap_text + "\n\n" + chunk.content
                new_chunk = Chunk(
                    id=chunk.id,
                    block_ids=chunk.block_ids,
                    content=new_content,
                    token_count=self._count_tokens(new_content),
                    hierarchy_path=chunk.hierarchy_path,
                    system_metadata={
                        **chunk.system_metadata,
                        "has_overlap": True,
                    },
                )
                result.append(new_chunk)
            else:
                result.append(chunk)

        return result

    def _get_overlap_text(self, text: str) -> str:
        """Get the last N tokens worth of text for overlap."""
        if not text:
            return ""

        # Work backwards from end
        words = text.split()
        overlap_words = []
        token_count = 0

        for word in reversed(words):
            word_tokens = self._count_tokens(word)
            if token_count + word_tokens <= self.config.overlap_tokens:
                overlap_words.insert(0, word)
                token_count += word_tokens
            else:
                break

        return " ".join(overlap_words)
