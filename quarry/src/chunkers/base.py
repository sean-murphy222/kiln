"""
Base chunker class and registry for document chunking strategies.

All chunkers inherit from BaseChunker and register themselves
with the ChunkerRegistry for easy selection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar

from chonk.core.document import Block, Chunk, ChonkDocument


@dataclass
class ChunkerConfig:
    """Configuration for chunking strategies."""

    # Size targets
    target_tokens: int = 400  # Target chunk size in tokens
    max_tokens: int = 600  # Maximum chunk size
    min_tokens: int = 100  # Minimum chunk size (prefer merging smaller)
    overlap_tokens: int = 50  # Token overlap between chunks

    # Behavior
    respect_boundaries: bool = True  # Don't split sentences/paragraphs mid-way
    preserve_tables: bool = True  # Keep tables as single chunks
    preserve_code: bool = True  # Keep code blocks as single chunks

    # Hierarchy options
    heading_weight: float = 1.5  # How much headings influence chunk boundaries
    group_under_headings: bool = True  # Keep content with its heading

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_tokens": self.target_tokens,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "overlap_tokens": self.overlap_tokens,
            "respect_boundaries": self.respect_boundaries,
            "preserve_tables": self.preserve_tables,
            "preserve_code": self.preserve_code,
            "heading_weight": self.heading_weight,
            "group_under_headings": self.group_under_headings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkerConfig:
        return cls(
            target_tokens=data.get("target_tokens", 400),
            max_tokens=data.get("max_tokens", 600),
            min_tokens=data.get("min_tokens", 100),
            overlap_tokens=data.get("overlap_tokens", 50),
            respect_boundaries=data.get("respect_boundaries", True),
            preserve_tables=data.get("preserve_tables", True),
            preserve_code=data.get("preserve_code", True),
            heading_weight=data.get("heading_weight", 1.5),
            group_under_headings=data.get("group_under_headings", True),
        )


class BaseChunker(ABC):
    """
    Abstract base class for document chunkers.

    Chunkers take a list of Blocks and group them into Chunks
    according to their strategy.
    """

    CHUNKER_NAME: ClassVar[str] = "base"

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()
        self._token_counter: TokenCounter | None = None

    @property
    def token_counter(self) -> "TokenCounter":
        """Lazy-load token counter."""
        if self._token_counter is None:
            self._token_counter = TokenCounter()
        return self._token_counter

    @abstractmethod
    def chunk(self, blocks: list[Block]) -> list[Chunk]:
        """
        Convert blocks into chunks.

        Args:
            blocks: List of blocks from document loader

        Returns:
            List of chunks
        """
        pass

    def chunk_document(self, document: ChonkDocument) -> ChonkDocument:
        """
        Chunk a document and update it with the results.

        Args:
            document: Document to chunk

        Returns:
            Updated document with chunks
        """
        chunks = self.chunk(document.blocks)

        # Update document
        document.chunks = chunks
        document.chunker_used = self.CHUNKER_NAME
        document.chunker_config = self.config.to_dict()
        document.last_chunked_at = datetime.now()

        return document

    def _create_chunk(
        self,
        blocks: list[Block],
        hierarchy_path: str = "",
    ) -> Chunk:
        """Helper to create a chunk from blocks."""
        content = self._merge_block_content(blocks)
        token_count = self.token_counter.count(content)

        # Get page range
        pages = [b.page for b in blocks]
        start_page = min(pages) if pages else 1
        end_page = max(pages) if pages else 1

        return Chunk(
            id=Chunk.generate_id(),
            block_ids=[b.id for b in blocks],
            content=content,
            token_count=token_count,
            hierarchy_path=hierarchy_path,
            system_metadata={
                "start_page": start_page,
                "end_page": end_page,
                "block_count": len(blocks),
            },
        )

    def _merge_block_content(self, blocks: list[Block]) -> str:
        """Merge block content with appropriate separators."""
        parts = []
        for block in blocks:
            parts.append(block.content)
        return "\n\n".join(parts)

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self.token_counter.count(text)


class TokenCounter:
    """
    Token counter using tiktoken for accurate token counts.

    Uses cl100k_base encoding (GPT-4/ChatGPT tokenizer).
    """

    def __init__(self, model: str = "cl100k_base") -> None:
        self._model = model
        self._encoder = None

    @property
    def encoder(self):
        """Lazy-load encoder."""
        if self._encoder is None:
            import tiktoken

            self._encoder = tiktoken.get_encoding(self._model)
        return self._encoder

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        return len(self.encoder.encode(text))


class ChunkerRegistry:
    """
    Registry of available chunking strategies.

    Use this to select chunkers by name.
    """

    _chunkers: ClassVar[dict[str, type[BaseChunker]]] = {}

    @classmethod
    def register(cls, chunker_class: type[BaseChunker]) -> type[BaseChunker]:
        """
        Register a chunker class. Can be used as a decorator.

        @ChunkerRegistry.register
        class MyChunker(BaseChunker):
            ...
        """
        cls._chunkers[chunker_class.CHUNKER_NAME] = chunker_class
        return chunker_class

    @classmethod
    def get_chunker(
        cls, name: str, config: ChunkerConfig | None = None
    ) -> BaseChunker | None:
        """Get a chunker by name."""
        chunker_class = cls._chunkers.get(name)
        if chunker_class:
            return chunker_class(config)
        return None

    @classmethod
    def available_chunkers(cls) -> list[str]:
        """Get list of available chunker names."""
        return list(cls._chunkers.keys())

    @classmethod
    def chunk_document(
        cls, document: ChonkDocument, chunker_name: str, config: ChunkerConfig | None = None
    ) -> ChonkDocument:
        """
        Chunk a document using the specified chunker.

        Raises:
            ValueError: If chunker not found
        """
        chunker = cls.get_chunker(chunker_name, config)
        if chunker is None:
            available = ", ".join(cls.available_chunkers())
            raise ValueError(f"Unknown chunker: {chunker_name}. Available: {available}")
        return chunker.chunk_document(document)
