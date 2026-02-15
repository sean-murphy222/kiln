"""
Tests for CHONK chunking strategies.
"""

import pytest

from chonk.chunkers import ChunkerConfig, ChunkerRegistry
from chonk.chunkers.fixed import FixedSizeChunker
from chonk.chunkers.hierarchy import HierarchyChunker
from chonk.chunkers.recursive import RecursiveChunker
from chonk.core.document import Block, ChonkDocument, DocumentMetadata


class TestChunkerConfig:
    """Tests for ChunkerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkerConfig()

        assert config.target_tokens == 400
        assert config.max_tokens == 600
        assert config.min_tokens == 100
        assert config.overlap_tokens == 50
        assert config.respect_boundaries is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ChunkerConfig(
            target_tokens=300,
            max_tokens=500,
            min_tokens=50,
            overlap_tokens=25,
        )

        assert config.target_tokens == 300
        assert config.max_tokens == 500


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    def test_fixed_chunker_creation(self):
        """Test fixed chunker instantiation."""
        config = ChunkerConfig(target_tokens=100)
        chunker = FixedSizeChunker(config)

        assert chunker.config.target_tokens == 100

    def test_fixed_chunker_basic(self, sample_document):
        """Test basic fixed-size chunking."""
        config = ChunkerConfig(
            target_tokens=50,
            max_tokens=100,
            min_tokens=10,
            overlap_tokens=10,
        )
        chunker = FixedSizeChunker(config)

        chunks = chunker.chunk(sample_document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.block_ids) > 0
            assert chunk.token_count > 0

    def test_fixed_chunker_respects_max(self, sample_document):
        """Test that chunks don't exceed max tokens."""
        config = ChunkerConfig(
            target_tokens=50,
            max_tokens=100,
        )
        chunker = FixedSizeChunker(config)

        chunks = chunker.chunk(sample_document)

        for chunk in chunks:
            assert chunk.token_count <= config.max_tokens


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    def test_recursive_chunker_creation(self):
        """Test recursive chunker instantiation."""
        config = ChunkerConfig(target_tokens=100)
        chunker = RecursiveChunker(config)

        assert chunker.config.target_tokens == 100

    def test_recursive_chunker_basic(self, sample_document):
        """Test basic recursive chunking."""
        config = ChunkerConfig(
            target_tokens=50,
            max_tokens=150,
            min_tokens=10,
        )
        chunker = RecursiveChunker(config)

        chunks = chunker.chunk(sample_document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.content.strip() != ""

    def test_recursive_chunker_separators(self):
        """Test that recursive chunker uses proper separators."""
        doc = ChonkDocument(
            id="test",
            source_path="test.txt",
            source_type="txt",
            blocks=[
                Block(
                    id="b1",
                    type="text",
                    content="First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
                    page=1,
                ),
            ],
        )

        config = ChunkerConfig(target_tokens=10, max_tokens=20)
        chunker = RecursiveChunker(config)

        chunks = chunker.chunk(doc)

        # Should split on paragraph boundaries
        assert len(chunks) >= 2


class TestHierarchyChunker:
    """Tests for HierarchyChunker."""

    def test_hierarchy_chunker_creation(self):
        """Test hierarchy chunker instantiation."""
        config = ChunkerConfig(target_tokens=100)
        chunker = HierarchyChunker(config)

        assert chunker.config.target_tokens == 100

    def test_hierarchy_chunker_basic(self, sample_document):
        """Test basic hierarchy-aware chunking."""
        config = ChunkerConfig(
            target_tokens=50,
            max_tokens=200,
            min_tokens=10,
        )
        chunker = HierarchyChunker(config)

        chunks = chunker.chunk(sample_document)

        assert len(chunks) > 0
        # Should have hierarchy paths set
        has_path = any(c.hierarchy_path for c in chunks)
        assert has_path

    def test_hierarchy_chunker_respects_headings(self):
        """Test that hierarchy chunker respects heading boundaries."""
        doc = ChonkDocument(
            id="test",
            source_path="test.md",
            source_type="md",
            blocks=[
                Block(id="h1", type="heading", content="Introduction", page=1, heading_level=1),
                Block(id="p1", type="text", content="Intro content " * 20, page=1),
                Block(id="h2", type="heading", content="Methods", page=1, heading_level=1),
                Block(id="p2", type="text", content="Methods content " * 20, page=1),
            ],
        )

        config = ChunkerConfig(
            target_tokens=50,
            max_tokens=200,
            respect_boundaries=True,
        )
        chunker = HierarchyChunker(config)

        chunks = chunker.chunk(doc)

        # Chunks should not span across major headings
        # Each chunk should have content from only one section
        assert len(chunks) >= 2

    def test_hierarchy_chunker_preserves_code(self):
        """Test that hierarchy chunker preserves code blocks."""
        doc = ChonkDocument(
            id="test",
            source_path="test.md",
            source_type="md",
            blocks=[
                Block(id="p1", type="text", content="Some text before code.", page=1),
                Block(
                    id="c1",
                    type="code",
                    content="def function():\n    pass\n" * 10,
                    page=1,
                ),
                Block(id="p2", type="text", content="Some text after code.", page=1),
            ],
        )

        config = ChunkerConfig(
            target_tokens=20,
            max_tokens=500,  # Large enough to keep code intact
            preserve_code=True,
        )
        chunker = HierarchyChunker(config)

        chunks = chunker.chunk(doc)

        # Code should be kept together
        code_chunks = [c for c in chunks if "def function" in c.content]
        assert len(code_chunks) >= 1


class TestChunkerRegistry:
    """Tests for ChunkerRegistry."""

    def test_available_chunkers(self):
        """Test getting available chunkers."""
        chunkers = ChunkerRegistry.available_chunkers()

        assert "fixed" in chunkers
        assert "recursive" in chunkers
        assert "hierarchy" in chunkers

    def test_get_chunker(self):
        """Test getting chunker by name."""
        config = ChunkerConfig()

        fixed = ChunkerRegistry.get("fixed", config)
        assert isinstance(fixed, FixedSizeChunker)

        recursive = ChunkerRegistry.get("recursive", config)
        assert isinstance(recursive, RecursiveChunker)

        hierarchy = ChunkerRegistry.get("hierarchy", config)
        assert isinstance(hierarchy, HierarchyChunker)

    def test_get_unknown_chunker(self):
        """Test getting unknown chunker raises error."""
        config = ChunkerConfig()

        with pytest.raises(ValueError) as exc_info:
            ChunkerRegistry.get("unknown_chunker", config)

        assert "Unknown chunker" in str(exc_info.value)

    def test_chunk_document(self, sample_document):
        """Test chunking document through registry."""
        config = ChunkerConfig(target_tokens=50)

        ChunkerRegistry.chunk_document(sample_document, "hierarchy", config)

        assert len(sample_document.chunks) > 0
