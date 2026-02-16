"""
Tests for CHONK core document models.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from chonk.core.document import (
    Block,
    BlockType,
    Chunk,
    ChunkMetadata,
    ChonkDocument,
    ChonkProject,
    DocumentMetadata,
    ProjectSettings,
    QualityScore,
    TestQuery,
    TestSuite,
)


class TestBlock:
    """Tests for Block dataclass."""

    def test_block_creation(self):
        """Test basic block creation."""
        block = Block(
            id="test_block",
            type=BlockType.TEXT,
            content="Hello, world!",
            page=1,
        )

        assert block.id == "test_block"
        assert block.type == BlockType.TEXT
        assert block.content == "Hello, world!"
        assert block.page == 1
        assert block.bbox is None
        assert block.parent_id is None
        assert block.children_ids == []

    def test_block_with_heading(self):
        """Test block with heading level."""
        block = Block(
            id="heading_block",
            type=BlockType.HEADING,
            content="Introduction",
            page=1,
            heading_level=1,
        )

        assert block.type == BlockType.HEADING
        assert block.heading_level == 1

    def test_block_with_bbox(self):
        """Test block with bounding box."""
        block = Block(
            id="positioned_block",
            type=BlockType.TEXT,
            content="Positioned text",
            page=2,
            bbox={"x1": 100, "y1": 200, "x2": 300, "y2": 250, "page": 2},
        )

        assert block.bbox is not None
        assert block.bbox["x1"] == 100
        assert block.bbox["page"] == 2

    def test_block_serialization(self):
        """Test block to_dict and from_dict."""
        block = Block(
            id="test_block",
            type=BlockType.TEXT,
            content="Test content",
            page=1,
            metadata={"custom_key": "value"},
        )

        data = block.to_dict()
        restored = Block.from_dict(data)

        assert restored.id == block.id
        assert restored.type == block.type
        assert restored.content == block.content
        assert restored.metadata == block.metadata

    def test_block_generate_id(self):
        """Test block ID generation."""
        id1 = Block.generate_id()
        id2 = Block.generate_id()

        assert id1 != id2
        assert id1.startswith("block_")
        assert len(id1) == 18  # "block_" + 12 hex chars


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_chunk_creation(self):
        """Test basic chunk creation."""
        chunk = Chunk(
            id="test_chunk",
            block_ids=["block_1", "block_2"],
            content="Combined content from blocks",
            token_count=10,
        )

        assert chunk.id == "test_chunk"
        assert chunk.block_ids == ["block_1", "block_2"]
        assert chunk.token_count == 10
        assert chunk.is_locked is False
        assert chunk.is_modified is False

    def test_chunk_with_metadata(self):
        """Test chunk with user metadata."""
        metadata = ChunkMetadata(
            tags=["important", "review"],
            hierarchy_hint="Chapter 1 > Section 2",
            notes="This chunk needs review",
        )
        chunk = Chunk(
            id="tagged_chunk",
            block_ids=["block_1"],
            content="Tagged content",
            token_count=5,
            user_metadata=metadata,
        )

        assert chunk.user_metadata.tags == ["important", "review"]
        assert chunk.user_metadata.hierarchy_hint == "Chapter 1 > Section 2"
        assert chunk.user_metadata.notes == "This chunk needs review"

    def test_chunk_quality_score(self):
        """Test chunk with quality score."""
        quality = QualityScore(
            token_range=0.9,
            sentence_complete=0.85,
            hierarchy_preserved=1.0,
            table_integrity=1.0,
            reference_complete=1.0,
        )
        chunk = Chunk(
            id="quality_chunk",
            block_ids=["block_1"],
            content="Quality content",
            token_count=100,
            quality=quality,
        )

        assert chunk.quality.overall == 0.945
        assert chunk.quality.token_range == 0.9

    def test_chunk_serialization(self):
        """Test chunk to_dict and from_dict."""
        chunk = Chunk(
            id="test_chunk",
            block_ids=["block_1", "block_2"],
            content="Test content",
            token_count=15,
            hierarchy_path="Introduction > Overview",
            is_locked=True,
        )

        data = chunk.to_dict()
        restored = Chunk.from_dict(data)

        assert restored.id == chunk.id
        assert restored.block_ids == chunk.block_ids
        assert restored.hierarchy_path == chunk.hierarchy_path
        assert restored.is_locked == chunk.is_locked


class TestChonkDocument:
    """Tests for ChonkDocument dataclass."""

    def test_document_creation(self, sample_blocks):
        """Test basic document creation."""
        doc = ChonkDocument(
            id="test_doc",
            source_path=Path("document.pdf"),
            source_type="pdf",
            blocks=sample_blocks,
            chunks=[],
        )

        assert doc.id == "test_doc"
        assert doc.source_type == "pdf"
        assert len(doc.blocks) == 8
        assert doc.chunks == []

    def test_document_metadata(self, sample_document):
        """Test document metadata."""
        assert sample_document.metadata.title == "Test Document"
        assert sample_document.metadata.page_count == 3
        assert sample_document.metadata.word_count == 150

    def test_document_get_block(self, sample_document):
        """Test getting block by ID."""
        block = sample_document.get_block("block_1")
        assert block is not None
        assert block.content == "Introduction"

        missing = sample_document.get_block("nonexistent")
        assert missing is None

    def test_document_serialization(self, sample_document):
        """Test document serialization."""
        data = sample_document.to_dict()

        assert data["id"] == sample_document.id
        assert "blocks" in data
        assert "chunks" in data
        assert "metadata" in data


class TestChonkProject:
    """Tests for ChonkProject dataclass."""

    def test_project_creation(self):
        """Test basic project creation."""
        project = ChonkProject(
            id="test_project",
            name="My Test Project",
        )

        assert project.id == "test_project"
        assert project.name == "My Test Project"
        assert project.documents == []
        assert project.test_suites == []
        assert project.project_path is None

    def test_project_settings(self):
        """Test project settings."""
        project = ChonkProject(
            id="test_project",
            name="Test Project",
        )

        assert project.settings.default_chunker == "hierarchy"
        assert project.settings.default_chunk_size == 400
        assert project.settings.embedding_model == "all-MiniLM-L6-v2"

    def test_project_get_document(self, sample_project):
        """Test getting document by ID."""
        doc = sample_project.get_document("doc_1")
        assert doc is not None
        assert doc.metadata.title == "Test Document"

        missing = sample_project.get_document("nonexistent")
        assert missing is None

    def test_project_save_load(self, sample_project, temp_dir):
        """Test project save and load."""
        save_path = temp_dir / "test_project.chonk"

        # Save
        saved_path = sample_project.save(save_path)
        assert saved_path.exists()

        # Load
        loaded = ChonkProject.load(saved_path)
        assert loaded.id == sample_project.id
        assert loaded.name == sample_project.name
        assert len(loaded.documents) == len(sample_project.documents)


class TestTestSuite:
    """Tests for TestSuite and TestQuery dataclasses."""

    def test_test_query_creation(self):
        """Test basic test query creation."""
        query = TestQuery(
            id="query_1",
            query="What is the introduction about?",
            expected_chunk_ids=["chunk_1", "chunk_2"],
            excluded_chunk_ids=["chunk_5"],
        )

        assert query.query == "What is the introduction about?"
        assert len(query.expected_chunk_ids) == 2
        assert len(query.excluded_chunk_ids) == 1

    def test_test_suite_creation(self):
        """Test basic test suite creation."""
        suite = TestSuite(
            id="suite_1",
            name="Introduction Tests",
        )

        assert suite.name == "Introduction Tests"
        assert suite.queries == []

    def test_test_suite_with_queries(self):
        """Test test suite with queries."""
        queries = [
            TestQuery(
                id="q1",
                query="Question 1",
                expected_chunk_ids=["c1"],
            ),
            TestQuery(
                id="q2",
                query="Question 2",
                expected_chunk_ids=["c2", "c3"],
            ),
        ]

        suite = TestSuite(
            id="suite_1",
            name="Test Suite",
            queries=queries,
        )

        assert len(suite.queries) == 2

    def test_test_suite_serialization(self):
        """Test test suite serialization."""
        query = TestQuery(
            id="q1",
            query="Test query",
            expected_chunk_ids=["c1"],
            notes="Test notes",
        )
        suite = TestSuite(
            id="suite_1",
            name="Test Suite",
            queries=[query],
        )

        data = suite.to_dict()
        restored = TestSuite.from_dict(data)

        assert restored.id == suite.id
        assert restored.name == suite.name
        assert len(restored.queries) == 1
        assert restored.queries[0].notes == "Test notes"
