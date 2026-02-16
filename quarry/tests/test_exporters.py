"""
Tests for CHONK exporters.
"""

import json
from pathlib import Path

import pytest

from chonk.core.document import Chunk, ChonkDocument, ChonkProject
from chonk.exporters import ExporterRegistry
from chonk.exporters.base import BaseExporter


class TestExporterRegistry:
    """Tests for ExporterRegistry."""

    def test_available_exporters(self):
        """Test getting available exporters."""
        exporters = ExporterRegistry.available_exporters()

        assert "jsonl" in exporters
        assert "json" in exporters
        assert "csv" in exporters

    def test_get_exporter(self):
        """Test getting exporter by format."""
        jsonl_exporter = ExporterRegistry.get_exporter("jsonl")
        assert jsonl_exporter is not None

        json_exporter = ExporterRegistry.get_exporter("json")
        assert json_exporter is not None

        csv_exporter = ExporterRegistry.get_exporter("csv")
        assert csv_exporter is not None

    def test_get_unknown_exporter(self):
        """Test getting unknown exporter returns None."""
        exporter = ExporterRegistry.get_exporter("unknown_format")
        assert exporter is None


class TestJSONLExporter:
    """Tests for JSONL exporter."""

    @pytest.fixture
    def document_with_chunks(self, sample_document):
        """Create document with chunks for export testing."""
        sample_document.chunks = [
            Chunk(
                id="chunk_1",
                block_ids=["block_1", "block_2"],
                content="First chunk content.",
                token_count=5,
                hierarchy_path="Introduction",
            ),
            Chunk(
                id="chunk_2",
                block_ids=["block_3"],
                content="Second chunk content.",
                token_count=4,
                hierarchy_path="Introduction",
            ),
        ]
        return sample_document

    def test_jsonl_export_document(self, document_with_chunks, temp_dir):
        """Test exporting document to JSONL."""
        output_path = temp_dir / "export.jsonl"

        path = ExporterRegistry.export_document(
            document_with_chunks, output_path, "jsonl"
        )

        assert path.exists()

        # Read and verify content
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2  # Two chunks

        # Verify JSON structure
        for line in lines:
            data = json.loads(line)
            assert "content" in data
            assert "metadata" in data

    def test_jsonl_export_project(self, sample_project, temp_dir):
        """Test exporting project to JSONL."""
        # Add chunks to document
        doc = sample_project.documents[0]
        doc.chunks = [
            Chunk(id="c1", block_ids=["b1"], content="Chunk 1", token_count=3),
            Chunk(id="c2", block_ids=["b2"], content="Chunk 2", token_count=3),
        ]

        output_path = temp_dir / "project_export.jsonl"

        path = ExporterRegistry.export_project(sample_project, output_path, "jsonl")

        assert path.exists()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2


class TestJSONExporter:
    """Tests for JSON exporter."""

    @pytest.fixture
    def document_with_chunks(self, sample_document):
        """Create document with chunks for export testing."""
        sample_document.chunks = [
            Chunk(
                id="chunk_1",
                block_ids=["block_1"],
                content="Chunk content here.",
                token_count=4,
            ),
        ]
        return sample_document

    def test_json_export_document(self, document_with_chunks, temp_dir):
        """Test exporting document to JSON."""
        output_path = temp_dir / "export.json"

        path = ExporterRegistry.export_document(
            document_with_chunks, output_path, "json"
        )

        assert path.exists()

        # Verify JSON structure
        data = json.loads(path.read_text())
        assert "document_id" in data
        assert "chunks" in data
        assert len(data["chunks"]) == 1

    def test_json_export_includes_metadata(self, document_with_chunks, temp_dir):
        """Test that JSON export includes metadata."""
        output_path = temp_dir / "metadata.json"

        path = ExporterRegistry.export_document(
            document_with_chunks, output_path, "json"
        )

        data = json.loads(path.read_text())

        # Should include document metadata
        assert "metadata" in data or "document_name" in data


class TestCSVExporter:
    """Tests for CSV exporter."""

    @pytest.fixture
    def document_with_chunks(self, sample_document):
        """Create document with chunks for export testing."""
        sample_document.chunks = [
            Chunk(
                id="chunk_1",
                block_ids=["block_1"],
                content="First chunk with content.",
                token_count=5,
                hierarchy_path="Section 1",
            ),
            Chunk(
                id="chunk_2",
                block_ids=["block_2"],
                content="Second chunk here.",
                token_count=4,
                hierarchy_path="Section 2",
            ),
        ]
        sample_document.chunks[0].user_metadata.tags = ["important"]
        return sample_document

    def test_csv_export_document(self, document_with_chunks, temp_dir):
        """Test exporting document to CSV."""
        output_path = temp_dir / "export.csv"

        path = ExporterRegistry.export_document(
            document_with_chunks, output_path, "csv"
        )

        assert path.exists()

        # Read and verify
        lines = path.read_text().strip().split("\n")
        assert len(lines) >= 3  # Header + 2 chunks

        # Header should contain expected columns
        header = lines[0].lower()
        assert "content" in header or "chunk" in header

    def test_csv_export_handles_special_chars(self, sample_document, temp_dir):
        """Test CSV export handles special characters."""
        sample_document.chunks = [
            Chunk(
                id="special",
                block_ids=["b1"],
                content='Content with "quotes" and, commas, and\nnewlines.',
                token_count=10,
            ),
        ]

        output_path = temp_dir / "special.csv"
        path = ExporterRegistry.export_document(sample_document, output_path, "csv")

        # Should not raise and should be readable
        assert path.exists()
        content = path.read_text()
        assert "quotes" in content


class TestExporterIntegration:
    """Integration tests for exporters."""

    def test_export_roundtrip_jsonl(self, sample_project, temp_dir):
        """Test that JSONL export can be parsed back."""
        doc = sample_project.documents[0]
        doc.chunks = [
            Chunk(
                id="c1",
                block_ids=["b1"],
                content="Test content for roundtrip.",
                token_count=6,
                hierarchy_path="Test > Section",
            ),
        ]
        doc.chunks[0].user_metadata.tags = ["tag1", "tag2"]

        output_path = temp_dir / "roundtrip.jsonl"
        ExporterRegistry.export_document(doc, output_path, "jsonl")

        # Read back
        data = json.loads(output_path.read_text().strip())
        assert data["content"] == "Test content for roundtrip."
        assert "metadata" in data

    def test_export_empty_document(self, temp_dir):
        """Test exporting document with no chunks."""
        doc = ChonkDocument(
            id="empty",
            source_path=Path("empty.txt"),
            source_type="txt",
            blocks=[],
            chunks=[],
        )

        output_path = temp_dir / "empty.jsonl"
        path = ExporterRegistry.export_document(doc, output_path, "jsonl")

        # Should create empty or minimal file
        assert path.exists()
