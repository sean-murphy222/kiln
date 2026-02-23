"""Tests for export schema, versioning, and vector DB adapters."""

from __future__ import annotations

from pathlib import Path

import pytest
from chonk.core.document import ChonkDocument, Chunk, DocumentMetadata
from chonk.enrichment.extractor import ENRICHMENT_FIELDS_KEY
from chonk.exporters.schema import (
    SCHEMA_HISTORY,
    SCHEMA_VERSION,
    ChonkRecord,
    VectorDBAdapter,
    chunk_to_record,
)

# --- Fixtures ---


@pytest.fixture
def sample_record() -> ChonkRecord:
    """A fully-populated ChonkRecord for testing."""
    return ChonkRecord(
        id="chunk_001",
        content="Remove the air filter element per TM 9-2320-280-20.",
        token_count=12,
        hierarchy_path="Chapter 3 > Engine Maintenance > Air Filter",
        quality_score=0.92,
        source="TM-9-2320-280-20.pdf",
        source_type="pdf",
        document_id="doc_abc123",
        page_start=42,
        page_end=43,
        enrichment_fields={
            "tm_number": "TM 9-2320-280-20",
            "maintenance_level": "unit",
            "nsn": "2940-01-234-5678",
        },
        user_metadata={
            "tags": ["maintenance", "engine"],
            "notes": "Reviewed by SME",
        },
        system_metadata={
            "enrichment_applied": True,
            "enrichment_quality": 0.85,
        },
    )


@pytest.fixture
def minimal_record() -> ChonkRecord:
    """A minimal ChonkRecord with only required fields."""
    return ChonkRecord(
        id="chunk_min",
        content="Minimal content.",
        token_count=3,
    )


@pytest.fixture
def enriched_chunk() -> tuple[Chunk, ChonkDocument]:
    """A chunk with enrichment metadata and its parent document."""
    chunk = Chunk(
        id="chunk_enr",
        block_ids=["b1", "b2"],
        content="Replace filter per TM 9-2320-280-20. NSN 2940-01-234-5678.",
        token_count=15,
        hierarchy_path="Chapter 3 > Air Filter",
        system_metadata={
            ENRICHMENT_FIELDS_KEY: {
                "tm_number": "TM 9-2320-280-20",
                "nsn": "2940-01-234-5678",
            },
            "enrichment_applied": True,
            "enrichment_quality": 0.9,
            "start_page": 10,
            "end_page": 11,
        },
    )
    chunk.user_metadata.tags = ["engine", "filter"]
    chunk.user_metadata.notes = "Verified"

    document = ChonkDocument(
        id="doc_test",
        source_path=Path("manuals/TM-9-2320.pdf"),
        source_type="pdf",
        blocks=[],
        chunks=[chunk],
        metadata=DocumentMetadata(title="TM 9-2320", page_count=100),
    )
    return chunk, document


# --- Schema Version Tests ---


class TestSchemaVersion:
    """Tests for schema versioning."""

    def test_version_is_string(self) -> None:
        """Schema version is a non-empty string."""
        assert isinstance(SCHEMA_VERSION, str)
        assert len(SCHEMA_VERSION) > 0

    def test_version_format(self) -> None:
        """Schema version follows MAJOR.MINOR format."""
        parts = SCHEMA_VERSION.split(".")
        assert len(parts) == 2
        assert all(p.isdigit() for p in parts)

    def test_current_version_in_history(self) -> None:
        """Current version has a history entry."""
        assert SCHEMA_VERSION in SCHEMA_HISTORY

    def test_history_non_empty(self) -> None:
        """Version history has entries."""
        assert len(SCHEMA_HISTORY) >= 2


# --- ChonkRecord Tests ---


class TestChonkRecord:
    """Tests for ChonkRecord dataclass."""

    def test_construction(self, sample_record: ChonkRecord) -> None:
        """Test full construction."""
        assert sample_record.id == "chunk_001"
        assert sample_record.token_count == 12
        assert sample_record.quality_score == 0.92
        assert sample_record.page_start == 42

    def test_minimal_construction(self, minimal_record: ChonkRecord) -> None:
        """Test minimal construction with defaults."""
        assert minimal_record.hierarchy_path == ""
        assert minimal_record.quality_score == 1.0
        assert minimal_record.page_start is None
        assert minimal_record.enrichment_fields == {}

    def test_to_dict(self, sample_record: ChonkRecord) -> None:
        """Test serialization includes schema version."""
        d = sample_record.to_dict()
        assert d["schema_version"] == SCHEMA_VERSION
        assert d["id"] == "chunk_001"
        assert d["content"] == sample_record.content
        assert d["enrichment_fields"]["tm_number"] == "TM 9-2320-280-20"
        assert d["user_metadata"]["tags"] == ["maintenance", "engine"]

    def test_to_dict_has_all_fields(self, sample_record: ChonkRecord) -> None:
        """Test serialization includes all expected keys."""
        d = sample_record.to_dict()
        expected_keys = {
            "schema_version",
            "id",
            "content",
            "token_count",
            "hierarchy_path",
            "quality_score",
            "source",
            "source_type",
            "document_id",
            "page_start",
            "page_end",
            "enrichment_fields",
            "user_metadata",
            "system_metadata",
        }
        assert set(d.keys()) == expected_keys

    def test_from_dict(self, sample_record: ChonkRecord) -> None:
        """Test deserialization round-trip."""
        d = sample_record.to_dict()
        restored = ChonkRecord.from_dict(d)
        assert restored.id == sample_record.id
        assert restored.content == sample_record.content
        assert restored.enrichment_fields == sample_record.enrichment_fields
        assert restored.page_start == sample_record.page_start

    def test_from_dict_minimal(self) -> None:
        """Test deserialization with minimal data."""
        data = {"id": "c1", "content": "Hello", "token_count": 1}
        record = ChonkRecord.from_dict(data)
        assert record.id == "c1"
        assert record.enrichment_fields == {}
        assert record.quality_score == 1.0


# --- chunk_to_record Tests ---


class TestChunkToRecord:
    """Tests for chunk_to_record converter."""

    def test_basic_conversion(self, enriched_chunk: tuple[Chunk, ChonkDocument]) -> None:
        """Test converting an enriched chunk to a record."""
        chunk, doc = enriched_chunk
        record = chunk_to_record(chunk, doc)

        assert record.id == "chunk_enr"
        assert record.content == chunk.content
        assert record.token_count == 15
        assert record.source == str(Path("manuals/TM-9-2320.pdf"))
        assert record.source_type == "pdf"
        assert record.document_id == "doc_test"

    def test_enrichment_extracted(self, enriched_chunk: tuple[Chunk, ChonkDocument]) -> None:
        """Test enrichment fields are extracted from system_metadata."""
        chunk, doc = enriched_chunk
        record = chunk_to_record(chunk, doc)

        assert record.enrichment_fields["tm_number"] == "TM 9-2320-280-20"
        assert record.enrichment_fields["nsn"] == "2940-01-234-5678"

    def test_enrichment_not_in_system_metadata(
        self, enriched_chunk: tuple[Chunk, ChonkDocument]
    ) -> None:
        """Test enrichment_fields key is excluded from system_metadata."""
        chunk, doc = enriched_chunk
        record = chunk_to_record(chunk, doc)

        assert ENRICHMENT_FIELDS_KEY not in record.system_metadata
        assert "enrichment_applied" in record.system_metadata

    def test_page_range_extracted(self, enriched_chunk: tuple[Chunk, ChonkDocument]) -> None:
        """Test page range is extracted."""
        chunk, doc = enriched_chunk
        record = chunk_to_record(chunk, doc)

        assert record.page_start == 10
        assert record.page_end == 11

    def test_user_metadata_flattened(self, enriched_chunk: tuple[Chunk, ChonkDocument]) -> None:
        """Test user metadata is flattened into dict."""
        chunk, doc = enriched_chunk
        record = chunk_to_record(chunk, doc)

        assert record.user_metadata["tags"] == ["engine", "filter"]
        assert record.user_metadata["notes"] == "Verified"

    def test_no_enrichment(self) -> None:
        """Test converting a chunk without enrichment metadata."""
        chunk = Chunk(id="plain", block_ids=["b1"], content="Plain text.", token_count=3)
        doc = ChonkDocument(
            id="doc_plain",
            source_path=Path("file.txt"),
            source_type="txt",
            blocks=[],
            chunks=[chunk],
        )
        record = chunk_to_record(chunk, doc)

        assert record.enrichment_fields == {}
        assert record.page_start is None

    def test_custom_user_metadata(self) -> None:
        """Test custom user metadata fields are prefixed."""
        chunk = Chunk(id="custom", block_ids=["b1"], content="Custom.", token_count=1)
        chunk.user_metadata.custom = {"priority": "high", "domain": "aviation"}
        doc = ChonkDocument(
            id="doc_c",
            source_path=Path("f.pdf"),
            source_type="pdf",
            blocks=[],
            chunks=[chunk],
        )
        record = chunk_to_record(chunk, doc)

        assert record.user_metadata["custom_priority"] == "high"
        assert record.user_metadata["custom_domain"] == "aviation"


# --- VectorDBAdapter: ChromaDB Tests ---


class TestChromaDBAdapter:
    """Tests for ChromaDB adapter."""

    def test_structure(self, sample_record: ChonkRecord) -> None:
        """Test output has required ChromaDB keys."""
        result = VectorDBAdapter.to_chromadb(sample_record)
        assert "id" in result
        assert "document" in result
        assert "metadata" in result

    def test_content_as_document(self, sample_record: ChonkRecord) -> None:
        """Test content is in 'document' field."""
        result = VectorDBAdapter.to_chromadb(sample_record)
        assert result["document"] == sample_record.content

    def test_metadata_flat(self, sample_record: ChonkRecord) -> None:
        """Test metadata values are ChromaDB-compatible types."""
        result = VectorDBAdapter.to_chromadb(sample_record)
        for value in result["metadata"].values():
            assert isinstance(value, (str, int, float, bool))

    def test_enrichment_prefixed(self, sample_record: ChonkRecord) -> None:
        """Test enrichment fields have enrichment_ prefix."""
        result = VectorDBAdapter.to_chromadb(sample_record)
        meta = result["metadata"]
        assert meta["enrichment_tm_number"] == "TM 9-2320-280-20"
        assert meta["enrichment_maintenance_level"] == "unit"

    def test_tags_joined(self, sample_record: ChonkRecord) -> None:
        """Test tags are joined with semicolons."""
        result = VectorDBAdapter.to_chromadb(sample_record)
        assert result["metadata"]["tags"] == "maintenance; engine"

    def test_pages_included(self, sample_record: ChonkRecord) -> None:
        """Test page numbers are in metadata."""
        result = VectorDBAdapter.to_chromadb(sample_record)
        assert result["metadata"]["page_start"] == 42
        assert result["metadata"]["page_end"] == 43

    def test_no_pages_omitted(self, minimal_record: ChonkRecord) -> None:
        """Test page fields omitted when None."""
        result = VectorDBAdapter.to_chromadb(minimal_record)
        assert "page_start" not in result["metadata"]
        assert "page_end" not in result["metadata"]


# --- VectorDBAdapter: Qdrant Tests ---


class TestQdrantAdapter:
    """Tests for Qdrant adapter."""

    def test_structure(self, sample_record: ChonkRecord) -> None:
        """Test output has required Qdrant keys."""
        result = VectorDBAdapter.to_qdrant(sample_record)
        assert "id" in result
        assert "payload" in result

    def test_content_in_payload(self, sample_record: ChonkRecord) -> None:
        """Test content is in payload."""
        result = VectorDBAdapter.to_qdrant(sample_record)
        assert result["payload"]["content"] == sample_record.content

    def test_nested_enrichment(self, sample_record: ChonkRecord) -> None:
        """Test enrichment is nested dict (Qdrant supports rich types)."""
        result = VectorDBAdapter.to_qdrant(sample_record)
        enrichment = result["payload"]["enrichment"]
        assert isinstance(enrichment, dict)
        assert enrichment["tm_number"] == "TM 9-2320-280-20"

    def test_user_metadata_nested(self, sample_record: ChonkRecord) -> None:
        """Test user metadata is nested dict."""
        result = VectorDBAdapter.to_qdrant(sample_record)
        user_meta = result["payload"]["user_metadata"]
        assert user_meta["tags"] == ["maintenance", "engine"]

    def test_no_vector_key(self, sample_record: ChonkRecord) -> None:
        """Test no vector key (user adds their own)."""
        result = VectorDBAdapter.to_qdrant(sample_record)
        assert "vector" not in result


# --- VectorDBAdapter: Weaviate Tests ---


class TestWeaviateAdapter:
    """Tests for Weaviate adapter."""

    def test_structure(self, sample_record: ChonkRecord) -> None:
        """Test output has required Weaviate keys."""
        result = VectorDBAdapter.to_weaviate(sample_record)
        assert result["class"] == "ChonkChunk"
        assert "properties" in result

    def test_camel_case_properties(self, sample_record: ChonkRecord) -> None:
        """Test property names use camelCase."""
        props = VectorDBAdapter.to_weaviate(sample_record)["properties"]
        assert "sourceType" in props
        assert "documentId" in props
        assert "hierarchyPath" in props
        assert "qualityScore" in props
        assert "tokenCount" in props

    def test_chunk_id_in_properties(self, sample_record: ChonkRecord) -> None:
        """Test chunk ID is in properties as chunkId."""
        props = VectorDBAdapter.to_weaviate(sample_record)["properties"]
        assert props["chunkId"] == "chunk_001"

    def test_enrichment_flattened(self, sample_record: ChonkRecord) -> None:
        """Test enrichment fields are flattened into properties."""
        props = VectorDBAdapter.to_weaviate(sample_record)["properties"]
        assert props["enrichment_tm_number"] == "TM 9-2320-280-20"


# --- VectorDBAdapter: Pinecone Tests ---


class TestPineconeAdapter:
    """Tests for Pinecone adapter."""

    def test_structure(self, sample_record: ChonkRecord) -> None:
        """Test output has required Pinecone keys."""
        result = VectorDBAdapter.to_pinecone(sample_record)
        assert "id" in result
        assert "metadata" in result

    def test_content_in_metadata(self, sample_record: ChonkRecord) -> None:
        """Test content is stored in metadata (Pinecone pattern)."""
        result = VectorDBAdapter.to_pinecone(sample_record)
        assert result["metadata"]["content"] == sample_record.content

    def test_metadata_types(self, sample_record: ChonkRecord) -> None:
        """Test metadata values are Pinecone-compatible."""
        result = VectorDBAdapter.to_pinecone(sample_record)
        for value in result["metadata"].values():
            assert isinstance(value, (str, int, float, bool, list))

    def test_tags_as_list(self, sample_record: ChonkRecord) -> None:
        """Test tags are kept as list[str] (Pinecone supports it)."""
        result = VectorDBAdapter.to_pinecone(sample_record)
        assert result["metadata"]["tags"] == ["maintenance", "engine"]

    def test_enrichment_prefixed(self, sample_record: ChonkRecord) -> None:
        """Test enrichment fields have enrichment_ prefix."""
        result = VectorDBAdapter.to_pinecone(sample_record)
        meta = result["metadata"]
        assert meta["enrichment_nsn"] == "2940-01-234-5678"

    def test_no_values_key(self, sample_record: ChonkRecord) -> None:
        """Test no values key (user adds their own embedding)."""
        result = VectorDBAdapter.to_pinecone(sample_record)
        assert "values" not in result

    def test_pages_omitted_when_none(self, minimal_record: ChonkRecord) -> None:
        """Test page fields omitted when None."""
        result = VectorDBAdapter.to_pinecone(minimal_record)
        assert "page_start" not in result["metadata"]


# --- Cross-Adapter Consistency Tests ---


class TestAdapterConsistency:
    """Tests that all adapters handle the same record consistently."""

    def test_all_preserve_id(self, sample_record: ChonkRecord) -> None:
        """Test all adapters preserve the chunk ID."""
        chromadb = VectorDBAdapter.to_chromadb(sample_record)
        qdrant = VectorDBAdapter.to_qdrant(sample_record)
        pinecone = VectorDBAdapter.to_pinecone(sample_record)

        assert chromadb["id"] == "chunk_001"
        assert qdrant["id"] == "chunk_001"
        assert pinecone["id"] == "chunk_001"

    def test_all_include_content(self, sample_record: ChonkRecord) -> None:
        """Test all adapters include the content text."""
        text = sample_record.content

        chromadb = VectorDBAdapter.to_chromadb(sample_record)
        qdrant = VectorDBAdapter.to_qdrant(sample_record)
        weaviate = VectorDBAdapter.to_weaviate(sample_record)
        pinecone = VectorDBAdapter.to_pinecone(sample_record)

        assert chromadb["document"] == text
        assert qdrant["payload"]["content"] == text
        assert weaviate["properties"]["content"] == text
        assert pinecone["metadata"]["content"] == text

    def test_all_include_source(self, sample_record: ChonkRecord) -> None:
        """Test all adapters include source document info."""
        chromadb = VectorDBAdapter.to_chromadb(sample_record)
        qdrant = VectorDBAdapter.to_qdrant(sample_record)
        weaviate = VectorDBAdapter.to_weaviate(sample_record)
        pinecone = VectorDBAdapter.to_pinecone(sample_record)

        assert chromadb["metadata"]["source"] == sample_record.source
        assert qdrant["payload"]["source"] == sample_record.source
        assert weaviate["properties"]["source"] == sample_record.source
        assert pinecone["metadata"]["source"] == sample_record.source

    def test_minimal_record_all_adapters(self, minimal_record: ChonkRecord) -> None:
        """Test all adapters handle minimal records without error."""
        VectorDBAdapter.to_chromadb(minimal_record)
        VectorDBAdapter.to_qdrant(minimal_record)
        VectorDBAdapter.to_weaviate(minimal_record)
        VectorDBAdapter.to_pinecone(minimal_record)
