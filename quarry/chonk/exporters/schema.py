"""Export schema definitions, versioning, and vector DB adapters.

Defines the canonical CHONK export record format and provides
adapter functions that transform records into payloads for major
vector databases (ChromaDB, Qdrant, Weaviate, Pinecone).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chonk.core.document import ChonkDocument, Chunk
from chonk.enrichment.extractor import ENRICHMENT_FIELDS_KEY

# Current export schema version (semver: MAJOR.MINOR)
# Increment MAJOR on breaking changes, MINOR on additive changes.
SCHEMA_VERSION = "1.1"

# Version history for documentation and migration.
SCHEMA_HISTORY: dict[str, str] = {
    "1.0": "Initial export format with document/chunk structure",
    "1.1": "Added enrichment_fields, schema versioning, vector DB adapters",
}


@dataclass
class ChonkRecord:
    """Canonical export record for a single chunk.

    All exporters and vector DB adapters consume this intermediate
    representation rather than raw Chunk objects.

    Attributes:
        id: Unique chunk identifier.
        content: Full text content.
        token_count: Number of tokens.
        hierarchy_path: Structural path (e.g. ``Chapter 1 > Safety``).
        quality_score: Overall quality score (0.0-1.0).
        source: Source document path.
        source_type: Document format (pdf, docx, etc.).
        document_id: Parent document identifier.
        page_start: First page number (None if unknown).
        page_end: Last page number (None if unknown).
        enrichment_fields: Extracted metadata from enrichment pipeline.
        user_metadata: User-defined tags, notes, custom fields.
        system_metadata: System-generated metadata (excluding enrichment).
    """

    id: str
    content: str
    token_count: int
    hierarchy_path: str = ""
    quality_score: float = 1.0
    source: str = ""
    source_type: str = ""
    document_id: str = ""
    page_start: int | None = None
    page_end: int | None = None
    enrichment_fields: dict[str, Any] = field(default_factory=dict)
    user_metadata: dict[str, Any] = field(default_factory=dict)
    system_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to standard CHONK record dict.

        Returns:
            Dictionary with schema_version and all fields.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "id": self.id,
            "content": self.content,
            "token_count": self.token_count,
            "hierarchy_path": self.hierarchy_path,
            "quality_score": self.quality_score,
            "source": self.source,
            "source_type": self.source_type,
            "document_id": self.document_id,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "enrichment_fields": self.enrichment_fields,
            "user_metadata": self.user_metadata,
            "system_metadata": self.system_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChonkRecord:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with record fields.

        Returns:
            ChonkRecord instance.
        """
        return cls(
            id=data["id"],
            content=data["content"],
            token_count=data.get("token_count", 0),
            hierarchy_path=data.get("hierarchy_path", ""),
            quality_score=data.get("quality_score", 1.0),
            source=data.get("source", ""),
            source_type=data.get("source_type", ""),
            document_id=data.get("document_id", ""),
            page_start=data.get("page_start"),
            page_end=data.get("page_end"),
            enrichment_fields=data.get("enrichment_fields", {}),
            user_metadata=data.get("user_metadata", {}),
            system_metadata=data.get("system_metadata", {}),
        )


def chunk_to_record(chunk: Chunk, document: ChonkDocument) -> ChonkRecord:
    """Convert a Chunk and its parent document into a ChonkRecord.

    Extracts enrichment fields from system_metadata and flattens
    user_metadata into a simple dict.

    Args:
        chunk: The chunk to convert.
        document: The parent document for source context.

    Returns:
        ChonkRecord ready for export or vector DB adaptation.
    """
    enrichment = chunk.system_metadata.get(ENRICHMENT_FIELDS_KEY, {})
    page_range = chunk.page_range

    user_meta: dict[str, Any] = {}
    if chunk.user_metadata.tags:
        user_meta["tags"] = chunk.user_metadata.tags
    if chunk.user_metadata.hierarchy_hint:
        user_meta["hierarchy_hint"] = chunk.user_metadata.hierarchy_hint
    if chunk.user_metadata.notes:
        user_meta["notes"] = chunk.user_metadata.notes
    for key, value in chunk.user_metadata.custom.items():
        user_meta[f"custom_{key}"] = value

    # System metadata minus enrichment fields (stored separately)
    sys_meta = {k: v for k, v in chunk.system_metadata.items() if k != ENRICHMENT_FIELDS_KEY}

    return ChonkRecord(
        id=chunk.id,
        content=chunk.content,
        token_count=chunk.token_count,
        hierarchy_path=chunk.hierarchy_path,
        quality_score=chunk.quality.overall,
        source=str(document.source_path),
        source_type=document.source_type,
        document_id=document.id,
        page_start=page_range[0] if page_range else None,
        page_end=page_range[1] if page_range else None,
        enrichment_fields=enrichment,
        user_metadata=user_meta,
        system_metadata=sys_meta,
    )


class VectorDBAdapter:
    """Transform ChonkRecords into vector-DB-specific payloads.

    Each method returns a dict ready for the target DB's upsert API,
    minus the embedding vector (generated by the user's chosen model).

    Example::

        record = chunk_to_record(chunk, document)
        payload = VectorDBAdapter.to_chromadb(record)
        collection.add(
            ids=[payload["id"]],
            documents=[payload["document"]],
            metadatas=[payload["metadata"]],
            embeddings=[my_embedding],
        )
    """

    @staticmethod
    def to_chromadb(record: ChonkRecord) -> dict[str, Any]:
        """Format for ChromaDB ``collection.add()``.

        ChromaDB metadata values must be str, int, float, or bool.
        Lists are joined with semicolons.

        Args:
            record: Canonical CHONK record.

        Returns:
            Dict with keys: id, document, metadata.
        """
        metadata: dict[str, str | int | float | bool] = {
            "source": record.source,
            "source_type": record.source_type,
            "document_id": record.document_id,
            "hierarchy_path": record.hierarchy_path,
            "quality_score": record.quality_score,
            "token_count": record.token_count,
        }

        if record.page_start is not None:
            metadata["page_start"] = record.page_start
        if record.page_end is not None:
            metadata["page_end"] = record.page_end

        for key, value in record.enrichment_fields.items():
            if isinstance(value, (int, float, bool)):
                metadata[f"enrichment_{key}"] = value
            else:
                metadata[f"enrichment_{key}"] = str(value)

        tags = record.user_metadata.get("tags", [])
        if tags:
            metadata["tags"] = "; ".join(tags) if isinstance(tags, list) else str(tags)

        return {
            "id": record.id,
            "document": record.content,
            "metadata": metadata,
        }

    @staticmethod
    def to_qdrant(record: ChonkRecord) -> dict[str, Any]:
        """Format for Qdrant ``PointStruct``.

        Qdrant supports nested payloads with rich types.
        Add ``vector`` key before upserting.

        Args:
            record: Canonical CHONK record.

        Returns:
            Dict with keys: id, payload.
        """
        payload: dict[str, Any] = {
            "content": record.content,
            "source": record.source,
            "source_type": record.source_type,
            "document_id": record.document_id,
            "hierarchy_path": record.hierarchy_path,
            "quality_score": record.quality_score,
            "token_count": record.token_count,
            "page_start": record.page_start,
            "page_end": record.page_end,
            "enrichment": record.enrichment_fields,
            "user_metadata": record.user_metadata,
        }

        return {
            "id": record.id,
            "payload": payload,
        }

    @staticmethod
    def to_weaviate(record: ChonkRecord) -> dict[str, Any]:
        """Format for Weaviate data object.

        Weaviate uses flat properties with camelCase naming convention.
        Add ``vector`` key for custom vectors, or let Weaviate
        auto-vectorize.

        Args:
            record: Canonical CHONK record.

        Returns:
            Dict with keys: class, properties.
        """
        properties: dict[str, Any] = {
            "content": record.content,
            "chunkId": record.id,
            "source": record.source,
            "sourceType": record.source_type,
            "documentId": record.document_id,
            "hierarchyPath": record.hierarchy_path,
            "qualityScore": record.quality_score,
            "tokenCount": record.token_count,
            "pageStart": record.page_start,
            "pageEnd": record.page_end,
        }

        for key, value in record.enrichment_fields.items():
            if isinstance(value, list):
                properties[f"enrichment_{key}"] = str(value)
            else:
                properties[f"enrichment_{key}"] = value

        return {
            "class": "ChonkChunk",
            "properties": properties,
        }

    @staticmethod
    def to_pinecone(record: ChonkRecord) -> dict[str, Any]:
        """Format for Pinecone ``upsert()``.

        Pinecone metadata must be str, number, bool, or list[str].
        Add ``values`` key with the embedding vector before upserting.

        Args:
            record: Canonical CHONK record.

        Returns:
            Dict with keys: id, metadata.
        """
        metadata: dict[str, Any] = {
            "content": record.content,
            "source": record.source,
            "source_type": record.source_type,
            "document_id": record.document_id,
            "hierarchy_path": record.hierarchy_path,
            "quality_score": record.quality_score,
            "token_count": record.token_count,
        }

        if record.page_start is not None:
            metadata["page_start"] = record.page_start
        if record.page_end is not None:
            metadata["page_end"] = record.page_end

        for key, value in record.enrichment_fields.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[f"enrichment_{key}"] = value
            elif isinstance(value, list):
                metadata[f"enrichment_{key}"] = [str(v) for v in value]
            else:
                metadata[f"enrichment_{key}"] = str(value)

        tags = record.user_metadata.get("tags", [])
        if tags:
            metadata["tags"] = tags if isinstance(tags, list) else [str(tags)]

        return {
            "id": record.id,
            "metadata": metadata,
        }
