# CHONK Export Format Specification

**Schema Version:** 1.1
**Last Updated:** 2026-02-23

## Overview

CHONK exports document chunks in a standardized format designed to be vector-database-agnostic. The canonical record format (`ChonkRecord`) serves as the intermediate representation that all exporters and vector DB adapters consume.

## Schema Versioning

Every export includes a `schema_version` field following `MAJOR.MINOR` semantics:

- **MAJOR** increment: Breaking changes to field names, types, or removal of fields
- **MINOR** increment: Additive changes (new optional fields, new adapters)

### Version History

| Version | Description |
|---------|-------------|
| 1.0     | Initial export format with document/chunk structure |
| 1.1     | Added enrichment_fields, schema versioning, vector DB adapters |

## Canonical Record Format (ChonkRecord)

Each chunk exports as a JSON object with the following fields:

```json
{
  "schema_version": "1.1",
  "id": "chunk_abc123def4",
  "content": "Remove the air filter element per TM 9-2320-280-20...",
  "token_count": 128,
  "hierarchy_path": "Chapter 3 > Engine Maintenance > Air Filter",
  "quality_score": 0.92,
  "source": "TM-9-2320-280-20.pdf",
  "source_type": "pdf",
  "document_id": "doc_abc123def4",
  "page_start": 42,
  "page_end": 43,
  "enrichment_fields": {
    "tm_number": "TM 9-2320-280-20",
    "maintenance_level": "unit",
    "nsn": "2940-01-234-5678"
  },
  "user_metadata": {
    "tags": ["maintenance", "engine"],
    "notes": "Reviewed by SME"
  },
  "system_metadata": {
    "enrichment_applied": true,
    "enrichment_quality": 0.85
  }
}
```

### Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | string | yes | Export format version |
| `id` | string | yes | Unique chunk identifier |
| `content` | string | yes | Full text content of the chunk |
| `token_count` | integer | yes | Number of tokens in content |
| `hierarchy_path` | string | no | Structural path (e.g. "Chapter 1 > Safety") |
| `quality_score` | float | no | Overall quality score (0.0-1.0), default 1.0 |
| `source` | string | no | Source document file path |
| `source_type` | string | no | Document format (pdf, docx, txt, etc.) |
| `document_id` | string | no | Parent document identifier |
| `page_start` | integer | no | First page number (null if unknown) |
| `page_end` | integer | no | Last page number (null if unknown) |
| `enrichment_fields` | object | no | Extracted metadata from enrichment pipeline |
| `user_metadata` | object | no | User-defined tags, notes, custom fields |
| `system_metadata` | object | no | System-generated metadata |

### Enrichment Fields

When the metadata enrichment pipeline (T-007) has processed a chunk, the `enrichment_fields` object may contain:

| Field | Example | Description |
|-------|---------|-------------|
| `tm_number` | "TM 9-2320-280-20" | Technical manual number |
| `nsn` | "2940-01-234-5678" | National Stock Number |
| `maintenance_level` | "unit" | Maintenance level (unit, direct, general, depot) |
| `work_package` | "WP 0045 00" | Work package reference |
| `lin` | "T54321" | Line Item Number |
| `smr_code` | "PAOAF" | Source, Maintenance, Recoverability code |
| `section_number` | "3-2" | Section reference |
| `figure_ref` | "Figure 3-1" | Figure reference |
| `table_ref` | "Table 4-2" | Table reference |

## Export Formats

### JSON (Full Document)

Exports the complete document structure with processing metadata.

```bash
ExporterRegistry.export_document(document, path, "json")
```

### JSONL (Per-Chunk)

One JSON object per line, each representing a single chunk. Best for streaming ingestion into RAG pipelines.

```bash
ExporterRegistry.export_document(document, path, "jsonl")
```

### CSV (Tabular)

Flat tabular format for spreadsheet viewing. Loses nested metadata structure.

```bash
ExporterRegistry.export_document(document, path, "csv")
```

## Vector Database Adapters

The `VectorDBAdapter` class transforms `ChonkRecord` objects into payloads ready for each vector DB's upsert API. The user supplies their own embedding vectors.

### Usage Pattern

```python
from chonk.exporters.schema import chunk_to_record, VectorDBAdapter

# Convert chunk to canonical record
record = chunk_to_record(chunk, document)

# Transform for your target DB
payload = VectorDBAdapter.to_chromadb(record)
# payload = VectorDBAdapter.to_qdrant(record)
# payload = VectorDBAdapter.to_weaviate(record)
# payload = VectorDBAdapter.to_pinecone(record)
```

### ChromaDB

```python
payload = VectorDBAdapter.to_chromadb(record)
# Returns: {"id": "...", "document": "...", "metadata": {...}}

collection.add(
    ids=[payload["id"]],
    documents=[payload["document"]],
    metadatas=[payload["metadata"]],
    embeddings=[my_embedding],  # User-provided
)
```

**Metadata constraints:** All values are str, int, float, or bool. Lists are joined with `"; "`. Enrichment fields are prefixed with `enrichment_`.

### Qdrant

```python
payload = VectorDBAdapter.to_qdrant(record)
# Returns: {"id": "...", "payload": {...}}

from qdrant_client.models import PointStruct
point = PointStruct(
    id=payload["id"],
    vector=my_embedding,  # User-provided
    payload=payload["payload"],
)
client.upsert(collection_name="chunks", points=[point])
```

**Payload structure:** Supports nested objects. Enrichment fields stored as nested `enrichment` dict. User metadata preserved as nested dict.

### Weaviate

```python
payload = VectorDBAdapter.to_weaviate(record)
# Returns: {"class": "ChonkChunk", "properties": {...}}

client.data_object.create(
    data_object=payload["properties"],
    class_name=payload["class"],
    vector=my_embedding,  # User-provided (or let Weaviate auto-vectorize)
)
```

**Property naming:** Uses camelCase (e.g. `sourceType`, `hierarchyPath`, `qualityScore`). Enrichment fields are flattened with `enrichment_` prefix.

### Pinecone

```python
payload = VectorDBAdapter.to_pinecone(record)
# Returns: {"id": "...", "metadata": {...}}

index.upsert(vectors=[{
    "id": payload["id"],
    "values": my_embedding,  # User-provided
    "metadata": payload["metadata"],
}])
```

**Metadata constraints:** Values must be str, number, bool, or list[str]. Content is stored in metadata for retrieval. Tags preserved as list[str]. Enrichment fields prefixed with `enrichment_`.

## Conversion Helper

The `chunk_to_record()` function converts a `Chunk` + `ChonkDocument` pair into a `ChonkRecord`:

```python
from chonk.exporters.schema import chunk_to_record

for chunk in document.chunks:
    record = chunk_to_record(chunk, document)
    # record.enrichment_fields contains extracted metadata
    # record.user_metadata contains flattened user tags/notes
    # record.system_metadata excludes enrichment (stored separately)
```

This function:
- Extracts `enrichment_fields` from `system_metadata` into a separate field
- Flattens user metadata (tags, notes, custom fields with `custom_` prefix)
- Extracts page range from system metadata
- Passes quality score through from the chunk's QualityScore
