# Kiln Export Format Specification

Kiln produces two categories of export data: **Quarry chunks** (document processing output) and **Forge training data** (curriculum for model fine-tuning). This document specifies the exact format for each.

---

## Quarry Export: ChonkRecord

The canonical export format for processed document chunks is the `ChonkRecord`, defined in `quarry/chonk/exporters/schema.py`.

### Schema Version

Current version: **1.1**

| Version | Description |
|---|---|
| 1.0 | Initial format with document/chunk structure |
| 1.1 | Added enrichment_fields, schema versioning, vector DB adapters |

The `schema_version` field is included in every serialized record. Consumers should check this field and handle unknown versions gracefully.

### ChonkRecord Fields

```json
{
  "schema_version": "1.1",
  "id": "chunk_abc123",
  "content": "Full text content of the chunk...",
  "token_count": 256,
  "hierarchy_path": "Chapter 1 > Safety > General Precautions",
  "quality_score": 0.92,
  "source": "/path/to/document.pdf",
  "source_type": "pdf",
  "document_id": "doc_def456",
  "page_start": 14,
  "page_end": 15,
  "enrichment_fields": {
    "document_type": "technical_manual",
    "has_warnings": true,
    "section_level": 3
  },
  "user_metadata": {
    "tags": ["safety", "electrical"],
    "hierarchy_hint": "Chapter 1.2.1",
    "notes": "Reviewer note here",
    "custom_priority": "high"
  },
  "system_metadata": {
    "processing_timestamp": "2026-02-23T10:30:00",
    "classifier_confidence": 0.87
  }
}
```

### Field Reference

| Field | Type | Required | Description |
|---|---|---|---|
| schema_version | string | Yes | Semver format (currently "1.1") |
| id | string | Yes | Unique chunk identifier |
| content | string | Yes | Full text content |
| token_count | int | Yes | Number of tokens (tiktoken) |
| hierarchy_path | string | No | Structural path separated by " > " |
| quality_score | float | No | 0.0 to 1.0 (default 1.0) |
| source | string | No | Source document path |
| source_type | string | No | Document format ("pdf", "docx", etc.) |
| document_id | string | No | Parent document identifier |
| page_start | int or null | No | First page number |
| page_end | int or null | No | Last page number |
| enrichment_fields | object | No | Metadata from enrichment pipeline |
| user_metadata | object | No | User-defined tags and custom fields |
| system_metadata | object | No | System-generated processing metadata |

---

## JSONL Export (Quarry)

The JSONL exporter (`quarry/chonk/exporters/jsonl.py`) writes one JSON object per line, compatible with LangChain, LlamaIndex, and most RAG frameworks.

### JSONL Record Format

Each line is a self-contained JSON object:

```json
{"id": "chunk_abc123", "text": "Full text content...", "metadata": {"source": "/path/to/doc.pdf", "source_type": "pdf", "page": 14, "page_end": 15, "hierarchy_path": "Chapter 1 > Safety", "quality_score": 0.92, "token_count": 256, "tags": ["safety"]}}
```

Note: The JSONL format uses `text` as the content field (matching LangChain conventions), while ChonkRecord uses `content`. The `metadata` object flattens source information, location, hierarchy, quality, and user metadata into a single level.

---

## Vector DB Adapter Formats

The `VectorDBAdapter` class in `quarry/chonk/exporters/schema.py` transforms ChonkRecords into payloads for four major vector databases. All adapters exclude the embedding vector, which must be generated separately by the user's chosen embedding model.

### ChromaDB

```python
VectorDBAdapter.to_chromadb(record)
```

Returns:

```json
{
  "id": "chunk_abc123",
  "document": "Full text content...",
  "metadata": {
    "source": "/path/to/doc.pdf",
    "source_type": "pdf",
    "document_id": "doc_def456",
    "hierarchy_path": "Chapter 1 > Safety",
    "quality_score": 0.92,
    "token_count": 256,
    "page_start": 14,
    "page_end": 15,
    "enrichment_document_type": "technical_manual",
    "tags": "safety; electrical"
  }
}
```

**Notes:**
- ChromaDB metadata values must be str, int, float, or bool
- Lists are joined with semicolons
- Enrichment fields are prefixed with `enrichment_`
- Use with: `collection.add(ids=[payload["id"]], documents=[payload["document"]], metadatas=[payload["metadata"]], embeddings=[vector])`

### Qdrant

```python
VectorDBAdapter.to_qdrant(record)
```

Returns:

```json
{
  "id": "chunk_abc123",
  "payload": {
    "content": "Full text content...",
    "source": "/path/to/doc.pdf",
    "source_type": "pdf",
    "document_id": "doc_def456",
    "hierarchy_path": "Chapter 1 > Safety",
    "quality_score": 0.92,
    "token_count": 256,
    "page_start": 14,
    "page_end": 15,
    "enrichment": {"document_type": "technical_manual"},
    "user_metadata": {"tags": ["safety", "electrical"]}
  }
}
```

**Notes:**
- Qdrant supports nested payloads with rich types
- Enrichment and user metadata are kept as nested objects
- Add `vector` key before upserting

### Weaviate

```python
VectorDBAdapter.to_weaviate(record)
```

Returns:

```json
{
  "class": "ChonkChunk",
  "properties": {
    "content": "Full text content...",
    "chunkId": "chunk_abc123",
    "source": "/path/to/doc.pdf",
    "sourceType": "pdf",
    "documentId": "doc_def456",
    "hierarchyPath": "Chapter 1 > Safety",
    "qualityScore": 0.92,
    "tokenCount": 256,
    "pageStart": 14,
    "pageEnd": 15,
    "enrichment_document_type": "technical_manual"
  }
}
```

**Notes:**
- Weaviate uses camelCase naming convention
- The class name is `ChonkChunk`
- Enrichment fields are flattened with `enrichment_` prefix
- List values are converted to strings
- Add `vector` key for custom vectors, or let Weaviate auto-vectorize

### Pinecone

```python
VectorDBAdapter.to_pinecone(record)
```

Returns:

```json
{
  "id": "chunk_abc123",
  "metadata": {
    "content": "Full text content...",
    "source": "/path/to/doc.pdf",
    "source_type": "pdf",
    "document_id": "doc_def456",
    "hierarchy_path": "Chapter 1 > Safety",
    "quality_score": 0.92,
    "token_count": 256,
    "page_start": 14,
    "page_end": 15,
    "enrichment_document_type": "technical_manual",
    "tags": ["safety", "electrical"]
  }
}
```

**Notes:**
- Pinecone metadata must be str, number, bool, or list[str]
- Enrichment fields are prefixed with `enrichment_`
- List values are converted to list[str]
- Add `values` key with the embedding vector before upserting

---

## Forge Export: Alpaca JSONL

Forge exports training examples in Alpaca format for consumption by Foundry's LoRA training pipeline.

### Alpaca Record Format

```json
{
  "instruction": "What are the safety precautions for replacing a hydraulic filter?",
  "input": "",
  "output": "Before replacing a hydraulic filter, ensure the system is depressurized...",
  "metadata": {
    "example_id": "ex_abc123def456",
    "discipline_id": "disc_maint789012",
    "competency_id": "comp_safety345678",
    "contributor_id": "contrib_jane901234",
    "review_status": "approved",
    "created_at": "2026-02-20T14:30:00"
  }
}
```

### Field Reference

| Field | Type | Required | Description |
|---|---|---|---|
| instruction | string | Yes | The question or prompt (from `Example.question`) |
| input | string | Yes | Always empty string ("") in Kiln |
| output | string | Yes | The ideal answer (from `Example.ideal_answer`) |
| metadata | object | Yes | Provenance and audit information |

### Metadata Fields

| Field | Type | Description |
|---|---|---|
| example_id | string | Unique ID (prefixed with "ex_") |
| discipline_id | string | Parent discipline (prefixed with "disc_") |
| competency_id | string | Associated competency (prefixed with "comp_") |
| contributor_id | string | Creator (prefixed with "contrib_") |
| review_status | string | "pending", "approved", "rejected", or "needs_revision" |
| created_at | string | ISO 8601 timestamp |

### Export Methods

Training examples (excluding test set):
```python
storage.export_to_jsonl(discipline_id, "curriculum.jsonl")
```

Test set only:
```python
storage.export_test_set_jsonl(discipline_id, "test_set.jsonl")
```

All examples including test set:
```python
storage.export_to_jsonl(discipline_id, "all.jsonl", include_test_set=True)
```

### Important Notes

- The `input` field is always an empty string. Kiln uses instruction-only format because domain-specific context comes from RAG at inference time, not from training examples.
- Only examples with `is_test_set = False` are included in the default training export. Test-set examples are exported separately for use by Foundry's evaluation system.
- The `metadata` field is not used during training (LoRA trainers consume only instruction/input/output). It provides audit trail and provenance tracking.
- Review status is included in metadata but does not filter the export. Filtering by review status should be done before creating a curriculum version.

---

## Schema Versioning

Both Quarry and Forge exports include versioning for forward compatibility.

### Quarry ChonkRecord

The `schema_version` field (currently "1.1") is included in every `ChonkRecord.to_dict()` output. Version history is tracked in the `SCHEMA_HISTORY` constant in `quarry/chonk/exporters/schema.py`.

**Migration guidance:**
- MAJOR version increments indicate breaking changes. Consumers should reject unknown major versions.
- MINOR version increments are additive only. New fields may appear; old consumers should ignore unknown fields.

### Forge Alpaca JSONL

The Forge JSONL format follows the Alpaca standard and does not include a version field. The `metadata` object provides extensibility -- new metadata fields can be added without breaking existing consumers.

---

## Integration Examples

### Loading Quarry JSONL into LangChain

```python
from langchain.document_loaders import JSONLoader

loader = JSONLoader(
    file_path="export.jsonl",
    jq_schema=".",
    content_key="text",
    metadata_func=lambda record, metadata: record.get("metadata", {}),
)
documents = loader.load()
```

### Loading Forge JSONL for Foundry Training

```python
from foundry.src.training import CurriculumLoader

loader = CurriculumLoader()
records = loader.load("curriculum.jsonl")
train, val = loader.split_train_val(records, val_ratio=0.1)
stats = loader.get_statistics(records)
print(f"Total examples: {stats['total_records']}")
```

### Using Vector DB Adapters

```python
from chonk.exporters.schema import ChonkRecord, VectorDBAdapter

record = ChonkRecord.from_dict(data)

# ChromaDB
payload = VectorDBAdapter.to_chromadb(record)
collection.add(
    ids=[payload["id"]],
    documents=[payload["document"]],
    metadatas=[payload["metadata"]],
    embeddings=[embedding_vector],
)

# Qdrant
payload = VectorDBAdapter.to_qdrant(record)
client.upsert(
    collection_name="chunks",
    points=[{"id": payload["id"], "payload": payload["payload"], "vector": embedding_vector}],
)

# Pinecone
payload = VectorDBAdapter.to_pinecone(record)
index.upsert([(payload["id"], embedding_vector, payload["metadata"])])
```
