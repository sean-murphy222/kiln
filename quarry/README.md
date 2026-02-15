# Quarry

Document processing pipeline for Kiln. Transforms complex domain documents into high-quality, metadata-enriched, retrieval-ready chunks.

## Status

~70% complete at project start. Core extraction and hierarchy construction functional. Remaining work:
- Tier 1 structural fingerprinting (ML classifier)
- Metadata enrichment pipeline
- QA pass and diagnostic output
- Retrieval pipeline integration

## Architecture

**Tier 1: Structural Fingerprinting**
- Statistical document analysis
- ML classifier (random forest/gradient boost, NOT LLM)
- Produces structural profile for document type

**Tier 2: Content Extraction**
- Docling for layout-aware PDF parsing
- Table extraction, multi-column layouts
- Figure/caption association

**Tier 3: Hierarchy & QA**
- Hierarchical structure construction
- Classification and filtering (remove boilerplate)
- Cleaning and normalization
- Metadata enrichment from formatting cues

**Metadata-Filtered Retrieval**
- Stage 1: Structural pre-filter (deterministic, zero-cost)
- Stage 2: Semantic search within filtered set
- Stage 3: Validation pass

## Export Format

Structured JSON with:
- `body`: Clean content for embedding
- `metadata`: Filterable attributes (section, equipment, procedure, etc.)
- Vector-database-agnostic (ChromaDB, Qdrant, Weaviate, Pinecone)

## MVP Scope

- PDF documents with embedded text layers
- No OCR (scanned/image-based PDFs out of scope)
- Military technical manuals as primary test domain
