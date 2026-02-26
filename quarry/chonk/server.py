"""FastAPI server for CHONK (Quarry document processing).

Provides REST API endpoints for the Electron UI to communicate
with the Python backend. Endpoints are registered on an ``APIRouter``
so that the unified ``kiln_server.py`` can mount them under a prefix.

The standalone ``app`` object includes the router directly and can
still be run independently::

    uvicorn chonk.server:app --reload --port 8420
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chonk.chunkers import ChunkerConfig, ChunkerRegistry
from chonk.comparison import StrategyComparer
from chonk.core.document import (
    Chunk,
    ChunkMetadata,
    ChonkDocument,
    ChonkProject,
    TestQuery,
    TestSuite,
)
from chonk.diagnostics import DiagnosticAnalyzer, FixOrchestrator
from chonk.diagnostics.question_generator import QuestionGenerator
from chonk.diagnostics.test_runner import QuestionTestRunner
from chonk.exporters import ExporterRegistry
from chonk.extraction import (
    ExtractionStrategy,
    ExtractionTier,
    get_available_tiers,
    get_extractor,
)
from chonk.hierarchy import HierarchyBuilder
from chonk.loaders import LoaderRegistry
from chonk.testing import RetrievalTester
from chonk.utils.quality import QualityAnalyzer

# ---------------------------------------------------------------------------
# Router: all Quarry/CHONK endpoints live here so that external apps
# (e.g. kiln_server.py) can mount them via ``include_router(router)``.
# ---------------------------------------------------------------------------
router = APIRouter()

# Initialize standalone FastAPI app (backward-compatible entry point)
app = FastAPI(
    title="CHONK API",
    description="Visual Document Chunking Studio for RAG Pipelines",
    version="0.1.0",
)

# Enable CORS for standalone mode
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory state (would be replaced with proper state management in production)
_state: dict[str, Any] = {
    "project": None,
    "tester": None,
    "settings": {
        "default_chunker": "hierarchy",
        "default_target_tokens": 400,
        "default_max_tokens": 600,
        "default_min_tokens": 100,
        "default_overlap_tokens": 50,
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_batch_size": 32,
        "theme": "dark",
        "show_quality_warnings": True,
        "auto_save": True,
        # Extraction settings
        "extraction_tier": "fast",  # fast, enhanced, ai, auto
        "extraction_auto_upgrade": False,
    },
}


# ============================================================================
# Pydantic Models for API
# ============================================================================


class ChunkConfigRequest(BaseModel):
    """Request model for chunking configuration."""

    chunker: str = "hierarchy"
    target_tokens: int = 400
    max_tokens: int = 600
    min_tokens: int = 100
    overlap_tokens: int = 50
    respect_boundaries: bool = True
    preserve_tables: bool = True
    preserve_code: bool = True


class ChunkUpdateRequest(BaseModel):
    """Request model for updating chunk metadata."""

    tags: list[str] | None = None
    hierarchy_hint: str | None = None
    notes: str | None = None
    custom: dict[str, Any] | None = None
    is_locked: bool | None = None


class MergeRequest(BaseModel):
    """Request for merging chunks."""

    chunk_ids: list[str]


class SplitRequest(BaseModel):
    """Request for splitting a chunk."""

    chunk_id: str
    split_position: int  # Character position to split at


class SearchRequest(BaseModel):
    """Request for searching chunks."""

    query: str
    top_k: int = 5
    document_ids: list[str] | None = None


class TestQueryRequest(BaseModel):
    """Request for creating a test query."""

    query: str
    expected_chunk_ids: list[str] = []
    excluded_chunk_ids: list[str] = []
    notes: str | None = None


class ExportRequest(BaseModel):
    """Request for exporting chunks."""

    format: str = "jsonl"
    output_path: str
    document_id: str | None = None  # None = export entire project


class ProjectCreateRequest(BaseModel):
    """Request for creating a new project."""

    name: str
    output_directory: str | None = None


# ============================================================================
# Project Endpoints
# ============================================================================


@router.post("/api/project/new")
async def create_project(request: ProjectCreateRequest) -> dict[str, Any]:
    """Create a new project."""
    project = ChonkProject(
        id=ChonkProject.generate_id(),
        name=request.name,
    )

    if request.output_directory:
        project.settings.output_directory = Path(request.output_directory)

    _state["project"] = project
    _state["tester"] = RetrievalTester()

    return {
        "id": project.id,
        "name": project.name,
        "created_at": project.created_at.isoformat(),
    }


@router.post("/api/project/open")
async def open_project(path: str) -> dict[str, Any]:
    """Open an existing project."""
    try:
        project = ChonkProject.load(Path(path))
        _state["project"] = project
        _state["tester"] = RetrievalTester()

        # Re-index chunks if there are documents
        if project.documents:
            tester: RetrievalTester = _state["tester"]
            tester.index_documents(project.documents)

        return {
            "id": project.id,
            "name": project.name,
            "document_count": len(project.documents),
            "created_at": project.created_at.isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/project/save")
async def save_project(path: str | None = None) -> dict[str, Any]:
    """Save the current project."""
    project = _get_project()
    try:
        save_path = project.save(Path(path) if path else None)
        return {
            "path": str(save_path),
            "saved_at": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/project")
async def get_project() -> dict[str, Any]:
    """Get current project info."""
    project = _get_project()
    return project.to_dict()


# ============================================================================
# Document Endpoints
# ============================================================================


@router.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    extraction_tier: str | None = None,
) -> dict[str, Any]:
    """Upload and process a document.

    Args:
        file: The document file to upload
        extraction_tier: Override extraction tier (fast, enhanced, ai, auto)
    """
    project = _get_project()
    settings = _state["settings"]

    # Save uploaded file to temp location
    suffix = Path(file.filename or "document").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Determine extraction tier to use
        tier_str = extraction_tier or settings.get("extraction_tier", "fast")
        auto_upgrade = settings.get("extraction_auto_upgrade", False)

        # Use extraction strategy for PDF files
        extraction_info = {}
        if suffix.lower() == ".pdf":
            result = _extract_with_strategy(tmp_path, tier_str, auto_upgrade)

            # Create document from extraction result
            from chonk.core.document import ChonkDocument, DocumentMetadata
            document = ChonkDocument(
                id=ChonkDocument.generate_id(),
                source_path=Path(file.filename or "document"),
                source_type=suffix.lower().lstrip('.'),
                blocks=result.blocks,
                chunks=[],  # Will be populated by chunker
                metadata=result.metadata,
                loader_used=f"extraction:{result.tier_used.value}",
            )
            extraction_info = {
                "tier_used": result.tier_used.value,
                "extraction_info": result.extraction_info,
                "warnings": result.warnings,
            }
        else:
            # Use standard loader for non-PDF files
            document = LoaderRegistry.load_document(tmp_path)
            document.source_path = Path(file.filename or "document")

        # Auto-chunk with default settings
        ChunkerRegistry.chunk_document(
            document,
            project.settings.default_chunker,
            ChunkerConfig(
                target_tokens=project.settings.default_chunk_size,
                overlap_tokens=project.settings.default_overlap,
            ),
        )

        # Analyze quality
        analyzer = QualityAnalyzer()
        analyzer.analyze_document(document)

        # Add to project
        project.documents.append(document)

        # Update search index
        tester: RetrievalTester = _state["tester"]
        tester.index_documents(project.documents)

        return {
            "document_id": document.id,
            "filename": file.filename,
            "page_count": document.metadata.page_count,
            "chunk_count": len(document.chunks),
            "word_count": document.metadata.word_count,
            **extraction_info,
        }

    finally:
        # Clean up temp file
        tmp_path.unlink(missing_ok=True)


def _extract_with_strategy(
    path: Path, tier_str: str, auto_upgrade: bool
) -> Any:
    """Extract document using the configured extraction strategy."""
    from chonk.extraction.strategy import ExtractionResult

    # Map tier string to ExtractionTier
    tier_map = {
        "fast": ExtractionTier.FAST,
        "enhanced": ExtractionTier.ENHANCED,
        "ai": ExtractionTier.AI,
    }

    if tier_str == "auto" or auto_upgrade:
        # Use strategy with auto-upgrade
        strategy = ExtractionStrategy(
            preferred_tier=tier_map.get(tier_str, ExtractionTier.FAST),
            auto_fallback=True,
            auto_upgrade=True,
        )
        return strategy.extract(path)
    else:
        # Use specific tier
        tier = tier_map.get(tier_str, ExtractionTier.FAST)
        extractor = get_extractor(tier)
        return extractor.extract(path)


@router.get("/api/documents/{document_id}")
async def get_document(document_id: str) -> dict[str, Any]:
    """Get a document by ID."""
    project = _get_project()
    document = project.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document.to_dict()


@router.delete("/api/documents/{document_id}")
async def delete_document(document_id: str) -> dict[str, Any]:
    """Remove a document from the project."""
    project = _get_project()
    document = project.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    project.documents = [d for d in project.documents if d.id != document_id]

    # Re-index
    tester: RetrievalTester = _state["tester"]
    tester.index_documents(project.documents)

    return {"deleted": document_id}


# ============================================================================
# Chunking Endpoints
# ============================================================================


@router.post("/api/documents/{document_id}/rechunk")
async def rechunk_document(
    document_id: str, config: ChunkConfigRequest
) -> dict[str, Any]:
    """Re-chunk a document with new settings."""
    project = _get_project()
    document = project.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    # Keep locked chunks
    locked_chunks = [c for c in document.chunks if c.is_locked]

    # Re-chunk
    chunker_config = ChunkerConfig(
        target_tokens=config.target_tokens,
        max_tokens=config.max_tokens,
        min_tokens=config.min_tokens,
        overlap_tokens=config.overlap_tokens,
        respect_boundaries=config.respect_boundaries,
        preserve_tables=config.preserve_tables,
        preserve_code=config.preserve_code,
    )

    ChunkerRegistry.chunk_document(document, config.chunker, chunker_config)

    # Restore locked chunks (by replacing new chunks that overlap with locked ones)
    # For MVP, we just re-add locked chunks at the end
    # TODO: Smarter merging that preserves locked chunk positions
    document.chunks.extend(locked_chunks)

    # Analyze quality
    analyzer = QualityAnalyzer()
    quality_report = analyzer.analyze_document(document)

    # Re-index
    tester: RetrievalTester = _state["tester"]
    tester.index_documents(project.documents)

    return {
        "chunk_count": len(document.chunks),
        "quality": quality_report,
    }


@router.post("/api/chunks/merge")
async def merge_chunks(request: MergeRequest) -> dict[str, Any]:
    """Merge multiple chunks into one."""
    project = _get_project()

    # Find chunks and their document
    chunks_to_merge: list[Chunk] = []
    source_doc: ChonkDocument | None = None

    for doc in project.documents:
        for chunk in doc.chunks:
            if chunk.id in request.chunk_ids:
                if source_doc is not None and source_doc.id != doc.id:
                    raise HTTPException(
                        status_code=400,
                        detail="Cannot merge chunks from different documents",
                    )
                source_doc = doc
                chunks_to_merge.append(chunk)

    if len(chunks_to_merge) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 chunks to merge")

    if source_doc is None:
        raise HTTPException(status_code=404, detail="Chunks not found")

    # Sort by position (using block_ids as proxy for order)
    chunks_to_merge.sort(key=lambda c: c.block_ids[0] if c.block_ids else "")

    # Create merged chunk
    merged_content = "\n\n".join(c.content for c in chunks_to_merge)
    merged_block_ids = []
    for c in chunks_to_merge:
        merged_block_ids.extend(c.block_ids)

    from chonk.utils.tokens import count_tokens

    merged_chunk = Chunk(
        id=Chunk.generate_id(),
        block_ids=merged_block_ids,
        content=merged_content,
        token_count=count_tokens(merged_content),
        hierarchy_path=chunks_to_merge[0].hierarchy_path,
        is_modified=True,
    )

    # Update quality score
    analyzer = QualityAnalyzer()
    merged_chunk.quality = analyzer.analyze_chunk(merged_chunk, source_doc)

    # Replace chunks in document
    source_doc.chunks = [
        c for c in source_doc.chunks if c.id not in request.chunk_ids
    ]
    # Insert at position of first merged chunk
    source_doc.chunks.append(merged_chunk)

    # Re-index
    tester: RetrievalTester = _state["tester"]
    tester.index_documents(project.documents)

    return merged_chunk.to_dict()


@router.post("/api/chunks/split")
async def split_chunk(request: SplitRequest) -> dict[str, Any]:
    """Split a chunk at the specified position."""
    project = _get_project()

    # Find chunk and document
    source_doc: ChonkDocument | None = None
    source_chunk: Chunk | None = None

    for doc in project.documents:
        for chunk in doc.chunks:
            if chunk.id == request.chunk_id:
                source_doc = doc
                source_chunk = chunk
                break

    if source_chunk is None or source_doc is None:
        raise HTTPException(status_code=404, detail="Chunk not found")

    content = source_chunk.content
    if request.split_position <= 0 or request.split_position >= len(content):
        raise HTTPException(status_code=400, detail="Invalid split position")

    # Split content
    content_a = content[: request.split_position].strip()
    content_b = content[request.split_position :].strip()

    if not content_a or not content_b:
        raise HTTPException(
            status_code=400, detail="Split would create empty chunk"
        )

    from chonk.utils.tokens import count_tokens

    # Create two new chunks
    chunk_a = Chunk(
        id=Chunk.generate_id(),
        block_ids=source_chunk.block_ids,  # Both keep reference to original blocks
        content=content_a,
        token_count=count_tokens(content_a),
        hierarchy_path=source_chunk.hierarchy_path,
        is_modified=True,
        system_metadata={
            **source_chunk.system_metadata,
            "split_from": source_chunk.id,
        },
    )

    chunk_b = Chunk(
        id=Chunk.generate_id(),
        block_ids=source_chunk.block_ids,
        content=content_b,
        token_count=count_tokens(content_b),
        hierarchy_path=source_chunk.hierarchy_path,
        is_modified=True,
        system_metadata={
            **source_chunk.system_metadata,
            "split_from": source_chunk.id,
        },
    )

    # Update quality scores
    analyzer = QualityAnalyzer()
    chunk_a.quality = analyzer.analyze_chunk(chunk_a, source_doc)
    chunk_b.quality = analyzer.analyze_chunk(chunk_b, source_doc)

    # Replace original chunk
    idx = next(
        i for i, c in enumerate(source_doc.chunks) if c.id == source_chunk.id
    )
    source_doc.chunks = (
        source_doc.chunks[:idx] + [chunk_a, chunk_b] + source_doc.chunks[idx + 1 :]
    )

    # Re-index
    tester: RetrievalTester = _state["tester"]
    tester.index_documents(project.documents)

    return {
        "chunk_a": chunk_a.to_dict(),
        "chunk_b": chunk_b.to_dict(),
    }


@router.put("/api/chunks/{chunk_id}")
async def update_chunk(chunk_id: str, request: ChunkUpdateRequest) -> dict[str, Any]:
    """Update chunk metadata."""
    project = _get_project()

    # Find chunk
    for doc in project.documents:
        for chunk in doc.chunks:
            if chunk.id == chunk_id:
                # Update metadata
                if request.tags is not None:
                    chunk.user_metadata.tags = request.tags
                if request.hierarchy_hint is not None:
                    chunk.user_metadata.hierarchy_hint = request.hierarchy_hint
                if request.notes is not None:
                    chunk.user_metadata.notes = request.notes
                if request.custom is not None:
                    chunk.user_metadata.custom = request.custom
                if request.is_locked is not None:
                    chunk.is_locked = request.is_locked

                chunk.is_modified = True
                chunk.modified_at = datetime.now()

                return chunk.to_dict()

    raise HTTPException(status_code=404, detail="Chunk not found")


# ============================================================================
# Search & Testing Endpoints
# ============================================================================


@router.post("/api/test/search")
async def search_chunks(request: SearchRequest) -> dict[str, Any]:
    """Search chunks with a query."""
    tester: RetrievalTester = _state.get("tester")
    if tester is None or not tester.is_indexed:
        raise HTTPException(status_code=400, detail="No chunks indexed")

    results = tester.search(request.query, request.top_k, request.document_ids)

    return {
        "query": request.query,
        "results": [r.to_dict() for r in results],
    }


@router.get("/api/test/status")
async def get_test_status() -> dict[str, Any]:
    """Get indexing status."""
    tester: RetrievalTester = _state.get("tester")
    if tester is None:
        return {"indexed": False, "chunk_count": 0}

    return {
        "indexed": tester.is_indexed,
        "chunk_count": tester.chunk_count,
    }


@router.post("/api/test/reindex")
async def reindex_chunks() -> dict[str, Any]:
    """Force re-indexing of all chunks."""
    project = _get_project()
    tester: RetrievalTester = _state["tester"]

    count = tester.index_documents(project.documents)

    return {
        "indexed": True,
        "chunk_count": count,
    }


@router.post("/api/test-suites")
async def create_test_suite(name: str) -> dict[str, Any]:
    """Create a new test suite."""
    project = _get_project()

    suite = TestSuite(
        id=TestSuite.generate_id(),
        name=name,
    )
    project.test_suites.append(suite)

    return suite.to_dict()


@router.post("/api/test-suites/{suite_id}/queries")
async def add_test_query(
    suite_id: str, request: TestQueryRequest
) -> dict[str, Any]:
    """Add a query to a test suite."""
    project = _get_project()

    suite = next((s for s in project.test_suites if s.id == suite_id), None)
    if suite is None:
        raise HTTPException(status_code=404, detail="Test suite not found")

    query = TestQuery(
        id=TestQuery.generate_id(),
        query=request.query,
        expected_chunk_ids=request.expected_chunk_ids,
        excluded_chunk_ids=request.excluded_chunk_ids,
        notes=request.notes,
    )
    suite.queries.append(query)
    suite.modified_at = datetime.now()

    return query.to_dict()


@router.post("/api/test-suites/{suite_id}/run")
async def run_test_suite(
    suite_id: str,
    top_k: int = 5,
    document_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Run a test suite and get results."""
    project = _get_project()
    tester: RetrievalTester = _state.get("tester")

    if tester is None or not tester.is_indexed:
        raise HTTPException(status_code=400, detail="No chunks indexed")

    suite = next((s for s in project.test_suites if s.id == suite_id), None)
    if suite is None:
        raise HTTPException(status_code=404, detail="Test suite not found")

    report = tester.run_test_suite(suite, top_k, document_ids)

    return report.to_dict()


@router.get("/api/test-suites/{suite_id}/coverage")
async def get_coverage(suite_id: str, top_k: int = 5) -> dict[str, Any]:
    """Get coverage analysis for a test suite."""
    project = _get_project()
    tester: RetrievalTester = _state.get("tester")

    if tester is None or not tester.is_indexed:
        raise HTTPException(status_code=400, detail="No chunks indexed")

    suite = next((s for s in project.test_suites if s.id == suite_id), None)
    if suite is None:
        raise HTTPException(status_code=404, detail="Test suite not found")

    return tester.get_chunk_coverage(suite, top_k)


# ============================================================================
# Export Endpoints
# ============================================================================


@router.post("/api/export")
async def export_chunks(request: ExportRequest) -> dict[str, Any]:
    """Export chunks to file."""
    project = _get_project()
    output_path = Path(request.output_path)

    try:
        if request.document_id:
            # Export single document
            document = project.get_document(request.document_id)
            if document is None:
                raise HTTPException(status_code=404, detail="Document not found")

            path = ExporterRegistry.export_document(
                document, output_path, request.format
            )
        else:
            # Export entire project
            path = ExporterRegistry.export_project(
                project, output_path, request.format
            )

        return {
            "path": str(path),
            "format": request.format,
            "exported_at": datetime.now().isoformat(),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/export/formats")
async def get_export_formats() -> dict[str, Any]:
    """Get available export formats."""
    return {
        "formats": ExporterRegistry.available_exporters(),
    }


# ============================================================================
# Quality Endpoints
# ============================================================================


@router.get("/api/documents/{document_id}/quality")
async def get_quality_report(document_id: str) -> dict[str, Any]:
    """Get quality analysis for a document."""
    project = _get_project()
    document = project.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    analyzer = QualityAnalyzer()
    return analyzer.analyze_document(document)


@router.get("/api/chunks/{chunk_id}/suggestions")
async def get_chunk_suggestions(chunk_id: str) -> dict[str, Any]:
    """Get improvement suggestions for a chunk."""
    project = _get_project()

    for doc in project.documents:
        for chunk in doc.chunks:
            if chunk.id == chunk_id:
                analyzer = QualityAnalyzer()
                suggestions = analyzer.get_improvement_suggestions(chunk, doc)
                return {
                    "chunk_id": chunk_id,
                    "quality_score": chunk.quality.overall,
                    "suggestions": suggestions,
                }

    raise HTTPException(status_code=404, detail="Chunk not found")


# ============================================================================
# Utility Endpoints
# ============================================================================


@router.get("/api/loaders")
async def get_available_loaders() -> dict[str, Any]:
    """Get available document loaders and supported formats."""
    return {
        "extensions": LoaderRegistry.supported_extensions(),
    }


@router.get("/api/chunkers")
async def get_available_chunkers() -> dict[str, Any]:
    """Get available chunking strategies."""
    return {
        "chunkers": ChunkerRegistry.available_chunkers(),
    }


@router.get("/api/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "0.1.0",
    }


@router.get("/api/extractors")
async def get_available_extractors() -> dict[str, Any]:
    """Get available extraction tiers and their status."""
    available = get_available_tiers()

    extractors = [
        {
            "id": "fast",
            "name": "Fast (PyMuPDF)",
            "description": "No GPU required, instant processing. Uses PyMuPDF + pdfplumber.",
            "available": ExtractionTier.FAST in available,
            "tier": 1,
            "install_hint": None,  # Always available
        },
        {
            "id": "enhanced",
            "name": "Enhanced (Docling)",
            "description": "Better tables, formulas, and reading order. Uses IBM Docling.",
            "available": ExtractionTier.ENHANCED in available,
            "tier": 2,
            "install_hint": "pip install chonk[enhanced]",
        },
        {
            "id": "ai",
            "name": "AI (LayoutParser)",
            "description": "Deep learning layout detection. Best for complex/scanned documents.",
            "available": ExtractionTier.AI in available,
            "tier": 3,
            "install_hint": "pip install chonk[ai]",
        },
    ]

    return {
        "extractors": extractors,
        "available_count": len(available),
    }


# ============================================================================
# Settings Endpoints
# ============================================================================


@router.get("/api/settings")
async def get_settings() -> dict[str, Any]:
    """Get application settings."""
    return _state["settings"]


@router.post("/api/settings")
async def save_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Save application settings."""
    # Update settings, preserving any that weren't sent
    _state["settings"].update(settings)
    return {"saved": True}


# ============================================================================
# Hierarchy Endpoints
# ============================================================================


class HierarchyBuildRequest(BaseModel):
    document_id: str


@router.post("/api/hierarchy/build")
async def build_hierarchy(request: HierarchyBuildRequest) -> dict[str, Any]:
    """Build hierarchy tree from document blocks."""
    project = _get_project()

    # Find document
    document = None
    for doc in project.documents:
        if doc.id == request.document_id:
            document = doc
            break

    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    # Build hierarchy
    tree = HierarchyBuilder.build_from_blocks(document.blocks, document.id)

    # Return tree as dict
    return tree.to_dict()


@router.get("/api/hierarchy/{document_id}")
async def get_hierarchy(document_id: str) -> dict[str, Any]:
    """Get hierarchy tree for a document."""
    project = _get_project()

    # Find document
    document = None
    for doc in project.documents:
        if doc.id == document_id:
            document = doc
            break

    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    # Build hierarchy (in the future, this could be cached)
    tree = HierarchyBuilder.build_from_blocks(document.blocks, document.id)

    return tree.to_dict()


@router.get("/api/hierarchy/{document_id}/stats")
async def get_hierarchy_stats(document_id: str) -> dict[str, Any]:
    """Get hierarchy statistics for a document."""
    project = _get_project()

    # Find document
    document = None
    for doc in project.documents:
        if doc.id == document_id:
            document = doc
            break

    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    # Build hierarchy
    tree = HierarchyBuilder.build_from_blocks(document.blocks, document.id)
    stats = tree.get_statistics()

    return {
        "total_nodes": stats["total_nodes"],
        "total_headings": len(tree._get_level_distribution()),
        "max_depth": stats["max_depth"],
        "quality_score": 1.0,  # Placeholder - would need HierarchyAnalyzer for real score
    }


# ============================================================================
# Comparison Endpoints
# ============================================================================


class CompareStrategiesRequest(BaseModel):
    document_id: str
    strategies: list[dict[str, Any]]


@router.post("/api/chunk/compare")
async def compare_strategies(request: CompareStrategiesRequest) -> dict[str, Any]:
    """Compare different chunking strategies."""
    project = _get_project()

    # Find document
    document = None
    for doc in project.documents:
        if doc.id == request.document_id:
            document = doc
            break

    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    # Prepare chunkers for comparison
    chunkers = []
    for strat in request.strategies:
        name = strat["name"]
        config = strat.get("config", {})

        # Get chunker
        chunker_config = ChunkerConfig(
            target_tokens=config.get("target_tokens", 400),
            max_tokens=config.get("max_tokens", 600),
            min_tokens=config.get("min_tokens", 100),
            overlap_tokens=config.get("overlap_tokens", 50),
            respect_boundaries=config.get("preserve_tables", True),
            preserve_tables=config.get("preserve_tables", True),
            preserve_code=config.get("preserve_code", True),
        )

        chunker = ChunkerRegistry.get_chunker(name, chunker_config)
        chunkers.append((name, chunker))

    # Compare strategies
    comparison = StrategyComparer.compare(document.blocks, chunkers)

    return comparison.to_dict()


class PreviewChunksRequest(BaseModel):
    document_id: str
    chunker: str
    target_tokens: int = 400
    max_tokens: int = 600
    overlap_tokens: int = 50
    preserve_tables: bool = True
    preserve_code: bool = True
    group_under_headings: bool = True
    heading_weight: float = 1.5


@router.post("/api/chunk/preview")
async def preview_chunks(request: PreviewChunksRequest) -> dict[str, Any]:
    """Preview chunks without saving them to the document."""
    project = _get_project()

    # Find document
    document = None
    for doc in project.documents:
        if doc.id == request.document_id:
            document = doc
            break

    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    # Create chunker
    config = ChunkerConfig(
        target_tokens=request.target_tokens,
        max_tokens=request.max_tokens,
        overlap_tokens=request.overlap_tokens,
        preserve_tables=request.preserve_tables,
        preserve_code=request.preserve_code,
    )

    chunker = ChunkerRegistry.get_chunker(request.chunker, config)

    # Generate chunks
    chunks = chunker.chunk(document.blocks)

    # Analyze quality - create temporary document for analysis
    temp_doc = ChonkDocument(
        id=document.id,
        source_path=document.source_path,
        source_type=document.source_type,
        blocks=document.blocks,
        chunks=chunks,
        metadata=document.metadata,
        loader_used=document.loader_used,
    )

    analyzer = QualityAnalyzer()
    quality = analyzer.analyze_document(temp_doc)

    return {
        "chunks": [chunk.to_dict() for chunk in chunks],
        "quality": quality,
    }


# ============================================================================
# Query Testing Endpoints
# ============================================================================


class StrategyTestRequest(BaseModel):
    query: str
    strategies: list[str]
    document_id: str | None = None


@router.post("/api/test/query")
async def test_query(request: StrategyTestRequest) -> dict[str, Any]:
    """Test a query against different chunking strategies."""
    project = _get_project()

    # Get document if specified
    document = None
    if request.document_id:
        for doc in project.documents:
            if doc.id == request.document_id:
                document = doc
                break
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")

    # If no document specified, use all documents
    documents_to_search = [document] if document else project.documents

    if not documents_to_search:
        raise HTTPException(status_code=400, detail="No documents available")

    # Prepare results for each strategy
    strategy_results = []

    for strategy_name in request.strategies:
        # For each strategy, we need to chunk the documents with that strategy
        # and then search
        all_chunks = []

        for doc in documents_to_search:
            # Create chunker for this strategy
            config = ChunkerConfig(
                target_tokens=400,
                max_tokens=600,
                overlap_tokens=50,
                preserve_tables=True,
                preserve_code=True,
            )

            chunker = ChunkerRegistry.get_chunker(strategy_name, config)
            chunks = chunker.chunk(doc.blocks)
            all_chunks.extend(chunks)

        if not all_chunks:
            continue

        # Search using retrieval tester
        tester = _get_retrieval_tester()
        tester.index_chunks(all_chunks)
        results = tester.search(request.query, top_k=3)

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "chunk_id": result.chunk_id,
                "score": result.score,
                "content_preview": result.content_preview[:200],
                "hierarchy_path": result.hierarchy_path,
                "token_count": result.token_count,
            })

        strategy_results.append({
            "strategy_name": strategy_name,
            "query": request.query,
            "results": formatted_results,
            "top_score": results[0].score if results else 0.0,
            "retrieved_count": len(results),
        })

    return {
        "query": request.query,
        "strategies": strategy_results,
    }


class CompareStrategiesQueryRequest(BaseModel):
    queries: list[str]
    strategies: list[str]
    document_id: str | None = None


@router.post("/api/test/compare-strategies")
async def compare_strategies_query(request: CompareStrategiesQueryRequest) -> dict[str, Any]:
    """Compare strategies across multiple queries."""
    results = []

    for query in request.queries:
        query_request = TestQueryRequest(
            query=query,
            strategies=request.strategies,
            document_id=request.document_id,
        )
        result = await test_query(query_request)
        results.append(result)

    # Calculate average scores per strategy
    avg_scores = {}
    for strategy_name in request.strategies:
        scores = []
        for result in results:
            for strat_result in result["strategies"]:
                if strat_result["strategy_name"] == strategy_name:
                    scores.append(strat_result["top_score"])
        avg_scores[strategy_name] = sum(scores) / len(scores) if scores else 0.0

    # Find best strategy
    best_strategy = max(avg_scores.items(), key=lambda x: x[1])[0] if avg_scores else None

    return {
        "queries": results,
        "summary": {
            "best_strategy": best_strategy,
            "avg_scores": avg_scores,
        },
    }


# ============================================================================
# Diagnostic Endpoints (MVP Phase 1-2)
# ============================================================================


class DiagnosticRequest(BaseModel):
    """Request for running diagnostics on a document."""

    document_id: str
    include_questions: bool = False  # Whether to generate and test questions
    top_k: int = 5  # For question testing


@router.post("/api/diagnostics/analyze")
async def analyze_chunks(request: DiagnosticRequest) -> dict[str, Any]:
    """
    Run diagnostic analysis on document chunks.

    Returns:
    - Static analysis problems (heuristics)
    - Question-based test results (if include_questions=True)
    - Statistics and recommendations
    """
    project = _get_project()

    # Find document
    document = project.get_document(request.document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    # Run static analysis
    analyzer = DiagnosticAnalyzer()
    problems = analyzer.analyze_document(document)
    stats = analyzer.get_statistics(problems)

    result = {
        "document_id": request.document_id,
        "static_analysis": {
            "problems": [p.to_dict() for p in problems],
            "statistics": stats,
        },
    }

    # Optionally run question-based testing
    if request.include_questions:
        tester = _get_retrieval_tester()

        # Make sure chunks are indexed
        if not tester.is_indexed:
            tester.index_chunks(document.chunks)

        runner = QuestionTestRunner(tester)
        test_report = runner.run_diagnostic_tests(document, request.top_k)

        result["question_tests"] = test_report.to_dict()

    return result


@router.get("/api/diagnostics/{document_id}/problems")
async def get_problems(document_id: str) -> dict[str, Any]:
    """Get all detected problems for a document (static analysis only)."""
    project = _get_project()

    document = project.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    analyzer = DiagnosticAnalyzer()
    problems = analyzer.analyze_document(document)
    stats = analyzer.get_statistics(problems)

    return {
        "document_id": document_id,
        "problems": [p.to_dict() for p in problems],
        "statistics": stats,
    }


class GenerateQuestionsRequest(BaseModel):
    """Request for generating diagnostic questions."""

    document_id: str


@router.post("/api/diagnostics/generate-questions")
async def generate_questions(request: GenerateQuestionsRequest) -> dict[str, Any]:
    """Generate diagnostic questions from chunks."""
    project = _get_project()

    document = project.get_document(request.document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    generator = QuestionGenerator()
    questions = generator.generate_all_questions(document.chunks)

    # Group by test type
    by_type = {}
    for q in questions:
        if q.test_type not in by_type:
            by_type[q.test_type] = []
        by_type[q.test_type].append({
            "question": q.question,
            "expected_chunks": q.expected_chunk_ids,
            "source_chunk": q.source_chunk_id,
            "metadata": q.metadata,
        })

    return {
        "document_id": request.document_id,
        "total_questions": len(questions),
        "by_type": {k: len(v) for k, v in by_type.items()},
        "questions": by_type,
    }


class TestQuestionsRequest(BaseModel):
    """Request for testing generated questions."""

    document_id: str
    top_k: int = 5


@router.post("/api/diagnostics/test-questions")
async def test_questions(request: TestQuestionsRequest) -> dict[str, Any]:
    """Run question-based diagnostic tests."""
    project = _get_project()

    document = project.get_document(request.document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    tester = _get_retrieval_tester()

    # Index chunks if needed
    if not tester.is_indexed:
        tester.index_chunks(document.chunks)

    runner = QuestionTestRunner(tester)
    report = runner.run_diagnostic_tests(document, request.top_k)

    return report.to_dict()


# ============================================================================
# Fix Endpoints (Automatic Problem Resolution)
# ============================================================================


class PreviewFixesRequest(BaseModel):
    """Request for previewing automatic fixes."""

    document_id: str
    auto_resolve_conflicts: bool = True


@router.post("/api/diagnostics/preview-fixes")
async def preview_fixes(request: PreviewFixesRequest) -> dict[str, Any]:
    """
    Preview automatic fixes for detected problems.

    Returns a fix plan without applying changes.
    User can review before applying.
    """
    project = _get_project()

    document = project.get_document(request.document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    # Run diagnostics to find problems
    analyzer = DiagnosticAnalyzer()
    problems = analyzer.analyze_document(document)

    # Plan fixes
    orchestrator = FixOrchestrator()
    plan = orchestrator.plan_fixes(
        problems,
        document.chunks,
        auto_resolve_conflicts=request.auto_resolve_conflicts,
    )

    return {
        "document_id": request.document_id,
        "problems_found": len(problems),
        "fix_plan": plan.to_dict(),
    }


class ApplyFixesRequest(BaseModel):
    """Request for applying automatic fixes."""

    document_id: str
    auto_resolve_conflicts: bool = True
    validate: bool = True  # Re-run diagnostics after fixes


@router.post("/api/diagnostics/apply-fixes")
async def apply_fixes(request: ApplyFixesRequest) -> dict[str, Any]:
    """
    Apply automatic fixes to document chunks.

    1. Run diagnostics
    2. Plan fixes
    3. Execute fixes
    4. Optionally validate improvements
    5. Update document with fixed chunks
    """
    project = _get_project()

    document = project.get_document(request.document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    # Run diagnostics
    analyzer = DiagnosticAnalyzer()
    problems_before = analyzer.analyze_document(document)
    stats_before = analyzer.get_statistics(problems_before)

    if not problems_before:
        return {
            "document_id": request.document_id,
            "result": "no_problems",
            "message": "No problems detected - nothing to fix",
        }

    # Plan and execute fixes
    orchestrator = FixOrchestrator()
    plan = orchestrator.plan_fixes(
        problems_before,
        document.chunks,
        auto_resolve_conflicts=request.auto_resolve_conflicts,
    )

    result = orchestrator.execute_plan(
        plan,
        document.chunks,
        validate=request.validate,
    )

    if not result.success:
        return {
            "document_id": request.document_id,
            "result": "failed",
            "errors": result.errors,
            "fix_result": result.to_dict(),
        }

    # Update document with fixed chunks
    document.chunks = result.new_chunks

    # Re-run diagnostics on fixed chunks
    problems_after = analyzer.analyze_document(document)
    stats_after = analyzer.get_statistics(problems_after)

    # Re-index chunks for retrieval
    tester = _get_retrieval_tester()
    tester.index_documents(project.documents)

    return {
        "document_id": request.document_id,
        "result": "success",
        "fix_result": result.to_dict(),
        "before": {
            "problems": len(problems_before),
            "statistics": stats_before,
        },
        "after": {
            "problems": len(problems_after),
            "statistics": stats_after,
        },
        "improvement": {
            "problems_fixed": len(problems_before) - len(problems_after),
            "reduction_rate": (len(problems_before) - len(problems_after)) / len(problems_before) if len(problems_before) > 0 else 0,
        },
    }


# ============================================================================
# Helper Functions
# ============================================================================


def _get_project() -> ChonkProject:
    """Get the current project or raise error."""
    project = _state.get("project")
    if project is None:
        raise HTTPException(
            status_code=400,
            detail="No project loaded. Create or open a project first.",
        )
    return project


def _get_retrieval_tester() -> RetrievalTester:
    """Get the retrieval tester or create one if needed."""
    tester = _state.get("tester")
    if tester is None:
        tester = RetrievalTester()
        _state["tester"] = tester
    return tester


# ============================================================================
# Mount router on the standalone app (backward compatibility)
# ============================================================================

app.include_router(router)


# ============================================================================
# Main Entry Point
# ============================================================================


def run_server(host: str = "127.0.0.1", port: int = 8420) -> None:
    """Run the CHONK server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
