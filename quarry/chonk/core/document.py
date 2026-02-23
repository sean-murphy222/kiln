"""
Unified Document Model (UDM) for CHONK.

This module defines the core data structures that represent documents,
blocks, chunks, and projects throughout the application.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class BlockType(Enum):
    """Types of content blocks that can be extracted from documents."""

    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"
    LIST = "list"
    LIST_ITEM = "list_item"
    FOOTER = "footer"
    HEADER = "header"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    PAGE_BREAK = "page_break"


@dataclass
class BoundingBox:
    """
    Represents the visual location of a block on a page.

    Coordinates are in points (1/72 inch) from top-left origin.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    page: int

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict[str, Any]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "page": self.page,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoundingBox:
        return cls(
            x1=data["x1"],
            y1=data["y1"],
            x2=data["x2"],
            y2=data["y2"],
            page=data["page"],
        )


@dataclass
class Block:
    """
    A semantic unit extracted from a source document.

    Blocks are the atomic units of content - paragraphs, headings, tables,
    images, etc. They are grouped into Chunks for embedding.
    """

    id: str
    type: BlockType
    content: str
    bbox: BoundingBox | None = None
    page: int = 1
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # Parser confidence score (0-1)
    heading_level: int | None = None  # For HEADING blocks: 1-6

    @staticmethod
    def generate_id() -> str:
        return f"block_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "page": self.page,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "heading_level": self.heading_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Block:
        return cls(
            id=data["id"],
            type=BlockType(data["type"]),
            content=data["content"],
            bbox=BoundingBox.from_dict(data["bbox"]) if data.get("bbox") else None,
            page=data.get("page", 1),
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            metadata=data.get("metadata", {}),
            confidence=data.get("confidence", 1.0),
            heading_level=data.get("heading_level"),
        )


@dataclass
class QualityScore:
    """
    Detailed quality assessment for a chunk.

    Each component is scored 0-1 with a weight for the final score.
    """

    token_range: float = 1.0  # Is token count in optimal range?
    sentence_complete: float = 1.0  # Does it start/end properly?
    hierarchy_preserved: float = 1.0  # No orphan headings?
    table_integrity: float = 1.0  # Tables not split?
    reference_complete: float = 1.0  # No orphan references?

    # Weights for final score calculation
    WEIGHTS: dict[str, float] = field(
        default_factory=lambda: {
            "token_range": 0.25,
            "sentence_complete": 0.20,
            "hierarchy_preserved": 0.25,
            "table_integrity": 0.15,
            "reference_complete": 0.15,
        }
    )

    @property
    def overall(self) -> float:
        """Calculate weighted overall quality score."""
        components = {
            "token_range": self.token_range,
            "sentence_complete": self.sentence_complete,
            "hierarchy_preserved": self.hierarchy_preserved,
            "table_integrity": self.table_integrity,
            "reference_complete": self.reference_complete,
        }
        total = sum(score * self.WEIGHTS[name] for name, score in components.items())
        return round(total, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_range": self.token_range,
            "sentence_complete": self.sentence_complete,
            "hierarchy_preserved": self.hierarchy_preserved,
            "table_integrity": self.table_integrity,
            "reference_complete": self.reference_complete,
            "overall": self.overall,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualityScore:
        return cls(
            token_range=data.get("token_range", 1.0),
            sentence_complete=data.get("sentence_complete", 1.0),
            hierarchy_preserved=data.get("hierarchy_preserved", 1.0),
            table_integrity=data.get("table_integrity", 1.0),
            reference_complete=data.get("reference_complete", 1.0),
        )


@dataclass
class ChunkMetadata:
    """
    User-defined metadata that can be attached to chunks.

    This is separate from system metadata to keep user additions clean.
    """

    tags: list[str] = field(default_factory=list)
    hierarchy_hint: str | None = None  # e.g., "Section 3 > Safety"
    notes: str | None = None
    custom: dict[str, Any] = field(default_factory=dict)  # Key-value pairs

    def to_dict(self) -> dict[str, Any]:
        return {
            "tags": self.tags,
            "hierarchy_hint": self.hierarchy_hint,
            "notes": self.notes,
            "custom": self.custom,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkMetadata:
        return cls(
            tags=data.get("tags", []),
            hierarchy_hint=data.get("hierarchy_hint"),
            notes=data.get("notes"),
            custom=data.get("custom", {}),
        )


@dataclass
class Chunk:
    """
    A group of blocks that will become one embedding.

    Chunks are the unit of retrieval - each chunk becomes one vector
    in the embedding space.
    """

    id: str
    block_ids: list[str]
    content: str
    token_count: int
    quality: QualityScore = field(default_factory=QualityScore)
    hierarchy_path: str = ""  # e.g., "Chapter 1 > Section A > Safety"
    user_metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    system_metadata: dict[str, Any] = field(default_factory=dict)

    # Edit tracking
    is_modified: bool = False  # User has edited this chunk
    is_locked: bool = False  # User has locked this chunk from re-chunking
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime | None = None

    @staticmethod
    def generate_id() -> str:
        return f"chunk_{uuid.uuid4().hex[:12]}"

    @property
    def page_range(self) -> tuple[int, int] | None:
        """Get the page range from system metadata if available."""
        start = self.system_metadata.get("start_page")
        end = self.system_metadata.get("end_page")
        if start is not None and end is not None:
            return (start, end)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "block_ids": self.block_ids,
            "content": self.content,
            "token_count": self.token_count,
            "quality": self.quality.to_dict(),
            "hierarchy_path": self.hierarchy_path,
            "user_metadata": self.user_metadata.to_dict(),
            "system_metadata": self.system_metadata,
            "is_modified": self.is_modified,
            "is_locked": self.is_locked,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(
            id=data["id"],
            block_ids=data["block_ids"],
            content=data["content"],
            token_count=data["token_count"],
            quality=QualityScore.from_dict(data.get("quality", {})),
            hierarchy_path=data.get("hierarchy_path", ""),
            user_metadata=ChunkMetadata.from_dict(data.get("user_metadata", {})),
            system_metadata=data.get("system_metadata", {}),
            is_modified=data.get("is_modified", False),
            is_locked=data.get("is_locked", False),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(),
            modified_at=datetime.fromisoformat(data["modified_at"])
            if data.get("modified_at")
            else None,
        )


@dataclass
class DocumentMetadata:
    """Metadata extracted from the source document."""

    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: list[str] = field(default_factory=list)
    created_date: datetime | None = None
    modified_date: datetime | None = None
    page_count: int = 0
    word_count: int = 0
    file_size_bytes: int = 0
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "author": self.author,
            "subject": self.subject,
            "keywords": self.keywords,
            "created_date": self.created_date.isoformat() if self.created_date else None,
            "modified_date": self.modified_date.isoformat() if self.modified_date else None,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "file_size_bytes": self.file_size_bytes,
            "custom": self.custom,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentMetadata:
        return cls(
            title=data.get("title"),
            author=data.get("author"),
            subject=data.get("subject"),
            keywords=data.get("keywords", []),
            created_date=datetime.fromisoformat(data["created_date"])
            if data.get("created_date")
            else None,
            modified_date=datetime.fromisoformat(data["modified_date"])
            if data.get("modified_date")
            else None,
            page_count=data.get("page_count", 0),
            word_count=data.get("word_count", 0),
            file_size_bytes=data.get("file_size_bytes", 0),
            custom=data.get("custom", {}),
        )


@dataclass
class ChonkDocument:
    """
    A fully processed document with blocks and chunks.

    This is the main data structure for a single document within CHONK.
    """

    id: str
    source_path: Path
    source_type: str  # pdf, docx, md, txt, etc.
    blocks: list[Block]
    chunks: list[Chunk]
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)

    # Processing info
    loader_used: str = ""
    parser_used: str = ""
    chunker_used: str = ""
    chunker_config: dict[str, Any] = field(default_factory=dict)

    # Timestamps
    loaded_at: datetime = field(default_factory=datetime.now)
    last_chunked_at: datetime | None = None

    @staticmethod
    def generate_id() -> str:
        return f"doc_{uuid.uuid4().hex[:12]}"

    def get_block(self, block_id: str) -> Block | None:
        """Get a block by ID."""
        for block in self.blocks:
            if block.id == block_id:
                return block
        return None

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Get a chunk by ID."""
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None

    def get_blocks_for_chunk(self, chunk: Chunk) -> list[Block]:
        """Get all blocks that belong to a chunk."""
        return [b for b in self.blocks if b.id in chunk.block_ids]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_path": str(self.source_path),
            "source_type": self.source_type,
            "blocks": [b.to_dict() for b in self.blocks],
            "chunks": [c.to_dict() for c in self.chunks],
            "metadata": self.metadata.to_dict(),
            "loader_used": self.loader_used,
            "parser_used": self.parser_used,
            "chunker_used": self.chunker_used,
            "chunker_config": self.chunker_config,
            "loaded_at": self.loaded_at.isoformat(),
            "last_chunked_at": self.last_chunked_at.isoformat()
            if self.last_chunked_at
            else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChonkDocument:
        return cls(
            id=data["id"],
            source_path=Path(data["source_path"]),
            source_type=data["source_type"],
            blocks=[Block.from_dict(b) for b in data.get("blocks", [])],
            chunks=[Chunk.from_dict(c) for c in data.get("chunks", [])],
            metadata=DocumentMetadata.from_dict(data.get("metadata", {})),
            loader_used=data.get("loader_used", ""),
            parser_used=data.get("parser_used", ""),
            chunker_used=data.get("chunker_used", ""),
            chunker_config=data.get("chunker_config", {}),
            loaded_at=datetime.fromisoformat(data["loaded_at"])
            if data.get("loaded_at")
            else datetime.now(),
            last_chunked_at=datetime.fromisoformat(data["last_chunked_at"])
            if data.get("last_chunked_at")
            else None,
        )


@dataclass
class TestQuery:
    """A saved test query for retrieval testing."""

    id: str
    query: str
    expected_chunk_ids: list[str] = field(default_factory=list)  # Should match
    excluded_chunk_ids: list[str] = field(default_factory=list)  # Should NOT match
    notes: str | None = None
    created_at: datetime = field(default_factory=datetime.now)

    @staticmethod
    def generate_id() -> str:
        return f"query_{uuid.uuid4().hex[:8]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "expected_chunk_ids": self.expected_chunk_ids,
            "excluded_chunk_ids": self.excluded_chunk_ids,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestQuery:
        return cls(
            id=data["id"],
            query=data["query"],
            expected_chunk_ids=data.get("expected_chunk_ids", []),
            excluded_chunk_ids=data.get("excluded_chunk_ids", []),
            notes=data.get("notes"),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(),
        )


@dataclass
class TestSuite:
    """A collection of test queries for a project."""

    id: str
    name: str
    queries: list[TestQuery] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime | None = None

    @staticmethod
    def generate_id() -> str:
        return f"suite_{uuid.uuid4().hex[:8]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "queries": [q.to_dict() for q in self.queries],
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestSuite:
        return cls(
            id=data["id"],
            name=data["name"],
            queries=[TestQuery.from_dict(q) for q in data.get("queries", [])],
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(),
            modified_at=datetime.fromisoformat(data["modified_at"])
            if data.get("modified_at")
            else None,
        )


@dataclass
class ProjectSettings:
    """User-configurable settings for a project."""

    default_chunker: str = "hierarchy"
    default_chunk_size: int = 400
    default_overlap: int = 50
    embedding_model: str = "all-MiniLM-L6-v2"
    output_directory: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_chunker": self.default_chunker,
            "default_chunk_size": self.default_chunk_size,
            "default_overlap": self.default_overlap,
            "embedding_model": self.embedding_model,
            "output_directory": str(self.output_directory) if self.output_directory else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectSettings:
        return cls(
            default_chunker=data.get("default_chunker", "hierarchy"),
            default_chunk_size=data.get("default_chunk_size", 400),
            default_overlap=data.get("default_overlap", 50),
            embedding_model=data.get("embedding_model", "all-MiniLM-L6-v2"),
            output_directory=Path(data["output_directory"])
            if data.get("output_directory")
            else None,
        )


@dataclass
class ChonkProject:
    """
    A CHONK project containing multiple documents and shared settings.

    Projects are saved as .chonk files (JSON format).
    """

    id: str
    name: str
    documents: list[ChonkDocument] = field(default_factory=list)
    test_suites: list[TestSuite] = field(default_factory=list)
    settings: ProjectSettings = field(default_factory=ProjectSettings)

    # File location
    project_path: Path | None = None  # Where the .chonk file is saved

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime | None = None

    @staticmethod
    def generate_id() -> str:
        return f"proj_{uuid.uuid4().hex[:8]}"

    def get_document(self, doc_id: str) -> ChonkDocument | None:
        """Get a document by ID."""
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None

    def get_all_chunks(self) -> list[tuple[ChonkDocument, Chunk]]:
        """Get all chunks across all documents with their parent document."""
        result = []
        for doc in self.documents:
            for chunk in doc.chunks:
                result.append((doc, chunk))
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "documents": [d.to_dict() for d in self.documents],
            "test_suites": [t.to_dict() for t in self.test_suites],
            "settings": self.settings.to_dict(),
            "project_path": str(self.project_path) if self.project_path else None,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChonkProject:
        return cls(
            id=data["id"],
            name=data["name"],
            documents=[ChonkDocument.from_dict(d) for d in data.get("documents", [])],
            test_suites=[TestSuite.from_dict(t) for t in data.get("test_suites", [])],
            settings=ProjectSettings.from_dict(data.get("settings", {})),
            project_path=Path(data["project_path"]) if data.get("project_path") else None,
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(),
            modified_at=datetime.fromisoformat(data["modified_at"])
            if data.get("modified_at")
            else None,
        )

    def save(self, path: Path | None = None) -> Path:
        """Save project to a .chonk file."""
        import json

        save_path = path or self.project_path
        if save_path is None:
            raise ValueError("No path specified and project has no saved path")

        if not save_path.suffix == ".chonk":
            save_path = save_path.with_suffix(".chonk")

        self.project_path = save_path
        self.modified_at = datetime.now()

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        return save_path

    @classmethod
    def load(cls, path: Path) -> ChonkProject:
        """Load a project from a .chonk file."""
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        project = cls.from_dict(data)
        project.project_path = path
        return project
