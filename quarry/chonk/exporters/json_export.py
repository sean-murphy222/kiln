"""
JSON exporter for CHONK.

Exports chunks as a single JSON file with full metadata,
suitable for debugging or custom integrations.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from chonk.core.document import ChonkDocument, ChonkProject
from chonk.exporters.base import BaseExporter, ExporterRegistry
from chonk.exporters.schema import SCHEMA_VERSION


@ExporterRegistry.register
class JSONExporter(BaseExporter):
    """
    Export chunks as a JSON file with full structure.

    Includes document metadata, chunk details, and processing info.
    """

    EXPORTER_NAME: ClassVar[str] = "json"
    FILE_EXTENSION: ClassVar[str] = ".json"

    def export_document(self, document: ChonkDocument, path: Path) -> Path:
        """Export document to JSON with full metadata."""
        path = self._ensure_extension(path)

        export_data = {
            "version": SCHEMA_VERSION,
            "exported_at": datetime.now().isoformat(),
            "exporter": "chonk",
            "document": self._document_to_dict(document),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return path

    def export_project(self, project: ChonkProject, path: Path) -> Path:
        """Export entire project to JSON."""
        path = self._ensure_extension(path)

        export_data = {
            "version": SCHEMA_VERSION,
            "exported_at": datetime.now().isoformat(),
            "exporter": "chonk",
            "project": {
                "id": project.id,
                "name": project.name,
                "created_at": project.created_at.isoformat(),
                "settings": project.settings.to_dict(),
                "documents": [self._document_to_dict(doc) for doc in project.documents],
                "test_suites": [suite.to_dict() for suite in project.test_suites],
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return path

    def _document_to_dict(self, document: ChonkDocument) -> dict[str, Any]:
        """Convert document to export dictionary."""
        return {
            "id": document.id,
            "source": str(document.source_path),
            "source_type": document.source_type,
            "metadata": document.metadata.to_dict(),
            "processing": {
                "loader": document.loader_used,
                "parser": document.parser_used,
                "chunker": document.chunker_used,
                "chunker_config": document.chunker_config,
                "loaded_at": document.loaded_at.isoformat(),
                "chunked_at": (
                    document.last_chunked_at.isoformat() if document.last_chunked_at else None
                ),
            },
            "chunks": [self._chunk_to_dict(c) for c in document.chunks],
        }

    def _chunk_to_dict(self, chunk) -> dict[str, Any]:
        """Convert chunk to export dictionary."""
        return {
            "id": chunk.id,
            "content": chunk.content,
            "token_count": chunk.token_count,
            "quality": chunk.quality.to_dict(),
            "hierarchy_path": chunk.hierarchy_path,
            "page_range": chunk.page_range,
            "block_ids": chunk.block_ids,
            "user_metadata": chunk.user_metadata.to_dict(),
            "system_metadata": chunk.system_metadata,
            "is_modified": chunk.is_modified,
            "is_locked": chunk.is_locked,
        }
