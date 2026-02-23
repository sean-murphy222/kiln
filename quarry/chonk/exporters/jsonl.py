"""
JSONL exporter for CHONK.

Exports chunks as newline-delimited JSON, compatible with
LangChain, LlamaIndex, and most RAG frameworks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from chonk.core.document import Chunk, ChonkDocument, ChonkProject
from chonk.exporters.base import BaseExporter, ExporterRegistry


@ExporterRegistry.register
class JSONLExporter(BaseExporter):
    """
    Export chunks as JSONL (newline-delimited JSON).

    Each line is a complete JSON object representing one chunk.
    This is the most common format for RAG pipelines.
    """

    EXPORTER_NAME: ClassVar[str] = "jsonl"
    FILE_EXTENSION: ClassVar[str] = ".jsonl"

    def export_document(self, document: ChonkDocument, path: Path) -> Path:
        """Export document chunks to JSONL."""
        path = self._ensure_extension(path)

        with open(path, "w", encoding="utf-8") as f:
            for chunk in document.chunks:
                record = self._chunk_to_record(chunk, document)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return path

    def export_project(self, project: ChonkProject, path: Path) -> Path:
        """Export all project chunks to a single JSONL file."""
        path = self._ensure_extension(path)

        with open(path, "w", encoding="utf-8") as f:
            for doc in project.documents:
                for chunk in doc.chunks:
                    record = self._chunk_to_record(chunk, doc)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return path

    def _chunk_to_record(
        self,
        chunk: Chunk,
        document: ChonkDocument,
    ) -> dict[str, Any]:
        """Convert a chunk to a JSONL record."""
        # Standard fields expected by most RAG frameworks
        record = {
            "id": chunk.id,
            "text": chunk.content,
            "metadata": {
                # Source information
                "source": str(document.source_path),
                "source_type": document.source_type,
                # Location
                "page": chunk.page_range[0] if chunk.page_range else None,
                "page_end": chunk.page_range[1] if chunk.page_range else None,
                # Hierarchy
                "hierarchy_path": chunk.hierarchy_path,
                # Quality
                "quality_score": chunk.quality.overall,
                # Token count
                "token_count": chunk.token_count,
                # User metadata (flattened)
                **self._flatten_user_metadata(chunk),
            },
        }

        return record

    def _flatten_user_metadata(self, chunk: Chunk) -> dict[str, Any]:
        """Flatten user metadata for inclusion in record."""
        result = {}

        if chunk.user_metadata.tags:
            result["tags"] = chunk.user_metadata.tags

        if chunk.user_metadata.hierarchy_hint:
            result["hierarchy_hint"] = chunk.user_metadata.hierarchy_hint

        if chunk.user_metadata.notes:
            result["notes"] = chunk.user_metadata.notes

        # Add custom key-value pairs with prefix
        for key, value in chunk.user_metadata.custom.items():
            result[f"custom_{key}"] = value

        return result
