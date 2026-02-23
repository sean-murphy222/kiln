"""
CSV exporter for CHONK.

Exports chunks as a CSV file for use in spreadsheets
or simple data analysis.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, ClassVar

from chonk.core.document import Chunk, ChonkDocument, ChonkProject
from chonk.exporters.base import BaseExporter, ExporterRegistry


@ExporterRegistry.register
class CSVExporter(BaseExporter):
    """
    Export chunks as CSV.

    Simple format with one row per chunk. Good for spreadsheet
    viewing but loses some metadata structure.
    """

    EXPORTER_NAME: ClassVar[str] = "csv"
    FILE_EXTENSION: ClassVar[str] = ".csv"

    # CSV columns
    COLUMNS = [
        "chunk_id",
        "document",
        "content",
        "token_count",
        "quality_score",
        "page_start",
        "page_end",
        "hierarchy_path",
        "tags",
        "notes",
    ]

    def export_document(self, document: ChonkDocument, path: Path) -> Path:
        """Export document chunks to CSV."""
        path = self._ensure_extension(path)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()

            for chunk in document.chunks:
                row = self._chunk_to_row(chunk, document)
                writer.writerow(row)

        return path

    def export_project(self, project: ChonkProject, path: Path) -> Path:
        """Export all project chunks to CSV."""
        path = self._ensure_extension(path)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()

            for doc in project.documents:
                for chunk in doc.chunks:
                    row = self._chunk_to_row(chunk, doc)
                    writer.writerow(row)

        return path

    def _chunk_to_row(
        self,
        chunk: Chunk,
        document: ChonkDocument,
    ) -> dict[str, Any]:
        """Convert a chunk to a CSV row."""
        page_range = chunk.page_range

        return {
            "chunk_id": chunk.id,
            "document": document.source_path.name if document.source_path else document.id,
            "content": chunk.content,
            "token_count": chunk.token_count,
            "quality_score": round(chunk.quality.overall, 3),
            "page_start": page_range[0] if page_range else "",
            "page_end": page_range[1] if page_range else "",
            "hierarchy_path": chunk.hierarchy_path,
            "tags": "; ".join(chunk.user_metadata.tags),
            "notes": chunk.user_metadata.notes or "",
        }
