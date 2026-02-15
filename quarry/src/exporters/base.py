"""
Base exporter class and registry.

All exporters inherit from BaseExporter and register themselves
with the ExporterRegistry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from chonk.core.document import ChonkDocument, ChonkProject


class BaseExporter(ABC):
    """
    Abstract base class for chunk exporters.

    Exporters convert chunks to various output formats for use
    in downstream RAG pipelines.
    """

    EXPORTER_NAME: ClassVar[str] = "base"
    FILE_EXTENSION: ClassVar[str] = ""

    @abstractmethod
    def export_document(self, document: ChonkDocument, path: Path) -> Path:
        """
        Export a single document's chunks.

        Args:
            document: Document to export
            path: Output file path

        Returns:
            Path to exported file
        """
        pass

    def export_project(self, project: ChonkProject, path: Path) -> Path:
        """
        Export all documents in a project as a single file.

        Args:
            project: Project to export
            path: Output file path

        Returns:
            Path to exported file
        """
        # Default: combine all documents
        # Subclasses can override for format-specific handling
        combined_chunks = []
        for doc in project.documents:
            for chunk in doc.chunks:
                combined_chunks.append((doc, chunk))

        return self._export_combined(combined_chunks, path)

    def _export_combined(
        self,
        chunks: list[tuple[ChonkDocument, Any]],
        path: Path,
    ) -> Path:
        """Export combined chunks from multiple documents."""
        # Default implementation - subclasses should override
        raise NotImplementedError("Subclass must implement _export_combined")

    def _ensure_extension(self, path: Path) -> Path:
        """Ensure the path has the correct extension."""
        if path.suffix.lower() != self.FILE_EXTENSION.lower():
            return path.with_suffix(self.FILE_EXTENSION)
        return path


class ExporterRegistry:
    """Registry of available exporters."""

    _exporters: ClassVar[dict[str, type[BaseExporter]]] = {}

    @classmethod
    def register(cls, exporter_class: type[BaseExporter]) -> type[BaseExporter]:
        """Register an exporter class."""
        cls._exporters[exporter_class.EXPORTER_NAME] = exporter_class
        return exporter_class

    @classmethod
    def get_exporter(cls, name: str) -> BaseExporter | None:
        """Get an exporter by name."""
        exporter_class = cls._exporters.get(name)
        if exporter_class:
            return exporter_class()
        return None

    @classmethod
    def available_exporters(cls) -> list[str]:
        """Get list of available exporter names."""
        return list(cls._exporters.keys())

    @classmethod
    def export_document(
        cls,
        document: ChonkDocument,
        path: Path,
        format: str,
    ) -> Path:
        """Export a document using the specified format."""
        exporter = cls.get_exporter(format)
        if exporter is None:
            available = ", ".join(cls.available_exporters())
            raise ValueError(f"Unknown export format: {format}. Available: {available}")
        return exporter.export_document(document, path)

    @classmethod
    def export_project(
        cls,
        project: ChonkProject,
        path: Path,
        format: str,
    ) -> Path:
        """Export a project using the specified format."""
        exporter = cls.get_exporter(format)
        if exporter is None:
            available = ", ".join(cls.available_exporters())
            raise ValueError(f"Unknown export format: {format}. Available: {available}")
        return exporter.export_project(project, path)
