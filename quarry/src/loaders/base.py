"""
Base loader class and registry for document loaders.

All document loaders inherit from BaseLoader and register themselves
with the LoaderRegistry for automatic format detection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from chonk.core.document import Block, ChonkDocument, DocumentMetadata


class LoaderError(Exception):
    """Base exception for loader errors."""

    def __init__(self, message: str, source_path: Path | None = None, details: str | None = None):
        self.source_path = source_path
        self.details = details
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": str(self),
            "source_path": str(self.source_path) if self.source_path else None,
            "details": self.details,
        }


class BaseLoader(ABC):
    """
    Abstract base class for document loaders.

    Each loader is responsible for:
    1. Detecting if it can handle a file type
    2. Loading the file and extracting content as Blocks
    3. Extracting document metadata
    """

    # Subclasses should define these
    SUPPORTED_EXTENSIONS: ClassVar[list[str]] = []
    LOADER_NAME: ClassVar[str] = "base"

    def __init__(self) -> None:
        self._errors: list[str] = []
        self._warnings: list[str] = []
        self._info: list[str] = []

    @classmethod
    def can_load(cls, path: Path) -> bool:
        """Check if this loader can handle the given file."""
        return path.suffix.lower() in cls.SUPPORTED_EXTENSIONS

    @abstractmethod
    def load(self, path: Path) -> tuple[list[Block], DocumentMetadata]:
        """
        Load a document and extract blocks and metadata.

        Args:
            path: Path to the document file

        Returns:
            Tuple of (blocks, metadata)

        Raises:
            LoaderError: If loading fails
        """
        pass

    def load_document(self, path: Path) -> ChonkDocument:
        """
        Load a document into a full ChonkDocument.

        This is a convenience method that wraps load() and creates
        the document structure.
        """
        if not path.exists():
            raise LoaderError(f"File not found: {path}", source_path=path)

        if not self.can_load(path):
            raise LoaderError(
                f"Unsupported file type: {path.suffix}",
                source_path=path,
                details=f"Supported types: {', '.join(self.SUPPORTED_EXTENSIONS)}",
            )

        blocks, metadata = self.load(path)

        # Update metadata with file info
        metadata.file_size_bytes = path.stat().st_size

        return ChonkDocument(
            id=ChonkDocument.generate_id(),
            source_path=path,
            source_type=path.suffix.lower().lstrip("."),
            blocks=blocks,
            chunks=[],  # Chunks are created by chunkers, not loaders
            metadata=metadata,
            loader_used=self.LOADER_NAME,
        )

    @property
    def errors(self) -> list[str]:
        """Get any errors that occurred during loading."""
        return self._errors

    @property
    def warnings(self) -> list[str]:
        """Get any warnings that occurred during loading."""
        return self._warnings

    def _add_error(self, error: str) -> None:
        """Record an error during loading."""
        self._errors.append(error)

    def _add_warning(self, warning: str) -> None:
        """Record a warning during loading."""
        self._warnings.append(warning)

    def _add_info(self, info: str) -> None:
        """Record an informational message during loading."""
        self._info.append(info)

    def _reset_messages(self) -> None:
        """Reset errors, warnings, and info for a new load operation."""
        self._errors = []
        self._warnings = []
        self._info = []

    @property
    def info(self) -> list[str]:
        """Get any informational messages from loading."""
        return self._info


class LoaderRegistry:
    """
    Registry of available document loaders.

    Use this to automatically select the appropriate loader for a file.
    """

    _loaders: ClassVar[list[type[BaseLoader]]] = []

    @classmethod
    def register(cls, loader_class: type[BaseLoader]) -> type[BaseLoader]:
        """
        Register a loader class. Can be used as a decorator.

        @LoaderRegistry.register
        class MyLoader(BaseLoader):
            ...
        """
        if loader_class not in cls._loaders:
            cls._loaders.append(loader_class)
        return loader_class

    @classmethod
    def get_loader(cls, path: Path) -> BaseLoader | None:
        """Get an appropriate loader for the given file path."""
        for loader_class in cls._loaders:
            if loader_class.can_load(path):
                return loader_class()
        return None

    @classmethod
    def get_loader_by_name(cls, name: str) -> type[BaseLoader] | None:
        """Get a loader class by its name."""
        for loader_class in cls._loaders:
            if loader_class.LOADER_NAME == name:
                return loader_class
        return None

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """Get all supported file extensions."""
        extensions = []
        for loader_class in cls._loaders:
            extensions.extend(loader_class.SUPPORTED_EXTENSIONS)
        return list(set(extensions))

    @classmethod
    def load_document(cls, path: Path) -> ChonkDocument:
        """
        Load a document using the appropriate loader.

        Raises:
            LoaderError: If no loader is available or loading fails
        """
        loader = cls.get_loader(path)
        if loader is None:
            supported = ", ".join(cls.supported_extensions())
            raise LoaderError(
                f"No loader available for file type: {path.suffix}",
                source_path=path,
                details=f"Supported types: {supported}",
            )
        return loader.load_document(path)
