"""Persistent storage for human-labeled document fingerprint examples.

Provides an append-only JSONL-backed queue of manual classifications
that feeds into classifier retraining.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from chonk.tier1.fingerprinter import DocumentFingerprint
from chonk.tier1.taxonomy import DocumentType

logger = logging.getLogger(__name__)


@dataclass
class ManualExample:
    """One human-labeled fingerprint with provenance.

    Attributes:
        example_id: UUID string, stable identifier.
        fingerprint: The 49-float DocumentFingerprint.
        document_type: Human-assigned label (never UNKNOWN).
        original_confidence: Classifier confidence that triggered review.
        source_document_name: Filename for reference (no path, no PII).
        created_at: ISO 8601 timestamp string.
        reviewer_note: Optional free-text note from reviewer.
    """

    example_id: str
    fingerprint: DocumentFingerprint
    document_type: DocumentType
    original_confidence: float
    source_document_name: str
    created_at: str
    reviewer_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "example_id": self.example_id,
            "fingerprint": self.fingerprint.to_dict(),
            "document_type": self.document_type.value,
            "original_confidence": self.original_confidence,
            "source_document_name": self.source_document_name,
            "created_at": self.created_at,
            "reviewer_note": self.reviewer_note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManualExample:
        """Deserialize from dictionary."""
        return cls(
            example_id=data["example_id"],
            fingerprint=DocumentFingerprint.from_dict(data["fingerprint"]),
            document_type=DocumentType(data["document_type"]),
            original_confidence=data["original_confidence"],
            source_document_name=data["source_document_name"],
            created_at=data["created_at"],
            reviewer_note=data.get("reviewer_note", ""),
        )

    @staticmethod
    def create(
        fingerprint: DocumentFingerprint,
        document_type: DocumentType,
        original_confidence: float,
        source_document_name: str,
        reviewer_note: str = "",
    ) -> ManualExample:
        """Create a new ManualExample with generated ID and timestamp.

        Args:
            fingerprint: The document fingerprint.
            document_type: Human-assigned type (must not be UNKNOWN).
            original_confidence: Classifier confidence that triggered review.
            source_document_name: Display filename.
            reviewer_note: Optional note.

        Returns:
            New ManualExample with UUID and current timestamp.
        """
        return ManualExample(
            example_id=uuid.uuid4().hex,
            fingerprint=fingerprint,
            document_type=document_type,
            original_confidence=original_confidence,
            source_document_name=Path(source_document_name).name,
            created_at=datetime.now().isoformat(),
            reviewer_note=reviewer_note,
        )


class ManualLabelStore:
    """Append-only file-backed queue of human-labeled fingerprint examples.

    Each example is stored as a single JSON line in a .jsonl file.

    Args:
        store_path: Path to the .jsonl file. Created on first add().
            Parent directory must exist.

    Example::

        store = ManualLabelStore(Path("data/manual_labels.jsonl"))
        example = ManualExample.create(fp, DocumentType.FORM, 0.3, "doc.pdf")
        store.add(example)
        all_examples = store.load_all()
    """

    def __init__(self, store_path: str | Path) -> None:
        self._path = Path(store_path).resolve()
        if not self._path.parent.exists():
            raise ValueError(f"Parent directory does not exist: {self._path.parent}")

    @property
    def path(self) -> Path:
        """Return the resolved store file path."""
        return self._path

    def add(self, example: ManualExample) -> None:
        """Append one labeled example to the store.

        Args:
            example: A ManualExample with document_type != UNKNOWN.

        Raises:
            ValueError: If document_type is UNKNOWN.
        """
        if example.document_type == DocumentType.UNKNOWN:
            raise ValueError("Cannot store UNKNOWN as manual classification")
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(example.to_dict()) + "\n")

    def load_all(self) -> list[ManualExample]:
        """Load all stored examples.

        Returns:
            List of ManualExample ordered by file position.
            Corrupt lines are skipped with a warning.
        """
        if not self._path.exists():
            return []
        examples: list[ManualExample] = []
        with open(self._path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    examples.append(ManualExample.from_dict(data))
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    logger.warning(
                        "Skipping corrupt line %d in %s: %s",
                        line_num,
                        self._path,
                        exc,
                    )
        return examples

    def count(self) -> int:
        """Return the number of stored examples without loading all data."""
        if not self._path.exists():
            return 0
        count = 0
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def count_by_type(self) -> dict[str, int]:
        """Return counts per DocumentType value string."""
        counts: dict[str, int] = {}
        for example in self.load_all():
            key = example.document_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def clear(self) -> None:
        """Delete all stored examples. Irreversible."""
        if self._path.exists():
            self._path.unlink()
