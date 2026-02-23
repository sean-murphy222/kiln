"""Filter log data structures for QA pass audit trail.

Records every filtering decision so reviewers can inspect
what was filtered and why.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class FilterCategory(str, Enum):
    """Why a block was filtered."""

    TOC = "toc"
    INDEX = "index"
    DISTRIBUTION = "distribution_statement"
    BOILERPLATE = "boilerplate"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    REPETITION = "repetition"


@dataclass
class FilterLogEntry:
    """A single filtering decision.

    Attributes:
        block_id: ID of the filtered block.
        block_type: BlockType value as string.
        category: Why the block was filtered.
        reason: Human-readable explanation.
        rule_name: Which rule or pattern matched.
        page: Page number of the block.
        content_preview: First 80 chars of content.
        timestamp: When the decision was made.
    """

    block_id: str
    block_type: str
    category: FilterCategory
    reason: str
    rule_name: str
    page: int
    content_preview: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "block_id": self.block_id,
            "block_type": self.block_type,
            "category": self.category.value,
            "reason": self.reason,
            "rule_name": self.rule_name,
            "page": self.page,
            "content_preview": self.content_preview,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FilterLogEntry:
        """Deserialize from dictionary."""
        return cls(
            block_id=data["block_id"],
            block_type=data["block_type"],
            category=FilterCategory(data["category"]),
            reason=data["reason"],
            rule_name=data["rule_name"],
            page=data["page"],
            content_preview=data["content_preview"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class FilterLog:
    """Complete log of all filtering decisions for a document.

    Attributes:
        document_id: ID of the document that was filtered.
        document_type: Document type used for rule selection.
        entries: All filtering decisions.
        created_at: When the log was created.
    """

    document_id: str
    document_type: str = "unknown"
    entries: list[FilterLogEntry] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def filtered_count(self) -> int:
        """Total number of filtered blocks."""
        return len(self.entries)

    def by_category(self) -> dict[FilterCategory, list[FilterLogEntry]]:
        """Group entries by filter category.

        Returns:
            Dict mapping category to list of entries.
        """
        result: dict[FilterCategory, list[FilterLogEntry]] = {}
        for entry in self.entries:
            result.setdefault(entry.category, []).append(entry)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "document_id": self.document_id,
            "document_type": self.document_type,
            "entries": [e.to_dict() for e in self.entries],
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FilterLog:
        """Deserialize from dictionary."""
        return cls(
            document_id=data["document_id"],
            document_type=data.get("document_type", "unknown"),
            entries=[FilterLogEntry.from_dict(e) for e in data.get("entries", [])],
            created_at=datetime.fromisoformat(data["created_at"]),
        )

    def save(self, path: Path) -> None:
        """Save log to JSON file.

        Args:
            path: Destination file path.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> FilterLog:
        """Load log from JSON file.

        Args:
            path: Source file path.

        Returns:
            Deserialized FilterLog.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
