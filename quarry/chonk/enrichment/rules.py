"""Extraction rules for metadata enrichment.

Defines regex-based rules that extract structured metadata fields
from chunk content and hierarchy paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ExtractionSource(str, Enum):
    """Where to search for metadata matches.

    Attributes:
        CONTENT: Search in chunk text content.
        HIERARCHY_PATH: Search in hierarchy path string.
        BOTH: Search content first, fall back to hierarchy path.
    """

    CONTENT = "content"
    HIERARCHY_PATH = "hierarchy_path"
    BOTH = "both"


@dataclass
class ExtractionRule:
    """A regex-based rule for extracting a metadata field.

    Args:
        field_name: Name of the metadata field to extract.
        pattern: Compiled regex pattern. First capture group is the value.
        source: Where to search for the pattern.
        required: Whether this field is required for the profile.
        description: Human-readable description of what this rule extracts.
    """

    field_name: str
    pattern: re.Pattern[str]
    source: ExtractionSource = ExtractionSource.CONTENT
    required: bool = False
    description: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize to dictionary."""
        return {
            "field_name": self.field_name,
            "pattern": self.pattern.pattern,
            "source": self.source.value,
            "required": self.required,
            "description": self.description,
        }


@dataclass
class FieldExtraction:
    """Result of applying a single extraction rule.

    Args:
        field_name: Name of the extracted field.
        value: Extracted string value.
        confidence: Confidence score (0.0-1.0) based on match quality.
        source: Which source the value was extracted from.
        rule_description: Description of the rule that produced this.
    """

    field_name: str
    value: str
    confidence: float
    source: ExtractionSource
    rule_description: str = ""
