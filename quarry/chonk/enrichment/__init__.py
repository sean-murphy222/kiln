"""Metadata enrichment for document chunks.

Extracts structured metadata from chunk content using regex-based
rules organized into document-type profiles. Validates extracted
values and computes enrichment quality scores.
"""

from chonk.enrichment.extractor import MetadataExtractor
from chonk.enrichment.profiles import (
    MetadataProfile,
    MetadataProfileRegistry,
)
from chonk.enrichment.result import ChunkEnrichmentRecord, EnrichmentResult
from chonk.enrichment.rules import ExtractionRule, ExtractionSource, FieldExtraction
from chonk.enrichment.validators import FieldValidationResult, FieldValidator

__all__ = [
    "ChunkEnrichmentRecord",
    "EnrichmentResult",
    "ExtractionRule",
    "ExtractionSource",
    "FieldExtraction",
    "FieldValidationResult",
    "FieldValidator",
    "MetadataExtractor",
    "MetadataProfile",
    "MetadataProfileRegistry",
]
