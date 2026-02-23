"""Result dataclasses for metadata enrichment.

Holds per-chunk and per-document enrichment results with
quality scoring and field coverage tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chonk.enrichment.rules import FieldExtraction
from chonk.enrichment.validators import FieldValidationResult


@dataclass
class ChunkEnrichmentRecord:
    """Enrichment result for a single chunk.

    Args:
        chunk_id: ID of the enriched chunk.
        extracted_fields: Mapping of field_name to FieldExtraction.
        validation_results: Mapping of field_name to FieldValidationResult.
        quality_score: Enrichment quality score (0.0-1.0).
    """

    chunk_id: str
    extracted_fields: dict[str, FieldExtraction] = field(default_factory=dict)
    validation_results: dict[str, FieldValidationResult] = field(default_factory=dict)
    quality_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "extracted_fields": {
                name: {
                    "value": ext.value,
                    "confidence": ext.confidence,
                    "source": ext.source.value,
                }
                for name, ext in self.extracted_fields.items()
            },
            "validation_results": {
                name: {
                    "is_valid": res.is_valid,
                    "error_message": res.error_message,
                }
                for name, res in self.validation_results.items()
            },
            "quality_score": self.quality_score,
        }


@dataclass
class EnrichmentResult:
    """Result of enriching all chunks in a document.

    Args:
        document_id: ID of the enriched document.
        total_chunks: Total chunks processed.
        enriched_count: Chunks where at least one field was extracted.
        field_coverage: Per-field extraction rate across all chunks.
        quality_score: Overall enrichment quality (0.0-1.0).
        records: Per-chunk enrichment records.
    """

    document_id: str
    total_chunks: int
    enriched_count: int
    field_coverage: dict[str, float] = field(default_factory=dict)
    quality_score: float = 0.0
    records: list[ChunkEnrichmentRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "document_id": self.document_id,
            "total_chunks": self.total_chunks,
            "enriched_count": self.enriched_count,
            "field_coverage": self.field_coverage,
            "quality_score": self.quality_score,
            "records": [r.to_dict() for r in self.records],
        }
