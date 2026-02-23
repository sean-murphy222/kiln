"""Metadata extractor for chunk enrichment.

Orchestrates the extraction, validation, and scoring of metadata
fields from document chunks using type-specific profiles.
"""

from __future__ import annotations

import re

from chonk.core.document import Chunk
from chonk.enrichment.profiles import MetadataProfileRegistry
from chonk.enrichment.result import ChunkEnrichmentRecord, EnrichmentResult
from chonk.enrichment.rules import ExtractionRule, ExtractionSource, FieldExtraction
from chonk.enrichment.validators import FieldValidator
from chonk.tier1.taxonomy import DocumentType

# Metadata keys stamped onto enriched chunks
ENRICHMENT_APPLIED_KEY = "enrichment_applied"
ENRICHMENT_FIELDS_KEY = "enrichment_fields"
ENRICHMENT_QUALITY_KEY = "enrichment_quality"

# Quality score weights
_WEIGHT_REQUIRED_COVERAGE = 0.5
_WEIGHT_VALIDATION_RATE = 0.3
_WEIGHT_AVG_CONFIDENCE = 0.2


class MetadataExtractor:
    """Extracts and validates metadata from chunks.

    Uses type-specific profiles to apply extraction rules, validates
    extracted values, computes quality scores, and stamps metadata
    onto chunks.

    Args:
        registry: Profile registry to use. Defaults to built-in.
        validator: Field validator to use. Defaults to built-in.

    Example::

        extractor = MetadataExtractor()
        result = extractor.enrich(
            chunks, DocumentType.TECHNICAL_MANUAL, "doc1"
        )
        print(f"{result.enriched_count} chunks enriched")
    """

    def __init__(
        self,
        registry: MetadataProfileRegistry | None = None,
        validator: FieldValidator | None = None,
    ) -> None:
        self._registry = registry or MetadataProfileRegistry()
        self._validator = validator or FieldValidator()

    def enrich(
        self,
        chunks: list[Chunk],
        document_type: DocumentType,
        document_id: str = "unknown",
    ) -> EnrichmentResult:
        """Enrich all chunks with metadata from the type profile.

        Modifies chunk system_metadata in-place.

        Args:
            chunks: Chunks to enrich.
            document_type: Document type for profile selection.
            document_id: Document ID for the result.

        Returns:
            EnrichmentResult with per-chunk records and quality.
        """
        profile = self._registry.get(document_type)
        records: list[ChunkEnrichmentRecord] = []
        enriched_count = 0
        field_hits: dict[str, int] = {}
        all_field_names = [r.field_name for r in profile.rules]

        for chunk in chunks:
            record = self._extract_from_chunk(chunk, profile)
            records.append(record)

            if record.extracted_fields:
                enriched_count += 1
                self._stamp_chunk(chunk, record)

            for fname in record.extracted_fields:
                field_hits[fname] = field_hits.get(fname, 0) + 1

        total = len(chunks)
        field_coverage = {
            fname: field_hits.get(fname, 0) / total if total > 0 else 0.0
            for fname in all_field_names
        }

        overall_quality = self._compute_overall_quality(records, profile)

        return EnrichmentResult(
            document_id=document_id,
            total_chunks=total,
            enriched_count=enriched_count,
            field_coverage=field_coverage,
            quality_score=overall_quality,
            records=records,
        )

    def _extract_from_chunk(
        self,
        chunk: Chunk,
        profile: MetadataProfileRegistry | object,
    ) -> ChunkEnrichmentRecord:
        """Apply all profile rules to a single chunk.

        Args:
            chunk: Chunk to extract from.
            profile: Metadata profile with rules.

        Returns:
            ChunkEnrichmentRecord with extractions and validations.
        """
        from chonk.enrichment.profiles import MetadataProfile

        if not isinstance(profile, MetadataProfile):
            return ChunkEnrichmentRecord(chunk_id=chunk.id)

        extracted: dict[str, FieldExtraction] = {}

        for rule in profile.rules:
            extraction = self._apply_rule(chunk, rule)
            if extraction is not None:
                extracted[extraction.field_name] = extraction

        validations = {}
        for fname, ext in extracted.items():
            result = self._validator.validate(fname, ext.value)
            validations[fname] = result

        record = ChunkEnrichmentRecord(
            chunk_id=chunk.id,
            extracted_fields=extracted,
            validation_results=validations,
        )
        record.quality_score = self._compute_chunk_quality(record, profile)
        return record

    def _apply_rule(self, chunk: Chunk, rule: ExtractionRule) -> FieldExtraction | None:
        """Apply a single extraction rule to a chunk.

        Args:
            chunk: Chunk to extract from.
            rule: Rule to apply.

        Returns:
            FieldExtraction if matched, None otherwise.
        """
        sources = self._resolve_sources(chunk, rule.source)

        for source_type, text in sources:
            match = rule.pattern.search(text)
            if match:
                value = match.group(1) if match.lastindex else match.group(0)
                confidence = self._estimate_confidence(match, text)
                return FieldExtraction(
                    field_name=rule.field_name,
                    value=value.strip(),
                    confidence=confidence,
                    source=source_type,
                    rule_description=rule.description,
                )

        return None

    @staticmethod
    def _resolve_sources(
        chunk: Chunk, source: ExtractionSource
    ) -> list[tuple[ExtractionSource, str]]:
        """Get text sources to search in priority order.

        Args:
            chunk: Chunk to extract text from.
            source: Source specification.

        Returns:
            List of (source_type, text) tuples to search.
        """
        sources: list[tuple[ExtractionSource, str]] = []
        if source in (ExtractionSource.CONTENT, ExtractionSource.BOTH):
            sources.append((ExtractionSource.CONTENT, chunk.content))
        if source in (
            ExtractionSource.HIERARCHY_PATH,
            ExtractionSource.BOTH,
        ):
            sources.append((ExtractionSource.HIERARCHY_PATH, chunk.hierarchy_path))
        return sources

    @staticmethod
    def _estimate_confidence(match: re.Match[str], text: str) -> float:
        """Estimate extraction confidence based on match quality.

        Longer matches relative to the text get lower confidence
        (more likely a false positive). Exact short patterns get
        higher confidence.

        Args:
            match: The regex match object.
            text: The full text that was searched.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if not text:
            return 0.5
        match_len = len(match.group(0))
        text_len = len(text)
        ratio = match_len / text_len if text_len > 0 else 1.0
        if ratio < 0.3:
            return 0.9
        if ratio < 0.6:
            return 0.7
        return 0.5

    @staticmethod
    def _compute_chunk_quality(
        record: ChunkEnrichmentRecord,
        profile: object,
    ) -> float:
        """Compute quality score for a single chunk enrichment.

        Formula: 0.5 * required_coverage + 0.3 * validation_rate
                 + 0.2 * avg_confidence

        Args:
            record: The chunk's enrichment record.
            profile: The metadata profile used.

        Returns:
            Quality score between 0.0 and 1.0.
        """
        from chonk.enrichment.profiles import MetadataProfile

        if not isinstance(profile, MetadataProfile):
            return 0.0

        required = [r.field_name for r in profile.rules if r.required]
        required_found = sum(1 for f in required if f in record.extracted_fields)
        required_coverage = required_found / len(required) if required else 1.0

        validations = record.validation_results
        valid_count = sum(1 for v in validations.values() if v.is_valid)
        validation_rate = valid_count / len(validations) if validations else 1.0

        extractions = record.extracted_fields
        avg_confidence = (
            sum(e.confidence for e in extractions.values()) / len(extractions)
            if extractions
            else 0.0
        )

        return round(
            _WEIGHT_REQUIRED_COVERAGE * required_coverage
            + _WEIGHT_VALIDATION_RATE * validation_rate
            + _WEIGHT_AVG_CONFIDENCE * avg_confidence,
            3,
        )

    @staticmethod
    def _compute_overall_quality(
        records: list[ChunkEnrichmentRecord],
        profile: object,
    ) -> float:
        """Compute average quality across all chunk records.

        Args:
            records: All chunk enrichment records.
            profile: The metadata profile used.

        Returns:
            Average quality score between 0.0 and 1.0.
        """
        if not records:
            return 0.0
        return round(sum(r.quality_score for r in records) / len(records), 3)

    @staticmethod
    def _stamp_chunk(chunk: Chunk, record: ChunkEnrichmentRecord) -> None:
        """Write enrichment metadata onto a chunk.

        Args:
            chunk: Chunk to stamp.
            record: Enrichment record with extracted fields.
        """
        chunk.system_metadata[ENRICHMENT_APPLIED_KEY] = True
        chunk.system_metadata[ENRICHMENT_FIELDS_KEY] = {
            fname: ext.value for fname, ext in record.extracted_fields.items()
        }
        chunk.system_metadata[ENRICHMENT_QUALITY_KEY] = record.quality_score
