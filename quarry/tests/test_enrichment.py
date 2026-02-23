"""Tests for metadata enrichment pipeline."""

from __future__ import annotations

import re

from chonk.core.document import Chunk
from chonk.enrichment.extractor import (
    ENRICHMENT_APPLIED_KEY,
    ENRICHMENT_FIELDS_KEY,
    ENRICHMENT_QUALITY_KEY,
    MetadataExtractor,
)
from chonk.enrichment.profiles import (
    MetadataProfile,
    MetadataProfileRegistry,
)
from chonk.enrichment.result import ChunkEnrichmentRecord, EnrichmentResult
from chonk.enrichment.rules import ExtractionRule, ExtractionSource, FieldExtraction
from chonk.enrichment.validators import FieldValidationResult, FieldValidator
from chonk.tier1.taxonomy import DocumentType

# --- Helpers ---


def _chunk(
    content: str,
    hierarchy_path: str = "",
    chunk_id: str | None = None,
) -> Chunk:
    """Create a test chunk with given content."""
    return Chunk(
        id=chunk_id or Chunk.generate_id(),
        block_ids=["block_1"],
        content=content,
        token_count=len(content.split()),
        hierarchy_path=hierarchy_path,
    )


# --- ExtractionRule Tests ---


class TestExtractionRule:
    """Tests for ExtractionRule dataclass."""

    def test_construction(self) -> None:
        """Test basic construction."""
        rule = ExtractionRule(
            field_name="tm_number",
            pattern=re.compile(r"(TM\s+\d+-\d+)"),
            source=ExtractionSource.CONTENT,
            required=True,
            description="TM number",
        )
        assert rule.field_name == "tm_number"
        assert rule.required is True
        assert rule.source == ExtractionSource.CONTENT

    def test_to_dict(self) -> None:
        """Test serialization."""
        rule = ExtractionRule(
            field_name="nsn",
            pattern=re.compile(r"(\d{4}-\d{2}-\d{3}-\d{4})"),
        )
        d = rule.to_dict()
        assert d["field_name"] == "nsn"
        assert "pattern" in d
        assert d["source"] == "content"

    def test_defaults(self) -> None:
        """Test default values."""
        rule = ExtractionRule(
            field_name="test",
            pattern=re.compile(r"(.+)"),
        )
        assert rule.source == ExtractionSource.CONTENT
        assert rule.required is False
        assert rule.description == ""


class TestExtractionSource:
    """Tests for ExtractionSource enum."""

    def test_values(self) -> None:
        """Test all enum values exist."""
        assert ExtractionSource.CONTENT.value == "content"
        assert ExtractionSource.HIERARCHY_PATH.value == "hierarchy_path"
        assert ExtractionSource.BOTH.value == "both"


class TestFieldExtraction:
    """Tests for FieldExtraction dataclass."""

    def test_construction(self) -> None:
        """Test basic construction."""
        ext = FieldExtraction(
            field_name="tm_number",
            value="TM 9-2320-272-20",
            confidence=0.9,
            source=ExtractionSource.CONTENT,
        )
        assert ext.field_name == "tm_number"
        assert ext.value == "TM 9-2320-272-20"
        assert ext.confidence == 0.9


# --- FieldValidator Tests ---


class TestFieldValidator:
    """Tests for FieldValidator."""

    def test_valid_nsn(self) -> None:
        """Test valid NSN passes validation."""
        v = FieldValidator()
        result = v.validate("nsn", "2320-01-107-7155")
        assert result.is_valid is True

    def test_invalid_nsn(self) -> None:
        """Test invalid NSN fails validation."""
        v = FieldValidator()
        result = v.validate("nsn", "not-a-nsn")
        assert result.is_valid is False
        assert result.error_message != ""

    def test_valid_tm_number(self) -> None:
        """Test valid TM number passes."""
        v = FieldValidator()
        result = v.validate("tm_number", "TM 9-2320-272-20")
        assert result.is_valid is True

    def test_invalid_tm_number(self) -> None:
        """Test invalid TM number fails."""
        v = FieldValidator()
        result = v.validate("tm_number", "XX 123")
        assert result.is_valid is False

    def test_valid_lin(self) -> None:
        """Test valid LIN passes."""
        v = FieldValidator()
        result = v.validate("lin", "T51687")
        assert result.is_valid is True

    def test_invalid_lin(self) -> None:
        """Test invalid LIN fails."""
        v = FieldValidator()
        result = v.validate("lin", "12345")
        assert result.is_valid is False

    def test_valid_smr_code(self) -> None:
        """Test valid SMR code passes."""
        v = FieldValidator()
        result = v.validate("smr_code", "PAOAF")
        assert result.is_valid is True

    def test_valid_maintenance_level(self) -> None:
        """Test valid maintenance level passes."""
        v = FieldValidator()
        result = v.validate("maintenance_level", "organizational")
        assert result.is_valid is True

    def test_unknown_field_passes(self) -> None:
        """Test fields without validators always pass."""
        v = FieldValidator()
        result = v.validate("unknown_field", "any value")
        assert result.is_valid is True

    def test_register_custom(self) -> None:
        """Test registering a custom validator."""
        v = FieldValidator()
        v.register("custom_id", re.compile(r"^CID-\d{4}$"))
        assert v.validate("custom_id", "CID-1234").is_valid is True
        assert v.validate("custom_id", "bad").is_valid is False

    def test_registered_fields(self) -> None:
        """Test registered fields listing."""
        v = FieldValidator()
        fields = v.registered_fields
        assert "nsn" in fields
        assert "tm_number" in fields
        assert "lin" in fields

    def test_valid_work_package(self) -> None:
        """Test valid work package passes."""
        v = FieldValidator()
        result = v.validate("work_package", "WP 0001 00")
        assert result.is_valid is True


# --- FieldValidationResult Tests ---


class TestFieldValidationResult:
    """Tests for FieldValidationResult dataclass."""

    def test_valid_result(self) -> None:
        """Test valid result construction."""
        r = FieldValidationResult(field_name="nsn", is_valid=True)
        assert r.is_valid is True
        assert r.error_message == ""

    def test_invalid_result(self) -> None:
        """Test invalid result with error message."""
        r = FieldValidationResult(
            field_name="nsn",
            is_valid=False,
            error_message="bad format",
        )
        assert r.is_valid is False
        assert "bad format" in r.error_message


# --- MetadataProfile Tests ---


class TestMetadataProfile:
    """Tests for MetadataProfile dataclass."""

    def test_construction(self) -> None:
        """Test basic construction."""
        profile = MetadataProfile(
            name="test",
            document_type=DocumentType.TECHNICAL_MANUAL,
        )
        assert profile.name == "test"
        assert profile.document_type == DocumentType.TECHNICAL_MANUAL

    def test_required_fields(self) -> None:
        """Test required fields property."""
        profile = MetadataProfile(
            name="test",
            document_type=DocumentType.TECHNICAL_MANUAL,
            rules=[
                ExtractionRule(
                    field_name="required_field",
                    pattern=re.compile(r"(.+)"),
                    required=True,
                ),
                ExtractionRule(
                    field_name="optional_field",
                    pattern=re.compile(r"(.+)"),
                    required=False,
                ),
            ],
        )
        assert "required_field" in profile.required_fields
        assert "optional_field" not in profile.required_fields

    def test_optional_fields(self) -> None:
        """Test optional fields property."""
        profile = MetadataProfile(
            name="test",
            document_type=DocumentType.TECHNICAL_MANUAL,
            rules=[
                ExtractionRule(
                    field_name="required_field",
                    pattern=re.compile(r"(.+)"),
                    required=True,
                ),
                ExtractionRule(
                    field_name="optional_field",
                    pattern=re.compile(r"(.+)"),
                    required=False,
                ),
            ],
        )
        assert "optional_field" in profile.optional_fields
        assert "required_field" not in profile.optional_fields

    def test_to_dict(self) -> None:
        """Test serialization."""
        profile = MetadataProfile(
            name="test",
            document_type=DocumentType.TECHNICAL_MANUAL,
            description="Test profile",
        )
        d = profile.to_dict()
        assert d["name"] == "test"
        assert d["document_type"] == "technical_manual"


# --- MetadataProfileRegistry Tests ---


class TestMetadataProfileRegistry:
    """Tests for MetadataProfileRegistry."""

    def test_builtin_profiles_loaded(self) -> None:
        """Test built-in profiles are loaded on construction."""
        registry = MetadataProfileRegistry()
        types = registry.registered_types
        assert DocumentType.TECHNICAL_MANUAL in types
        assert DocumentType.MAINTENANCE_PROCEDURE in types
        assert DocumentType.PARTS_CATALOG in types
        assert DocumentType.REGULATION in types

    def test_get_known_type(self) -> None:
        """Test getting a registered profile."""
        registry = MetadataProfileRegistry()
        profile = registry.get(DocumentType.TECHNICAL_MANUAL)
        assert profile.name == "technical_manual"
        assert len(profile.rules) > 0

    def test_get_unknown_type_returns_default(self) -> None:
        """Test unknown type falls back to default profile."""
        registry = MetadataProfileRegistry()
        profile = registry.get(DocumentType.ACADEMIC_PAPER)
        assert profile.name == "default"

    def test_register_custom_profile(self) -> None:
        """Test registering a custom profile."""
        registry = MetadataProfileRegistry()
        custom = MetadataProfile(
            name="custom_spec",
            document_type=DocumentType.SPECIFICATION,
            rules=[
                ExtractionRule(
                    field_name="spec_number",
                    pattern=re.compile(r"(SPEC-\d+)"),
                    required=True,
                ),
            ],
        )
        registry.register(custom)
        retrieved = registry.get(DocumentType.SPECIFICATION)
        assert retrieved.name == "custom_spec"

    def test_technical_manual_has_tm_number_rule(self) -> None:
        """Test TM profile requires tm_number."""
        registry = MetadataProfileRegistry()
        profile = registry.get(DocumentType.TECHNICAL_MANUAL)
        assert "tm_number" in profile.required_fields

    def test_parts_catalog_has_nsn_rule(self) -> None:
        """Test parts catalog profile requires NSN."""
        registry = MetadataProfileRegistry()
        profile = registry.get(DocumentType.PARTS_CATALOG)
        assert "nsn" in profile.required_fields


# --- ChunkEnrichmentRecord Tests ---


class TestChunkEnrichmentRecord:
    """Tests for ChunkEnrichmentRecord dataclass."""

    def test_construction(self) -> None:
        """Test basic construction."""
        record = ChunkEnrichmentRecord(chunk_id="chunk_1")
        assert record.chunk_id == "chunk_1"
        assert record.extracted_fields == {}
        assert record.quality_score == 0.0

    def test_to_dict(self) -> None:
        """Test serialization."""
        record = ChunkEnrichmentRecord(
            chunk_id="chunk_1",
            extracted_fields={
                "tm_number": FieldExtraction(
                    field_name="tm_number",
                    value="TM 9-2320",
                    confidence=0.9,
                    source=ExtractionSource.CONTENT,
                ),
            },
            quality_score=0.8,
        )
        d = record.to_dict()
        assert d["chunk_id"] == "chunk_1"
        assert d["extracted_fields"]["tm_number"]["value"] == "TM 9-2320"
        assert d["quality_score"] == 0.8


# --- EnrichmentResult Tests ---


class TestEnrichmentResult:
    """Tests for EnrichmentResult dataclass."""

    def test_construction(self) -> None:
        """Test basic construction."""
        result = EnrichmentResult(
            document_id="doc1",
            total_chunks=10,
            enriched_count=5,
        )
        assert result.total_chunks == 10
        assert result.enriched_count == 5

    def test_to_dict(self) -> None:
        """Test serialization."""
        result = EnrichmentResult(
            document_id="doc1",
            total_chunks=5,
            enriched_count=3,
            field_coverage={"tm_number": 0.6},
            quality_score=0.75,
        )
        d = result.to_dict()
        assert d["document_id"] == "doc1"
        assert d["field_coverage"]["tm_number"] == 0.6


# --- MetadataExtractor Tests ---


class TestExtractorBasic:
    """Tests for basic MetadataExtractor functionality."""

    def test_construction(self) -> None:
        """Test default construction."""
        extractor = MetadataExtractor()
        assert extractor is not None

    def test_enrich_empty_chunks(self) -> None:
        """Test enriching empty chunk list."""
        extractor = MetadataExtractor()
        result = extractor.enrich([], DocumentType.TECHNICAL_MANUAL, "doc1")
        assert result.total_chunks == 0
        assert result.enriched_count == 0

    def test_enrich_no_matches(self) -> None:
        """Test enriching chunks with no matching content."""
        extractor = MetadataExtractor()
        chunks = [_chunk("This is plain text with no metadata.")]
        result = extractor.enrich(chunks, DocumentType.TECHNICAL_MANUAL, "doc1")
        assert result.total_chunks == 1
        assert result.enriched_count == 0


class TestExtractorTMExtraction:
    """Tests for TM number extraction."""

    def test_extract_tm_number_from_content(self) -> None:
        """Test TM number extracted from content."""
        extractor = MetadataExtractor()
        chunks = [_chunk("Refer to TM 9-2320-272-20 for maintenance procedures.")]
        result = extractor.enrich(chunks, DocumentType.TECHNICAL_MANUAL, "doc1")
        assert result.enriched_count == 1
        record = result.records[0]
        assert "tm_number" in record.extracted_fields
        assert "TM 9-2320-272-20" in record.extracted_fields["tm_number"].value

    def test_extract_tm_number_from_hierarchy(self) -> None:
        """Test TM number extracted from hierarchy path."""
        extractor = MetadataExtractor()
        chunks = [
            _chunk(
                "Some procedure text.",
                hierarchy_path="TM 9-2320-272-20 > Chapter 3 > Safety",
            )
        ]
        result = extractor.enrich(chunks, DocumentType.TECHNICAL_MANUAL, "doc1")
        record = result.records[0]
        assert "tm_number" in record.extracted_fields

    def test_extract_tm_number_case_insensitive(self) -> None:
        """Test TM number extraction is case insensitive."""
        extractor = MetadataExtractor()
        chunks = [_chunk("Per tm 1-1500-204-23, inspect rotor.")]
        result = extractor.enrich(chunks, DocumentType.TECHNICAL_MANUAL, "doc1")
        record = result.records[0]
        assert "tm_number" in record.extracted_fields


class TestExtractorNSNExtraction:
    """Tests for NSN extraction."""

    def test_extract_nsn(self) -> None:
        """Test NSN extracted from content."""
        extractor = MetadataExtractor()
        chunks = [_chunk("Replace filter (NSN 2940-01-234-5678) and inspect.")]
        result = extractor.enrich(chunks, DocumentType.TECHNICAL_MANUAL, "doc1")
        record = result.records[0]
        assert "nsn" in record.extracted_fields
        assert record.extracted_fields["nsn"].value == "2940-01-234-5678"

    def test_nsn_validated(self) -> None:
        """Test extracted NSN is validated."""
        extractor = MetadataExtractor()
        chunks = [_chunk("NSN 2940-01-234-5678 required.")]
        result = extractor.enrich(chunks, DocumentType.TECHNICAL_MANUAL, "doc1")
        record = result.records[0]
        assert "nsn" in record.validation_results
        assert record.validation_results["nsn"].is_valid is True


class TestExtractorMilitaryFields:
    """Tests for military-specific field extraction."""

    def test_extract_maintenance_level(self) -> None:
        """Test maintenance level extraction."""
        extractor = MetadataExtractor()
        chunks = [_chunk("This is an organizational maintenance procedure.")]
        result = extractor.enrich(chunks, DocumentType.TECHNICAL_MANUAL, "doc1")
        record = result.records[0]
        assert "maintenance_level" in record.extracted_fields
        assert "organizational" in record.extracted_fields["maintenance_level"].value.lower()

    def test_extract_work_package(self) -> None:
        """Test work package extraction."""
        extractor = MetadataExtractor()
        chunks = [_chunk("See WP 0023 00 for removal procedure.")]
        result = extractor.enrich(chunks, DocumentType.TECHNICAL_MANUAL, "doc1")
        record = result.records[0]
        assert "work_package" in record.extracted_fields
        assert "WP 0023 00" in record.extracted_fields["work_package"].value

    def test_extract_lin(self) -> None:
        """Test LIN extraction."""
        extractor = MetadataExtractor()
        chunks = [_chunk("Equipment LIN: T51687 assigned to unit.")]
        result = extractor.enrich(chunks, DocumentType.TECHNICAL_MANUAL, "doc1")
        record = result.records[0]
        assert "lin" in record.extracted_fields
        assert record.extracted_fields["lin"].value == "T51687"

    def test_extract_smr_code(self) -> None:
        """Test SMR code extraction."""
        extractor = MetadataExtractor()
        chunks = [_chunk("SMR CODE: PAOAF for this component.")]
        result = extractor.enrich(chunks, DocumentType.TECHNICAL_MANUAL, "doc1")
        record = result.records[0]
        assert "smr_code" in record.extracted_fields
        assert record.extracted_fields["smr_code"].value == "PAOAF"

    def test_extract_section_number(self) -> None:
        """Test section number extraction."""
        extractor = MetadataExtractor()
        chunks = [
            _chunk(
                "Procedure details here.",
                hierarchy_path="Section 3.2 > Safety",
            )
        ]
        result = extractor.enrich(chunks, DocumentType.TECHNICAL_MANUAL, "doc1")
        record = result.records[0]
        assert "section_number" in record.extracted_fields

    def test_extract_figure_ref(self) -> None:
        """Test figure reference extraction."""
        extractor = MetadataExtractor()
        chunks = [_chunk("See Figure 3-12 for exploded view.")]
        result = extractor.enrich(chunks, DocumentType.TECHNICAL_MANUAL, "doc1")
        record = result.records[0]
        assert "figure_ref" in record.extracted_fields

    def test_extract_table_ref(self) -> None:
        """Test table reference extraction."""
        extractor = MetadataExtractor()
        chunks = [_chunk("Torque values in Table 2-5.")]
        result = extractor.enrich(chunks, DocumentType.TECHNICAL_MANUAL, "doc1")
        record = result.records[0]
        assert "table_ref" in record.extracted_fields


class TestExtractorPartsCatalog:
    """Tests for parts catalog profile extraction."""

    def test_parts_catalog_nsn_required(self) -> None:
        """Test NSN is required in parts catalog profile."""
        extractor = MetadataExtractor()
        chunks = [
            _chunk(
                "Filter assembly, NSN 2940-01-234-5678, P/N: AB-12345, "
                "CAGE: 12345, per TM 9-2320-272-20."
            )
        ]
        result = extractor.enrich(chunks, DocumentType.PARTS_CATALOG, "doc1")
        record = result.records[0]
        assert "nsn" in record.extracted_fields
        assert "part_number" in record.extracted_fields
        assert "cage_code" in record.extracted_fields

    def test_parts_catalog_part_number(self) -> None:
        """Test part number extraction."""
        extractor = MetadataExtractor()
        chunks = [_chunk("Item 5, P/N: XY-98765-A, qty 2.")]
        result = extractor.enrich(chunks, DocumentType.PARTS_CATALOG, "doc1")
        record = result.records[0]
        assert "part_number" in record.extracted_fields
        assert record.extracted_fields["part_number"].value == "XY-98765-A"


class TestExtractorRegulation:
    """Tests for regulation profile extraction."""

    def test_regulation_number(self) -> None:
        """Test regulation number extraction."""
        extractor = MetadataExtractor()
        chunks = [_chunk("IAW AR 750-1, all units shall comply.")]
        result = extractor.enrich(chunks, DocumentType.REGULATION, "doc1")
        record = result.records[0]
        assert "regulation_number" in record.extracted_fields
        assert "AR 750-1" in record.extracted_fields["regulation_number"].value


class TestExtractorDefaultProfile:
    """Tests for default profile fallback."""

    def test_unknown_type_uses_default(self) -> None:
        """Test unknown document type uses default profile."""
        extractor = MetadataExtractor()
        chunks = [_chunk("See Section 5.2 for details.")]
        result = extractor.enrich(chunks, DocumentType.UNKNOWN, "doc1")
        record = result.records[0]
        assert "section_number" in record.extracted_fields

    def test_unregistered_type_uses_default(self) -> None:
        """Test unregistered type uses default profile."""
        extractor = MetadataExtractor()
        chunks = [_chunk("Figure 1-2 shows the layout.")]
        result = extractor.enrich(chunks, DocumentType.PRESENTATION, "doc1")
        record = result.records[0]
        assert "figure_ref" in record.extracted_fields


# --- Quality Scoring Tests ---


class TestQualityScoring:
    """Tests for enrichment quality scoring."""

    def test_quality_score_range(self) -> None:
        """Test quality score is between 0 and 1."""
        extractor = MetadataExtractor()
        chunks = [_chunk("TM 9-2320-272-20 organizational maintenance procedure.")]
        result = extractor.enrich(chunks, DocumentType.TECHNICAL_MANUAL, "doc1")
        for record in result.records:
            assert 0.0 <= record.quality_score <= 1.0
        assert 0.0 <= result.quality_score <= 1.0

    def test_higher_quality_with_required_fields(self) -> None:
        """Test chunks with required fields score higher."""
        extractor = MetadataExtractor()
        chunk_with = _chunk(
            "TM 9-2320-272-20 organizational maintenance.",
            chunk_id="with",
        )
        chunk_without = _chunk(
            "Generic text about engines.",
            chunk_id="without",
        )
        result = extractor.enrich(
            [chunk_with, chunk_without],
            DocumentType.TECHNICAL_MANUAL,
            "doc1",
        )
        record_with = next(r for r in result.records if r.chunk_id == "with")
        record_without = next(r for r in result.records if r.chunk_id == "without")
        assert record_with.quality_score > record_without.quality_score

    def test_empty_chunks_zero_quality(self) -> None:
        """Test no chunks produces zero quality."""
        extractor = MetadataExtractor()
        result = extractor.enrich([], DocumentType.TECHNICAL_MANUAL, "doc1")
        assert result.quality_score == 0.0


# --- Field Coverage Tests ---


class TestFieldCoverage:
    """Tests for field coverage tracking."""

    def test_coverage_calculated(self) -> None:
        """Test field coverage is calculated per field."""
        extractor = MetadataExtractor()
        chunks = [
            _chunk("TM 9-2320-272-20 procedure."),
            _chunk("Plain text with no metadata."),
        ]
        result = extractor.enrich(chunks, DocumentType.TECHNICAL_MANUAL, "doc1")
        assert "tm_number" in result.field_coverage
        assert result.field_coverage["tm_number"] == 0.5

    def test_all_profile_fields_in_coverage(self) -> None:
        """Test all profile fields appear in coverage dict."""
        extractor = MetadataExtractor()
        registry = MetadataProfileRegistry()
        profile = registry.get(DocumentType.TECHNICAL_MANUAL)
        chunks = [_chunk("No matches here.")]
        result = extractor.enrich(chunks, DocumentType.TECHNICAL_MANUAL, "doc1")
        for rule in profile.rules:
            assert rule.field_name in result.field_coverage


# --- Metadata Stamping Tests ---


class TestMetadataStamping:
    """Tests for chunk metadata stamping."""

    def test_enriched_chunk_stamped(self) -> None:
        """Test enriched chunks get metadata stamps."""
        extractor = MetadataExtractor()
        chunk = _chunk("TM 9-2320-272-20 maintenance.")
        extractor.enrich([chunk], DocumentType.TECHNICAL_MANUAL, "doc1")
        assert chunk.system_metadata.get(ENRICHMENT_APPLIED_KEY) is True
        assert ENRICHMENT_FIELDS_KEY in chunk.system_metadata
        assert ENRICHMENT_QUALITY_KEY in chunk.system_metadata

    def test_unenriched_chunk_not_stamped(self) -> None:
        """Test chunks with no extractions are not stamped."""
        extractor = MetadataExtractor()
        chunk = _chunk("Nothing to extract here.")
        extractor.enrich([chunk], DocumentType.TECHNICAL_MANUAL, "doc1")
        assert ENRICHMENT_APPLIED_KEY not in chunk.system_metadata

    def test_stamped_fields_are_values(self) -> None:
        """Test stamped fields contain extracted values."""
        extractor = MetadataExtractor()
        chunk = _chunk("NSN 2940-01-234-5678 per TM 9-2320-272-20.")
        extractor.enrich([chunk], DocumentType.TECHNICAL_MANUAL, "doc1")
        fields = chunk.system_metadata[ENRICHMENT_FIELDS_KEY]
        assert "tm_number" in fields
        assert "nsn" in fields
        assert "2940-01-234-5678" in fields["nsn"]


# --- Confidence Scoring Tests ---


class TestConfidenceScoring:
    """Tests for extraction confidence scoring."""

    def test_confidence_range(self) -> None:
        """Test confidence is between 0 and 1."""
        extractor = MetadataExtractor()
        chunks = [
            _chunk(
                "This is a long paragraph about maintenance. "
                "The TM 9-2320-272-20 covers the full procedure. "
                "Additional details follow for the complete system."
            )
        ]
        result = extractor.enrich(chunks, DocumentType.TECHNICAL_MANUAL, "doc1")
        for record in result.records:
            for ext in record.extracted_fields.values():
                assert 0.0 <= ext.confidence <= 1.0

    def test_short_match_in_long_text_high_confidence(self) -> None:
        """Test short match in long text gets high confidence."""
        extractor = MetadataExtractor()
        long_text = "A" * 200 + " TM 9-2320-272-20 " + "B" * 200
        chunks = [_chunk(long_text)]
        result = extractor.enrich(chunks, DocumentType.TECHNICAL_MANUAL, "doc1")
        record = result.records[0]
        if "tm_number" in record.extracted_fields:
            assert record.extracted_fields["tm_number"].confidence >= 0.7


# --- Integration Tests ---


class TestEnrichmentIntegration:
    """Integration tests for the full enrichment pipeline."""

    def test_full_tm_enrichment(self) -> None:
        """Test full enrichment of a typical TM chunk."""
        extractor = MetadataExtractor()
        chunk = _chunk(
            "TM 9-2320-272-20\n"
            "Organizational Maintenance\n"
            "Engine Assembly, NSN 2815-01-234-5678\n"
            "See WP 0023 00 for removal.\n"
            "Refer to Figure 3-12 and Table 2-5.\n"
            "LIN: T51687\n"
            "SMR CODE: PAOAF",
            hierarchy_path="TM 9-2320-272-20 > Chapter 3 > Engine",
        )
        result = extractor.enrich([chunk], DocumentType.TECHNICAL_MANUAL, "doc1")
        assert result.enriched_count == 1
        record = result.records[0]
        assert "tm_number" in record.extracted_fields
        assert "nsn" in record.extracted_fields
        assert "work_package" in record.extracted_fields
        assert "figure_ref" in record.extracted_fields
        assert "table_ref" in record.extracted_fields
        assert "lin" in record.extracted_fields
        assert "smr_code" in record.extracted_fields
        assert record.quality_score > 0.5

    def test_mixed_chunks_enrichment(self) -> None:
        """Test enriching a mix of metadata-rich and plain chunks."""
        extractor = MetadataExtractor()
        chunks = [
            _chunk("TM 9-2320-272-20 maintenance.", chunk_id="c1"),
            _chunk("Plain paragraph about safety.", chunk_id="c2"),
            _chunk("NSN 2940-01-234-5678 filter.", chunk_id="c3"),
        ]
        result = extractor.enrich(chunks, DocumentType.TECHNICAL_MANUAL, "doc1")
        assert result.total_chunks == 3
        assert result.enriched_count == 2
        assert result.document_id == "doc1"

    def test_custom_registry(self) -> None:
        """Test using a custom registry."""
        registry = MetadataProfileRegistry()
        registry.register(
            MetadataProfile(
                name="custom",
                document_type=DocumentType.SPECIFICATION,
                rules=[
                    ExtractionRule(
                        field_name="spec_id",
                        pattern=re.compile(r"(SPEC-\d{4})"),
                        required=True,
                    ),
                ],
            )
        )
        extractor = MetadataExtractor(registry=registry)
        chunks = [_chunk("Per SPEC-1234 requirements.")]
        result = extractor.enrich(chunks, DocumentType.SPECIFICATION, "doc1")
        record = result.records[0]
        assert "spec_id" in record.extracted_fields
        assert record.extracted_fields["spec_id"].value == "SPEC-1234"

    def test_result_serialization(self) -> None:
        """Test full result can be serialized to dict."""
        extractor = MetadataExtractor()
        chunks = [_chunk("TM 9-2320-272-20 procedure.")]
        result = extractor.enrich(chunks, DocumentType.TECHNICAL_MANUAL, "doc1")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "records" in d
        assert len(d["records"]) == 1
