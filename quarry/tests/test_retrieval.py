"""Tests for 3-stage metadata-filtered retrieval pipeline."""

from __future__ import annotations

from chonk.core.document import Chunk
from chonk.enrichment.extractor import ENRICHMENT_FIELDS_KEY
from chonk.retrieval.filters import (
    FilterCriteria,
    FilterOperator,
    MetadataFilter,
)
from chonk.retrieval.pipeline import (
    RetrievalConfig,
    RetrievalPipeline,
    RetrievalResult,
    StageMetrics,
)
from chonk.retrieval.search import KeywordSearch, ScoredChunk
from chonk.retrieval.validation import (
    ResultValidator,
    ValidationCheck,
    ValidationRule,
)

# --- Helpers ---


def _chunk(
    content: str,
    chunk_id: str | None = None,
    hierarchy_path: str = "",
    enrichment_fields: dict | None = None,
    document_type: str | None = None,
) -> Chunk:
    """Create a test chunk with optional enrichment metadata."""
    c = Chunk(
        id=chunk_id or Chunk.generate_id(),
        block_ids=["block_1"],
        content=content,
        token_count=len(content.split()),
        hierarchy_path=hierarchy_path,
    )
    if enrichment_fields:
        c.system_metadata[ENRICHMENT_FIELDS_KEY] = enrichment_fields
    if document_type:
        c.system_metadata["document_type"] = document_type
    return c


def _make_corpus() -> list[Chunk]:
    """Create a realistic test corpus of enriched chunks."""
    return [
        _chunk(
            "Remove air filter element per TM 9-2320-272-20. "
            "Inspect for damage and replace as needed.",
            chunk_id="c1",
            hierarchy_path="TM 9-2320-272-20 > Chapter 3 > Air Filter",
            enrichment_fields={
                "tm_number": "TM 9-2320-272-20",
                "maintenance_level": "organizational",
                "section_number": "3.2",
                "figure_ref": "3-12",
            },
            document_type="technical_manual",
        ),
        _chunk(
            "Torque the bolts to 45 ft-lbs. Verify alignment "
            "with the engine block mounting surface.",
            chunk_id="c2",
            hierarchy_path="TM 9-2320-272-20 > Chapter 4 > Engine Mount",
            enrichment_fields={
                "tm_number": "TM 9-2320-272-20",
                "maintenance_level": "direct support",
                "section_number": "4.1",
            },
            document_type="technical_manual",
        ),
        _chunk(
            "NSN 2940-01-234-5678 Air Filter Assembly. " "Qty per vehicle: 1. Unit price: $23.50.",
            chunk_id="c3",
            hierarchy_path="Parts List > Section III > Filters",
            enrichment_fields={
                "nsn": "2940-01-234-5678",
                "part_number": "AF-12345",
                "tm_number": "TM 9-2320-272-24P",
            },
            document_type="parts_catalog",
        ),
        _chunk(
            "Safety requirements for all maintenance operations. " "PPE must be worn at all times.",
            chunk_id="c4",
            hierarchy_path="TM 9-2320-272-20 > Chapter 1 > Safety",
            enrichment_fields={
                "tm_number": "TM 9-2320-272-20",
                "section_number": "1.1",
            },
            document_type="technical_manual",
        ),
        _chunk(
            "AR 750-1 requires all units to maintain readiness " "levels above 90 percent.",
            chunk_id="c5",
            hierarchy_path="AR 750-1 > Chapter 3",
            enrichment_fields={
                "regulation_number": "AR 750-1",
            },
            document_type="regulation",
        ),
        _chunk(
            "Transmission fluid level check procedure. "
            "Remove dipstick and verify level is within range.",
            chunk_id="c6",
            hierarchy_path="TM 9-2320-272-20 > Chapter 5 > Transmission",
            enrichment_fields={
                "tm_number": "TM 9-2320-272-20",
                "maintenance_level": "organizational",
                "section_number": "5.3",
            },
            document_type="technical_manual",
        ),
        _chunk(
            "Distribution statement A: Approved for public release.",
            chunk_id="c7",
            document_type="technical_manual",
        ),
        _chunk(
            "Table of Contents",
            chunk_id="c8",
            document_type="technical_manual",
        ),
        _chunk(
            "Replace the air filter element. Ensure new filter "
            "is seated properly in the housing.",
            chunk_id="c9",
            hierarchy_path="Maintenance Procedure > Step 3",
            enrichment_fields={
                "tm_number": "TM 9-2320-272-20",
                "maintenance_level": "organizational",
                "work_package": "WP 0023 00",
            },
            document_type="maintenance_procedure",
        ),
        _chunk(
            "Engine oil specifications require MIL-PRF-2104 "
            "compliant lubricant for all operating conditions.",
            chunk_id="c10",
            hierarchy_path="TM 9-2320-272-20 > Chapter 6 > Lubrication",
            enrichment_fields={
                "tm_number": "TM 9-2320-272-20",
                "maintenance_level": "organizational",
                "section_number": "6.1",
            },
            document_type="technical_manual",
        ),
    ]


# --- FilterOperator Tests ---


class TestFilterOperator:
    """Tests for FilterOperator enum."""

    def test_all_operators(self) -> None:
        """Test all operators exist."""
        assert FilterOperator.EQ.value == "eq"
        assert FilterOperator.CONTAINS.value == "contains"
        assert FilterOperator.REGEX.value == "regex"
        assert FilterOperator.EXISTS.value == "exists"
        assert FilterOperator.NOT_EXISTS.value == "not_exists"
        assert FilterOperator.GT.value == "gt"
        assert FilterOperator.LT.value == "lt"


# --- FilterCriteria Tests ---


class TestFilterCriteria:
    """Tests for FilterCriteria."""

    def test_empty_criteria(self) -> None:
        """Test empty criteria has no conditions."""
        criteria = FilterCriteria()
        assert criteria.conditions == []
        assert criteria.document_type is None

    def test_add_chaining(self) -> None:
        """Test add returns self for chaining."""
        criteria = FilterCriteria()
        result = criteria.add("tm_number", FilterOperator.EQ, "TM 1-1")
        assert result is criteria
        assert len(criteria.conditions) == 1

    def test_multiple_conditions(self) -> None:
        """Test adding multiple conditions."""
        criteria = FilterCriteria()
        criteria.add("tm_number", FilterOperator.CONTAINS, "9-2320")
        criteria.add("maintenance_level", FilterOperator.EQ, "organizational")
        assert len(criteria.conditions) == 2


# --- MetadataFilter Stage 1 Tests ---


class TestMetadataFilter:
    """Tests for Stage 1 metadata pre-filter."""

    def test_no_criteria_passes_all(self) -> None:
        """Test empty criteria passes all chunks."""
        mf = MetadataFilter()
        chunks = _make_corpus()
        result = mf.filter(chunks, FilterCriteria())
        assert len(result.passed) == len(chunks)
        assert result.reduction_ratio == 0.0

    def test_eq_filter(self) -> None:
        """Test exact match filter."""
        mf = MetadataFilter()
        chunks = _make_corpus()
        criteria = FilterCriteria()
        criteria.add("maintenance_level", FilterOperator.EQ, "organizational")
        result = mf.filter(chunks, criteria)
        for c in result.passed:
            fields = c.system_metadata.get(ENRICHMENT_FIELDS_KEY, {})
            assert fields.get("maintenance_level", "").lower() == "organizational"

    def test_contains_filter(self) -> None:
        """Test substring match filter."""
        mf = MetadataFilter()
        chunks = _make_corpus()
        criteria = FilterCriteria()
        criteria.add("tm_number", FilterOperator.CONTAINS, "9-2320")
        result = mf.filter(chunks, criteria)
        assert len(result.passed) > 0
        for c in result.passed:
            fields = c.system_metadata.get(ENRICHMENT_FIELDS_KEY, {})
            assert "9-2320" in fields.get("tm_number", "")

    def test_exists_filter(self) -> None:
        """Test field existence filter."""
        mf = MetadataFilter()
        chunks = _make_corpus()
        criteria = FilterCriteria()
        criteria.add("nsn", FilterOperator.EXISTS)
        result = mf.filter(chunks, criteria)
        assert len(result.passed) == 1
        assert result.passed[0].id == "c3"

    def test_not_exists_filter(self) -> None:
        """Test field non-existence filter."""
        mf = MetadataFilter()
        chunks = _make_corpus()
        criteria = FilterCriteria()
        criteria.add("nsn", FilterOperator.NOT_EXISTS)
        result = mf.filter(chunks, criteria)
        assert len(result.passed) == len(chunks) - 1

    def test_regex_filter(self) -> None:
        """Test regex pattern filter."""
        mf = MetadataFilter()
        chunks = _make_corpus()
        criteria = FilterCriteria()
        criteria.add("section_number", FilterOperator.REGEX, r"^\d+\.\d+$")
        result = mf.filter(chunks, criteria)
        assert len(result.passed) > 0

    def test_document_type_filter(self) -> None:
        """Test document type filtering."""
        mf = MetadataFilter()
        chunks = _make_corpus()
        criteria = FilterCriteria(document_type="technical_manual")
        result = mf.filter(chunks, criteria)
        for c in result.passed:
            assert c.system_metadata.get("document_type") == "technical_manual"

    def test_combined_filters(self) -> None:
        """Test multiple conditions with AND logic."""
        mf = MetadataFilter()
        chunks = _make_corpus()
        criteria = FilterCriteria(document_type="technical_manual")
        criteria.add("maintenance_level", FilterOperator.EQ, "organizational")
        criteria.add("tm_number", FilterOperator.CONTAINS, "9-2320-272-20")
        result = mf.filter(chunks, criteria)
        # Should match c1, c6, c10
        assert len(result.passed) == 3

    def test_reduction_ratio(self) -> None:
        """Test reduction ratio calculation."""
        mf = MetadataFilter()
        chunks = _make_corpus()
        criteria = FilterCriteria(document_type="regulation")
        result = mf.filter(chunks, criteria)
        assert result.total_input == 10
        assert result.reduction_ratio > 0.5

    def test_empty_chunks(self) -> None:
        """Test filtering empty chunk list."""
        mf = MetadataFilter()
        criteria = FilterCriteria()
        criteria.add("tm_number", FilterOperator.EXISTS)
        result = mf.filter([], criteria)
        assert result.passed == []
        assert result.total_input == 0

    def test_case_insensitive_eq(self) -> None:
        """Test EQ filter is case insensitive."""
        mf = MetadataFilter()
        chunk = _chunk(
            "test",
            enrichment_fields={"maintenance_level": "Organizational"},
        )
        criteria = FilterCriteria()
        criteria.add("maintenance_level", FilterOperator.EQ, "organizational")
        result = mf.filter([chunk], criteria)
        assert len(result.passed) == 1


# --- KeywordSearch Stage 2 Tests ---


class TestKeywordSearch:
    """Tests for Stage 2 keyword search."""

    def test_empty_query(self) -> None:
        """Test empty query returns no results."""
        search = KeywordSearch()
        chunks = _make_corpus()
        results = search.search("", chunks)
        assert results == []

    def test_empty_chunks(self) -> None:
        """Test empty chunks returns no results."""
        search = KeywordSearch()
        results = search.search("air filter", [])
        assert results == []

    def test_relevant_results_ranked_first(self) -> None:
        """Test most relevant chunks appear first."""
        search = KeywordSearch()
        chunks = _make_corpus()
        results = search.search("replace air filter element", chunks)
        assert len(results) > 0
        # c1 and c9 mention "air filter" directly
        top_ids = [r.chunk.id for r in results[:3]]
        assert "c1" in top_ids or "c9" in top_ids

    def test_scores_normalized(self) -> None:
        """Test scores are between 0 and 1."""
        search = KeywordSearch()
        chunks = _make_corpus()
        results = search.search("maintenance procedure", chunks)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_top_result_score_is_one(self) -> None:
        """Test top result has score 1.0 (normalized)."""
        search = KeywordSearch()
        chunks = _make_corpus()
        results = search.search("air filter", chunks)
        if results:
            assert results[0].score == 1.0

    def test_ranks_are_sequential(self) -> None:
        """Test ranks start at 1 and are sequential."""
        search = KeywordSearch()
        chunks = _make_corpus()
        results = search.search("engine oil", chunks)
        for i, r in enumerate(results):
            assert r.rank == i + 1

    def test_top_k_limits_results(self) -> None:
        """Test top_k parameter limits results."""
        search = KeywordSearch()
        chunks = _make_corpus()
        results = search.search("maintenance", chunks, top_k=3)
        assert len(results) <= 3

    def test_no_matching_terms(self) -> None:
        """Test query with no matching terms returns empty."""
        search = KeywordSearch()
        chunks = _make_corpus()
        results = search.search("xyzzy foobar baz", chunks)
        assert results == []


# --- ResultValidator Stage 3 Tests ---


class TestResultValidator:
    """Tests for Stage 3 result validation."""

    def test_no_rules_passes_all(self) -> None:
        """Test no rules passes all results unchanged."""
        validator = ResultValidator()
        chunk = _chunk("test content", chunk_id="c1")
        scored = [ScoredChunk(chunk=chunk, score=0.8, rank=1)]
        results = validator.validate(scored)
        assert len(results) == 1
        assert results[0].validation_score == 1.0

    def test_has_field_passes(self) -> None:
        """Test HAS_FIELD check passes with field present."""
        validator = ResultValidator(
            rules=[
                ValidationRule(
                    check=ValidationCheck.HAS_FIELD,
                    value="tm_number",
                ),
            ]
        )
        chunk = _chunk(
            "test",
            enrichment_fields={"tm_number": "TM 1-1"},
        )
        scored = [ScoredChunk(chunk=chunk, score=0.8, rank=1)]
        results = validator.validate(scored)
        assert results[0].validation_score == 1.0

    def test_has_field_fails(self) -> None:
        """Test HAS_FIELD check fails with field missing."""
        validator = ResultValidator(
            rules=[
                ValidationRule(
                    check=ValidationCheck.HAS_FIELD,
                    value="nsn",
                ),
            ]
        )
        chunk = _chunk("test", enrichment_fields={"tm_number": "TM 1-1"})
        scored = [ScoredChunk(chunk=chunk, score=0.8, rank=1)]
        results = validator.validate(scored)
        assert results[0].validation_score == 0.0

    def test_content_matches_passes(self) -> None:
        """Test CONTENT_MATCHES check passes with match."""
        validator = ResultValidator(
            rules=[
                ValidationRule(
                    check=ValidationCheck.CONTENT_MATCHES,
                    value=r"step \d+",
                ),
            ]
        )
        chunk = _chunk("Follow step 3 to complete.")
        scored = [ScoredChunk(chunk=chunk, score=0.8, rank=1)]
        results = validator.validate(scored)
        assert results[0].validation_score == 1.0

    def test_min_length_passes(self) -> None:
        """Test MIN_LENGTH check passes with sufficient content."""
        validator = ResultValidator(
            rules=[
                ValidationRule(
                    check=ValidationCheck.MIN_LENGTH,
                    value=10,
                ),
            ]
        )
        chunk = _chunk("This is a sufficiently long chunk of text.")
        scored = [ScoredChunk(chunk=chunk, score=0.8, rank=1)]
        results = validator.validate(scored)
        assert results[0].validation_score == 1.0

    def test_min_length_fails(self) -> None:
        """Test MIN_LENGTH check fails with short content."""
        validator = ResultValidator(
            rules=[
                ValidationRule(
                    check=ValidationCheck.MIN_LENGTH,
                    value=100,
                ),
            ]
        )
        chunk = _chunk("Short.")
        scored = [ScoredChunk(chunk=chunk, score=0.8, rank=1)]
        results = validator.validate(scored)
        assert results[0].validation_score == 0.0

    def test_has_hierarchy_passes(self) -> None:
        """Test HAS_HIERARCHY check passes with path."""
        validator = ResultValidator(
            rules=[
                ValidationRule(
                    check=ValidationCheck.HAS_HIERARCHY,
                ),
            ]
        )
        chunk = _chunk("test", hierarchy_path="Chapter 1 > Section A")
        scored = [ScoredChunk(chunk=chunk, score=0.8, rank=1)]
        results = validator.validate(scored)
        assert results[0].validation_score == 1.0

    def test_has_hierarchy_fails(self) -> None:
        """Test HAS_HIERARCHY check fails without path."""
        validator = ResultValidator(
            rules=[
                ValidationRule(
                    check=ValidationCheck.HAS_HIERARCHY,
                ),
            ]
        )
        chunk = _chunk("test")
        scored = [ScoredChunk(chunk=chunk, score=0.8, rank=1)]
        results = validator.validate(scored)
        assert results[0].validation_score == 0.0

    def test_weighted_rules(self) -> None:
        """Test weighted validation scoring."""
        validator = ResultValidator(
            rules=[
                ValidationRule(
                    check=ValidationCheck.HAS_FIELD,
                    value="tm_number",
                    weight=2.0,
                    description="Has TM",
                ),
                ValidationRule(
                    check=ValidationCheck.HAS_FIELD,
                    value="nsn",
                    weight=1.0,
                    description="Has NSN",
                ),
            ]
        )
        chunk = _chunk(
            "test",
            enrichment_fields={"tm_number": "TM 1-1"},
        )
        scored = [ScoredChunk(chunk=chunk, score=0.9, rank=1)]
        results = validator.validate(scored)
        # TM passes (weight 2), NSN fails (weight 1)
        # validation_score = 2.0/3.0 = 0.667
        assert 0.6 < results[0].validation_score < 0.7

    def test_adjusted_score(self) -> None:
        """Test adjusted score combines search and validation."""
        validator = ResultValidator(
            rules=[
                ValidationRule(
                    check=ValidationCheck.HAS_FIELD,
                    value="tm_number",
                ),
            ]
        )
        chunk = _chunk(
            "test",
            enrichment_fields={"tm_number": "TM 1-1"},
        )
        scored = [ScoredChunk(chunk=chunk, score=0.8, rank=1)]
        results = validator.validate(scored)
        # adjusted = 0.8 * 1.0 = 0.8
        assert results[0].adjusted_score == 0.8

    def test_min_validation_score_filters(self) -> None:
        """Test minimum validation score filters out results."""
        validator = ResultValidator(
            rules=[
                ValidationRule(
                    check=ValidationCheck.HAS_FIELD,
                    value="nsn",
                ),
            ],
            min_validation_score=0.5,
        )
        chunk_with = _chunk(
            "test",
            chunk_id="with",
            enrichment_fields={"nsn": "2940-01-234-5678"},
        )
        chunk_without = _chunk("test", chunk_id="without")
        scored = [
            ScoredChunk(chunk=chunk_with, score=0.8, rank=1),
            ScoredChunk(chunk=chunk_without, score=0.9, rank=2),
        ]
        results = validator.validate(scored)
        assert len(results) == 1
        assert results[0].chunk.chunk.id == "with"

    def test_results_sorted_by_adjusted_score(self) -> None:
        """Test results are sorted by adjusted score."""
        validator = ResultValidator(
            rules=[
                ValidationRule(
                    check=ValidationCheck.HAS_FIELD,
                    value="tm_number",
                ),
            ]
        )
        c1 = _chunk("a", chunk_id="a", enrichment_fields={"tm_number": "TM 1"})
        c2 = _chunk("b", chunk_id="b", enrichment_fields={"tm_number": "TM 2"})
        scored = [
            ScoredChunk(chunk=c1, score=0.5, rank=2),
            ScoredChunk(chunk=c2, score=0.9, rank=1),
        ]
        results = validator.validate(scored)
        assert results[0].adjusted_score >= results[1].adjusted_score


# --- RetrievalPipeline Integration Tests ---


class TestRetrievalPipeline:
    """Tests for the full 3-stage retrieval pipeline."""

    def test_basic_retrieval(self) -> None:
        """Test basic end-to-end retrieval."""
        pipeline = RetrievalPipeline()
        chunks = _make_corpus()
        result = pipeline.retrieve("air filter replacement", chunks)
        assert isinstance(result, RetrievalResult)
        assert result.query == "air filter replacement"
        assert len(result.stage_metrics) == 3

    def test_filtered_retrieval(self) -> None:
        """Test retrieval with metadata pre-filter."""
        pipeline = RetrievalPipeline()
        chunks = _make_corpus()
        criteria = FilterCriteria(document_type="technical_manual")
        criteria.add("maintenance_level", FilterOperator.EQ, "organizational")
        result = pipeline.retrieve("air filter", chunks, criteria)
        # Only organizational TM chunks should be searched
        for vr in result.results:
            fields = vr.chunk.chunk.system_metadata.get(ENRICHMENT_FIELDS_KEY, {})
            assert fields.get("maintenance_level", "").lower() == "organizational"

    def test_stage_metrics_recorded(self) -> None:
        """Test all 3 stage metrics are recorded."""
        pipeline = RetrievalPipeline()
        chunks = _make_corpus()
        criteria = FilterCriteria(document_type="technical_manual")
        result = pipeline.retrieve("engine", chunks, criteria)
        assert len(result.stage_metrics) == 3
        assert result.stage_metrics[0].stage == "metadata_filter"
        assert result.stage_metrics[1].stage == "semantic_search"
        assert result.stage_metrics[2].stage == "validation"

    def test_prefilter_reduces_search_space(self) -> None:
        """Test metadata filter reduces search space significantly."""
        pipeline = RetrievalPipeline()
        chunks = _make_corpus()
        criteria = FilterCriteria(document_type="regulation")
        result = pipeline.retrieve("readiness", chunks, criteria)
        filter_metrics = result.stage_metrics[0]
        assert filter_metrics.reduction_ratio >= 0.8

    def test_no_criteria_skips_filter(self) -> None:
        """Test no criteria passes all chunks to search."""
        pipeline = RetrievalPipeline()
        chunks = _make_corpus()
        result = pipeline.retrieve("maintenance", chunks)
        filter_metrics = result.stage_metrics[0]
        assert filter_metrics.reduction_ratio == 0.0

    def test_with_validation_rules(self) -> None:
        """Test retrieval with Stage 3 validation rules."""
        pipeline = RetrievalPipeline(
            validation_rules=[
                ValidationRule(
                    check=ValidationCheck.HAS_FIELD,
                    value="tm_number",
                    description="Must reference a TM",
                ),
                ValidationRule(
                    check=ValidationCheck.MIN_LENGTH,
                    value=30,
                    description="Must have content",
                ),
            ]
        )
        chunks = _make_corpus()
        result = pipeline.retrieve("air filter replacement", chunks)
        for vr in result.results:
            assert vr.validation_score > 0

    def test_top_k_config(self) -> None:
        """Test top_k configuration limits results."""
        config = RetrievalConfig(top_k=2)
        pipeline = RetrievalPipeline(config=config)
        chunks = _make_corpus()
        result = pipeline.retrieve("maintenance", chunks)
        assert len(result.results) <= 2

    def test_total_reduction(self) -> None:
        """Test total reduction is calculated."""
        pipeline = RetrievalPipeline(config=RetrievalConfig(top_k=3))
        chunks = _make_corpus()
        criteria = FilterCriteria(document_type="technical_manual")
        result = pipeline.retrieve("engine oil", chunks, criteria)
        assert result.total_reduction > 0

    def test_to_dict_serialization(self) -> None:
        """Test result serialization."""
        pipeline = RetrievalPipeline()
        chunks = _make_corpus()
        result = pipeline.retrieve("filter", chunks)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "query" in d
        assert "results" in d
        assert "stage_metrics" in d

    def test_top_chunks_property(self) -> None:
        """Test top_chunks convenience property."""
        pipeline = RetrievalPipeline()
        chunks = _make_corpus()
        result = pipeline.retrieve("air filter", chunks)
        top = result.top_chunks
        assert all(isinstance(c, Chunk) for c in top)

    def test_min_score_threshold(self) -> None:
        """Test minimum score threshold filters low-confidence results."""
        config = RetrievalConfig(min_score=0.5)
        pipeline = RetrievalPipeline(config=config)
        chunks = _make_corpus()
        result = pipeline.retrieve("air filter", chunks)
        for vr in result.results:
            assert vr.chunk.score >= 0.5

    def test_realistic_workflow(self) -> None:
        """Test a realistic military maintenance query workflow."""
        pipeline = RetrievalPipeline(
            validation_rules=[
                ValidationRule(
                    check=ValidationCheck.HAS_FIELD,
                    value="tm_number",
                    weight=2.0,
                    description="References a TM",
                ),
                ValidationRule(
                    check=ValidationCheck.HAS_HIERARCHY,
                    weight=1.0,
                    description="Has structural context",
                ),
                ValidationRule(
                    check=ValidationCheck.MIN_LENGTH,
                    value=20,
                    weight=1.0,
                    description="Has substantive content",
                ),
            ],
            config=RetrievalConfig(top_k=5),
        )
        chunks = _make_corpus()
        criteria = FilterCriteria(document_type="technical_manual")
        criteria.add("tm_number", FilterOperator.CONTAINS, "9-2320")
        result = pipeline.retrieve("How to replace the air filter?", chunks, criteria)
        # Should find air filter content from TM chunks
        assert len(result.results) > 0
        # Best result should reference air filter
        top_content = result.results[0].chunk.chunk.content.lower()
        assert "filter" in top_content or "air" in top_content

    def test_empty_query(self) -> None:
        """Test empty query returns no results."""
        pipeline = RetrievalPipeline()
        chunks = _make_corpus()
        result = pipeline.retrieve("", chunks)
        assert len(result.results) == 0

    def test_empty_corpus(self) -> None:
        """Test empty corpus returns no results."""
        pipeline = RetrievalPipeline()
        result = pipeline.retrieve("air filter", [])
        assert len(result.results) == 0
        assert result.total_reduction == 0.0


# --- ScoredChunk Tests ---


class TestScoredChunk:
    """Tests for ScoredChunk dataclass."""

    def test_to_dict(self) -> None:
        """Test serialization."""
        chunk = _chunk("test", chunk_id="c1")
        sc = ScoredChunk(chunk=chunk, score=0.85, rank=1)
        d = sc.to_dict()
        assert d["chunk_id"] == "c1"
        assert d["score"] == 0.85
        assert d["rank"] == 1


# --- StageMetrics Tests ---


class TestStageMetrics:
    """Tests for StageMetrics dataclass."""

    def test_construction(self) -> None:
        """Test basic construction."""
        sm = StageMetrics(
            stage="metadata_filter",
            input_count=100,
            output_count=15,
            reduction_ratio=0.85,
        )
        assert sm.stage == "metadata_filter"
        assert sm.reduction_ratio == 0.85
