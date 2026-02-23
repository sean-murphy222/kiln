"""Tests for Quarry integration module for example scaffolding."""

from __future__ import annotations

import pytest

from forge.src.models import Competency, Example
from forge.src.quarry_integration import (
    CandidateExample,
    CandidateStatus,
    ChunkSource,
    QuarryBridge,
    ScaffoldConfig,
)
from forge.src.storage import ForgeStorage

# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def sample_chunk_dict() -> dict:
    """A raw chunk dict as it would come from Quarry serialization."""
    return {
        "id": "chunk_abc123",
        "block_ids": ["block_001", "block_002"],
        "content": (
            "The hydraulic pump assembly requires inspection every 500 hours. "
            "Remove the access panel and check fluid levels. Verify pressure "
            "readings are within the 2000-3000 PSI range. Replace seals if "
            "any leakage is detected during the pressure test."
        ),
        "token_count": 120,
        "quality": {"overall": 0.85},
        "hierarchy_path": "Chapter 3 > Section 3.2 > Hydraulic Maintenance",
        "user_metadata": {
            "tags": ["hydraulic", "maintenance"],
            "hierarchy_hint": "Section 3.2",
            "notes": None,
            "custom": {},
        },
        "system_metadata": {
            "start_page": 42,
            "end_page": 43,
            "section_title": "Hydraulic Maintenance",
            "document_title": "TM-1-1500-204-23",
        },
    }


@pytest.fixture
def sample_chunk_dict_small() -> dict:
    """A chunk dict that is too small (below min tokens)."""
    return {
        "id": "chunk_tiny001",
        "block_ids": ["block_010"],
        "content": "See figure 3.",
        "token_count": 10,
        "quality": {"overall": 0.5},
        "hierarchy_path": "",
        "user_metadata": {"tags": [], "custom": {}},
        "system_metadata": {},
    }


@pytest.fixture
def sample_chunk_dict_large() -> dict:
    """A chunk dict that exceeds max tokens."""
    return {
        "id": "chunk_huge001",
        "block_ids": ["block_020"],
        "content": "A " * 2500,
        "token_count": 2500,
        "quality": {"overall": 0.7},
        "hierarchy_path": "Chapter 10 > Appendix",
        "user_metadata": {"tags": [], "custom": {}},
        "system_metadata": {},
    }


@pytest.fixture
def chunk_source() -> ChunkSource:
    """A pre-built ChunkSource for direct testing."""
    return ChunkSource(
        chunk_id="chunk_abc123",
        content=(
            "The hydraulic pump assembly requires inspection every 500 hours. "
            "Remove the access panel and check fluid levels."
        ),
        hierarchy_path="Chapter 3 > Section 3.2 > Hydraulic Maintenance",
        source_document="TM-1-1500-204-23",
        page_range="42-43",
        section_title="Hydraulic Maintenance",
        metadata={"start_page": 42, "end_page": 43},
    )


@pytest.fixture
def chunk_source_no_section() -> ChunkSource:
    """A ChunkSource missing section_title and hierarchy_path."""
    return ChunkSource(
        chunk_id="chunk_nosec001",
        content="Perform daily checks on all pneumatic lines and fittings.",
        hierarchy_path="",
        source_document="TM-1-1500-204-23",
        page_range="",
        section_title="",
        metadata={},
    )


@pytest.fixture
def bridge(populated_store: ForgeStorage) -> QuarryBridge:
    """QuarryBridge with a populated ForgeStorage."""
    return QuarryBridge(populated_store)


@pytest.fixture
def safety_competency() -> Competency:
    """A second competency for matching tests."""
    return Competency(
        id="comp_safety001",
        name="Safety Protocols",
        description="Identify and follow safety procedures and warnings",
        discipline_id="disc_test001",
        coverage_target=20,
    )


@pytest.fixture
def bridge_with_competencies(
    populated_store: ForgeStorage,
    safety_competency: Competency,
) -> QuarryBridge:
    """QuarryBridge with multiple competencies available."""
    populated_store.create_competency(safety_competency)
    return QuarryBridge(populated_store)


# ---------------------------------------------------------------
# TestChunkSource
# ---------------------------------------------------------------


class TestChunkSource:
    """Tests for ChunkSource dataclass."""

    def test_construction(self, chunk_source: ChunkSource) -> None:
        """ChunkSource stores all fields correctly."""
        assert chunk_source.chunk_id == "chunk_abc123"
        assert "hydraulic pump" in chunk_source.content
        assert chunk_source.hierarchy_path == "Chapter 3 > Section 3.2 > Hydraulic Maintenance"
        assert chunk_source.source_document == "TM-1-1500-204-23"
        assert chunk_source.page_range == "42-43"
        assert chunk_source.section_title == "Hydraulic Maintenance"
        assert chunk_source.metadata == {"start_page": 42, "end_page": 43}

    def test_to_dict(self, chunk_source: ChunkSource) -> None:
        """to_dict produces a serializable dictionary."""
        data = chunk_source.to_dict()
        assert data["chunk_id"] == "chunk_abc123"
        assert data["source_document"] == "TM-1-1500-204-23"
        assert data["page_range"] == "42-43"
        assert isinstance(data["metadata"], dict)

    def test_from_dict_roundtrip(self, chunk_source: ChunkSource) -> None:
        """from_dict restores a ChunkSource from its dict form."""
        data = chunk_source.to_dict()
        restored = ChunkSource.from_dict(data)
        assert restored.chunk_id == chunk_source.chunk_id
        assert restored.content == chunk_source.content
        assert restored.hierarchy_path == chunk_source.hierarchy_path
        assert restored.source_document == chunk_source.source_document
        assert restored.page_range == chunk_source.page_range
        assert restored.section_title == chunk_source.section_title
        assert restored.metadata == chunk_source.metadata


# ---------------------------------------------------------------
# TestCandidateExample
# ---------------------------------------------------------------


class TestCandidateExample:
    """Tests for CandidateExample dataclass."""

    def test_construction(self, chunk_source: ChunkSource) -> None:
        """CandidateExample stores all fields correctly."""
        candidate = CandidateExample(
            id="cand_test001",
            chunk_source=chunk_source,
            suggested_question="What is the procedure?",
            suggested_answer="Check fluid levels.",
            suggested_competency_id="comp_test001",
            confidence=0.8,
            provenance="Source: TM | Page: 42",
        )
        assert candidate.id == "cand_test001"
        assert candidate.suggested_question == "What is the procedure?"
        assert candidate.suggested_answer == "Check fluid levels."
        assert candidate.suggested_competency_id == "comp_test001"
        assert candidate.confidence == 0.8
        assert candidate.status == CandidateStatus.PENDING

    def test_generate_id_prefix(self) -> None:
        """Generated IDs start with 'cand_'."""
        generated_id = CandidateExample.generate_id()
        assert generated_id.startswith("cand_")
        assert len(generated_id) > 5

    def test_to_dict_from_dict_roundtrip(self, chunk_source: ChunkSource) -> None:
        """Serialization roundtrip preserves all fields."""
        candidate = CandidateExample(
            id="cand_round001",
            chunk_source=chunk_source,
            suggested_question="Q?",
            suggested_answer="A.",
            suggested_competency_id=None,
            confidence=0.5,
            provenance="Source: TM",
        )
        data = candidate.to_dict()
        restored = CandidateExample.from_dict(data)
        assert restored.id == candidate.id
        assert restored.chunk_source.chunk_id == chunk_source.chunk_id
        assert restored.suggested_question == candidate.suggested_question
        assert restored.suggested_answer == candidate.suggested_answer
        assert restored.suggested_competency_id is None
        assert restored.confidence == 0.5
        assert restored.status == CandidateStatus.PENDING
        assert restored.provenance == "Source: TM"

    def test_status_default_is_pending(self, chunk_source: ChunkSource) -> None:
        """Default status is PENDING."""
        candidate = CandidateExample(
            id="cand_def001",
            chunk_source=chunk_source,
            suggested_question="Q?",
            suggested_answer="A.",
            confidence=0.5,
            provenance="",
        )
        assert candidate.status == CandidateStatus.PENDING


# ---------------------------------------------------------------
# TestScaffoldConfig
# ---------------------------------------------------------------


class TestScaffoldConfig:
    """Tests for ScaffoldConfig dataclass."""

    def test_defaults(self) -> None:
        """Default config has sensible values."""
        config = ScaffoldConfig()
        assert config.min_chunk_tokens == 50
        assert config.max_chunk_tokens == 2000
        assert len(config.question_templates) > 0

    def test_custom_templates(self) -> None:
        """Custom templates override defaults."""
        templates = ["What about {section_title}?"]
        config = ScaffoldConfig(question_templates=templates)
        assert config.question_templates == templates
        assert len(config.question_templates) == 1


# ---------------------------------------------------------------
# TestIngestChunks
# ---------------------------------------------------------------


class TestIngestChunks:
    """Tests for QuarryBridge.ingest_chunks."""

    def test_converts_dicts_to_chunk_sources(
        self,
        bridge: QuarryBridge,
        sample_chunk_dict: dict,
    ) -> None:
        """Raw chunk dicts become ChunkSource objects."""
        sources = bridge.ingest_chunks([sample_chunk_dict], source_document="TM-1-1500-204-23")
        assert len(sources) == 1
        src = sources[0]
        assert isinstance(src, ChunkSource)
        assert src.chunk_id == "chunk_abc123"
        assert src.source_document == "TM-1-1500-204-23"
        assert "hydraulic pump" in src.content

    def test_filters_by_min_token_count(
        self,
        bridge: QuarryBridge,
        sample_chunk_dict: dict,
        sample_chunk_dict_small: dict,
    ) -> None:
        """Chunks below min_chunk_tokens are filtered out."""
        sources = bridge.ingest_chunks(
            [sample_chunk_dict, sample_chunk_dict_small],
            source_document="TM-1-1500-204-23",
        )
        assert len(sources) == 1
        assert sources[0].chunk_id == "chunk_abc123"

    def test_filters_by_max_token_count(
        self,
        bridge: QuarryBridge,
        sample_chunk_dict: dict,
        sample_chunk_dict_large: dict,
    ) -> None:
        """Chunks above max_chunk_tokens are filtered out."""
        sources = bridge.ingest_chunks(
            [sample_chunk_dict, sample_chunk_dict_large],
            source_document="TM-1-1500-204-23",
        )
        assert len(sources) == 1
        assert sources[0].chunk_id == "chunk_abc123"

    def test_preserves_metadata(
        self,
        bridge: QuarryBridge,
        sample_chunk_dict: dict,
    ) -> None:
        """System metadata is preserved in ChunkSource.metadata."""
        sources = bridge.ingest_chunks([sample_chunk_dict], source_document="TM-1-1500-204-23")
        src = sources[0]
        assert src.metadata.get("start_page") == 42
        assert src.metadata.get("end_page") == 43
        assert src.section_title == "Hydraulic Maintenance"

    def test_empty_input_returns_empty(self, bridge: QuarryBridge) -> None:
        """Empty chunk list yields empty result."""
        sources = bridge.ingest_chunks([], source_document="TM-1-1500-204-23")
        assert sources == []


# ---------------------------------------------------------------
# TestScaffoldExamples
# ---------------------------------------------------------------


class TestScaffoldExamples:
    """Tests for QuarryBridge.scaffold_examples."""

    def test_generates_candidates_from_chunks(
        self,
        bridge: QuarryBridge,
        chunk_source: ChunkSource,
    ) -> None:
        """Each chunk source produces a candidate example."""
        candidates = bridge.scaffold_examples([chunk_source], discipline_id="disc_test001")
        assert len(candidates) == 1
        assert isinstance(candidates[0], CandidateExample)

    def test_question_is_populated(
        self,
        bridge: QuarryBridge,
        chunk_source: ChunkSource,
    ) -> None:
        """Candidate has a non-empty suggested question."""
        candidates = bridge.scaffold_examples([chunk_source], discipline_id="disc_test001")
        assert len(candidates[0].suggested_question) > 0

    def test_answer_is_populated(
        self,
        bridge: QuarryBridge,
        chunk_source: ChunkSource,
    ) -> None:
        """Candidate has a non-empty suggested answer."""
        candidates = bridge.scaffold_examples([chunk_source], discipline_id="disc_test001")
        assert len(candidates[0].suggested_answer) > 0

    def test_provenance_is_set(
        self,
        bridge: QuarryBridge,
        chunk_source: ChunkSource,
    ) -> None:
        """Candidate includes provenance string."""
        candidates = bridge.scaffold_examples([chunk_source], discipline_id="disc_test001")
        prov = candidates[0].provenance
        assert "TM-1-1500-204-23" in prov
        assert "chunk_abc123" in prov

    def test_confidence_scoring(
        self,
        bridge: QuarryBridge,
        chunk_source: ChunkSource,
        chunk_source_no_section: ChunkSource,
    ) -> None:
        """Chunks with more metadata get higher confidence."""
        candidates_good = bridge.scaffold_examples([chunk_source], discipline_id="disc_test001")
        candidates_sparse = bridge.scaffold_examples(
            [chunk_source_no_section], discipline_id="disc_test001"
        )
        assert candidates_good[0].confidence > candidates_sparse[0].confidence


# ---------------------------------------------------------------
# TestAcceptCandidate
# ---------------------------------------------------------------


class TestAcceptCandidate:
    """Tests for QuarryBridge.accept_candidate."""

    def _make_candidate(self, chunk_source: ChunkSource) -> CandidateExample:
        """Helper to build a candidate for acceptance tests."""
        return CandidateExample(
            id="cand_accept001",
            chunk_source=chunk_source,
            suggested_question="What does Hydraulic Maintenance describe?",
            suggested_answer="Inspection of the hydraulic pump assembly.",
            suggested_competency_id="comp_test001",
            confidence=0.8,
            provenance=(
                "Source: TM-1-1500-204-23 | Page: 42-43 "
                "| Section: Chapter 3 > Section 3.2 > Hydraulic Maintenance "
                "| Chunk: chunk_abc123"
            ),
        )

    def test_creates_example_in_storage(
        self,
        bridge: QuarryBridge,
        chunk_source: ChunkSource,
    ) -> None:
        """Accepting a candidate persists an Example in storage."""
        candidate = self._make_candidate(chunk_source)
        example = bridge.accept_candidate(
            candidate=candidate,
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            competency_id="comp_test001",
        )
        assert isinstance(example, Example)
        assert example.id.startswith("ex_")
        # Verify it was stored
        stored = bridge.storage.get_example(example.id)
        assert stored is not None
        assert stored.question == candidate.suggested_question

    def test_uses_edited_question_and_answer(
        self,
        bridge: QuarryBridge,
        chunk_source: ChunkSource,
    ) -> None:
        """Expert can override question and answer at acceptance."""
        candidate = self._make_candidate(chunk_source)
        example = bridge.accept_candidate(
            candidate=candidate,
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            competency_id="comp_test001",
            question="My custom question?",
            answer="My custom answer.",
        )
        assert example.question == "My custom question?"
        assert example.ideal_answer == "My custom answer."

    def test_preserves_provenance_in_context(
        self,
        bridge: QuarryBridge,
        chunk_source: ChunkSource,
    ) -> None:
        """Provenance string is stored in example.context."""
        candidate = self._make_candidate(chunk_source)
        example = bridge.accept_candidate(
            candidate=candidate,
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            competency_id="comp_test001",
        )
        assert "|provenance:" in example.context
        assert "TM-1-1500-204-23" in example.context

    def test_sets_contributor_id(
        self,
        bridge: QuarryBridge,
        chunk_source: ChunkSource,
    ) -> None:
        """Accepted example carries the accepting contributor's ID."""
        candidate = self._make_candidate(chunk_source)
        example = bridge.accept_candidate(
            candidate=candidate,
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            competency_id="comp_test001",
        )
        assert example.contributor_id == "contrib_test001"

    def test_tracks_competency_id(
        self,
        bridge: QuarryBridge,
        chunk_source: ChunkSource,
    ) -> None:
        """Accepted example is linked to the specified competency."""
        candidate = self._make_candidate(chunk_source)
        example = bridge.accept_candidate(
            candidate=candidate,
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            competency_id="comp_test001",
        )
        assert example.competency_id == "comp_test001"


# ---------------------------------------------------------------
# TestRejectCandidate
# ---------------------------------------------------------------


class TestRejectCandidate:
    """Tests for QuarryBridge.reject_candidate."""

    def test_marks_rejected(
        self,
        bridge: QuarryBridge,
        chunk_source: ChunkSource,
    ) -> None:
        """Rejecting sets status to REJECTED."""
        candidate = CandidateExample(
            id="cand_rej001",
            chunk_source=chunk_source,
            suggested_question="Q?",
            suggested_answer="A.",
            confidence=0.5,
            provenance="",
        )
        rejected = bridge.reject_candidate(candidate)
        assert rejected.status == CandidateStatus.REJECTED

    def test_does_not_create_example(
        self,
        bridge: QuarryBridge,
        chunk_source: ChunkSource,
        populated_store: ForgeStorage,
    ) -> None:
        """Rejected candidates do not persist any Example."""
        initial_examples = populated_store.get_examples_for_competency("comp_test001")
        initial_count = len(initial_examples)

        candidate = CandidateExample(
            id="cand_rej002",
            chunk_source=chunk_source,
            suggested_question="Q?",
            suggested_answer="A.",
            confidence=0.5,
            provenance="",
        )
        bridge.reject_candidate(candidate)

        after_examples = populated_store.get_examples_for_competency("comp_test001")
        assert len(after_examples) == initial_count


# ---------------------------------------------------------------
# TestSuggestCompetency
# ---------------------------------------------------------------


class TestSuggestCompetency:
    """Tests for QuarryBridge.suggest_competency."""

    def test_matches_by_keyword(
        self,
        bridge_with_competencies: QuarryBridge,
    ) -> None:
        """Chunk containing competency keywords gets a match."""
        source = ChunkSource(
            chunk_id="chunk_fault001",
            content="Isolate the fault by checking continuity across terminals.",
            hierarchy_path="Chapter 5 > Fault Isolation",
            source_document="TM-1-1500-204-23",
            page_range="80-81",
            section_title="Fault Isolation",
            metadata={},
        )
        competencies = bridge_with_competencies.storage.get_competencies_for_discipline(
            "disc_test001"
        )
        result = bridge_with_competencies.suggest_competency(source, competencies)
        # Should match "Fault Isolation" competency (comp_test001)
        assert result == "comp_test001"

    def test_returns_none_for_no_match(
        self,
        bridge_with_competencies: QuarryBridge,
    ) -> None:
        """Returns None when no competency keywords overlap."""
        source = ChunkSource(
            chunk_id="chunk_nomatch",
            content="The weather today is sunny and warm.",
            hierarchy_path="",
            source_document="WEATHER.pdf",
            page_range="",
            section_title="",
            metadata={},
        )
        competencies = bridge_with_competencies.storage.get_competencies_for_discipline(
            "disc_test001"
        )
        result = bridge_with_competencies.suggest_competency(source, competencies)
        assert result is None

    def test_case_insensitive(
        self,
        bridge_with_competencies: QuarryBridge,
    ) -> None:
        """Matching is case-insensitive."""
        source = ChunkSource(
            chunk_id="chunk_case001",
            content="FAULT ISOLATION is critical for system reliability.",
            hierarchy_path="",
            source_document="TM-1-1500-204-23",
            page_range="",
            section_title="",
            metadata={},
        )
        competencies = bridge_with_competencies.storage.get_competencies_for_discipline(
            "disc_test001"
        )
        result = bridge_with_competencies.suggest_competency(source, competencies)
        assert result == "comp_test001"

    def test_best_match_selected(
        self,
        bridge_with_competencies: QuarryBridge,
    ) -> None:
        """When multiple competencies match, highest overlap wins."""
        source = ChunkSource(
            chunk_id="chunk_best001",
            content=(
                "Follow all safety protocols during fault isolation. "
                "Safety procedures require protective equipment. "
                "Safety warnings must be heeded."
            ),
            hierarchy_path="",
            source_document="TM-1-1500-204-23",
            page_range="",
            section_title="",
            metadata={},
        )
        competencies = bridge_with_competencies.storage.get_competencies_for_discipline(
            "disc_test001"
        )
        result = bridge_with_competencies.suggest_competency(source, competencies)
        # "safety" appears 3 times vs "fault isolation" once
        assert result == "comp_safety001"


# ---------------------------------------------------------------
# TestGetProvenance
# ---------------------------------------------------------------


class TestGetProvenance:
    """Tests for QuarryBridge.get_provenance."""

    def test_extracts_from_context(self, bridge: QuarryBridge) -> None:
        """Extracts provenance from example context field."""
        example = Example(
            id="ex_prov001",
            question="Q?",
            ideal_answer="A.",
            competency_id="comp_test001",
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            context="Some context|provenance:Source: TM-1 | Page: 42 | Chunk: c1",
        )
        prov = bridge.get_provenance(example)
        assert prov is not None
        assert "Source: TM-1" in prov

    def test_returns_none_if_no_provenance(self, bridge: QuarryBridge) -> None:
        """Returns None when context lacks provenance marker."""
        example = Example(
            id="ex_noprov001",
            question="Q?",
            ideal_answer="A.",
            competency_id="comp_test001",
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            context="Just some regular context",
        )
        prov = bridge.get_provenance(example)
        assert prov is None

    def test_handles_multiple_pipes(self, bridge: QuarryBridge) -> None:
        """Provenance with multiple pipe separators is extracted correctly."""
        example = Example(
            id="ex_pipes001",
            question="Q?",
            ideal_answer="A.",
            competency_id="comp_test001",
            contributor_id="contrib_test001",
            discipline_id="disc_test001",
            context=(
                "Context info|provenance:Source: TM-1-1500-204-23 "
                "| Page: 42-43 | Section: Ch3 > S3.2 | Chunk: chunk_abc123"
            ),
        )
        prov = bridge.get_provenance(example)
        assert prov is not None
        assert "TM-1-1500-204-23" in prov
        assert "chunk_abc123" in prov


# ---------------------------------------------------------------
# TestQuestionGeneration
# ---------------------------------------------------------------


class TestQuestionGeneration:
    """Tests for internal _generate_question method."""

    def test_uses_templates(
        self,
        bridge: QuarryBridge,
        chunk_source: ChunkSource,
    ) -> None:
        """Generated question uses section title or hierarchy path."""
        question = bridge._generate_question(chunk_source)
        assert len(question) > 0
        assert question.endswith("?")

    def test_handles_missing_section_title(
        self,
        bridge: QuarryBridge,
        chunk_source_no_section: ChunkSource,
    ) -> None:
        """Works even when section_title is empty."""
        question = bridge._generate_question(chunk_source_no_section)
        assert len(question) > 0
        assert question.endswith("?")

    def test_extracts_topic_from_content(
        self,
        bridge: QuarryBridge,
        chunk_source_no_section: ChunkSource,
    ) -> None:
        """Falls back to extracting topic from content when no title."""
        question = bridge._generate_question(chunk_source_no_section)
        assert len(question) > 10  # Not a degenerate question
