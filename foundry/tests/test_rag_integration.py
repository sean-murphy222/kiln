"""Tests for the RAG integration layer connecting LoRA models with Quarry retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundry.src.rag_integration import (
    CitationSource,
    ContextBuilder,
    MockRetrievalAdapter,
    RAGConfig,
    RAGError,
    RAGEvaluator,
    RAGPipeline,
    RAGResponse,
    RAGSession,
)

# ---------------------------------------------------------------------------
# TestRAGConfig
# ---------------------------------------------------------------------------


class TestRAGConfig:
    """Tests for RAGConfig dataclass."""

    def test_default_values(self) -> None:
        """RAGConfig should have sensible defaults."""
        config = RAGConfig()
        assert config.max_context_chunks == 5
        assert config.max_context_tokens == 1500
        assert config.citation_format == "inline"
        assert config.include_metadata is True
        assert isinstance(config.system_prompt_template, str)
        assert len(config.system_prompt_template) > 0

    def test_custom_values(self) -> None:
        """RAGConfig should accept custom values."""
        config = RAGConfig(
            max_context_chunks=10,
            max_context_tokens=3000,
            citation_format="footnote",
            include_metadata=False,
            system_prompt_template="Custom template: {context}",
        )
        assert config.max_context_chunks == 10
        assert config.max_context_tokens == 3000
        assert config.citation_format == "footnote"
        assert config.include_metadata is False
        assert config.system_prompt_template == "Custom template: {context}"

    def test_to_dict(self) -> None:
        """to_dict should serialize all fields."""
        config = RAGConfig()
        data = config.to_dict()
        assert data["max_context_chunks"] == 5
        assert data["max_context_tokens"] == 1500
        assert data["citation_format"] == "inline"
        assert data["include_metadata"] is True
        assert "system_prompt_template" in data

    def test_from_dict(self) -> None:
        """from_dict should deserialize correctly."""
        original = RAGConfig(max_context_chunks=8, citation_format="footnote")
        data = original.to_dict()
        restored = RAGConfig.from_dict(data)
        assert restored.max_context_chunks == original.max_context_chunks
        assert restored.citation_format == original.citation_format
        assert restored.max_context_tokens == original.max_context_tokens

    def test_roundtrip_serialization(self) -> None:
        """Serialization and deserialization should be lossless."""
        config = RAGConfig(
            max_context_chunks=3,
            max_context_tokens=500,
            citation_format="footnote",
            include_metadata=False,
        )
        restored = RAGConfig.from_dict(config.to_dict())
        assert restored.to_dict() == config.to_dict()


# ---------------------------------------------------------------------------
# TestCitationSource
# ---------------------------------------------------------------------------


class TestCitationSource:
    """Tests for CitationSource dataclass."""

    def test_construction(self) -> None:
        """CitationSource should store all fields."""
        citation = CitationSource(
            chunk_id="chunk_001",
            document_title="TM 9-2320-280-20",
            section="Chapter 3: Engine",
            page=42,
            relevance_score=0.95,
        )
        assert citation.chunk_id == "chunk_001"
        assert citation.document_title == "TM 9-2320-280-20"
        assert citation.section == "Chapter 3: Engine"
        assert citation.page == 42
        assert citation.relevance_score == 0.95

    def test_optional_page(self) -> None:
        """page should be optional (None)."""
        citation = CitationSource(
            chunk_id="chunk_002",
            document_title="Manual",
            section="Section 1",
            page=None,
            relevance_score=0.8,
        )
        assert citation.page is None

    def test_to_dict(self) -> None:
        """to_dict should serialize all fields including None page."""
        citation = CitationSource(
            chunk_id="chunk_003",
            document_title="Doc",
            section="Sec",
            page=None,
            relevance_score=0.7,
        )
        data = citation.to_dict()
        assert data["chunk_id"] == "chunk_003"
        assert data["page"] is None
        assert data["relevance_score"] == 0.7

    def test_to_dict_with_page(self) -> None:
        """to_dict should include page number when present."""
        citation = CitationSource(
            chunk_id="c1",
            document_title="D1",
            section="S1",
            page=10,
            relevance_score=0.9,
        )
        data = citation.to_dict()
        assert data["page"] == 10


# ---------------------------------------------------------------------------
# TestRAGResponse
# ---------------------------------------------------------------------------


class TestRAGResponse:
    """Tests for RAGResponse dataclass."""

    def test_construction(self) -> None:
        """RAGResponse should store all fields."""
        response = RAGResponse(
            query="How do you replace a filter?",
            answer="Remove the old filter and install a new one.",
            citations=[],
            context_used=["chunk text 1"],
            retrieval_time_ms=50.0,
            generation_time_ms=200.0,
            total_time_ms=250.0,
            model_name="test-model",
        )
        assert response.query == "How do you replace a filter?"
        assert response.answer == "Remove the old filter and install a new one."
        assert response.citations == []
        assert response.context_used == ["chunk text 1"]
        assert response.model_name == "test-model"

    def test_timing_fields(self) -> None:
        """Timing fields should be stored correctly."""
        response = RAGResponse(
            query="q",
            answer="a",
            citations=[],
            context_used=[],
            retrieval_time_ms=100.5,
            generation_time_ms=300.2,
            total_time_ms=400.7,
            model_name="m",
        )
        assert response.retrieval_time_ms == 100.5
        assert response.generation_time_ms == 300.2
        assert response.total_time_ms == 400.7

    def test_to_dict(self) -> None:
        """to_dict should serialize all fields including citations."""
        citation = CitationSource(
            chunk_id="c1",
            document_title="Doc",
            section="Sec",
            page=5,
            relevance_score=0.9,
        )
        response = RAGResponse(
            query="q",
            answer="a",
            citations=[citation],
            context_used=["text"],
            retrieval_time_ms=10.0,
            generation_time_ms=20.0,
            total_time_ms=30.0,
            model_name="model",
        )
        data = response.to_dict()
        assert data["query"] == "q"
        assert data["answer"] == "a"
        assert len(data["citations"]) == 1
        assert data["citations"][0]["chunk_id"] == "c1"
        assert data["model_name"] == "model"

    def test_to_dict_empty_citations(self) -> None:
        """to_dict with no citations should produce empty list."""
        response = RAGResponse(
            query="q",
            answer="a",
            citations=[],
            context_used=[],
            retrieval_time_ms=0.0,
            generation_time_ms=0.0,
            total_time_ms=0.0,
            model_name="m",
        )
        data = response.to_dict()
        assert data["citations"] == []


# ---------------------------------------------------------------------------
# TestContextBuilder
# ---------------------------------------------------------------------------


class TestContextBuilder:
    """Tests for ContextBuilder."""

    def test_build_context_with_chunks(self, sample_chunks: list[dict]) -> None:
        """build_context should format chunks into context string."""
        builder = ContextBuilder()
        config = RAGConfig(max_context_chunks=5)
        context = builder.build_context(sample_chunks, config)
        assert isinstance(context, str)
        assert len(context) > 0
        # All chunk texts should appear
        for chunk in sample_chunks:
            assert chunk["text"] in context

    def test_build_context_respects_max_chunks(self, sample_chunks: list[dict]) -> None:
        """build_context should limit to max_context_chunks."""
        builder = ContextBuilder()
        config = RAGConfig(max_context_chunks=2)
        context = builder.build_context(sample_chunks, config)
        # Should only include first 2 chunks
        assert sample_chunks[0]["text"] in context
        assert sample_chunks[1]["text"] in context
        assert sample_chunks[4]["text"] not in context

    def test_build_context_respects_max_tokens(self) -> None:
        """build_context should stop adding chunks when token limit is reached."""
        builder = ContextBuilder()
        # Create chunks with known sizes
        chunks = [
            {"text": "word " * 100, "metadata": {}, "score": 0.9},
            {"text": "more " * 100, "metadata": {}, "score": 0.8},
            {"text": "extra " * 100, "metadata": {}, "score": 0.7},
        ]
        # Very low token limit should truncate
        config = RAGConfig(max_context_tokens=50, max_context_chunks=10)
        context = builder.build_context(chunks, config)
        # Should have stopped before all chunks
        assert "extra" not in context or len(context.split()) < 300

    def test_build_context_includes_metadata(self, sample_chunks: list[dict]) -> None:
        """build_context with include_metadata=True should add metadata."""
        builder = ContextBuilder()
        config = RAGConfig(include_metadata=True)
        context = builder.build_context(sample_chunks, config)
        # Should include section or page info from metadata
        assert "Chapter" in context or "Section" in context or "page" in context.lower()

    def test_build_context_excludes_metadata(self, sample_chunks: list[dict]) -> None:
        """build_context with include_metadata=False should omit metadata."""
        builder = ContextBuilder()
        config = RAGConfig(include_metadata=False)
        context = builder.build_context(sample_chunks, config)
        # Should still have text content
        assert sample_chunks[0]["text"] in context

    def test_build_context_empty_chunks(self) -> None:
        """build_context with empty list should return empty string."""
        builder = ContextBuilder()
        config = RAGConfig()
        context = builder.build_context([], config)
        assert context == ""

    def test_build_prompt(self) -> None:
        """build_prompt should combine system template, context, and query."""
        builder = ContextBuilder()
        context = "Relevant info about filters."
        query = "How do I replace a filter?"
        template = "Use the following context:\n{context}\n\nQuestion: {query}"
        prompt = builder.build_prompt(query, context, template)
        assert "Relevant info about filters." in prompt
        assert "How do I replace a filter?" in prompt

    def test_build_prompt_default_template(self) -> None:
        """build_prompt with default template should include context and query."""
        builder = ContextBuilder()
        config = RAGConfig()
        prompt = builder.build_prompt("my question", "my context", config.system_prompt_template)
        assert "my question" in prompt
        assert "my context" in prompt

    def test_extract_citations(self, sample_chunks: list[dict]) -> None:
        """extract_citations should produce CitationSource from chunk metadata."""
        builder = ContextBuilder()
        citations = builder.extract_citations(sample_chunks)
        assert len(citations) == len(sample_chunks)
        assert all(isinstance(c, CitationSource) for c in citations)
        assert citations[0].chunk_id == sample_chunks[0]["metadata"]["chunk_id"]

    def test_extract_citations_empty(self) -> None:
        """extract_citations with empty list should return empty list."""
        builder = ContextBuilder()
        citations = builder.extract_citations([])
        assert citations == []

    def test_extract_citations_missing_metadata(self) -> None:
        """extract_citations should handle chunks with minimal metadata."""
        builder = ContextBuilder()
        chunks = [
            {"text": "some text", "metadata": {}, "score": 0.5},
        ]
        citations = builder.extract_citations(chunks)
        assert len(citations) == 1
        assert citations[0].chunk_id == "unknown"


# ---------------------------------------------------------------------------
# TestMockRetrievalAdapter
# ---------------------------------------------------------------------------


class TestMockRetrievalAdapter:
    """Tests for MockRetrievalAdapter."""

    def test_returns_configured_chunks(self, sample_chunks: list[dict]) -> None:
        """MockRetrievalAdapter should return the chunks it was initialized with."""
        adapter = MockRetrievalAdapter(chunks=sample_chunks)
        result = adapter.retrieve("any query")
        assert result == sample_chunks

    def test_empty_results(self) -> None:
        """MockRetrievalAdapter with empty chunks should return empty list."""
        adapter = MockRetrievalAdapter(chunks=[])
        result = adapter.retrieve("query")
        assert result == []

    def test_ignores_filters(self, sample_chunks: list[dict]) -> None:
        """MockRetrievalAdapter should return all chunks regardless of filters."""
        adapter = MockRetrievalAdapter(chunks=sample_chunks)
        result = adapter.retrieve("query", filters={"doc_type": "TM"})
        assert result == sample_chunks

    def test_accepts_none_filters(self, sample_chunks: list[dict]) -> None:
        """MockRetrievalAdapter should handle None filters."""
        adapter = MockRetrievalAdapter(chunks=sample_chunks)
        result = adapter.retrieve("query", filters=None)
        assert result == sample_chunks


# ---------------------------------------------------------------------------
# TestRAGPipeline
# ---------------------------------------------------------------------------


class TestRAGPipeline:
    """Tests for RAGPipeline."""

    def test_single_query(self, rag_pipeline: RAGPipeline) -> None:
        """query should return a RAGResponse with answer and citations."""
        response = rag_pipeline.query("How do you replace a filter?")
        assert isinstance(response, RAGResponse)
        assert isinstance(response.answer, str)
        assert response.query == "How do you replace a filter?"
        assert isinstance(response.citations, list)
        assert response.model_name != ""

    def test_query_has_timing(self, rag_pipeline: RAGPipeline) -> None:
        """query should record retrieval and generation timing."""
        response = rag_pipeline.query("test query")
        assert response.retrieval_time_ms >= 0
        assert response.generation_time_ms >= 0
        assert response.total_time_ms >= 0
        assert response.total_time_ms >= response.retrieval_time_ms

    def test_query_has_context_used(
        self, rag_pipeline: RAGPipeline, sample_chunks: list[dict]
    ) -> None:
        """query should populate context_used with chunk texts."""
        response = rag_pipeline.query("test query")
        assert isinstance(response.context_used, list)
        assert len(response.context_used) > 0

    def test_query_has_citations(self, rag_pipeline: RAGPipeline) -> None:
        """query should extract citations from retrieval results."""
        response = rag_pipeline.query("test query")
        assert isinstance(response.citations, list)
        assert len(response.citations) > 0
        assert all(isinstance(c, CitationSource) for c in response.citations)

    def test_batch_query(self, rag_pipeline: RAGPipeline) -> None:
        """batch_query should process multiple queries."""
        queries = ["query 1", "query 2", "query 3"]
        responses = rag_pipeline.batch_query(queries)
        assert len(responses) == 3
        assert all(isinstance(r, RAGResponse) for r in responses)
        assert responses[0].query == "query 1"
        assert responses[2].query == "query 3"

    def test_batch_query_empty(self, rag_pipeline: RAGPipeline) -> None:
        """batch_query with empty list should return empty list."""
        responses = rag_pipeline.batch_query([])
        assert responses == []

    def test_query_with_filters(self, rag_pipeline: RAGPipeline) -> None:
        """query should pass filters to retrieval adapter."""
        response = rag_pipeline.query("test", filters={"doc_type": "TM"})
        assert isinstance(response, RAGResponse)

    def test_empty_retrieval(self) -> None:
        """query with no retrieval results should still return a response."""
        from foundry.src.evaluation import MockInference

        model = MockInference(default_response="I don't have enough context.")
        adapter = MockRetrievalAdapter(chunks=[])
        pipeline = RAGPipeline(model=model, retrieval=adapter)
        response = pipeline.query("unknown question")
        assert isinstance(response, RAGResponse)
        assert response.citations == []
        assert response.context_used == []

    def test_query_response_to_dict(self, rag_pipeline: RAGPipeline) -> None:
        """Response from query should be serializable via to_dict."""
        response = rag_pipeline.query("test")
        data = response.to_dict()
        assert isinstance(data, dict)
        assert "query" in data
        assert "answer" in data
        assert "citations" in data


# ---------------------------------------------------------------------------
# TestRAGEvaluator
# ---------------------------------------------------------------------------


class TestRAGEvaluator:
    """Tests for RAGEvaluator."""

    def test_evaluate_with_rag(
        self,
        rag_pipeline: RAGPipeline,
        sample_test_jsonl: Path,
        competency_names: dict[str, str],
    ) -> None:
        """evaluate_with_rag should produce an EvaluationReport."""
        from foundry.src.evaluation import EvaluationRunner

        runner = EvaluationRunner()
        test_cases = runner.load_test_cases(sample_test_jsonl)
        # Take a small subset for speed
        subset = test_cases[:3]

        evaluator = RAGEvaluator(rag_pipeline=rag_pipeline)
        report = evaluator.evaluate_with_rag(
            test_cases=subset,
            competency_names=competency_names,
            discipline_id="disc_maint",
        )
        assert report.discipline_id == "disc_maint"
        assert report.total_cases == 3
        assert report.status.value == "completed"

    def test_evaluate_with_rag_empty_cases(
        self,
        rag_pipeline: RAGPipeline,
        competency_names: dict[str, str],
    ) -> None:
        """evaluate_with_rag with no test cases should raise error."""
        from foundry.src.evaluation import EvaluationError

        evaluator = RAGEvaluator(rag_pipeline=rag_pipeline)
        with pytest.raises(EvaluationError, match="No test cases"):
            evaluator.evaluate_with_rag(
                test_cases=[],
                competency_names=competency_names,
                discipline_id="disc_maint",
            )

    def test_compare_rag_vs_direct(
        self,
        rag_pipeline: RAGPipeline,
        mock_model,
        sample_test_jsonl: Path,
        competency_names: dict[str, str],
    ) -> None:
        """compare_rag_vs_direct should return comparison dict."""
        from foundry.src.evaluation import EvaluationRunner

        runner = EvaluationRunner()
        test_cases = runner.load_test_cases(sample_test_jsonl)
        subset = test_cases[:3]

        evaluator = RAGEvaluator(rag_pipeline=rag_pipeline)
        comparison = evaluator.compare_rag_vs_direct(
            model=mock_model,
            test_cases=subset,
            competency_names=competency_names,
            discipline_id="disc_maint",
        )
        assert isinstance(comparison, dict)
        assert "rag_report" in comparison
        assert "direct_report" in comparison
        assert "improvements" in comparison
        assert "regressions" in comparison


# ---------------------------------------------------------------------------
# TestRAGSession
# ---------------------------------------------------------------------------


class TestRAGSession:
    """Tests for RAGSession."""

    def test_ask_returns_response(self, rag_pipeline: RAGPipeline, tmp_path: Path) -> None:
        """ask should return RAGResponse and add to history."""
        session = RAGSession(pipeline=rag_pipeline, session_dir=tmp_path)
        response = session.ask("How do I replace a filter?")
        assert isinstance(response, RAGResponse)
        assert len(session.get_history()) == 1

    def test_multiple_asks(self, rag_pipeline: RAGPipeline, tmp_path: Path) -> None:
        """Multiple asks should accumulate in history."""
        session = RAGSession(pipeline=rag_pipeline, session_dir=tmp_path)
        session.ask("question 1")
        session.ask("question 2")
        session.ask("question 3")
        history = session.get_history()
        assert len(history) == 3
        assert history[0].query == "question 1"
        assert history[2].query == "question 3"

    def test_save_and_load(self, rag_pipeline: RAGPipeline, tmp_path: Path) -> None:
        """save should persist session; load should restore it."""
        session = RAGSession(pipeline=rag_pipeline, session_dir=tmp_path)
        session.ask("saved question")
        saved_path = session.save()
        assert saved_path.exists()

        loaded = RAGSession.load(saved_path, pipeline=rag_pipeline)
        assert len(loaded.get_history()) == 1
        assert loaded.get_history()[0].query == "saved question"

    def test_clear_history(self, rag_pipeline: RAGPipeline, tmp_path: Path) -> None:
        """clear_history should empty the history."""
        session = RAGSession(pipeline=rag_pipeline, session_dir=tmp_path)
        session.ask("question")
        assert len(session.get_history()) == 1
        session.clear_history()
        assert len(session.get_history()) == 0

    def test_save_empty_session(self, rag_pipeline: RAGPipeline, tmp_path: Path) -> None:
        """Saving an empty session should work without error."""
        session = RAGSession(pipeline=rag_pipeline, session_dir=tmp_path)
        saved_path = session.save()
        assert saved_path.exists()
        loaded = RAGSession.load(saved_path, pipeline=rag_pipeline)
        assert len(loaded.get_history()) == 0

    def test_load_nonexistent_file(self, rag_pipeline: RAGPipeline, tmp_path: Path) -> None:
        """Loading a nonexistent session file should raise RAGError."""
        bad_path = tmp_path / "nonexistent.json"
        with pytest.raises(RAGError, match="not found"):
            RAGSession.load(bad_path, pipeline=rag_pipeline)


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


class TestIntegration:
    """Full pipeline integration tests: mock model + mock retrieval -> RAG response."""

    def test_full_pipeline_flow(self, sample_chunks: list[dict]) -> None:
        """End-to-end: query -> retrieval -> context -> generation -> citations."""
        from foundry.src.evaluation import MockInference

        model = MockInference(
            default_response=(
                "To replace the hydraulic filter, remove the old filter, "
                "clean the housing, and install the new filter per TM specs."
            ),
        )
        adapter = MockRetrievalAdapter(chunks=sample_chunks)
        config = RAGConfig(max_context_chunks=3)
        pipeline = RAGPipeline(model=model, retrieval=adapter, config=config)

        response = pipeline.query("How do I replace a hydraulic filter?")

        assert response.query == "How do I replace a hydraulic filter?"
        assert "filter" in response.answer.lower()
        assert len(response.citations) > 0
        assert len(response.context_used) > 0
        assert response.retrieval_time_ms >= 0
        assert response.generation_time_ms >= 0
        assert response.total_time_ms >= 0

    def test_full_pipeline_serialization(self, sample_chunks: list[dict]) -> None:
        """Full pipeline response should be JSON-serializable."""
        from foundry.src.evaluation import MockInference

        model = MockInference(default_response="Test answer.")
        adapter = MockRetrievalAdapter(chunks=sample_chunks)
        pipeline = RAGPipeline(model=model, retrieval=adapter)

        response = pipeline.query("test")
        data = response.to_dict()
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
        restored = json.loads(json_str)
        assert restored["query"] == "test"
        assert restored["answer"] == "Test answer."

    def test_session_with_pipeline(self, sample_chunks: list[dict], tmp_path: Path) -> None:
        """Session should work with full pipeline through save/load cycle."""
        from foundry.src.evaluation import MockInference

        model = MockInference(default_response="Session answer.")
        adapter = MockRetrievalAdapter(chunks=sample_chunks)
        pipeline = RAGPipeline(model=model, retrieval=adapter)

        session = RAGSession(pipeline=pipeline, session_dir=tmp_path)
        session.ask("first question")
        session.ask("second question")
        saved = session.save()

        loaded = RAGSession.load(saved, pipeline=pipeline)
        assert len(loaded.get_history()) == 2
        assert loaded.get_history()[0].query == "first question"

    def test_rag_config_affects_context(self, sample_chunks: list[dict]) -> None:
        """Changing RAGConfig should affect the context passed to the model."""
        from foundry.src.evaluation import MockInference

        model = MockInference(default_response="answer")
        adapter = MockRetrievalAdapter(chunks=sample_chunks)

        # With only 1 chunk allowed
        config_small = RAGConfig(max_context_chunks=1)
        pipeline_small = RAGPipeline(model=model, retrieval=adapter, config=config_small)
        response_small = pipeline_small.query("test")

        # With all chunks allowed
        config_large = RAGConfig(max_context_chunks=10)
        pipeline_large = RAGPipeline(model=model, retrieval=adapter, config=config_large)
        response_large = pipeline_large.query("test")

        # More context chunks should mean more context_used entries
        assert len(response_small.context_used) <= len(response_large.context_used)
        assert len(response_small.context_used) == 1
