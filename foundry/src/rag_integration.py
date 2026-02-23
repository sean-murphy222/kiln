"""RAG integration layer connecting LoRA models with Quarry retrieval.

Provides the end-to-end flow: query -> Quarry retrieval -> context building
-> LoRA generation -> citation extraction. All components are abstracted
behind protocols for testability without requiring real models or a live
Quarry instance.

Example::

    from foundry.src.evaluation import MockInference
    from foundry.src.rag_integration import (
        MockRetrievalAdapter, RAGConfig, RAGPipeline,
    )

    model = MockInference(default_response="Answer based on context.")
    retrieval = MockRetrievalAdapter(chunks=[...])
    pipeline = RAGPipeline(model=model, retrieval=retrieval)
    response = pipeline.query("How do I replace a filter?")
    print(response.answer)
    print(response.citations)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from foundry.src.evaluation import (
    EvaluationConfig,
    EvaluationError,
    EvaluationReport,
    EvaluationRunner,
    ModelInference,
    TestCase,
)

# ===================================================================
# Exceptions
# ===================================================================


class RAGError(Exception):
    """Raised for RAG pipeline errors."""


# ===================================================================
# Constants
# ===================================================================

_DEFAULT_SYSTEM_TEMPLATE = (
    "You are a knowledgeable technical assistant. "
    "Answer the question using ONLY the provided context. "
    "If the context does not contain enough information, say so.\n\n"
    "Context:\n{context}\n\n"
    "Question: {query}"
)

_TOKEN_ESTIMATE_DIVISOR = 4  # rough chars-per-token estimate


# ===================================================================
# RAGConfig
# ===================================================================


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline.

    Attributes:
        max_context_chunks: Maximum number of chunks to include in context.
        max_context_tokens: Approximate maximum tokens in the context block.
        citation_format: Citation style, either 'inline' or 'footnote'.
        system_prompt_template: Template with {context} and {query} placeholders.
        include_metadata: Whether to include metadata in the context block.
    """

    max_context_chunks: int = 5
    max_context_tokens: int = 1500
    citation_format: str = "inline"
    system_prompt_template: str = _DEFAULT_SYSTEM_TEMPLATE
    include_metadata: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with all configuration fields.
        """
        return {
            "max_context_chunks": self.max_context_chunks,
            "max_context_tokens": self.max_context_tokens,
            "citation_format": self.citation_format,
            "system_prompt_template": self.system_prompt_template,
            "include_metadata": self.include_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RAGConfig:
        """Deserialize from dictionary.

        Args:
            data: Dict with configuration values.

        Returns:
            RAGConfig instance.
        """
        return cls(
            max_context_chunks=data.get("max_context_chunks", 5),
            max_context_tokens=data.get("max_context_tokens", 1500),
            citation_format=data.get("citation_format", "inline"),
            system_prompt_template=data.get("system_prompt_template", _DEFAULT_SYSTEM_TEMPLATE),
            include_metadata=data.get("include_metadata", True),
        )


# ===================================================================
# CitationSource
# ===================================================================


@dataclass
class CitationSource:
    """A single citation source extracted from a retrieval chunk.

    Attributes:
        chunk_id: Unique identifier for the source chunk.
        document_title: Title of the source document.
        section: Section heading or path within the document.
        page: Page number, or None if unknown.
        relevance_score: How relevant this chunk was to the query (0-1).
    """

    chunk_id: str
    document_title: str
    section: str
    page: int | None
    relevance_score: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with all citation fields.
        """
        return {
            "chunk_id": self.chunk_id,
            "document_title": self.document_title,
            "section": self.section,
            "page": self.page,
            "relevance_score": self.relevance_score,
        }


# ===================================================================
# RAGResponse
# ===================================================================


@dataclass
class RAGResponse:
    """Response from the RAG pipeline.

    Attributes:
        query: The original query text.
        answer: The model-generated answer.
        citations: List of citation sources from retrieved chunks.
        context_used: The chunk texts that were fed to the model.
        retrieval_time_ms: Time spent on retrieval in milliseconds.
        generation_time_ms: Time spent on generation in milliseconds.
        total_time_ms: Total end-to-end time in milliseconds.
        model_name: Identifier for the model used.
    """

    query: str
    answer: str
    citations: list[CitationSource]
    context_used: list[str]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    model_name: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with all response fields.
        """
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "context_used": list(self.context_used),
            "retrieval_time_ms": self.retrieval_time_ms,
            "generation_time_ms": self.generation_time_ms,
            "total_time_ms": self.total_time_ms,
            "model_name": self.model_name,
        }


# ===================================================================
# RetrievalAdapter protocol
# ===================================================================


@runtime_checkable
class RetrievalAdapter(Protocol):
    """Protocol for retrieval backends.

    Abstracts the Quarry retrieval pipeline so the RAG layer
    can be tested with mock data.
    """

    def retrieve(self, query: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Retrieve relevant chunks for a query.

        Args:
            query: The search query text.
            filters: Optional metadata filters for pre-filtering.

        Returns:
            List of chunk dicts with 'text', 'metadata', and 'score' keys.
        """
        ...


# ===================================================================
# MockRetrievalAdapter
# ===================================================================


class MockRetrievalAdapter:
    """Mock retrieval adapter that returns pre-configured chunks.

    Useful for testing the RAG pipeline without a live Quarry instance.

    Args:
        chunks: List of chunk dicts to return on every retrieve call.

    Example::

        adapter = MockRetrievalAdapter(chunks=[
            {"text": "some content", "metadata": {"chunk_id": "c1"}, "score": 0.9},
        ])
        results = adapter.retrieve("any query")
    """

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = list(chunks)

    def retrieve(self, query: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Return pre-configured chunks regardless of query or filters.

        Args:
            query: Ignored for mock.
            filters: Ignored for mock.

        Returns:
            The list of chunks this adapter was initialized with.
        """
        return list(self._chunks)


# ===================================================================
# ContextBuilder
# ===================================================================


class ContextBuilder:
    """Builds prompt context from retrieved chunks.

    Handles chunk formatting, token limiting, metadata inclusion,
    and citation extraction.

    Example::

        builder = ContextBuilder()
        context = builder.build_context(chunks, config)
        prompt = builder.build_prompt(query, context, template)
        citations = builder.extract_citations(chunks)
    """

    def build_context(self, chunks: list[dict[str, Any]], config: RAGConfig) -> str:
        """Format retrieved chunks into a context string for the prompt.

        Respects max_context_chunks and max_context_tokens limits.
        Includes metadata (section, page) if configured.

        Args:
            chunks: List of chunk dicts from retrieval.
            config: RAG configuration with limits.

        Returns:
            Formatted context string, or empty string if no chunks.
        """
        if not chunks:
            return ""

        limited = chunks[: config.max_context_chunks]
        return self._format_chunks(limited, config)

    def build_prompt(self, query: str, context: str, system_template: str) -> str:
        """Combine system template, context, and query into a final prompt.

        Args:
            query: The user question.
            context: The formatted context block.
            system_template: Template with {context} and {query} placeholders.

        Returns:
            The assembled prompt string.
        """
        return system_template.format(context=context, query=query)

    def extract_citations(self, chunks: list[dict[str, Any]]) -> list[CitationSource]:
        """Extract citation metadata from retrieval chunks.

        Args:
            chunks: List of chunk dicts with metadata.

        Returns:
            List of CitationSource instances.
        """
        return [self._chunk_to_citation(chunk) for chunk in chunks]

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _format_chunks(self, chunks: list[dict[str, Any]], config: RAGConfig) -> str:
        """Format chunks into context, respecting token limit.

        Args:
            chunks: Pre-limited list of chunks.
            config: RAG configuration.

        Returns:
            Formatted context string.
        """
        parts: list[str] = []
        token_count = 0

        for i, chunk in enumerate(chunks):
            block = self._format_single_chunk(chunk, i + 1, config)
            block_tokens = self._estimate_tokens(block)

            if token_count + block_tokens > config.max_context_tokens:
                break

            parts.append(block)
            token_count += block_tokens

        return "\n\n".join(parts)

    def _format_single_chunk(self, chunk: dict[str, Any], index: int, config: RAGConfig) -> str:
        """Format a single chunk with optional metadata header.

        Args:
            chunk: The chunk dict.
            index: 1-based chunk index.
            config: RAG configuration.

        Returns:
            Formatted chunk string.
        """
        lines: list[str] = [f"[Source {index}]"]

        if config.include_metadata:
            meta = chunk.get("metadata", {})
            meta_line = self._format_metadata(meta)
            if meta_line:
                lines.append(meta_line)

        lines.append(chunk.get("text", ""))
        return "\n".join(lines)

    @staticmethod
    def _format_metadata(meta: dict[str, Any]) -> str:
        """Format metadata fields into a header line.

        Args:
            meta: Metadata dictionary.

        Returns:
            Formatted metadata string, or empty string if no metadata.
        """
        parts: list[str] = []
        if "section" in meta:
            parts.append(meta["section"])
        if "page" in meta and meta["page"] is not None:
            parts.append(f"page {meta['page']}")
        if "document_title" in meta:
            parts.append(f"({meta['document_title']})")
        return " | ".join(parts) if parts else ""

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count from character length.

        Uses a rough heuristic of 4 characters per token.

        Args:
            text: Input text.

        Returns:
            Estimated token count.
        """
        return max(1, len(text) // _TOKEN_ESTIMATE_DIVISOR)

    @staticmethod
    def _chunk_to_citation(chunk: dict[str, Any]) -> CitationSource:
        """Convert a chunk dict to a CitationSource.

        Args:
            chunk: Chunk dict with metadata and score.

        Returns:
            CitationSource instance.
        """
        meta = chunk.get("metadata", {})
        return CitationSource(
            chunk_id=meta.get("chunk_id", "unknown"),
            document_title=meta.get("document_title", "Unknown Document"),
            section=meta.get("section", ""),
            page=meta.get("page"),
            relevance_score=chunk.get("score", 0.0),
        )


# ===================================================================
# RAGPipeline
# ===================================================================


class RAGPipeline:
    """Orchestrates the RAG query flow.

    Connects a model inference backend with a retrieval adapter,
    building context from retrieved chunks and generating answers
    with citations.

    Args:
        model: Model inference implementation.
        retrieval: Retrieval adapter for fetching relevant chunks.
        config: RAG configuration.
        model_name: Optional model identifier for response metadata.

    Example::

        pipeline = RAGPipeline(model=my_model, retrieval=my_retrieval)
        response = pipeline.query("How do I replace a filter?")
    """

    def __init__(
        self,
        model: ModelInference,
        retrieval: RetrievalAdapter,
        config: RAGConfig | None = None,
        model_name: str = "rag-model",
    ) -> None:
        self._model = model
        self._retrieval = retrieval
        self._config = config or RAGConfig()
        self._model_name = model_name
        self._context_builder = ContextBuilder()

    def query(
        self,
        query_text: str,
        filters: dict[str, Any] | None = None,
    ) -> RAGResponse:
        """Run a single RAG query.

        Args:
            query_text: The user question.
            filters: Optional metadata filters for retrieval.

        Returns:
            RAGResponse with answer, citations, and timing.
        """
        chunks, retrieval_ms = self._retrieve_context(query_text, filters)
        context_texts = [c.get("text", "") for c in chunks]
        context_str = self._context_builder.build_context(chunks, self._config)
        prompt = self._build_prompt(query_text, context_str)
        answer, generation_ms = self._generate_response(prompt)
        citations = self._context_builder.extract_citations(chunks)

        return self._build_response(
            query=query_text,
            answer=answer,
            chunks=chunks,
            context_texts=context_texts,
            citations=citations,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )

    def batch_query(
        self,
        queries: list[str],
        filters: dict[str, Any] | None = None,
    ) -> list[RAGResponse]:
        """Run multiple RAG queries sequentially.

        Args:
            queries: List of query strings.
            filters: Optional metadata filters applied to all queries.

        Returns:
            List of RAGResponse instances, one per query.
        """
        return [self.query(q, filters=filters) for q in queries]

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _retrieve_context(
        self,
        query_text: str,
        filters: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], float]:
        """Retrieve relevant chunks and measure time.

        Args:
            query_text: The search query.
            filters: Optional metadata filters.

        Returns:
            Tuple of (chunks, elapsed_ms).
        """
        start = time.monotonic()
        chunks = self._retrieval.retrieve(query_text, filters=filters)
        elapsed_ms = (time.monotonic() - start) * 1000
        return chunks, elapsed_ms

    def _build_prompt(self, query_text: str, context_str: str) -> str:
        """Build the final prompt from query and context.

        Args:
            query_text: The user question.
            context_str: The formatted context block.

        Returns:
            Complete prompt string.
        """
        return self._context_builder.build_prompt(
            query_text, context_str, self._config.system_prompt_template
        )

    def _generate_response(self, prompt: str) -> tuple[str, float]:
        """Generate a response and measure time.

        Args:
            prompt: The complete prompt.

        Returns:
            Tuple of (answer_text, elapsed_ms).
        """
        start = time.monotonic()
        answer = self._model.generate(prompt)
        elapsed_ms = (time.monotonic() - start) * 1000
        return answer, elapsed_ms

    def _build_response(
        self,
        query: str,
        answer: str,
        chunks: list[dict[str, Any]],
        context_texts: list[str],
        citations: list[CitationSource],
        retrieval_ms: float,
        generation_ms: float,
    ) -> RAGResponse:
        """Assemble the final RAGResponse.

        Args:
            query: Original query.
            answer: Generated answer text.
            chunks: Retrieved chunks.
            context_texts: Text content of used chunks.
            citations: Extracted citation sources.
            retrieval_ms: Retrieval time.
            generation_ms: Generation time.

        Returns:
            Complete RAGResponse.
        """
        # Limit context_texts to chunks actually used
        max_chunks = self._config.max_context_chunks
        used_texts = context_texts[:max_chunks]

        return RAGResponse(
            query=query,
            answer=answer,
            citations=citations,
            context_used=used_texts,
            retrieval_time_ms=retrieval_ms,
            generation_time_ms=generation_ms,
            total_time_ms=retrieval_ms + generation_ms,
            model_name=self._model_name,
        )


# ===================================================================
# RAGEvaluator
# ===================================================================


class RAGEvaluator:
    """Evaluates model quality with RAG-augmented inference.

    Uses the RAG pipeline instead of direct model inference for
    each test case, measuring how well the model performs when
    given retrieved context.

    Args:
        rag_pipeline: The RAG pipeline to use for augmented inference.
        config: Optional evaluation thresholds.

    Example::

        evaluator = RAGEvaluator(rag_pipeline=pipeline)
        report = evaluator.evaluate_with_rag(test_cases, names, "disc_001")
    """

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        config: EvaluationConfig | None = None,
    ) -> None:
        self._pipeline = rag_pipeline
        self._config = config or EvaluationConfig()
        self._runner = EvaluationRunner(config=self._config)

    def evaluate_with_rag(
        self,
        test_cases: list[TestCase],
        competency_names: dict[str, str],
        discipline_id: str,
    ) -> EvaluationReport:
        """Evaluate test cases using RAG-augmented inference.

        For each test case, runs the RAG pipeline to get an answer,
        then evaluates quality using the standard evaluation runner.

        Args:
            test_cases: List of test cases to evaluate.
            competency_names: Map of competency_id to display name.
            discipline_id: Discipline being evaluated.

        Returns:
            EvaluationReport with results from RAG-augmented inference.

        Raises:
            EvaluationError: If no test cases provided.
        """
        if not test_cases:
            raise EvaluationError("No test cases provided for evaluation")

        rag_model = self._build_rag_model(test_cases)
        return self._runner.run_evaluation(
            model=rag_model,
            test_cases=test_cases,
            competency_names=competency_names,
            model_name="rag-augmented",
            discipline_id=discipline_id,
        )

    def compare_rag_vs_direct(
        self,
        model: ModelInference,
        test_cases: list[TestCase],
        competency_names: dict[str, str],
        discipline_id: str,
    ) -> dict[str, Any]:
        """Compare RAG-augmented vs direct model inference.

        Runs the same test cases through both the RAG pipeline and
        direct model inference, returning a comparison.

        Args:
            model: Direct model inference implementation.
            test_cases: Shared test cases.
            competency_names: Map of competency_id to display name.
            discipline_id: Discipline being evaluated.

        Returns:
            Dict with 'rag_report', 'direct_report', 'improvements',
            and 'regressions' keys.
        """
        rag_report = self.evaluate_with_rag(test_cases, competency_names, discipline_id)
        direct_report = self._runner.run_evaluation(
            model=model,
            test_cases=test_cases,
            competency_names=competency_names,
            model_name="direct",
            discipline_id=discipline_id,
        )
        improvements, regressions = self._find_changes(direct_report, rag_report)
        return {
            "rag_report": rag_report,
            "direct_report": direct_report,
            "improvements": improvements,
            "regressions": regressions,
        }

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _build_rag_model(self, test_cases: list[TestCase]) -> _RAGInferenceAdapter:
        """Build a model adapter that routes through the RAG pipeline.

        Args:
            test_cases: Test cases to pre-process for question extraction.

        Returns:
            _RAGInferenceAdapter wrapping the RAG pipeline.
        """
        return _RAGInferenceAdapter(pipeline=self._pipeline)

    @staticmethod
    def _find_changes(
        baseline: EvaluationReport,
        augmented: EvaluationReport,
    ) -> tuple[list[str], list[str]]:
        """Detect improvements and regressions.

        Args:
            baseline: Report from direct inference.
            augmented: Report from RAG-augmented inference.

        Returns:
            Tuple of (improvements, regressions) as competency ID lists.
        """
        improvements: list[str] = []
        regressions: list[str] = []
        all_ids = set(baseline.competency_scores) | set(augmented.competency_scores)

        for comp_id in all_ids:
            base_score = baseline.competency_scores.get(comp_id)
            aug_score = augmented.competency_scores.get(comp_id)
            base_pct = _correct_pct(base_score)
            aug_pct = _correct_pct(aug_score)

            if aug_pct > base_pct:
                improvements.append(comp_id)
            elif aug_pct < base_pct:
                regressions.append(comp_id)

        return improvements, regressions


def _correct_pct(score: Any) -> float:
    """Compute correct percentage from a CompetencyScore.

    Args:
        score: CompetencyScore or None.

    Returns:
        Fraction of correct answers, or 0.0 if None.
    """
    if score is None or score.total_cases == 0:
        return 0.0
    return score.correct / score.total_cases


class _RAGInferenceAdapter:
    """Adapter that makes a RAGPipeline look like a ModelInference.

    Wraps RAG pipeline queries so the EvaluationRunner can use it
    as if it were a simple model inference backend.

    Args:
        pipeline: The RAG pipeline to route queries through.
    """

    def __init__(self, pipeline: RAGPipeline) -> None:
        self._pipeline = pipeline

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate a response by routing through the RAG pipeline.

        Args:
            prompt: The input question (used as the RAG query).
            max_tokens: Ignored; pipeline manages its own limits.

        Returns:
            The RAG pipeline answer text.
        """
        response = self._pipeline.query(prompt)
        return response.answer


# ===================================================================
# RAGSession
# ===================================================================


class RAGSession:
    """Manages a conversational RAG session with history and persistence.

    Tracks query/response history and supports save/load for
    session continuity.

    Args:
        pipeline: The RAG pipeline to use for queries.
        session_dir: Directory for saving session files.

    Example::

        session = RAGSession(pipeline=my_pipeline, session_dir=Path("./sessions"))
        response = session.ask("How do I replace a filter?")
        session.save()
    """

    def __init__(self, pipeline: RAGPipeline, session_dir: Path) -> None:
        self._pipeline = pipeline
        self._session_dir = session_dir
        self._history: list[RAGResponse] = []
        self._session_id = f"session_{uuid.uuid4().hex[:12]}"

    def ask(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
    ) -> RAGResponse:
        """Ask a question and add the response to history.

        Args:
            query: The user question.
            filters: Optional metadata filters for retrieval.

        Returns:
            RAGResponse from the pipeline.
        """
        response = self._pipeline.query(query, filters=filters)
        self._history.append(response)
        return response

    def get_history(self) -> list[RAGResponse]:
        """Return the session query/response history.

        Returns:
            List of RAGResponse instances in chronological order.
        """
        return list(self._history)

    def save(self) -> Path:
        """Persist the session to a JSON file.

        Returns:
            Path to the saved session file.
        """
        self._session_dir.mkdir(parents=True, exist_ok=True)
        path = self._session_dir / f"{self._session_id}.json"
        data = self._serialize()
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, session_path: Path, pipeline: RAGPipeline) -> RAGSession:
        """Load a session from a JSON file.

        Args:
            session_path: Path to the saved session JSON.
            pipeline: RAG pipeline for future queries.

        Returns:
            Restored RAGSession instance.

        Raises:
            RAGError: If session file not found or invalid.
        """
        if not session_path.exists():
            raise RAGError(f"Session file not found: {session_path}")

        raw = session_path.read_text(encoding="utf-8")
        data = _parse_session_json(raw, session_path)
        session = cls(pipeline=pipeline, session_dir=session_path.parent)
        session._session_id = data.get("session_id", session._session_id)
        session._history = _deserialize_history(data.get("history", []))
        return session

    def clear_history(self) -> None:
        """Clear all session history."""
        self._history.clear()

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _serialize(self) -> dict[str, Any]:
        """Serialize the session to a dictionary.

        Returns:
            Dict with session_id and history.
        """
        return {
            "session_id": self._session_id,
            "history": [r.to_dict() for r in self._history],
        }


# ===================================================================
# Module-level helpers
# ===================================================================


def _parse_session_json(raw: str, path: Path) -> dict[str, Any]:
    """Parse session JSON, raising RAGError on failure.

    Args:
        raw: Raw JSON string.
        path: File path for error messages.

    Returns:
        Parsed dictionary.

    Raises:
        RAGError: If JSON parsing fails.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RAGError(f"Failed to parse session file {path}: {exc}") from exc


def _deserialize_history(
    raw_history: list[dict[str, Any]],
) -> list[RAGResponse]:
    """Deserialize a list of RAGResponse dicts.

    Args:
        raw_history: List of serialized response dicts.

    Returns:
        List of RAGResponse instances.
    """
    return [_deserialize_response(item) for item in raw_history]


def _deserialize_response(data: dict[str, Any]) -> RAGResponse:
    """Deserialize a single RAGResponse from dict.

    Args:
        data: Serialized response dict.

    Returns:
        RAGResponse instance.
    """
    citations = [_deserialize_citation(c) for c in data.get("citations", [])]
    return RAGResponse(
        query=data["query"],
        answer=data["answer"],
        citations=citations,
        context_used=data.get("context_used", []),
        retrieval_time_ms=data.get("retrieval_time_ms", 0.0),
        generation_time_ms=data.get("generation_time_ms", 0.0),
        total_time_ms=data.get("total_time_ms", 0.0),
        model_name=data.get("model_name", "unknown"),
    )


def _deserialize_citation(data: dict[str, Any]) -> CitationSource:
    """Deserialize a single CitationSource from dict.

    Args:
        data: Serialized citation dict.

    Returns:
        CitationSource instance.
    """
    return CitationSource(
        chunk_id=data.get("chunk_id", "unknown"),
        document_title=data.get("document_title", "Unknown Document"),
        section=data.get("section", ""),
        page=data.get("page"),
        relevance_score=data.get("relevance_score", 0.0),
    )
