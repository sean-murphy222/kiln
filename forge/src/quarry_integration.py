"""Quarry integration for example scaffolding in Forge.

Bridges Quarry's processed chunks into Forge's example workflow.
Generates template-based candidate Q/A pairs from chunk content and
metadata, which domain experts review and edit before acceptance.
No LLM calls -- all generation uses templates and extraction.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from forge.src.models import Competency, Example, ReviewStatus
from forge.src.storage import ForgeStorage


class ScaffoldingError(Exception):
    """Raised for scaffolding workflow errors."""


class CandidateStatus(str, Enum):
    """Status of a candidate example scaffold.

    Attributes:
        PENDING: Awaiting expert review.
        ACCEPTED: Approved and converted to Example.
        REJECTED: Discarded by expert.
        EDITED: Modified by expert before acceptance.
    """

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EDITED = "edited"


_DEFAULT_QUESTION_TEMPLATES = [
    "What does {section_title} describe?",
    "What is the procedure described in {section_title}?",
    "Explain the key points covered in {section_title}.",
]

_FALLBACK_QUESTION_TEMPLATES = [
    "What does this section describe regarding {topic}?",
    "Explain the procedure for {topic}.",
    "What are the key points about {topic}?",
]


@dataclass
class ScaffoldConfig:
    """Configuration for example scaffolding.

    Attributes:
        min_chunk_tokens: Minimum token count to accept a chunk.
        max_chunk_tokens: Maximum token count to accept a chunk.
        question_templates: Template strings for question generation.
    """

    min_chunk_tokens: int = 50
    max_chunk_tokens: int = 2000
    question_templates: list[str] = field(default_factory=lambda: list(_DEFAULT_QUESTION_TEMPLATES))


@dataclass
class ChunkSource:
    """Represents a Quarry chunk prepared for scaffolding.

    Attributes:
        chunk_id: Original Quarry chunk identifier.
        content: Text content of the chunk.
        hierarchy_path: Structural path (e.g. "Chapter 1 > Section 2").
        source_document: Name or identifier of the source document.
        page_range: Page range string (e.g. "42-43").
        section_title: Title of the containing section.
        metadata: Additional metadata from Quarry system_metadata.
    """

    chunk_id: str
    content: str
    hierarchy_path: str = ""
    source_document: str = ""
    page_range: str = ""
    section_title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation of this ChunkSource.
        """
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "hierarchy_path": self.hierarchy_path,
            "source_document": self.source_document,
            "page_range": self.page_range,
            "section_title": self.section_title,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkSource:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with ChunkSource fields.

        Returns:
            Restored ChunkSource instance.
        """
        return cls(
            chunk_id=data["chunk_id"],
            content=data["content"],
            hierarchy_path=data.get("hierarchy_path", ""),
            source_document=data.get("source_document", ""),
            page_range=data.get("page_range", ""),
            section_title=data.get("section_title", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CandidateExample:
    """A scaffold candidate that needs expert review before becoming an Example.

    Attributes:
        id: Unique identifier (prefixed with 'cand_').
        chunk_source: The Quarry chunk this candidate was derived from.
        suggested_question: Template-generated question scaffold.
        suggested_answer: Extracted answer scaffold from chunk content.
        suggested_competency_id: Auto-suggested competency (may be None).
        confidence: Quality score for the scaffold (0.0-1.0).
        status: Current review status.
        provenance: Formatted source reference string.
        created_at: Creation timestamp.
    """

    id: str
    chunk_source: ChunkSource
    suggested_question: str
    suggested_answer: str
    suggested_competency_id: str | None = None
    confidence: float = 0.0
    status: CandidateStatus = CandidateStatus.PENDING
    provenance: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    @staticmethod
    def generate_id() -> str:
        """Generate a unique candidate ID.

        Returns:
            String ID prefixed with 'cand_'.
        """
        return f"cand_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation of this CandidateExample.
        """
        return {
            "id": self.id,
            "chunk_source": self.chunk_source.to_dict(),
            "suggested_question": self.suggested_question,
            "suggested_answer": self.suggested_answer,
            "suggested_competency_id": self.suggested_competency_id,
            "confidence": self.confidence,
            "status": self.status.value,
            "provenance": self.provenance,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateExample:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with CandidateExample fields.

        Returns:
            Restored CandidateExample instance.
        """
        return cls(
            id=data["id"],
            chunk_source=ChunkSource.from_dict(data["chunk_source"]),
            suggested_question=data["suggested_question"],
            suggested_answer=data["suggested_answer"],
            suggested_competency_id=data.get("suggested_competency_id"),
            confidence=data.get("confidence", 0.0),
            status=CandidateStatus(data.get("status", "pending")),
            provenance=data.get("provenance", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


class QuarryBridge:
    """Bridges Quarry chunks into Forge example scaffolding.

    Converts processed Quarry chunks into candidate Q/A pairs using
    template-based generation. Domain experts review, edit, and accept
    candidates, which become real Examples with full provenance tracking.

    Args:
        storage: ForgeStorage instance for persisting accepted examples.
        config: Optional scaffold configuration. Uses defaults if None.

    Example::

        bridge = QuarryBridge(storage)
        sources = bridge.ingest_chunks(chunk_dicts, "TM-1-1500")
        candidates = bridge.scaffold_examples(sources, "disc_001")
        example = bridge.accept_candidate(
            candidates[0], "contrib_001", "disc_001", "comp_001"
        )
    """

    def __init__(
        self,
        storage: ForgeStorage,
        config: ScaffoldConfig | None = None,
    ) -> None:
        self.storage = storage
        self.config = config or ScaffoldConfig()

    def ingest_chunks(
        self,
        chunks_data: list[dict[str, Any]],
        source_document: str,
    ) -> list[ChunkSource]:
        """Convert raw Quarry chunk dicts into ChunkSource objects.

        Filters chunks by token count bounds from config.

        Args:
            chunks_data: List of serialized Quarry chunk dictionaries.
            source_document: Name/ID of the source document.

        Returns:
            List of ChunkSource objects that pass token filters.
        """
        sources: list[ChunkSource] = []
        for chunk_dict in chunks_data:
            token_count = chunk_dict.get("token_count", 0)
            if not self._is_valid_token_count(token_count):
                continue
            source = self._chunk_dict_to_source(chunk_dict, source_document)
            sources.append(source)
        return sources

    def scaffold_examples(
        self,
        chunk_sources: list[ChunkSource],
        discipline_id: str,
    ) -> list[CandidateExample]:
        """Generate candidate Q/A pairs from chunk sources.

        Uses template-based question generation and content extraction
        for answer scaffolds. Optionally suggests competency matches.

        Args:
            chunk_sources: ChunkSource objects to scaffold from.
            discipline_id: Discipline ID for competency lookup.

        Returns:
            List of CandidateExample objects ready for expert review.
        """
        competencies = self.storage.get_competencies_for_discipline(discipline_id)
        candidates: list[CandidateExample] = []
        for source in chunk_sources:
            candidate = self._build_candidate(source, competencies)
            candidates.append(candidate)
        return candidates

    def accept_candidate(
        self,
        candidate: CandidateExample,
        contributor_id: str,
        discipline_id: str,
        competency_id: str,
        question: str | None = None,
        answer: str | None = None,
    ) -> Example:
        """Accept a candidate, creating a real Example in storage.

        Expert may optionally override question and answer text.
        Provenance is preserved in the Example's context field.

        Args:
            candidate: The candidate to accept.
            contributor_id: ID of the accepting contributor.
            discipline_id: Target discipline ID.
            competency_id: Target competency ID.
            question: Override question text (uses suggested if None).
            answer: Override answer text (uses suggested if None).

        Returns:
            The persisted Example with provenance in context.
        """
        final_question = question or candidate.suggested_question
        final_answer = answer or candidate.suggested_answer
        context = self._format_context_with_provenance(candidate.provenance)

        example = Example(
            id=Example.generate_id(),
            question=final_question,
            ideal_answer=final_answer,
            competency_id=competency_id,
            contributor_id=contributor_id,
            discipline_id=discipline_id,
            context=context,
            review_status=ReviewStatus.PENDING,
        )
        candidate.status = CandidateStatus.ACCEPTED
        return self.storage.create_example(example)

    def reject_candidate(
        self,
        candidate: CandidateExample,
    ) -> CandidateExample:
        """Mark a candidate as rejected without creating an Example.

        Args:
            candidate: The candidate to reject.

        Returns:
            The candidate with status set to REJECTED.
        """
        candidate.status = CandidateStatus.REJECTED
        return candidate

    def get_provenance(self, example: Example) -> str | None:
        """Extract provenance string from an example's context.

        Args:
            example: Example to check for provenance.

        Returns:
            Provenance string if found, None otherwise.
        """
        marker = "|provenance:"
        if marker not in example.context:
            return None
        idx = example.context.index(marker)
        return example.context[idx + len(marker) :]

    def suggest_competency(
        self,
        chunk_source: ChunkSource,
        competencies: list[Competency],
    ) -> str | None:
        """Suggest the best-matching competency for a chunk.

        Uses simple keyword overlap between chunk content and
        competency name/description. Returns the competency with
        the highest overlap score, or None if no meaningful match.

        Args:
            chunk_source: The chunk to match.
            competencies: Available competencies to match against.

        Returns:
            Competency ID of the best match, or None.
        """
        if not competencies:
            return None

        chunk_words = self._extract_words(chunk_source.content)
        chunk_words |= self._extract_words(chunk_source.section_title)
        chunk_words |= self._extract_words(chunk_source.hierarchy_path)

        best_id: str | None = None
        best_score = 0

        for comp in competencies:
            score = self._compute_keyword_overlap(chunk_words, comp)
            if score > best_score:
                best_score = score
                best_id = comp.id

        return best_id if best_score > 0 else None

    # ---------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------

    def _is_valid_token_count(self, token_count: int) -> bool:
        """Check whether a token count is within configured bounds.

        Args:
            token_count: Number of tokens in the chunk.

        Returns:
            True if within [min_chunk_tokens, max_chunk_tokens].
        """
        return self.config.min_chunk_tokens <= token_count <= self.config.max_chunk_tokens

    def _chunk_dict_to_source(
        self,
        chunk_dict: dict[str, Any],
        source_document: str,
    ) -> ChunkSource:
        """Convert a raw Quarry chunk dict to a ChunkSource.

        Args:
            chunk_dict: Serialized Quarry chunk dictionary.
            source_document: Name/ID of the source document.

        Returns:
            ChunkSource populated from the chunk dict.
        """
        sys_meta = chunk_dict.get("system_metadata", {})
        page_range = self._extract_page_range(sys_meta)
        section_title = sys_meta.get("section_title", "")

        return ChunkSource(
            chunk_id=chunk_dict["id"],
            content=chunk_dict.get("content", ""),
            hierarchy_path=chunk_dict.get("hierarchy_path", ""),
            source_document=source_document,
            page_range=page_range,
            section_title=section_title,
            metadata=sys_meta,
        )

    def _extract_page_range(self, sys_meta: dict[str, Any]) -> str:
        """Build a page range string from system metadata.

        Args:
            sys_meta: Quarry system_metadata dictionary.

        Returns:
            Page range string like "42-43", or empty string.
        """
        start = sys_meta.get("start_page")
        end = sys_meta.get("end_page")
        if start is not None and end is not None:
            return f"{start}-{end}"
        if start is not None:
            return str(start)
        return ""

    def _build_candidate(
        self,
        source: ChunkSource,
        competencies: list[Competency],
    ) -> CandidateExample:
        """Build a CandidateExample from a ChunkSource.

        Args:
            source: The chunk source to scaffold from.
            competencies: Available competencies for suggestion.

        Returns:
            A new CandidateExample with generated fields.
        """
        question = self._generate_question(source)
        answer = self._generate_answer_excerpt(source)
        provenance = self._build_provenance(source)
        confidence = self._compute_confidence(source)
        comp_id = self.suggest_competency(source, competencies)

        return CandidateExample(
            id=CandidateExample.generate_id(),
            chunk_source=source,
            suggested_question=question,
            suggested_answer=answer,
            suggested_competency_id=comp_id,
            confidence=confidence,
            provenance=provenance,
        )

    def _generate_question(self, chunk: ChunkSource) -> str:
        """Generate a template-based question from chunk metadata.

        Uses section_title if available, otherwise extracts a topic
        phrase from the first sentence of the chunk content.

        Args:
            chunk: The chunk source to generate a question from.

        Returns:
            A question string ending with '?'.
        """
        if chunk.section_title:
            template = self.config.question_templates[0]
            return template.format(section_title=chunk.section_title)

        topic = self._extract_topic(chunk.content)
        fallback = _FALLBACK_QUESTION_TEMPLATES[0]
        return fallback.format(topic=topic)

    def _generate_answer_excerpt(self, chunk: ChunkSource) -> str:
        """Extract an answer scaffold from chunk content.

        Takes the first 500 characters of content, trimmed to the
        last complete sentence boundary when possible.

        Args:
            chunk: The chunk source to extract an answer from.

        Returns:
            Cleaned excerpt string for expert editing.
        """
        excerpt = chunk.content[:500].strip()
        last_period = excerpt.rfind(".")
        if last_period > 50:
            excerpt = excerpt[: last_period + 1]
        return excerpt

    def _build_provenance(self, chunk: ChunkSource) -> str:
        """Format a provenance reference string.

        Args:
            chunk: The chunk source to build provenance for.

        Returns:
            Formatted provenance like "Source: X | Page: Y | ...".
        """
        parts = [f"Source: {chunk.source_document}"]
        if chunk.page_range:
            parts.append(f"Page: {chunk.page_range}")
        if chunk.hierarchy_path:
            parts.append(f"Section: {chunk.hierarchy_path}")
        parts.append(f"Chunk: {chunk.chunk_id}")
        return " | ".join(parts)

    def _compute_confidence(self, source: ChunkSource) -> float:
        """Compute a confidence score for a scaffold candidate.

        Scores are based on metadata completeness: having a hierarchy
        path, section title, and page range each contribute to the
        score. Range is 0.0 to 1.0.

        Args:
            source: The chunk source to score.

        Returns:
            Confidence float between 0.0 and 1.0.
        """
        score = 0.25  # Base score for having content
        if source.hierarchy_path:
            score += 0.25
        if source.section_title:
            score += 0.25
        if source.page_range:
            score += 0.25
        return score

    def _extract_topic(self, content: str) -> str:
        """Extract a short topic phrase from content text.

        Takes the first sentence and truncates to a reasonable
        length for use in question templates.

        Args:
            content: Raw text content from a chunk.

        Returns:
            A topic phrase suitable for template insertion.
        """
        first_sentence = content.split(".")[0].strip()
        if len(first_sentence) > 80:
            first_sentence = first_sentence[:80].rsplit(" ", 1)[0]
        return first_sentence.lower()

    def _format_context_with_provenance(self, provenance: str) -> str:
        """Format the context string with embedded provenance.

        Args:
            provenance: The provenance string to embed.

        Returns:
            Context string with provenance marker.
        """
        return f"|provenance:{provenance}"

    def _extract_words(self, text: str) -> set[str]:
        """Extract unique lowercase words from text.

        Args:
            text: Input text to tokenize.

        Returns:
            Set of lowercase word strings.
        """
        return set(re.findall(r"[a-z]+", text.lower()))

    def _compute_keyword_overlap(
        self,
        chunk_words: set[str],
        competency: Competency,
    ) -> int:
        """Count keyword matches between chunk words and a competency.

        Matches against both the competency name and description.
        Common stop words are excluded from matching.

        Args:
            chunk_words: Pre-extracted words from the chunk.
            competency: Competency to match against.

        Returns:
            Integer count of overlapping meaningful words.
        """
        stop_words = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "is",
            "are",
            "in",
            "on",
            "of",
            "to",
            "for",
            "with",
            "by",
            "at",
            "from",
            "that",
            "this",
            "it",
            "be",
            "as",
            "do",
            "not",
            "but",
            "all",
            "can",
            "has",
            "have",
            "was",
            "were",
            "will",
            "would",
        }
        comp_words = self._extract_words(competency.name)
        comp_words |= self._extract_words(competency.description)
        meaningful_comp = comp_words - stop_words
        meaningful_chunk = chunk_words - stop_words
        return len(meaningful_comp & meaningful_chunk)
