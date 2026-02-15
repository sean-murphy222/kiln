"""
Automatic question generation for chunk diagnostic testing.

Core idea: Generate questions that SHOULD retrieve specific chunks,
then test if retrieval actually works. Failures indicate chunking problems.

Strategy types:
1. Boundary-spanning: Questions requiring info from chunk edges
2. Reference-chasing: Questions following cross-references
3. Structural: Questions about lists/procedures
4. Semantic coherence: Questions testing topic specificity
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from chonk.core.document import Chunk


@dataclass
class GeneratedQuestion:
    """A question generated from chunk analysis for diagnostic testing."""

    question: str
    expected_chunk_ids: list[str]
    test_type: str  # boundary_span, reference, structure, coherence
    source_chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


class QuestionGenerator:
    """Generate diagnostic questions from chunks."""

    # Reference patterns for detection
    FORWARD_REFERENCES = [
        r'\bas follows:?\b',
        r'\bbelow:?\b',
        r'\bnext\b',
        r'\bsubsequent\b',
        r'\blater\b',
    ]

    BACKWARD_REFERENCES = [
        r'\bas mentioned\b',
        r'\bas noted\b',
        r'\babove\b',
        r'\bprevious(?:ly)?\b',
        r'\bearlier\b',
        r'\bsee (?:section|figure|table)\b',
    ]

    CONNECTIVES = [
        'however', 'therefore', 'additionally', 'furthermore',
        'moreover', 'consequently', 'thus', 'hence', 'nevertheless',
    ]

    def __init__(self):
        self.forward_ref_patterns = [re.compile(p, re.IGNORECASE) for p in self.FORWARD_REFERENCES]
        self.backward_ref_patterns = [re.compile(p, re.IGNORECASE) for p in self.BACKWARD_REFERENCES]

    def generate_all_questions(
        self,
        chunks: list[Chunk],
    ) -> list[GeneratedQuestion]:
        """Generate all diagnostic questions for a set of chunks."""
        questions = []

        for i, chunk in enumerate(chunks):
            prev_chunk = chunks[i - 1] if i > 0 else None
            next_chunk = chunks[i + 1] if i < len(chunks) - 1 else None

            # Generate different question types
            questions.extend(self.generate_boundary_questions(chunk, prev_chunk, next_chunk))
            questions.extend(self.generate_reference_questions(chunk, chunks))
            questions.extend(self.generate_structural_questions(chunk, prev_chunk, next_chunk))
            questions.extend(self.generate_coherence_questions(chunk))

        return questions

    def generate_boundary_questions(
        self,
        chunk: Chunk,
        prev_chunk: Chunk | None,
        next_chunk: Chunk | None,
    ) -> list[GeneratedQuestion]:
        """Generate questions testing chunk boundary integrity."""
        questions = []

        # Test if chunk starts with dangling connective (likely fragment)
        content = chunk.content.strip().lower()
        for connective in self.CONNECTIVES:
            if content.startswith(connective):
                # This chunk probably needs the previous chunk for context
                expected_chunks = [chunk.id]
                if prev_chunk:
                    expected_chunks.insert(0, prev_chunk.id)

                questions.append(GeneratedQuestion(
                    question=f"What is the complete context for the statement that {connective}...?",
                    expected_chunk_ids=expected_chunks,
                    test_type="boundary_span",
                    source_chunk_id=chunk.id,
                    metadata={
                        "connective": connective,
                        "note": "Chunk starts with connective, likely needs previous chunk",
                    },
                ))
                break

        # Test if chunk ends mid-sentence (no period, exclamation, question mark)
        if chunk.content.strip() and not chunk.content.strip()[-1] in '.!?':
            if next_chunk:
                # Generate question spanning the boundary
                # Extract last few words
                words = chunk.content.strip().split()
                if len(words) >= 5:
                    last_phrase = ' '.join(words[-5:])
                    questions.append(GeneratedQuestion(
                        question=f"What is the complete information about '{last_phrase}'?",
                        expected_chunk_ids=[chunk.id, next_chunk.id],
                        test_type="boundary_span",
                        source_chunk_id=chunk.id,
                        metadata={
                            "note": "Chunk ends without sentence terminator",
                        },
                    ))

        return questions

    def generate_reference_questions(
        self,
        chunk: Chunk,
        all_chunks: list[Chunk],
    ) -> list[GeneratedQuestion]:
        """Generate questions testing reference completeness."""
        questions = []

        # Check for forward references
        for pattern in self.forward_ref_patterns:
            if pattern.search(chunk.content):
                questions.append(GeneratedQuestion(
                    question=f"What information follows regarding {self._extract_subject(chunk)}?",
                    expected_chunk_ids=[chunk.id],  # Should be self-contained if chunked well
                    test_type="forward_reference",
                    source_chunk_id=chunk.id,
                    metadata={
                        "pattern": pattern.pattern,
                        "note": "Contains forward reference - may be orphaned if content is in next chunk",
                    },
                ))
                break

        # Check for backward references
        for pattern in self.backward_ref_patterns:
            if pattern.search(chunk.content):
                questions.append(GeneratedQuestion(
                    question=f"What is the context and background for {self._extract_subject(chunk)}?",
                    expected_chunk_ids=[chunk.id],  # Ideally should have context
                    test_type="backward_reference",
                    source_chunk_id=chunk.id,
                    metadata={
                        "pattern": pattern.pattern,
                        "note": "Contains backward reference - may need previous chunks",
                    },
                ))
                break

        return questions

    def generate_structural_questions(
        self,
        chunk: Chunk,
        prev_chunk: Chunk | None,
        next_chunk: Chunk | None,
    ) -> list[GeneratedQuestion]:
        """Generate questions testing structural integrity (lists, procedures)."""
        questions = []

        # Detect numbered lists
        list_pattern = re.compile(r'^\s*(\d+)[\.\)]\s+', re.MULTILINE)
        matches = list_pattern.findall(chunk.content)

        if matches:
            numbers = [int(m) for m in matches]
            min_num = min(numbers)
            max_num = max(numbers)

            # Check if list starts mid-sequence
            if min_num > 1:
                expected_chunks = [chunk.id]
                if prev_chunk:
                    expected_chunks.insert(0, prev_chunk.id)

                questions.append(GeneratedQuestion(
                    question=f"What are all the items in the numbered list (starting from item 1)?",
                    expected_chunk_ids=expected_chunks,
                    test_type="list_completion",
                    source_chunk_id=chunk.id,
                    metadata={
                        "list_range": f"{min_num}-{max_num}",
                        "note": f"List starts at {min_num}, not 1 - likely split",
                    },
                ))

            # Check if list likely continues beyond chunk
            if max_num >= 5 and not chunk.content.strip().endswith('.'):
                expected_chunks = [chunk.id]
                if next_chunk:
                    expected_chunks.append(next_chunk.id)

                questions.append(GeneratedQuestion(
                    question=f"What is the complete numbered list?",
                    expected_chunk_ids=expected_chunks,
                    test_type="list_completion",
                    source_chunk_id=chunk.id,
                    metadata={
                        "list_range": f"{min_num}-{max_num}",
                        "note": "List may continue in next chunk",
                    },
                ))

        # Detect procedural language
        procedural_markers = [
            r'\bstep\s+\d+',
            r'\bfirst\b.*\bthen\b',
            r'\bshall\b',
            r'\bmust\b.*\bfollowing\b',
        ]

        for marker_pattern in procedural_markers:
            if re.search(marker_pattern, chunk.content, re.IGNORECASE):
                questions.append(GeneratedQuestion(
                    question=f"What is the complete procedure or sequence of steps?",
                    expected_chunk_ids=[chunk.id],
                    test_type="procedure_completion",
                    source_chunk_id=chunk.id,
                    metadata={
                        "pattern": marker_pattern,
                        "note": "Contains procedural language",
                    },
                ))
                break

        return questions

    def generate_coherence_questions(
        self,
        chunk: Chunk,
    ) -> list[GeneratedQuestion]:
        """Generate questions testing semantic coherence."""
        questions = []

        # Very large chunks likely contain multiple topics
        if chunk.token_count > 500:
            # Extract potential topics (simple heuristic: capitalized phrases)
            topic_pattern = re.compile(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*')
            topics = set(topic_pattern.findall(chunk.content[:500]))  # First 500 chars

            if len(topics) > 10:  # Many topics = likely contaminated
                questions.append(GeneratedQuestion(
                    question=f"What specific information is provided about [topic]?",
                    expected_chunk_ids=[chunk.id],
                    test_type="topic_specificity",
                    source_chunk_id=chunk.id,
                    metadata={
                        "token_count": chunk.token_count,
                        "topic_count": len(topics),
                        "note": "Very large chunk with many topics - may contain irrelevant content",
                    },
                ))

        # Very small chunks may be fragments
        if chunk.token_count < 20:
            questions.append(GeneratedQuestion(
                question=f"What is the complete information about '{chunk.content.strip()[:50]}'?",
                expected_chunk_ids=[chunk.id],
                test_type="semantic_incompleteness",
                source_chunk_id=chunk.id,
                metadata={
                    "token_count": chunk.token_count,
                    "note": "Very small chunk - may be incomplete fragment",
                },
            ))

        return questions

    def _extract_subject(self, chunk: Chunk) -> str:
        """Extract a subject/topic from chunk for question generation."""
        # Simple heuristic: find capitalized words/phrases in first 100 chars
        content = chunk.content[:100]
        words = content.split()

        # Look for capitalized sequences
        subject_words = []
        for word in words:
            if word and word[0].isupper() and word.lower() not in ['the', 'a', 'an', 'in', 'on', 'at']:
                subject_words.append(word)
                if len(subject_words) >= 3:
                    break

        if subject_words:
            return ' '.join(subject_words)
        else:
            # Fallback: first few words
            return ' '.join(words[:5]) if len(words) >= 5 else chunk.content[:30]
