"""
Core diagnostic analyzer for chunk quality.

Combines:
1. Static analysis (heuristics)
2. Query-based testing (generated questions)
3. Problem classification
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from chonk.core.document import Chunk, ChonkDocument


class ProblemType(Enum):
    """Types of chunking problems that can be detected."""

    SEMANTIC_INCOMPLETE = "semantic_incomplete"
    SEMANTIC_CONTAMINATION = "semantic_contamination"
    STRUCTURAL_BREAKAGE = "structural_breakage"
    REFERENCE_ORPHANING = "reference_orphaning"


class Severity(Enum):
    """Problem severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ChunkProblem:
    """A detected problem with a chunk."""

    id: str
    chunk_id: str
    problem_type: ProblemType
    severity: Severity
    description: str
    suggested_fix: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "chunk_id": self.chunk_id,
            "type": self.problem_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "suggested_fix": self.suggested_fix,
            "metadata": self.metadata,
        }


class DiagnosticAnalyzer:
    """Analyze chunks for quality problems."""

    # Reference patterns
    FORWARD_REF_PATTERNS = [
        r'\bas follows:?\b',
        r'\bbelow:?\b',
        r'\bsee table\s+\d+',
        r'\bsee figure\s+\d+',
    ]

    BACKWARD_REF_PATTERNS = [
        r'\bas mentioned\b',
        r'\bas noted\b',
        r'\babove\b',
        r'\bprevious(?:ly)?\b',
        r'\bsee section\b',
    ]

    DANGLING_CONNECTIVES = [
        'however', 'therefore', 'additionally', 'furthermore',
        'moreover', 'consequently', 'thus', 'hence',
    ]

    def __init__(self):
        self.forward_patterns = [re.compile(p, re.IGNORECASE) for p in self.FORWARD_REF_PATTERNS]
        self.backward_patterns = [re.compile(p, re.IGNORECASE) for p in self.BACKWARD_REF_PATTERNS]

    def analyze_document(self, document: ChonkDocument) -> list[ChunkProblem]:
        """Run all diagnostic tests on a document's chunks."""
        problems = []

        for i, chunk in enumerate(document.chunks):
            prev_chunk = document.chunks[i - 1] if i > 0 else None
            next_chunk = document.chunks[i + 1] if i < len(document.chunks) - 1 else None

            # Run static analysis
            problems.extend(self._check_size_issues(chunk))
            problems.extend(self._check_sentence_completeness(chunk, prev_chunk, next_chunk))
            problems.extend(self._check_reference_orphaning(chunk))
            problems.extend(self._check_structural_integrity(chunk, prev_chunk, next_chunk))

        return problems

    def _check_size_issues(self, chunk: Chunk) -> list[ChunkProblem]:
        """Detect chunks that are too small or too large."""
        problems = []

        # Very small chunks (semantic incompleteness)
        if chunk.token_count < 20:
            severity = Severity.HIGH if chunk.token_count < 10 else Severity.MEDIUM

            problems.append(ChunkProblem(
                id=f"problem_{chunk.id}_size_small",
                chunk_id=chunk.id,
                problem_type=ProblemType.SEMANTIC_INCOMPLETE,
                severity=severity,
                description=f"Chunk has only {chunk.token_count} tokens - likely an incomplete fragment",
                suggested_fix="Consider merging with adjacent chunks to form complete semantic unit",
                metadata={"token_count": chunk.token_count},
            ))

        # Very large chunks (semantic contamination)
        elif chunk.token_count > 500:
            severity = Severity.HIGH if chunk.token_count > 800 else Severity.MEDIUM

            problems.append(ChunkProblem(
                id=f"problem_{chunk.id}_size_large",
                chunk_id=chunk.id,
                problem_type=ProblemType.SEMANTIC_CONTAMINATION,
                severity=severity,
                description=f"Chunk has {chunk.token_count} tokens - may contain multiple unrelated topics",
                suggested_fix="Consider splitting chunk at topic boundaries or major section breaks",
                metadata={"token_count": chunk.token_count},
            ))

        return problems

    def _check_sentence_completeness(
        self,
        chunk: Chunk,
        prev_chunk: Chunk | None,
        next_chunk: Chunk | None,
    ) -> list[ChunkProblem]:
        """Detect incomplete sentences and dangling connectives."""
        problems = []
        content = chunk.content.strip()

        if not content:
            return problems

        # Check for lowercase start (mid-sentence fragment)
        if content[0].islower() and not content[0].isdigit():
            problems.append(ChunkProblem(
                id=f"problem_{chunk.id}_lowercase_start",
                chunk_id=chunk.id,
                problem_type=ProblemType.SEMANTIC_INCOMPLETE,
                severity=Severity.HIGH,
                description="Chunk starts with lowercase letter - likely mid-sentence fragment",
                suggested_fix="Merge with previous chunk to complete the sentence" if prev_chunk else None,
                metadata={"start_char": content[0]},
            ))

        # Check for dangling connectives
        content_lower = content.lower()
        for connective in self.DANGLING_CONNECTIVES:
            if content_lower.startswith(connective):
                problems.append(ChunkProblem(
                    id=f"problem_{chunk.id}_connective_{connective}",
                    chunk_id=chunk.id,
                    problem_type=ProblemType.SEMANTIC_INCOMPLETE,
                    severity=Severity.MEDIUM,
                    description=f"Chunk starts with '{connective}' - needs context from previous chunk",
                    suggested_fix="Merge with previous chunk to preserve logical flow" if prev_chunk else None,
                    metadata={"connective": connective},
                ))
                break

        # Check for incomplete ending (no sentence terminator)
        if content and content[-1] not in '.!?':
            if len(content) > 50:  # Only flag if chunk is substantial
                problems.append(ChunkProblem(
                    id=f"problem_{chunk.id}_incomplete_end",
                    chunk_id=chunk.id,
                    problem_type=ProblemType.SEMANTIC_INCOMPLETE,
                    severity=Severity.MEDIUM,
                    description="Chunk ends without sentence terminator - may be split mid-sentence",
                    suggested_fix="Merge with next chunk to complete the sentence" if next_chunk else None,
                    metadata={"last_char": content[-1]},
                ))

        return problems

    def _check_reference_orphaning(self, chunk: Chunk) -> list[ChunkProblem]:
        """Detect broken cross-references."""
        problems = []

        # Forward references
        for pattern in self.forward_patterns:
            if pattern.search(chunk.content):
                problems.append(ChunkProblem(
                    id=f"problem_{chunk.id}_forward_ref",
                    chunk_id=chunk.id,
                    problem_type=ProblemType.REFERENCE_ORPHANING,
                    severity=Severity.MEDIUM,
                    description=f"Contains forward reference ('{pattern.pattern}') - referenced content may be in different chunk",
                    suggested_fix="Verify referenced content is retrievable or merge chunks",
                    metadata={"pattern": pattern.pattern},
                ))
                break

        # Backward references
        for pattern in self.backward_patterns:
            if pattern.search(chunk.content):
                problems.append(ChunkProblem(
                    id=f"problem_{chunk.id}_backward_ref",
                    chunk_id=chunk.id,
                    problem_type=ProblemType.REFERENCE_ORPHANING,
                    severity=Severity.LOW,
                    description=f"Contains backward reference ('{pattern.pattern}') - may need previous chunks for context",
                    suggested_fix="Ensure referenced content is in same chunk or clearly linked",
                    metadata={"pattern": pattern.pattern},
                ))
                break

        return problems

    def _check_structural_integrity(
        self,
        chunk: Chunk,
        prev_chunk: Chunk | None,
        next_chunk: Chunk | None,
    ) -> list[ChunkProblem]:
        """Detect split lists, tables, and procedures."""
        problems = []

        # Check for numbered lists
        list_pattern = re.compile(r'^\s*(\d+)[\.\)]\s+', re.MULTILINE)
        matches = list_pattern.findall(chunk.content)

        if matches:
            numbers = [int(m) for m in matches]
            min_num = min(numbers)
            max_num = max(numbers)

            # List starts mid-sequence (doesn't start at 1)
            if min_num > 1:
                problems.append(ChunkProblem(
                    id=f"problem_{chunk.id}_list_start",
                    chunk_id=chunk.id,
                    problem_type=ProblemType.STRUCTURAL_BREAKAGE,
                    severity=Severity.HIGH,
                    description=f"Numbered list starts at item {min_num}, not 1 - list beginning is in different chunk",
                    suggested_fix="Merge with previous chunk(s) to include complete list",
                    metadata={"list_range": f"{min_num}-{max_num}"},
                ))

            # List has gaps (missing numbers)
            expected_nums = set(range(min_num, max_num + 1))
            actual_nums = set(numbers)
            missing = expected_nums - actual_nums

            if missing:
                problems.append(ChunkProblem(
                    id=f"problem_{chunk.id}_list_gaps",
                    chunk_id=chunk.id,
                    problem_type=ProblemType.STRUCTURAL_BREAKAGE,
                    severity=Severity.MEDIUM,
                    description=f"Numbered list missing items: {sorted(missing)}",
                    suggested_fix="List items may be split across chunks - consider merging",
                    metadata={"missing_items": list(missing)},
                ))

        # Check for table markers (simple heuristic)
        table_markers = ['|', '\t\t', '────']
        if any(marker in chunk.content for marker in table_markers):
            # Count lines with markers
            lines = chunk.content.split('\n')
            marker_lines = sum(1 for line in lines if any(m in line for m in table_markers))

            # If less than 3 marker lines, table might be split
            if marker_lines < 3 and marker_lines > 0:
                problems.append(ChunkProblem(
                    id=f"problem_{chunk.id}_table_partial",
                    chunk_id=chunk.id,
                    problem_type=ProblemType.STRUCTURAL_BREAKAGE,
                    severity=Severity.MEDIUM,
                    description="Partial table detected - table may be split across chunks",
                    suggested_fix="Merge chunks to preserve complete table structure",
                    metadata={"marker_lines": marker_lines},
                ))

        return problems

    def get_statistics(self, problems: list[ChunkProblem]) -> dict[str, Any]:
        """Calculate diagnostic statistics."""
        total = len(problems)

        if total == 0:
            return {
                "total_problems": 0,
                "by_type": {},
                "by_severity": {},
                "problem_rate": 0.0,
            }

        by_type = {}
        for ptype in ProblemType:
            count = sum(1 for p in problems if p.problem_type == ptype)
            if count > 0:
                by_type[ptype.value] = count

        by_severity = {}
        for severity in Severity:
            count = sum(1 for p in problems if p.severity == severity)
            if count > 0:
                by_severity[severity.value] = count

        # Count unique chunks with problems
        unique_chunks = len(set(p.chunk_id for p in problems))

        return {
            "total_problems": total,
            "unique_chunks_with_problems": unique_chunks,
            "by_type": by_type,
            "by_severity": by_severity,
        }
