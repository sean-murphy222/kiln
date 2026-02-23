"""
Automatic fix strategies for detected chunk problems.

Each problem type has corresponding fix strategies:
- Semantic Incompleteness → Merge with adjacent chunks
- Semantic Contamination → Split at topic boundaries
- Structural Breakage → Merge to preserve structures
- Reference Orphaning → Merge with referenced content

Fixes can be previewed before application.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from chonk.core.document import Chunk
from chonk.diagnostics.analyzer import ChunkProblem, ProblemType
from chonk.utils.tokens import count_tokens


@dataclass
class FixAction:
    """A proposed fix action."""

    action_type: str  # "merge", "split", "reorder"
    chunk_ids: list[str]  # Chunks affected
    description: str
    confidence: float  # 0-1, how confident we are this will help
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "chunk_ids": self.chunk_ids,
            "description": self.description,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class FixResult:
    """Result of applying a fix."""

    success: bool
    chunks_before: int
    chunks_after: int
    actions_applied: list[FixAction]
    new_chunks: list[Chunk]
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "chunks_before": self.chunks_before,
            "chunks_after": self.chunks_after,
            "actions_applied": [a.to_dict() for a in self.actions_applied],
            "errors": self.errors,
        }


class FixStrategy(ABC):
    """Base class for fix strategies."""

    @abstractmethod
    def can_fix(self, problem: ChunkProblem, chunks: list[Chunk]) -> bool:
        """Check if this strategy can fix the given problem."""
        pass

    @abstractmethod
    def plan_fix(self, problem: ChunkProblem, chunks: list[Chunk]) -> FixAction | None:
        """Plan a fix action for the problem."""
        pass


class MergeAdjacentFix(FixStrategy):
    """
    Merge chunk with adjacent chunks.

    Used for:
    - Small fragments (< 20 tokens)
    - Dangling connectives (needs previous chunk)
    - Incomplete sentences (needs next chunk)
    - Split lists/tables
    """

    def can_fix(self, problem: ChunkProblem, chunks: list[Chunk]) -> bool:
        """Can fix semantic incompleteness and structural breakage."""
        return problem.problem_type in [
            ProblemType.SEMANTIC_INCOMPLETE,
            ProblemType.STRUCTURAL_BREAKAGE,
        ]

    def plan_fix(self, problem: ChunkProblem, chunks: list[Chunk]) -> FixAction | None:
        """Plan merge operation based on problem type."""

        # Find the chunk
        chunk_idx = None
        for i, chunk in enumerate(chunks):
            if chunk.id == problem.chunk_id:
                chunk_idx = i
                break

        if chunk_idx is None:
            return None

        chunk = chunks[chunk_idx]

        # Determine merge direction based on problem metadata
        merge_with_prev = False
        merge_with_next = False
        confidence = 0.8

        # Small chunks: try to merge with both neighbors
        if chunk.token_count < 20:
            # Prefer merging with next unless at end
            if chunk_idx < len(chunks) - 1:
                merge_with_next = True
            elif chunk_idx > 0:
                merge_with_prev = True
            else:
                return None  # Can't merge (only chunk)

        # Dangling connective: needs previous chunk
        elif "connective" in problem.metadata:
            if chunk_idx > 0:
                merge_with_prev = True
            else:
                return None

        # Incomplete ending: needs next chunk
        elif "last_char" in problem.metadata:
            if chunk_idx < len(chunks) - 1:
                merge_with_next = True
            else:
                return None

        # Lowercase start: needs previous chunk
        elif "start_char" in problem.metadata:
            if chunk_idx > 0:
                merge_with_prev = True
            else:
                return None

        # List split: merge with previous to get list start
        elif "list_range" in problem.metadata:
            if chunk_idx > 0:
                merge_with_prev = True
                confidence = 0.9  # High confidence for structural issues
            else:
                return None

        else:
            # Default: try next, then prev
            if chunk_idx < len(chunks) - 1:
                merge_with_next = True
            elif chunk_idx > 0:
                merge_with_prev = True
            else:
                return None

        # Build action
        if merge_with_prev:
            prev_chunk = chunks[chunk_idx - 1]
            return FixAction(
                action_type="merge",
                chunk_ids=[prev_chunk.id, chunk.id],
                description=f"Merge '{chunk.id}' with previous chunk to resolve {problem.problem_type.value}",
                confidence=confidence,
                metadata={"direction": "prev", "problem_id": problem.id},
            )
        elif merge_with_next:
            next_chunk = chunks[chunk_idx + 1]
            return FixAction(
                action_type="merge",
                chunk_ids=[chunk.id, next_chunk.id],
                description=f"Merge '{chunk.id}' with next chunk to resolve {problem.problem_type.value}",
                confidence=confidence,
                metadata={"direction": "next", "problem_id": problem.id},
            )

        return None


class SplitLargeChunkFix(FixStrategy):
    """
    Split large chunks at natural boundaries.

    Used for:
    - Semantic contamination (>500 tokens, multiple topics)

    Split strategies:
    1. Paragraph boundaries (double newline)
    2. Section breaks (heading patterns)
    3. Topic shifts (if embeddings available - future)
    """

    def can_fix(self, problem: ChunkProblem, chunks: list[Chunk]) -> bool:
        """Can fix semantic contamination."""
        return problem.problem_type == ProblemType.SEMANTIC_CONTAMINATION

    def plan_fix(self, problem: ChunkProblem, chunks: list[Chunk]) -> FixAction | None:
        """Plan split operation."""

        # Find the chunk
        chunk = None
        for c in chunks:
            if c.id == problem.chunk_id:
                chunk = c
                break

        if chunk is None or chunk.token_count < 500:
            return None

        # Strategy 1: Split at paragraph boundaries
        paragraphs = chunk.content.split('\n\n')

        if len(paragraphs) >= 3:  # Need at least 3 paragraphs to split meaningfully
            # Find best split point (aim for ~40-60% through content)
            target_tokens = chunk.token_count // 2
            current_tokens = 0
            split_idx = 1  # Start after first paragraph

            for i, para in enumerate(paragraphs[:-1]):  # Don't split at last paragraph
                current_tokens += count_tokens(para)
                if current_tokens >= target_tokens * 0.4:  # Between 40-60%
                    split_idx = i + 1
                    break

            return FixAction(
                action_type="split",
                chunk_ids=[chunk.id],
                description=f"Split large chunk ({chunk.token_count} tokens) at paragraph boundary",
                confidence=0.7,
                metadata={
                    "split_index": split_idx,
                    "split_type": "paragraph",
                    "problem_id": problem.id,
                },
            )

        # Strategy 2: Split at heading patterns (if present)
        heading_pattern = re.compile(r'^#+\s+|^\d+\.\d+\s+|^[A-Z][A-Z\s]+:?$', re.MULTILINE)
        matches = list(heading_pattern.finditer(chunk.content))

        if len(matches) >= 2:  # Multiple headings found
            # Find middle heading
            mid_idx = len(matches) // 2
            split_pos = matches[mid_idx].start()

            return FixAction(
                action_type="split",
                chunk_ids=[chunk.id],
                description=f"Split large chunk ({chunk.token_count} tokens) at section heading",
                confidence=0.8,
                metadata={
                    "split_position": split_pos,
                    "split_type": "heading",
                    "problem_id": problem.id,
                },
            )

        # Strategy 3: Simple midpoint split (last resort)
        return FixAction(
            action_type="split",
            chunk_ids=[chunk.id],
            description=f"Split large chunk ({chunk.token_count} tokens) at midpoint",
            confidence=0.5,  # Lower confidence - not ideal
            metadata={
                "split_position": len(chunk.content) // 2,
                "split_type": "midpoint",
                "problem_id": problem.id,
            },
        )


class MergeReferencedChunksFix(FixStrategy):
    """
    Merge chunks with orphaned references.

    For now, this is conservative - only suggest merge, don't auto-apply
    since determining the referenced chunk requires deeper analysis.
    """

    def can_fix(self, problem: ChunkProblem, chunks: list[Chunk]) -> bool:
        """Can suggest fixes for reference orphaning."""
        return problem.problem_type == ProblemType.REFERENCE_ORPHANING

    def plan_fix(self, problem: ChunkProblem, chunks: list[Chunk]) -> FixAction | None:
        """Plan merge with referenced content (conservative)."""

        # Find the chunk
        chunk_idx = None
        for i, chunk in enumerate(chunks):
            if chunk.id == problem.chunk_id:
                chunk_idx = i
                break

        if chunk_idx is None:
            return None

        chunk = chunks[chunk_idx]

        # For forward references ("as follows", "below"), merge with next
        if "forward_ref" in problem.id:
            if chunk_idx < len(chunks) - 1:
                next_chunk = chunks[chunk_idx + 1]
                return FixAction(
                    action_type="merge",
                    chunk_ids=[chunk.id, next_chunk.id],
                    description=f"Merge chunk with next to resolve forward reference",
                    confidence=0.6,  # Medium confidence - reference might be further away
                    metadata={"reference_type": "forward", "problem_id": problem.id},
                )

        # For backward references ("as mentioned", "above"), merge with previous
        elif "backward_ref" in problem.id:
            if chunk_idx > 0:
                prev_chunk = chunks[chunk_idx - 1]
                return FixAction(
                    action_type="merge",
                    chunk_ids=[prev_chunk.id, chunk.id],
                    description=f"Merge chunk with previous to resolve backward reference",
                    confidence=0.6,
                    metadata={"reference_type": "backward", "problem_id": problem.id},
                )

        return None


# Registry of fix strategies
FIX_STRATEGIES: list[FixStrategy] = [
    MergeAdjacentFix(),
    SplitLargeChunkFix(),
    MergeReferencedChunksFix(),
]


def find_fix_strategy(problem: ChunkProblem, chunks: list[Chunk]) -> FixAction | None:
    """Find and plan a fix for the given problem."""
    for strategy in FIX_STRATEGIES:
        if strategy.can_fix(problem, chunks):
            action = strategy.plan_fix(problem, chunks)
            if action:
                return action
    return None
