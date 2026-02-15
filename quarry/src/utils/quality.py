"""
Chunk quality analysis.

Scores chunks based on various quality metrics to help users
identify potentially problematic chunks.
"""

from __future__ import annotations

import re
from typing import Any

from chonk.core.document import Block, BlockType, Chunk, ChonkDocument, QualityScore
from chonk.utils.tokens import TokenCounter


class QualityAnalyzer:
    """
    Analyze chunk quality and compute scores.

    Quality is scored across multiple dimensions:
    - Token count (is it in optimal range?)
    - Sentence completeness (proper start/end?)
    - Hierarchy preservation (no orphan headings?)
    - Table integrity (tables not split?)
    - Reference completeness (no orphan references?)
    """

    # Optimal token range
    OPTIMAL_MIN = 200
    OPTIMAL_MAX = 500
    WARN_MIN = 100
    WARN_MAX = 600

    # Patterns for reference detection
    ORPHAN_REFERENCE_PATTERNS = [
        r"\bsee above\b",
        r"\bsee below\b",
        r"\bas shown above\b",
        r"\bas shown below\b",
        r"\bas mentioned\b",
        r"\bthe following\b$",  # At end of chunk
        r"^the above\b",  # At start of chunk
        r"\brefer to\s*$",
    ]

    def __init__(self) -> None:
        self._token_counter = TokenCounter()

    def analyze_chunk(self, chunk: Chunk, document: ChonkDocument) -> QualityScore:
        """
        Analyze a single chunk and compute quality scores.

        Returns:
            QualityScore with all components filled
        """
        blocks = document.get_blocks_for_chunk(chunk)

        return QualityScore(
            token_range=self._score_token_range(chunk.token_count),
            sentence_complete=self._score_sentence_completeness(chunk.content),
            hierarchy_preserved=self._score_hierarchy(blocks),
            table_integrity=self._score_table_integrity(blocks),
            reference_complete=self._score_references(chunk.content),
        )

    def analyze_document(self, document: ChonkDocument) -> dict[str, Any]:
        """
        Analyze all chunks in a document.

        Returns summary statistics and per-chunk scores.
        """
        chunk_scores = []
        total_score = 0.0

        for chunk in document.chunks:
            score = self.analyze_chunk(chunk, document)
            chunk.quality = score  # Update chunk in place
            chunk_scores.append(
                {
                    "chunk_id": chunk.id,
                    "score": score.overall,
                    "details": score.to_dict(),
                }
            )
            total_score += score.overall

        avg_score = total_score / len(document.chunks) if document.chunks else 0

        # Find problem chunks
        problems = [cs for cs in chunk_scores if cs["score"] < 0.7]
        warnings = [cs for cs in chunk_scores if 0.7 <= cs["score"] < 0.85]

        return {
            "total_chunks": len(document.chunks),
            "average_score": round(avg_score, 3),
            "problem_count": len(problems),
            "warning_count": len(warnings),
            "problems": problems,
            "warnings": warnings,
            "all_scores": chunk_scores,
        }

    def _score_token_range(self, token_count: int) -> float:
        """Score based on whether token count is in optimal range."""
        if self.OPTIMAL_MIN <= token_count <= self.OPTIMAL_MAX:
            return 1.0

        if token_count < self.WARN_MIN:
            # Too short - linear penalty
            return max(0.3, token_count / self.WARN_MIN)

        if token_count < self.OPTIMAL_MIN:
            # Below optimal but not too short
            return 0.8

        if token_count <= self.WARN_MAX:
            # Above optimal but not too long
            return 0.85

        # Too long - gradual penalty
        excess = token_count - self.WARN_MAX
        penalty = min(0.5, excess / 500)
        return max(0.3, 1.0 - penalty)

    def _score_sentence_completeness(self, content: str) -> float:
        """Score based on whether chunk starts and ends properly."""
        content = content.strip()
        if not content:
            return 1.0

        score = 1.0

        # Check start
        first_char = content[0]
        if not first_char.isupper() and first_char not in "\"'([":
            score -= 0.3

        # Check end
        last_char = content[-1]
        if last_char not in ".!?:\"')]":
            score -= 0.3

        # Check for sentence fragments at boundaries
        if content.startswith(("and ", "or ", "but ", "however, ", "therefore, ")):
            score -= 0.2

        return max(0.3, score)

    def _score_hierarchy(self, blocks: list[Block]) -> float:
        """Score based on hierarchy preservation."""
        if not blocks:
            return 1.0

        # Check for orphan heading at end of chunk
        last_block = blocks[-1]
        if last_block.type == BlockType.HEADING:
            return 0.4  # Heading should be followed by content

        # Check for heading without content
        has_heading = any(b.type == BlockType.HEADING for b in blocks)
        has_content = any(b.type in (BlockType.TEXT, BlockType.LIST, BlockType.TABLE) for b in blocks)

        if has_heading and not has_content:
            return 0.5

        return 1.0

    def _score_table_integrity(self, blocks: list[Block]) -> float:
        """Score based on table handling."""
        # For now, just check if we have partial tables
        # This would need more sophisticated detection in practice

        for block in blocks:
            if block.type == BlockType.TABLE:
                content = block.content
                # Check for signs of truncation
                lines = content.split("\n")
                if len(lines) >= 2:
                    # Check if header and separator exist
                    if "|" not in lines[0] or "---" not in lines[1]:
                        return 0.7

        return 1.0

    def _score_references(self, content: str) -> float:
        """Score based on orphan references."""
        content_lower = content.lower()

        for pattern in self.ORPHAN_REFERENCE_PATTERNS:
            if re.search(pattern, content_lower):
                return 0.6

        return 1.0

    def get_improvement_suggestions(
        self, chunk: Chunk, document: ChonkDocument
    ) -> list[str]:
        """Get suggestions for improving a chunk."""
        suggestions = []
        score = chunk.quality

        if score.token_range < 0.8:
            if chunk.token_count < self.OPTIMAL_MIN:
                suggestions.append(
                    f"Consider merging with adjacent chunks (only {chunk.token_count} tokens)"
                )
            else:
                suggestions.append(
                    f"Consider splitting this chunk ({chunk.token_count} tokens)"
                )

        if score.sentence_complete < 0.8:
            content = chunk.content.strip()
            if content and not content[0].isupper():
                suggestions.append("Chunk starts mid-sentence - consider adjusting boundary")
            if content and content[-1] not in ".!?:\"')]":
                suggestions.append("Chunk ends mid-sentence - consider adjusting boundary")

        if score.hierarchy_preserved < 0.8:
            blocks = document.get_blocks_for_chunk(chunk)
            if blocks and blocks[-1].type == BlockType.HEADING:
                suggestions.append("Heading at end of chunk - consider including following content")

        if score.reference_complete < 0.8:
            suggestions.append(
                "Chunk contains orphan references - consider including referenced content"
            )

        return suggestions
