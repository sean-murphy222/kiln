"""
Tests for CHONK quality analysis.
"""

import pytest

from chonk.core.document import Block, Chunk, ChonkDocument, QualityScore
from chonk.utils.quality import QualityAnalyzer


class TestQualityAnalyzer:
    """Tests for QualityAnalyzer."""

    def test_analyzer_creation(self):
        """Test quality analyzer instantiation."""
        analyzer = QualityAnalyzer()
        assert analyzer is not None

    def test_analyze_chunk_basic(self, sample_document):
        """Test basic chunk quality analysis."""
        chunk = Chunk(
            id="test_chunk",
            block_ids=["block_1", "block_2"],
            content="This is a complete sentence. Here is another one. The content is well-formed.",
            token_count=20,
        )

        analyzer = QualityAnalyzer()
        score = analyzer.analyze_chunk(chunk, sample_document)

        assert isinstance(score, QualityScore)
        assert 0 <= score.overall <= 1
        assert 0 <= score.token_range <= 1
        assert 0 <= score.sentence_complete <= 1

    def test_analyze_chunk_token_range(self, sample_document):
        """Test token range scoring."""
        analyzer = QualityAnalyzer()

        # Good token count (within target range)
        good_chunk = Chunk(
            id="good",
            block_ids=["block_1"],
            content="Content " * 100,  # ~100 tokens
            token_count=100,
        )
        good_score = analyzer.analyze_chunk(good_chunk, sample_document)

        # Too small
        small_chunk = Chunk(
            id="small",
            block_ids=["block_1"],
            content="Small",
            token_count=1,
        )
        small_score = analyzer.analyze_chunk(small_chunk, sample_document)

        # Good chunks should score higher on token range
        assert good_score.token_range > small_score.token_range

    def test_analyze_chunk_sentence_complete(self, sample_document):
        """Test sentence completeness scoring."""
        analyzer = QualityAnalyzer()

        # Complete sentence
        complete = Chunk(
            id="complete",
            block_ids=["block_1"],
            content="This is a complete sentence. Here is another one.",
            token_count=10,
        )
        complete_score = analyzer.analyze_chunk(complete, sample_document)

        # Incomplete sentence
        incomplete = Chunk(
            id="incomplete",
            block_ids=["block_1"],
            content="This sentence is cut off in the middle of",
            token_count=10,
        )
        incomplete_score = analyzer.analyze_chunk(incomplete, sample_document)

        assert complete_score.sentence_complete >= incomplete_score.sentence_complete

    def test_analyze_document(self, sample_document):
        """Test document-level quality analysis."""
        # Add some chunks to the document
        sample_document.chunks = [
            Chunk(
                id="c1",
                block_ids=["block_1"],
                content="Good quality content. Well-formed sentences.",
                token_count=50,
            ),
            Chunk(
                id="c2",
                block_ids=["block_2"],
                content="Another chunk with proper content.",
                token_count=40,
            ),
            Chunk(
                id="c3",
                block_ids=["block_3"],
                content="tiny",  # Very small chunk
                token_count=1,
            ),
        ]

        analyzer = QualityAnalyzer()
        report = analyzer.analyze_document(sample_document)

        assert "total_chunks" in report
        assert "average_score" in report
        assert report["total_chunks"] == 3

    def test_get_improvement_suggestions(self, sample_document):
        """Test getting improvement suggestions."""
        # Create a problematic chunk
        chunk = Chunk(
            id="problem",
            block_ids=["block_1"],
            content="tiny",  # Very small
            token_count=1,
        )

        analyzer = QualityAnalyzer()
        suggestions = analyzer.get_improvement_suggestions(chunk, sample_document)

        assert len(suggestions) > 0
        # Should suggest merging or expanding the chunk
        assert any("small" in s.lower() or "merge" in s.lower() for s in suggestions)

    def test_quality_score_serialization(self):
        """Test quality score serialization."""
        score = QualityScore(
            token_range=0.8,
            sentence_complete=0.9,
            hierarchy_preserved=1.0,
            table_integrity=1.0,
            reference_complete=0.7,
            overall=0.88,
        )

        data = score.to_dict()
        restored = QualityScore.from_dict(data)

        assert restored.overall == score.overall
        assert restored.token_range == score.token_range


class TestQualityThresholds:
    """Tests for quality thresholds and categorization."""

    def test_good_quality_chunk(self, sample_document):
        """Test that well-formed chunks get good scores."""
        chunk = Chunk(
            id="good",
            block_ids=["block_1", "block_2"],
            content=(
                "This is a well-formed paragraph with complete sentences. "
                "It has proper structure and formatting. The content is coherent "
                "and self-contained. It provides useful information that can "
                "be retrieved independently."
            ),
            token_count=50,
        )

        analyzer = QualityAnalyzer()
        score = analyzer.analyze_chunk(chunk, sample_document)

        # Good chunks should have overall score >= 0.7
        assert score.overall >= 0.7

    def test_problem_detection(self, sample_document):
        """Test that problematic chunks are flagged."""
        # Create document with a problematic chunk
        sample_document.chunks = [
            Chunk(
                id="tiny",
                block_ids=["block_1"],
                content="Too small.",
                token_count=2,
            ),
        ]

        analyzer = QualityAnalyzer()
        report = analyzer.analyze_document(sample_document)

        # Should detect problems
        assert report["problem_count"] > 0 or report["warning_count"] > 0
