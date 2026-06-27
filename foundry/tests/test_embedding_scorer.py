"""Tests for foundry.src.evaluation.EmbeddingSimilarityScorer.

The encoder is faked so the cosine logic, clamping, and empty-input handling
are verified without loading a real sentence-transformer model.
"""

from __future__ import annotations

import numpy as np
import pytest

from foundry.src.evaluation import EmbeddingSimilarityScorer, SimilarityScorer


class _FakeEncoder:
    """Returns pre-mapped normalized vectors for known texts."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping
        self.calls = 0

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        self.calls += 1
        return np.array([self.mapping[t] for t in texts], dtype=float)


def _scorer(mapping: dict[str, list[float]]) -> EmbeddingSimilarityScorer:
    return EmbeddingSimilarityScorer(model=_FakeEncoder(mapping))


def test_protocol_conformance() -> None:
    assert isinstance(EmbeddingSimilarityScorer(), SimilarityScorer)


def test_identical_vectors_score_one() -> None:
    scorer = _scorer({"a": [1.0, 0.0], "b": [1.0, 0.0]})
    assert scorer.score("a", "b") == pytest.approx(1.0)


def test_orthogonal_vectors_score_zero() -> None:
    scorer = _scorer({"a": [1.0, 0.0], "b": [0.0, 1.0]})
    assert scorer.score("a", "b") == pytest.approx(0.0)


def test_opposite_vectors_clamped_to_zero() -> None:
    scorer = _scorer({"a": [1.0, 0.0], "b": [-1.0, 0.0]})
    assert scorer.score("a", "b") == 0.0


def test_partial_similarity_in_unit_range() -> None:
    v = 1.0 / np.sqrt(2)
    scorer = _scorer({"a": [1.0, 0.0], "b": [v, v]})
    result = scorer.score("a", "b")
    assert 0.0 < result < 1.0


def test_empty_inputs_score_zero_without_model_call() -> None:
    encoder = _FakeEncoder({})
    scorer = EmbeddingSimilarityScorer(model=encoder)
    assert scorer.score("", "something") == 0.0
    assert scorer.score("something", "   ") == 0.0
    assert encoder.calls == 0


def test_lazy_model_construction_uses_injected_model() -> None:
    encoder = _FakeEncoder({"x": [1.0, 0.0], "y": [1.0, 0.0]})
    scorer = EmbeddingSimilarityScorer(model=encoder)
    scorer.score("x", "y")
    assert encoder.calls == 1
