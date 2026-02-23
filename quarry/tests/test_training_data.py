"""Tests for synthetic training corpus generation."""

from __future__ import annotations

import numpy as np

from chonk.tier1.fingerprinter import DocumentFingerprint
from chonk.tier1.taxonomy import TRAINABLE_TYPES, DocumentType
from chonk.tier1.training_data import TrainingCorpus, generate_training_corpus


class TestTrainingCorpus:
    """Tests for TrainingCorpus dataclass."""

    def test_serialization_roundtrip(self):
        """Corpus can be serialized and deserialized."""
        corpus = generate_training_corpus(samples_per_type=5, random_seed=1)
        d = corpus.to_dict()
        restored = TrainingCorpus.from_dict(d)
        assert np.allclose(restored.feature_matrix, corpus.feature_matrix)
        assert restored.labels == corpus.labels
        assert restored.feature_names == corpus.feature_names


class TestGenerateTrainingCorpus:
    """Tests for generate_training_corpus function."""

    def test_default_shape(self):
        """Default corpus has shape (560, 49): 40 samples x 14 types."""
        corpus = generate_training_corpus()
        assert corpus.feature_matrix.shape == (560, 49)

    def test_custom_samples_per_type(self):
        """Custom samples_per_type changes corpus size."""
        corpus = generate_training_corpus(samples_per_type=10)
        assert corpus.feature_matrix.shape == (140, 49)

    def test_labels_match_rows(self):
        """Labels list has same length as feature matrix rows."""
        corpus = generate_training_corpus(samples_per_type=20)
        assert len(corpus.labels) == corpus.feature_matrix.shape[0]

    def test_all_labels_valid(self):
        """All labels are valid DocumentType values (not UNKNOWN)."""
        corpus = generate_training_corpus()
        valid_values = {dt.value for dt in TRAINABLE_TYPES}
        for label in corpus.labels:
            assert label in valid_values, f"Invalid label: {label}"

    def test_all_types_represented(self):
        """Every trainable type has samples in the corpus."""
        corpus = generate_training_corpus()
        label_set = set(corpus.labels)
        for dt in TRAINABLE_TYPES:
            assert dt.value in label_set, f"Missing type: {dt.value}"

    def test_feature_names_match_fingerprint(self):
        """Feature names match DocumentFingerprint.feature_names()."""
        corpus = generate_training_corpus()
        expected = DocumentFingerprint().feature_names()
        assert corpus.feature_names == expected

    def test_ratios_clamped(self):
        """Ratio features are clamped to [0, 1]."""
        corpus = generate_training_corpus(samples_per_type=100, random_seed=99)
        ratio_cols = [
            i
            for i, name in enumerate(corpus.feature_names)
            if "ratio" in name
            or name
            in {
                "byte_has_metadata",
                "byte_has_xmp_metadata",
                "byte_encrypted",
                "byte_has_acroform",
                "rep_has_page_numbers",
                "rep_has_headers",
                "rep_has_footers",
                "rhythm_has_toc",
            }
        ]
        for col in ratio_cols:
            values = corpus.feature_matrix[:, col]
            assert np.all(values >= 0.0), (
                f"Feature {corpus.feature_names[col]} has negative values"
            )
            assert np.all(values <= 1.0), (
                f"Feature {corpus.feature_names[col]} has values > 1"
            )

    def test_counts_nonnegative(self):
        """Count features are non-negative."""
        corpus = generate_training_corpus(samples_per_type=100, random_seed=99)
        count_cols = [
            i
            for i, name in enumerate(corpus.feature_names)
            if "count" in name or "depth" in name or "levels" in name
        ]
        for col in count_cols:
            values = corpus.feature_matrix[:, col]
            assert np.all(values >= 0.0), (
                f"Feature {corpus.feature_names[col]} has negative values"
            )

    def test_reproducible_with_same_seed(self):
        """Same seed produces identical corpus."""
        c1 = generate_training_corpus(random_seed=42)
        c2 = generate_training_corpus(random_seed=42)
        assert np.array_equal(c1.feature_matrix, c2.feature_matrix)
        assert c1.labels == c2.labels

    def test_different_seeds_differ(self):
        """Different seeds produce different corpus."""
        c1 = generate_training_corpus(random_seed=1)
        c2 = generate_training_corpus(random_seed=2)
        assert not np.array_equal(c1.feature_matrix, c2.feature_matrix)

    def test_samples_per_type_counts(self):
        """samples_per_type dict has correct counts."""
        corpus = generate_training_corpus(samples_per_type=25)
        for dt in TRAINABLE_TYPES:
            assert corpus.samples_per_type[dt.value] == 25

    def test_no_nan_or_inf(self):
        """No NaN or Inf in generated data."""
        corpus = generate_training_corpus()
        assert not np.any(np.isnan(corpus.feature_matrix))
        assert not np.any(np.isinf(corpus.feature_matrix))
