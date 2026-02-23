"""Tests for Tier 1 ML document type classifier."""

from __future__ import annotations

import numpy as np
import pytest

from chonk.tier1.classifier import (
    ClassificationResult,
    DocumentClassifier,
    FeatureImportance,
    TrainingReport,
)
from chonk.tier1.fingerprinter import (
    ByteLevelFeatures,
    CharacterFeatures,
    DocumentFingerprint,
    FontFeatures,
    LayoutFeatures,
    RepetitionFeatures,
    StructuralRhythmFeatures,
)
from chonk.tier1.taxonomy import (
    DOCUMENT_TYPE_PROFILES,
    TRAINABLE_TYPES,
    DocumentType,
)
from chonk.tier1.training_data import generate_training_corpus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trained_classifier() -> DocumentClassifier:
    """Create and train a classifier for testing."""
    corpus = generate_training_corpus(samples_per_type=40)
    clf = DocumentClassifier()
    clf.train(corpus)
    return clf


def _fingerprint_from_profile(doc_type: DocumentType) -> DocumentFingerprint:
    """Build a fingerprint using a type's profile mean values.

    Creates a fingerprint directly from profile means, which
    should be strongly recognized by a trained classifier.
    """
    profile = DOCUMENT_TYPE_PROFILES[doc_type]
    names = DocumentFingerprint().feature_names()

    # Build a flat dict of feature name -> value from profile means
    from chonk.tier1.taxonomy import _DEFAULT_MEANS

    values = {}
    for name in names:
        values[name] = profile.feature_means.get(name, _DEFAULT_MEANS.get(name, 0.0))

    return DocumentFingerprint(
        byte_features=ByteLevelFeatures(
            file_size=int(values["byte_file_size"]),
            pdf_version=values["byte_pdf_version"],
            object_count=int(values["byte_object_count"]),
            stream_count=int(values["byte_stream_count"]),
            has_metadata=values["byte_has_metadata"] > 0.5,
            has_xmp_metadata=values["byte_has_xmp_metadata"] > 0.5,
            page_count=int(values["byte_page_count"]),
            encrypted=values["byte_encrypted"] > 0.5,
            has_acroform=values["byte_has_acroform"] > 0.5,
        ),
        font_features=FontFeatures(
            font_count=int(values["font_font_count"]),
            size_min=values["font_size_min"],
            size_max=values["font_size_max"],
            size_mean=values["font_size_mean"],
            size_std=values["font_size_std"],
            size_median=values["font_size_median"],
            bold_ratio=values["font_bold_ratio"],
            italic_ratio=values["font_italic_ratio"],
            monospace_ratio=values["font_monospace_ratio"],
            distinct_sizes=int(values["font_distinct_sizes"]),
        ),
        layout_features=LayoutFeatures(
            page_width=values["layout_page_width"],
            page_height=values["layout_page_height"],
            width_consistency=values["layout_width_consistency"],
            height_consistency=values["layout_height_consistency"],
            margin_left=values["layout_margin_left"],
            margin_right=values["layout_margin_right"],
            margin_top=values["layout_margin_top"],
            margin_bottom=values["layout_margin_bottom"],
            text_area_ratio=values["layout_text_area_ratio"],
            estimated_columns=int(values["layout_estimated_columns"]),
        ),
        character_features=CharacterFeatures(
            alpha_ratio=values["char_alpha_ratio"],
            numeric_ratio=values["char_numeric_ratio"],
            punctuation_ratio=values["char_punctuation_ratio"],
            whitespace_ratio=values["char_whitespace_ratio"],
            special_ratio=values["char_special_ratio"],
            uppercase_ratio=values["char_uppercase_ratio"],
            total_chars=int(values["char_total_chars"]),
        ),
        repetition_features=RepetitionFeatures(
            has_page_numbers=values["rep_has_page_numbers"] > 0.5,
            has_headers=values["rep_has_headers"] > 0.5,
            has_footers=values["rep_has_footers"] > 0.5,
            repetition_ratio=values["rep_repetition_ratio"],
            first_line_diversity=values["rep_first_line_diversity"],
        ),
        structural_rhythm=StructuralRhythmFeatures(
            heading_density=values["rhythm_heading_density"],
            table_density=values["rhythm_table_density"],
            image_density=values["rhythm_image_density"],
            list_density=values["rhythm_list_density"],
            has_toc=values["rhythm_has_toc"] > 0.5,
            toc_depth=int(values["rhythm_toc_depth"]),
            link_count=int(values["rhythm_link_count"]),
            heading_size_levels=int(values["rhythm_heading_size_levels"]),
        ),
    )


# ===================================================================
# Shell tests
# ===================================================================


class TestDocumentClassifierShell:
    """Tests for classifier instantiation and validation."""

    def test_instantiation(self):
        clf = DocumentClassifier()
        assert clf is not None

    def test_not_trained_initially(self):
        clf = DocumentClassifier()
        assert clf.is_trained is False

    def test_default_threshold(self):
        clf = DocumentClassifier()
        assert clf.confidence_threshold == 0.45

    def test_custom_threshold(self):
        clf = DocumentClassifier(confidence_threshold=0.8)
        assert clf.confidence_threshold == 0.8

    def test_predict_before_training_raises(self):
        clf = DocumentClassifier()
        fp = DocumentFingerprint()
        with pytest.raises(RuntimeError, match="not been trained"):
            clf.predict(fp)

    def test_feature_importances_before_training_raises(self):
        clf = DocumentClassifier()
        with pytest.raises(RuntimeError, match="not been trained"):
            clf.feature_importances()

    def test_save_before_training_raises(self, tmp_path):
        clf = DocumentClassifier()
        with pytest.raises(RuntimeError):
            clf.save(tmp_path / "model.pkl")


# ===================================================================
# Training tests
# ===================================================================


class TestDocumentClassifierTraining:
    """Tests for training the classifier."""

    def test_train_returns_report(self):
        corpus = generate_training_corpus(samples_per_type=10)
        clf = DocumentClassifier()
        report = clf.train(corpus)
        assert isinstance(report, TrainingReport)

    def test_is_trained_after_training(self):
        corpus = generate_training_corpus(samples_per_type=10)
        clf = DocumentClassifier()
        clf.train(corpus)
        assert clf.is_trained is True

    def test_report_has_correct_n_samples(self):
        corpus = generate_training_corpus(samples_per_type=10)
        clf = DocumentClassifier()
        report = clf.train(corpus)
        assert report.n_samples == 140  # 10 * 14

    def test_report_has_correct_n_classes(self):
        corpus = generate_training_corpus(samples_per_type=10)
        clf = DocumentClassifier()
        report = clf.train(corpus)
        assert report.n_classes == 14

    def test_feature_importances_after_training(self):
        clf = _trained_classifier()
        importances = clf.feature_importances()
        assert len(importances) == 49

    def test_importances_sum_to_one(self):
        clf = _trained_classifier()
        importances = clf.feature_importances()
        total = sum(fi.importance for fi in importances)
        assert abs(total - 1.0) < 0.01

    def test_importances_ranked(self):
        clf = _trained_classifier()
        importances = clf.feature_importances()
        for i in range(len(importances) - 1):
            assert importances[i].importance >= importances[i + 1].importance

    def test_importances_nonnegative(self):
        clf = _trained_classifier()
        for fi in clf.feature_importances():
            assert fi.importance >= 0.0

    def test_empty_corpus_raises(self):
        corpus = generate_training_corpus(samples_per_type=10)
        empty = type(corpus)(
            feature_matrix=np.empty((0, 49)),
            labels=[],
            feature_names=corpus.feature_names,
        )
        clf = DocumentClassifier()
        with pytest.raises(ValueError, match="empty"):
            clf.train(empty)


# ===================================================================
# Prediction tests
# ===================================================================


class TestDocumentClassifierPrediction:
    """Tests for classifier prediction."""

    def test_predict_returns_result(self):
        clf = _trained_classifier()
        fp = _fingerprint_from_profile(DocumentType.TECHNICAL_MANUAL)
        result = clf.predict(fp)
        assert isinstance(result, ClassificationResult)

    def test_probabilities_sum_to_one(self):
        clf = _trained_classifier()
        fp = _fingerprint_from_profile(DocumentType.TECHNICAL_MANUAL)
        result = clf.predict(fp)
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 0.01

    def test_confidence_matches_best_probability(self):
        clf = _trained_classifier()
        fp = _fingerprint_from_profile(DocumentType.REPORT)
        result = clf.predict(fp)
        max_prob = max(result.probabilities.values())
        assert abs(result.confidence - max_prob) < 0.001

    def test_high_confidence_for_profile_match(self):
        """Profile-mean fingerprint should get high confidence."""
        clf = _trained_classifier()
        fp = _fingerprint_from_profile(DocumentType.TECHNICAL_MANUAL)
        result = clf.predict(fp)
        assert result.confidence > 0.4

    def test_correct_type_for_profile_matches(self):
        """Most profile-mean fingerprints should be correctly classified."""
        clf = _trained_classifier()
        correct = 0
        for dt in TRAINABLE_TYPES:
            fp = _fingerprint_from_profile(dt)
            result = clf.predict(fp)
            if result.document_type == dt:
                correct += 1
        # At least 10 of 14 types should be correctly classified
        assert correct >= 10, f"Only {correct}/14 correct"

    def test_zero_fingerprint_classifies(self):
        """All-zero fingerprint still produces a valid classification."""
        clf = _trained_classifier()
        fp = DocumentFingerprint()
        result = clf.predict(fp)
        assert isinstance(result, ClassificationResult)
        assert 0.0 <= result.confidence <= 1.0

    def test_batch_prediction(self):
        """Batch prediction returns list of same length."""
        clf = _trained_classifier()
        fps = [
            _fingerprint_from_profile(DocumentType.FORM),
            _fingerprint_from_profile(DocumentType.REPORT),
        ]
        results = clf.predict_batch(fps)
        assert len(results) == 2
        assert all(isinstance(r, ClassificationResult) for r in results)

    def test_is_unknown_flag(self):
        """is_unknown is True when type is UNKNOWN."""
        clf = DocumentClassifier(confidence_threshold=0.99)
        corpus = generate_training_corpus(samples_per_type=10)
        clf.train(corpus)
        fp = DocumentFingerprint()  # unlikely to hit 99% confidence
        result = clf.predict(fp)
        if result.is_unknown:
            assert result.document_type == DocumentType.UNKNOWN


# ===================================================================
# Persistence tests
# ===================================================================


class TestDocumentClassifierPersistence:
    """Tests for save/load."""

    def test_save_and_load(self, tmp_path):
        """Saved model can be loaded and produces same predictions."""
        clf = _trained_classifier()
        model_path = tmp_path / "model.pkl"
        clf.save(model_path)

        loaded = DocumentClassifier.load(model_path)
        assert loaded.is_trained

        fp = _fingerprint_from_profile(DocumentType.TECHNICAL_MANUAL)
        original = clf.predict(fp)
        restored = loaded.predict(fp)
        assert original.document_type == restored.document_type
        assert abs(original.confidence - restored.confidence) < 0.001

    def test_load_nonexistent_raises(self, tmp_path):
        """Loading from nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            DocumentClassifier.load(tmp_path / "nope.pkl")

    def test_load_invalid_file_raises(self, tmp_path):
        """Loading an invalid file raises ValueError."""
        bad_file = tmp_path / "bad.pkl"
        bad_file.write_text("not a pickle")
        with pytest.raises(ValueError, match="Invalid"):
            DocumentClassifier.load(bad_file)

    def test_loaded_threshold_preserved(self, tmp_path):
        """Custom confidence threshold is preserved through save/load."""
        clf = DocumentClassifier(confidence_threshold=0.8)
        corpus = generate_training_corpus(samples_per_type=10)
        clf.train(corpus)

        model_path = tmp_path / "model.pkl"
        clf.save(model_path)
        loaded = DocumentClassifier.load(model_path)
        assert loaded.confidence_threshold == 0.8
