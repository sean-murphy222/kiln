"""Tests for RetrainingService."""

from __future__ import annotations

from pathlib import Path

from chonk.tier1.fingerprinter import DocumentFingerprint
from chonk.tier1.manual_store import ManualExample, ManualLabelStore
from chonk.tier1.retraining import RetrainingService
from chonk.tier1.taxonomy import DocumentType

# ===================================================================
# Helpers
# ===================================================================


def _make_example(
    doc_type: DocumentType = DocumentType.TECHNICAL_MANUAL,
) -> ManualExample:
    return ManualExample.create(
        fingerprint=DocumentFingerprint(),
        document_type=doc_type,
        original_confidence=0.25,
        source_document_name="test.pdf",
    )


def _make_service(tmp_path: Path) -> tuple[RetrainingService, ManualLabelStore]:
    store = ManualLabelStore(tmp_path / "labels.jsonl")
    model_path = tmp_path / "model.pkl"
    service = RetrainingService(
        store=store,
        model_path=model_path,
        min_new_examples=5,
        synthetic_samples_per_type=10,  # Small for fast tests
    )
    return service, store


# ===================================================================
# build_combined_corpus
# ===================================================================


class TestBuildCombinedCorpus:
    """Tests for corpus merging."""

    def test_synthetic_only_when_store_empty(self, tmp_path):
        service, store = _make_service(tmp_path)
        corpus = service.build_combined_corpus()
        # 10 samples * 14 types = 140
        assert corpus.feature_matrix.shape[0] == 140

    def test_includes_manual_examples(self, tmp_path):
        service, store = _make_service(tmp_path)
        store.add(_make_example())
        store.add(_make_example(DocumentType.FORM))
        corpus = service.build_combined_corpus()
        assert corpus.feature_matrix.shape[0] == 142  # 140 synthetic + 2 manual

    def test_combined_feature_count_correct(self, tmp_path):
        service, store = _make_service(tmp_path)
        store.add(_make_example())
        corpus = service.build_combined_corpus()
        assert corpus.feature_matrix.shape[1] == 49

    def test_combined_labels_contain_both(self, tmp_path):
        service, store = _make_service(tmp_path)
        store.add(_make_example(DocumentType.FORM))
        corpus = service.build_combined_corpus()
        # Manual label should be present
        assert "form" in corpus.labels
        # Synthetic labels should be present
        assert "technical_manual" in corpus.labels

    def test_no_unknown_labels(self, tmp_path):
        service, store = _make_service(tmp_path)
        store.add(_make_example())
        corpus = service.build_combined_corpus()
        assert "unknown" not in corpus.labels


# ===================================================================
# should_retrain
# ===================================================================


class TestShouldRetrain:
    """Tests for retrain policy."""

    def test_true_when_no_model(self, tmp_path):
        service, store = _make_service(tmp_path)
        assert service.should_retrain() is True

    def test_false_when_recently_retrained(self, tmp_path):
        service, store = _make_service(tmp_path)
        # Retrain to create model and meta
        service.retrain(force=True)
        assert service.should_retrain() is False

    def test_false_with_four_new_examples(self, tmp_path):
        service, store = _make_service(tmp_path)
        service.retrain(force=True)
        for _ in range(4):
            store.add(_make_example())
        assert service.should_retrain() is False

    def test_true_after_five_new_examples(self, tmp_path):
        service, store = _make_service(tmp_path)
        service.retrain(force=True)
        for _ in range(5):
            store.add(_make_example())
        assert service.should_retrain() is True


# ===================================================================
# retrain
# ===================================================================


class TestRetrain:
    """Tests for retrain execution."""

    def test_retrain_returns_result(self, tmp_path):
        service, store = _make_service(tmp_path)
        result = service.retrain(force=True)
        assert result.success is True

    def test_retrain_saves_model(self, tmp_path):
        service, store = _make_service(tmp_path)
        result = service.retrain(force=True)
        assert result.model_path is not None
        assert result.model_path.exists()

    def test_retrain_skipped_when_not_warranted(self, tmp_path):
        service, store = _make_service(tmp_path)
        service.retrain(force=True)
        result = service.retrain(force=False)
        assert result.success is False
        assert result.skipped_reason == "not_enough_new_examples"

    def test_force_overrides_skip(self, tmp_path):
        service, store = _make_service(tmp_path)
        service.retrain(force=True)
        result = service.retrain(force=True)
        assert result.success is True

    def test_report_after_populated(self, tmp_path):
        service, store = _make_service(tmp_path)
        result = service.retrain(force=True)
        assert result.report_after is not None
        assert result.report_after.overall_accuracy > 0

    def test_meta_file_updated(self, tmp_path):
        service, store = _make_service(tmp_path)
        store.add(_make_example())
        service.retrain(force=True)
        assert service._meta_path.exists()

    def test_n_synthetic_and_manual_counts(self, tmp_path):
        service, store = _make_service(tmp_path)
        store.add(_make_example())
        store.add(_make_example(DocumentType.FORM))
        result = service.retrain(force=True)
        assert result.n_manual_examples == 2
        assert result.n_synthetic_examples == 140
        assert result.n_total == 142

    def test_load_current_classifier_none_when_no_file(self, tmp_path):
        service, store = _make_service(tmp_path)
        assert service.load_current_classifier() is None

    def test_load_current_classifier_after_retrain(self, tmp_path):
        service, store = _make_service(tmp_path)
        service.retrain(force=True)
        clf = service.load_current_classifier()
        assert clf is not None
        assert clf.is_trained

    def test_retrain_result_to_dict(self, tmp_path):
        service, store = _make_service(tmp_path)
        result = service.retrain(force=True)
        d = result.to_dict()
        assert d["success"] is True
        assert d["report_after"] is not None
