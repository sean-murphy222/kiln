"""Tests for ClassificationFallbackWorkflow."""

from __future__ import annotations

from pathlib import Path

import pytest
from chonk.tier1.classifier import ClassificationResult
from chonk.tier1.fallback import ClassificationFallbackWorkflow, FallbackContext
from chonk.tier1.fingerprinter import DocumentFingerprint
from chonk.tier1.manual_store import ManualLabelStore
from chonk.tier1.retraining import RetrainingService
from chonk.tier1.taxonomy import TRAINABLE_TYPES, DocumentType

# ===================================================================
# Helpers
# ===================================================================


def _make_unknown_result() -> ClassificationResult:
    """Create a ClassificationResult with is_unknown=True."""
    return ClassificationResult(
        document_type=DocumentType.UNKNOWN,
        confidence=0.2,
        probabilities={
            "technical_manual": 0.2,
            "form": 0.15,
            "report": 0.1,
            "correspondence": 0.08,
            "reference_card": 0.07,
            "maintenance_procedure": 0.06,
            "specification": 0.05,
            "training_material": 0.05,
            "parts_catalog": 0.04,
            "engineering_drawing": 0.04,
            "policy_directive": 0.04,
            "safety_bulletin": 0.04,
            "logistics_document": 0.04,
            "test_report": 0.04,
        },
        is_unknown=True,
    )


def _make_known_result() -> ClassificationResult:
    """Create a ClassificationResult with is_unknown=False."""
    return ClassificationResult(
        document_type=DocumentType.TECHNICAL_MANUAL,
        confidence=0.85,
        probabilities={"technical_manual": 0.85, "report": 0.15},
        is_unknown=False,
    )


def _make_workflow(tmp_path: Path) -> ClassificationFallbackWorkflow:
    store = ManualLabelStore(tmp_path / "labels.jsonl")
    service = RetrainingService(
        store=store,
        model_path=tmp_path / "model.pkl",
        synthetic_samples_per_type=10,
    )
    return ClassificationFallbackWorkflow(store, service)


# ===================================================================
# FallbackContext
# ===================================================================


class TestFallbackContext:
    """Tests for FallbackContext."""

    def test_to_dict_contains_required_keys(self, tmp_path):
        workflow = _make_workflow(tmp_path)
        ctx = workflow.build_fallback_context(
            DocumentFingerprint(), _make_unknown_result(), "doc.pdf"
        )
        d = ctx.to_dict()
        assert "classifier_result" in d
        assert "top_candidates" in d
        assert "available_types" in d
        assert "source_document_name" in d

    def test_top_candidates_sorted_descending(self, tmp_path):
        workflow = _make_workflow(tmp_path)
        ctx = workflow.build_fallback_context(
            DocumentFingerprint(), _make_unknown_result(), "doc.pdf"
        )
        probs = [p for _, p in ctx.top_candidates]
        assert probs == sorted(probs, reverse=True)

    def test_top_candidates_max_three(self, tmp_path):
        workflow = _make_workflow(tmp_path)
        ctx = workflow.build_fallback_context(
            DocumentFingerprint(), _make_unknown_result(), "doc.pdf"
        )
        assert len(ctx.top_candidates) <= 3

    def test_available_types_excludes_unknown(self, tmp_path):
        workflow = _make_workflow(tmp_path)
        ctx = workflow.build_fallback_context(
            DocumentFingerprint(), _make_unknown_result(), "doc.pdf"
        )
        assert "unknown" not in ctx.available_types
        assert len(ctx.available_types) == len(TRAINABLE_TYPES)


# ===================================================================
# ClassificationFallbackWorkflow
# ===================================================================


class TestClassificationFallbackWorkflow:
    """Tests for the fallback workflow."""

    def test_build_context_raises_if_not_unknown(self, tmp_path):
        workflow = _make_workflow(tmp_path)
        with pytest.raises(ValueError, match="UNKNOWN"):
            workflow.build_fallback_context(DocumentFingerprint(), _make_known_result(), "doc.pdf")

    def test_build_context_returns_for_unknown(self, tmp_path):
        workflow = _make_workflow(tmp_path)
        ctx = workflow.build_fallback_context(
            DocumentFingerprint(), _make_unknown_result(), "doc.pdf"
        )
        assert isinstance(ctx, FallbackContext)
        assert ctx.source_document_name == "doc.pdf"

    def test_top_candidates_match_probabilities(self, tmp_path):
        workflow = _make_workflow(tmp_path)
        result = _make_unknown_result()
        ctx = workflow.build_fallback_context(DocumentFingerprint(), result, "doc.pdf")
        # Top candidate should be technical_manual (0.2)
        assert ctx.top_candidates[0][0] == "technical_manual"
        assert ctx.top_candidates[0][1] == 0.2

    def test_submit_label_stores_example(self, tmp_path):
        workflow = _make_workflow(tmp_path)
        result = _make_unknown_result()
        workflow.submit_manual_label(
            fingerprint=DocumentFingerprint(),
            chosen_type=DocumentType.TECHNICAL_MANUAL,
            original_result=result,
            source_document_name="doc.pdf",
        )
        status = workflow.get_queue_status()
        assert status["total_examples"] == 1

    def test_submit_unknown_raises(self, tmp_path):
        workflow = _make_workflow(tmp_path)
        with pytest.raises(ValueError, match="UNKNOWN"):
            workflow.submit_manual_label(
                fingerprint=DocumentFingerprint(),
                chosen_type=DocumentType.UNKNOWN,
                original_result=_make_unknown_result(),
                source_document_name="doc.pdf",
            )

    def test_submit_returns_manual_example(self, tmp_path):
        workflow = _make_workflow(tmp_path)
        ex = workflow.submit_manual_label(
            fingerprint=DocumentFingerprint(),
            chosen_type=DocumentType.FORM,
            original_result=_make_unknown_result(),
            source_document_name="form.pdf",
            reviewer_note="Definitely a form",
        )
        assert ex.document_type == DocumentType.FORM
        assert ex.reviewer_note == "Definitely a form"

    def test_submit_assigns_example_id(self, tmp_path):
        workflow = _make_workflow(tmp_path)
        ex = workflow.submit_manual_label(
            fingerprint=DocumentFingerprint(),
            chosen_type=DocumentType.FORM,
            original_result=_make_unknown_result(),
            source_document_name="form.pdf",
        )
        assert len(ex.example_id) == 32

    def test_get_queue_status_initial(self, tmp_path):
        workflow = _make_workflow(tmp_path)
        status = workflow.get_queue_status()
        assert status["total_examples"] == 0
        assert status["by_type"] == {}
        assert status["should_retrain"] is True  # No model yet

    def test_get_queue_status_after_adds(self, tmp_path):
        workflow = _make_workflow(tmp_path)
        for _ in range(3):
            workflow.submit_manual_label(
                fingerprint=DocumentFingerprint(),
                chosen_type=DocumentType.REPORT,
                original_result=_make_unknown_result(),
                source_document_name="report.pdf",
            )
        status = workflow.get_queue_status()
        assert status["total_examples"] == 3
        assert status["by_type"]["report"] == 3

    def test_trigger_retrain_delegates(self, tmp_path):
        workflow = _make_workflow(tmp_path)
        result = workflow.trigger_retrain(force=True)
        assert result.success is True
