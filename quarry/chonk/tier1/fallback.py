"""Manual classification fallback workflow for unknown document types.

Coordinates the process when the classifier returns UNKNOWN: builds
context for the UI, stores human decisions, and triggers retraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chonk.tier1.classifier import ClassificationResult
from chonk.tier1.fingerprinter import DocumentFingerprint
from chonk.tier1.manual_store import ManualExample, ManualLabelStore
from chonk.tier1.retraining import RetrainingResult, RetrainingService
from chonk.tier1.taxonomy import TRAINABLE_TYPES, DocumentType


@dataclass
class FallbackContext:
    """Context for manual classification UI when classifier returns UNKNOWN.

    Attributes:
        fingerprint: The fingerprint that triggered UNKNOWN.
        classifier_result: The original ClassificationResult.
        top_candidates: Top 3 probable types as (type_value, probability).
        available_types: All valid DocumentType values for UI dropdown.
        source_document_name: Filename for display context.
    """

    fingerprint: DocumentFingerprint
    classifier_result: ClassificationResult
    top_candidates: list[tuple[str, float]]
    available_types: list[str]
    source_document_name: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "classifier_result": self.classifier_result.to_dict(),
            "top_candidates": [{"type": t, "probability": p} for t, p in self.top_candidates],
            "available_types": self.available_types,
            "source_document_name": self.source_document_name,
        }


class ClassificationFallbackWorkflow:
    """Coordinates the manual classification fallback process.

    Single entry point for T-003. The API layer only needs this class.

    Args:
        store: ManualLabelStore for persistence.
        retraining_service: RetrainingService for triggering retrains.

    Example::

        workflow = ClassificationFallbackWorkflow(store, service)
        ctx = workflow.build_fallback_context(fp, result, "doc.pdf")
        workflow.submit_manual_label(
            fingerprint=ctx.fingerprint,
            chosen_type=DocumentType.TECHNICAL_MANUAL,
            original_result=ctx.classifier_result,
            source_document_name="doc.pdf",
        )
    """

    def __init__(
        self,
        store: ManualLabelStore,
        retraining_service: RetrainingService,
    ) -> None:
        self._store = store
        self._retraining = retraining_service

    def build_fallback_context(
        self,
        fingerprint: DocumentFingerprint,
        classifier_result: ClassificationResult,
        source_document_name: str,
    ) -> FallbackContext:
        """Build context needed for manual classification UI.

        Args:
            fingerprint: Fingerprint that produced UNKNOWN.
            classifier_result: The ClassificationResult with is_unknown=True.
            source_document_name: Display filename.

        Returns:
            FallbackContext for the UI.

        Raises:
            ValueError: If classifier_result.is_unknown is False.
        """
        if not classifier_result.is_unknown:
            raise ValueError("Fallback context only valid for UNKNOWN classifications")

        # Sort probabilities descending, take top 3
        sorted_probs = sorted(
            classifier_result.probabilities.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        top_candidates = sorted_probs[:3]

        available = [dt.value for dt in TRAINABLE_TYPES]

        return FallbackContext(
            fingerprint=fingerprint,
            classifier_result=classifier_result,
            top_candidates=top_candidates,
            available_types=available,
            source_document_name=source_document_name,
        )

    def submit_manual_label(
        self,
        fingerprint: DocumentFingerprint,
        chosen_type: DocumentType,
        original_result: ClassificationResult,
        source_document_name: str,
        reviewer_note: str = "",
    ) -> ManualExample:
        """Persist a human classification decision.

        Args:
            fingerprint: The fingerprint being labeled.
            chosen_type: The human-selected DocumentType.
            original_result: The classifier result that was UNKNOWN.
            source_document_name: Filename for audit trail.
            reviewer_note: Optional human note.

        Returns:
            The stored ManualExample.

        Raises:
            ValueError: If chosen_type is DocumentType.UNKNOWN.
        """
        if chosen_type == DocumentType.UNKNOWN:
            raise ValueError("Cannot submit UNKNOWN as manual classification")

        example = ManualExample.create(
            fingerprint=fingerprint,
            document_type=chosen_type,
            original_confidence=original_result.confidence,
            source_document_name=source_document_name,
            reviewer_note=reviewer_note,
        )
        self._store.add(example)
        return example

    def get_queue_status(self) -> dict[str, Any]:
        """Return current state of the manual classification queue.

        Returns:
            Dict with total_examples, by_type, should_retrain.
        """
        return {
            "total_examples": self._store.count(),
            "by_type": self._store.count_by_type(),
            "should_retrain": self._retraining.should_retrain(),
        }

    def trigger_retrain(self, force: bool = False) -> RetrainingResult:
        """Explicitly trigger classifier retraining.

        Args:
            force: Bypass the should_retrain() check.

        Returns:
            RetrainingResult with outcome details.
        """
        return self._retraining.retrain(force=force)
