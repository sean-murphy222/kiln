"""Classifier retraining orchestration with manual + synthetic data.

Combines human-labeled examples from ManualLabelStore with synthetic
training data to retrain the document classifier.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chonk.tier1.classifier import DocumentClassifier
from chonk.tier1.evaluation import EvaluationReport, evaluate_classifier
from chonk.tier1.manual_store import ManualLabelStore
from chonk.tier1.training_data import TrainingCorpus, generate_training_corpus

logger = logging.getLogger(__name__)


@dataclass
class RetrainingResult:
    """Outcome of a retrain attempt.

    Attributes:
        success: True if classifier was retrained.
        skipped_reason: If not success, why (e.g. 'insufficient_examples').
        n_manual_examples: Number of manual examples included.
        n_synthetic_examples: Number of synthetic examples included.
        n_total: Total training samples.
        report_before: EvaluationReport before retraining (None if first).
        report_after: EvaluationReport after retraining.
        model_path: Path where new model was saved (None if not saved).
    """

    success: bool
    skipped_reason: str = ""
    n_manual_examples: int = 0
    n_synthetic_examples: int = 0
    n_total: int = 0
    report_before: EvaluationReport | None = None
    report_after: EvaluationReport | None = None
    model_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "skipped_reason": self.skipped_reason,
            "n_manual_examples": self.n_manual_examples,
            "n_synthetic_examples": self.n_synthetic_examples,
            "n_total": self.n_total,
            "report_before": (self.report_before.to_dict() if self.report_before else None),
            "report_after": (self.report_after.to_dict() if self.report_after else None),
            "model_path": str(self.model_path) if self.model_path else None,
        }


class RetrainingService:
    """Orchestrates classifier retraining with manual + synthetic data.

    Maintains a policy for when retraining is triggered and manages
    model persistence. Caller decides when to invoke retrain().

    Args:
        store: ManualLabelStore with accumulated human labels.
        model_path: Where to save/load the classifier .pkl file.
        min_new_examples: Minimum new examples since last retrain
            before should_retrain() returns True. Default 5.
        synthetic_samples_per_type: Synthetic samples per type. Default 40.

    Example::

        service = RetrainingService(store, Path("models/classifier.pkl"))
        result = service.retrain()
        if result.success:
            for line in result.report_after.summary_lines():
                print(line)
    """

    _META_SUFFIX = ".meta.json"

    def __init__(
        self,
        store: ManualLabelStore,
        model_path: str | Path,
        min_new_examples: int = 5,
        synthetic_samples_per_type: int = 40,
    ) -> None:
        self._store = store
        self._model_path = Path(model_path).resolve()
        self._meta_path = Path(str(self._model_path) + self._META_SUFFIX)
        self._min_new_examples = min_new_examples
        self._synthetic_samples_per_type = synthetic_samples_per_type

    @property
    def model_path(self) -> Path:
        """Return the resolved model file path."""
        return self._model_path

    def should_retrain(self) -> bool:
        """Return True if retraining is warranted.

        Policy: True when no model exists, or when the store has
        grown by at least min_new_examples since the last retrain.
        """
        if not self._model_path.exists():
            return True
        last_count = self._read_meta_count()
        current_count = self._store.count()
        return current_count >= last_count + self._min_new_examples

    def retrain(self, force: bool = False) -> RetrainingResult:
        """Build combined corpus and retrain the classifier.

        Args:
            force: Retrain even if should_retrain() returns False.

        Returns:
            RetrainingResult with evaluation reports and model path.
        """
        if not force and not self.should_retrain():
            return RetrainingResult(
                success=False,
                skipped_reason="not_enough_new_examples",
            )

        # Load current model for before-report if it exists
        report_before = None
        current_clf = self.load_current_classifier()

        # Build combined corpus
        corpus = self.build_combined_corpus()
        n_manual = self._store.count()
        n_synthetic = corpus.feature_matrix.shape[0] - n_manual

        if current_clf is not None:
            try:
                report_before = evaluate_classifier(current_clf, corpus, n_folds=3)
            except Exception:
                logger.warning("Could not evaluate existing classifier")

        # Train new classifier
        clf = DocumentClassifier()
        try:
            clf.train(corpus)
        except Exception as exc:
            logger.error("Training failed: %s", exc)
            return RetrainingResult(
                success=False,
                skipped_reason="training_failed",
                n_manual_examples=n_manual,
                n_synthetic_examples=n_synthetic,
                n_total=corpus.feature_matrix.shape[0],
            )

        # Evaluate new classifier
        report_after = evaluate_classifier(clf, corpus, n_folds=3)

        # Save atomically: write to temp, then rename
        try:
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_path = tempfile.mkstemp(
                dir=str(self._model_path.parent),
                suffix=".pkl.tmp",
            )
            os.close(fd)
            clf.save(temp_path)

            # Verify the saved model loads
            DocumentClassifier.load(temp_path)

            # Atomic replace
            os.replace(temp_path, str(self._model_path))
        except Exception as exc:
            logger.error("Failed to save model: %s", exc)
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return RetrainingResult(
                success=False,
                skipped_reason="save_failed",
                n_manual_examples=n_manual,
                n_synthetic_examples=n_synthetic,
                n_total=corpus.feature_matrix.shape[0],
                report_after=report_after,
            )

        # Update meta file
        self._write_meta_count(self._store.count())

        return RetrainingResult(
            success=True,
            n_manual_examples=n_manual,
            n_synthetic_examples=n_synthetic,
            n_total=corpus.feature_matrix.shape[0],
            report_before=report_before,
            report_after=report_after,
            model_path=self._model_path,
        )

    def build_combined_corpus(self) -> TrainingCorpus:
        """Merge synthetic baseline with manual examples.

        Returns:
            TrainingCorpus combining synthetic Gaussian samples with
            the real labeled fingerprints from the store.
        """
        synthetic = generate_training_corpus(samples_per_type=self._synthetic_samples_per_type)
        manual_examples = self._store.load_all()

        if not manual_examples:
            return synthetic

        # Build manual feature matrix
        manual_vectors = []
        manual_labels = []
        for ex in manual_examples:
            manual_vectors.append(ex.fingerprint.to_feature_vector())
            manual_labels.append(ex.document_type.value)

        manual_matrix = np.array(manual_vectors, dtype=float)

        # Combine
        combined_matrix = np.vstack([synthetic.feature_matrix, manual_matrix])
        combined_labels = synthetic.labels + manual_labels

        return TrainingCorpus(
            feature_matrix=combined_matrix,
            labels=combined_labels,
            feature_names=synthetic.feature_names,
            samples_per_type=synthetic.samples_per_type,
        )

    def load_current_classifier(self) -> DocumentClassifier | None:
        """Load the currently saved classifier, or None if not found.

        WARNING: Do not load classifier files from untrusted sources.
        """
        if not self._model_path.exists():
            return None
        try:
            return DocumentClassifier.load(self._model_path)
        except (ValueError, FileNotFoundError):
            return None

    def _read_meta_count(self) -> int:
        """Read the last retrain example count from meta file."""
        if not self._meta_path.exists():
            return 0
        try:
            data = json.loads(self._meta_path.read_text(encoding="utf-8"))
            return int(data.get("last_retrain_example_count", 0))
        except (json.JSONDecodeError, ValueError):
            return 0

    def _write_meta_count(self, count: int) -> None:
        """Write the current example count to meta file."""
        self._meta_path.write_text(
            json.dumps({"last_retrain_example_count": count}),
            encoding="utf-8",
        )
