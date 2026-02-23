"""Tier 1 ML document type classifier.

Wraps a GradientBoostingClassifier trained on synthetic fingerprint
profiles. Inference < 1ms per document. Feature importances inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from chonk.tier1.fingerprinter import DocumentFingerprint
from chonk.tier1.taxonomy import TRAINABLE_TYPES, DocumentType
from chonk.tier1.training_data import TrainingCorpus

_EXPECTED_FEATURES = 49


@dataclass
class ClassificationResult:
    """Result of classifying a single document fingerprint.

    Attributes:
        document_type: Predicted document type (UNKNOWN if low confidence).
        confidence: Probability of the predicted class.
        probabilities: All class probabilities keyed by type value.
        is_unknown: True if confidence below threshold.
    """

    document_type: DocumentType
    confidence: float
    probabilities: dict[str, float]
    is_unknown: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "document_type": self.document_type.value,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "is_unknown": self.is_unknown,
        }


@dataclass
class FeatureImportance:
    """A single feature's importance score from the trained model.

    Attributes:
        feature_name: Name matching DocumentFingerprint.feature_names().
        importance: Relative importance (0 to 1, sums to 1 across all).
        rank: 1-based rank (1 = most important).
    """

    feature_name: str
    importance: float
    rank: int


@dataclass
class TrainingReport:
    """Summary of a completed training run.

    Attributes:
        n_samples: Total training samples used.
        n_classes: Number of document type classes.
        feature_names: The 49 feature names.
        top_features: Top 10 features by importance.
    """

    n_samples: int
    n_classes: int
    feature_names: list[str]
    top_features: list[FeatureImportance] = field(default_factory=list)


class DocumentClassifier:
    """ML classifier for document type detection from fingerprints.

    Uses GradientBoostingClassifier trained on synthetic data derived
    from document type profiles. Returns UNKNOWN when confidence is
    below threshold.

    Args:
        confidence_threshold: Minimum probability for a definitive
            classification. Below this returns UNKNOWN.
        n_estimators: Number of boosting stages.
        max_depth: Maximum tree depth.
        random_state: Random seed for reproducibility.

    Example::

        from chonk.tier1.training_data import generate_training_corpus
        from chonk.tier1.classifier import DocumentClassifier

        corpus = generate_training_corpus(samples_per_type=40)
        clf = DocumentClassifier()
        report = clf.train(corpus)
        result = clf.predict(fingerprint)
        print(result.document_type, result.confidence)
    """

    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.45

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        n_estimators: int = 200,
        max_depth: int = 4,
        random_state: int = 42,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self._model: GradientBoostingClassifier | None = None
        self._feature_names: list[str] = []
        self._class_labels: list[str] = []
        self._sklearn_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        }

    @property
    def is_trained(self) -> bool:
        """True if the classifier has been trained."""
        return self._model is not None

    def train(self, corpus: TrainingCorpus) -> TrainingReport:
        """Fit the model on a training corpus.

        Args:
            corpus: Labeled feature matrix from generate_training_corpus().

        Returns:
            TrainingReport with fit statistics.

        Raises:
            ValueError: If corpus is empty or has wrong feature count.
        """
        if len(corpus.labels) == 0:
            raise ValueError("Training corpus is empty")
        if corpus.feature_matrix.shape[1] != _EXPECTED_FEATURES:
            raise ValueError(
                f"Expected {_EXPECTED_FEATURES} features, " f"got {corpus.feature_matrix.shape[1]}"
            )

        self._feature_names = list(corpus.feature_names)
        self._class_labels = [t.value for t in TRAINABLE_TYPES]

        self._model = GradientBoostingClassifier(**self._sklearn_params)
        self._model.fit(corpus.feature_matrix, corpus.labels)

        top = self.feature_importances()[:10]
        return TrainingReport(
            n_samples=len(corpus.labels),
            n_classes=len(set(corpus.labels)),
            feature_names=self._feature_names,
            top_features=top,
        )

    def predict(self, fingerprint: DocumentFingerprint) -> ClassificationResult:
        """Classify a single document fingerprint.

        Args:
            fingerprint: Extracted fingerprint from DocumentFingerprinter.

        Returns:
            ClassificationResult with type, confidence, and probabilities.

        Raises:
            RuntimeError: If classifier has not been trained.
            ValueError: If feature vector has wrong length.
        """
        self._check_trained()
        vector = fingerprint.to_feature_vector()
        if len(vector) != _EXPECTED_FEATURES:
            raise ValueError(f"Expected {_EXPECTED_FEATURES} features, got {len(vector)}")
        return self._predict_vector(np.array(vector, dtype=float))

    def predict_batch(self, fingerprints: list[DocumentFingerprint]) -> list[ClassificationResult]:
        """Classify multiple fingerprints efficiently.

        Args:
            fingerprints: List of fingerprints to classify.

        Returns:
            List of ClassificationResult, one per input.

        Raises:
            RuntimeError: If classifier has not been trained.
        """
        self._check_trained()
        return [self.predict(fp) for fp in fingerprints]

    def feature_importances(self) -> list[FeatureImportance]:
        """Return features ranked by importance, highest first.

        Returns:
            List of FeatureImportance, sorted by descending importance.

        Raises:
            RuntimeError: If classifier has not been trained.
        """
        self._check_trained()
        assert self._model is not None
        importances = self._model.feature_importances_
        ranked_indices = np.argsort(importances)[::-1]
        return [
            FeatureImportance(
                feature_name=self._feature_names[i],
                importance=float(importances[i]),
                rank=rank + 1,
            )
            for rank, i in enumerate(ranked_indices)
        ]

    def save(self, path: str | Path) -> None:
        """Persist trained classifier to disk using joblib.

        WARNING: Do not load classifier files from untrusted sources.
        joblib deserialization can execute arbitrary code.

        Args:
            path: Destination file path.

        Raises:
            RuntimeError: If classifier has not been trained.
        """
        self._check_trained()
        payload = {
            "model": self._model,
            "feature_names": self._feature_names,
            "class_labels": self._class_labels,
            "confidence_threshold": self.confidence_threshold,
        }
        joblib.dump(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> DocumentClassifier:
        """Load a trained classifier from disk.

        WARNING: Do not load classifier files from untrusted sources.

        Args:
            path: Path to file saved by save().

        Returns:
            Loaded DocumentClassifier ready for prediction.

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If file is not a valid classifier payload.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Classifier file not found: {path}")
        try:
            payload = joblib.load(path)
        except Exception as exc:
            raise ValueError("Invalid classifier file") from exc
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError("Invalid classifier file: missing 'model' key")

        instance = cls(
            confidence_threshold=payload.get(
                "confidence_threshold", cls.DEFAULT_CONFIDENCE_THRESHOLD
            )
        )
        instance._model = payload["model"]
        instance._feature_names = payload["feature_names"]
        instance._class_labels = payload["class_labels"]
        return instance

    def _check_trained(self) -> None:
        """Raise RuntimeError if model is not trained."""
        if not self.is_trained:
            raise RuntimeError("Classifier has not been trained. Call train() first.")

    def _predict_vector(self, vector: np.ndarray) -> ClassificationResult:
        """Classify a single raw feature vector.

        Args:
            vector: 1-D array of 49 floats.

        Returns:
            ClassificationResult.
        """
        assert self._model is not None
        x = vector.reshape(1, -1)
        proba = self._model.predict_proba(x)[0]
        classes = self._model.classes_

        proba_dict = {str(cls): float(p) for cls, p in zip(classes, proba)}
        best_idx = int(np.argmax(proba))
        best_label = str(classes[best_idx])
        confidence = float(proba[best_idx])

        is_unknown = confidence < self.confidence_threshold
        doc_type = DocumentType.UNKNOWN if is_unknown else DocumentType(best_label)

        return ClassificationResult(
            document_type=doc_type,
            confidence=confidence,
            probabilities=proba_dict,
            is_unknown=is_unknown,
        )
