"""Tier 1 classifier evaluation with k-fold cross-validation.

Provides structured evaluation reports with per-class accuracy
breakdowns in SME-friendly language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from chonk.tier1.classifier import DocumentClassifier
from chonk.tier1.taxonomy import DocumentType
from chonk.tier1.training_data import TrainingCorpus


@dataclass
class ClassAccuracyResult:
    """Per-class accuracy from evaluation.

    Attributes:
        document_type: The document type evaluated.
        correct: Number of correctly classified samples.
        total: Total samples of this type.
        accuracy: Fraction correct (0 to 1).
    """

    document_type: DocumentType
    correct: int
    total: int
    accuracy: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "document_type": self.document_type.value,
            "correct": self.correct,
            "total": self.total,
            "accuracy": self.accuracy,
        }


@dataclass
class EvaluationReport:
    """Structured evaluation report from k-fold cross-validation.

    Attributes:
        overall_accuracy: Weighted accuracy across all classes.
        per_class_results: Accuracy breakdown per document type.
        n_folds: Number of cross-validation folds used.
        n_samples: Total samples evaluated.
        meets_baseline: True if overall_accuracy >= 0.70.
        confusion_matrix: Square matrix [actual][predicted] counts.
        class_labels: Class label for each row/column of confusion matrix.
    """

    overall_accuracy: float
    per_class_results: list[ClassAccuracyResult]
    n_folds: int
    n_samples: int
    meets_baseline: bool
    confusion_matrix: list[list[int]] = field(default_factory=list)
    class_labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "overall_accuracy": self.overall_accuracy,
            "per_class_results": [r.to_dict() for r in self.per_class_results],
            "n_folds": self.n_folds,
            "n_samples": self.n_samples,
            "meets_baseline": self.meets_baseline,
            "confusion_matrix": self.confusion_matrix,
            "class_labels": self.class_labels,
        }

    def summary_lines(self) -> list[str]:
        """Return human-readable summary lines.

        Returns:
            List of formatted strings describing results.
        """
        lines = [
            f"Overall accuracy: {self.overall_accuracy:.1%} "
            f"({'PASS' if self.meets_baseline else 'FAIL'} — "
            f"baseline 70%)",
            f"Evaluated {self.n_samples} samples across "
            f"{len(self.per_class_results)} document types "
            f"({self.n_folds}-fold CV)",
            "",
        ]
        for result in sorted(self.per_class_results, key=lambda r: r.accuracy):
            status = "ok" if result.accuracy >= 0.7 else "WEAK"
            lines.append(
                f"  {result.document_type.value:25s} "
                f"{result.correct:3d}/{result.total:3d} "
                f"({result.accuracy:.0%}) [{status}]"
            )
        return lines


def evaluate_classifier(
    classifier: DocumentClassifier,
    corpus: TrainingCorpus,
    n_folds: int = 5,
) -> EvaluationReport:
    """Run k-fold cross-validation and return structured report.

    Creates new classifier instances for each fold using the same
    hyperparameters as the provided classifier.

    Args:
        classifier: A DocumentClassifier (used for hyperparameter config).
        corpus: Training corpus to evaluate on.
        n_folds: Number of stratified folds.

    Returns:
        EvaluationReport with per-class accuracy and confusion matrix.
    """
    x = corpus.feature_matrix
    y = np.array(corpus.labels)
    feature_names = corpus.feature_names

    class_labels = sorted(set(corpus.labels))
    label_to_idx = {label: i for i, label in enumerate(class_labels)}
    n_classes = len(class_labels)

    # Accumulate predictions across folds
    all_true: list[str] = []
    all_pred: list[str] = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, test_idx in skf.split(x, y):
        fold_corpus = TrainingCorpus(
            feature_matrix=x[train_idx],
            labels=[corpus.labels[i] for i in train_idx],
            feature_names=feature_names,
        )
        fold_clf = DocumentClassifier(
            confidence_threshold=0.0,  # No UNKNOWN during eval
            n_estimators=classifier._sklearn_params["n_estimators"],
            max_depth=classifier._sklearn_params["max_depth"],
            random_state=classifier._sklearn_params["random_state"],
        )
        fold_clf.train(fold_corpus)

        for idx in test_idx:
            vec = x[idx]
            proba = fold_clf._model.predict_proba(vec.reshape(1, -1))[0]  # type: ignore[union-attr]
            pred_label = str(fold_clf._model.classes_[np.argmax(proba)])  # type: ignore[union-attr]
            all_true.append(corpus.labels[idx])
            all_pred.append(pred_label)

    # Build confusion matrix
    confusion = [[0] * n_classes for _ in range(n_classes)]
    for true_label, pred_label in zip(all_true, all_pred):
        true_idx = label_to_idx[true_label]
        pred_idx = label_to_idx.get(pred_label, -1)
        if pred_idx >= 0:
            confusion[true_idx][pred_idx] += 1

    # Per-class accuracy
    per_class: list[ClassAccuracyResult] = []
    total_correct = 0
    total_samples = 0

    for label in class_labels:
        idx = label_to_idx[label]
        correct = confusion[idx][idx]
        total = sum(confusion[idx])
        accuracy = correct / total if total > 0 else 0.0
        total_correct += correct
        total_samples += total
        per_class.append(
            ClassAccuracyResult(
                document_type=DocumentType(label),
                correct=correct,
                total=total,
                accuracy=accuracy,
            )
        )

    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return EvaluationReport(
        overall_accuracy=overall_accuracy,
        per_class_results=per_class,
        n_folds=n_folds,
        n_samples=total_samples,
        meets_baseline=overall_accuracy >= 0.70,
        confusion_matrix=confusion,
        class_labels=class_labels,
    )
