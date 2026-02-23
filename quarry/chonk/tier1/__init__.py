"""Tier 1: Structural fingerprinting and ML-based document classification."""

from chonk.tier1.classifier import (
    ClassificationResult,
    DocumentClassifier,
    FeatureImportance,
    TrainingReport,
)
from chonk.tier1.evaluation import (
    ClassAccuracyResult,
    EvaluationReport,
    evaluate_classifier,
)
from chonk.tier1.fingerprinter import (
    ByteLevelFeatures,
    CharacterFeatures,
    DocumentFingerprint,
    DocumentFingerprinter,
    FontFeatures,
    LayoutFeatures,
    RepetitionFeatures,
    StructuralRhythmFeatures,
)
from chonk.tier1.taxonomy import DocumentType, DocumentTypeProfile
from chonk.tier1.training_data import TrainingCorpus, generate_training_corpus

__all__ = [
    "ByteLevelFeatures",
    "CharacterFeatures",
    "ClassAccuracyResult",
    "ClassificationResult",
    "DocumentClassifier",
    "DocumentFingerprint",
    "DocumentFingerprinter",
    "DocumentType",
    "DocumentTypeProfile",
    "EvaluationReport",
    "FeatureImportance",
    "FontFeatures",
    "LayoutFeatures",
    "RepetitionFeatures",
    "StructuralRhythmFeatures",
    "TrainingCorpus",
    "TrainingReport",
    "evaluate_classifier",
    "generate_training_corpus",
]
