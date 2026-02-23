"""Synthetic training corpus generator for Tier 1 classifier.

Produces labeled feature vectors by sampling from per-type Gaussian
profiles defined in taxonomy.py. No real PDFs needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chonk.tier1.fingerprinter import DocumentFingerprint
from chonk.tier1.taxonomy import (
    _DEFAULT_MEANS,
    _DEFAULT_STD,
    DOCUMENT_TYPE_PROFILES,
    TRAINABLE_TYPES,
)

# Features that represent ratios and should be clamped to [0, 1]
_RATIO_FEATURES = {
    "byte_has_metadata",
    "byte_has_xmp_metadata",
    "byte_encrypted",
    "byte_has_acroform",
    "font_bold_ratio",
    "font_italic_ratio",
    "font_monospace_ratio",
    "layout_width_consistency",
    "layout_height_consistency",
    "layout_text_area_ratio",
    "char_alpha_ratio",
    "char_numeric_ratio",
    "char_punctuation_ratio",
    "char_whitespace_ratio",
    "char_special_ratio",
    "char_uppercase_ratio",
    "rep_has_page_numbers",
    "rep_has_headers",
    "rep_has_footers",
    "rep_repetition_ratio",
    "rep_first_line_diversity",
    "rhythm_has_toc",
}

# Features that must be non-negative integers
_COUNT_FEATURES = {
    "byte_file_size",
    "byte_object_count",
    "byte_stream_count",
    "byte_page_count",
    "font_font_count",
    "font_distinct_sizes",
    "char_total_chars",
    "rhythm_toc_depth",
    "rhythm_link_count",
    "rhythm_heading_size_levels",
}


@dataclass
class TrainingCorpus:
    """Labeled training data for the document classifier.

    Attributes:
        feature_matrix: Array of shape (n_samples, 49).
        labels: Parallel list of DocumentType string values.
        feature_names: The 49 feature names from DocumentFingerprint.
        samples_per_type: Count of samples generated per type.
    """

    feature_matrix: np.ndarray
    labels: list[str]
    feature_names: list[str]
    samples_per_type: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (feature_matrix as nested list)."""
        return {
            "feature_matrix": self.feature_matrix.tolist(),
            "labels": self.labels,
            "feature_names": self.feature_names,
            "samples_per_type": self.samples_per_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingCorpus:
        """Deserialize from dictionary."""
        return cls(
            feature_matrix=np.array(data["feature_matrix"]),
            labels=data["labels"],
            feature_names=data["feature_names"],
            samples_per_type=data.get("samples_per_type", {}),
        )


def generate_training_corpus(
    samples_per_type: int = 40,
    random_seed: int = 42,
) -> TrainingCorpus:
    """Generate synthetic labeled training corpus from type profiles.

    Samples feature vectors from Gaussian distributions defined by
    each document type's profile. Applies realistic constraints
    (ratios clamped to [0,1], counts non-negative).

    Args:
        samples_per_type: Number of synthetic samples per document type.
        random_seed: Random seed for reproducibility.

    Returns:
        TrainingCorpus with feature_matrix and labels.
    """
    rng = np.random.RandomState(random_seed)
    feature_names = DocumentFingerprint().feature_names()
    n_features = len(feature_names)

    all_vectors: list[np.ndarray] = []
    all_labels: list[str] = []
    per_type_counts: dict[str, int] = {}

    for doc_type in TRAINABLE_TYPES:
        profile = DOCUMENT_TYPE_PROFILES[doc_type]

        # Build mean and std vectors from profile + defaults
        means = np.array(
            [
                profile.feature_means.get(name, _DEFAULT_MEANS.get(name, 0.0))
                for name in feature_names
            ]
        )
        stds = np.array([profile.feature_stds.get(name, _DEFAULT_STD) for name in feature_names])

        # Sample from Gaussian
        samples = rng.normal(loc=means, scale=stds, size=(samples_per_type, n_features))

        # Apply constraints
        samples = _apply_constraints(samples, feature_names)

        all_vectors.append(samples)
        all_labels.extend([doc_type.value] * samples_per_type)
        per_type_counts[doc_type.value] = samples_per_type

    feature_matrix = np.vstack(all_vectors)

    return TrainingCorpus(
        feature_matrix=feature_matrix,
        labels=all_labels,
        feature_names=feature_names,
        samples_per_type=per_type_counts,
    )


def _apply_constraints(samples: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Apply realistic constraints to sampled feature vectors.

    Args:
        samples: Array of shape (n, 49).
        feature_names: Feature name for each column.

    Returns:
        Constrained array.
    """
    for col_idx, name in enumerate(feature_names):
        if name in _RATIO_FEATURES:
            samples[:, col_idx] = np.clip(samples[:, col_idx], 0.0, 1.0)
        elif name in _COUNT_FEATURES:
            samples[:, col_idx] = np.maximum(samples[:, col_idx], 0.0)
            samples[:, col_idx] = np.round(samples[:, col_idx])
        else:
            # General non-negative for sizes, dimensions, densities
            if "size" in name or "margin" in name or "width" in name or "height" in name:
                samples[:, col_idx] = np.maximum(samples[:, col_idx], 0.0)
            if "density" in name:
                samples[:, col_idx] = np.maximum(samples[:, col_idx], 0.0)
            if "columns" in name:
                samples[:, col_idx] = np.clip(samples[:, col_idx], 1.0, 3.0)
                samples[:, col_idx] = np.round(samples[:, col_idx])

    return samples
