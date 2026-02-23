"""Tests for Tier 1 document type taxonomy and profiles."""

from __future__ import annotations

from chonk.tier1.fingerprinter import DocumentFingerprint
from chonk.tier1.taxonomy import (
    DOCUMENT_TYPE_PROFILES,
    TRAINABLE_TYPES,
    DocumentType,
    DocumentTypeProfile,
)


class TestDocumentType:
    """Tests for DocumentType enum."""

    def test_has_15_members(self):
        """Enum has exactly 15 members (14 trainable + UNKNOWN)."""
        assert len(DocumentType) == 15

    def test_unknown_exists(self):
        """UNKNOWN sentinel type exists."""
        assert DocumentType.UNKNOWN.value == "unknown"

    def test_all_values_are_strings(self):
        """All enum values are snake_case strings."""
        for dt in DocumentType:
            assert isinstance(dt.value, str)
            assert dt.value == dt.value.lower()


class TestTrainableTypes:
    """Tests for TRAINABLE_TYPES list."""

    def test_excludes_unknown(self):
        """TRAINABLE_TYPES does not include UNKNOWN."""
        assert DocumentType.UNKNOWN not in TRAINABLE_TYPES

    def test_has_14_types(self):
        """14 trainable document types."""
        assert len(TRAINABLE_TYPES) == 14

    def test_all_are_document_types(self):
        """All entries are DocumentType enum values."""
        for dt in TRAINABLE_TYPES:
            assert isinstance(dt, DocumentType)


class TestDocumentTypeProfiles:
    """Tests for DOCUMENT_TYPE_PROFILES."""

    def test_all_trainable_types_have_profiles(self):
        """Every trainable type has a profile."""
        for dt in TRAINABLE_TYPES:
            assert dt in DOCUMENT_TYPE_PROFILES

    def test_profiles_have_feature_means(self):
        """Every profile has feature_means dict."""
        for dt, profile in DOCUMENT_TYPE_PROFILES.items():
            assert isinstance(profile.feature_means, dict)
            assert len(profile.feature_means) > 0

    def test_profiles_have_valid_feature_names(self):
        """All feature names in profiles match actual fingerprint names."""
        valid_names = set(DocumentFingerprint().feature_names())
        for dt, profile in DOCUMENT_TYPE_PROFILES.items():
            for name in profile.feature_means:
                assert name in valid_names, (
                    f"Profile {dt.value} has invalid feature name: {name}"
                )
            for name in profile.feature_stds:
                assert name in valid_names, (
                    f"Profile {dt.value} has invalid std feature name: {name}"
                )

    def test_profiles_have_positive_stds(self):
        """All feature_stds values are positive."""
        for dt, profile in DOCUMENT_TYPE_PROFILES.items():
            for name, std in profile.feature_stds.items():
                assert std > 0, (
                    f"Profile {dt.value} has non-positive std for {name}: {std}"
                )

    def test_profile_labels_match_keys(self):
        """Profile label field matches its dict key."""
        for dt, profile in DOCUMENT_TYPE_PROFILES.items():
            assert profile.label == dt

    def test_profiles_have_descriptions(self):
        """All profiles have non-empty descriptions."""
        for dt, profile in DOCUMENT_TYPE_PROFILES.items():
            assert len(profile.description) > 0

    def test_profile_serialization_roundtrip(self):
        """Profile can be serialized and deserialized."""
        profile = DOCUMENT_TYPE_PROFILES[DocumentType.TECHNICAL_MANUAL]
        d = profile.to_dict()
        restored = DocumentTypeProfile.from_dict(d)
        assert restored.label == profile.label
        assert restored.description == profile.description
        assert restored.feature_means == profile.feature_means
