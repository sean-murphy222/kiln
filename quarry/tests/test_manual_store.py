"""Tests for ManualLabelStore and ManualExample."""

from __future__ import annotations

import pytest
from chonk.tier1.fingerprinter import DocumentFingerprint
from chonk.tier1.manual_store import ManualExample, ManualLabelStore
from chonk.tier1.taxonomy import DocumentType

# ===================================================================
# ManualExample
# ===================================================================


class TestManualExample:
    """Tests for ManualExample dataclass."""

    def test_to_dict_roundtrip(self):
        fp = DocumentFingerprint()
        ex = ManualExample.create(
            fingerprint=fp,
            document_type=DocumentType.TECHNICAL_MANUAL,
            original_confidence=0.3,
            source_document_name="test.pdf",
            reviewer_note="Looks like a TM",
        )
        data = ex.to_dict()
        restored = ManualExample.from_dict(data)
        assert restored.example_id == ex.example_id
        assert restored.document_type == DocumentType.TECHNICAL_MANUAL
        assert restored.original_confidence == 0.3
        assert restored.source_document_name == "test.pdf"
        assert restored.reviewer_note == "Looks like a TM"

    def test_created_at_is_iso8601(self):
        ex = ManualExample.create(
            fingerprint=DocumentFingerprint(),
            document_type=DocumentType.FORM,
            original_confidence=0.2,
            source_document_name="form.pdf",
        )
        # Should be parseable as ISO 8601
        from datetime import datetime

        datetime.fromisoformat(ex.created_at)

    def test_example_id_is_hex(self):
        ex = ManualExample.create(
            fingerprint=DocumentFingerprint(),
            document_type=DocumentType.FORM,
            original_confidence=0.2,
            source_document_name="form.pdf",
        )
        assert len(ex.example_id) == 32
        int(ex.example_id, 16)  # Should be valid hex

    def test_source_name_sanitized(self):
        """Path components stripped, only filename kept."""
        ex = ManualExample.create(
            fingerprint=DocumentFingerprint(),
            document_type=DocumentType.FORM,
            original_confidence=0.2,
            source_document_name="/some/path/to/form.pdf",
        )
        assert ex.source_document_name == "form.pdf"

    def test_unique_ids(self):
        ids = set()
        for _ in range(100):
            ex = ManualExample.create(
                fingerprint=DocumentFingerprint(),
                document_type=DocumentType.FORM,
                original_confidence=0.2,
                source_document_name="f.pdf",
            )
            ids.add(ex.example_id)
        assert len(ids) == 100


# ===================================================================
# ManualLabelStore
# ===================================================================


class TestManualLabelStore:
    """Tests for ManualLabelStore."""

    def _make_example(
        self, doc_type: DocumentType = DocumentType.TECHNICAL_MANUAL
    ) -> ManualExample:
        return ManualExample.create(
            fingerprint=DocumentFingerprint(),
            document_type=doc_type,
            original_confidence=0.25,
            source_document_name="test.pdf",
        )

    def test_add_single_example(self, tmp_path):
        store = ManualLabelStore(tmp_path / "labels.jsonl")
        ex = self._make_example()
        store.add(ex)
        assert store.count() == 1

    def test_load_all_returns_added(self, tmp_path):
        store = ManualLabelStore(tmp_path / "labels.jsonl")
        ex = self._make_example()
        store.add(ex)
        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0].example_id == ex.example_id
        assert loaded[0].document_type == DocumentType.TECHNICAL_MANUAL

    def test_count_matches_adds(self, tmp_path):
        store = ManualLabelStore(tmp_path / "labels.jsonl")
        for _ in range(5):
            store.add(self._make_example())
        assert store.count() == 5

    def test_count_by_type_accurate(self, tmp_path):
        store = ManualLabelStore(tmp_path / "labels.jsonl")
        store.add(self._make_example(DocumentType.TECHNICAL_MANUAL))
        store.add(self._make_example(DocumentType.TECHNICAL_MANUAL))
        store.add(self._make_example(DocumentType.FORM))
        counts = store.count_by_type()
        assert counts["technical_manual"] == 2
        assert counts["form"] == 1

    def test_add_unknown_raises(self, tmp_path):
        store = ManualLabelStore(tmp_path / "labels.jsonl")
        ex = ManualExample(
            example_id="test",
            fingerprint=DocumentFingerprint(),
            document_type=DocumentType.UNKNOWN,
            original_confidence=0.1,
            source_document_name="test.pdf",
            created_at="2026-01-01T00:00:00",
        )
        with pytest.raises(ValueError, match="UNKNOWN"):
            store.add(ex)

    def test_load_all_empty_store(self, tmp_path):
        store = ManualLabelStore(tmp_path / "labels.jsonl")
        assert store.load_all() == []

    def test_load_all_skips_corrupt_lines(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        store = ManualLabelStore(path)
        ex = self._make_example()
        store.add(ex)
        # Append corrupt line
        with open(path, "a", encoding="utf-8") as f:
            f.write("not valid json\n")
        store.add(self._make_example())
        loaded = store.load_all()
        assert len(loaded) == 2  # Corrupt line skipped

    def test_persistence_across_instances(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        store1 = ManualLabelStore(path)
        store1.add(self._make_example())
        store1.add(self._make_example())

        store2 = ManualLabelStore(path)
        assert store2.count() == 2
        loaded = store2.load_all()
        assert len(loaded) == 2

    def test_clear_removes_all(self, tmp_path):
        store = ManualLabelStore(tmp_path / "labels.jsonl")
        store.add(self._make_example())
        store.add(self._make_example())
        store.clear()
        assert store.count() == 0
        assert store.load_all() == []

    def test_nonexistent_parent_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Parent directory"):
            ManualLabelStore(tmp_path / "no_such_dir" / "labels.jsonl")

    def test_append_does_not_lose_data(self, tmp_path):
        store = ManualLabelStore(tmp_path / "labels.jsonl")
        for i in range(10):
            store.add(self._make_example())
        loaded = store.load_all()
        assert len(loaded) == 10

    def test_count_empty_file(self, tmp_path):
        """Count on non-existent file returns 0."""
        store = ManualLabelStore(tmp_path / "labels.jsonl")
        assert store.count() == 0

    def test_count_by_type_empty(self, tmp_path):
        store = ManualLabelStore(tmp_path / "labels.jsonl")
        assert store.count_by_type() == {}
