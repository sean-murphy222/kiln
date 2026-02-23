"""Tests for Forge SQLite storage layer."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from forge.src.models import (
    Competency,
    ContributorRole,
    CurriculumStatus,
    Discipline,
    DisciplineContributor,
    DisciplineStatus,
    Example,
    ReviewStatus,
)
from forge.src.storage import ForgeStorage, ForgeStorageError

# ===================================================================
# Schema
# ===================================================================


class TestSchema:
    """Schema initialization tests."""

    def test_schema_creates_tables(self, memory_store):
        tables = memory_store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = [t["name"] for t in tables]
        assert "contributors" in names
        assert "disciplines" in names
        assert "competencies" in names
        assert "examples" in names
        assert "discipline_contributors" in names
        assert "curriculum_versions" in names

    def test_schema_creates_indexes(self, memory_store):
        indexes = memory_store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        ).fetchall()
        names = [i["name"] for i in indexes]
        assert "idx_examples_discipline" in names
        assert "idx_examples_competency" in names
        assert "idx_examples_discipline_test" in names
        assert "idx_competencies_discipline" in names
        assert "idx_curriculum_versions_discipline" in names

    def test_foreign_keys_enabled(self, memory_store):
        result = memory_store._conn.execute("PRAGMA foreign_keys").fetchone()
        assert result[0] == 1

    def test_idempotent_schema_init(self, memory_store):
        """Calling initialize_schema twice doesn't raise."""
        memory_store.initialize_schema()


# ===================================================================
# Contributors
# ===================================================================


class TestContributorCRUD:
    """CRUD operations for Contributor."""

    def test_create_and_get_contributor(self, memory_store, sample_contributor):
        memory_store.create_contributor(sample_contributor)
        fetched = memory_store.get_contributor(sample_contributor.id)
        assert fetched is not None
        assert fetched.name == sample_contributor.name
        assert fetched.email == sample_contributor.email

    def test_get_nonexistent_contributor(self, memory_store):
        assert memory_store.get_contributor("contrib_ghost") is None

    def test_duplicate_contributor_raises(self, memory_store, sample_contributor):
        memory_store.create_contributor(sample_contributor)
        with pytest.raises(ForgeStorageError):
            memory_store.create_contributor(sample_contributor)

    def test_update_contributor(self, memory_store, sample_contributor):
        memory_store.create_contributor(sample_contributor)
        sample_contributor.name = "Alice Updated"
        result = memory_store.update_contributor(sample_contributor)
        assert result.name == "Alice Updated"
        fetched = memory_store.get_contributor(sample_contributor.id)
        assert fetched.name == "Alice Updated"

    def test_update_nonexistent_contributor_raises(self, memory_store, sample_contributor):
        with pytest.raises(ForgeStorageError):
            memory_store.update_contributor(sample_contributor)

    def test_delete_contributor(self, memory_store, sample_contributor):
        memory_store.create_contributor(sample_contributor)
        result = memory_store.delete_contributor(sample_contributor.id)
        assert result is True
        assert memory_store.get_contributor(sample_contributor.id) is None

    def test_delete_nonexistent_contributor_returns_false(self, memory_store):
        result = memory_store.delete_contributor("contrib_ghost")
        assert result is False


# ===================================================================
# Disciplines
# ===================================================================


class TestDisciplineCRUD:
    """CRUD operations for Discipline."""

    def test_create_and_get_discipline(self, memory_store, sample_contributor, sample_discipline):
        memory_store.create_contributor(sample_contributor)
        memory_store.create_discipline(sample_discipline)
        fetched = memory_store.get_discipline(sample_discipline.id)
        assert fetched is not None
        assert fetched.name == sample_discipline.name
        assert fetched.status == DisciplineStatus.DRAFT
        assert fetched.vocabulary == sample_discipline.vocabulary
        assert fetched.document_types == sample_discipline.document_types

    def test_get_all_disciplines_filtered_by_status(
        self, memory_store, sample_contributor, sample_discipline
    ):
        memory_store.create_contributor(sample_contributor)
        memory_store.create_discipline(sample_discipline)
        active = Discipline(
            id=Discipline.generate_id(),
            name="Another Discipline",
            description="desc",
            status=DisciplineStatus.ACTIVE,
            created_by=sample_contributor.id,
        )
        memory_store.create_discipline(active)
        drafts = memory_store.get_all_disciplines(status=DisciplineStatus.DRAFT)
        assert len(drafts) == 1
        assert drafts[0].id == sample_discipline.id

    def test_get_all_disciplines_no_filter(
        self, memory_store, sample_contributor, sample_discipline
    ):
        memory_store.create_contributor(sample_contributor)
        memory_store.create_discipline(sample_discipline)
        all_disc = memory_store.get_all_disciplines()
        assert len(all_disc) == 1

    def test_update_discipline_refreshes_updated_at(
        self, memory_store, sample_contributor, sample_discipline
    ):
        memory_store.create_contributor(sample_contributor)
        memory_store.create_discipline(sample_discipline)
        original_ts = sample_discipline.updated_at
        sample_discipline.name = "Renamed Discipline"
        result = memory_store.update_discipline(sample_discipline)
        assert result.updated_at >= original_ts

    def test_delete_discipline_cascades_to_competencies(
        self, populated_store, sample_discipline, sample_competency
    ):
        populated_store.delete_discipline(sample_discipline.id)
        assert populated_store.get_competency(sample_competency.id) is None


# ===================================================================
# Competencies
# ===================================================================


class TestCompetencyCRUD:
    """CRUD operations for Competency."""

    def test_create_and_get_competency(self, populated_store, sample_competency):
        fetched = populated_store.get_competency(sample_competency.id)
        assert fetched is not None
        assert fetched.name == sample_competency.name
        assert fetched.discipline_id == sample_competency.discipline_id
        assert fetched.coverage_target == 25

    def test_competency_with_parent(self, populated_store, sample_discipline, sample_competency):
        child = Competency(
            id=Competency.generate_id(),
            name="Hydraulic System Faults",
            description="Identify hydraulic faults.",
            discipline_id=sample_discipline.id,
            parent_id=sample_competency.id,
        )
        populated_store.create_competency(child)
        fetched = populated_store.get_competency(child.id)
        assert fetched.parent_id == sample_competency.id

    def test_get_competencies_for_discipline(self, populated_store, sample_discipline):
        results = populated_store.get_competencies_for_discipline(sample_discipline.id)
        assert len(results) >= 1
        assert all(c.discipline_id == sample_discipline.id for c in results)

    def test_delete_competency_cascades_to_examples(
        self, populated_store, sample_competency, sample_example
    ):
        populated_store.delete_competency(sample_competency.id)
        assert populated_store.get_example(sample_example.id) is None


# ===================================================================
# Examples
# ===================================================================


class TestExampleCRUD:
    """CRUD operations for Example."""

    def test_create_and_get_example(self, populated_store, sample_example):
        fetched = populated_store.get_example(sample_example.id)
        assert fetched is not None
        assert fetched.question == sample_example.question
        assert fetched.ideal_answer == sample_example.ideal_answer
        assert fetched.variants == sample_example.variants
        assert fetched.review_status == ReviewStatus.PENDING
        assert fetched.is_test_set is False

    def test_get_examples_for_competency_excludes_test_set(
        self, populated_store, sample_competency, sample_contributor, sample_discipline
    ):
        test_ex = Example(
            id=Example.generate_id(),
            question="Test question?",
            ideal_answer="Test answer.",
            competency_id=sample_competency.id,
            contributor_id=sample_contributor.id,
            discipline_id=sample_discipline.id,
            is_test_set=True,
        )
        populated_store.create_example(test_ex)
        training = populated_store.get_examples_for_competency(
            sample_competency.id, include_test_set=False
        )
        assert all(not ex.is_test_set for ex in training)

    def test_get_training_examples_excludes_test_set(
        self,
        populated_store,
        sample_competency,
        sample_contributor,
        sample_discipline,
        sample_example,
    ):
        test_ex = Example(
            id=Example.generate_id(),
            question="Held-out question?",
            ideal_answer="Held-out answer.",
            competency_id=sample_competency.id,
            contributor_id=sample_contributor.id,
            discipline_id=sample_discipline.id,
            is_test_set=True,
        )
        populated_store.create_example(test_ex)
        training = populated_store.get_training_examples(sample_discipline.id)
        ids = [ex.id for ex in training]
        assert sample_example.id in ids
        assert test_ex.id not in ids

    def test_get_test_set_examples(
        self, populated_store, sample_competency, sample_contributor, sample_discipline
    ):
        test_ex = Example(
            id=Example.generate_id(),
            question="Test question?",
            ideal_answer="Test answer.",
            competency_id=sample_competency.id,
            contributor_id=sample_contributor.id,
            discipline_id=sample_discipline.id,
            is_test_set=True,
        )
        populated_store.create_example(test_ex)
        results = populated_store.get_test_set_examples(sample_discipline.id)
        assert len(results) == 1
        assert results[0].id == test_ex.id

    def test_update_example_review_status(
        self, populated_store, sample_example, sample_contributor
    ):
        sample_example.review_status = ReviewStatus.APPROVED
        sample_example.reviewed_by = sample_contributor.id
        sample_example.reviewed_at = datetime.now()
        populated_store.update_example(sample_example)
        fetched = populated_store.get_example(sample_example.id)
        assert fetched.review_status == ReviewStatus.APPROVED
        assert fetched.reviewed_by == sample_contributor.id
        assert fetched.reviewed_at is not None

    def test_update_example_updated_at_changes(self, populated_store, sample_example):
        original_ts = sample_example.updated_at
        sample_example.question = "Revised question?"
        populated_store.update_example(sample_example)
        fetched = populated_store.get_example(sample_example.id)
        assert fetched.updated_at >= original_ts

    def test_delete_example(self, populated_store, sample_example):
        result = populated_store.delete_example(sample_example.id)
        assert result is True
        assert populated_store.get_example(sample_example.id) is None

    def test_example_variants_roundtrip(self, populated_store, sample_example):
        fetched = populated_store.get_example(sample_example.id)
        assert fetched.variants == sample_example.variants


# ===================================================================
# Discipline Contributors
# ===================================================================


class TestDisciplineContributors:
    """Operations on the discipline_contributors join table."""

    def test_add_and_get_contributor_in_discipline(
        self, populated_store, sample_discipline, sample_contributor
    ):
        dc = DisciplineContributor(
            discipline_id=sample_discipline.id,
            contributor_id=sample_contributor.id,
            role=ContributorRole.LEAD,
            competency_area_ids=[],
        )
        populated_store.add_contributor_to_discipline(dc)
        results = populated_store.get_discipline_contributors(sample_discipline.id)
        assert len(results) == 1
        assert results[0].contributor_id == sample_contributor.id
        assert results[0].role == ContributorRole.LEAD

    def test_duplicate_contributor_in_discipline_raises(
        self, populated_store, sample_discipline, sample_contributor
    ):
        dc = DisciplineContributor(
            discipline_id=sample_discipline.id,
            contributor_id=sample_contributor.id,
            role=ContributorRole.CONTRIBUTOR,
        )
        populated_store.add_contributor_to_discipline(dc)
        with pytest.raises(ForgeStorageError):
            populated_store.add_contributor_to_discipline(dc)

    def test_update_contributor_competency_areas(
        self, populated_store, sample_discipline, sample_contributor, sample_competency
    ):
        dc = DisciplineContributor(
            discipline_id=sample_discipline.id,
            contributor_id=sample_contributor.id,
            role=ContributorRole.CONTRIBUTOR,
            competency_area_ids=[],
        )
        populated_store.add_contributor_to_discipline(dc)
        dc.competency_area_ids = [sample_competency.id]
        populated_store.update_contributor_in_discipline(dc)
        results = populated_store.get_discipline_contributors(sample_discipline.id)
        assert sample_competency.id in results[0].competency_area_ids

    def test_remove_contributor_from_discipline(
        self, populated_store, sample_discipline, sample_contributor
    ):
        dc = DisciplineContributor(
            discipline_id=sample_discipline.id,
            contributor_id=sample_contributor.id,
            role=ContributorRole.CONTRIBUTOR,
        )
        populated_store.add_contributor_to_discipline(dc)
        result = populated_store.remove_contributor_from_discipline(
            sample_discipline.id, sample_contributor.id
        )
        assert result is True
        assert populated_store.get_discipline_contributors(sample_discipline.id) == []


# ===================================================================
# Curriculum Versioning
# ===================================================================


class TestCurriculumVersioning:
    """Snapshot-based curriculum versioning."""

    def test_create_curriculum_version(
        self, populated_store, sample_discipline, sample_contributor
    ):
        version = populated_store.create_curriculum_version(
            sample_discipline.id, sample_contributor.id
        )
        assert version.version_number == 1
        assert version.discipline_id == sample_discipline.id
        assert version.example_count == 1
        assert version.status == CurriculumStatus.DRAFT

    def test_version_numbers_increment(
        self, populated_store, sample_discipline, sample_contributor
    ):
        populated_store.create_curriculum_version(sample_discipline.id, sample_contributor.id)
        v2 = populated_store.create_curriculum_version(sample_discipline.id, sample_contributor.id)
        assert v2.version_number == 2

    def test_get_latest_curriculum_version(
        self, populated_store, sample_discipline, sample_contributor
    ):
        populated_store.create_curriculum_version(sample_discipline.id, sample_contributor.id)
        v2 = populated_store.create_curriculum_version(sample_discipline.id, sample_contributor.id)
        latest = populated_store.get_latest_curriculum_version(sample_discipline.id)
        assert latest is not None
        assert latest.version_number == v2.version_number

    def test_get_latest_returns_none_when_no_versions(self, populated_store, sample_discipline):
        result = populated_store.get_latest_curriculum_version(sample_discipline.id)
        assert result is None

    def test_curriculum_version_snapshot_roundtrip(
        self, populated_store, sample_discipline, sample_contributor, sample_example
    ):
        version = populated_store.create_curriculum_version(
            sample_discipline.id, sample_contributor.id
        )
        examples = version.get_examples()
        assert len(examples) == 1
        assert examples[0].id == sample_example.id
        assert examples[0].question == sample_example.question

    def test_publish_curriculum_version(
        self, populated_store, sample_discipline, sample_contributor
    ):
        version = populated_store.create_curriculum_version(
            sample_discipline.id, sample_contributor.id
        )
        published = populated_store.publish_curriculum_version(version.id)
        assert published.status == CurriculumStatus.PUBLISHED

    def test_publish_nonexistent_version_raises(self, populated_store):
        with pytest.raises(ForgeStorageError):
            populated_store.publish_curriculum_version("curv_doesnotexist")

    def test_get_curriculum_history(self, populated_store, sample_discipline, sample_contributor):
        populated_store.create_curriculum_version(sample_discipline.id, sample_contributor.id)
        populated_store.create_curriculum_version(sample_discipline.id, sample_contributor.id)
        history = populated_store.get_curriculum_history(sample_discipline.id)
        assert len(history) == 2
        assert history[0].version_number > history[1].version_number


# ===================================================================
# Coverage Report
# ===================================================================


class TestCoverageReport:
    """Competency coverage reporting."""

    def test_coverage_report_structure(self, populated_store, sample_discipline):
        report = populated_store.get_coverage_report(sample_discipline.id)
        assert "discipline_id" in report
        assert "total_examples" in report
        assert "total_test_examples" in report
        assert "competency_coverage" in report
        assert "gaps" in report
        assert "coverage_complete" in report

    def test_coverage_below_target_creates_gap(self, populated_store, sample_discipline):
        report = populated_store.get_coverage_report(sample_discipline.id)
        assert not report["coverage_complete"]
        assert len(report["gaps"]) > 0

    def test_coverage_complete_when_target_met(self, memory_store, sample_contributor):
        memory_store.create_contributor(sample_contributor)
        disc = Discipline(
            id=Discipline.generate_id(),
            name="Small Discipline",
            description="desc",
            status=DisciplineStatus.ACTIVE,
            created_by=sample_contributor.id,
        )
        memory_store.create_discipline(disc)
        comp = Competency(
            id=Competency.generate_id(),
            name="Single Competency",
            description="desc",
            discipline_id=disc.id,
            coverage_target=2,
        )
        memory_store.create_competency(comp)
        for _ in range(2):
            ex = Example(
                id=Example.generate_id(),
                question="Question?",
                ideal_answer="Answer.",
                competency_id=comp.id,
                contributor_id=sample_contributor.id,
                discipline_id=disc.id,
            )
            memory_store.create_example(ex)
        report = memory_store.get_coverage_report(disc.id)
        assert report["coverage_complete"] is True
        assert report["gaps"] == []

    def test_coverage_excludes_test_set_from_training_count(
        self, populated_store, sample_competency, sample_contributor, sample_discipline
    ):
        test_ex = Example(
            id=Example.generate_id(),
            question="Test question?",
            ideal_answer="Test answer.",
            competency_id=sample_competency.id,
            contributor_id=sample_contributor.id,
            discipline_id=sample_discipline.id,
            is_test_set=True,
        )
        populated_store.create_example(test_ex)
        report = populated_store.get_coverage_report(sample_discipline.id)
        comp_coverage = next(
            c for c in report["competency_coverage"] if c["competency_id"] == sample_competency.id
        )
        assert comp_coverage["example_count"] == 1
        assert report["total_test_examples"] == 1


# ===================================================================
# JSONL Export
# ===================================================================


class TestJSONLExport:
    """JSONL export for Foundry consumption."""

    def test_export_produces_file(self, populated_store, sample_discipline, temp_dir):
        output = temp_dir / "curriculum.jsonl"
        result = populated_store.export_to_jsonl(sample_discipline.id, output)
        assert result.exists()
        assert result == output

    def test_export_alpaca_format(self, populated_store, sample_discipline, temp_dir):
        output = temp_dir / "curriculum.jsonl"
        populated_store.export_to_jsonl(sample_discipline.id, output)
        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert "instruction" in record
        assert "input" in record
        assert "output" in record
        assert "metadata" in record
        assert record["input"] == ""

    def test_export_excludes_test_set_by_default(
        self,
        populated_store,
        sample_competency,
        sample_contributor,
        sample_discipline,
        sample_example,
        temp_dir,
    ):
        test_ex = Example(
            id=Example.generate_id(),
            question="Test question?",
            ideal_answer="Test answer.",
            competency_id=sample_competency.id,
            contributor_id=sample_contributor.id,
            discipline_id=sample_discipline.id,
            is_test_set=True,
        )
        populated_store.create_example(test_ex)
        output = temp_dir / "train.jsonl"
        populated_store.export_to_jsonl(sample_discipline.id, output)
        lines = output.read_text(encoding="utf-8").strip().split("\n")
        exported_ids = [json.loads(line)["metadata"]["example_id"] for line in lines]
        assert sample_example.id in exported_ids
        assert test_ex.id not in exported_ids

    def test_export_includes_test_set_when_requested(
        self,
        populated_store,
        sample_competency,
        sample_contributor,
        sample_discipline,
        sample_example,
        temp_dir,
    ):
        test_ex = Example(
            id=Example.generate_id(),
            question="Test question?",
            ideal_answer="Test answer.",
            competency_id=sample_competency.id,
            contributor_id=sample_contributor.id,
            discipline_id=sample_discipline.id,
            is_test_set=True,
        )
        populated_store.create_example(test_ex)
        output = temp_dir / "all.jsonl"
        populated_store.export_to_jsonl(sample_discipline.id, output, include_test_set=True)
        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_export_test_set_jsonl(
        self, populated_store, sample_competency, sample_contributor, sample_discipline, temp_dir
    ):
        test_ex = Example(
            id=Example.generate_id(),
            question="Test question?",
            ideal_answer="Test answer.",
            competency_id=sample_competency.id,
            contributor_id=sample_contributor.id,
            discipline_id=sample_discipline.id,
            is_test_set=True,
        )
        populated_store.create_example(test_ex)
        output = temp_dir / "testset.jsonl"
        populated_store.export_test_set_jsonl(sample_discipline.id, output)
        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["metadata"]["example_id"] == test_ex.id

    def test_export_metadata_contains_lineage_fields(
        self, populated_store, sample_discipline, temp_dir
    ):
        output = temp_dir / "check.jsonl"
        populated_store.export_to_jsonl(sample_discipline.id, output)
        record = json.loads(output.read_text(encoding="utf-8").strip())
        meta = record["metadata"]
        assert "discipline_id" in meta
        assert "competency_id" in meta
        assert "contributor_id" in meta
        assert "review_status" in meta
        assert "created_at" in meta

    def test_export_empty_discipline_creates_empty_file(
        self, memory_store, sample_contributor, temp_dir
    ):
        memory_store.create_contributor(sample_contributor)
        disc = Discipline(
            id=Discipline.generate_id(),
            name="Empty Discipline",
            description="desc",
            status=DisciplineStatus.ACTIVE,
            created_by=sample_contributor.id,
        )
        memory_store.create_discipline(disc)
        output = temp_dir / "empty.jsonl"
        memory_store.export_to_jsonl(disc.id, output)
        assert output.exists()
        assert output.read_text(encoding="utf-8").strip() == ""


# ===================================================================
# Context Manager
# ===================================================================


class TestStorageContextManager:
    """ForgeStorage as a context manager."""

    def test_context_manager_closes_connection(self, tmp_path):
        db_path = tmp_path / "test.db"
        with ForgeStorage(db_path) as store:
            store.initialize_schema()
        with pytest.raises(Exception):
            store._conn.execute("SELECT 1")
