"""Tests for Forge data models."""

from __future__ import annotations

import json
from datetime import datetime

from forge.src.models import (
    Competency,
    Contributor,
    ContributorRole,
    CurriculumStatus,
    CurriculumVersion,
    Discipline,
    DisciplineContributor,
    DisciplineStatus,
    Example,
    ReviewStatus,
)

# ===================================================================
# Enums
# ===================================================================


class TestEnums:
    """Enum value and membership tests."""

    def test_contributor_role_values(self):
        assert ContributorRole.CONTRIBUTOR.value == "contributor"
        assert ContributorRole.LEAD.value == "lead"
        assert ContributorRole.ADMIN.value == "admin"

    def test_discipline_status_values(self):
        assert DisciplineStatus.DRAFT.value == "draft"
        assert DisciplineStatus.ACTIVE.value == "active"
        assert DisciplineStatus.ARCHIVED.value == "archived"

    def test_review_status_values(self):
        assert ReviewStatus.PENDING.value == "pending"
        assert ReviewStatus.APPROVED.value == "approved"
        assert ReviewStatus.REJECTED.value == "rejected"
        assert ReviewStatus.NEEDS_REVISION.value == "needs_revision"

    def test_curriculum_status_values(self):
        assert CurriculumStatus.DRAFT.value == "draft"
        assert CurriculumStatus.PUBLISHED.value == "published"


# ===================================================================
# Contributor
# ===================================================================


class TestContributor:
    """Contributor model tests."""

    def test_generate_id_prefix(self):
        cid = Contributor.generate_id()
        assert cid.startswith("contrib_")

    def test_generate_id_unique(self):
        ids = {Contributor.generate_id() for _ in range(100)}
        assert len(ids) == 100

    def test_to_dict_roundtrip(self):
        c = Contributor(id="contrib_abc123", name="Alice", email="alice@test.com")
        d = c.to_dict()
        restored = Contributor.from_dict(d)
        assert restored.id == c.id
        assert restored.name == c.name
        assert restored.email == c.email

    def test_default_timestamps(self):
        c = Contributor(id="contrib_test", name="Test")
        assert isinstance(c.created_at, datetime)
        assert isinstance(c.updated_at, datetime)


# ===================================================================
# Discipline
# ===================================================================


class TestDiscipline:
    """Discipline model tests."""

    def test_generate_id_prefix(self):
        did = Discipline.generate_id()
        assert did.startswith("disc_")

    def test_generate_id_unique(self):
        ids = {Discipline.generate_id() for _ in range(100)}
        assert len(ids) == 100

    def test_default_status_is_draft(self):
        d = Discipline(id="disc_test", name="Test", description="desc")
        assert d.status == DisciplineStatus.DRAFT

    def test_to_dict_roundtrip(self):
        d = Discipline(
            id="disc_abc",
            name="Military Maintenance",
            description="Maintenance procedures",
            vocabulary=["torque", "clearance"],
            document_types=["technical_manual", "procedure"],
            created_by="contrib_abc",
        )
        data = d.to_dict()
        restored = Discipline.from_dict(data)
        assert restored.id == d.id
        assert restored.name == d.name
        assert restored.vocabulary == d.vocabulary
        assert restored.document_types == d.document_types
        assert restored.status == DisciplineStatus.DRAFT

    def test_vocabulary_default_empty(self):
        d = Discipline(id="disc_test", name="Test", description="desc")
        assert d.vocabulary == []
        assert d.document_types == []


# ===================================================================
# Competency
# ===================================================================


class TestCompetency:
    """Competency model tests."""

    def test_generate_id_prefix(self):
        cid = Competency.generate_id()
        assert cid.startswith("comp_")

    def test_default_coverage_target(self):
        c = Competency(id="comp_test", name="Test", description="d", discipline_id="disc_x")
        assert c.coverage_target == 25

    def test_parent_id_optional(self):
        c = Competency(id="comp_test", name="Test", description="d", discipline_id="disc_x")
        assert c.parent_id is None

    def test_to_dict_roundtrip(self):
        c = Competency(
            id="comp_abc",
            name="Fault Isolation",
            description="Identify faults",
            discipline_id="disc_xyz",
            parent_id="comp_parent",
            coverage_target=30,
        )
        data = c.to_dict()
        restored = Competency.from_dict(data)
        assert restored.id == c.id
        assert restored.parent_id == c.parent_id
        assert restored.coverage_target == 30


# ===================================================================
# Example
# ===================================================================


class TestExample:
    """Example model tests."""

    def test_generate_id_prefix(self):
        eid = Example.generate_id()
        assert eid.startswith("ex_")

    def test_default_review_status(self):
        e = Example(
            id="ex_test",
            question="Q?",
            ideal_answer="A.",
            competency_id="comp_x",
            contributor_id="contrib_x",
            discipline_id="disc_x",
        )
        assert e.review_status == ReviewStatus.PENDING
        assert e.is_test_set is False

    def test_to_dict_roundtrip(self):
        e = Example(
            id="ex_abc",
            question="What is torque?",
            ideal_answer="Rotational force.",
            competency_id="comp_x",
            contributor_id="contrib_x",
            discipline_id="disc_x",
            variants=["Define torque.", "Explain torque."],
            context="Maintenance context",
        )
        data = e.to_dict()
        restored = Example.from_dict(data)
        assert restored.id == e.id
        assert restored.question == e.question
        assert restored.variants == e.variants
        assert restored.context == e.context

    def test_to_dict_reviewed_at_none(self):
        e = Example(
            id="ex_test",
            question="Q?",
            ideal_answer="A.",
            competency_id="comp_x",
            contributor_id="contrib_x",
            discipline_id="disc_x",
        )
        d = e.to_dict()
        assert d["reviewed_at"] is None

    def test_to_training_record_alpaca_format(self):
        e = Example(
            id="ex_abc",
            question="What is torque?",
            ideal_answer="Rotational force.",
            competency_id="comp_x",
            contributor_id="contrib_y",
            discipline_id="disc_z",
        )
        record = e.to_training_record()
        assert record["instruction"] == "What is torque?"
        assert record["input"] == ""
        assert record["output"] == "Rotational force."
        assert record["metadata"]["example_id"] == "ex_abc"
        assert record["metadata"]["discipline_id"] == "disc_z"
        assert record["metadata"]["competency_id"] == "comp_x"
        assert record["metadata"]["contributor_id"] == "contrib_y"

    def test_from_dict_with_reviewed_at(self):
        now = datetime.now()
        e = Example(
            id="ex_test",
            question="Q?",
            ideal_answer="A.",
            competency_id="comp_x",
            contributor_id="contrib_x",
            discipline_id="disc_x",
            reviewed_at=now,
            reviewed_by="contrib_y",
            review_status=ReviewStatus.APPROVED,
        )
        data = e.to_dict()
        restored = Example.from_dict(data)
        assert restored.reviewed_at is not None
        assert restored.reviewed_by == "contrib_y"
        assert restored.review_status == ReviewStatus.APPROVED


# ===================================================================
# DisciplineContributor
# ===================================================================


class TestDisciplineContributor:
    """DisciplineContributor model tests."""

    def test_default_role(self):
        dc = DisciplineContributor(discipline_id="disc_x", contributor_id="contrib_x")
        assert dc.role == ContributorRole.CONTRIBUTOR

    def test_to_dict_roundtrip(self):
        dc = DisciplineContributor(
            discipline_id="disc_x",
            contributor_id="contrib_x",
            role=ContributorRole.LEAD,
            competency_area_ids=["comp_a", "comp_b"],
        )
        data = dc.to_dict()
        restored = DisciplineContributor.from_dict(data)
        assert restored.role == ContributorRole.LEAD
        assert restored.competency_area_ids == ["comp_a", "comp_b"]


# ===================================================================
# CurriculumVersion
# ===================================================================


class TestCurriculumVersion:
    """CurriculumVersion model tests."""

    def test_generate_id_prefix(self):
        vid = CurriculumVersion.generate_id()
        assert vid.startswith("curv_")

    def test_default_status_is_draft(self):
        v = CurriculumVersion(
            id="curv_test",
            discipline_id="disc_x",
            version_number=1,
            created_by="contrib_x",
            example_count=0,
        )
        assert v.status == CurriculumStatus.DRAFT

    def test_get_examples_empty_snapshot(self):
        v = CurriculumVersion(
            id="curv_test",
            discipline_id="disc_x",
            version_number=1,
            created_by="contrib_x",
            example_count=0,
            snapshot_json="",
        )
        assert v.get_examples() == []

    def test_get_examples_deserializes(self):
        ex = Example(
            id="ex_abc",
            question="Q?",
            ideal_answer="A.",
            competency_id="comp_x",
            contributor_id="contrib_x",
            discipline_id="disc_x",
        )
        snapshot = json.dumps([ex.to_dict()])
        v = CurriculumVersion(
            id="curv_test",
            discipline_id="disc_x",
            version_number=1,
            created_by="contrib_x",
            example_count=1,
            snapshot_json=snapshot,
        )
        examples = v.get_examples()
        assert len(examples) == 1
        assert examples[0].id == "ex_abc"
        assert examples[0].question == "Q?"

    def test_to_dict_roundtrip(self):
        v = CurriculumVersion(
            id="curv_abc",
            discipline_id="disc_x",
            version_number=3,
            created_by="contrib_x",
            example_count=42,
            status=CurriculumStatus.PUBLISHED,
        )
        data = v.to_dict()
        restored = CurriculumVersion.from_dict(data)
        assert restored.id == v.id
        assert restored.version_number == 3
        assert restored.status == CurriculumStatus.PUBLISHED
