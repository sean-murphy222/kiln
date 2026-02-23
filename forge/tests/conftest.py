"""Shared fixtures for Forge tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.src.models import (
    Competency,
    Contributor,
    Discipline,
    DisciplineStatus,
    Example,
)
from forge.src.storage import ForgeStorage


@pytest.fixture
def memory_store() -> ForgeStorage:
    """In-memory ForgeStorage with schema initialized."""
    store = ForgeStorage(":memory:")
    store.initialize_schema()
    return store


@pytest.fixture
def sample_contributor() -> Contributor:
    """A sample contributor for testing."""
    return Contributor(
        id="contrib_test001",
        name="Alice Smith",
        email="alice@example.com",
    )


@pytest.fixture
def sample_discipline(sample_contributor: Contributor) -> Discipline:
    """A sample discipline for testing."""
    return Discipline(
        id="disc_test001",
        name="Military Maintenance",
        description="Aircraft maintenance procedures",
        status=DisciplineStatus.DRAFT,
        created_by=sample_contributor.id,
        vocabulary=["torque", "clearance", "tolerance"],
        document_types=["technical_manual", "procedure"],
    )


@pytest.fixture
def sample_competency(sample_discipline: Discipline) -> Competency:
    """A sample competency for testing."""
    return Competency(
        id="comp_test001",
        name="Fault Isolation",
        description="Identify and isolate equipment faults",
        discipline_id=sample_discipline.id,
        coverage_target=25,
    )


@pytest.fixture
def sample_example(
    sample_competency: Competency,
    sample_contributor: Contributor,
    sample_discipline: Discipline,
) -> Example:
    """A sample example for testing."""
    return Example(
        id="ex_test001",
        question="How do you isolate a hydraulic leak?",
        ideal_answer="Check pressure readings, isolate sections, inspect fittings.",
        competency_id=sample_competency.id,
        contributor_id=sample_contributor.id,
        discipline_id=sample_discipline.id,
        variants=["Describe hydraulic leak isolation.", "Steps for finding a hydraulic leak?"],
    )


@pytest.fixture
def populated_store(
    memory_store: ForgeStorage,
    sample_contributor: Contributor,
    sample_discipline: Discipline,
    sample_competency: Competency,
    sample_example: Example,
) -> ForgeStorage:
    """Memory store pre-populated with one contributor, discipline, competency, and example."""
    memory_store.create_contributor(sample_contributor)
    memory_store.create_discipline(sample_discipline)
    memory_store.create_competency(sample_competency)
    memory_store.create_example(sample_example)
    return memory_store


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Temporary directory for file output tests."""
    return tmp_path
