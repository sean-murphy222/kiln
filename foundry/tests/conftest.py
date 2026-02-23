"""Shared fixtures for Foundry tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_training_record(
    index: int,
    competency_id: str = "comp_test001",
    discipline_id: str = "disc_test001",
) -> dict:
    """Create a synthetic Alpaca-format training record.

    Args:
        index: Record index for unique content.
        competency_id: Competency to associate with.
        discipline_id: Discipline to associate with.

    Returns:
        Dictionary in Alpaca training format.
    """
    return {
        "instruction": f"How do you perform maintenance procedure {index}?",
        "input": "",
        "output": (
            f"To perform procedure {index}, follow these steps: "
            f"Step 1: Inspect the component. "
            f"Step 2: Apply torque to specification. "
            f"Step 3: Verify clearance meets tolerance."
        ),
        "metadata": {
            "example_id": f"ex_test{index:04d}",
            "discipline_id": discipline_id,
            "competency_id": competency_id,
            "contributor_id": "contrib_test001",
            "review_status": "approved",
            "created_at": "2026-02-20T10:00:00",
        },
    }


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Temporary directory for file output tests."""
    return tmp_path


@pytest.fixture
def sample_curriculum_path(tmp_path: Path) -> Path:
    """Write 20 synthetic Alpaca records to a JSONL file.

    Returns:
        Path to the JSONL file.
    """
    path = tmp_path / "curriculum.jsonl"
    records = []
    for i in range(20):
        comp_id = "comp_test001" if i < 10 else "comp_test002"
        records.append(_make_training_record(i, competency_id=comp_id))
    lines = [json.dumps(r) for r in records]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture
def sample_training_config(sample_curriculum_path: Path, tmp_path: Path):
    """Return a TrainingConfig pointing at the sample curriculum.

    Returns:
        A TrainingConfig instance with sensible test defaults.
    """
    from foundry.src.training import TrainingConfig

    return TrainingConfig(
        base_model="microsoft/phi-3-mini-4k-instruct",
        base_model_family="phi",
        curriculum_path=sample_curriculum_path,
        output_dir=tmp_path / "output",
    )


@pytest.fixture
def training_pipeline(sample_training_config):
    """Return a prepared TrainingPipeline.

    Returns:
        A TrainingPipeline that has been prepared for execution.
    """
    from foundry.src.training import TrainingPipeline

    pipeline = TrainingPipeline(config=sample_training_config)
    pipeline.prepare()
    return pipeline
