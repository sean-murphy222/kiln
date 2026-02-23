"""Shared fixtures for Foundry tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundry.src.diagnostics import DiagnosticConfig, TrainingDiagnostics


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


@pytest.fixture
def competency_names() -> dict[str, str]:
    """Returns dict mapping competency IDs to human-readable names."""
    return {
        "comp_proc": "Procedural Comprehension",
        "comp_fault": "Fault Isolation",
        "comp_safety": "Safety Protocol Awareness",
        "comp_parts": "Parts Identification",
        "comp_tools": "Tool Selection",
    }


@pytest.fixture
def sample_test_jsonl(tmp_path: Path) -> Path:
    """Writes 15 synthetic test cases to JSONL and returns path.

    Creates 3 examples per competency across 5 competencies:
    - comp_proc: Procedural Comprehension
    - comp_fault: Fault Isolation
    - comp_safety: Safety Protocol Awareness
    - comp_parts: Parts Identification
    - comp_tools: Tool Selection
    """
    path = tmp_path / "test_set.jsonl"
    records = [
        {
            "instruction": "What are the steps to replace a hydraulic filter?",
            "input": "",
            "output": "Remove old filter, clean housing, install new filter, torque to spec.",
            "metadata": {
                "example_id": "ex_001",
                "discipline_id": "disc_maint",
                "competency_id": "comp_proc",
                "contributor_id": "contrib_alice",
                "review_status": "approved",
                "created_at": "2026-01-15T10:00:00",
            },
        },
        {
            "instruction": "Describe the engine oil change procedure.",
            "input": "",
            "output": "Drain oil, replace drain plug, install new filter, fill with specified oil.",
            "metadata": {
                "example_id": "ex_002",
                "discipline_id": "disc_maint",
                "competency_id": "comp_proc",
                "contributor_id": "contrib_alice",
                "review_status": "approved",
                "created_at": "2026-01-15T10:05:00",
            },
        },
        {
            "instruction": "How do you perform a brake pad replacement?",
            "input": "",
            "output": (
                "Remove wheel, remove caliper, replace pads," " reinstall caliper, torque bolts."
            ),
            "metadata": {
                "example_id": "ex_003",
                "discipline_id": "disc_maint",
                "competency_id": "comp_proc",
                "contributor_id": "contrib_bob",
                "review_status": "approved",
                "created_at": "2026-01-15T10:10:00",
            },
        },
        {
            "instruction": "How do you isolate a hydraulic system leak?",
            "input": "",
            "output": "Check pressure gauges, isolate sections systematically, inspect fittings.",
            "metadata": {
                "example_id": "ex_004",
                "discipline_id": "disc_maint",
                "competency_id": "comp_fault",
                "contributor_id": "contrib_alice",
                "review_status": "approved",
                "created_at": "2026-01-15T10:15:00",
            },
        },
        {
            "instruction": "What is the troubleshooting process for an electrical fault?",
            "input": "",
            "output": (
                "Check power source, test fuses, trace wiring,"
                " use multimeter on suspect circuits."
            ),
            "metadata": {
                "example_id": "ex_005",
                "discipline_id": "disc_maint",
                "competency_id": "comp_fault",
                "contributor_id": "contrib_bob",
                "review_status": "approved",
                "created_at": "2026-01-15T10:20:00",
            },
        },
        {
            "instruction": "How do you diagnose a fuel system malfunction?",
            "input": "",
            "output": (
                "Inspect fuel lines, check pump pressure," " test injectors, verify fuel quality."
            ),
            "metadata": {
                "example_id": "ex_006",
                "discipline_id": "disc_maint",
                "competency_id": "comp_fault",
                "contributor_id": "contrib_alice",
                "review_status": "approved",
                "created_at": "2026-01-15T10:25:00",
            },
        },
        {
            "instruction": "What safety steps are required before engine work?",
            "input": "",
            "output": "Lock out power, verify zero energy, use PPE, follow safety checklist.",
            "metadata": {
                "example_id": "ex_007",
                "discipline_id": "disc_maint",
                "competency_id": "comp_safety",
                "contributor_id": "contrib_alice",
                "review_status": "approved",
                "created_at": "2026-01-15T10:30:00",
            },
        },
        {
            "instruction": "What PPE is required for welding operations?",
            "input": "",
            "output": "Welding helmet, gloves, fire-resistant clothing, steel-toe boots.",
            "metadata": {
                "example_id": "ex_008",
                "discipline_id": "disc_maint",
                "competency_id": "comp_safety",
                "contributor_id": "contrib_bob",
                "review_status": "approved",
                "created_at": "2026-01-15T10:35:00",
            },
        },
        {
            "instruction": "How do you handle hazardous material spills?",
            "input": "",
            "output": "Evacuate area, don PPE, contain spill, report to supervisor, follow MSDS.",
            "metadata": {
                "example_id": "ex_009",
                "discipline_id": "disc_maint",
                "competency_id": "comp_safety",
                "contributor_id": "contrib_alice",
                "review_status": "approved",
                "created_at": "2026-01-15T10:40:00",
            },
        },
        {
            "instruction": "How do you identify the correct replacement part?",
            "input": "",
            "output": "Check part number in TM, verify NSN, match specifications to original.",
            "metadata": {
                "example_id": "ex_010",
                "discipline_id": "disc_maint",
                "competency_id": "comp_parts",
                "contributor_id": "contrib_bob",
                "review_status": "approved",
                "created_at": "2026-01-15T10:45:00",
            },
        },
        {
            "instruction": "What is the difference between AN and MS hardware?",
            "input": "",
            "output": (
                "AN is Army-Navy standard, MS is Military Standard." " Check dash number for size."
            ),
            "metadata": {
                "example_id": "ex_011",
                "discipline_id": "disc_maint",
                "competency_id": "comp_parts",
                "contributor_id": "contrib_alice",
                "review_status": "approved",
                "created_at": "2026-01-15T10:50:00",
            },
        },
        {
            "instruction": "How do you read a parts breakdown diagram?",
            "input": "",
            "output": "Identify assembly, locate item number, cross-reference to parts list table.",
            "metadata": {
                "example_id": "ex_012",
                "discipline_id": "disc_maint",
                "competency_id": "comp_parts",
                "contributor_id": "contrib_bob",
                "review_status": "approved",
                "created_at": "2026-01-15T10:55:00",
            },
        },
        {
            "instruction": "What torque wrench is needed for engine bolts?",
            "input": "",
            "output": (
                "Use calibrated torque wrench rated for specified range," " check TM for values."
            ),
            "metadata": {
                "example_id": "ex_013",
                "discipline_id": "disc_maint",
                "competency_id": "comp_tools",
                "contributor_id": "contrib_alice",
                "review_status": "approved",
                "created_at": "2026-01-15T11:00:00",
            },
        },
        {
            "instruction": "When should you use a dial indicator vs a micrometer?",
            "input": "",
            "output": "Dial indicator for runout and alignment. Micrometer for precise dimensions.",
            "metadata": {
                "example_id": "ex_014",
                "discipline_id": "disc_maint",
                "competency_id": "comp_tools",
                "contributor_id": "contrib_bob",
                "review_status": "approved",
                "created_at": "2026-01-15T11:05:00",
            },
        },
        {
            "instruction": "What tool removes a pressed-in bearing?",
            "input": "",
            "output": "Use a bearing puller or hydraulic press. Never use a hammer directly.",
            "metadata": {
                "example_id": "ex_015",
                "discipline_id": "disc_maint",
                "competency_id": "comp_tools",
                "contributor_id": "contrib_alice",
                "review_status": "approved",
                "created_at": "2026-01-15T11:10:00",
            },
        },
    ]

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return path


@pytest.fixture
def mock_model():
    """Returns a MockInference with pre-configured responses for test cases."""
    from foundry.src.evaluation import MockInference

    responses = {
        "What are the steps to replace a hydraulic filter?": (
            "Remove old filter, clean housing, install new filter, torque to spec."
        ),
        "Describe the engine oil change procedure.": (
            "Drain oil, replace drain plug, install new filter, fill with specified oil."
        ),
        "How do you perform a brake pad replacement?": (
            "Remove wheel, remove caliper, replace pads, reinstall caliper, torque bolts."
        ),
        "How do you isolate a hydraulic system leak?": (
            "Check pressure gauges, isolate sections, inspect fittings and connections."
        ),
        "What is the troubleshooting process for an electrical fault?": (
            "Check power source and test fuses."
        ),
        "How do you diagnose a fuel system malfunction?": ("Look at the fuel lines for damage."),
        "What safety steps are required before engine work?": (
            "Lock out power, verify zero energy, use PPE, follow safety checklist."
        ),
        "What PPE is required for welding operations?": (
            "Welding helmet, gloves, fire-resistant clothing, steel-toe boots."
        ),
        "How do you handle hazardous material spills?": (
            "Evacuate area, don PPE, contain spill, report to supervisor, follow MSDS."
        ),
        "How do you identify the correct replacement part?": ("Check the part number."),
        "What is the difference between AN and MS hardware?": ("They are different standards."),
        "How do you read a parts breakdown diagram?": ("Look at the diagram and find the part."),
        "What torque wrench is needed for engine bolts?": (
            "Use calibrated torque wrench rated for specified range, check TM for values."
        ),
        "When should you use a dial indicator vs a micrometer?": (
            "Dial indicator for runout. Micrometer for precise dimensions."
        ),
        "What tool removes a pressed-in bearing?": (
            "Use a bearing puller or hydraulic press. Never use a hammer directly."
        ),
    }

    return MockInference(responses=responses)


@pytest.fixture
def evaluation_runner():
    """Returns an EvaluationRunner with default config."""
    from foundry.src.evaluation import EvaluationRunner

    return EvaluationRunner()


@pytest.fixture
def diagnostic_config() -> DiagnosticConfig:
    """Returns DiagnosticConfig with defaults."""
    return DiagnosticConfig()


@pytest.fixture
def training_diagnostics() -> TrainingDiagnostics:
    """Returns TrainingDiagnostics instance with default config."""
    return TrainingDiagnostics()


# ---------------------------------------------------------------------------
# RAG integration fixtures (T-024)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_chunks() -> list[dict]:
    """Returns list of 5 mock chunk dicts with text, metadata, and scores.

    Simulates retrieval results from Quarry with realistic TM content.
    """
    return [
        {
            "text": (
                "To replace the hydraulic filter, first depressurize the system. "
                "Remove the filter housing cover and extract the old element."
            ),
            "metadata": {
                "chunk_id": "chunk_001",
                "document_title": "TM 9-2320-280-20",
                "section": "Chapter 3: Hydraulic System",
                "page": 42,
            },
            "score": 0.95,
        },
        {
            "text": (
                "Install the new filter element ensuring the O-ring is properly seated. "
                "Torque the housing cover to 25 ft-lbs."
            ),
            "metadata": {
                "chunk_id": "chunk_002",
                "document_title": "TM 9-2320-280-20",
                "section": "Chapter 3: Hydraulic System",
                "page": 43,
            },
            "score": 0.90,
        },
        {
            "text": (
                "After filter replacement, bleed the hydraulic system by cycling "
                "the controls through full range of motion three times."
            ),
            "metadata": {
                "chunk_id": "chunk_003",
                "document_title": "TM 9-2320-280-20",
                "section": "Chapter 3: Hydraulic System",
                "page": 44,
            },
            "score": 0.85,
        },
        {
            "text": (
                "Verify system pressure is within 2800-3200 PSI after bleeding. "
                "Check all fittings for leaks."
            ),
            "metadata": {
                "chunk_id": "chunk_004",
                "document_title": "TM 9-2320-280-20",
                "section": "Section 3.5: Pressure Verification",
                "page": 45,
            },
            "score": 0.80,
        },
        {
            "text": (
                "Record the filter replacement in the equipment maintenance log. "
                "Note the date, mileage, and filter part number."
            ),
            "metadata": {
                "chunk_id": "chunk_005",
                "document_title": "TM 9-2320-280-20",
                "section": "Section 3.6: Maintenance Records",
                "page": 46,
            },
            "score": 0.75,
        },
    ]


@pytest.fixture
def mock_retrieval(sample_chunks: list[dict]):
    """Returns a MockRetrievalAdapter pre-loaded with sample chunks.

    Returns:
        MockRetrievalAdapter instance.
    """
    from foundry.src.rag_integration import MockRetrievalAdapter

    return MockRetrievalAdapter(chunks=sample_chunks)


@pytest.fixture
def rag_config():
    """Returns a RAGConfig with default settings.

    Returns:
        RAGConfig instance.
    """
    from foundry.src.rag_integration import RAGConfig

    return RAGConfig()


@pytest.fixture
def rag_pipeline(mock_model, mock_retrieval, rag_config):
    """Returns a RAGPipeline with mock model and mock retrieval.

    Args:
        mock_model: MockInference fixture from T-022.
        mock_retrieval: MockRetrievalAdapter fixture.
        rag_config: RAGConfig fixture.

    Returns:
        RAGPipeline instance ready for testing.
    """
    from foundry.src.rag_integration import RAGPipeline

    return RAGPipeline(model=mock_model, retrieval=mock_retrieval, config=rag_config)
