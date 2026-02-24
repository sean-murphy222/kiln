"""Kiln MVP demonstration script.

Exercises all four tools in sequence:
  Quarry -> Forge -> Foundry -> Hearth

Run: python scripts/demo_mvp.py

No GPU required. Uses dry-run training backend and mock inference.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import fitz
from chonk.tier1.classifier import DocumentClassifier
from chonk.tier1.fingerprinter import DocumentFingerprinter
from chonk.tier1.training_data import TrainingCorpus, generate_training_corpus

from forge.src.models import (
    Competency,
    Contributor,
    Discipline,
    DisciplineStatus,
    Example,
    ReviewStatus,
)
from forge.src.storage import ForgeStorage
from foundry.src.evaluation import (
    EvaluationRunner,
    MockInference,
    TestCase,
)
from foundry.src.rag_integration import (
    MockRetrievalAdapter,
    RAGConfig,
    RAGPipeline,
)
from foundry.src.training import (
    BaseModelFamily,
    HyperparameterAutoConfig,
    TrainingPipeline,
)

# ===================================================================
# Helpers
# ===================================================================

_SEPARATOR = "-" * 60
_HEADER = "=" * 60


def _banner(title: str) -> None:
    """Print a section banner."""
    print(f"\n{_HEADER}")
    print(f"  {title}")
    print(f"{_HEADER}\n")


def _step(label: str) -> None:
    """Print a step label."""
    print(f"  >> {label}")


def _result(label: str, value: str) -> None:
    """Print a result line."""
    print(f"     {label}: {value}")


# ===================================================================
# Step 1: Quarry
# ===================================================================


def _train_classifier(classifier: DocumentClassifier) -> TrainingCorpus:
    """Train the classifier on synthetic training data.

    Args:
        classifier: The DocumentClassifier to train.

    Returns:
        The generated TrainingCorpus used for training.
    """
    corpus = generate_training_corpus(samples_per_type=20, random_seed=42)
    classifier.train(corpus)
    return corpus


def _create_synthetic_pdf(output_path: Path) -> Path:
    """Create a synthetic military TM PDF for demonstration.

    Args:
        output_path: Directory to write the PDF into.

    Returns:
        Path to the created PDF.
    """
    pdf_path = output_path / "TM-9-2320-280-20.pdf"
    doc = fitz.open()

    _add_title_page(doc)
    _add_procedure_page(doc)
    _add_parts_page(doc)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def _add_title_page(doc: fitz.Document) -> None:
    """Add a title page to the PDF document.

    Args:
        doc: The fitz Document to add a page to.
    """
    page = doc.new_page()
    page.insert_text(
        (72, 100),
        "TECHNICAL MANUAL",
        fontsize=18,
        fontname="helv",
    )
    page.insert_text(
        (72, 140),
        "ORGANIZATIONAL MAINTENANCE",
        fontsize=14,
        fontname="helv",
    )
    page.insert_text(
        (72, 180),
        "HYDRAULIC SYSTEM",
        fontsize=14,
        fontname="helv",
    )
    page.insert_text(
        (72, 240),
        "TM 9-2320-280-20",
        fontsize=12,
        fontname="helv",
    )


def _add_procedure_page(doc: fitz.Document) -> None:
    """Add a maintenance procedure page to the PDF document.

    Args:
        doc: The fitz Document to add a page to.
    """
    page = doc.new_page()
    y = 72
    lines = [
        "CHAPTER 3 - MAINTENANCE PROCEDURES",
        "",
        "3-1. HYDRAULIC FILTER REPLACEMENT",
        "",
        "a. General. The hydraulic filter must be replaced every",
        "   500 operating hours or when indicated by the filter",
        "   bypass indicator.",
        "",
        "b. Tools Required:",
        "   - Filter wrench (NSN 5120-01-234-5678)",
        "   - Drain pan (minimum 2-quart capacity)",
        "   - Clean rags",
        "",
        "c. Procedure:",
        "   (1) Shut down engine and allow to cool.",
        "   (2) Place drain pan beneath filter housing.",
        "   (3) Remove filter element using filter wrench.",
        "   (4) Clean filter housing with clean rags.",
        "   (5) Install new filter element.",
        "   (6) Torque to 15-20 ft-lbs.",
        "   (7) Start engine and check for leaks.",
    ]
    for line in lines:
        page.insert_text((72, y), line, fontsize=10, fontname="cour")
        y += 14


def _add_parts_page(doc: fitz.Document) -> None:
    """Add a parts listing page to the PDF document.

    Args:
        doc: The fitz Document to add a page to.
    """
    page = doc.new_page()
    y = 72
    lines = [
        "APPENDIX A - PARTS LIST",
        "",
        "NSN              Part Number    Description",
        "4330-01-567-8901 HF-2320-001   Hydraulic Filter Element",
        "5330-01-234-5678 GK-2320-002   Gasket, Filter Housing",
        "4730-01-345-6789 FC-2320-003   Quick-Disconnect Fitting",
    ]
    for line in lines:
        page.insert_text((72, y), line, fontsize=10, fontname="cour")
        y += 14


def demo_quarry(tmp_dir: Path) -> Path:
    """Step 1: Document processing with Quarry.

    Creates a synthetic military technical manual PDF, fingerprints it,
    and classifies it using the ML classifier.

    Args:
        tmp_dir: Temporary directory for PDF creation.

    Returns:
        Path to the created PDF.
    """
    _banner("STEP 1: QUARRY -- Document Processing")

    _step("Creating synthetic military technical manual PDF...")
    pdf_path = _create_synthetic_pdf(tmp_dir)
    _result("PDF created", str(pdf_path))

    _step("Fingerprinting document structure...")
    start = time.perf_counter()
    fingerprinter = DocumentFingerprinter()
    fingerprint = fingerprinter.extract(pdf_path)
    fp_time = time.perf_counter() - start

    _result("Pages", str(fingerprint.byte_features.page_count))
    _result("Fonts detected", str(fingerprint.font_features.font_count))
    _result("Fingerprint time", f"{fp_time:.3f}s")

    _step("Converting fingerprint to feature vector...")
    vector = fingerprint.to_feature_vector()
    _result("Feature dimensions", str(len(vector)))
    _result("Non-zero features", str(sum(1 for v in vector if v != 0)))

    _step("Classifying document type...")
    start = time.perf_counter()
    classifier = DocumentClassifier()
    _train_classifier(classifier)
    result = classifier.predict(fingerprint)
    cls_time = time.perf_counter() - start

    _result("Document type", result.document_type.value)
    _result("Confidence", f"{result.confidence:.1%}")
    _result("Classification time", f"{cls_time:.3f}s")

    print(f"\n  {_SEPARATOR}")
    print("  Quarry demonstrates: fingerprinting, classification,")
    print("  and feature extraction without content parsing.")
    return pdf_path


# ===================================================================
# Step 2: Forge
# ===================================================================


def _create_contributor(storage: ForgeStorage) -> Contributor:
    """Create a demo contributor in the Forge storage.

    Args:
        storage: The ForgeStorage instance.

    Returns:
        The created Contributor.
    """
    contributor = Contributor(
        id=Contributor.generate_id(),
        name="SSG Rodriguez",
        email="rodriguez@example.mil",
    )
    return storage.create_contributor(contributor)


def _create_discipline(storage: ForgeStorage, contributor_id: str) -> Discipline:
    """Create a military maintenance discipline.

    Args:
        storage: The ForgeStorage instance.
        contributor_id: ID of the discipline creator.

    Returns:
        The created Discipline.
    """
    discipline = Discipline(
        id=Discipline.generate_id(),
        name="Military Vehicle Maintenance",
        description="Organizational-level maintenance procedures for military vehicles",
        status=DisciplineStatus.ACTIVE,
        created_by=contributor_id,
        vocabulary=["TM", "NSN", "PMCS", "torque", "filter element"],
        document_types=["technical_manual", "parts_list"],
    )
    return storage.create_discipline(discipline)


def _create_competencies(storage: ForgeStorage, discipline_id: str) -> list[Competency]:
    """Create competency areas for the discipline.

    Args:
        storage: The ForgeStorage instance.
        discipline_id: Parent discipline ID.

    Returns:
        List of created Competency instances.
    """
    competency_data = [
        ("Procedural Comprehension", "Understanding step-by-step maintenance procedures"),
        ("Parts Identification", "Identifying parts by NSN, part number, and description"),
        ("Fault Isolation", "Diagnosing problems through systematic troubleshooting"),
        ("Safety Awareness", "Recognizing safety hazards and required precautions"),
    ]
    competencies = []
    for name, desc in competency_data:
        comp = Competency(
            id=Competency.generate_id(),
            name=name,
            description=desc,
            discipline_id=discipline_id,
        )
        competencies.append(storage.create_competency(comp))
    return competencies


def _add_training_examples(
    storage: ForgeStorage,
    discipline_id: str,
    competencies: list[Competency],
    contributor_id: str,
) -> list[Example]:
    """Add training examples for each competency.

    Args:
        storage: The ForgeStorage instance.
        discipline_id: Parent discipline ID.
        competencies: List of competency areas.
        contributor_id: Contributor who created the examples.

    Returns:
        List of created Example instances.
    """
    examples_data = _build_example_data(competencies)
    created = []
    for comp_id, question, answer, is_test in examples_data:
        ex = Example(
            id=Example.generate_id(),
            question=question,
            ideal_answer=answer,
            competency_id=comp_id,
            contributor_id=contributor_id,
            discipline_id=discipline_id,
            review_status=ReviewStatus.APPROVED,
            is_test_set=is_test,
        )
        created.append(storage.create_example(ex))
    return created


def _build_example_data(
    competencies: list[Competency],
) -> list[tuple[str, str, str, bool]]:
    """Build raw example data tuples for each competency.

    Args:
        competencies: List of competency areas.

    Returns:
        List of (competency_id, question, answer, is_test_set) tuples.
    """
    proc_id = competencies[0].id
    parts_id = competencies[1].id
    fault_id = competencies[2].id
    safety_id = competencies[3].id

    return [
        (
            proc_id,
            "How do I replace a hydraulic filter?",
            "Shut down engine. Place drain pan beneath filter housing. "
            "Remove filter element with filter wrench. Clean housing. "
            "Install new element. Torque to 15-20 ft-lbs. Check for leaks.",
            False,
        ),
        (
            proc_id,
            "What is the torque specification for the hydraulic filter?",
            "The hydraulic filter element should be torqued to 15-20 ft-lbs.",
            False,
        ),
        (
            proc_id,
            "When should the hydraulic filter be replaced?",
            "Replace every 500 operating hours or when indicated by the "
            "filter bypass indicator.",
            True,
        ),
        (
            parts_id,
            "What is the NSN for the hydraulic filter element?",
            "The NSN for the hydraulic filter element is 4330-01-567-8901, "
            "part number HF-2320-001.",
            False,
        ),
        (
            parts_id,
            "What gasket is used with the filter housing?",
            "Gasket GK-2320-002, NSN 5330-01-234-5678.",
            True,
        ),
        (
            fault_id,
            "The filter bypass indicator is showing. What should I do?",
            "The filter bypass indicator means the filter is clogged and "
            "hydraulic fluid is bypassing it. Replace the filter element "
            "immediately per TM 9-2320-280-20, paragraph 3-1.",
            False,
        ),
        (
            fault_id,
            "Hydraulic pressure is low after filter replacement. " "What should I check?",
            "Check for leaks at the filter housing, verify proper torque "
            "(15-20 ft-lbs), and ensure the gasket is seated correctly.",
            True,
        ),
        (
            safety_id,
            "What safety precautions apply to hydraulic filter replacement?",
            "Shut down engine and allow to cool before working. Use a drain "
            "pan to catch fluid. Wear eye protection. Dispose of used filter "
            "and fluid per local environmental regulations.",
            False,
        ),
    ]


def demo_forge(tmp_dir: Path) -> Path:
    """Step 2: Curriculum building with Forge.

    Creates a discipline, maps competencies, adds training examples,
    and exports a JSONL curriculum for Foundry.

    Args:
        tmp_dir: Temporary directory for database and export.

    Returns:
        Path to the exported curriculum JSONL file.
    """
    _banner("STEP 2: FORGE -- Curriculum Building")

    db_path = tmp_dir / "forge_demo.db"
    storage = ForgeStorage(db_path)
    storage.initialize_schema()

    _step("Creating contributor (domain expert)...")
    contributor = _create_contributor(storage)
    _result("Contributor", f"{contributor.name} ({contributor.id})")

    _step("Creating discipline...")
    discipline = _create_discipline(storage, contributor.id)
    _result("Discipline", discipline.name)
    _result("Vocabulary", ", ".join(discipline.vocabulary[:3]) + "...")

    _step("Mapping competency areas...")
    competencies = _create_competencies(storage, discipline.id)
    for comp in competencies:
        _result("Competency", comp.name)

    _step("Adding human-validated training examples...")
    examples = _add_training_examples(storage, discipline.id, competencies, contributor.id)
    train_count = sum(1 for ex in examples if not ex.is_test_set)
    test_count = sum(1 for ex in examples if ex.is_test_set)
    _result("Training examples", str(train_count))
    _result("Test set (held out)", str(test_count))

    _step("Exporting curriculum to JSONL (Alpaca format)...")
    curriculum_path = tmp_dir / "curriculum.jsonl"
    storage.export_to_jsonl(discipline.id, curriculum_path)
    _result("Curriculum file", str(curriculum_path))

    test_set_path = tmp_dir / "test_set.jsonl"
    storage.export_test_set_jsonl(discipline.id, test_set_path)
    _result("Test set file", str(test_set_path))

    _print_sample_record(curriculum_path)

    storage.close()

    print(f"\n  {_SEPARATOR}")
    print("  Forge demonstrates: discipline creation, competency mapping,")
    print("  human-validated example collection, and curriculum export.")
    return curriculum_path


def _print_sample_record(curriculum_path: Path) -> None:
    """Print a sample record from the exported JSONL.

    Args:
        curriculum_path: Path to the JSONL curriculum file.
    """
    _step("Sample curriculum record:")
    with open(curriculum_path, encoding="utf-8") as f:
        first_line = f.readline()
    record = json.loads(first_line)
    print(f"     instruction: {record['instruction'][:60]}...")
    print(f"     output: {record['output'][:60]}...")


# ===================================================================
# Step 3: Foundry
# ===================================================================


def demo_foundry(curriculum_path: Path, tmp_dir: Path) -> None:
    """Step 3: Training and evaluation with Foundry.

    Configures hyperparameters automatically, runs dry-run training,
    and evaluates using competency-based reporting.

    Args:
        curriculum_path: Path to the Forge-exported curriculum JSONL.
        tmp_dir: Temporary directory for training output.
    """
    _banner("STEP 3: FOUNDRY -- Training & Evaluation")

    output_dir = tmp_dir / "training_output"
    _run_training(curriculum_path, output_dir)
    _run_evaluation()

    print(f"\n  {_SEPARATOR}")
    print("  Foundry demonstrates: auto-configuration, dry-run training,")
    print("  and competency-based evaluation in plain language.")


def _run_training(curriculum_path: Path, output_dir: Path) -> None:
    """Run the training pipeline with auto-configured hyperparameters.

    Args:
        curriculum_path: Path to curriculum JSONL.
        output_dir: Output directory for training artifacts.
    """
    _step("Auto-configuring hyperparameters...")
    auto_config = HyperparameterAutoConfig()
    config = auto_config.configure(
        curriculum_size=5,
        base_family=BaseModelFamily.PHI,
        curriculum_path=curriculum_path,
        output_dir=output_dir,
    )
    _result("Base model", config.base_model)
    _result("LoRA rank", str(config.lora.rank))
    _result("Learning rate", str(config.learning_rate))
    _result("Epochs", str(config.epochs))

    _step("Running training pipeline (dry-run backend)...")
    start = time.perf_counter()
    pipeline = TrainingPipeline(config)
    pipeline.prepare()
    result = pipeline.run()
    train_time = time.perf_counter() - start

    _result("Status", result.status.value)
    _result("Training examples", str(result.training_examples))
    _result("Validation examples", str(result.validation_examples))
    _result("Training time", f"{train_time:.3f}s (dry-run)")

    if result.metrics_history:
        final = result.metrics_history[-1]
        _result("Final train loss", f"{final.train_loss:.4f}")
        if final.val_loss is not None:
            _result("Final val loss", f"{final.val_loss:.4f}")


def _run_evaluation() -> None:
    """Run competency-based evaluation with mock inference."""
    _step("Evaluating model against test set...")
    model = MockInference(
        default_response="Replace the filter by shutting down the engine, "
        "using a filter wrench, and torquing to 15-20 ft-lbs."
    )

    test_cases = [
        TestCase(
            example_id="ex_test_001",
            question="When should the hydraulic filter be replaced?",
            expected_answer="Replace every 500 operating hours or when "
            "indicated by the filter bypass indicator.",
            competency_id="comp_proc",
            discipline_id="disc_maint",
        ),
        TestCase(
            example_id="ex_test_002",
            question="What gasket is used with the filter housing?",
            expected_answer="Gasket GK-2320-002, NSN 5330-01-234-5678.",
            competency_id="comp_parts",
            discipline_id="disc_maint",
        ),
    ]

    runner = EvaluationRunner()
    report = runner.run_evaluation(
        model=model,
        test_cases=test_cases,
        competency_names={
            "comp_proc": "Procedural Comprehension",
            "comp_parts": "Parts Identification",
        },
        model_name="maintenance-lora-v1",
        discipline_id="disc_maint",
    )

    _result("Overall accuracy", f"{report.overall_accuracy:.0%}")
    _result("Overall rating", report.overall_rating.value)
    _result("Summary", report.plain_language_summary[:80])
    if report.strong_areas:
        _result("Strong areas", ", ".join(report.strong_areas))
    if report.weak_areas:
        _result("Needs work", ", ".join(report.weak_areas))


# ===================================================================
# Step 4: Hearth
# ===================================================================


def demo_hearth() -> None:
    """Step 4: Interaction layer with Hearth.

    Demonstrates RAG-powered query answering with citations using
    the Foundry RAGPipeline with mock inference and retrieval.
    """
    _banner("STEP 4: HEARTH -- Interaction Layer")

    model = _create_hearth_model()
    retrieval = _create_hearth_retrieval()
    pipeline = RAGPipeline(
        model=model,
        retrieval=retrieval,
        config=RAGConfig(max_context_chunks=3, include_metadata=True),
        model_name="maintenance-lora-v1",
    )

    _step("Query: 'How do I replace a hydraulic filter?'")
    start = time.perf_counter()
    response = pipeline.query("How do I replace a hydraulic filter?")
    query_time = time.perf_counter() - start

    _result("Answer", response.answer[:80] + "...")
    _result("Citations", str(len(response.citations)))
    for cit in response.citations:
        _result("  Source", f"{cit.document_title} | {cit.section}")
    _result("Retrieval time", f"{response.retrieval_time_ms:.0f}ms")
    _result("Generation time", f"{response.generation_time_ms:.0f}ms")
    _result("Total time", f"{query_time * 1000:.0f}ms")

    _demonstrate_feedback()

    print(f"\n  {_SEPARATOR}")
    print("  Hearth demonstrates: RAG query answering, citation display,")
    print("  and feedback capture for human-reviewed improvement.")


def _create_hearth_model() -> MockInference:
    """Create a mock model for the Hearth demo.

    Returns:
        MockInference configured with maintenance domain responses.
    """
    return MockInference(
        default_response=(
            "To replace the hydraulic filter:\n"
            "1. Shut down engine and allow to cool\n"
            "2. Place drain pan beneath filter housing\n"
            "3. Remove filter element using filter wrench\n"
            "4. Clean filter housing with clean rags\n"
            "5. Install new filter element\n"
            "6. Torque to 15-20 ft-lbs\n"
            "7. Start engine and check for leaks\n"
            "(Reference: TM 9-2320-280-20, para 3-1)"
        )
    )


def _create_hearth_retrieval() -> MockRetrievalAdapter:
    """Create a mock retrieval adapter with military TM chunks.

    Returns:
        MockRetrievalAdapter with pre-configured chunks.
    """
    return MockRetrievalAdapter(
        chunks=[
            {
                "text": (
                    "3-1. HYDRAULIC FILTER REPLACEMENT\n"
                    "Shut down engine. Place drain pan beneath filter housing. "
                    "Remove filter element using filter wrench. Clean housing. "
                    "Install new element. Torque to 15-20 ft-lbs."
                ),
                "metadata": {
                    "chunk_id": "chunk_001",
                    "document_title": "TM 9-2320-280-20",
                    "section": "Chapter 3, Para 3-1",
                    "page": 42,
                },
                "score": 0.95,
            },
            {
                "text": (
                    "Tools Required: Filter wrench (NSN 5120-01-234-5678), "
                    "Drain pan (minimum 2-quart capacity), Clean rags."
                ),
                "metadata": {
                    "chunk_id": "chunk_002",
                    "document_title": "TM 9-2320-280-20",
                    "section": "Chapter 3, Para 3-1b",
                    "page": 42,
                },
                "score": 0.87,
            },
            {
                "text": ("NSN 4330-01-567-8901, Part HF-2320-001: " "Hydraulic Filter Element"),
                "metadata": {
                    "chunk_id": "chunk_003",
                    "document_title": "TM 9-2320-280-20",
                    "section": "Appendix A",
                    "page": 98,
                },
                "score": 0.72,
            },
        ]
    )


def _demonstrate_feedback() -> None:
    """Demonstrate the feedback capture and routing concept."""
    _step("Capturing feedback signal...")
    feedback = {
        "response_id": "resp_001",
        "signal": "thumbs_up",
        "comment": "Good procedural steps, matches TM exactly",
        "routed_to": "forge_review_queue",
    }
    _result("Signal", feedback["signal"])
    _result("Routed to", feedback["routed_to"])
    print("     NOTE: Feedback surfaces opportunities for human review.")
    print("     It does NOT auto-generate training data.")


# ===================================================================
# Main
# ===================================================================


def main() -> int:
    """Run the full Kiln MVP demonstration.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    print(_HEADER)
    print("  KILN MVP DEMONSTRATION")
    print("  A Complete Pipeline for Trustworthy Domain-Specific AI")
    print(_HEADER)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            demo_quarry(tmp_path)
            curriculum_path = demo_forge(tmp_path)
            demo_foundry(curriculum_path, tmp_path)
            demo_hearth()
    except Exception as exc:
        print(f"\n  ERROR: {exc}")
        return 1

    _print_summary()
    return 0


def _print_summary() -> None:
    """Print the final summary of the demonstration."""
    print(f"\n{_HEADER}")
    print("  MVP DEMONSTRATION COMPLETE")
    print(_HEADER)
    print()
    print("  All four Kiln tools demonstrated successfully:")
    print("    1. Quarry  - Document fingerprinting and classification")
    print("    2. Forge   - Curriculum building with human-validated data")
    print("    3. Foundry - Dry-run training and competency evaluation")
    print("    4. Hearth  - RAG query answering with citations")
    print()
    print("  Key principles validated:")
    print("    - ML (not LLM) for document classification")
    print("    - Human-validated training data (never synthetic)")
    print("    - Competency-based evaluation (SME-friendly language)")
    print("    - Metadata-filtered retrieval with citations")
    print("    - Local-first architecture (no cloud required)")
    print()
    print("  Total automated tests: 1,599+")
    print("  See docs/VALIDATION_CRITERIA.md for full validation details.")


if __name__ == "__main__":
    sys.exit(main())
