"""End-to-end integration tests for the Kiln pipeline.

Validates cross-module boundaries: Quarry -> Forge -> Foundry.
Uses synthetic data and mock inference -- no real PDFs or models required.

Test classes:
    TestQuarryToForge: ChonkRecord -> QuarryBridge -> ForgeStorage
    TestForgeToFoundry: ForgeStorage -> JSONL export -> CurriculumLoader -> TrainingPipeline
    TestFoundryRAGPipeline: MockRetrieval + MockInference -> RAGPipeline -> citations
    TestFoundryRegressionFlow: EvaluationRunner -> RegressionChecker -> VersionManager
    TestFoundryMergingFlow: AdapterInfo -> MergePipeline -> MergeResult
    TestFullPipelineE2E: Quarry -> Forge -> Foundry (full flow)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from chonk.exporters.schema import ChonkRecord

from forge.src.models import (
    Competency,
    Contributor,
    Discipline,
    DisciplineStatus,
    Example,
    ReviewStatus,
)
from forge.src.quarry_integration import QuarryBridge, ScaffoldConfig
from forge.src.storage import ForgeStorage
from foundry.src.evaluation import (
    CompetencyRating,
    CompetencyScore,
    EvaluationReport,
    EvaluationRunner,
    EvaluationStatus,
    MockInference,
    TestCase,
)
from foundry.src.merging import (
    AdapterInfo,
    MergeConfig,
    MergeMethod,
    MergePipeline,
    MergeStatus,
)
from foundry.src.rag_integration import (
    MockRetrievalAdapter,
    RAGConfig,
    RAGEvaluator,
    RAGPipeline,
    RAGSession,
)
from foundry.src.regression import (
    ChangeType,
    RegressionChecker,
    RegressionRunner,
    VersionEntry,
    VersionManager,
)
from foundry.src.training import (
    BaseModelFamily,
    CurriculumLoader,
    HyperparameterAutoConfig,
    TrainingConfig,
    TrainingPipeline,
    TrainingRegistry,
    TrainingStatus,
)

# ===================================================================
# Shared fixtures
# ===================================================================


@pytest.fixture()
def forge_storage() -> ForgeStorage:
    """Create an in-memory ForgeStorage with schema initialized."""
    storage = ForgeStorage(":memory:")
    storage.initialize_schema()
    return storage


@pytest.fixture()
def contributor(forge_storage: ForgeStorage) -> Contributor:
    """Create and persist a test contributor."""
    contrib = Contributor(
        id=Contributor.generate_id(),
        name="Integration Test SME",
        email="sme@integration.test",
    )
    forge_storage.create_contributor(contrib)
    return contrib


@pytest.fixture()
def discipline(
    forge_storage: ForgeStorage,
    contributor: Contributor,
) -> Discipline:
    """Create and persist a test discipline."""
    disc = Discipline(
        id=Discipline.generate_id(),
        name="Military Maintenance",
        description="Maintenance procedures for military equipment",
        status=DisciplineStatus.ACTIVE,
        created_by=contributor.id,
        vocabulary=["TM", "WP", "PMCS", "fault isolation"],
        document_types=["technical_manual", "maintenance_procedure"],
    )
    forge_storage.create_discipline(disc)
    return disc


@pytest.fixture()
def competencies(
    forge_storage: ForgeStorage,
    discipline: Discipline,
) -> list[Competency]:
    """Create and persist two competencies for the test discipline."""
    comp_proc = Competency(
        id=Competency.generate_id(),
        name="Procedural Comprehension",
        description="Understanding step-by-step maintenance procedures",
        discipline_id=discipline.id,
        coverage_target=5,
    )
    comp_fault = Competency(
        id=Competency.generate_id(),
        name="Fault Isolation",
        description="Diagnosing and isolating equipment faults",
        discipline_id=discipline.id,
        coverage_target=5,
    )
    forge_storage.create_competency(comp_proc)
    forge_storage.create_competency(comp_fault)
    return [comp_proc, comp_fault]


def _make_sample_chonk_records() -> list[dict[str, str | int | dict]]:
    """Build synthetic Quarry chunk dicts for integration testing.

    Returns:
        List of chunk dictionaries matching ChonkRecord.to_dict() shape.
    """
    return [
        {
            "id": "chunk_001",
            "content": (
                "Remove the oil filter by turning counterclockwise. "
                "Install new filter and torque to 15 ft-lbs. "
                "Verify no leaks after engine start."
            ),
            "token_count": 120,
            "hierarchy_path": "Chapter 3 > Oil System > Filter Replacement",
            "quality_score": 0.92,
            "source": "TM-1-1500-328.pdf",
            "source_type": "pdf",
            "document_id": "doc_tm1500",
            "page_start": 42,
            "page_end": 43,
            "system_metadata": {
                "section_title": "Oil Filter Replacement",
                "start_page": 42,
                "end_page": 43,
            },
        },
        {
            "id": "chunk_002",
            "content": (
                "If engine oil pressure warning illuminates during operation, "
                "immediately reduce power and check oil level. "
                "If level is normal, suspect oil pressure sensor or pump failure."
            ),
            "token_count": 95,
            "hierarchy_path": "Chapter 5 > Fault Isolation > Oil Pressure",
            "quality_score": 0.88,
            "source": "TM-1-1500-328.pdf",
            "source_type": "pdf",
            "document_id": "doc_tm1500",
            "page_start": 78,
            "page_end": 78,
            "system_metadata": {
                "section_title": "Oil Pressure Fault Isolation",
                "start_page": 78,
                "end_page": 78,
            },
        },
        {
            "id": "chunk_003",
            "content": (
                "Perform PMCS checks before each mission. Inspect fluid levels, "
                "tire pressure, and all safety equipment. Report deficiencies "
                "on DA Form 5988-E."
            ),
            "token_count": 80,
            "hierarchy_path": "Chapter 1 > PMCS > Before Operations",
            "quality_score": 0.95,
            "source": "TM-1-1500-328.pdf",
            "source_type": "pdf",
            "document_id": "doc_tm1500",
            "page_start": 5,
            "page_end": 6,
            "system_metadata": {
                "section_title": "Before Operations PMCS",
                "start_page": 5,
                "end_page": 6,
            },
        },
    ]


def _populate_examples(
    forge_storage: ForgeStorage,
    discipline: Discipline,
    contributor: Contributor,
    competencies: list[Competency],
    count_per_competency: int = 5,
) -> list[Example]:
    """Create training examples and persist them.

    Args:
        forge_storage: Storage backend.
        discipline: Parent discipline.
        contributor: Who created the examples.
        competencies: Competencies to distribute examples across.
        count_per_competency: Number of examples per competency.

    Returns:
        List of all created Example objects.
    """
    examples: list[Example] = []
    for comp in competencies:
        for idx in range(count_per_competency):
            is_test = idx == count_per_competency - 1
            ex = Example(
                id=Example.generate_id(),
                question=_make_question(comp, idx),
                ideal_answer=_make_answer(comp, idx),
                competency_id=comp.id,
                contributor_id=contributor.id,
                discipline_id=discipline.id,
                review_status=ReviewStatus.APPROVED,
                is_test_set=is_test,
            )
            forge_storage.create_example(ex)
            examples.append(ex)
    return examples


def _make_question(comp: Competency, idx: int) -> str:
    """Generate a synthetic question for a competency.

    Args:
        comp: Target competency.
        idx: Example index within the competency.

    Returns:
        Question string at least 10 characters long.
    """
    return f"How do you apply {comp.name.lower()} " f"in scenario {idx + 1} for maintenance tasks?"


def _make_answer(comp: Competency, idx: int) -> str:
    """Generate a synthetic answer for a competency.

    Args:
        comp: Target competency.
        idx: Example index within the competency.

    Returns:
        Answer string at least 10 characters long.
    """
    return (
        f"Apply {comp.name.lower()} by following step-by-step "
        f"procedures in scenario {idx + 1}. Verify completion "
        f"with supervisor checklist."
    )


# ===================================================================
# TestQuarryToForge
# ===================================================================


class TestQuarryToForge:
    """Verify Quarry output feeds into Forge via QuarryBridge."""

    def test_chonk_record_round_trip(self) -> None:
        """ChonkRecord serializes and deserializes without data loss."""
        record = ChonkRecord(
            id="chunk_rt_001",
            content="Test content for round trip verification.",
            token_count=15,
            hierarchy_path="Chapter 1 > Section 2",
            quality_score=0.9,
            source="test.pdf",
            source_type="pdf",
            document_id="doc_001",
            page_start=1,
            page_end=2,
            enrichment_fields={"doc_type": "technical_manual"},
            user_metadata={"tags": ["safety"]},
        )
        as_dict = record.to_dict()
        restored = ChonkRecord.from_dict(as_dict)

        assert restored.id == record.id
        assert restored.content == record.content
        assert restored.token_count == record.token_count
        assert restored.hierarchy_path == record.hierarchy_path
        assert restored.enrichment_fields == record.enrichment_fields

    def test_ingest_chunks_filters_by_token_count(
        self,
        forge_storage: ForgeStorage,
    ) -> None:
        """QuarryBridge.ingest_chunks filters chunks outside token bounds."""
        config = ScaffoldConfig(min_chunk_tokens=50, max_chunk_tokens=200)
        bridge = QuarryBridge(forge_storage, config=config)

        chunks = _make_sample_chonk_records()
        # Add a too-small chunk
        chunks.append(
            {
                "id": "chunk_tiny",
                "content": "Too small.",
                "token_count": 5,
                "system_metadata": {},
            }
        )
        sources = bridge.ingest_chunks(chunks, "TM-1-1500")

        assert len(sources) == 3
        assert all(s.source_document == "TM-1-1500" for s in sources)

    def test_scaffold_generates_candidates(
        self,
        forge_storage: ForgeStorage,
        discipline: Discipline,
        competencies: list[Competency],
    ) -> None:
        """scaffold_examples produces candidates from ChunkSources."""
        bridge = QuarryBridge(forge_storage)
        chunks = _make_sample_chonk_records()
        sources = bridge.ingest_chunks(chunks, "TM-1-1500")

        candidates = bridge.scaffold_examples(sources, discipline.id)

        assert len(candidates) == len(sources)
        for cand in candidates:
            assert cand.suggested_question
            assert cand.suggested_answer
            assert cand.provenance
            assert cand.confidence > 0.0

    def test_accept_candidate_creates_example(
        self,
        forge_storage: ForgeStorage,
        discipline: Discipline,
        competencies: list[Competency],
        contributor: Contributor,
    ) -> None:
        """Accepting a candidate persists an Example in ForgeStorage."""
        bridge = QuarryBridge(forge_storage)
        chunks = _make_sample_chonk_records()
        sources = bridge.ingest_chunks(chunks, "TM-1-1500")
        candidates = bridge.scaffold_examples(sources, discipline.id)

        first = candidates[0]
        example = bridge.accept_candidate(
            candidate=first,
            contributor_id=contributor.id,
            discipline_id=discipline.id,
            competency_id=competencies[0].id,
        )

        assert example.id.startswith("ex_")
        assert example.discipline_id == discipline.id
        assert example.competency_id == competencies[0].id

        fetched = forge_storage.get_example(example.id)
        assert fetched is not None
        assert fetched.question == first.suggested_question

    def test_provenance_tracked_in_context(
        self,
        forge_storage: ForgeStorage,
        discipline: Discipline,
        competencies: list[Competency],
        contributor: Contributor,
    ) -> None:
        """Accepted examples carry provenance in their context field."""
        bridge = QuarryBridge(forge_storage)
        chunks = _make_sample_chonk_records()
        sources = bridge.ingest_chunks(chunks, "TM-1-1500")
        candidates = bridge.scaffold_examples(sources, discipline.id)

        example = bridge.accept_candidate(
            candidate=candidates[0],
            contributor_id=contributor.id,
            discipline_id=discipline.id,
            competency_id=competencies[0].id,
        )

        provenance = bridge.get_provenance(example)
        assert provenance is not None
        assert "TM-1-1500" in provenance


# ===================================================================
# TestForgeToFoundry
# ===================================================================


class TestForgeToFoundry:
    """Verify Forge curriculum feeds into Foundry training pipeline."""

    def test_jsonl_export_and_curriculum_load(
        self,
        tmp_path: Path,
        forge_storage: ForgeStorage,
        discipline: Discipline,
        contributor: Contributor,
        competencies: list[Competency],
    ) -> None:
        """JSONL export from Forge is loadable by CurriculumLoader."""
        _populate_examples(forge_storage, discipline, contributor, competencies)

        jsonl_path = tmp_path / "curriculum.jsonl"
        forge_storage.export_to_jsonl(discipline.id, jsonl_path)

        loader = CurriculumLoader()
        records = loader.load(jsonl_path)

        # 5 per comp, minus 1 test each = 4 training per comp = 8 total
        assert len(records) == 8
        for rec in records:
            assert "instruction" in rec
            assert "output" in rec
            assert "metadata" in rec
            assert rec["metadata"]["discipline_id"] == discipline.id

    def test_curriculum_statistics(
        self,
        tmp_path: Path,
        forge_storage: ForgeStorage,
        discipline: Discipline,
        contributor: Contributor,
        competencies: list[Competency],
    ) -> None:
        """CurriculumLoader.get_statistics reports accurate counts."""
        _populate_examples(forge_storage, discipline, contributor, competencies)

        jsonl_path = tmp_path / "curriculum.jsonl"
        forge_storage.export_to_jsonl(discipline.id, jsonl_path)

        loader = CurriculumLoader()
        records = loader.load(jsonl_path)
        stats = loader.get_statistics(records)

        assert stats["total_records"] == 8
        assert discipline.id in stats["discipline_counts"]
        assert len(stats["competency_counts"]) == 2

    def test_training_pipeline_prepare_and_run(
        self,
        tmp_path: Path,
        forge_storage: ForgeStorage,
        discipline: Discipline,
        contributor: Contributor,
        competencies: list[Competency],
    ) -> None:
        """TrainingPipeline runs dry-run training on exported curriculum."""
        _populate_examples(forge_storage, discipline, contributor, competencies)

        jsonl_path = tmp_path / "curriculum.jsonl"
        forge_storage.export_to_jsonl(discipline.id, jsonl_path)
        output_dir = tmp_path / "training_output"

        config = TrainingConfig(
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family=BaseModelFamily.PHI,
            curriculum_path=jsonl_path,
            output_dir=output_dir,
            epochs=2,
            batch_size=4,
        )

        pipeline = TrainingPipeline(config)
        pipeline.prepare()
        assert pipeline.get_status() == TrainingStatus.PREPARING

        result = pipeline.run()
        assert result.status == TrainingStatus.COMPLETED
        assert result.total_examples == 8
        assert result.training_examples > 0
        assert result.adapter_path is not None
        assert result.adapter_path.exists()
        assert len(result.metrics_history) > 0

    def test_hyperparameter_auto_config(
        self,
        tmp_path: Path,
        forge_storage: ForgeStorage,
        discipline: Discipline,
        contributor: Contributor,
        competencies: list[Competency],
    ) -> None:
        """HyperparameterAutoConfig tunes params for curriculum size."""
        _populate_examples(forge_storage, discipline, contributor, competencies)

        jsonl_path = tmp_path / "curriculum.jsonl"
        forge_storage.export_to_jsonl(discipline.id, jsonl_path)

        auto_config = HyperparameterAutoConfig()
        config = auto_config.configure(
            curriculum_size=8,
            base_family=BaseModelFamily.PHI,
            curriculum_path=jsonl_path,
            output_dir=tmp_path / "auto_output",
        )

        # Small curriculum (<100) should get 5 epochs
        assert config.epochs == 5
        assert config.lora.rank == 16
        assert config.base_model == "microsoft/phi-3-mini-4k-instruct"

    def test_evaluation_on_test_set(
        self,
        tmp_path: Path,
        forge_storage: ForgeStorage,
        discipline: Discipline,
        contributor: Contributor,
        competencies: list[Competency],
    ) -> None:
        """EvaluationRunner evaluates against Forge test set examples."""
        examples = _populate_examples(forge_storage, discipline, contributor, competencies)

        test_examples = [ex for ex in examples if ex.is_test_set]
        assert len(test_examples) == 2

        # Build test cases from test examples
        test_cases = [
            TestCase(
                example_id=ex.id,
                question=ex.question,
                expected_answer=ex.ideal_answer,
                competency_id=ex.competency_id,
                discipline_id=ex.discipline_id,
            )
            for ex in test_examples
        ]

        # MockInference returns answers similar to ideal
        responses = {tc.question: tc.expected_answer for tc in test_cases}
        model = MockInference(responses=responses)

        comp_names = {c.id: c.name for c in competencies}
        runner = EvaluationRunner()
        report = runner.run_evaluation(
            model=model,
            test_cases=test_cases,
            competency_names=comp_names,
            model_name="integration-test-lora",
            discipline_id=discipline.id,
        )

        assert report.status == EvaluationStatus.COMPLETED
        assert report.total_cases == 2
        assert report.overall_correct == 2
        assert report.overall_accuracy == 1.0
        assert report.plain_language_summary

    def test_training_registry(
        self,
        tmp_path: Path,
        forge_storage: ForgeStorage,
        discipline: Discipline,
        contributor: Contributor,
        competencies: list[Competency],
    ) -> None:
        """TrainingRegistry persists and retrieves training runs."""
        _populate_examples(forge_storage, discipline, contributor, competencies)

        jsonl_path = tmp_path / "curriculum.jsonl"
        forge_storage.export_to_jsonl(discipline.id, jsonl_path)
        output_dir = tmp_path / "reg_output"

        config = TrainingConfig(
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family=BaseModelFamily.PHI,
            curriculum_path=jsonl_path,
            output_dir=output_dir,
            epochs=1,
        )

        pipeline = TrainingPipeline(config)
        pipeline.prepare()
        result = pipeline.run()

        registry = TrainingRegistry(tmp_path / "registry")
        run_ref = registry.register_run(result)

        assert run_ref.run_id.startswith("run_")
        assert run_ref.discipline_id == discipline.id

        fetched = registry.get_run(run_ref.run_id)
        assert fetched is not None
        assert fetched.discipline_id == discipline.id

        latest = registry.get_latest_run(discipline_id=discipline.id)
        assert latest is not None
        assert latest.run_id == run_ref.run_id


# ===================================================================
# TestFoundryRAGPipeline
# ===================================================================


class TestFoundryRAGPipeline:
    """Verify RAG integration with mock retrieval and inference."""

    def _make_mock_chunks(self) -> list[dict]:
        """Build mock retrieval chunks.

        Returns:
            List of chunk dicts with text, metadata, and score.
        """
        return [
            {
                "text": (
                    "Remove the oil filter by turning counterclockwise. "
                    "Install new filter and torque to 15 ft-lbs."
                ),
                "metadata": {
                    "chunk_id": "c_001",
                    "document_title": "TM-1-1500-328",
                    "section": "Oil Filter Replacement",
                    "page": 42,
                },
                "score": 0.95,
            },
            {
                "text": (
                    "Verify no leaks after engine start. "
                    "Run engine at idle for 5 minutes and inspect filter area."
                ),
                "metadata": {
                    "chunk_id": "c_002",
                    "document_title": "TM-1-1500-328",
                    "section": "Post-Installation Checks",
                    "page": 43,
                },
                "score": 0.82,
            },
        ]

    def test_rag_query_returns_answer_with_citations(self) -> None:
        """RAGPipeline.query returns answer and citation metadata."""
        chunks = self._make_mock_chunks()
        retrieval = MockRetrievalAdapter(chunks=chunks)
        model = MockInference(
            default_response="Remove the filter counterclockwise and install new one."
        )
        pipeline = RAGPipeline(
            model=model,
            retrieval=retrieval,
            model_name="test-rag-model",
        )

        response = pipeline.query("How do I replace the oil filter?")

        assert response.answer
        assert len(response.citations) == 2
        assert response.citations[0].chunk_id == "c_001"
        assert response.citations[0].document_title == "TM-1-1500-328"
        assert response.retrieval_time_ms >= 0
        assert response.generation_time_ms >= 0
        assert response.total_time_ms >= 0

    def test_rag_batch_query(self) -> None:
        """RAGPipeline.batch_query processes multiple queries."""
        chunks = self._make_mock_chunks()
        retrieval = MockRetrievalAdapter(chunks=chunks)
        model = MockInference(default_response="Batch answer.")
        pipeline = RAGPipeline(model=model, retrieval=retrieval)

        responses = pipeline.batch_query(["Question 1?", "Question 2?", "Question 3?"])

        assert len(responses) == 3
        for resp in responses:
            assert resp.answer == "Batch answer."
            assert len(resp.citations) == 2

    def test_rag_evaluator(self) -> None:
        """RAGEvaluator evaluates test cases with RAG context."""
        chunks = self._make_mock_chunks()
        retrieval = MockRetrievalAdapter(chunks=chunks)
        model = MockInference(default_response="Remove filter counterclockwise install new torque")
        pipeline = RAGPipeline(model=model, retrieval=retrieval)

        evaluator = RAGEvaluator(rag_pipeline=pipeline)
        test_cases = [
            TestCase(
                example_id="ex_001",
                question="How to replace oil filter?",
                expected_answer="Remove filter counterclockwise install new torque",
                competency_id="comp_proc",
                discipline_id="disc_maint",
            ),
        ]

        report = evaluator.evaluate_with_rag(
            test_cases=test_cases,
            competency_names={"comp_proc": "Procedural Comprehension"},
            discipline_id="disc_maint",
        )

        assert report.status == EvaluationStatus.COMPLETED
        assert report.total_cases == 1
        assert report.model_name == "rag-augmented"

    def test_rag_session_save_load(self, tmp_path: Path) -> None:
        """RAGSession persists and reloads conversation history."""
        chunks = self._make_mock_chunks()
        retrieval = MockRetrievalAdapter(chunks=chunks)
        model = MockInference(default_response="Session answer.")
        pipeline = RAGPipeline(model=model, retrieval=retrieval)

        session = RAGSession(pipeline=pipeline, session_dir=tmp_path)
        session.ask("First question?")
        session.ask("Second question?")
        assert len(session.get_history()) == 2

        saved_path = session.save()
        assert saved_path.exists()

        loaded = RAGSession.load(saved_path, pipeline=pipeline)
        assert len(loaded.get_history()) == 2
        assert loaded.get_history()[0].query == "First question?"

    def test_rag_config_customization(self) -> None:
        """RAGConfig controls context building behavior."""
        config = RAGConfig(max_context_chunks=1, max_context_tokens=500)
        chunks = self._make_mock_chunks()
        retrieval = MockRetrievalAdapter(chunks=chunks)
        model = MockInference(default_response="Limited context answer.")
        pipeline = RAGPipeline(
            model=model,
            retrieval=retrieval,
            config=config,
        )

        response = pipeline.query("Test with limited context?")

        # Only 1 chunk context should be used (max_context_chunks=1)
        assert len(response.context_used) <= 2
        assert response.answer == "Limited context answer."


# ===================================================================
# TestFoundryRegressionFlow
# ===================================================================


class TestFoundryRegressionFlow:
    """Verify version management and regression detection."""

    def _make_eval_report(
        self,
        run_id: str,
        discipline_id: str,
        comp_id: str,
        comp_name: str,
        correct: int,
        total: int,
    ) -> EvaluationReport:
        """Build a minimal EvaluationReport for testing.

        Args:
            run_id: Unique evaluation run identifier.
            discipline_id: Discipline being evaluated.
            comp_id: Competency identifier.
            comp_name: Competency display name.
            correct: Number of correct responses.
            total: Total test cases.

        Returns:
            EvaluationReport with a single competency score.
        """
        accuracy = correct / total if total > 0 else 0.0
        rating = self._compute_rating(accuracy)
        score = CompetencyScore(
            competency_id=comp_id,
            competency_name=comp_name,
            total_cases=total,
            correct=correct,
            partially_correct=0,
            incorrect=total - correct,
            no_response=0,
            rating=rating,
            summary=f"{correct}/{total} correct",
        )
        return EvaluationReport(
            run_id=run_id,
            model_name="test-model",
            discipline_id=discipline_id,
            status=EvaluationStatus.COMPLETED,
            competency_scores={comp_id: score},
            test_results=[],
            total_cases=total,
            overall_correct=correct,
            overall_accuracy=accuracy,
            overall_rating=rating,
            plain_language_summary=f"{correct}/{total} correct overall",
            weak_areas=[],
            strong_areas=[],
            started_at=__import__("datetime").datetime.now(),
            completed_at=__import__("datetime").datetime.now(),
        )

    @staticmethod
    def _compute_rating(accuracy: float) -> CompetencyRating:
        """Compute a competency rating from accuracy.

        Args:
            accuracy: Fraction of correct answers.

        Returns:
            CompetencyRating value.
        """
        if accuracy >= 0.8:
            return CompetencyRating.STRONG
        if accuracy >= 0.6:
            return CompetencyRating.ADEQUATE
        if accuracy >= 0.4:
            return CompetencyRating.NEEDS_IMPROVEMENT
        return CompetencyRating.WEAK

    def test_regression_checker_detects_drop(self) -> None:
        """RegressionChecker detects when performance decreases."""
        baseline = self._make_eval_report("eval_base", "disc_001", "comp_proc", "Procedural", 9, 10)
        current = self._make_eval_report("eval_curr", "disc_001", "comp_proc", "Procedural", 5, 10)

        checker = RegressionChecker()
        report = checker.compare(baseline, current, ChangeType.RETRAIN)

        assert report.overall_verdict == "fail"
        assert len(report.regressions) == 1
        assert report.regressions[0].competency_name == "Procedural"
        assert report.plain_language_summary

    def test_regression_checker_detects_improvement(self) -> None:
        """RegressionChecker detects when performance improves."""
        baseline = self._make_eval_report("eval_base", "disc_001", "comp_proc", "Procedural", 5, 10)
        current = self._make_eval_report("eval_curr", "disc_001", "comp_proc", "Procedural", 9, 10)

        checker = RegressionChecker()
        report = checker.compare(baseline, current, ChangeType.RETRAIN)

        assert report.overall_verdict == "pass"
        assert len(report.improvements) == 1
        assert len(report.regressions) == 0

    def test_version_manager_tracks_versions(self, tmp_path: Path) -> None:
        """VersionManager registers and retrieves model versions."""
        vm = VersionManager(tmp_path / "versions")

        entry_v1 = VersionEntry(
            version_id="ver_001",
            model_name="maintenance-lora-v1",
            discipline_id="disc_maint",
            training_run_id="run_001",
            evaluation_run_id="eval_001",
            adapter_path="/path/to/adapter_v1",
            change_type=ChangeType.RETRAIN,
            change_description="Initial training",
            created_at=__import__("datetime").datetime(2026, 1, 1),
            is_active=True,
        )
        vm.register_version(entry_v1)

        entry_v2 = VersionEntry(
            version_id="ver_002",
            model_name="maintenance-lora-v2",
            discipline_id="disc_maint",
            training_run_id="run_002",
            evaluation_run_id="eval_002",
            adapter_path="/path/to/adapter_v2",
            change_type=ChangeType.CURRICULUM_UPDATE,
            change_description="Added 50 examples",
            created_at=__import__("datetime").datetime(2026, 1, 15),
            is_active=False,
        )
        vm.register_version(entry_v2)

        versions = vm.list_versions(discipline_id="disc_maint")
        assert len(versions) == 2

        active = vm.get_active_version("disc_maint")
        assert active is not None
        assert active.version_id == "ver_001"

    def test_version_manager_rollback(self, tmp_path: Path) -> None:
        """VersionManager.rollback restores previous active version."""
        vm = VersionManager(tmp_path / "versions")

        entry_v1 = VersionEntry(
            version_id="ver_001",
            model_name="v1",
            discipline_id="disc_maint",
            training_run_id="run_001",
            evaluation_run_id="eval_001",
            adapter_path=None,
            change_type=ChangeType.RETRAIN,
            change_description="v1",
            created_at=__import__("datetime").datetime(2026, 1, 1),
            is_active=False,
        )
        entry_v2 = VersionEntry(
            version_id="ver_002",
            model_name="v2",
            discipline_id="disc_maint",
            training_run_id="run_002",
            evaluation_run_id="eval_002",
            adapter_path=None,
            change_type=ChangeType.RETRAIN,
            change_description="v2",
            created_at=__import__("datetime").datetime(2026, 1, 15),
            is_active=True,
        )
        vm.register_version(entry_v1)
        vm.register_version(entry_v2)

        rolled_back = vm.rollback("disc_maint")
        assert rolled_back is not None
        assert rolled_back.version_id == "ver_001"
        assert rolled_back.is_active is True

    def test_regression_runner_saves_report(self, tmp_path: Path) -> None:
        """RegressionRunner persists and loads regression reports."""
        baseline = self._make_eval_report("eval_base", "disc_001", "comp_proc", "Procedural", 8, 10)
        current = self._make_eval_report("eval_curr", "disc_001", "comp_proc", "Procedural", 7, 10)

        checker = RegressionChecker()
        vm = VersionManager(tmp_path / "versions")
        runner = RegressionRunner(checker, vm, tmp_path / "reports")

        report = runner.run_regression_check(baseline, current, ChangeType.RETRAIN)
        saved_path = runner.save_report(report)
        assert saved_path.exists()

        loaded = runner.load_report(report.report_id)
        assert loaded.report_id == report.report_id
        assert loaded.overall_verdict == report.overall_verdict


# ===================================================================
# TestFoundryMergingFlow
# ===================================================================


class TestFoundryMergingFlow:
    """Verify adapter merging pipeline."""

    def test_linear_merge_two_adapters(self, tmp_path: Path) -> None:
        """MergePipeline merges two compatible adapters with LINEAR method."""
        adapter_a_path = tmp_path / "adapter_a"
        adapter_b_path = tmp_path / "adapter_b"
        adapter_a_path.mkdir()
        adapter_b_path.mkdir()

        adapter_a = AdapterInfo(
            adapter_path=adapter_a_path,
            discipline_id="disc_maint",
            discipline_name="Military Maintenance",
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family="phi",
            lora_rank=16,
            training_run_id="run_001",
        )
        adapter_b = AdapterInfo(
            adapter_path=adapter_b_path,
            discipline_id="disc_safety",
            discipline_name="Safety Procedures",
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family="phi",
            lora_rank=16,
            training_run_id="run_002",
        )

        config = MergeConfig(
            method=MergeMethod.LINEAR,
            weights=[0.6, 0.4],
            output_dir=tmp_path / "merged",
        )
        pipeline = MergePipeline(config=config)
        result = pipeline.merge([adapter_a, adapter_b])

        assert result.status == MergeStatus.COMPLETED
        assert result.merged_adapter_path is not None
        assert result.merged_adapter_path.exists()
        assert len(result.adapters) == 2
        assert len(result.weights_used) == 2
        assert result.plain_language_summary
        assert "Military Maintenance" in result.plain_language_summary

    def test_ties_merge(self, tmp_path: Path) -> None:
        """MergePipeline merges adapters with TIES method."""
        adapter_a_path = tmp_path / "ties_a"
        adapter_b_path = tmp_path / "ties_b"
        adapter_a_path.mkdir()
        adapter_b_path.mkdir()

        adapter_a = AdapterInfo(
            adapter_path=adapter_a_path,
            discipline_id="disc_001",
            discipline_name="Discipline A",
            base_model="meta-llama/Llama-3-8B-Instruct",
            base_model_family="llama",
            lora_rank=16,
        )
        adapter_b = AdapterInfo(
            adapter_path=adapter_b_path,
            discipline_id="disc_002",
            discipline_name="Discipline B",
            base_model="meta-llama/Llama-3-8B-Instruct",
            base_model_family="llama",
            lora_rank=16,
        )

        config = MergeConfig(
            method=MergeMethod.TIES,
            ties_density=0.3,
            output_dir=tmp_path / "ties_merged",
        )
        pipeline = MergePipeline(config=config)
        result = pipeline.merge([adapter_a, adapter_b])

        assert result.status == MergeStatus.COMPLETED
        assert result.method == MergeMethod.TIES
        # Check metadata file was written
        metadata_path = result.merged_adapter_path / "merge_metadata.json"
        assert metadata_path.exists()

        with open(metadata_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        assert meta["method"] == "ties"
        assert meta["ties_density"] == 0.3

    def test_merge_incompatible_base_models_rejected(self, tmp_path: Path) -> None:
        """MergePipeline raises MergingError for mismatched base models."""
        from foundry.src.merging import MergingError

        adapter_a = AdapterInfo(
            adapter_path=tmp_path / "a",
            discipline_id="disc_001",
            discipline_name="A",
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family="phi",
            lora_rank=16,
        )
        adapter_b = AdapterInfo(
            adapter_path=tmp_path / "b",
            discipline_id="disc_002",
            discipline_name="B",
            base_model="meta-llama/Llama-3-8B-Instruct",
            base_model_family="llama",
            lora_rank=16,
        )

        pipeline = MergePipeline(config=MergeConfig())
        with pytest.raises(MergingError, match="Compatibility check failed"):
            pipeline.merge([adapter_a, adapter_b])


# ===================================================================
# TestFullPipelineE2E
# ===================================================================


class TestFullPipelineE2E:
    """End-to-end pipeline: Quarry -> Forge -> Foundry."""

    def test_full_pipeline_no_exceptions(
        self,
        tmp_path: Path,
        forge_storage: ForgeStorage,
        discipline: Discipline,
        contributor: Contributor,
        competencies: list[Competency],
    ) -> None:
        """Complete pipeline runs without exceptions and data flows correctly."""
        # --- Stage 1: Quarry produces ChonkRecords ---
        chunks = _make_sample_chonk_records()
        records = [ChonkRecord.from_dict(c) for c in chunks]
        assert len(records) == 3
        assert all(r.content for r in records)

        # --- Stage 2: QuarryBridge scaffolds candidates ---
        bridge = QuarryBridge(forge_storage)
        sources = bridge.ingest_chunks(chunks, "TM-1-1500")
        candidates = bridge.scaffold_examples(sources, discipline.id)
        assert len(candidates) >= 1

        # Accept first candidate
        accepted = bridge.accept_candidate(
            candidate=candidates[0],
            contributor_id=contributor.id,
            discipline_id=discipline.id,
            competency_id=competencies[0].id,
        )
        assert forge_storage.get_example(accepted.id) is not None

        # --- Stage 3: Populate more examples for training ---
        _populate_examples(forge_storage, discipline, contributor, competencies)

        # --- Stage 4: Export JSONL for Foundry ---
        jsonl_path = tmp_path / "full_pipeline.jsonl"
        forge_storage.export_to_jsonl(discipline.id, jsonl_path)
        assert jsonl_path.exists()

        loader = CurriculumLoader()
        curriculum = loader.load(jsonl_path)
        assert len(curriculum) >= 8  # at least 8 training examples

        # --- Stage 5: Train (dry-run) ---
        output_dir = tmp_path / "e2e_training"
        config = TrainingConfig(
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family=BaseModelFamily.PHI,
            curriculum_path=jsonl_path,
            output_dir=output_dir,
            epochs=1,
            batch_size=4,
        )
        pipeline = TrainingPipeline(config)
        pipeline.prepare()
        result = pipeline.run()

        assert result.status == TrainingStatus.COMPLETED
        assert result.adapter_path is not None

        # --- Stage 6: Evaluate against test set ---
        test_set_path = tmp_path / "test_set.jsonl"
        forge_storage.export_test_set_jsonl(discipline.id, test_set_path)
        assert test_set_path.exists()

        runner = EvaluationRunner()
        test_cases = runner.load_test_cases(test_set_path)
        assert len(test_cases) == 2

        # Mock model returns exact answers for perfect scores
        responses = {tc.question: tc.expected_answer for tc in test_cases}
        model = MockInference(responses=responses)

        comp_names = {c.id: c.name for c in competencies}
        report = runner.run_evaluation(
            model=model,
            test_cases=test_cases,
            competency_names=comp_names,
            model_name="e2e-test-model",
            discipline_id=discipline.id,
        )

        assert report.status == EvaluationStatus.COMPLETED
        assert report.overall_accuracy == 1.0
        assert report.plain_language_summary

        # --- Stage 7: RAG integration ---
        mock_chunks = [
            {
                "text": records[0].content,
                "metadata": {
                    "chunk_id": records[0].id,
                    "document_title": "TM-1-1500-328",
                    "section": "Oil Filter Replacement",
                    "page": 42,
                },
                "score": 0.95,
            },
        ]
        retrieval = MockRetrievalAdapter(chunks=mock_chunks)
        rag_model = MockInference(default_response="Replace filter per TM.")
        rag_pipeline = RAGPipeline(
            model=rag_model,
            retrieval=retrieval,
            model_name="e2e-rag",
        )
        rag_response = rag_pipeline.query("How to replace oil filter?")

        assert rag_response.answer
        assert len(rag_response.citations) == 1
        assert rag_response.model_name == "e2e-rag"

    def test_curriculum_version_snapshot(
        self,
        forge_storage: ForgeStorage,
        discipline: Discipline,
        contributor: Contributor,
        competencies: list[Competency],
    ) -> None:
        """Curriculum versioning snapshots training examples correctly."""
        _populate_examples(forge_storage, discipline, contributor, competencies)

        version = forge_storage.create_curriculum_version(discipline.id, contributor.id)

        assert version.version_number == 1
        assert version.example_count == 8  # 4 training per comp x 2 comps

        snapshot_examples = version.get_examples()
        assert len(snapshot_examples) == 8

        # Second version should be v2
        version2 = forge_storage.create_curriculum_version(discipline.id, contributor.id)
        assert version2.version_number == 2

    def test_coverage_report_integration(
        self,
        forge_storage: ForgeStorage,
        discipline: Discipline,
        contributor: Contributor,
        competencies: list[Competency],
    ) -> None:
        """Coverage report reflects actual example counts per competency."""
        _populate_examples(forge_storage, discipline, contributor, competencies)

        report = forge_storage.get_coverage_report(discipline.id)

        assert report["discipline_id"] == discipline.id
        assert report["total_examples"] == 8  # 4 training per comp x 2
        assert report["total_test_examples"] == 2
        assert len(report["competency_coverage"]) == 2

        for entry in report["competency_coverage"]:
            assert entry["example_count"] == 4
            assert entry["coverage_target"] == 5
            assert entry["met"] is False

    def test_evaluation_report_serialization(
        self,
        tmp_path: Path,
        forge_storage: ForgeStorage,
        discipline: Discipline,
        contributor: Contributor,
        competencies: list[Competency],
    ) -> None:
        """EvaluationReport round-trips through JSON correctly."""
        from foundry.src.evaluation import EvaluationHistory

        examples = _populate_examples(forge_storage, discipline, contributor, competencies)
        test_examples = [ex for ex in examples if ex.is_test_set]
        test_cases = [
            TestCase(
                example_id=ex.id,
                question=ex.question,
                expected_answer=ex.ideal_answer,
                competency_id=ex.competency_id,
                discipline_id=ex.discipline_id,
            )
            for ex in test_examples
        ]

        responses = {tc.question: tc.expected_answer for tc in test_cases}
        model = MockInference(responses=responses)
        comp_names = {c.id: c.name for c in competencies}

        runner = EvaluationRunner()
        report = runner.run_evaluation(
            model=model,
            test_cases=test_cases,
            competency_names=comp_names,
            model_name="serialization-test",
            discipline_id=discipline.id,
        )

        history = EvaluationHistory(tmp_path / "eval_history")
        saved_path = history.save_report(report)
        assert saved_path.exists()

        loaded = history.load_report(report.run_id)
        assert loaded.run_id == report.run_id
        assert loaded.overall_accuracy == report.overall_accuracy
        assert loaded.total_cases == report.total_cases
        assert len(loaded.competency_scores) == len(report.competency_scores)
