"""Tests for real-time coverage analysis."""

from __future__ import annotations

import pytest

from forge.src.coverage import (
    CompetencyCoverage,
    CoverageAnalysisError,
    CoverageAnalyzer,
    CoverageRecommendation,
    CoverageSnapshot,
)
from forge.src.models import Competency, Example
from forge.src.storage import ForgeStorage

# --- Fixtures ---


@pytest.fixture
def analyzer(populated_store: ForgeStorage) -> CoverageAnalyzer:
    """CoverageAnalyzer with a populated store."""
    return CoverageAnalyzer(populated_store)


@pytest.fixture
def multi_comp_store(populated_store: ForgeStorage) -> ForgeStorage:
    """Store with multiple competencies at various coverage levels."""
    # populated_store already has comp_test001 (Fault Isolation, target=25, 1 example)
    # Add more competencies with varying states
    populated_store.create_competency(
        Competency(
            id="comp_procedures",
            name="Procedural Comprehension",
            description="Understanding maintenance procedures",
            discipline_id="disc_test001",
            coverage_target=20,
        )
    )
    populated_store.create_competency(
        Competency(
            id="comp_safety",
            name="Safety Protocols",
            description="Safety procedures and PPE",
            discipline_id="disc_test001",
            coverage_target=15,
        )
    )
    populated_store.create_competency(
        Competency(
            id="comp_tools",
            name="Tool Identification",
            description="Identifying and using tools",
            discipline_id="disc_test001",
            coverage_target=10,
        )
    )

    # Add examples to reach different coverage levels
    # comp_procedures: 12 examples (60% of 20)
    for i in range(12):
        populated_store.create_example(
            Example(
                id=f"ex_proc_{i:03d}",
                question=f"Procedure question {i}",
                ideal_answer=f"Procedure answer {i}",
                competency_id="comp_procedures",
                contributor_id="contrib_test001",
                discipline_id="disc_test001",
            )
        )

    # comp_safety: 15 examples (100% of 15 = met)
    for i in range(15):
        populated_store.create_example(
            Example(
                id=f"ex_safe_{i:03d}",
                question=f"Safety question {i}",
                ideal_answer=f"Safety answer {i}",
                competency_id="comp_safety",
                contributor_id="contrib_test001",
                discipline_id="disc_test001",
            )
        )

    # comp_tools: 0 examples (empty)
    # comp_test001 (Fault Isolation): 1 example (4% of 25)

    return populated_store


@pytest.fixture
def multi_analyzer(multi_comp_store: ForgeStorage) -> CoverageAnalyzer:
    """CoverageAnalyzer with multi-competency store."""
    return CoverageAnalyzer(multi_comp_store)


# --- CompetencyCoverage Tests ---


class TestCompetencyCoverage:
    """Tests for CompetencyCoverage dataclass."""

    def test_construction(self) -> None:
        """Test basic construction."""
        cc = CompetencyCoverage(
            competency_id="comp_1",
            competency_name="Fault Isolation",
            parent_id=None,
            example_count=10,
            coverage_target=25,
            coverage_ratio=0.4,
            is_met=False,
            gap=15,
        )
        assert cc.competency_name == "Fault Isolation"
        assert cc.gap == 15

    def test_to_dict(self) -> None:
        """Test serialization."""
        cc = CompetencyCoverage(
            competency_id="comp_1",
            competency_name="Test",
            parent_id="comp_parent",
            example_count=5,
            coverage_target=10,
            coverage_ratio=0.5,
            is_met=False,
            gap=5,
        )
        d = cc.to_dict()
        assert d["competency_id"] == "comp_1"
        assert d["parent_id"] == "comp_parent"
        assert d["coverage_ratio"] == 0.5


# --- CoverageRecommendation Tests ---


class TestCoverageRecommendation:
    """Tests for CoverageRecommendation dataclass."""

    def test_construction(self) -> None:
        """Test basic construction."""
        rec = CoverageRecommendation(
            competency_id="comp_1",
            competency_name="Test",
            priority=1,
            reason="No examples yet",
            examples_needed=25,
        )
        assert rec.priority == 1
        assert rec.examples_needed == 25

    def test_to_dict(self) -> None:
        """Test serialization."""
        rec = CoverageRecommendation(
            competency_id="comp_1",
            competency_name="Test",
            priority=2,
            reason="Low coverage",
            examples_needed=10,
        )
        d = rec.to_dict()
        assert d["priority"] == 2
        assert d["reason"] == "Low coverage"


# --- CoverageSnapshot Tests ---


class TestCoverageSnapshot:
    """Tests for CoverageSnapshot dataclass."""

    def test_construction(self) -> None:
        """Test basic construction."""
        snap = CoverageSnapshot(
            discipline_id="disc_1",
            total_competencies=5,
            met_count=2,
            gap_count=3,
            overall_ratio=0.45,
            total_examples=50,
            total_test_examples=10,
            total_gap=75,
            coverage_complete=False,
        )
        assert snap.gap_count == 3
        assert not snap.coverage_complete

    def test_to_dict(self) -> None:
        """Test serialization includes all fields."""
        snap = CoverageSnapshot(
            discipline_id="disc_1",
            total_competencies=1,
            met_count=0,
            gap_count=1,
            overall_ratio=0.0,
            total_examples=0,
            total_test_examples=0,
            total_gap=25,
            coverage_complete=False,
            competencies=[
                CompetencyCoverage(
                    competency_id="c1",
                    competency_name="Test",
                    parent_id=None,
                    example_count=0,
                    coverage_target=25,
                    coverage_ratio=0.0,
                    is_met=False,
                    gap=25,
                )
            ],
        )
        d = snap.to_dict()
        assert "competencies" in d
        assert "recommendations" in d
        assert len(d["competencies"]) == 1


# --- CoverageAnalyzer Basic Tests ---


class TestCoverageAnalyzerBasic:
    """Tests for basic CoverageAnalyzer functionality."""

    def test_construction(self, populated_store: ForgeStorage) -> None:
        """Test analyzer can be constructed."""
        analyzer = CoverageAnalyzer(populated_store)
        assert analyzer is not None

    def test_analyze_returns_snapshot(self, analyzer: CoverageAnalyzer) -> None:
        """Test analyze returns a CoverageSnapshot."""
        snapshot = analyzer.analyze("disc_test001")
        assert isinstance(snapshot, CoverageSnapshot)

    def test_empty_discipline_raises(self, memory_store: ForgeStorage) -> None:
        """Test analyzing a discipline with no competencies raises error."""
        analyzer = CoverageAnalyzer(memory_store)
        with pytest.raises(CoverageAnalysisError, match="No competencies"):
            analyzer.analyze("nonexistent")

    def test_single_competency_snapshot(self, analyzer: CoverageAnalyzer) -> None:
        """Test snapshot for a discipline with one competency."""
        snapshot = analyzer.analyze("disc_test001")
        assert snapshot.total_competencies == 1
        assert snapshot.discipline_id == "disc_test001"
        assert snapshot.total_examples == 1


# --- Coverage Metrics Tests ---


class TestCoverageMetrics:
    """Tests for coverage metric calculations."""

    def test_gap_count(self, multi_analyzer: CoverageAnalyzer) -> None:
        """Test gap count is correct."""
        snapshot = multi_analyzer.analyze("disc_test001")
        # comp_safety is met, others are not
        assert snapshot.met_count == 1
        assert snapshot.gap_count == 3

    def test_total_gap(self, multi_analyzer: CoverageAnalyzer) -> None:
        """Test total gap across all competencies."""
        snapshot = multi_analyzer.analyze("disc_test001")
        # Fault: 25-1=24, Procedures: 20-12=8, Tools: 10-0=10, Safety: 0
        assert snapshot.total_gap == 24 + 8 + 10

    def test_overall_ratio(self, multi_analyzer: CoverageAnalyzer) -> None:
        """Test overall ratio is weighted by target."""
        snapshot = multi_analyzer.analyze("disc_test001")
        # Total target: 25+20+15+10=70
        # Achieved (capped): min(1,25)+min(12,20)+min(15,15)+min(0,10) = 1+12+15+0=28
        # Ratio: 28/70 = 0.4
        assert snapshot.overall_ratio == 0.4

    def test_coverage_complete_false(self, multi_analyzer: CoverageAnalyzer) -> None:
        """Test coverage_complete is False when gaps exist."""
        snapshot = multi_analyzer.analyze("disc_test001")
        assert not snapshot.coverage_complete

    def test_per_competency_ratios(self, multi_analyzer: CoverageAnalyzer) -> None:
        """Test individual competency coverage ratios."""
        snapshot = multi_analyzer.analyze("disc_test001")
        by_id = {c.competency_id: c for c in snapshot.competencies}

        assert by_id["comp_test001"].coverage_ratio == 0.04  # 1/25
        assert by_id["comp_procedures"].coverage_ratio == 0.6  # 12/20
        assert by_id["comp_safety"].coverage_ratio == 1.0  # 15/15
        assert by_id["comp_tools"].coverage_ratio == 0.0  # 0/10

    def test_is_met_flags(self, multi_analyzer: CoverageAnalyzer) -> None:
        """Test is_met flags per competency."""
        snapshot = multi_analyzer.analyze("disc_test001")
        by_id = {c.competency_id: c for c in snapshot.competencies}

        assert not by_id["comp_test001"].is_met
        assert not by_id["comp_procedures"].is_met
        assert by_id["comp_safety"].is_met
        assert not by_id["comp_tools"].is_met


# --- Recommendation Tests ---


class TestRecommendations:
    """Tests for coverage recommendations."""

    def test_recommendations_generated(self, multi_analyzer: CoverageAnalyzer) -> None:
        """Test recommendations are generated for gaps."""
        snapshot = multi_analyzer.analyze("disc_test001")
        assert len(snapshot.recommendations) > 0

    def test_no_recommendations_for_met(self, multi_analyzer: CoverageAnalyzer) -> None:
        """Test no recommendation for met competencies."""
        snapshot = multi_analyzer.analyze("disc_test001")
        rec_ids = [r.competency_id for r in snapshot.recommendations]
        assert "comp_safety" not in rec_ids

    def test_empty_competency_priority_1(self, multi_analyzer: CoverageAnalyzer) -> None:
        """Test empty competencies get priority 1."""
        snapshot = multi_analyzer.analyze("disc_test001")
        tools_rec = next(r for r in snapshot.recommendations if r.competency_id == "comp_tools")
        assert tools_rec.priority == 1
        assert "No examples" in tools_rec.reason

    def test_low_coverage_priority_2(self, multi_analyzer: CoverageAnalyzer) -> None:
        """Test < 50% coverage gets priority 2."""
        snapshot = multi_analyzer.analyze("disc_test001")
        fault_rec = next(r for r in snapshot.recommendations if r.competency_id == "comp_test001")
        assert fault_rec.priority == 2

    def test_medium_coverage_priority_3(self, multi_analyzer: CoverageAnalyzer) -> None:
        """Test 50-99% coverage gets priority 3."""
        snapshot = multi_analyzer.analyze("disc_test001")
        proc_rec = next(r for r in snapshot.recommendations if r.competency_id == "comp_procedures")
        assert proc_rec.priority == 3
        assert "Nearly there" in proc_rec.reason

    def test_recommendations_sorted(self, multi_analyzer: CoverageAnalyzer) -> None:
        """Test recommendations sorted by priority then gap."""
        snapshot = multi_analyzer.analyze("disc_test001")
        priorities = [r.priority for r in snapshot.recommendations]
        assert priorities == sorted(priorities)

    def test_examples_needed_in_recommendations(self, multi_analyzer: CoverageAnalyzer) -> None:
        """Test examples_needed reflects actual gap."""
        snapshot = multi_analyzer.analyze("disc_test001")
        tools_rec = next(r for r in snapshot.recommendations if r.competency_id == "comp_tools")
        assert tools_rec.examples_needed == 10


# --- Helper Method Tests ---


class TestHelperMethods:
    """Tests for helper methods on CoverageAnalyzer."""

    def test_get_competency_coverage(self, multi_analyzer: CoverageAnalyzer) -> None:
        """Test getting coverage for a single competency."""
        cc = multi_analyzer.get_competency_coverage("disc_test001", "comp_safety")
        assert cc is not None
        assert cc.is_met is True

    def test_get_competency_coverage_not_found(self, multi_analyzer: CoverageAnalyzer) -> None:
        """Test getting coverage for nonexistent competency."""
        cc = multi_analyzer.get_competency_coverage("disc_test001", "comp_nonexistent")
        assert cc is None

    def test_is_discipline_ready_false(self, multi_analyzer: CoverageAnalyzer) -> None:
        """Test discipline readiness when incomplete."""
        assert not multi_analyzer.is_discipline_ready("disc_test001")

    def test_is_discipline_ready_nonexistent(self, memory_store: ForgeStorage) -> None:
        """Test discipline readiness for nonexistent discipline."""
        analyzer = CoverageAnalyzer(memory_store)
        assert not analyzer.is_discipline_ready("nonexistent")


# --- Full Coverage Tests ---


class TestFullCoverage:
    """Tests for fully-covered disciplines."""

    def test_all_met(self, populated_store: ForgeStorage) -> None:
        """Test snapshot when all competencies meet targets."""
        # Set target to 1 so the existing example meets it
        comp = populated_store.get_competency("comp_test001")
        assert comp is not None
        comp.coverage_target = 1
        populated_store.update_competency(comp)

        analyzer = CoverageAnalyzer(populated_store)
        snapshot = analyzer.analyze("disc_test001")
        assert snapshot.coverage_complete is True
        assert snapshot.gap_count == 0
        assert snapshot.total_gap == 0
        assert snapshot.overall_ratio == 1.0
        assert len(snapshot.recommendations) == 0

    def test_is_ready_when_all_met(self, populated_store: ForgeStorage) -> None:
        """Test is_discipline_ready returns True when fully covered."""
        comp = populated_store.get_competency("comp_test001")
        assert comp is not None
        comp.coverage_target = 1
        populated_store.update_competency(comp)
        analyzer = CoverageAnalyzer(populated_store)
        assert analyzer.is_discipline_ready("disc_test001")


# --- Serialization Integration Tests ---


class TestSerializationIntegration:
    """Tests for full snapshot serialization."""

    def test_full_snapshot_to_dict(self, multi_analyzer: CoverageAnalyzer) -> None:
        """Test complete snapshot serializes correctly."""
        snapshot = multi_analyzer.analyze("disc_test001")
        d = snapshot.to_dict()

        assert d["discipline_id"] == "disc_test001"
        assert d["total_competencies"] == 4
        assert isinstance(d["competencies"], list)
        assert isinstance(d["recommendations"], list)
        assert len(d["competencies"]) == 4
        assert d["overall_ratio"] == 0.4

    def test_round_trip_data(self, multi_analyzer: CoverageAnalyzer) -> None:
        """Test all data survives serialization."""
        snapshot = multi_analyzer.analyze("disc_test001")
        d = snapshot.to_dict()

        # Verify competencies have all fields
        for comp_dict in d["competencies"]:
            assert "competency_id" in comp_dict
            assert "coverage_ratio" in comp_dict
            assert "gap" in comp_dict

        # Verify recommendations have all fields
        for rec_dict in d["recommendations"]:
            assert "priority" in rec_dict
            assert "reason" in rec_dict
            assert "examples_needed" in rec_dict
