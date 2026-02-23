"""Tests for competency mapping system."""

from __future__ import annotations

import pytest

from forge.src.competency import (
    CompetencyMapError,
    CompetencyMapper,
    CompetencyMapSummary,
)
from forge.src.models import (
    Competency,
    Contributor,
    Discipline,
    DisciplineStatus,
    DiscoverySession,
    Example,
    SessionStatus,
)
from forge.src.storage import ForgeStorage

# --- Fixtures ---


@pytest.fixture
def store() -> ForgeStorage:
    """In-memory store with schema initialized."""
    s = ForgeStorage(":memory:")
    s.initialize_schema()
    return s


@pytest.fixture
def contributor(store: ForgeStorage) -> Contributor:
    """Persisted contributor."""
    c = Contributor(id="contrib_map01", name="Bob Expert")
    store.create_contributor(c)
    return c


@pytest.fixture
def discipline(store: ForgeStorage, contributor: Contributor) -> Discipline:
    """Persisted discipline."""
    d = Discipline(
        id="disc_map01",
        name="Test Discipline",
        description="For mapping tests",
        status=DisciplineStatus.DRAFT,
        created_by=contributor.id,
    )
    store.create_discipline(d)
    return d


@pytest.fixture
def seeded_competencies(store: ForgeStorage, discipline: Discipline) -> list[Competency]:
    """Three roots and one child, mimicking discovery seeds."""
    root1 = Competency(
        id="comp_root1",
        name="Fault Isolation",
        description="Identify and isolate equipment faults",
        discipline_id=discipline.id,
        coverage_target=25,
    )
    root2 = Competency(
        id="comp_root2",
        name="Preventive Maintenance",
        description="Scheduled maintenance procedures",
        discipline_id=discipline.id,
        coverage_target=30,
    )
    root3 = Competency(
        id="comp_root3",
        name="Safety Protocols",
        description="Safety procedures and hazard awareness",
        discipline_id=discipline.id,
        coverage_target=20,
    )
    child1 = Competency(
        id="comp_child1",
        name="Hydraulic Faults",
        description="Hydraulic system fault isolation",
        discipline_id=discipline.id,
        parent_id=root1.id,
        coverage_target=15,
    )
    for c in [root1, root2, root3, child1]:
        store.create_competency(c)
    return [root1, root2, root3, child1]


@pytest.fixture
def mapper(store: ForgeStorage) -> CompetencyMapper:
    """Mapper instance."""
    return CompetencyMapper(store)


@pytest.fixture
def completed_session(discipline: Discipline, contributor: Contributor) -> DiscoverySession:
    """A completed discovery session linked to the discipline."""
    return DiscoverySession(
        id="dsess_map01",
        discipline_name=discipline.name,
        contributor_id=contributor.id,
        status=SessionStatus.COMPLETED,
        generated_discipline_id=discipline.id,
    )


# --- Tests ---


class TestCompetencyMapSummary:
    """Tests for CompetencyMapSummary dataclass."""

    def test_construction(self) -> None:
        """Test full construction with all fields."""
        s = CompetencyMapSummary(
            discipline_id="disc_1",
            total_competencies=5,
            root_competencies=3,
            child_competencies=2,
            total_coverage_target=100,
            coverage_complete=False,
            gap_competency_names=["A", "B"],
            estimated_examples_needed=50,
        )
        assert s.total_competencies == 5
        assert s.root_competencies == 3
        assert not s.coverage_complete

    def test_defaults(self) -> None:
        """Test default values for optional fields."""
        s = CompetencyMapSummary(
            discipline_id="disc_1",
            total_competencies=0,
            root_competencies=0,
            child_competencies=0,
            total_coverage_target=0,
            coverage_complete=True,
        )
        assert s.gap_competency_names == []
        assert s.estimated_examples_needed == 0

    def test_gap_names_mutable(self) -> None:
        """Test that gap names list is mutable."""
        s = CompetencyMapSummary(
            discipline_id="disc_1",
            total_competencies=2,
            root_competencies=2,
            child_competencies=0,
            total_coverage_target=50,
            coverage_complete=False,
            gap_competency_names=["A"],
            estimated_examples_needed=25,
        )
        s.gap_competency_names.append("B")
        assert len(s.gap_competency_names) == 2


class TestLoadFromDiscovery:
    """Tests for load_from_discovery."""

    def test_returns_seeded_competencies(
        self,
        mapper: CompetencyMapper,
        discipline: Discipline,
        seeded_competencies: list[Competency],
        completed_session: DiscoverySession,
    ) -> None:
        """Test loading competencies seeded by discovery."""
        result = mapper.load_from_discovery(discipline.id, completed_session)
        assert len(result) == 4
        ids = {c.id for c in result}
        assert "comp_root1" in ids

    def test_wrong_session_raises(
        self,
        mapper: CompetencyMapper,
        discipline: Discipline,
        seeded_competencies: list[Competency],
        contributor: Contributor,
    ) -> None:
        """Test that mismatched session raises error."""
        wrong_session = DiscoverySession(
            id="dsess_wrong",
            discipline_name="Other",
            contributor_id=contributor.id,
            generated_discipline_id="disc_other",
        )
        with pytest.raises(CompetencyMapError, match="did not generate"):
            mapper.load_from_discovery(discipline.id, wrong_session)

    def test_no_competencies_raises(
        self,
        mapper: CompetencyMapper,
        discipline: Discipline,
        completed_session: DiscoverySession,
    ) -> None:
        """Test that empty competency list raises error."""
        with pytest.raises(CompetencyMapError, match="No competencies"):
            mapper.load_from_discovery(discipline.id, completed_session)


class TestAddCompetency:
    """Tests for add_competency."""

    def test_adds_to_storage(
        self,
        mapper: CompetencyMapper,
        store: ForgeStorage,
        discipline: Discipline,
    ) -> None:
        """Test that added competency persists in storage."""
        comp = mapper.add_competency(discipline.id, "New Skill", "A new skill")
        assert comp.name == "New Skill"
        assert store.get_competency(comp.id) is not None

    def test_with_parent(
        self,
        mapper: CompetencyMapper,
        discipline: Discipline,
        seeded_competencies: list[Competency],
    ) -> None:
        """Test adding competency with parent relationship."""
        comp = mapper.add_competency(
            discipline.id,
            "Sub Skill",
            "Under root1",
            parent_id="comp_root1",
        )
        assert comp.parent_id == "comp_root1"

    def test_bad_discipline_raises(
        self,
        mapper: CompetencyMapper,
    ) -> None:
        """Test that nonexistent discipline raises error."""
        with pytest.raises(CompetencyMapError, match="Discipline not found"):
            mapper.add_competency("disc_nonexistent", "X", "Y")

    def test_bad_parent_raises(
        self,
        mapper: CompetencyMapper,
        discipline: Discipline,
    ) -> None:
        """Test that nonexistent parent raises error."""
        with pytest.raises(CompetencyMapError, match="Parent competency not found"):
            mapper.add_competency(discipline.id, "X", "Y", parent_id="comp_nonexistent")

    def test_parent_wrong_discipline_raises(
        self,
        mapper: CompetencyMapper,
        store: ForgeStorage,
        discipline: Discipline,
        contributor: Contributor,
        seeded_competencies: list[Competency],
    ) -> None:
        """Test that parent from another discipline raises error."""
        other_disc = Discipline(
            id="disc_other",
            name="Other",
            description="Another discipline",
            created_by=contributor.id,
        )
        store.create_discipline(other_disc)
        with pytest.raises(CompetencyMapError, match="belongs to discipline"):
            mapper.add_competency(other_disc.id, "X", "Y", parent_id="comp_root1")

    def test_default_coverage_target(
        self,
        mapper: CompetencyMapper,
        discipline: Discipline,
    ) -> None:
        """Test that default coverage target is 25."""
        comp = mapper.add_competency(discipline.id, "A", "B")
        assert comp.coverage_target == 25

    def test_unique_ids(
        self,
        mapper: CompetencyMapper,
        discipline: Discipline,
    ) -> None:
        """Test that generated IDs are unique."""
        c1 = mapper.add_competency(discipline.id, "A", "A desc")
        c2 = mapper.add_competency(discipline.id, "B", "B desc")
        assert c1.id != c2.id


class TestUpdateCompetency:
    """Tests for update_competency."""

    def test_update_name(
        self,
        mapper: CompetencyMapper,
        seeded_competencies: list[Competency],
    ) -> None:
        """Test updating competency name."""
        result = mapper.update_competency("comp_root1", name="Renamed")
        assert result.name == "Renamed"

    def test_update_description(
        self,
        mapper: CompetencyMapper,
        seeded_competencies: list[Competency],
    ) -> None:
        """Test updating competency description."""
        result = mapper.update_competency("comp_root1", description="New desc")
        assert result.description == "New desc"

    def test_update_coverage_target(
        self,
        mapper: CompetencyMapper,
        seeded_competencies: list[Competency],
    ) -> None:
        """Test updating coverage target."""
        result = mapper.update_competency("comp_root1", coverage_target=50)
        assert result.coverage_target == 50

    def test_partial_update_preserves_other_fields(
        self,
        mapper: CompetencyMapper,
        seeded_competencies: list[Competency],
    ) -> None:
        """Test that partial update doesn't clobber other fields."""
        result = mapper.update_competency("comp_root1", name="Renamed")
        assert result.description == ("Identify and isolate equipment faults")
        assert result.coverage_target == 25

    def test_nonexistent_raises(
        self,
        mapper: CompetencyMapper,
    ) -> None:
        """Test that updating nonexistent competency raises error."""
        with pytest.raises(CompetencyMapError, match="Competency not found"):
            mapper.update_competency("comp_nonexistent", name="X")


class TestSetParent:
    """Tests for set_parent."""

    def test_sets_parent(
        self,
        mapper: CompetencyMapper,
        seeded_competencies: list[Competency],
    ) -> None:
        """Test setting parent on a root competency."""
        result = mapper.set_parent("comp_root2", "comp_root1")
        assert result.parent_id == "comp_root1"

    def test_clears_parent(
        self,
        mapper: CompetencyMapper,
        seeded_competencies: list[Competency],
    ) -> None:
        """Test clearing parent makes competency a root."""
        result = mapper.set_parent("comp_child1", None)
        assert result.parent_id is None

    def test_parent_wrong_discipline_raises(
        self,
        mapper: CompetencyMapper,
        store: ForgeStorage,
        discipline: Discipline,
        contributor: Contributor,
        seeded_competencies: list[Competency],
    ) -> None:
        """Test that parent from another discipline raises error."""
        other_disc = Discipline(
            id="disc_other2",
            name="Other2",
            description="Another",
            created_by=contributor.id,
        )
        store.create_discipline(other_disc)
        other_comp = Competency(
            id="comp_other",
            name="Foreign",
            description="From other disc",
            discipline_id=other_disc.id,
        )
        store.create_competency(other_comp)
        with pytest.raises(CompetencyMapError, match="belongs to discipline"):
            mapper.set_parent("comp_root1", "comp_other")

    def test_self_parent_raises(
        self,
        mapper: CompetencyMapper,
        seeded_competencies: list[Competency],
    ) -> None:
        """Test that setting self as parent raises error."""
        with pytest.raises(CompetencyMapError, match="own parent"):
            mapper.set_parent("comp_root1", "comp_root1")

    def test_circular_reference_raises(
        self,
        mapper: CompetencyMapper,
        seeded_competencies: list[Competency],
    ) -> None:
        """Test that creating a cycle raises error."""
        with pytest.raises(CompetencyMapError, match="Circular reference"):
            mapper.set_parent("comp_root1", "comp_child1")

    def test_nonexistent_raises(
        self,
        mapper: CompetencyMapper,
    ) -> None:
        """Test that reparenting nonexistent competency raises error."""
        with pytest.raises(CompetencyMapError, match="Competency not found"):
            mapper.set_parent("comp_nonexistent", "comp_root1")


class TestRemoveCompetency:
    """Tests for remove_competency."""

    def test_removes_leaf(
        self,
        mapper: CompetencyMapper,
        store: ForgeStorage,
        seeded_competencies: list[Competency],
    ) -> None:
        """Test removing a leaf competency."""
        mapper.remove_competency("comp_child1")
        assert store.get_competency("comp_child1") is None

    def test_removes_with_reassignment(
        self,
        mapper: CompetencyMapper,
        store: ForgeStorage,
        seeded_competencies: list[Competency],
    ) -> None:
        """Test removing parent with child reassignment."""
        mapper.remove_competency("comp_root1", reassign_children_to="comp_root2")
        assert store.get_competency("comp_root1") is None
        child = store.get_competency("comp_child1")
        assert child is not None
        assert child.parent_id == "comp_root2"

    def test_has_children_no_reassign_raises(
        self,
        mapper: CompetencyMapper,
        seeded_competencies: list[Competency],
    ) -> None:
        """Test that removing parent without reassignment raises."""
        with pytest.raises(CompetencyMapError, match="1 children"):
            mapper.remove_competency("comp_root1")

    def test_nonexistent_raises(
        self,
        mapper: CompetencyMapper,
    ) -> None:
        """Test that removing nonexistent competency raises error."""
        with pytest.raises(CompetencyMapError, match="Competency not found"):
            mapper.remove_competency("comp_nonexistent")


class TestCoverageSummary:
    """Tests for get_coverage_summary."""

    def test_zero_examples_full_gaps(
        self,
        mapper: CompetencyMapper,
        discipline: Discipline,
        seeded_competencies: list[Competency],
    ) -> None:
        """Test that zero examples means all competencies are gaps."""
        summary = mapper.get_coverage_summary(discipline.id)
        assert summary.total_competencies == 4
        assert len(summary.gap_competency_names) == 4
        assert not summary.coverage_complete

    def test_counts_correct(
        self,
        mapper: CompetencyMapper,
        discipline: Discipline,
        seeded_competencies: list[Competency],
    ) -> None:
        """Test root/child/target counts are accurate."""
        summary = mapper.get_coverage_summary(discipline.id)
        assert summary.root_competencies == 3
        assert summary.child_competencies == 1
        assert summary.total_coverage_target == 90  # 25+30+20+15

    def test_estimated_examples_needed(
        self,
        mapper: CompetencyMapper,
        discipline: Discipline,
        seeded_competencies: list[Competency],
    ) -> None:
        """Test estimated examples needed with zero examples."""
        summary = mapper.get_coverage_summary(discipline.id)
        assert summary.estimated_examples_needed == 90

    def test_met_targets_not_in_gaps(
        self,
        mapper: CompetencyMapper,
        store: ForgeStorage,
        discipline: Discipline,
        contributor: Contributor,
        seeded_competencies: list[Competency],
    ) -> None:
        """Test that met targets are excluded from gaps."""
        for i in range(25):
            ex = Example(
                id=f"ex_fill_{i:03d}",
                question=f"Question {i}",
                ideal_answer=f"Answer {i}",
                competency_id="comp_root1",
                contributor_id=contributor.id,
                discipline_id=discipline.id,
            )
            store.create_example(ex)
        summary = mapper.get_coverage_summary(discipline.id)
        assert "Fault Isolation" not in summary.gap_competency_names
        assert summary.estimated_examples_needed == 65  # 90 - 25


class TestCompetencyTree:
    """Tests for get_competency_tree."""

    def test_flat_list_all_roots(
        self,
        mapper: CompetencyMapper,
        store: ForgeStorage,
        discipline: Discipline,
    ) -> None:
        """Test tree with only root nodes."""
        for i in range(2):
            store.create_competency(
                Competency(
                    id=f"comp_flat{i}",
                    name=f"Root {i}",
                    description=f"Root desc {i}",
                    discipline_id=discipline.id,
                )
            )
        tree = mapper.get_competency_tree(discipline.id)
        assert len(tree) == 2
        for node in tree:
            assert node["children"] == []

    def test_parent_child_nesting(
        self,
        mapper: CompetencyMapper,
        discipline: Discipline,
        seeded_competencies: list[Competency],
    ) -> None:
        """Test tree correctly nests children under parents."""
        tree = mapper.get_competency_tree(discipline.id)
        root_ids = {n["competency"].id for n in tree}
        assert "comp_root1" in root_ids
        root1_node = next(n for n in tree if n["competency"].id == "comp_root1")
        child_ids = {c["competency"].id for c in root1_node["children"]}
        assert "comp_child1" in child_ids

    def test_empty_discipline(
        self,
        mapper: CompetencyMapper,
        discipline: Discipline,
    ) -> None:
        """Test tree for discipline with no competencies."""
        tree = mapper.get_competency_tree(discipline.id)
        assert tree == []


class TestFinalizeMap:
    """Tests for finalize_map."""

    def test_valid_map_succeeds(
        self,
        mapper: CompetencyMapper,
        discipline: Discipline,
        seeded_competencies: list[Competency],
    ) -> None:
        """Test finalization of a valid competency map."""
        summary = mapper.finalize_map(discipline.id)
        assert isinstance(summary, CompetencyMapSummary)
        assert summary.total_competencies == 4

    def test_too_few_roots_raises(
        self,
        mapper: CompetencyMapper,
        store: ForgeStorage,
        discipline: Discipline,
    ) -> None:
        """Test that fewer than 3 roots fails validation."""
        for i in range(2):
            store.create_competency(
                Competency(
                    id=f"comp_few{i}",
                    name=f"Root {i}",
                    description=f"Desc {i}",
                    discipline_id=discipline.id,
                )
            )
        with pytest.raises(CompetencyMapError, match="at least 3"):
            mapper.finalize_map(discipline.id)

    def test_name_equals_description_raises(
        self,
        mapper: CompetencyMapper,
        store: ForgeStorage,
        discipline: Discipline,
    ) -> None:
        """Test that unrefined competency (name==desc) fails."""
        for i in range(3):
            name = f"Skill {i}" if i < 2 else "Same"
            desc = f"Description for skill {i}" if i < 2 else "Same"
            store.create_competency(
                Competency(
                    id=f"comp_same{i}",
                    name=name,
                    description=desc,
                    discipline_id=discipline.id,
                )
            )
        with pytest.raises(CompetencyMapError, match="name == description"):
            mapper.finalize_map(discipline.id)

    def test_zero_coverage_target_raises(
        self,
        mapper: CompetencyMapper,
        store: ForgeStorage,
        discipline: Discipline,
    ) -> None:
        """Test that zero coverage target fails validation."""
        for i in range(3):
            target = 25 if i < 2 else 0
            store.create_competency(
                Competency(
                    id=f"comp_zero{i}",
                    name=f"Root {i}",
                    description=f"Desc for root {i}",
                    discipline_id=discipline.id,
                    coverage_target=target,
                )
            )
        with pytest.raises(CompetencyMapError, match="coverage_target <= 0"):
            mapper.finalize_map(discipline.id)

    def test_multiple_violations_collected(
        self,
        mapper: CompetencyMapper,
        store: ForgeStorage,
        discipline: Discipline,
    ) -> None:
        """Test that all violations are collected in one error."""
        store.create_competency(
            Competency(
                id="comp_multi",
                name="Bad",
                description="Bad",
                discipline_id=discipline.id,
                coverage_target=0,
            )
        )
        with pytest.raises(CompetencyMapError) as exc_info:
            mapper.finalize_map(discipline.id)
        msg = str(exc_info.value)
        assert "at least 3" in msg
        assert "name == description" in msg
        assert "coverage_target <= 0" in msg
