"""Tests for held-out test set reservation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.src.models import Competency, Example
from forge.src.storage import ForgeStorage
from forge.src.test_split import (
    CompetencySplitInfo,
    SplitConfig,
    SplitResult,
    SplitStrategy,
    TestSetManager,
    TestSplitError,
)

# --- Fixtures ---


@pytest.fixture
def manager(populated_store: ForgeStorage) -> TestSetManager:
    """TestSetManager with default config and populated store."""
    return TestSetManager(populated_store)


@pytest.fixture
def multi_comp_store(populated_store: ForgeStorage) -> ForgeStorage:
    """Store with two competencies and 20 examples each."""
    populated_store.create_competency(
        Competency(
            id="comp_safety",
            name="Safety Protocols",
            description="Safety procedures",
            discipline_id="disc_test001",
            coverage_target=20,
        )
    )
    for i in range(20):
        populated_store.create_example(
            Example(
                id=f"ex_fault_{i:03d}",
                question=f"Fault isolation question {i} about components?",
                ideal_answer=f"Fault isolation answer {i} with detail.",
                competency_id="comp_test001",
                contributor_id="contrib_test001",
                discipline_id="disc_test001",
            )
        )
    for i in range(20):
        populated_store.create_example(
            Example(
                id=f"ex_safe_{i:03d}",
                question=f"Safety question {i} about procedures?",
                ideal_answer=f"Safety answer {i} with detail.",
                competency_id="comp_safety",
                contributor_id="contrib_test001",
                discipline_id="disc_test001",
            )
        )
    return populated_store


@pytest.fixture
def multi_manager(multi_comp_store: ForgeStorage) -> TestSetManager:
    """TestSetManager with multi-competency store."""
    return TestSetManager(multi_comp_store)


@pytest.fixture
def large_store(populated_store: ForgeStorage) -> ForgeStorage:
    """Store with 100 examples in a single competency."""
    for i in range(100):
        populated_store.create_example(
            Example(
                id=f"ex_large_{i:03d}",
                question=f"Large corpus question {i} about topic?",
                ideal_answer=f"Large corpus answer {i} with full detail.",
                competency_id="comp_test001",
                contributor_id="contrib_test001",
                discipline_id="disc_test001",
            )
        )
    return populated_store


# --- TestSplitConfig ---


class TestSplitConfig:
    """Tests for SplitConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default configuration values."""
        config = SplitConfig()
        assert config.test_percentage == 0.15
        assert config.min_test_per_competency == 1
        assert config.min_examples_to_split == 5
        assert config.random_seed == 42
        assert config.strategy == SplitStrategy.STRATIFIED

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = SplitConfig(
            test_percentage=0.20,
            min_test_per_competency=2,
            min_examples_to_split=10,
            random_seed=99,
            strategy=SplitStrategy.RANDOM,
        )
        assert config.test_percentage == 0.20
        assert config.min_test_per_competency == 2
        assert config.min_examples_to_split == 10
        assert config.random_seed == 99
        assert config.strategy == SplitStrategy.RANDOM

    def test_validation_rejects_invalid_percentage(self) -> None:
        """Test that invalid percentage raises on manager construction."""
        config = SplitConfig(test_percentage=0.0)
        store = ForgeStorage(":memory:")
        store.initialize_schema()
        with pytest.raises(TestSplitError, match="percentage"):
            TestSetManager(store, config=config)

    def test_validation_rejects_percentage_above_half(self) -> None:
        """Test that percentage above 0.5 raises on manager construction."""
        config = SplitConfig(test_percentage=0.6)
        store = ForgeStorage(":memory:")
        store.initialize_schema()
        with pytest.raises(TestSplitError, match="percentage"):
            TestSetManager(store, config=config)


# --- TestSplitResult ---


class TestSplitResult:
    """Tests for SplitResult dataclass."""

    def test_construction(self) -> None:
        """Test SplitResult construction."""
        info = CompetencySplitInfo(
            competency_id="comp_001",
            competency_name="Test Comp",
            total=20,
            training=17,
            test=3,
            challenge=1,
        )
        result = SplitResult(
            discipline_id="disc_001",
            training_count=17,
            test_count=3,
            challenge_count=1,
            per_competency=[info],
            config=SplitConfig(),
        )
        assert result.training_count == 17
        assert result.test_count == 3
        assert len(result.per_competency) == 1

    def test_to_dict(self) -> None:
        """Test SplitResult serialization."""
        info = CompetencySplitInfo(
            competency_id="comp_001",
            competency_name="Test Comp",
            total=10,
            training=8,
            test=2,
            challenge=0,
        )
        result = SplitResult(
            discipline_id="disc_001",
            training_count=8,
            test_count=2,
            challenge_count=0,
            per_competency=[info],
            config=SplitConfig(),
        )
        d = result.to_dict()
        assert d["discipline_id"] == "disc_001"
        assert d["training_count"] == 8
        assert d["test_count"] == 2
        assert "per_competency" in d
        assert len(d["per_competency"]) == 1
        assert d["per_competency"][0]["competency_id"] == "comp_001"
        assert "split_at" in d

    def test_competency_info_fields(self) -> None:
        """Test CompetencySplitInfo has correct fields."""
        info = CompetencySplitInfo(
            competency_id="comp_001",
            competency_name="Fault Isolation",
            total=25,
            training=21,
            test=4,
            challenge=2,
        )
        assert info.competency_id == "comp_001"
        assert info.competency_name == "Fault Isolation"
        assert info.total == 25
        assert info.training == 21
        assert info.test == 4
        assert info.challenge == 2


# --- TestStratifiedSplit ---


class TestStratifiedSplit:
    """Tests for stratified split strategy."""

    def test_correct_proportions(self, multi_manager: TestSetManager) -> None:
        """Test stratified split produces correct test/train proportions."""
        result = multi_manager.split_discipline("disc_test001")
        total = result.training_count + result.test_count
        # 20 + 20 + 1 (from populated_store) = 41 examples
        assert total == 41
        # Test set should be approximately 15% of total
        assert result.test_count >= 4
        assert result.test_count <= 10

    def test_per_competency_minimum(self, multi_manager: TestSetManager) -> None:
        """Test each competency gets at least min_test_per_competency."""
        result = multi_manager.split_discipline("disc_test001")
        for info in result.per_competency:
            if info.total >= multi_manager._config.min_examples_to_split:
                assert info.test >= multi_manager._config.min_test_per_competency

    def test_reproducible_with_seed(self, multi_comp_store: ForgeStorage) -> None:
        """Test same seed produces same split."""
        config = SplitConfig(random_seed=42)
        mgr1 = TestSetManager(multi_comp_store, config=config)
        result1 = mgr1.split_discipline("disc_test001")
        test_ids_1 = self._get_test_ids(multi_comp_store, "disc_test001")

        # Reset all to training
        self._reset_all(multi_comp_store, "disc_test001")

        mgr2 = TestSetManager(multi_comp_store, config=config)
        result2 = mgr2.split_discipline("disc_test001")
        test_ids_2 = self._get_test_ids(multi_comp_store, "disc_test001")

        assert test_ids_1 == test_ids_2
        assert result1.test_count == result2.test_count

    def test_different_seeds_differ(self, multi_comp_store: ForgeStorage) -> None:
        """Test different seeds produce different splits."""
        config1 = SplitConfig(random_seed=42)
        mgr1 = TestSetManager(multi_comp_store, config=config1)
        mgr1.split_discipline("disc_test001")
        test_ids_1 = self._get_test_ids(multi_comp_store, "disc_test001")

        self._reset_all(multi_comp_store, "disc_test001")

        config2 = SplitConfig(random_seed=99)
        mgr2 = TestSetManager(multi_comp_store, config=config2)
        mgr2.split_discipline("disc_test001")
        test_ids_2 = self._get_test_ids(multi_comp_store, "disc_test001")

        # With 41 examples, different seeds should produce different sets
        assert test_ids_1 != test_ids_2

    def test_respects_min_examples_to_split(self, populated_store: ForgeStorage) -> None:
        """Test competencies with fewer than min_examples_to_split are not split."""
        # populated_store has only 1 example in comp_test001
        config = SplitConfig(min_examples_to_split=5)
        mgr = TestSetManager(populated_store, config=config)
        result = mgr.split_discipline("disc_test001")
        # 1 example is below min_examples_to_split=5, so no split
        assert result.test_count == 0
        assert result.training_count == 1

    def test_single_competency(self, large_store: ForgeStorage) -> None:
        """Test split with a single competency having many examples."""
        config = SplitConfig(test_percentage=0.15, random_seed=42)
        mgr = TestSetManager(large_store, config=config)
        result = mgr.split_discipline("disc_test001")
        # 101 examples total (100 + 1 from populated_store)
        assert result.test_count >= 15
        assert result.test_count <= 20
        assert len(result.per_competency) == 1

    def test_multiple_competencies_balanced(self, multi_manager: TestSetManager) -> None:
        """Test split is balanced across multiple competencies."""
        result = multi_manager.split_discipline("disc_test001")
        # Both large competencies should have test examples
        large_comps = [c for c in result.per_competency if c.total >= 20]
        for comp_info in large_comps:
            assert comp_info.test >= 1

    @staticmethod
    def _get_test_ids(store: ForgeStorage, discipline_id: str) -> set[str]:
        """Helper to get IDs of test set examples."""
        test_examples = store.get_test_set_examples(discipline_id)
        return {ex.id for ex in test_examples}

    @staticmethod
    def _reset_all(store: ForgeStorage, discipline_id: str) -> None:
        """Helper to reset all examples to training set."""
        test_examples = store.get_test_set_examples(discipline_id)
        for ex in test_examples:
            ex.is_test_set = False
            ex.context = ex.context.replace("|challenge", "")
            store.update_example(ex)


# --- TestRandomSplit ---


class TestRandomSplit:
    """Tests for random split strategy."""

    def test_correct_proportions(self, multi_comp_store: ForgeStorage) -> None:
        """Test random split produces correct proportions."""
        config = SplitConfig(strategy=SplitStrategy.RANDOM)
        mgr = TestSetManager(multi_comp_store, config=config)
        result = mgr.split_discipline("disc_test001")
        total = result.training_count + result.test_count
        assert total == 41
        assert result.test_count >= 4
        assert result.test_count <= 10

    def test_reproducible(self, multi_comp_store: ForgeStorage) -> None:
        """Test random split is reproducible with same seed."""
        config = SplitConfig(strategy=SplitStrategy.RANDOM, random_seed=77)
        mgr1 = TestSetManager(multi_comp_store, config=config)
        mgr1.split_discipline("disc_test001")
        ids1 = {ex.id for ex in multi_comp_store.get_test_set_examples("disc_test001")}

        # Reset
        for ex in multi_comp_store.get_test_set_examples("disc_test001"):
            ex.is_test_set = False
            multi_comp_store.update_example(ex)

        mgr2 = TestSetManager(multi_comp_store, config=config)
        mgr2.split_discipline("disc_test001")
        ids2 = {ex.id for ex in multi_comp_store.get_test_set_examples("disc_test001")}

        assert ids1 == ids2

    def test_different_seeds(self, multi_comp_store: ForgeStorage) -> None:
        """Test random split differs with different seeds."""
        config1 = SplitConfig(strategy=SplitStrategy.RANDOM, random_seed=77)
        mgr1 = TestSetManager(multi_comp_store, config=config1)
        mgr1.split_discipline("disc_test001")
        ids1 = {ex.id for ex in multi_comp_store.get_test_set_examples("disc_test001")}

        for ex in multi_comp_store.get_test_set_examples("disc_test001"):
            ex.is_test_set = False
            multi_comp_store.update_example(ex)

        config2 = SplitConfig(strategy=SplitStrategy.RANDOM, random_seed=123)
        mgr2 = TestSetManager(multi_comp_store, config=config2)
        mgr2.split_discipline("disc_test001")
        ids2 = {ex.id for ex in multi_comp_store.get_test_set_examples("disc_test001")}

        assert ids1 != ids2

    def test_respects_minimum(self, populated_store: ForgeStorage) -> None:
        """Test random split respects min_examples_to_split."""
        config = SplitConfig(strategy=SplitStrategy.RANDOM, min_examples_to_split=5)
        mgr = TestSetManager(populated_store, config=config)
        result = mgr.split_discipline("disc_test001")
        assert result.test_count == 0


# --- TestChallengeExamples ---


class TestChallengeExamples:
    """Tests for challenge example marking."""

    def test_mark_challenge_sets_test_set(
        self, multi_manager: TestSetManager, multi_comp_store: ForgeStorage
    ) -> None:
        """Test mark_challenge sets is_test_set to True."""
        example = multi_comp_store.get_example("ex_fault_000")
        assert example is not None
        assert not example.is_test_set

        updated = multi_manager.mark_challenge("ex_fault_000")
        assert updated.is_test_set is True

    def test_mark_challenge_adds_context_marker(
        self, multi_manager: TestSetManager, multi_comp_store: ForgeStorage
    ) -> None:
        """Test mark_challenge appends challenge marker to context."""
        multi_manager.mark_challenge("ex_fault_001")
        example = multi_comp_store.get_example("ex_fault_001")
        assert example is not None
        assert "|challenge" in example.context

    def test_unmark_test_clears_flag(
        self, multi_manager: TestSetManager, multi_comp_store: ForgeStorage
    ) -> None:
        """Test unmark_test moves example back to training."""
        multi_manager.mark_challenge("ex_fault_002")
        example = multi_comp_store.get_example("ex_fault_002")
        assert example is not None
        assert example.is_test_set is True

        updated = multi_manager.unmark_test("ex_fault_002")
        assert updated.is_test_set is False
        assert "|challenge" not in updated.context

    def test_challenge_count_in_summary(self, multi_manager: TestSetManager) -> None:
        """Test challenge examples appear in split summary."""
        multi_manager.mark_challenge("ex_fault_000")
        multi_manager.mark_challenge("ex_fault_001")
        result = multi_manager.get_split_summary("disc_test001")
        assert result.challenge_count == 2


# --- TestSplitDiscipline ---


class TestSplitDiscipline:
    """Tests for full split_discipline pipeline."""

    def test_full_split_pipeline(self, multi_manager: TestSetManager) -> None:
        """Test full split creates test and training sets."""
        result = multi_manager.split_discipline("disc_test001")
        assert result.training_count > 0
        assert result.test_count > 0
        assert result.training_count + result.test_count == 41

    def test_empty_discipline(self, memory_store: ForgeStorage) -> None:
        """Test split on discipline with no examples."""
        from forge.src.models import Contributor, Discipline

        memory_store.create_contributor(Contributor(id="contrib_empty", name="Empty User"))
        memory_store.create_discipline(
            Discipline(
                id="disc_empty",
                name="Empty Discipline",
                description="No examples",
                created_by="contrib_empty",
            )
        )
        mgr = TestSetManager(memory_store)
        result = mgr.split_discipline("disc_empty")
        assert result.training_count == 0
        assert result.test_count == 0
        assert result.per_competency == []

    def test_single_example_not_split(self, populated_store: ForgeStorage) -> None:
        """Test discipline with single example is not split."""
        mgr = TestSetManager(populated_store)
        result = mgr.split_discipline("disc_test001")
        # 1 example is below default min_examples_to_split=5
        assert result.test_count == 0
        assert result.training_count == 1

    def test_large_corpus_proportions(self, large_store: ForgeStorage) -> None:
        """Test split proportions hold for large corpus."""
        config = SplitConfig(test_percentage=0.20, random_seed=42)
        mgr = TestSetManager(large_store, config=config)
        result = mgr.split_discipline("disc_test001")
        # 101 examples, ~20% = ~20 test examples
        assert result.test_count >= 18
        assert result.test_count <= 25

    def test_idempotent_resplit(self, multi_comp_store: ForgeStorage) -> None:
        """Test re-splitting resets previous split first."""
        config = SplitConfig(random_seed=42)
        mgr = TestSetManager(multi_comp_store, config=config)

        result1 = mgr.split_discipline("disc_test001")
        result2 = mgr.split_discipline("disc_test001")

        assert result1.test_count == result2.test_count
        assert result1.training_count == result2.training_count

        # Verify no double-counting: total should be unchanged
        total = result2.training_count + result2.test_count
        assert total == 41


# --- TestGetSplitSummary ---


class TestGetSplitSummary:
    """Tests for get_split_summary."""

    def test_shows_current_state(
        self, multi_manager: TestSetManager, multi_comp_store: ForgeStorage
    ) -> None:
        """Test summary reflects current split state."""
        multi_manager.split_discipline("disc_test001")
        result = multi_manager.get_split_summary("disc_test001")
        assert result.training_count > 0
        assert result.test_count > 0

    def test_counts_match(
        self, multi_manager: TestSetManager, multi_comp_store: ForgeStorage
    ) -> None:
        """Test summary counts match actual database state."""
        multi_manager.split_discipline("disc_test001")
        result = multi_manager.get_split_summary("disc_test001")

        actual_train = len(multi_comp_store.get_training_examples("disc_test001"))
        actual_test = len(multi_comp_store.get_test_set_examples("disc_test001"))

        assert result.training_count == actual_train
        assert result.test_count == actual_test

    def test_per_competency_breakdown(self, multi_manager: TestSetManager) -> None:
        """Test summary has per-competency breakdown."""
        multi_manager.split_discipline("disc_test001")
        result = multi_manager.get_split_summary("disc_test001")
        # populated_store has comp_test001, multi adds comp_safety
        assert len(result.per_competency) == 2
        for info in result.per_competency:
            assert info.total == info.training + info.test


# --- TestExportSplit ---


class TestExportSplit:
    """Tests for export_split."""

    def test_creates_two_files(self, multi_manager: TestSetManager, temp_dir: Path) -> None:
        """Test export creates training.jsonl and test.jsonl."""
        multi_manager.split_discipline("disc_test001")
        train_path, test_path = multi_manager.export_split("disc_test001", temp_dir)
        assert train_path.exists()
        assert test_path.exists()
        assert train_path.name == "training.jsonl"
        assert test_path.name == "test.jsonl"

    def test_training_file_has_no_test_examples(
        self, multi_manager: TestSetManager, temp_dir: Path
    ) -> None:
        """Test training file excludes test set examples."""
        multi_manager.split_discipline("disc_test001")
        train_path, _ = multi_manager.export_split("disc_test001", temp_dir)

        with open(train_path, encoding="utf-8") as f:
            lines = f.readlines()

        # Parse and verify none are test set
        for line in lines:
            record = json.loads(line)
            ex_id = record["metadata"]["example_id"]
            example = multi_manager._storage.get_example(ex_id)
            assert example is not None
            assert not example.is_test_set

    def test_test_file_has_only_test_examples(
        self, multi_manager: TestSetManager, temp_dir: Path
    ) -> None:
        """Test test file contains only test set examples."""
        multi_manager.split_discipline("disc_test001")
        _, test_path = multi_manager.export_split("disc_test001", temp_dir)

        with open(test_path, encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) > 0
        for line in lines:
            record = json.loads(line)
            ex_id = record["metadata"]["example_id"]
            example = multi_manager._storage.get_example(ex_id)
            assert example is not None
            assert example.is_test_set

    def test_alpaca_format_preserved(self, multi_manager: TestSetManager, temp_dir: Path) -> None:
        """Test exported records are in Alpaca format."""
        multi_manager.split_discipline("disc_test001")
        train_path, test_path = multi_manager.export_split("disc_test001", temp_dir)

        for filepath in [train_path, test_path]:
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    assert "instruction" in record
                    assert "input" in record
                    assert "output" in record
                    assert "metadata" in record


# --- TestEdgeCases ---


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_examples(self, memory_store: ForgeStorage) -> None:
        """Test split with zero examples produces empty result."""
        from forge.src.models import Contributor, Discipline

        memory_store.create_contributor(Contributor(id="contrib_zero", name="Zero User"))
        memory_store.create_discipline(
            Discipline(
                id="disc_zero",
                name="Zero Discipline",
                description="No examples",
                created_by="contrib_zero",
            )
        )
        mgr = TestSetManager(memory_store)
        result = mgr.split_discipline("disc_zero")
        assert result.training_count == 0
        assert result.test_count == 0

    def test_one_competency_many_examples(self, large_store: ForgeStorage) -> None:
        """Test split with one competency having many examples."""
        config = SplitConfig(test_percentage=0.15)
        mgr = TestSetManager(large_store, config=config)
        result = mgr.split_discipline("disc_test001")
        assert len(result.per_competency) == 1
        info = result.per_competency[0]
        assert info.total == 101  # 100 + 1 from populated_store
        assert info.test >= 15

    def test_all_examples_already_in_test_set(self, multi_comp_store: ForgeStorage) -> None:
        """Test re-split after all examples already marked as test."""
        # Mark all examples as test set manually
        all_examples = multi_comp_store.get_training_examples("disc_test001")
        all_test = multi_comp_store.get_test_set_examples("disc_test001")
        for ex in all_examples + all_test:
            ex.is_test_set = True
            multi_comp_store.update_example(ex)

        # Re-split should reset and produce a correct split
        config = SplitConfig(random_seed=42)
        mgr = TestSetManager(multi_comp_store, config=config)
        result = mgr.split_discipline("disc_test001")
        assert result.training_count > 0
        assert result.test_count > 0
        assert result.training_count + result.test_count == 41

    def test_mark_nonexistent_example_raises(self, manager: TestSetManager) -> None:
        """Test marking nonexistent example raises error."""
        with pytest.raises(TestSplitError, match="not found"):
            manager.mark_challenge("ex_nonexistent")

    def test_unmark_nonexistent_example_raises(self, manager: TestSetManager) -> None:
        """Test unmarking nonexistent example raises error."""
        with pytest.raises(TestSplitError, match="not found"):
            manager.unmark_test("ex_nonexistent")

    def test_challenge_first_strategy(self, multi_comp_store: ForgeStorage) -> None:
        """Test CHALLENGE_FIRST strategy keeps challenge examples in test."""
        config = SplitConfig(strategy=SplitStrategy.CHALLENGE_FIRST, random_seed=42)
        mgr = TestSetManager(multi_comp_store, config=config)

        # Mark two as challenge
        mgr.mark_challenge("ex_fault_000")
        mgr.mark_challenge("ex_safe_000")

        result = mgr.split_discipline("disc_test001")

        # Verify challenge examples are still in test set
        ex1 = multi_comp_store.get_example("ex_fault_000")
        ex2 = multi_comp_store.get_example("ex_safe_000")
        assert ex1 is not None and ex1.is_test_set
        assert ex2 is not None and ex2.is_test_set
        assert result.challenge_count >= 2
