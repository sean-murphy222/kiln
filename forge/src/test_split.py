"""Held-out test set reservation for Forge curriculum evaluation.

Provides configurable train/test splitting of examples with
stratified sampling per competency, challenge example marking,
and separate JSONL export for training and evaluation.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from forge.src.models import Example
from forge.src.storage import ForgeStorage

CHALLENGE_MARKER = "|challenge"


class TestSplitError(Exception):
    """Raised for test split workflow errors."""


class SplitStrategy(str, Enum):
    """Strategy for selecting test set examples.

    Attributes:
        RANDOM: Uniform random selection across all examples.
        STRATIFIED: Proportional selection per competency.
        CHALLENGE_FIRST: Challenge-marked examples first, then stratified.
    """

    RANDOM = "random"
    STRATIFIED = "stratified"
    CHALLENGE_FIRST = "challenge_first"


@dataclass
class SplitConfig:
    """Configuration for test set splitting.

    Attributes:
        test_percentage: Fraction of examples to hold out (0.0-0.5).
        min_test_per_competency: Minimum test examples per competency.
        min_examples_to_split: Minimum examples needed to split a competency.
        random_seed: Seed for reproducible splits.
        strategy: Split strategy to use.
    """

    test_percentage: float = 0.15
    min_test_per_competency: int = 1
    min_examples_to_split: int = 5
    random_seed: int = 42
    strategy: SplitStrategy = SplitStrategy.STRATIFIED


@dataclass
class CompetencySplitInfo:
    """Per-competency breakdown of a split result.

    Attributes:
        competency_id: The competency ID.
        competency_name: Human-readable competency name.
        total: Total examples in this competency.
        training: Number assigned to training set.
        test: Number assigned to test set.
        challenge: Number of challenge-marked test examples.
    """

    competency_id: str
    competency_name: str
    total: int
    training: int
    test: int
    challenge: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "competency_id": self.competency_id,
            "competency_name": self.competency_name,
            "total": self.total,
            "training": self.training,
            "test": self.test,
            "challenge": self.challenge,
        }


@dataclass
class SplitResult:
    """Result of a test set split operation.

    Attributes:
        discipline_id: The discipline that was split.
        training_count: Total training examples.
        test_count: Total test set examples.
        challenge_count: Number of challenge-marked examples.
        per_competency: Breakdown by competency.
        config: Configuration used for the split.
        split_at: When the split was performed.
    """

    discipline_id: str
    training_count: int
    test_count: int
    challenge_count: int
    per_competency: list[CompetencySplitInfo]
    config: SplitConfig
    split_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "discipline_id": self.discipline_id,
            "training_count": self.training_count,
            "test_count": self.test_count,
            "challenge_count": self.challenge_count,
            "per_competency": [c.to_dict() for c in self.per_competency],
            "split_at": self.split_at.isoformat(),
        }


class TestSetManager:
    """Manages held-out test set reservation for a discipline.

    Provides stratified or random splitting, challenge example
    marking, and separate JSONL export of training and test sets.

    Args:
        storage: ForgeStorage instance for persistence.
        config: Split configuration. Defaults to SplitConfig().

    Raises:
        TestSplitError: If config has invalid test_percentage.

    Example::

        mgr = TestSetManager(storage, config=SplitConfig(test_percentage=0.20))
        result = mgr.split_discipline("disc_001")
        train_path, test_path = mgr.export_split("disc_001", Path("./out"))
    """

    def __init__(
        self,
        storage: ForgeStorage,
        config: SplitConfig | None = None,
    ) -> None:
        self._storage = storage
        self._config = config or SplitConfig()
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the split configuration.

        Raises:
            TestSplitError: If test_percentage is out of range.
        """
        pct = self._config.test_percentage
        if pct <= 0.0 or pct > 0.5:
            raise TestSplitError(
                "test_percentage must be between 0.0 (exclusive) and 0.5 (inclusive)"
            )

    def split_discipline(self, discipline_id: str) -> SplitResult:
        """Perform train/test split for all examples in a discipline.

        Resets any existing split first, then applies the configured
        strategy. Persists changes via storage.update_example().

        Args:
            discipline_id: Discipline to split.

        Returns:
            SplitResult with counts and per-competency breakdown.
        """
        self._reset_split(discipline_id)

        if self._config.strategy == SplitStrategy.RANDOM:
            return self._do_random_split(discipline_id)
        elif self._config.strategy == SplitStrategy.CHALLENGE_FIRST:
            return self._do_challenge_first_split(discipline_id)
        return self._do_stratified_split(discipline_id)

    def mark_challenge(self, example_id: str) -> Example:
        """Mark an example as a challenge test example.

        Sets is_test_set=True and appends the challenge marker
        to the example's context field.

        Args:
            example_id: ID of the example to mark.

        Returns:
            Updated Example.

        Raises:
            TestSplitError: If example not found.
        """
        example = self._storage.get_example(example_id)
        if example is None:
            raise TestSplitError(f"Example not found: {example_id}")

        example.is_test_set = True
        if CHALLENGE_MARKER not in example.context:
            example.context = example.context + CHALLENGE_MARKER
        return self._storage.update_example(example)

    def unmark_test(self, example_id: str) -> Example:
        """Move an example back to the training set.

        Clears is_test_set and removes the challenge marker
        from context if present.

        Args:
            example_id: ID of the example to unmark.

        Returns:
            Updated Example.

        Raises:
            TestSplitError: If example not found.
        """
        example = self._storage.get_example(example_id)
        if example is None:
            raise TestSplitError(f"Example not found: {example_id}")

        example.is_test_set = False
        example.context = example.context.replace(CHALLENGE_MARKER, "")
        return self._storage.update_example(example)

    def get_split_summary(self, discipline_id: str) -> SplitResult:
        """Report current split state without modifying anything.

        Args:
            discipline_id: Discipline to summarize.

        Returns:
            SplitResult reflecting current database state.
        """
        return self._build_result(discipline_id)

    def export_split(
        self,
        discipline_id: str,
        output_dir: str | Path,
    ) -> tuple[Path, Path]:
        """Export training and test sets to separate JSONL files.

        Creates training.jsonl and test.jsonl in Alpaca format
        within the specified output directory.

        Args:
            discipline_id: Discipline to export.
            output_dir: Directory for output files.

        Returns:
            Tuple of (training_path, test_path).
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        train_path = self._export_training(discipline_id, out)
        test_path = self._export_test(discipline_id, out)
        return train_path, test_path

    # ---------------------------------------------------------------
    # Internal split implementations
    # ---------------------------------------------------------------

    def _do_stratified_split(self, discipline_id: str) -> SplitResult:
        """Perform stratified split across competencies.

        Args:
            discipline_id: Discipline to split.

        Returns:
            SplitResult with per-competency proportional allocation.
        """
        competencies = self._storage.get_competencies_for_discipline(discipline_id)
        rng = random.Random(self._config.random_seed)

        for comp in competencies:
            examples = self._storage.get_examples_for_competency(comp.id)
            self._split_competency_examples(examples, rng)

        return self._build_result(discipline_id)

    def _do_random_split(self, discipline_id: str) -> SplitResult:
        """Perform random split across all examples.

        Args:
            discipline_id: Discipline to split.

        Returns:
            SplitResult from uniform random selection.
        """
        training = self._storage.get_training_examples(discipline_id)
        if len(training) < self._config.min_examples_to_split:
            return self._build_result(discipline_id)

        rng = random.Random(self._config.random_seed)
        test_count = max(1, math.ceil(len(training) * self._config.test_percentage))

        test_ids = self._select_random_ids(training, test_count, rng)
        self._apply_test_ids(test_ids)
        return self._build_result(discipline_id)

    def _do_challenge_first_split(self, discipline_id: str) -> SplitResult:
        """Perform split preserving existing challenge markers.

        Challenge-marked examples stay in test set. Remaining budget
        is filled using stratified sampling.

        Args:
            discipline_id: Discipline to split.

        Returns:
            SplitResult preserving challenge examples.
        """
        competencies = self._storage.get_competencies_for_discipline(discipline_id)
        rng = random.Random(self._config.random_seed)

        # Restore challenge examples that were reset
        self._restore_challenges(discipline_id)

        for comp in competencies:
            examples = self._storage.get_examples_for_competency(comp.id)
            non_challenge = [ex for ex in examples if CHALLENGE_MARKER not in ex.context]
            challenge_count = len(examples) - len(non_challenge)
            self._split_competency_examples(non_challenge, rng, pre_selected=challenge_count)

        return self._build_result(discipline_id)

    # ---------------------------------------------------------------
    # Split helpers
    # ---------------------------------------------------------------

    def _split_competency_examples(
        self,
        examples: list[Example],
        rng: random.Random,
        pre_selected: int = 0,
    ) -> None:
        """Split a single competency's examples into train/test.

        Args:
            examples: Examples to consider for splitting.
            rng: Random number generator for selection.
            pre_selected: Number already selected (e.g., challenges).
        """
        total = len(examples) + pre_selected
        if total < self._config.min_examples_to_split:
            return

        target = math.ceil(total * self._config.test_percentage)
        target = max(target, self._config.min_test_per_competency)
        remaining = max(0, target - pre_selected)

        if remaining <= 0 or len(examples) == 0:
            return

        remaining = min(remaining, len(examples))
        selected = rng.sample(examples, remaining)
        for ex in selected:
            ex.is_test_set = True
            self._storage.update_example(ex)

    def _select_random_ids(
        self,
        examples: list[Example],
        count: int,
        rng: random.Random,
    ) -> set[str]:
        """Select random example IDs for the test set.

        Args:
            examples: Pool of examples to select from.
            count: Number to select.
            rng: Random number generator.

        Returns:
            Set of selected example IDs.
        """
        count = min(count, len(examples))
        selected = rng.sample(examples, count)
        return {ex.id for ex in selected}

    def _apply_test_ids(self, test_ids: set[str]) -> None:
        """Mark examples with given IDs as test set.

        Args:
            test_ids: IDs of examples to mark as test.
        """
        for ex_id in test_ids:
            example = self._storage.get_example(ex_id)
            if example is not None:
                example.is_test_set = True
                self._storage.update_example(example)

    def _reset_split(self, discipline_id: str) -> None:
        """Reset all test set flags for a discipline.

        Preserves challenge markers in context but resets is_test_set
        to False for all non-challenge examples. Challenge examples
        remain marked.

        Args:
            discipline_id: Discipline to reset.
        """
        test_examples = self._storage.get_test_set_examples(discipline_id)
        for ex in test_examples:
            if CHALLENGE_MARKER in ex.context:
                continue
            ex.is_test_set = False
            self._storage.update_example(ex)

    def _restore_challenges(self, discipline_id: str) -> None:
        """Restore challenge-marked examples to test set.

        After reset, challenge examples need their is_test_set
        flag restored based on the context marker.

        Args:
            discipline_id: Discipline to scan.
        """
        training = self._storage.get_training_examples(discipline_id)
        for ex in training:
            if CHALLENGE_MARKER in ex.context:
                ex.is_test_set = True
                self._storage.update_example(ex)

    # ---------------------------------------------------------------
    # Result building
    # ---------------------------------------------------------------

    def _build_result(self, discipline_id: str) -> SplitResult:
        """Build a SplitResult from current database state.

        Args:
            discipline_id: Discipline to summarize.

        Returns:
            SplitResult with computed counts.
        """
        competencies = self._storage.get_competencies_for_discipline(discipline_id)
        per_competency = [self._build_competency_info(comp.id, comp.name) for comp in competencies]

        training_total = sum(c.training for c in per_competency)
        test_total = sum(c.test for c in per_competency)
        challenge_total = sum(c.challenge for c in per_competency)

        return SplitResult(
            discipline_id=discipline_id,
            training_count=training_total,
            test_count=test_total,
            challenge_count=challenge_total,
            per_competency=per_competency,
            config=self._config,
        )

    def _build_competency_info(
        self,
        competency_id: str,
        competency_name: str,
    ) -> CompetencySplitInfo:
        """Build split info for a single competency.

        Args:
            competency_id: Competency to report on.
            competency_name: Human-readable name.

        Returns:
            CompetencySplitInfo with counts.
        """
        all_examples = self._storage.get_examples_for_competency(competency_id)
        test_examples = [ex for ex in all_examples if ex.is_test_set]
        training_examples = [ex for ex in all_examples if not ex.is_test_set]
        challenge_examples = [ex for ex in test_examples if CHALLENGE_MARKER in ex.context]

        return CompetencySplitInfo(
            competency_id=competency_id,
            competency_name=competency_name,
            total=len(all_examples),
            training=len(training_examples),
            test=len(test_examples),
            challenge=len(challenge_examples),
        )

    # ---------------------------------------------------------------
    # Export helpers
    # ---------------------------------------------------------------

    def _export_training(
        self,
        discipline_id: str,
        output_dir: Path,
    ) -> Path:
        """Export training examples to JSONL.

        Args:
            discipline_id: Discipline to export.
            output_dir: Output directory.

        Returns:
            Path to training.jsonl.
        """
        path = output_dir / "training.jsonl"
        examples = self._storage.get_training_examples(discipline_id)
        self._write_jsonl(examples, path)
        return path

    def _export_test(
        self,
        discipline_id: str,
        output_dir: Path,
    ) -> Path:
        """Export test set examples to JSONL.

        Args:
            discipline_id: Discipline to export.
            output_dir: Output directory.

        Returns:
            Path to test.jsonl.
        """
        path = output_dir / "test.jsonl"
        examples = self._storage.get_test_set_examples(discipline_id)
        self._write_jsonl(examples, path)
        return path

    @staticmethod
    def _write_jsonl(examples: list[Example], path: Path) -> None:
        """Write examples to JSONL in Alpaca format.

        Args:
            examples: Examples to write.
            path: Output file path.
        """
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_training_record()) + "\n")
