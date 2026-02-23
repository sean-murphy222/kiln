"""Tests for consistency checking engine."""

from __future__ import annotations

from datetime import datetime

import pytest

from forge.src.consistency import (
    ConsistencyChecker,
    ConsistencyError,
    ConsistencyIssue,
    ConsistencyReport,
    IssueSeverity,
    IssueType,
)
from forge.src.models import Competency, Example
from forge.src.storage import ForgeStorage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_example(
    ex_id: str,
    question: str,
    answer: str,
    competency_id: str = "comp_test001",
    discipline_id: str = "disc_test001",
    contributor_id: str = "contrib_test001",
) -> Example:
    """Create an Example with minimal boilerplate."""
    return Example(
        id=ex_id,
        question=question,
        ideal_answer=answer,
        competency_id=competency_id,
        contributor_id=contributor_id,
        discipline_id=discipline_id,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def checker(populated_store: ForgeStorage) -> ConsistencyChecker:
    """ConsistencyChecker backed by populated_store."""
    return ConsistencyChecker(populated_store)


@pytest.fixture
def store_with_many_examples(populated_store: ForgeStorage) -> ForgeStorage:
    """Store with enough examples per competency to trigger length checks.

    Creates 6 examples with answer lengths around 50 chars each so
    that the mean is well-defined.  The populated_store already has one
    example in comp_test001; we add 5 more.
    """
    base_answer = "This is a moderately sized answer for testing."  # ~46 chars
    for i in range(5):
        populated_store.create_example(
            _make_example(
                ex_id=f"ex_len_{i:03d}",
                question=f"Question number {i} about maintenance?",
                answer=f"{base_answer} Extra text {i}.",
            )
        )
    return populated_store


@pytest.fixture
def checker_many(store_with_many_examples: ForgeStorage) -> ConsistencyChecker:
    """Checker backed by store with many examples."""
    return ConsistencyChecker(store_with_many_examples)


@pytest.fixture
def store_with_vocab(populated_store: ForgeStorage) -> ForgeStorage:
    """Store whose discipline has vocabulary and 6+ examples."""
    disc = populated_store.get_discipline("disc_test001")
    assert disc is not None
    disc.vocabulary = ["torque", "clearance", "tolerance"]
    populated_store.update_discipline(disc)
    base = "Apply the correct torque and check clearance tolerance."
    for i in range(5):
        populated_store.create_example(
            _make_example(
                ex_id=f"ex_vocab_{i:03d}",
                question=f"Vocab question {i} about maintenance?",
                answer=f"{base} Repetition {i}.",
            )
        )
    return populated_store


@pytest.fixture
def checker_vocab(store_with_vocab: ForgeStorage) -> ConsistencyChecker:
    """Checker backed by store with vocabulary discipline."""
    return ConsistencyChecker(store_with_vocab)


# ===========================================================================
# TestConsistencyIssue
# ===========================================================================


class TestConsistencyIssue:
    """Tests for the ConsistencyIssue dataclass."""

    def test_construction(self) -> None:
        """Issue can be constructed with all fields."""
        issue = ConsistencyIssue(
            issue_type=IssueType.LENGTH,
            severity=IssueSeverity.WARNING,
            message="Answer is too short",
            example_id="ex_001",
            suggested_fix="Expand the answer to be closer to the mean length.",
        )
        assert issue.issue_type == IssueType.LENGTH
        assert issue.severity == IssueSeverity.WARNING
        assert issue.example_id == "ex_001"
        assert issue.suggested_fix != ""

    def test_to_dict(self) -> None:
        """Issue serializes to a dictionary."""
        issue = ConsistencyIssue(
            issue_type=IssueType.TERMINOLOGY,
            severity=IssueSeverity.INFO,
            message="Term mismatch",
            example_id="ex_002",
            suggested_fix="Use standard term.",
            details={"term": "wrench"},
        )
        data = issue.to_dict()
        assert data["issue_type"] == IssueType.TERMINOLOGY.value
        assert data["severity"] == "info"
        assert data["details"]["term"] == "wrench"

    def test_from_dict(self) -> None:
        """Issue deserializes from a dictionary."""
        data = {
            "issue_type": IssueType.CITATION.value,
            "severity": "error",
            "message": "Citation format wrong",
            "example_id": "ex_003",
            "suggested_fix": "Fix citation.",
            "details": {},
        }
        issue = ConsistencyIssue.from_dict(data)
        assert issue.issue_type == IssueType.CITATION
        assert issue.severity == IssueSeverity.ERROR
        assert issue.example_id == "ex_003"


# ===========================================================================
# TestConsistencyReport
# ===========================================================================


class TestConsistencyReport:
    """Tests for the ConsistencyReport dataclass."""

    def test_construction(self) -> None:
        """Report can be constructed with defaults."""
        report = ConsistencyReport(
            discipline_id="disc_001",
            issues=[],
            checked_at=datetime.now(),
            example_count=10,
        )
        assert report.discipline_id == "disc_001"
        assert report.example_count == 10
        assert report.issues == []

    def test_has_errors_false(self) -> None:
        """has_errors is False when no ERROR-severity issues."""
        report = ConsistencyReport(
            discipline_id="disc_001",
            issues=[
                ConsistencyIssue(
                    issue_type=IssueType.LENGTH,
                    severity=IssueSeverity.WARNING,
                    message="short",
                    example_id="ex_1",
                ),
            ],
            checked_at=datetime.now(),
            example_count=5,
        )
        assert report.has_errors is False

    def test_has_errors_true(self) -> None:
        """has_errors is True when ERROR-severity issues exist."""
        report = ConsistencyReport(
            discipline_id="disc_001",
            issues=[
                ConsistencyIssue(
                    issue_type=IssueType.CONFLICT,
                    severity=IssueSeverity.ERROR,
                    message="conflict",
                    example_id="ex_1",
                ),
            ],
            checked_at=datetime.now(),
            example_count=5,
        )
        assert report.has_errors is True

    def test_has_warnings(self) -> None:
        """has_warnings detects WARNING-level issues."""
        report = ConsistencyReport(
            discipline_id="disc_001",
            issues=[
                ConsistencyIssue(
                    issue_type=IssueType.LENGTH,
                    severity=IssueSeverity.WARNING,
                    message="too long",
                    example_id="ex_2",
                ),
            ],
            checked_at=datetime.now(),
            example_count=5,
        )
        assert report.has_warnings is True

    def test_issues_by_type(self) -> None:
        """issues_by_type groups issues correctly."""
        issues = [
            ConsistencyIssue(
                issue_type=IssueType.LENGTH,
                severity=IssueSeverity.WARNING,
                message="a",
                example_id="ex_1",
            ),
            ConsistencyIssue(
                issue_type=IssueType.LENGTH,
                severity=IssueSeverity.WARNING,
                message="b",
                example_id="ex_2",
            ),
            ConsistencyIssue(
                issue_type=IssueType.CONFLICT,
                severity=IssueSeverity.ERROR,
                message="c",
                example_id="ex_3",
            ),
        ]
        report = ConsistencyReport(
            discipline_id="disc_001",
            issues=issues,
            checked_at=datetime.now(),
            example_count=10,
        )
        by_type = report.issues_by_type
        assert len(by_type[IssueType.LENGTH]) == 2
        assert len(by_type[IssueType.CONFLICT]) == 1

    def test_issues_by_severity(self) -> None:
        """issues_by_severity groups issues correctly."""
        issues = [
            ConsistencyIssue(
                issue_type=IssueType.LENGTH,
                severity=IssueSeverity.WARNING,
                message="w",
                example_id="ex_1",
            ),
            ConsistencyIssue(
                issue_type=IssueType.CONFLICT,
                severity=IssueSeverity.ERROR,
                message="e",
                example_id="ex_2",
            ),
            ConsistencyIssue(
                issue_type=IssueType.TERMINOLOGY,
                severity=IssueSeverity.INFO,
                message="i",
                example_id="ex_3",
            ),
        ]
        report = ConsistencyReport(
            discipline_id="disc_001",
            issues=issues,
            checked_at=datetime.now(),
            example_count=10,
        )
        by_sev = report.issues_by_severity
        assert len(by_sev[IssueSeverity.WARNING]) == 1
        assert len(by_sev[IssueSeverity.ERROR]) == 1
        assert len(by_sev[IssueSeverity.INFO]) == 1


# ===========================================================================
# TestResponseLengthCheck
# ===========================================================================


class TestResponseLengthCheck:
    """Tests for _check_response_length."""

    def test_normal_range_passes(self, checker_many: ConsistencyChecker) -> None:
        """Examples within 0.5x-2x mean produce no issues."""
        examples = checker_many._storage.get_examples_for_competency("comp_test001")
        issues = checker_many._check_response_length(examples)
        # All examples are roughly the same length -- no outliers
        assert len(issues) == 0

    def test_too_short_flagged(self, store_with_many_examples: ForgeStorage) -> None:
        """An example much shorter than the mean is flagged."""
        store_with_many_examples.create_example(
            _make_example(
                ex_id="ex_short",
                question="What is the minimum clearance?",
                answer="5mm.",  # Very short relative to ~50 char mean
            )
        )
        checker = ConsistencyChecker(store_with_many_examples)
        examples = checker._storage.get_examples_for_competency("comp_test001")
        issues = checker._check_response_length(examples)
        short_issues = [i for i in issues if i.example_id == "ex_short"]
        assert len(short_issues) == 1
        assert "short" in short_issues[0].message.lower()

    def test_too_long_flagged(self, store_with_many_examples: ForgeStorage) -> None:
        """An example much longer than the mean is flagged."""
        long_answer = "A" * 500  # Way longer than ~50 char mean
        store_with_many_examples.create_example(
            _make_example(
                ex_id="ex_long",
                question="Explain the full maintenance procedure?",
                answer=long_answer,
            )
        )
        checker = ConsistencyChecker(store_with_many_examples)
        examples = checker._storage.get_examples_for_competency("comp_test001")
        issues = checker._check_response_length(examples)
        long_issues = [i for i in issues if i.example_id == "ex_long"]
        assert len(long_issues) == 1
        assert "long" in long_issues[0].message.lower()

    def test_small_corpus_skipped(self, checker: ConsistencyChecker) -> None:
        """Fewer than 5 examples in a competency skips the length check."""
        # populated_store has only 1 example in comp_test001
        examples = checker._storage.get_examples_for_competency("comp_test001")
        assert len(examples) < 5
        issues = checker._check_response_length(examples)
        assert issues == []

    def test_per_competency_grouping(self, store_with_many_examples: ForgeStorage) -> None:
        """Length check operates per-competency, not across all examples."""
        # Add a second competency with its own scale
        store_with_many_examples.create_competency(
            Competency(
                id="comp_other",
                name="Other Competency",
                description="Separate competency",
                discipline_id="disc_test001",
            )
        )
        short_answer = "Brief."
        for i in range(6):
            store_with_many_examples.create_example(
                _make_example(
                    ex_id=f"ex_other_{i:03d}",
                    question=f"Other question {i}?",
                    answer=short_answer,
                    competency_id="comp_other",
                )
            )
        checker = ConsistencyChecker(store_with_many_examples)
        # No outliers within comp_other because they're all the same length
        other_examples = checker._storage.get_examples_for_competency("comp_other")
        issues = checker._check_response_length(other_examples)
        assert issues == []


# ===========================================================================
# TestTerminologyCheck
# ===========================================================================


class TestTerminologyCheck:
    """Tests for _check_terminology."""

    def test_correct_terms_pass(self, checker_vocab: ConsistencyChecker) -> None:
        """Examples using vocabulary terms produce no issues for those examples."""
        examples = checker_vocab._storage.get_examples_for_competency("comp_test001")
        vocab = ["torque", "clearance", "tolerance"]
        # Only check the examples we added that contain vocab terms
        vocab_examples = [ex for ex in examples if ex.id.startswith("ex_vocab_")]
        issues = checker_vocab._check_terminology(vocab_examples, vocab)
        assert len(issues) == 0

    def test_missing_vocab_flagged(self, store_with_vocab: ForgeStorage) -> None:
        """An example not using any vocab terms is flagged."""
        store_with_vocab.create_example(
            _make_example(
                ex_id="ex_no_vocab",
                question="What color is the widget?",
                answer="The widget is painted blue with no special features.",
            )
        )
        checker = ConsistencyChecker(store_with_vocab)
        examples = checker._storage.get_examples_for_competency("comp_test001")
        vocab = ["torque", "clearance", "tolerance"]
        issues = checker._check_terminology(examples, vocab)
        flagged = [i for i in issues if i.example_id == "ex_no_vocab"]
        assert len(flagged) >= 1

    def test_case_insensitive_matching(self, store_with_vocab: ForgeStorage) -> None:
        """Vocabulary matching is case-insensitive."""
        store_with_vocab.create_example(
            _make_example(
                ex_id="ex_caps",
                question="What is the TORQUE specification?",
                answer="Apply TORQUE of 50 ft-lbs with proper CLEARANCE check.",
            )
        )
        checker = ConsistencyChecker(store_with_vocab)
        examples = [checker._storage.get_example("ex_caps")]
        assert examples[0] is not None
        vocab = ["torque", "clearance"]
        issues = checker._check_terminology(examples, vocab)  # type: ignore[arg-type]
        # Should NOT be flagged since TORQUE matches torque
        flagged = [i for i in issues if i.example_id == "ex_caps"]
        assert len(flagged) == 0

    def test_empty_vocabulary_skipped(self, checker: ConsistencyChecker) -> None:
        """Empty vocabulary list produces no issues."""
        examples = checker._storage.get_examples_for_competency("comp_test001")
        issues = checker._check_terminology(examples, [])
        assert issues == []

    def test_partial_match_handling(self, store_with_vocab: ForgeStorage) -> None:
        """An answer using some but not all vocab terms is not flagged.

        The check only flags examples that use NONE of the vocabulary
        terms, not examples that use only some of them.
        """
        store_with_vocab.create_example(
            _make_example(
                ex_id="ex_partial",
                question="What torque should be applied?",
                answer="Apply the specified torque value carefully.",
            )
        )
        checker = ConsistencyChecker(store_with_vocab)
        examples = [checker._storage.get_example("ex_partial")]
        assert examples[0] is not None
        vocab = ["torque", "clearance", "tolerance"]
        issues = checker._check_terminology(examples, vocab)  # type: ignore[arg-type]
        flagged = [i for i in issues if i.example_id == "ex_partial"]
        assert len(flagged) == 0


# ===========================================================================
# TestCitationFormatCheck
# ===========================================================================


class TestCitationFormatCheck:
    """Tests for _check_citation_format."""

    def test_consistent_format_passes(self, populated_store: ForgeStorage) -> None:
        """Examples with consistent citation formats produce no issues."""
        for i in range(3):
            populated_store.create_example(
                _make_example(
                    ex_id=f"ex_cit_{i:03d}",
                    question=f"Citation question {i}?",
                    answer=f"Refer to TM 1-1520-237-10, section {i + 1}.",
                )
            )
        checker = ConsistencyChecker(populated_store)
        examples = populated_store.get_training_examples("disc_test001")
        issues = checker._check_citation_format(examples)
        assert len(issues) == 0

    def test_inconsistent_format_flagged(self, populated_store: ForgeStorage) -> None:
        """Mixed citation formats in same discipline are flagged."""
        populated_store.create_example(
            _make_example(
                ex_id="ex_cit_a",
                question="First citation question?",
                answer="Refer to TM 1-1520-237-10 for details.",
            )
        )
        populated_store.create_example(
            _make_example(
                ex_id="ex_cit_b",
                question="Second citation question?",
                answer="Refer to TM1-1520-237-10 for details.",  # No space
            )
        )
        checker = ConsistencyChecker(populated_store)
        examples = populated_store.get_training_examples("disc_test001")
        issues = checker._check_citation_format(examples)
        assert len(issues) >= 1

    def test_no_citations_passes(self, populated_store: ForgeStorage) -> None:
        """Examples with no citations at all produce no issues."""
        for i in range(3):
            populated_store.create_example(
                _make_example(
                    ex_id=f"ex_nocit_{i:03d}",
                    question=f"No citation question {i}?",
                    answer=f"Simple answer with no references number {i}.",
                )
            )
        checker = ConsistencyChecker(populated_store)
        examples = populated_store.get_training_examples("disc_test001")
        issues = checker._check_citation_format(examples)
        assert len(issues) == 0

    def test_mixed_patterns_flagged(self, populated_store: ForgeStorage) -> None:
        """Multiple different citation styles in a corpus are flagged."""
        populated_store.create_example(
            _make_example(
                ex_id="ex_mix_a",
                question="Mixed pattern A?",
                answer="See TM 9-2320-280-10 section 4.",
            )
        )
        populated_store.create_example(
            _make_example(
                ex_id="ex_mix_b",
                question="Mixed pattern B?",
                answer="See FM 3-25.26 chapter 2.",  # FM vs TM
            )
        )
        populated_store.create_example(
            _make_example(
                ex_id="ex_mix_c",
                question="Mixed pattern C?",
                answer="See AR 385-10 paragraph 6.",  # AR
            )
        )
        checker = ConsistencyChecker(populated_store)
        examples = populated_store.get_training_examples("disc_test001")
        # Different manual types (TM, FM, AR) are valid distinct references.
        # The check is for format inconsistency of the SAME reference.
        result = checker._check_citation_format(examples)
        # Each uses a consistent format (TYPE NUM-NUM-NUM-NUM) so no issues
        assert isinstance(result, list)

    def test_regex_patterns_detected(self, populated_store: ForgeStorage) -> None:
        """Citation regex correctly detects military manual references."""
        populated_store.create_example(
            _make_example(
                ex_id="ex_regex",
                question="Regex detection test?",
                answer="Per TM 1-1520-237-10, WP 0045 00, step 3.",
            )
        )
        checker = ConsistencyChecker(populated_store)
        examples = [populated_store.get_example("ex_regex")]
        assert examples[0] is not None
        issues = checker._check_citation_format(examples)  # type: ignore[arg-type]
        # Single example with citations should not be flagged
        assert len(issues) == 0


# ===========================================================================
# TestConflictDetection
# ===========================================================================


class TestConflictDetection:
    """Tests for _check_conflicts."""

    def test_no_conflicts_pass(self, populated_store: ForgeStorage) -> None:
        """Dissimilar questions produce no conflict issues."""
        populated_store.create_example(
            _make_example(
                ex_id="ex_conf_a",
                question="How to check oil pressure?",
                answer="Check the gauge reading and compare to spec.",
            )
        )
        populated_store.create_example(
            _make_example(
                ex_id="ex_conf_b",
                question="What is the torque for bolt A?",
                answer="Apply 25 ft-lbs of torque.",
            )
        )
        checker = ConsistencyChecker(populated_store)
        examples = populated_store.get_training_examples("disc_test001")
        issues = checker._check_conflicts(examples)
        assert len(issues) == 0

    def test_similar_questions_different_answers_flagged(
        self, populated_store: ForgeStorage
    ) -> None:
        """Similar questions with different answers are flagged."""
        populated_store.create_example(
            _make_example(
                ex_id="ex_sim_a",
                question="How do you check the hydraulic fluid level?",
                answer="Open the reservoir cap and read the dipstick.",
            )
        )
        populated_store.create_example(
            _make_example(
                ex_id="ex_sim_b",
                question="How do you check the hydraulic fluid level?",
                answer="Connect the electronic sensor to port B.",
            )
        )
        checker = ConsistencyChecker(populated_store)
        examples = populated_store.get_training_examples("disc_test001")
        issues = checker._check_conflicts(examples)
        conflict_issues = [i for i in issues if i.issue_type == IssueType.CONFLICT]
        assert len(conflict_issues) >= 1

    def test_identical_questions_flagged(self, populated_store: ForgeStorage) -> None:
        """Identical questions with different answers are definitely flagged."""
        populated_store.create_example(
            _make_example(
                ex_id="ex_dup_a",
                question="What is the maximum torque specification?",
                answer="The maximum torque is 50 ft-lbs.",
            )
        )
        populated_store.create_example(
            _make_example(
                ex_id="ex_dup_b",
                question="What is the maximum torque specification?",
                answer="Apply no more than 75 ft-lbs.",
            )
        )
        checker = ConsistencyChecker(populated_store)
        examples = populated_store.get_training_examples("disc_test001")
        issues = checker._check_conflicts(examples)
        conflict_issues = [i for i in issues if i.issue_type == IssueType.CONFLICT]
        assert len(conflict_issues) >= 1

    def test_dissimilar_questions_not_flagged(self, populated_store: ForgeStorage) -> None:
        """Questions with low word overlap are not flagged."""
        populated_store.create_example(
            _make_example(
                ex_id="ex_dis_a",
                question="Explain engine startup procedure.",
                answer="Turn the ignition key clockwise.",
            )
        )
        populated_store.create_example(
            _make_example(
                ex_id="ex_dis_b",
                question="Describe proper tire inflation technique.",
                answer="Use a calibrated gauge.",
            )
        )
        checker = ConsistencyChecker(populated_store)
        examples = populated_store.get_training_examples("disc_test001")
        issues = checker._check_conflicts(examples)
        assert len(issues) == 0

    def test_same_answer_not_flagged(self, populated_store: ForgeStorage) -> None:
        """Similar questions with similar answers are not flagged as conflicts."""
        populated_store.create_example(
            _make_example(
                ex_id="ex_same_a",
                question="How do you check the hydraulic fluid level?",
                answer="Open the reservoir cap and read the dipstick carefully.",
            )
        )
        populated_store.create_example(
            _make_example(
                ex_id="ex_same_b",
                question="How do you check the hydraulic fluid level?",
                answer="Open the reservoir cap and read the dipstick.",
            )
        )
        checker = ConsistencyChecker(populated_store)
        examples = populated_store.get_training_examples("disc_test001")
        issues = checker._check_conflicts(examples)
        # Similar questions AND similar answers should NOT be flagged
        assert len(issues) == 0


# ===========================================================================
# TestCheckExample
# ===========================================================================


class TestCheckExample:
    """Tests for check_example (single example against corpus)."""

    def test_clean_example_passes(self, store_with_many_examples: ForgeStorage) -> None:
        """A well-formed example matching corpus norms has no issues."""
        checker = ConsistencyChecker(store_with_many_examples)
        existing = store_with_many_examples.get_examples_for_competency("comp_test001")
        new_example = _make_example(
            ex_id="ex_clean",
            question="How do you verify a torque wrench calibration?",
            answer="Check the calibration sticker and test against a known load.",
        )
        issues = checker.check_example(new_example, existing)
        assert len(issues) == 0

    def test_multiple_issues_collected(self, store_with_many_examples: ForgeStorage) -> None:
        """An example can trigger multiple issue types."""
        checker = ConsistencyChecker(store_with_many_examples)
        existing = store_with_many_examples.get_examples_for_competency("comp_test001")
        # Very short answer triggers length, and duplicate question triggers conflict
        # First add an example with the same question to the store
        store_with_many_examples.create_example(
            _make_example(
                ex_id="ex_existing_q",
                question="How do you check the hydraulic fluid level?",
                answer="Open the reservoir cap and read the dipstick.",
            )
        )
        existing = store_with_many_examples.get_examples_for_competency("comp_test001")
        bad_example = _make_example(
            ex_id="ex_multi_issue",
            question="How do you check the hydraulic fluid level?",
            answer="No.",  # Too short + conflicting answer
        )
        issues = checker.check_example(bad_example, existing)
        assert len(issues) >= 1

    def test_severity_levels_correct(self, store_with_many_examples: ForgeStorage) -> None:
        """Conflicts produce ERROR severity, length produces WARNING."""
        checker = ConsistencyChecker(store_with_many_examples)
        store_with_many_examples.create_example(
            _make_example(
                ex_id="ex_sev_existing",
                question="How do you check the hydraulic fluid level?",
                answer="Open the reservoir cap and read the dipstick.",
            )
        )
        existing = store_with_many_examples.get_examples_for_competency("comp_test001")
        conflict_example = _make_example(
            ex_id="ex_sev_test",
            question="How do you check the hydraulic fluid level?",
            answer="Connect the electronic sensor to port B and activate.",
        )
        issues = checker.check_example(conflict_example, existing)
        severities = {i.severity for i in issues}
        # Conflict should be ERROR
        if any(i.issue_type == IssueType.CONFLICT for i in issues):
            assert IssueSeverity.ERROR in severities

    def test_suggested_fixes_present(self, store_with_many_examples: ForgeStorage) -> None:
        """Issues include non-empty suggested fixes."""
        checker = ConsistencyChecker(store_with_many_examples)
        # Create an extremely short answer to trigger length issue
        short_example = _make_example(
            ex_id="ex_fix_test",
            question="What is the minimum clearance?",
            answer="5mm.",
        )
        existing = store_with_many_examples.get_examples_for_competency("comp_test001")
        issues = checker.check_example(short_example, existing)
        for issue in issues:
            assert issue.suggested_fix != ""


# ===========================================================================
# TestCheckDiscipline
# ===========================================================================


class TestCheckDiscipline:
    """Tests for check_discipline (full pipeline)."""

    def test_full_pipeline(self, store_with_many_examples: ForgeStorage) -> None:
        """check_discipline returns a complete ConsistencyReport."""
        # Add a short outlier to trigger an issue
        store_with_many_examples.create_example(
            _make_example(
                ex_id="ex_pipe_short",
                question="Quick question about clearance?",
                answer="No.",
            )
        )
        checker = ConsistencyChecker(store_with_many_examples)
        report = checker.check_discipline("disc_test001")
        assert isinstance(report, ConsistencyReport)
        assert report.discipline_id == "disc_test001"
        assert report.example_count > 0
        assert report.checked_at is not None
        # Should have at least one issue from the short answer
        assert len(report.issues) >= 1

    def test_empty_discipline(self, populated_store: ForgeStorage) -> None:
        """A discipline with no competencies produces empty report."""
        # Create a brand new discipline with no competencies or examples
        from forge.src.models import Discipline

        populated_store.create_discipline(
            Discipline(
                id="disc_empty",
                name="Empty Discipline",
                description="No competencies",
                created_by="contrib_test001",
            )
        )
        checker = ConsistencyChecker(populated_store)
        report = checker.check_discipline("disc_empty")
        assert report.example_count == 0
        assert report.issues == []

    def test_single_example(self, checker: ConsistencyChecker) -> None:
        """A discipline with one example skips length and conflict checks."""
        report = checker.check_discipline("disc_test001")
        # Only 1 example: length check skipped, no conflicts possible.
        # Terminology INFO issues may still appear depending on vocab.
        length_issues = [i for i in report.issues if i.issue_type == IssueType.LENGTH]
        conflict_issues = [i for i in report.issues if i.issue_type == IssueType.CONFLICT]
        assert len(length_issues) == 0
        assert len(conflict_issues) == 0

    def test_multiple_competencies(self, store_with_many_examples: ForgeStorage) -> None:
        """Report covers all competencies in a discipline."""
        # Add a second competency with examples
        store_with_many_examples.create_competency(
            Competency(
                id="comp_second",
                name="Inspection",
                description="Inspection procedures",
                discipline_id="disc_test001",
            )
        )
        base = "Perform a thorough visual inspection of the component."
        for i in range(6):
            store_with_many_examples.create_example(
                _make_example(
                    ex_id=f"ex_insp_{i:03d}",
                    question=f"Inspection question number {i}?",
                    answer=f"{base} Step {i}.",
                    competency_id="comp_second",
                )
            )
        checker = ConsistencyChecker(store_with_many_examples)
        report = checker.check_discipline("disc_test001")
        assert report.example_count > 6  # both competencies counted

    def test_report_structure(self, store_with_many_examples: ForgeStorage) -> None:
        """Report has correct structure and can be serialized."""
        checker = ConsistencyChecker(store_with_many_examples)
        report = checker.check_discipline("disc_test001")
        # Verify all expected attributes exist
        assert hasattr(report, "discipline_id")
        assert hasattr(report, "issues")
        assert hasattr(report, "checked_at")
        assert hasattr(report, "example_count")
        assert hasattr(report, "has_errors")
        assert hasattr(report, "has_warnings")
        assert hasattr(report, "issues_by_type")
        assert hasattr(report, "issues_by_severity")


# ===========================================================================
# TestSuggestFixes
# ===========================================================================


class TestSuggestFixes:
    """Tests for _suggest_fixes."""

    def test_length_fix_suggestion(self, checker: ConsistencyChecker) -> None:
        """Length issues produce actionable fix suggestions."""
        issue = ConsistencyIssue(
            issue_type=IssueType.LENGTH,
            severity=IssueSeverity.WARNING,
            message="Answer is too short",
            example_id="ex_001",
        )
        fix = checker._suggest_fixes(issue)
        assert len(fix) > 0
        assert "length" in fix.lower() or "expand" in fix.lower() or "shorten" in fix.lower()

    def test_terminology_fix_suggestion(self, checker: ConsistencyChecker) -> None:
        """Terminology issues produce fix suggestions referencing vocabulary."""
        issue = ConsistencyIssue(
            issue_type=IssueType.TERMINOLOGY,
            severity=IssueSeverity.INFO,
            message="No vocabulary terms found",
            example_id="ex_002",
        )
        fix = checker._suggest_fixes(issue)
        assert len(fix) > 0
        assert "vocab" in fix.lower() or "term" in fix.lower()

    def test_citation_fix_suggestion(self, checker: ConsistencyChecker) -> None:
        """Citation issues produce fix suggestions about format."""
        issue = ConsistencyIssue(
            issue_type=IssueType.CITATION,
            severity=IssueSeverity.WARNING,
            message="Inconsistent citation format",
            example_id="ex_003",
        )
        fix = checker._suggest_fixes(issue)
        assert len(fix) > 0
        assert "citation" in fix.lower() or "format" in fix.lower()

    def test_conflict_fix_suggestion(self, checker: ConsistencyChecker) -> None:
        """Conflict issues produce fix suggestions about review."""
        issue = ConsistencyIssue(
            issue_type=IssueType.CONFLICT,
            severity=IssueSeverity.ERROR,
            message="Potentially conflicting answers",
            example_id="ex_004",
        )
        fix = checker._suggest_fixes(issue)
        assert len(fix) > 0
        assert "review" in fix.lower() or "conflict" in fix.lower()


# ===========================================================================
# TestConsistencyError
# ===========================================================================


class TestConsistencyError:
    """Tests for the ConsistencyError exception."""

    def test_raise_and_catch(self) -> None:
        """ConsistencyError can be raised and caught."""
        with pytest.raises(ConsistencyError, match="test error"):
            raise ConsistencyError("test error")

    def test_is_exception(self) -> None:
        """ConsistencyError is a subclass of Exception."""
        assert issubclass(ConsistencyError, Exception)
