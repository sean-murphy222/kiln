"""Consistency checking engine for Step 4 of Forge curriculum building.

Analyzes training examples within a discipline for response length
consistency, terminology usage, citation format uniformity, and
potential conflicts between examples. Produces actionable reports
that guide contributors toward higher-quality training data.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from forge.src.models import Example
from forge.src.storage import ForgeStorage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_EXAMPLES_FOR_LENGTH_CHECK = 5
_LENGTH_SHORT_RATIO = 0.5
_LENGTH_LONG_RATIO = 2.0
_CONFLICT_SIMILARITY_THRESHOLD = 0.70
_CONFLICT_ANSWER_SIMILARITY_THRESHOLD = 0.70

# Military manual citation pattern: TYPE followed by numbers/dashes
_CITATION_PATTERN = re.compile(r"\b([A-Z]{2,4})\s*(\d[\d\-]+\d)\b")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class IssueSeverity(str, Enum):
    """Severity level of a consistency issue.

    Attributes:
        INFO: Informational, no action required.
        WARNING: Potential problem worth reviewing.
        ERROR: Likely error requiring resolution.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class IssueType(str, Enum):
    """Category of consistency issue.

    Attributes:
        LENGTH: Response length outlier.
        TERMINOLOGY: Vocabulary term not used.
        CITATION: Citation format inconsistency.
        CONFLICT: Potentially conflicting examples.
    """

    LENGTH = "length_outlier"
    TERMINOLOGY = "terminology"
    CITATION = "citation_format"
    CONFLICT = "conflict"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConsistencyError(Exception):
    """Raised for consistency checking errors."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ConsistencyIssue:
    """A single consistency issue found during checking.

    Attributes:
        issue_type: Category of the issue.
        severity: How serious the issue is.
        message: Human-readable description.
        example_id: ID of the example with the issue.
        suggested_fix: Actionable suggestion for resolving the issue.
        details: Additional structured data about the issue.
    """

    issue_type: IssueType
    severity: IssueSeverity
    message: str
    example_id: str
    suggested_fix: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation of the issue.
        """
        return {
            "issue_type": self.issue_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "example_id": self.example_id,
            "suggested_fix": self.suggested_fix,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConsistencyIssue:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with issue fields.

        Returns:
            ConsistencyIssue instance.
        """
        return cls(
            issue_type=IssueType(data["issue_type"]),
            severity=IssueSeverity(data["severity"]),
            message=data["message"],
            example_id=data["example_id"],
            suggested_fix=data.get("suggested_fix", ""),
            details=data.get("details", {}),
        )


@dataclass
class ConsistencyReport:
    """Report from a full consistency check of a discipline.

    Attributes:
        discipline_id: The discipline that was checked.
        issues: All issues found.
        checked_at: When the check was performed.
        example_count: Total examples checked.
    """

    discipline_id: str
    issues: list[ConsistencyIssue]
    checked_at: datetime
    example_count: int

    @property
    def has_errors(self) -> bool:
        """Return True if any issue has ERROR severity.

        Returns:
            Whether any error-level issues exist.
        """
        return any(i.severity == IssueSeverity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Return True if any issue has WARNING severity.

        Returns:
            Whether any warning-level issues exist.
        """
        return any(i.severity == IssueSeverity.WARNING for i in self.issues)

    @property
    def issues_by_type(self) -> dict[IssueType, list[ConsistencyIssue]]:
        """Group issues by their type.

        Returns:
            Dictionary mapping issue type to list of issues.
        """
        grouped: dict[IssueType, list[ConsistencyIssue]] = defaultdict(list)
        for issue in self.issues:
            grouped[issue.issue_type].append(issue)
        return dict(grouped)

    @property
    def issues_by_severity(self) -> dict[IssueSeverity, list[ConsistencyIssue]]:
        """Group issues by their severity.

        Returns:
            Dictionary mapping severity to list of issues.
        """
        grouped: dict[IssueSeverity, list[ConsistencyIssue]] = defaultdict(list)
        for issue in self.issues:
            grouped[issue.severity].append(issue)
        return dict(grouped)


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------


class ConsistencyChecker:
    """Checks training examples for consistency within a discipline.

    Runs response length, terminology, citation format, and conflict
    checks against the example corpus stored in ForgeStorage.

    Args:
        storage: ForgeStorage instance for reading examples.

    Example::

        checker = ConsistencyChecker(storage)
        report = checker.check_discipline("disc_001")
        if report.has_errors:
            for issue in report.issues:
                print(issue.message)
    """

    def __init__(self, storage: ForgeStorage) -> None:
        self._storage = storage

    def check_discipline(self, discipline_id: str) -> ConsistencyReport:
        """Run all consistency checks on a discipline.

        Args:
            discipline_id: Discipline to check.

        Returns:
            ConsistencyReport with all issues found.
        """
        issues: list[ConsistencyIssue] = []
        all_examples: list[Example] = []

        competencies = self._storage.get_competencies_for_discipline(discipline_id)
        discipline = self._storage.get_discipline(discipline_id)
        vocab = discipline.vocabulary if discipline else []

        for comp in competencies:
            examples = self._storage.get_examples_for_competency(comp.id)
            all_examples.extend(examples)
            issues.extend(self._check_response_length(examples))
            issues.extend(self._check_terminology(examples, vocab))

        issues.extend(self._check_citation_format(all_examples))
        issues.extend(self._check_conflicts(all_examples))

        # Populate suggested fixes for issues that lack them
        for issue in issues:
            if not issue.suggested_fix:
                issue.suggested_fix = self._suggest_fixes(issue)

        return ConsistencyReport(
            discipline_id=discipline_id,
            issues=issues,
            checked_at=datetime.now(),
            example_count=len(all_examples),
        )

    def check_example(
        self,
        example: Example,
        existing_examples: list[Example],
    ) -> list[ConsistencyIssue]:
        """Check a single example against an existing corpus.

        Useful for pre-submission validation of a new example.

        Args:
            example: The new example to validate.
            existing_examples: Existing examples in the same competency.

        Returns:
            List of issues found for this example.
        """
        issues: list[ConsistencyIssue] = []

        # Length check: include the new example in the pool
        combined = list(existing_examples) + [example]
        if len(combined) >= _MIN_EXAMPLES_FOR_LENGTH_CHECK:
            length_issues = self._check_response_length(combined)
            issues.extend(i for i in length_issues if i.example_id == example.id)

        # Conflict check against existing
        conflict_issues = self._check_conflicts_for_example(example, existing_examples)
        issues.extend(conflict_issues)

        for issue in issues:
            if not issue.suggested_fix:
                issue.suggested_fix = self._suggest_fixes(issue)

        return issues

    # -------------------------------------------------------------------
    # Length check
    # -------------------------------------------------------------------

    def _check_response_length(self, examples: list[Example]) -> list[ConsistencyIssue]:
        """Flag answers significantly shorter or longer than the mean.

        Skips check if fewer than MIN_EXAMPLES_FOR_LENGTH_CHECK examples
        are present (too few data points for meaningful statistics).

        Args:
            examples: Examples within a single competency.

        Returns:
            List of length-related issues.
        """
        if len(examples) < _MIN_EXAMPLES_FOR_LENGTH_CHECK:
            return []

        lengths = [len(ex.ideal_answer) for ex in examples]
        mean_length = sum(lengths) / len(lengths)

        issues: list[ConsistencyIssue] = []
        for ex in examples:
            answer_len = len(ex.ideal_answer)
            if answer_len < mean_length * _LENGTH_SHORT_RATIO:
                issues.append(self._build_length_issue(ex, answer_len, mean_length, "short"))
            elif answer_len > mean_length * _LENGTH_LONG_RATIO:
                issues.append(self._build_length_issue(ex, answer_len, mean_length, "long"))

        return issues

    @staticmethod
    def _build_length_issue(
        example: Example,
        actual: int,
        mean: float,
        direction: str,
    ) -> ConsistencyIssue:
        """Build a ConsistencyIssue for a length outlier.

        Args:
            example: The outlier example.
            actual: Actual answer length in characters.
            mean: Mean answer length for the competency.
            direction: 'short' or 'long'.

        Returns:
            ConsistencyIssue describing the outlier.
        """
        return ConsistencyIssue(
            issue_type=IssueType.LENGTH,
            severity=IssueSeverity.WARNING,
            message=(
                f"Answer is too {direction} ({actual} chars) "
                f"compared to competency mean ({mean:.0f} chars)"
            ),
            example_id=example.id,
            details={"actual_length": actual, "mean_length": round(mean, 1)},
        )

    # -------------------------------------------------------------------
    # Terminology check
    # -------------------------------------------------------------------

    def _check_terminology(
        self,
        examples: list[Example],
        vocabulary: list[str],
    ) -> list[ConsistencyIssue]:
        """Flag examples that use none of the discipline vocabulary terms.

        Args:
            examples: Examples to check.
            vocabulary: Discipline vocabulary terms.

        Returns:
            List of terminology-related issues.
        """
        if not vocabulary:
            return []

        lower_vocab = [term.lower() for term in vocabulary]
        issues: list[ConsistencyIssue] = []

        for ex in examples:
            combined_text = f"{ex.question} {ex.ideal_answer}".lower()
            if not self._text_contains_any_term(combined_text, lower_vocab):
                issues.append(
                    ConsistencyIssue(
                        issue_type=IssueType.TERMINOLOGY,
                        severity=IssueSeverity.INFO,
                        message=("Example does not use any discipline " "vocabulary terms"),
                        example_id=ex.id,
                        details={"vocabulary": vocabulary},
                    )
                )

        return issues

    @staticmethod
    def _text_contains_any_term(text: str, terms: list[str]) -> bool:
        """Check whether text contains at least one term.

        Args:
            text: Lowercased text to search.
            terms: Lowercased vocabulary terms.

        Returns:
            True if any term appears in the text.
        """
        return any(term in text for term in terms)

    # -------------------------------------------------------------------
    # Citation format check
    # -------------------------------------------------------------------

    def _check_citation_format(
        self,
        examples: list[Example],
    ) -> list[ConsistencyIssue]:
        """Detect inconsistent citation formatting across examples.

        Collects all citation-like references and checks whether the
        same manual number appears in different spacing formats (e.g.,
        'TM 1-1520-237-10' vs 'TM1-1520-237-10').

        Args:
            examples: All examples in a discipline.

        Returns:
            List of citation-format issues.
        """
        # Map from normalized citation (no spaces) -> set of original forms
        citation_forms: dict[str, set[str]] = defaultdict(set)
        citation_to_examples: dict[str, set[str]] = defaultdict(set)

        for ex in examples:
            self._collect_citations(ex, citation_forms, citation_to_examples)

        return self._build_citation_issues(citation_forms, citation_to_examples)

    def _collect_citations(
        self,
        example: Example,
        citation_forms: dict[str, set[str]],
        citation_to_examples: dict[str, set[str]],
    ) -> None:
        """Extract citations from one example into tracking dicts.

        Args:
            example: Example to scan.
            citation_forms: Maps normalized citation to original forms.
            citation_to_examples: Maps normalized citation to example IDs.
        """
        text = f"{example.question} {example.ideal_answer}"
        for match in _CITATION_PATTERN.finditer(text):
            original = match.group(0)
            normalized = original.replace(" ", "")
            citation_forms[normalized].add(original)
            citation_to_examples[normalized].add(example.id)

    @staticmethod
    def _build_citation_issues(
        citation_forms: dict[str, set[str]],
        citation_to_examples: dict[str, set[str]],
    ) -> list[ConsistencyIssue]:
        """Build issues for citations with inconsistent formatting.

        Args:
            citation_forms: Maps normalized citation to original forms.
            citation_to_examples: Maps normalized citation to example IDs.

        Returns:
            List of citation-format issues.
        """
        issues: list[ConsistencyIssue] = []
        for normalized, forms in citation_forms.items():
            if len(forms) <= 1:
                continue
            for ex_id in citation_to_examples[normalized]:
                issues.append(
                    ConsistencyIssue(
                        issue_type=IssueType.CITATION,
                        severity=IssueSeverity.WARNING,
                        message=(
                            f"Inconsistent citation format for "
                            f"'{normalized}': found forms "
                            f"{sorted(forms)}"
                        ),
                        example_id=ex_id,
                        details={
                            "normalized": normalized,
                            "forms": sorted(forms),
                        },
                    )
                )
        return issues

    # -------------------------------------------------------------------
    # Conflict check
    # -------------------------------------------------------------------

    def _check_conflicts(
        self,
        examples: list[Example],
    ) -> list[ConsistencyIssue]:
        """Find examples with similar questions but different answers.

        Uses Jaccard similarity on word tokens. If question similarity
        exceeds the threshold but answer similarity is below it, the
        pair is flagged as a potential conflict.

        Args:
            examples: All examples to compare pairwise.

        Returns:
            List of conflict issues.
        """
        issues: list[ConsistencyIssue] = []
        seen_pairs: set[tuple[str, str]] = set()

        for i, ex_a in enumerate(examples):
            for ex_b in examples[i + 1 :]:
                pair_key = (ex_a.id, ex_b.id)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                issue = self._compare_pair(ex_a, ex_b)
                if issue is not None:
                    issues.append(issue)

        return issues

    def _compare_pair(
        self,
        ex_a: Example,
        ex_b: Example,
    ) -> ConsistencyIssue | None:
        """Compare two examples for potential conflict.

        Args:
            ex_a: First example.
            ex_b: Second example.

        Returns:
            ConsistencyIssue if a conflict is detected, else None.
        """
        q_sim = self._jaccard_similarity(ex_a.question, ex_b.question)
        if q_sim < _CONFLICT_SIMILARITY_THRESHOLD:
            return None

        a_sim = self._jaccard_similarity(ex_a.ideal_answer, ex_b.ideal_answer)
        if a_sim >= _CONFLICT_ANSWER_SIMILARITY_THRESHOLD:
            return None

        return ConsistencyIssue(
            issue_type=IssueType.CONFLICT,
            severity=IssueSeverity.ERROR,
            message=(
                f"Potentially conflicting answers: "
                f"questions are {q_sim:.0%} similar but "
                f"answers are only {a_sim:.0%} similar"
            ),
            example_id=ex_a.id,
            details={
                "other_example_id": ex_b.id,
                "question_similarity": round(q_sim, 3),
                "answer_similarity": round(a_sim, 3),
            },
        )

    def _check_conflicts_for_example(
        self,
        example: Example,
        existing: list[Example],
    ) -> list[ConsistencyIssue]:
        """Check a single example for conflicts against existing corpus.

        Args:
            example: New example to validate.
            existing: Existing examples to compare against.

        Returns:
            List of conflict issues for the new example.
        """
        issues: list[ConsistencyIssue] = []
        for other in existing:
            issue = self._compare_pair(example, other)
            if issue is not None:
                # Ensure the issue references the new example
                issue.example_id = example.id
                issues.append(issue)
        return issues

    @staticmethod
    def _jaccard_similarity(text_a: str, text_b: str) -> float:
        """Compute Jaccard similarity between two texts.

        Tokenizes on whitespace and punctuation boundaries,
        then computes |intersection| / |union| of token sets.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        tokens_a = set(re.findall(r"\w+", text_a.lower()))
        tokens_b = set(re.findall(r"\w+", text_b.lower()))
        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    # -------------------------------------------------------------------
    # Fix suggestions
    # -------------------------------------------------------------------

    @staticmethod
    def _suggest_fixes(issue: ConsistencyIssue) -> str:
        """Generate an actionable fix suggestion for an issue.

        Args:
            issue: The issue to suggest a fix for.

        Returns:
            Human-readable suggestion string.
        """
        suggestions = {
            IssueType.LENGTH: (
                "Adjust the answer length to be closer to the "
                "competency mean. Expand short answers with more "
                "detail or shorten verbose answers."
            ),
            IssueType.TERMINOLOGY: (
                "Incorporate discipline vocabulary terms into "
                "the question or answer to maintain consistency."
            ),
            IssueType.CITATION: (
                "Standardize the citation format to match the "
                "most common pattern used in other examples."
            ),
            IssueType.CONFLICT: (
                "Review both examples and resolve the conflict. "
                "Either reconcile the answers or mark one for "
                "revision."
            ),
        }
        return suggestions.get(issue.issue_type, "Review this issue manually.")
