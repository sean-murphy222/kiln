"""Stage 3: Result validation against structural patterns.

Validates that retrieved chunks actually match expected structural
patterns for the query type, filtering out false positives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from chonk.core.document import Chunk
from chonk.enrichment.extractor import ENRICHMENT_FIELDS_KEY
from chonk.retrieval.search import ScoredChunk


class ValidationCheck(str, Enum):
    """Types of validation checks.

    Attributes:
        HAS_FIELD: Chunk must have a specific enrichment field.
        CONTENT_MATCHES: Chunk content must match a regex.
        MIN_LENGTH: Chunk content must meet minimum length.
        HAS_HIERARCHY: Chunk must have a non-empty hierarchy path.
    """

    HAS_FIELD = "has_field"
    CONTENT_MATCHES = "content_matches"
    MIN_LENGTH = "min_length"
    HAS_HIERARCHY = "has_hierarchy"


@dataclass
class ValidationRule:
    """A single validation rule for result checking.

    Args:
        check: Type of validation check.
        value: Parameter for the check (field name, regex, min length).
        weight: How much this rule affects the validation score.
        description: Human-readable description.
    """

    check: ValidationCheck
    value: Any = None
    weight: float = 1.0
    description: str = ""


@dataclass
class ValidationResult:
    """Result of validating a single scored chunk.

    Args:
        chunk: The validated scored chunk.
        passed_rules: Number of rules passed.
        total_rules: Total rules checked.
        validation_score: Weighted pass rate (0.0-1.0).
        adjusted_score: Original score * validation_score.
        details: Per-rule pass/fail details.
    """

    chunk: ScoredChunk
    passed_rules: int
    total_rules: int
    validation_score: float
    adjusted_score: float
    details: dict[str, bool] = field(default_factory=dict)


class ResultValidator:
    """Stage 3 validator for retrieved chunks.

    Checks each retrieved chunk against structural validation
    rules and adjusts scores accordingly. Chunks that fail
    critical rules are demoted.

    Args:
        rules: Validation rules to apply.
        min_validation_score: Minimum score to keep a result.

    Example::

        validator = ResultValidator(rules=[
            ValidationRule(
                check=ValidationCheck.HAS_FIELD,
                value="tm_number",
                description="Must reference a TM",
            ),
            ValidationRule(
                check=ValidationCheck.MIN_LENGTH,
                value=50,
                description="Must have substantive content",
            ),
        ])
        validated = validator.validate(scored_chunks)
    """

    def __init__(
        self,
        rules: list[ValidationRule] | None = None,
        min_validation_score: float = 0.0,
    ) -> None:
        self._rules = rules or []
        self._min_score = min_validation_score

    def validate(self, results: list[ScoredChunk]) -> list[ValidationResult]:
        """Validate and re-score search results.

        Args:
            results: Scored chunks from Stage 2.

        Returns:
            ValidationResults sorted by adjusted score descending.
        """
        if not self._rules:
            return [
                ValidationResult(
                    chunk=sc,
                    passed_rules=0,
                    total_rules=0,
                    validation_score=1.0,
                    adjusted_score=sc.score,
                )
                for sc in results
            ]

        validated: list[ValidationResult] = []
        for sc in results:
            vr = self._validate_chunk(sc)
            if vr.validation_score >= self._min_score:
                validated.append(vr)

        validated.sort(key=lambda v: v.adjusted_score, reverse=True)
        return validated

    def _validate_chunk(self, scored: ScoredChunk) -> ValidationResult:
        """Apply all validation rules to a single chunk.

        Args:
            scored: The scored chunk to validate.

        Returns:
            ValidationResult with pass/fail details.
        """
        details: dict[str, bool] = {}
        total_weight = 0.0
        passed_weight = 0.0
        passed_count = 0

        for rule in self._rules:
            passed = self._check_rule(scored.chunk, rule)
            key = rule.description or f"{rule.check.value}:{rule.value}"
            details[key] = passed
            total_weight += rule.weight
            if passed:
                passed_weight += rule.weight
                passed_count += 1

        vscore = passed_weight / total_weight if total_weight > 0 else 1.0

        return ValidationResult(
            chunk=scored,
            passed_rules=passed_count,
            total_rules=len(self._rules),
            validation_score=round(vscore, 3),
            adjusted_score=round(scored.score * vscore, 3),
            details=details,
        )

    @staticmethod
    def _check_rule(chunk: Chunk, rule: ValidationRule) -> bool:
        """Evaluate a single validation rule against a chunk.

        Args:
            chunk: Chunk to check.
            rule: Validation rule to apply.

        Returns:
            True if the rule passes.
        """
        check = rule.check

        if check == ValidationCheck.HAS_FIELD:
            fields = chunk.system_metadata.get(ENRICHMENT_FIELDS_KEY, {})
            return rule.value in fields and bool(fields[rule.value])

        if check == ValidationCheck.CONTENT_MATCHES:
            try:
                return bool(re.search(str(rule.value), chunk.content, re.IGNORECASE))
            except re.error:
                return False

        if check == ValidationCheck.MIN_LENGTH:
            try:
                return len(chunk.content) >= int(rule.value)
            except (ValueError, TypeError):
                return False

        if check == ValidationCheck.HAS_HIERARCHY:
            return bool(chunk.hierarchy_path.strip())

        return False
