"""Stage 1: Deterministic metadata pre-filter.

Reduces the search space by filtering chunks on metadata fields
before semantic search. Targets 80-90% reduction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from chonk.core.document import Chunk
from chonk.enrichment.extractor import ENRICHMENT_FIELDS_KEY


class FilterOperator(str, Enum):
    """Operators for metadata filter conditions.

    Attributes:
        EQ: Exact string match (case-insensitive).
        CONTAINS: Substring match (case-insensitive).
        REGEX: Regex pattern match.
        EXISTS: Field exists and is non-empty.
        NOT_EXISTS: Field does not exist or is empty.
        GT: Greater than (numeric comparison).
        LT: Less than (numeric comparison).
    """

    EQ = "eq"
    CONTAINS = "contains"
    REGEX = "regex"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    GT = "gt"
    LT = "lt"


@dataclass
class FilterCondition:
    """A single metadata filter condition.

    Args:
        field_name: Metadata field to check.
        operator: Comparison operator.
        value: Value to compare against. Not used for EXISTS/NOT_EXISTS.
    """

    field_name: str
    operator: FilterOperator
    value: Any = None


@dataclass
class FilterCriteria:
    """A set of filter conditions applied as AND logic.

    All conditions must match for a chunk to pass the filter.

    Args:
        conditions: List of filter conditions.
        document_type: Optional document type filter.
    """

    conditions: list[FilterCondition] = field(default_factory=list)
    document_type: str | None = None

    def add(
        self,
        field_name: str,
        operator: FilterOperator,
        value: Any = None,
    ) -> FilterCriteria:
        """Add a condition and return self for chaining.

        Args:
            field_name: Metadata field name.
            operator: Filter operator.
            value: Comparison value.

        Returns:
            Self for method chaining.
        """
        self.conditions.append(
            FilterCondition(field_name=field_name, operator=operator, value=value)
        )
        return self


@dataclass
class FilterResult:
    """Result of Stage 1 metadata filtering.

    Args:
        passed: Chunks that passed all conditions.
        total_input: Total chunks before filtering.
        reduction_ratio: Fraction of chunks removed (0.0-1.0).
    """

    passed: list[Chunk]
    total_input: int
    reduction_ratio: float


class MetadataFilter:
    """Stage 1 deterministic metadata pre-filter.

    Filters chunks based on enrichment metadata fields using
    exact match, contains, regex, and existence operators.

    Example::

        mf = MetadataFilter()
        criteria = FilterCriteria()
        criteria.add("tm_number", FilterOperator.CONTAINS, "9-2320")
        criteria.add("maintenance_level", FilterOperator.EQ, "organizational")
        result = mf.filter(chunks, criteria)
    """

    def filter(self, chunks: list[Chunk], criteria: FilterCriteria) -> FilterResult:
        """Apply filter criteria to chunks.

        Args:
            chunks: Chunks to filter.
            criteria: Filter conditions (AND logic).

        Returns:
            FilterResult with passed chunks and reduction stats.
        """
        if not criteria.conditions and not criteria.document_type:
            return FilterResult(
                passed=list(chunks),
                total_input=len(chunks),
                reduction_ratio=0.0,
            )

        passed = [c for c in chunks if self._matches(c, criteria)]
        total = len(chunks)
        removed = total - len(passed)
        ratio = removed / total if total > 0 else 0.0

        return FilterResult(
            passed=passed,
            total_input=total,
            reduction_ratio=round(ratio, 3),
        )

    def _matches(self, chunk: Chunk, criteria: FilterCriteria) -> bool:
        """Check if a chunk matches all filter conditions.

        Args:
            chunk: Chunk to check.
            criteria: Filter conditions.

        Returns:
            True if all conditions match.
        """
        if criteria.document_type:
            doc_type = chunk.system_metadata.get("document_type", "")
            if str(doc_type).lower() != criteria.document_type.lower():
                return False

        enrichment = chunk.system_metadata.get(ENRICHMENT_FIELDS_KEY, {})

        for condition in criteria.conditions:
            value = enrichment.get(condition.field_name)
            if not self._check_condition(value, condition):
                return False
        return True

    @staticmethod
    def _check_condition(value: Any, condition: FilterCondition) -> bool:
        """Evaluate a single condition against a field value.

        Args:
            value: The field value (may be None).
            condition: The condition to check.

        Returns:
            True if the condition is satisfied.
        """
        op = condition.operator

        if op == FilterOperator.EXISTS:
            return value is not None and str(value).strip() != ""

        if op == FilterOperator.NOT_EXISTS:
            return value is None or str(value).strip() == ""

        if value is None:
            return False

        str_value = str(value)
        cmp_value = str(condition.value) if condition.value is not None else ""

        if op == FilterOperator.EQ:
            return str_value.lower() == cmp_value.lower()

        if op == FilterOperator.CONTAINS:
            return cmp_value.lower() in str_value.lower()

        if op == FilterOperator.REGEX:
            try:
                return bool(re.search(cmp_value, str_value, re.IGNORECASE))
            except re.error:
                return False

        if op == FilterOperator.GT:
            try:
                return float(str_value) > float(cmp_value)
            except (ValueError, TypeError):
                return False

        if op == FilterOperator.LT:
            try:
                return float(str_value) < float(cmp_value)
            except (ValueError, TypeError):
                return False

        return False
