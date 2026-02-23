"""
Section numbering scheme detection and validation.

Detects numbering patterns (1.1, 1.1.1, A.1, etc.) from heading text
and validates consistency across the document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Pattern for decimal numbering: "1", "1.1", "1.1.1", etc.
_DECIMAL_RE = re.compile(r"^(\d+(?:\.\d+)*)")
# Pattern for letter-prefixed numbering: "A.1", "E.5.3.5", etc.
_LETTER_PREFIX_RE = re.compile(r"^([A-Z](?:\.\d+)+)")


def _extract_number(heading: str) -> str | None:
    """Extract the leading number from a heading, or None.

    Args:
        heading: Heading text to parse.

    Returns:
        The number string (e.g., "1.2.3") or None if not numbered.
    """
    text = heading.strip()
    match = _DECIMAL_RE.match(text)
    if match:
        return match.group(1)
    match = _LETTER_PREFIX_RE.match(text)
    if match:
        return match.group(1)
    return None


def _number_depth(number_str: str) -> int:
    """Count the depth of a numbering string.

    Args:
        number_str: A numbering string like "1.2.3" or "A.1.2".

    Returns:
        Number of components (e.g., "1.2.3" -> 3).
    """
    return len(number_str.split("."))


@dataclass
class NumberingScheme:
    """Detected numbering scheme for a document.

    Attributes:
        scheme_type: One of "decimal", "letter_prefix", "unnumbered", "mixed".
        max_depth: Maximum nesting depth found in numbering.
        total_numbered: Count of headings with detected numbers.
    """

    scheme_type: str
    max_depth: int = 0
    total_numbered: int = 0

    @classmethod
    def detect(cls, headings: list[str]) -> NumberingScheme:
        """Detect the numbering scheme from a list of heading texts.

        Args:
            headings: List of heading text strings.

        Returns:
            NumberingScheme describing the detected pattern.
        """
        if not headings:
            return cls(scheme_type="unnumbered")

        decimal_count = 0
        letter_count = 0
        max_depth = 0

        for h in headings:
            text = h.strip()
            if _DECIMAL_RE.match(text):
                decimal_count += 1
                num = _DECIMAL_RE.match(text).group(1)  # type: ignore[union-attr]
                max_depth = max(max_depth, _number_depth(num))
            elif _LETTER_PREFIX_RE.match(text):
                letter_count += 1
                num = _LETTER_PREFIX_RE.match(text).group(1)  # type: ignore[union-attr]
                max_depth = max(max_depth, _number_depth(num))

        total_numbered = decimal_count + letter_count

        if total_numbered == 0:
            return cls(scheme_type="unnumbered")

        # Determine scheme type
        ratio = total_numbered / len(headings)
        if letter_count > decimal_count:
            scheme_type = "letter_prefix"
        elif ratio >= 0.5:
            scheme_type = "decimal"
        else:
            scheme_type = "mixed"

        return cls(
            scheme_type=scheme_type,
            max_depth=max_depth,
            total_numbered=total_numbered,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize to dictionary.

        Returns:
            Dictionary with scheme details.
        """
        return {
            "scheme_type": self.scheme_type,
            "max_depth": self.max_depth,
            "total_numbered": self.total_numbered,
        }


class NumberingValidator:
    """Validates section numbering consistency."""

    @staticmethod
    def validate(headings: list[str]) -> list[dict[str, str]]:
        """Check numbering for gaps, duplicates, and inconsistencies.

        Args:
            headings: List of heading text strings.

        Returns:
            List of issue dictionaries with 'type' and 'message' keys.
        """
        issues: list[dict[str, str]] = []
        numbers: list[str] = []
        unnumbered_indices: list[int] = []

        for i, h in enumerate(headings):
            num = _extract_number(h)
            if num is not None:
                numbers.append(num)
            else:
                unnumbered_indices.append(i)

        if not numbers:
            return issues

        # Check for duplicates
        seen: set[str] = set()
        for num in numbers:
            if num in seen:
                issues.append({
                    "type": "duplicate_number",
                    "message": f"Duplicate section number: {num}",
                })
            seen.add(num)

        # Check for gaps at top level
        top_level = [n for n in numbers if "." not in n]
        NumberingValidator._check_gaps(top_level, issues)

        # Flag unnumbered headings in a mostly-numbered document
        if unnumbered_indices and len(numbers) > len(unnumbered_indices):
            for idx in unnumbered_indices:
                issues.append({
                    "type": "unnumbered_in_numbered_doc",
                    "message": (
                        f"Heading at index {idx} is unnumbered "
                        f"in a numbered document: {headings[idx]!r}"
                    ),
                })

        return issues

    @staticmethod
    def _check_gaps(
        top_level: list[str], issues: list[dict[str, str]]
    ) -> None:
        """Check for gaps in top-level numbering.

        Args:
            top_level: List of top-level number strings.
            issues: List to append issues to.
        """
        int_numbers = []
        for n in top_level:
            try:
                int_numbers.append(int(n))
            except ValueError:
                continue

        if len(int_numbers) < 2:
            return

        for i in range(1, len(int_numbers)):
            expected = int_numbers[i - 1] + 1
            actual = int_numbers[i]
            if actual > expected:
                issues.append({
                    "type": "numbering_gap",
                    "message": (
                        f"Gap in numbering: expected {expected} "
                        f"after {int_numbers[i-1]}, found {actual}"
                    ),
                })
