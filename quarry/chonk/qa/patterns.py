"""Pattern library for zero-value content detection.

Compiled regex patterns for identifying TOC entries, distribution
statements, boilerplate text, and other filterable content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from chonk.qa.filter_log import FilterCategory


@dataclass
class CompiledPattern:
    """A compiled regex pattern for content matching.

    Attributes:
        name: Rule name for audit trail.
        category: Filter category this pattern detects.
        pattern: Compiled regex.
        match_type: How to apply: 'full', 'prefix', or 'contains'.
    """

    name: str
    category: FilterCategory
    pattern: re.Pattern[str]
    match_type: str  # "full" | "prefix" | "contains"


# -----------------------------------------------------------------
# TOC Patterns
# -----------------------------------------------------------------

TOC_PATTERNS: list[CompiledPattern] = [
    CompiledPattern(
        name="toc_heading",
        category=FilterCategory.TOC,
        pattern=re.compile(r"^table\s+of\s+contents\s*$", re.IGNORECASE),
        match_type="full",
    ),
    CompiledPattern(
        name="toc_heading_short",
        category=FilterCategory.TOC,
        pattern=re.compile(r"^contents\s*$", re.IGNORECASE),
        match_type="full",
    ),
    CompiledPattern(
        name="toc_dotleader",
        category=FilterCategory.TOC,
        pattern=re.compile(r"^.*\.{3,}\s*\d+\s*$", re.IGNORECASE),
        match_type="full",
    ),
]

# -----------------------------------------------------------------
# Index Patterns
# -----------------------------------------------------------------

INDEX_PATTERNS: list[CompiledPattern] = [
    CompiledPattern(
        name="index_heading",
        category=FilterCategory.INDEX,
        pattern=re.compile(r"^index\s*$", re.IGNORECASE),
        match_type="full",
    ),
    CompiledPattern(
        name="index_alpha",
        category=FilterCategory.INDEX,
        pattern=re.compile(r"^alphabetical\s+index\s*$", re.IGNORECASE),
        match_type="full",
    ),
]

# -----------------------------------------------------------------
# Distribution Statement Patterns
# -----------------------------------------------------------------

DISTRIBUTION_PATTERNS: list[CompiledPattern] = [
    CompiledPattern(
        name="distro_statement",
        category=FilterCategory.DISTRIBUTION,
        pattern=re.compile(r"^distribution\s+statement", re.IGNORECASE),
        match_type="prefix",
    ),
    CompiledPattern(
        name="distro_approved",
        category=FilterCategory.DISTRIBUTION,
        pattern=re.compile(r"approved\s+for\s+public\s+release", re.IGNORECASE),
        match_type="contains",
    ),
    CompiledPattern(
        name="distro_unlimited",
        category=FilterCategory.DISTRIBUTION,
        pattern=re.compile(r"distribution\s+is\s+unlimited", re.IGNORECASE),
        match_type="contains",
    ),
]

# -----------------------------------------------------------------
# Boilerplate Patterns
# -----------------------------------------------------------------

BOILERPLATE_PATTERNS: list[CompiledPattern] = [
    CompiledPattern(
        name="intentionally_blank",
        category=FilterCategory.BOILERPLATE,
        pattern=re.compile(
            r"this\s+page\s+intentionally\s+left\s+blank",
            re.IGNORECASE,
        ),
        match_type="contains",
    ),
    CompiledPattern(
        name="fouo",
        category=FilterCategory.BOILERPLATE,
        pattern=re.compile(r"^for\s+official\s+use\s+only", re.IGNORECASE),
        match_type="prefix",
    ),
    CompiledPattern(
        name="proprietary",
        category=FilterCategory.BOILERPLATE,
        pattern=re.compile(r"proprietary.*confidential", re.IGNORECASE),
        match_type="contains",
    ),
]

# -----------------------------------------------------------------
# All patterns combined
# -----------------------------------------------------------------

ALL_PATTERNS: list[CompiledPattern] = (
    TOC_PATTERNS + INDEX_PATTERNS + DISTRIBUTION_PATTERNS + BOILERPLATE_PATTERNS
)


def match_patterns(
    text: str,
    candidates: list[CompiledPattern] | None = None,
) -> CompiledPattern | None:
    """Return the first matching pattern or None.

    Args:
        text: Text content to check.
        candidates: Patterns to try. Defaults to ALL_PATTERNS.

    Returns:
        First matching CompiledPattern, or None.
    """
    if candidates is None:
        candidates = ALL_PATTERNS

    stripped = text.strip()
    for cp in candidates:
        if cp.match_type == "full":
            if cp.pattern.match(stripped):
                return cp
        elif cp.match_type == "prefix":
            if cp.pattern.match(stripped):
                return cp
        elif cp.match_type == "contains":
            if cp.pattern.search(stripped):
                return cp
    return None
