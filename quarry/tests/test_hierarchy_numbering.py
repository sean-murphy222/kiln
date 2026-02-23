"""Tests for section numbering scheme detection and validation."""

from __future__ import annotations

from chonk.core.document import Block, BlockType
from chonk.hierarchy.numbering import NumberingScheme, NumberingValidator

# ===================================================================
# Helpers
# ===================================================================


def _heading(content: str, level: int = 1) -> Block:
    """Create a heading block."""
    return Block(
        id=Block.generate_id(),
        type=BlockType.HEADING,
        content=content,
        heading_level=level,
    )


def _text(content: str) -> Block:
    """Create a text block."""
    return Block(id=Block.generate_id(), type=BlockType.TEXT, content=content)


# ===================================================================
# NumberingScheme
# ===================================================================


class TestNumberingScheme:
    """Tests for numbering scheme dataclass."""

    def test_detect_decimal(self):
        headings = ["1 Intro", "1.1 Overview", "1.1.1 Details", "2 Methods"]
        scheme = NumberingScheme.detect(headings)
        assert scheme.scheme_type == "decimal"

    def test_detect_letter_prefix(self):
        headings = ["A.1 Appendix", "A.2 Another", "B.1 Next"]
        scheme = NumberingScheme.detect(headings)
        assert scheme.scheme_type == "letter_prefix"

    def test_detect_unnumbered(self):
        headings = ["Introduction", "Background", "Methods", "Results"]
        scheme = NumberingScheme.detect(headings)
        assert scheme.scheme_type == "unnumbered"

    def test_detect_mixed(self):
        headings = ["1 First", "FOREWORD", "2 Second", "Appendix A"]
        scheme = NumberingScheme.detect(headings)
        assert scheme.scheme_type in ("decimal", "mixed")

    def test_max_depth_decimal(self):
        headings = ["1 A", "1.1 B", "1.1.1 C", "1.1.1.1 D"]
        scheme = NumberingScheme.detect(headings)
        assert scheme.max_depth == 4

    def test_max_depth_flat(self):
        headings = ["1 A", "2 B", "3 C"]
        scheme = NumberingScheme.detect(headings)
        assert scheme.max_depth == 1

    def test_to_dict(self):
        scheme = NumberingScheme(scheme_type="decimal", max_depth=3, total_numbered=5)
        d = scheme.to_dict()
        assert d["scheme_type"] == "decimal"
        assert d["max_depth"] == 3


# ===================================================================
# NumberingValidator
# ===================================================================


class TestNumberingValidator:
    """Tests for numbering validation."""

    def test_valid_sequence(self):
        headings = ["1 Intro", "1.1 A", "1.2 B", "2 Methods"]
        issues = NumberingValidator.validate(headings)
        assert len(issues) == 0

    def test_gap_detected(self):
        """Gap from 1 to 3 at top level."""
        headings = ["1 Intro", "3 Results"]
        issues = NumberingValidator.validate(headings)
        gap_issues = [i for i in issues if i["type"] == "numbering_gap"]
        assert len(gap_issues) >= 1

    def test_duplicate_detected(self):
        headings = ["1 First", "1 Duplicate"]
        issues = NumberingValidator.validate(headings)
        dup_issues = [i for i in issues if i["type"] == "duplicate_number"]
        assert len(dup_issues) >= 1

    def test_unnumbered_no_issues(self):
        """Unnumbered headings should not generate numbering issues."""
        headings = ["Introduction", "Methods", "Results"]
        issues = NumberingValidator.validate(headings)
        assert len(issues) == 0

    def test_mixed_numbered_and_unnumbered(self):
        """Some numbered, some not - flags the unnumbered ones."""
        headings = ["1 Intro", "FOREWORD", "2 Methods"]
        issues = NumberingValidator.validate(headings)
        mixed_issues = [i for i in issues if i["type"] == "unnumbered_in_numbered_doc"]
        assert len(mixed_issues) >= 1
