"""Tests for letter-spacing collapse + the not-measurement-sensitive pattern."""

from __future__ import annotations

from chonk.cleaning.normalizer import collapse_letter_spacing, normalize_text
from chonk.qa.patterns import BOILERPLATE_PATTERNS, match_patterns


def test_collapse_two_words_with_wide_gap() -> None:
    assert collapse_letter_spacing("N O T   M E A S U R E M E N T") == "NOT MEASUREMENT"


def test_collapse_single_spaced_word() -> None:
    assert collapse_letter_spacing("S E N S I T I V E") == "SENSITIVE"


def test_collapse_leaves_normal_prose_untouched() -> None:
    s = "Depressurize the system before removing the filter."
    assert collapse_letter_spacing(s) == s


def test_short_run_not_treated_as_letter_spacing() -> None:
    # Two single chars is too short to be a spaced word.
    assert collapse_letter_spacing("a b") == "a b"


def test_mixed_line_collapses_gap_not_words() -> None:
    assert collapse_letter_spacing("w/Change   1") == "w/Change 1"


def test_normalize_text_runs_full_pipeline() -> None:
    assert normalize_text("N O T   M E A S U R E M E N T") == "NOT MEASUREMENT"


def test_not_measurement_sensitive_pattern_matches() -> None:
    assert match_patterns("NOT MEASUREMENT SENSITIVE", BOILERPLATE_PATTERNS) is not None


def test_normal_sentence_does_not_match_boilerplate() -> None:
    assert match_patterns("This is a normal requirement sentence.", BOILERPLATE_PATTERNS) is None
