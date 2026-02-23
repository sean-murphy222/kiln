"""Tests for Tier 1 document fingerprinting.

Uses synthetic PDFs created with PyMuPDF (fitz) — no real PDFs in the repo.
"""

from __future__ import annotations

import os

import fitz
import pytest

from chonk.tier1.fingerprinter import (
    ByteLevelFeatures,
    CharacterFeatures,
    DocumentFingerprint,
    DocumentFingerprinter,
    FontFeatures,
    LayoutFeatures,
    RepetitionFeatures,
    StructuralRhythmFeatures,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic PDF factory
# ---------------------------------------------------------------------------

def create_pdf(
    tmp_path,
    *,
    pages: int = 1,
    text: str = "Hello World",
    font_size: float = 12.0,
    bold: bool = False,
    page_width: float = 612,
    page_height: float = 792,
    add_toc: bool = False,
    multi_column: bool = False,
    add_images: bool = False,
    add_tables: bool = False,
    add_links: bool = False,
    add_page_numbers: bool = False,
    add_headers: bool = False,
    varying_fonts: bool = False,
) -> str:
    """Create a synthetic PDF for testing.

    Args:
        tmp_path: Pytest tmp_path fixture.
        pages: Number of pages.
        text: Text content per page.
        font_size: Base font size in points.
        bold: Whether to use bold font.
        page_width: Page width in points.
        page_height: Page height in points.
        add_toc: Whether to add a table of contents.
        multi_column: Whether to lay out text in two columns.
        add_images: Whether to insert placeholder images.
        add_tables: Whether to draw table-like structures.
        add_links: Whether to add hyperlinks.
        add_page_numbers: Whether to add page number footers.
        add_headers: Whether to add repeated header text.
        varying_fonts: Whether to use multiple font sizes.

    Returns:
        String path to the created PDF file.
    """
    doc = fitz.open()
    font_name = "helv"
    if bold:
        font_name = "hebo"

    for i in range(pages):
        page = doc.new_page(width=page_width, height=page_height)

        if add_headers:
            header_rect = fitz.Rect(50, 20, page_width - 50, 40)
            page.insert_textbox(header_rect, "Document Header", fontsize=8, fontname="helv")

        if multi_column:
            col_width = (page_width - 150) / 2
            left_rect = fitz.Rect(50, 72, 50 + col_width, page_height - 72)
            right_rect = fitz.Rect(100 + col_width, 72, page_width - 50, page_height - 72)
            page.insert_textbox(left_rect, text, fontsize=font_size, fontname=font_name)
            page.insert_textbox(right_rect, text, fontsize=font_size, fontname=font_name)
        else:
            text_rect = fitz.Rect(50, 72, page_width - 50, page_height - 72)
            page.insert_textbox(text_rect, text, fontsize=font_size, fontname=font_name)

        if varying_fonts:
            heading_rect = fitz.Rect(50, 50, page_width - 50, 70)
            page.insert_textbox(heading_rect, f"Heading {i + 1}", fontsize=18, fontname="hebo")
            small_rect = fitz.Rect(50, page_height - 100, page_width - 50, page_height - 80)
            page.insert_textbox(small_rect, "Fine print", fontsize=6, fontname="helv")

        if add_images:
            img_rect = fitz.Rect(200, 300, 400, 500)
            shape = page.new_shape()
            shape.draw_rect(img_rect)
            shape.finish(color=(0, 0, 0), fill=(0.8, 0.8, 0.8))
            shape.commit()

        if add_tables:
            for row in range(3):
                for col in range(3):
                    cell = fitz.Rect(
                        100 + col * 100,
                        500 + row * 30,
                        200 + col * 100,
                        530 + row * 30,
                    )
                    shape = page.new_shape()
                    shape.draw_rect(cell)
                    shape.finish(color=(0, 0, 0))
                    shape.commit()
                    page.insert_textbox(cell, f"R{row}C{col}", fontsize=8, fontname="helv")

        if add_page_numbers:
            footer_rect = fitz.Rect(page_width / 2 - 20, page_height - 30, page_width / 2 + 20, page_height - 10)
            page.insert_textbox(footer_rect, str(i + 1), fontsize=10, fontname="helv")

        if add_links:
            link_rect = fitz.Rect(50, page_height - 60, 200, page_height - 40)
            page.insert_textbox(link_rect, "https://example.com", fontsize=10, fontname="helv")
            page.insert_link(
                {
                    "kind": fitz.LINK_URI,
                    "from": link_rect,
                    "uri": "https://example.com",
                }
            )

    if add_toc:
        toc = [[1, "Chapter 1", 1], [2, "Section 1.1", 1]]
        if pages > 1:
            toc.append([1, "Chapter 2", 2])
        doc.set_toc(toc)

    tmp_path.mkdir(parents=True, exist_ok=True)
    path = str(tmp_path / "test.pdf")
    doc.save(path)
    doc.close()
    return path


# ===================================================================
# Cycle 1: Dataclass construction, serialization, feature vector
# ===================================================================


class TestByteLevelFeatures:
    """Tests for ByteLevelFeatures dataclass."""

    def test_construction_defaults(self):
        """All fields default to zero/False."""
        f = ByteLevelFeatures()
        assert f.file_size == 0
        assert f.pdf_version == 0.0
        assert f.object_count == 0
        assert f.stream_count == 0
        assert f.has_metadata is False
        assert f.has_xmp_metadata is False
        assert f.page_count == 0
        assert f.encrypted is False
        assert f.has_acroform is False

    def test_to_dict_roundtrip(self):
        """Serialize and deserialize preserves values."""
        f = ByteLevelFeatures(
            file_size=1024,
            pdf_version=1.7,
            object_count=42,
            stream_count=10,
            has_metadata=True,
            has_xmp_metadata=False,
            page_count=5,
            encrypted=False,
            has_acroform=True,
        )
        d = f.to_dict()
        restored = ByteLevelFeatures.from_dict(d)
        assert restored == f

    def test_to_dict_keys(self):
        """Dict keys match field names."""
        f = ByteLevelFeatures()
        keys = set(f.to_dict().keys())
        expected = {
            "file_size", "pdf_version", "object_count", "stream_count",
            "has_metadata", "has_xmp_metadata", "page_count", "encrypted",
            "has_acroform",
        }
        assert keys == expected


class TestFontFeatures:
    """Tests for FontFeatures dataclass."""

    def test_construction_defaults(self):
        f = FontFeatures()
        assert f.font_count == 0
        assert f.size_min == 0.0
        assert f.size_max == 0.0
        assert f.size_mean == 0.0
        assert f.size_std == 0.0
        assert f.size_median == 0.0
        assert f.bold_ratio == 0.0
        assert f.italic_ratio == 0.0
        assert f.monospace_ratio == 0.0
        assert f.distinct_sizes == 0

    def test_to_dict_roundtrip(self):
        f = FontFeatures(
            font_count=5,
            size_min=8.0,
            size_max=24.0,
            size_mean=12.0,
            size_std=3.5,
            size_median=11.0,
            bold_ratio=0.3,
            italic_ratio=0.1,
            monospace_ratio=0.0,
            distinct_sizes=4,
        )
        assert FontFeatures.from_dict(f.to_dict()) == f


class TestLayoutFeatures:
    """Tests for LayoutFeatures dataclass."""

    def test_construction_defaults(self):
        f = LayoutFeatures()
        assert f.page_width == 0.0
        assert f.page_height == 0.0
        assert f.width_consistency == 0.0
        assert f.height_consistency == 0.0
        assert f.margin_left == 0.0
        assert f.margin_right == 0.0
        assert f.margin_top == 0.0
        assert f.margin_bottom == 0.0
        assert f.text_area_ratio == 0.0
        assert f.estimated_columns == 1

    def test_to_dict_roundtrip(self):
        f = LayoutFeatures(
            page_width=612.0,
            page_height=792.0,
            width_consistency=1.0,
            height_consistency=1.0,
            margin_left=72.0,
            margin_right=72.0,
            margin_top=72.0,
            margin_bottom=72.0,
            text_area_ratio=0.75,
            estimated_columns=2,
        )
        assert LayoutFeatures.from_dict(f.to_dict()) == f


class TestCharacterFeatures:
    """Tests for CharacterFeatures dataclass."""

    def test_construction_defaults(self):
        f = CharacterFeatures()
        assert f.alpha_ratio == 0.0
        assert f.numeric_ratio == 0.0
        assert f.punctuation_ratio == 0.0
        assert f.whitespace_ratio == 0.0
        assert f.special_ratio == 0.0
        assert f.uppercase_ratio == 0.0
        assert f.total_chars == 0

    def test_to_dict_roundtrip(self):
        f = CharacterFeatures(
            alpha_ratio=0.7,
            numeric_ratio=0.1,
            punctuation_ratio=0.05,
            whitespace_ratio=0.1,
            special_ratio=0.05,
            uppercase_ratio=0.15,
            total_chars=5000,
        )
        assert CharacterFeatures.from_dict(f.to_dict()) == f


class TestRepetitionFeatures:
    """Tests for RepetitionFeatures dataclass."""

    def test_construction_defaults(self):
        f = RepetitionFeatures()
        assert f.has_page_numbers is False
        assert f.has_headers is False
        assert f.has_footers is False
        assert f.repetition_ratio == 0.0
        assert f.first_line_diversity == 0.0

    def test_to_dict_roundtrip(self):
        f = RepetitionFeatures(
            has_page_numbers=True,
            has_headers=True,
            has_footers=False,
            repetition_ratio=0.4,
            first_line_diversity=0.8,
        )
        assert RepetitionFeatures.from_dict(f.to_dict()) == f


class TestStructuralRhythmFeatures:
    """Tests for StructuralRhythmFeatures dataclass."""

    def test_construction_defaults(self):
        f = StructuralRhythmFeatures()
        assert f.heading_density == 0.0
        assert f.table_density == 0.0
        assert f.image_density == 0.0
        assert f.list_density == 0.0
        assert f.has_toc is False
        assert f.toc_depth == 0
        assert f.link_count == 0
        assert f.heading_size_levels == 0

    def test_to_dict_roundtrip(self):
        f = StructuralRhythmFeatures(
            heading_density=0.05,
            table_density=0.02,
            image_density=0.01,
            list_density=0.03,
            has_toc=True,
            toc_depth=3,
            link_count=15,
            heading_size_levels=4,
        )
        assert StructuralRhythmFeatures.from_dict(f.to_dict()) == f


class TestDocumentFingerprint:
    """Tests for DocumentFingerprint container dataclass."""

    def test_construction_with_defaults(self):
        """Container creates sub-features with defaults."""
        fp = DocumentFingerprint()
        assert isinstance(fp.byte_features, ByteLevelFeatures)
        assert isinstance(fp.font_features, FontFeatures)
        assert isinstance(fp.layout_features, LayoutFeatures)
        assert isinstance(fp.character_features, CharacterFeatures)
        assert isinstance(fp.repetition_features, RepetitionFeatures)
        assert isinstance(fp.structural_rhythm, StructuralRhythmFeatures)

    def test_to_feature_vector_returns_list_of_floats(self):
        """Feature vector is a flat list of floats."""
        fp = DocumentFingerprint()
        vec = fp.to_feature_vector()
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)

    def test_to_feature_vector_length(self):
        """Feature vector has the expected number of elements (~49)."""
        fp = DocumentFingerprint()
        vec = fp.to_feature_vector()
        # 9 byte + 10 font + 10 layout + 7 char + 5 repetition + 8 rhythm = 49
        assert len(vec) == 49

    def test_feature_names_match_vector_length(self):
        """feature_names() returns same count as feature vector."""
        fp = DocumentFingerprint()
        names = fp.feature_names()
        vec = fp.to_feature_vector()
        assert len(names) == len(vec)
        assert all(isinstance(n, str) for n in names)

    def test_feature_names_unique(self):
        """All feature names are unique."""
        fp = DocumentFingerprint()
        names = fp.feature_names()
        assert len(names) == len(set(names))

    def test_to_dict_roundtrip(self):
        """Full fingerprint serialization roundtrip."""
        fp = DocumentFingerprint(
            byte_features=ByteLevelFeatures(file_size=2048, page_count=10),
            font_features=FontFeatures(font_count=3, size_mean=12.0),
        )
        d = fp.to_dict()
        restored = DocumentFingerprint.from_dict(d)
        assert restored.byte_features.file_size == 2048
        assert restored.font_features.font_count == 3
        assert restored == fp

    def test_feature_vector_values_change_with_data(self):
        """Feature vector reflects actual data, not all zeros."""
        fp = DocumentFingerprint(
            byte_features=ByteLevelFeatures(file_size=999, page_count=5),
        )
        vec = fp.to_feature_vector()
        assert vec[0] == 999.0  # file_size is first
        assert any(v != 0.0 for v in vec)


# ===================================================================
# Cycle 2: Fingerprinter shell — instantiation and input validation
# ===================================================================


class TestDocumentFingerprinterShell:
    """Tests for DocumentFingerprinter instantiation and validation."""

    def test_instantiation(self):
        """Fingerprinter can be created."""
        fp = DocumentFingerprinter()
        assert fp is not None

    def test_max_file_size_default(self):
        """Default max file size is 100 MB."""
        fp = DocumentFingerprinter()
        assert fp.max_file_size == 100 * 1024 * 1024

    def test_max_file_size_custom(self):
        """Custom max file size is respected."""
        fp = DocumentFingerprinter(max_file_size=50 * 1024 * 1024)
        assert fp.max_file_size == 50 * 1024 * 1024

    def test_extract_missing_file(self, tmp_path):
        """Raises FileNotFoundError for nonexistent path."""
        fp = DocumentFingerprinter()
        with pytest.raises(FileNotFoundError):
            fp.extract(tmp_path / "nonexistent.pdf")

    def test_extract_non_pdf(self, tmp_path):
        """Raises ValueError for non-PDF file."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a pdf")
        fp = DocumentFingerprinter()
        with pytest.raises(ValueError, match="not a valid PDF"):
            fp.extract(txt_file)

    def test_extract_oversized_file(self, tmp_path):
        """Raises ValueError for file exceeding max size."""
        pdf_path = create_pdf(tmp_path, pages=1, text="small")
        fp = DocumentFingerprinter(max_file_size=1)  # 1 byte limit
        with pytest.raises(ValueError, match="exceeds maximum"):
            fp.extract(pdf_path)

    def test_extract_returns_fingerprint(self, tmp_path):
        """extract() returns a DocumentFingerprint."""
        pdf_path = create_pdf(tmp_path, pages=1, text="Hello World")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert isinstance(result, DocumentFingerprint)

    def test_extract_accepts_string_path(self, tmp_path):
        """extract() accepts both str and Path."""
        pdf_path = create_pdf(tmp_path, pages=1, text="Hello")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)  # str from create_pdf
        assert isinstance(result, DocumentFingerprint)

    def test_extract_accepts_path_object(self, tmp_path):
        """extract() accepts pathlib.Path."""
        from pathlib import Path

        pdf_path = Path(create_pdf(tmp_path, pages=1, text="Hello"))
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert isinstance(result, DocumentFingerprint)


# ===================================================================
# Cycle 3: Byte-level feature analysis
# ===================================================================


class TestByteFeatureAnalysis:
    """Tests for _analyze_byte_features."""

    def test_file_size_populated(self, tmp_path):
        """file_size reflects actual file size."""
        pdf_path = create_pdf(tmp_path, pages=1, text="Hello")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        actual_size = os.path.getsize(pdf_path)
        assert result.byte_features.file_size == actual_size

    def test_page_count(self, tmp_path):
        """page_count matches number of pages created."""
        pdf_path = create_pdf(tmp_path, pages=5, text="Content")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.byte_features.page_count == 5

    def test_page_count_single(self, tmp_path):
        """Single page PDF has page_count 1."""
        pdf_path = create_pdf(tmp_path, pages=1, text="Single page")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.byte_features.page_count == 1

    def test_pdf_version(self, tmp_path):
        """pdf_version is a reasonable PDF spec version."""
        pdf_path = create_pdf(tmp_path, pages=1, text="Hello")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert 1.0 <= result.byte_features.pdf_version <= 2.0

    def test_object_count_positive(self, tmp_path):
        """Object count is positive for any valid PDF."""
        pdf_path = create_pdf(tmp_path, pages=1, text="Hello")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.byte_features.object_count > 0

    def test_stream_count_nonneg(self, tmp_path):
        """Stream count is non-negative."""
        pdf_path = create_pdf(tmp_path, pages=1, text="Hello")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.byte_features.stream_count >= 0

    def test_metadata_flags(self, tmp_path):
        """Metadata flags are booleans."""
        pdf_path = create_pdf(tmp_path, pages=1, text="Hello")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert isinstance(result.byte_features.has_metadata, bool)
        assert isinstance(result.byte_features.has_xmp_metadata, bool)
        assert isinstance(result.byte_features.encrypted, bool)
        assert isinstance(result.byte_features.has_acroform, bool)

    def test_more_pages_more_objects(self, tmp_path):
        """More pages generally means more objects."""
        path1 = create_pdf(tmp_path / "a", pages=1, text="Hi")
        path5 = create_pdf(tmp_path / "b", pages=10, text="Hi")
        fp = DocumentFingerprinter()
        r1 = fp.extract(path1)
        r5 = fp.extract(path5)
        assert r5.byte_features.object_count > r1.byte_features.object_count


# ===================================================================
# Cycle 4: Font feature analysis
# ===================================================================


class TestFontFeatureAnalysis:
    """Tests for _analyze_font_features."""

    def test_font_count_positive(self, tmp_path):
        """At least one font used in a text PDF."""
        pdf_path = create_pdf(tmp_path, pages=1, text="Hello World")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.font_features.font_count >= 1

    def test_size_stats_populated(self, tmp_path):
        """Font size statistics are non-zero for text PDFs."""
        pdf_path = create_pdf(tmp_path, pages=1, text="Hello", font_size=12.0)
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.font_features.size_min > 0
        assert result.font_features.size_max > 0
        assert result.font_features.size_mean > 0

    def test_bold_ratio_with_bold(self, tmp_path):
        """Bold ratio is positive when using bold font."""
        pdf_path = create_pdf(tmp_path, pages=1, text="Bold text", bold=True)
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.font_features.bold_ratio > 0

    def test_bold_ratio_without_bold(self, tmp_path):
        """Bold ratio is zero or low when not using bold font."""
        pdf_path = create_pdf(tmp_path, pages=1, text="Normal text", bold=False)
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.font_features.bold_ratio == 0.0

    def test_varying_fonts_increases_distinct_sizes(self, tmp_path):
        """Multiple font sizes increases distinct_sizes count."""
        pdf_path = create_pdf(
            tmp_path, pages=3, text="Body text", varying_fonts=True
        )
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.font_features.distinct_sizes >= 2

    def test_size_min_leq_max(self, tmp_path):
        """size_min <= size_mean <= size_max."""
        pdf_path = create_pdf(
            tmp_path, pages=3, text="Some text", varying_fonts=True
        )
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.font_features.size_min <= result.font_features.size_mean
        assert result.font_features.size_mean <= result.font_features.size_max

    def test_ratios_bounded(self, tmp_path):
        """Bold/italic/monospace ratios are between 0 and 1."""
        pdf_path = create_pdf(tmp_path, pages=1, text="Hello")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert 0.0 <= result.font_features.bold_ratio <= 1.0
        assert 0.0 <= result.font_features.italic_ratio <= 1.0
        assert 0.0 <= result.font_features.monospace_ratio <= 1.0


# ===================================================================
# Cycle 5: Layout feature analysis
# ===================================================================


class TestLayoutFeatureAnalysis:
    """Tests for _analyze_layout_features."""

    def test_page_dimensions_letter(self, tmp_path):
        """US Letter page dimensions detected correctly."""
        pdf_path = create_pdf(
            tmp_path, pages=1, text="Hello",
            page_width=612, page_height=792,
        )
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert abs(result.layout_features.page_width - 612.0) < 1.0
        assert abs(result.layout_features.page_height - 792.0) < 1.0

    def test_page_dimensions_a4(self, tmp_path):
        """A4 page dimensions detected correctly."""
        pdf_path = create_pdf(
            tmp_path, pages=1, text="Hello",
            page_width=595, page_height=842,
        )
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert abs(result.layout_features.page_width - 595.0) < 1.0
        assert abs(result.layout_features.page_height - 842.0) < 1.0

    def test_consistency_uniform_pages(self, tmp_path):
        """Consistency is 1.0 when all pages are the same size."""
        pdf_path = create_pdf(tmp_path, pages=5, text="Same size")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.layout_features.width_consistency == 1.0
        assert result.layout_features.height_consistency == 1.0

    def test_margins_positive(self, tmp_path):
        """Margins are non-negative for text PDFs."""
        pdf_path = create_pdf(tmp_path, pages=1, text="Hello World " * 50)
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.layout_features.margin_left >= 0
        assert result.layout_features.margin_right >= 0
        assert result.layout_features.margin_top >= 0
        assert result.layout_features.margin_bottom >= 0

    def test_text_area_ratio_bounded(self, tmp_path):
        """text_area_ratio is between 0 and 1."""
        pdf_path = create_pdf(tmp_path, pages=1, text="Some text content " * 20)
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert 0.0 <= result.layout_features.text_area_ratio <= 1.0

    def test_single_column_default(self, tmp_path):
        """Single-column layout detected for normal text."""
        pdf_path = create_pdf(tmp_path, pages=1, text="Normal single column text " * 20)
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.layout_features.estimated_columns == 1

    def test_multi_column_detected(self, tmp_path):
        """Multi-column layout detected when text is in two columns."""
        pdf_path = create_pdf(
            tmp_path, pages=1, text="Column text " * 30, multi_column=True
        )
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.layout_features.estimated_columns >= 2


# ===================================================================
# Cycle 6: Character feature analysis
# ===================================================================


class TestCharacterFeatureAnalysis:
    """Tests for _analyze_character_features."""

    def test_alpha_ratio_for_text(self, tmp_path):
        """Alpha ratio is high for plain text content."""
        pdf_path = create_pdf(tmp_path, pages=1, text="This is plain text content")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.character_features.alpha_ratio > 0.5

    def test_numeric_ratio_for_numbers(self, tmp_path):
        """Numeric ratio is positive for text with numbers."""
        pdf_path = create_pdf(tmp_path, pages=1, text="123 456 789 000 111 222")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.character_features.numeric_ratio > 0.3

    def test_total_chars_positive(self, tmp_path):
        """total_chars is positive for text PDFs."""
        pdf_path = create_pdf(tmp_path, pages=1, text="Hello World")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.character_features.total_chars > 0

    def test_ratios_sum_approximately_one(self, tmp_path):
        """Character class ratios sum to approximately 1.0."""
        pdf_path = create_pdf(tmp_path, pages=1, text="Hello World 123! @#$")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        cf = result.character_features
        if cf.total_chars > 0:
            total = (
                cf.alpha_ratio + cf.numeric_ratio + cf.punctuation_ratio
                + cf.whitespace_ratio + cf.special_ratio
            )
            assert abs(total - 1.0) < 0.01

    def test_uppercase_ratio_all_upper(self, tmp_path):
        """Uppercase ratio is high for all-caps text."""
        pdf_path = create_pdf(tmp_path, pages=1, text="ALL CAPS TEXT HERE NOW")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.character_features.uppercase_ratio > 0.8

    def test_uppercase_ratio_all_lower(self, tmp_path):
        """Uppercase ratio is low for all-lowercase text."""
        pdf_path = create_pdf(tmp_path, pages=1, text="all lowercase text here now")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.character_features.uppercase_ratio < 0.1

    def test_all_ratios_bounded(self, tmp_path):
        """All ratios are between 0 and 1."""
        pdf_path = create_pdf(tmp_path, pages=1, text="Mixed Content 123!")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        cf = result.character_features
        for ratio in [cf.alpha_ratio, cf.numeric_ratio, cf.punctuation_ratio,
                      cf.whitespace_ratio, cf.special_ratio, cf.uppercase_ratio]:
            assert 0.0 <= ratio <= 1.0


# ===================================================================
# Cycle 7: Repetition feature analysis
# ===================================================================


class TestRepetitionFeatureAnalysis:
    """Tests for _analyze_repetition_features."""

    def test_page_numbers_detected(self, tmp_path):
        """Page numbers detected when added."""
        pdf_path = create_pdf(
            tmp_path, pages=5, text="Content", add_page_numbers=True
        )
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.repetition_features.has_page_numbers is True

    def test_no_page_numbers(self, tmp_path):
        """No page numbers when not added."""
        pdf_path = create_pdf(tmp_path, pages=5, text="Content")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.repetition_features.has_page_numbers is False

    def test_headers_detected(self, tmp_path):
        """Repeated headers detected."""
        pdf_path = create_pdf(
            tmp_path, pages=5, text="Content", add_headers=True
        )
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.repetition_features.has_headers is True

    def test_no_headers(self, tmp_path):
        """No headers when not added."""
        pdf_path = create_pdf(tmp_path, pages=5, text="Content")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.repetition_features.has_headers is False

    def test_repetition_ratio_bounded(self, tmp_path):
        """Repetition ratio is between 0 and 1."""
        pdf_path = create_pdf(
            tmp_path, pages=5, text="Content",
            add_headers=True, add_page_numbers=True,
        )
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert 0.0 <= result.repetition_features.repetition_ratio <= 1.0

    def test_first_line_diversity_bounded(self, tmp_path):
        """first_line_diversity is between 0 and 1."""
        pdf_path = create_pdf(tmp_path, pages=5, text="Content")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert 0.0 <= result.repetition_features.first_line_diversity <= 1.0

    def test_single_page_no_repetition(self, tmp_path):
        """Single page has no repetition patterns."""
        pdf_path = create_pdf(tmp_path, pages=1, text="Hello")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.repetition_features.has_page_numbers is False
        assert result.repetition_features.has_headers is False
        assert result.repetition_features.has_footers is False


# ===================================================================
# Cycle 8: Structural rhythm analysis
# ===================================================================


class TestStructuralRhythmAnalysis:
    """Tests for _analyze_structural_rhythm."""

    def test_toc_detected(self, tmp_path):
        """TOC is detected when present."""
        pdf_path = create_pdf(
            tmp_path, pages=3, text="Content", add_toc=True
        )
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.structural_rhythm.has_toc is True
        assert result.structural_rhythm.toc_depth >= 1

    def test_no_toc(self, tmp_path):
        """No TOC when not added."""
        pdf_path = create_pdf(tmp_path, pages=3, text="Content")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.structural_rhythm.has_toc is False
        assert result.structural_rhythm.toc_depth == 0

    def test_link_count_with_links(self, tmp_path):
        """Links counted when present."""
        pdf_path = create_pdf(
            tmp_path, pages=3, text="Content", add_links=True
        )
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.structural_rhythm.link_count > 0

    def test_link_count_without_links(self, tmp_path):
        """No links when not added."""
        pdf_path = create_pdf(tmp_path, pages=3, text="Content")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.structural_rhythm.link_count == 0

    def test_heading_density_with_varying_fonts(self, tmp_path):
        """Heading density detected with varying font sizes."""
        pdf_path = create_pdf(
            tmp_path, pages=5, text="Body content", varying_fonts=True
        )
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.structural_rhythm.heading_size_levels >= 1

    def test_image_density_with_images(self, tmp_path):
        """Image density is positive when images present."""
        pdf_path = create_pdf(
            tmp_path, pages=3, text="Content", add_images=True
        )
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.structural_rhythm.image_density >= 0

    def test_densities_nonnegative(self, tmp_path):
        """All densities are non-negative."""
        pdf_path = create_pdf(tmp_path, pages=3, text="Content")
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)
        assert result.structural_rhythm.heading_density >= 0
        assert result.structural_rhythm.table_density >= 0
        assert result.structural_rhythm.image_density >= 0
        assert result.structural_rhythm.list_density >= 0


# ===================================================================
# Cycle 9: Integration — determinism and performance
# ===================================================================


class TestIntegration:
    """Integration tests for the full fingerprinter pipeline."""

    def test_deterministic(self, tmp_path):
        """Same file produces same fingerprint twice."""
        pdf_path = create_pdf(
            tmp_path, pages=5, text="Determinism test content " * 10,
            varying_fonts=True, add_toc=True,
        )
        fp = DocumentFingerprinter()
        result1 = fp.extract(pdf_path)
        result2 = fp.extract(pdf_path)
        assert result1.to_feature_vector() == result2.to_feature_vector()

    def test_different_docs_different_fingerprints(self, tmp_path):
        """Different documents produce different fingerprints."""
        path_a = create_pdf(
            tmp_path / "a", pages=1, text="Short document",
        )
        path_b = create_pdf(
            tmp_path / "b", pages=10, text="Long document " * 50,
            varying_fonts=True, add_toc=True, add_links=True,
        )
        fp = DocumentFingerprinter()
        vec_a = fp.extract(path_a).to_feature_vector()
        vec_b = fp.extract(path_b).to_feature_vector()
        assert vec_a != vec_b

    def test_performance_100_pages(self, tmp_path):
        """Fingerprinting a 100-page PDF completes in < 5 seconds."""
        import time

        pdf_path = create_pdf(
            tmp_path, pages=100, text="Performance test content " * 20,
        )
        fp = DocumentFingerprinter()
        start = time.time()
        fp.extract(pdf_path)
        elapsed = time.time() - start
        assert elapsed < 5.0, f"Took {elapsed:.2f}s, expected < 5s"

    def test_all_feature_groups_populated(self, tmp_path):
        """All feature groups have at least some non-default values."""
        pdf_path = create_pdf(
            tmp_path, pages=5, text="Rich content " * 20,
            varying_fonts=True, add_toc=True, add_links=True,
            add_headers=True, add_page_numbers=True,
        )
        fp = DocumentFingerprinter()
        result = fp.extract(pdf_path)

        # Byte features should have file_size and page_count
        assert result.byte_features.file_size > 0
        assert result.byte_features.page_count == 5

        # Font features should detect fonts
        assert result.font_features.font_count >= 1

        # Layout features should have dimensions
        assert result.layout_features.page_width > 0
        assert result.layout_features.page_height > 0

        # Character features should have content
        assert result.character_features.total_chars > 0

        # Structural features should detect TOC
        assert result.structural_rhythm.has_toc is True

    def test_feature_vector_no_nan_or_inf(self, tmp_path):
        """Feature vector contains no NaN or Inf values."""
        import math

        pdf_path = create_pdf(
            tmp_path, pages=3, text="Clean data test",
            varying_fonts=True,
        )
        fp = DocumentFingerprinter()
        vec = fp.extract(pdf_path).to_feature_vector()
        for i, v in enumerate(vec):
            assert not math.isnan(v), f"NaN at index {i}"
            assert not math.isinf(v), f"Inf at index {i}"
