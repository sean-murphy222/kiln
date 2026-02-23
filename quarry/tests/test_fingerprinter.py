"""Tests for Tier 1 document fingerprinting.

Uses synthetic PDFs created with PyMuPDF (fitz) — no real PDFs in the repo.
"""

from __future__ import annotations

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
