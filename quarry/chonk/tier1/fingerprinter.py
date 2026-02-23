"""Tier 1 structural document fingerprinting.

Analyzes raw PDF structure statistically — no content parsing — to produce
a feature vector for ML document-type classification.

Data flow::

    PDF File → DocumentFingerprinter.extract(path) → DocumentFingerprint
        → .to_feature_vector() → list[float]  (for T-002 ML classifier)
"""

from __future__ import annotations

import os
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz
import numpy as np

# ---------------------------------------------------------------------------
# Sub-feature dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ByteLevelFeatures:
    """Low-level byte/object statistics from PDF structure.

    Attributes:
        file_size: File size in bytes.
        pdf_version: PDF spec version (e.g. 1.7).
        object_count: Number of indirect PDF objects.
        stream_count: Number of stream objects.
        has_metadata: Whether the document has an Info dict.
        has_xmp_metadata: Whether the document has XMP metadata.
        page_count: Total number of pages.
        encrypted: Whether the PDF is encrypted.
        has_acroform: Whether the PDF contains form fields.
    """

    file_size: int = 0
    pdf_version: float = 0.0
    object_count: int = 0
    stream_count: int = 0
    has_metadata: bool = False
    has_xmp_metadata: bool = False
    page_count: int = 0
    encrypted: bool = False
    has_acroform: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "file_size": self.file_size,
            "pdf_version": self.pdf_version,
            "object_count": self.object_count,
            "stream_count": self.stream_count,
            "has_metadata": self.has_metadata,
            "has_xmp_metadata": self.has_xmp_metadata,
            "page_count": self.page_count,
            "encrypted": self.encrypted,
            "has_acroform": self.has_acroform,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ByteLevelFeatures:
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class FontFeatures:
    """Font usage statistics across sampled pages.

    Attributes:
        font_count: Total distinct fonts used.
        size_min: Smallest font size observed.
        size_max: Largest font size observed.
        size_mean: Mean font size weighted by usage.
        size_std: Standard deviation of font sizes.
        size_median: Median font size.
        bold_ratio: Fraction of text spans using bold fonts.
        italic_ratio: Fraction of text spans using italic fonts.
        monospace_ratio: Fraction of text spans using monospace fonts.
        distinct_sizes: Number of unique font sizes.
    """

    font_count: int = 0
    size_min: float = 0.0
    size_max: float = 0.0
    size_mean: float = 0.0
    size_std: float = 0.0
    size_median: float = 0.0
    bold_ratio: float = 0.0
    italic_ratio: float = 0.0
    monospace_ratio: float = 0.0
    distinct_sizes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "font_count": self.font_count,
            "size_min": self.size_min,
            "size_max": self.size_max,
            "size_mean": self.size_mean,
            "size_std": self.size_std,
            "size_median": self.size_median,
            "bold_ratio": self.bold_ratio,
            "italic_ratio": self.italic_ratio,
            "monospace_ratio": self.monospace_ratio,
            "distinct_sizes": self.distinct_sizes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FontFeatures:
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class LayoutFeatures:
    """Page layout and geometry statistics.

    Attributes:
        page_width: Median page width in points.
        page_height: Median page height in points.
        width_consistency: 1.0 if all pages same width, lower otherwise.
        height_consistency: 1.0 if all pages same height, lower otherwise.
        margin_left: Estimated left margin in points.
        margin_right: Estimated right margin in points.
        margin_top: Estimated top margin in points.
        margin_bottom: Estimated bottom margin in points.
        text_area_ratio: Fraction of page area covered by text.
        estimated_columns: Estimated number of text columns (1, 2, or 3).
    """

    page_width: float = 0.0
    page_height: float = 0.0
    width_consistency: float = 0.0
    height_consistency: float = 0.0
    margin_left: float = 0.0
    margin_right: float = 0.0
    margin_top: float = 0.0
    margin_bottom: float = 0.0
    text_area_ratio: float = 0.0
    estimated_columns: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "page_width": self.page_width,
            "page_height": self.page_height,
            "width_consistency": self.width_consistency,
            "height_consistency": self.height_consistency,
            "margin_left": self.margin_left,
            "margin_right": self.margin_right,
            "margin_top": self.margin_top,
            "margin_bottom": self.margin_bottom,
            "text_area_ratio": self.text_area_ratio,
            "estimated_columns": self.estimated_columns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LayoutFeatures:
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class CharacterFeatures:
    """Character class distribution across sampled text.

    Attributes:
        alpha_ratio: Fraction of alphabetic characters.
        numeric_ratio: Fraction of numeric characters.
        punctuation_ratio: Fraction of punctuation characters.
        whitespace_ratio: Fraction of whitespace characters.
        special_ratio: Fraction of other/special characters.
        uppercase_ratio: Fraction of alpha chars that are uppercase.
        total_chars: Total character count sampled.
    """

    alpha_ratio: float = 0.0
    numeric_ratio: float = 0.0
    punctuation_ratio: float = 0.0
    whitespace_ratio: float = 0.0
    special_ratio: float = 0.0
    uppercase_ratio: float = 0.0
    total_chars: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "alpha_ratio": self.alpha_ratio,
            "numeric_ratio": self.numeric_ratio,
            "punctuation_ratio": self.punctuation_ratio,
            "whitespace_ratio": self.whitespace_ratio,
            "special_ratio": self.special_ratio,
            "uppercase_ratio": self.uppercase_ratio,
            "total_chars": self.total_chars,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CharacterFeatures:
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class RepetitionFeatures:
    """Cross-page repetition patterns (headers, footers, page numbers).

    Attributes:
        has_page_numbers: Whether sequential page numbers were detected.
        has_headers: Whether repeated header text was found.
        has_footers: Whether repeated footer text was found.
        repetition_ratio: Fraction of pages with repeated elements.
        first_line_diversity: Ratio of unique first lines to total pages.
    """

    has_page_numbers: bool = False
    has_headers: bool = False
    has_footers: bool = False
    repetition_ratio: float = 0.0
    first_line_diversity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "has_page_numbers": self.has_page_numbers,
            "has_headers": self.has_headers,
            "has_footers": self.has_footers,
            "repetition_ratio": self.repetition_ratio,
            "first_line_diversity": self.first_line_diversity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepetitionFeatures:
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class StructuralRhythmFeatures:
    """Document element density and structural patterns.

    Attributes:
        heading_density: Headings per page (estimated).
        table_density: Tables per page (estimated).
        image_density: Images per page (estimated).
        list_density: Lists per page (estimated).
        has_toc: Whether a table of contents was detected.
        toc_depth: Maximum TOC nesting depth (0 if no TOC).
        link_count: Total hyperlinks in the document.
        heading_size_levels: Number of distinct heading font sizes.
    """

    heading_density: float = 0.0
    table_density: float = 0.0
    image_density: float = 0.0
    list_density: float = 0.0
    has_toc: bool = False
    toc_depth: int = 0
    link_count: int = 0
    heading_size_levels: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "heading_density": self.heading_density,
            "table_density": self.table_density,
            "image_density": self.image_density,
            "list_density": self.list_density,
            "has_toc": self.has_toc,
            "toc_depth": self.toc_depth,
            "link_count": self.link_count,
            "heading_size_levels": self.heading_size_levels,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuralRhythmFeatures:
        """Deserialize from dictionary."""
        return cls(**data)


# ---------------------------------------------------------------------------
# Container dataclass
# ---------------------------------------------------------------------------


@dataclass
class DocumentFingerprint:
    """Complete structural fingerprint of a PDF document.

    Contains all sub-feature groups and provides methods to convert
    to a flat feature vector for ML classification.

    Example::

        fp = DocumentFingerprint(
            byte_features=ByteLevelFeatures(file_size=2048, page_count=10),
        )
        vector = fp.to_feature_vector()  # list of ~49 floats
        names = fp.feature_names()       # corresponding feature names
    """

    byte_features: ByteLevelFeatures = field(default_factory=ByteLevelFeatures)
    font_features: FontFeatures = field(default_factory=FontFeatures)
    layout_features: LayoutFeatures = field(default_factory=LayoutFeatures)
    character_features: CharacterFeatures = field(default_factory=CharacterFeatures)
    repetition_features: RepetitionFeatures = field(default_factory=RepetitionFeatures)
    structural_rhythm: StructuralRhythmFeatures = field(default_factory=StructuralRhythmFeatures)

    def to_feature_vector(self) -> list[float]:
        """Convert fingerprint to a flat list of floats for ML input.

        Boolean fields are converted to 1.0/0.0. Integer fields are
        cast to float. The order matches feature_names().

        Returns:
            List of ~49 float values.
        """
        vector: list[float] = []
        for group in self._feature_groups():
            for key, value in group.to_dict().items():
                if isinstance(value, bool):
                    vector.append(1.0 if value else 0.0)
                else:
                    vector.append(float(value))
        return vector

    def feature_names(self) -> list[str]:
        """Return ordered list of feature names matching to_feature_vector().

        Each name is prefixed with its group for uniqueness
        (e.g. 'byte_file_size', 'font_size_mean').

        Returns:
            List of string feature names.
        """
        prefixes = [
            "byte",
            "font",
            "layout",
            "char",
            "rep",
            "rhythm",
        ]
        names: list[str] = []
        for prefix, group in zip(prefixes, self._feature_groups()):
            for key in group.to_dict():
                names.append(f"{prefix}_{key}")
        return names

    def to_dict(self) -> dict[str, Any]:
        """Serialize to nested dictionary."""
        return {
            "byte_features": self.byte_features.to_dict(),
            "font_features": self.font_features.to_dict(),
            "layout_features": self.layout_features.to_dict(),
            "character_features": self.character_features.to_dict(),
            "repetition_features": self.repetition_features.to_dict(),
            "structural_rhythm": self.structural_rhythm.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentFingerprint:
        """Deserialize from nested dictionary."""
        return cls(
            byte_features=ByteLevelFeatures.from_dict(data["byte_features"]),
            font_features=FontFeatures.from_dict(data["font_features"]),
            layout_features=LayoutFeatures.from_dict(data["layout_features"]),
            character_features=CharacterFeatures.from_dict(data["character_features"]),
            repetition_features=RepetitionFeatures.from_dict(data["repetition_features"]),
            structural_rhythm=StructuralRhythmFeatures.from_dict(data["structural_rhythm"]),
        )

    def _feature_groups(self) -> list[Any]:
        """Return ordered list of sub-feature dataclasses."""
        return [
            self.byte_features,
            self.font_features,
            self.layout_features,
            self.character_features,
            self.repetition_features,
            self.structural_rhythm,
        ]


# ---------------------------------------------------------------------------
# Fingerprinter class
# ---------------------------------------------------------------------------

_DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
_MAX_SAMPLE_PAGES = 50


class DocumentFingerprinter:
    """Extracts structural fingerprints from PDF documents.

    Analyzes raw PDF structure statistically — no content parsing beyond
    character-level stats — to produce a feature vector suitable for ML
    document-type classification.

    Args:
        max_file_size: Maximum allowed file size in bytes.

    Example::

        fp = DocumentFingerprinter()
        fingerprint = fp.extract("document.pdf")
        vector = fingerprint.to_feature_vector()
    """

    def __init__(self, max_file_size: int = _DEFAULT_MAX_FILE_SIZE) -> None:
        self.max_file_size = max_file_size

    def extract(self, path: str | Path) -> DocumentFingerprint:
        """Extract a structural fingerprint from a PDF file.

        Args:
            path: Path to the PDF file (str or pathlib.Path).

        Returns:
            DocumentFingerprint with all sub-features populated.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a valid PDF or exceeds size limit.
        """
        path = Path(path)
        self._validate_path(path)

        doc = fitz.open(str(path))
        try:
            sampled = self._sample_pages(doc)
            fingerprint = DocumentFingerprint(
                byte_features=self._analyze_byte_features(doc, path),
                font_features=self._analyze_font_features(doc, sampled),
                layout_features=self._analyze_layout_features(doc, sampled),
                character_features=self._analyze_character_features(doc, sampled),
                repetition_features=self._analyze_repetition_features(doc, sampled),
                structural_rhythm=self._analyze_structural_rhythm(doc, sampled),
            )
        finally:
            doc.close()

        return fingerprint

    def _validate_path(self, path: Path) -> None:
        """Validate that path points to a readable PDF within size limits.

        Args:
            path: Path to validate.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If not a PDF or exceeds max size.
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_size = os.path.getsize(path)
        if file_size > self.max_file_size:
            raise ValueError(
                f"File size ({file_size} bytes) exceeds maximum " f"({self.max_file_size} bytes)"
            )

        # Verify it's actually a PDF by checking magic bytes
        with open(path, "rb") as f:
            header = f.read(5)
        if header != b"%PDF-":
            raise ValueError(f"{path.name} is not a valid PDF file")

    def _sample_pages(self, doc: fitz.Document) -> list[int]:
        """Select pages to sample for analysis.

        Strategy: first 3, last 2, and evenly-spaced middle pages,
        up to _MAX_SAMPLE_PAGES total.

        Args:
            doc: Open fitz.Document.

        Returns:
            Sorted list of 0-based page indices.
        """
        n = len(doc)
        if n <= _MAX_SAMPLE_PAGES:
            return list(range(n))

        indices: set[int] = set()

        # First 3
        for i in range(min(3, n)):
            indices.add(i)

        # Last 2
        for i in range(max(0, n - 2), n):
            indices.add(i)

        # Evenly spaced middle
        remaining = _MAX_SAMPLE_PAGES - len(indices)
        if remaining > 0 and n > 5:
            step = max(1, (n - 5) // remaining)
            for i in range(3, n - 2, step):
                indices.add(i)
                if len(indices) >= _MAX_SAMPLE_PAGES:
                    break

        return sorted(indices)

    # ------------------------------------------------------------------
    # Byte-level analysis (cycle 3)
    # ------------------------------------------------------------------

    def _analyze_byte_features(self, doc: fitz.Document, path: Path) -> ByteLevelFeatures:
        """Extract low-level byte and object statistics from PDF.

        Args:
            doc: Open fitz.Document.
            path: Path to the PDF file on disk.

        Returns:
            Populated ByteLevelFeatures.
        """
        file_size = os.path.getsize(path)

        # PDF version from metadata string (e.g. "PDF-1.7")
        pdf_version = 0.0
        fmt = doc.metadata.get("format", "") if doc.metadata else ""
        version_match = re.search(r"(\d+\.\d+)", fmt)
        if version_match:
            pdf_version = float(version_match.group(1))

        # Count objects and streams via xref table
        xref_len = doc.xref_length()
        object_count = max(0, xref_len - 1)  # xref 0 is free
        stream_count = 0
        for i in range(1, xref_len):
            if doc.xref_is_stream(i):
                stream_count += 1

        # Metadata flags
        meta = doc.metadata or {}
        has_metadata = bool(meta.get("author") or meta.get("title"))
        has_xmp_metadata = bool(doc.xref_xml_metadata())
        encrypted = doc.is_encrypted
        has_acroform = self._check_acroform(doc)

        return ByteLevelFeatures(
            file_size=file_size,
            pdf_version=pdf_version,
            object_count=object_count,
            stream_count=stream_count,
            has_metadata=has_metadata,
            has_xmp_metadata=has_xmp_metadata,
            page_count=len(doc),
            encrypted=encrypted,
            has_acroform=has_acroform,
        )

    @staticmethod
    def _check_acroform(doc: fitz.Document) -> bool:
        """Check if PDF contains AcroForm (interactive form fields).

        Args:
            doc: Open fitz.Document.

        Returns:
            True if AcroForm dictionary found in catalog.
        """
        try:
            cat = doc.pdf_catalog()
            if cat > 0:
                xref_str = doc.xref_object(cat)
                return "/AcroForm" in xref_str
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # Font analysis (cycle 4)
    # ------------------------------------------------------------------

    def _analyze_font_features(self, doc: fitz.Document, sampled: list[int]) -> FontFeatures:
        """Analyze font usage across sampled pages.

        Args:
            doc: Open fitz.Document.
            sampled: List of page indices to analyze.

        Returns:
            Populated FontFeatures.
        """
        all_sizes: list[float] = []
        font_names: set[str] = set()
        bold_spans = 0
        italic_spans = 0
        mono_spans = 0
        total_spans = 0

        for page_idx in sampled:
            page = doc[page_idx]
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            for block in blocks.get("blocks", []):
                if block.get("type") != 0:  # text blocks only
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = span.get("size", 0.0)
                        font = span.get("font", "")
                        flags = span.get("flags", 0)
                        text = span.get("text", "")

                        if not text.strip():
                            continue

                        total_spans += 1
                        all_sizes.append(size)
                        font_names.add(font)

                        # flags: bit 0 = superscript, bit 1 = italic,
                        #         bit 4 = bold, bit 3 = monospaced
                        if flags & (1 << 4):
                            bold_spans += 1
                        if flags & (1 << 1):
                            italic_spans += 1
                        if flags & (1 << 3):
                            mono_spans += 1

        if not all_sizes:
            return FontFeatures()

        sizes_arr = np.array(all_sizes)
        distinct = sorted(set(round(s, 1) for s in all_sizes))

        return FontFeatures(
            font_count=len(font_names),
            size_min=float(np.min(sizes_arr)),
            size_max=float(np.max(sizes_arr)),
            size_mean=float(np.mean(sizes_arr)),
            size_std=float(np.std(sizes_arr)),
            size_median=float(np.median(sizes_arr)),
            bold_ratio=bold_spans / total_spans if total_spans else 0.0,
            italic_ratio=italic_spans / total_spans if total_spans else 0.0,
            monospace_ratio=mono_spans / total_spans if total_spans else 0.0,
            distinct_sizes=len(distinct),
        )

    # ------------------------------------------------------------------
    # Layout analysis (cycle 5)
    # ------------------------------------------------------------------

    def _analyze_layout_features(self, doc: fitz.Document, sampled: list[int]) -> LayoutFeatures:
        """Analyze page layout geometry across sampled pages.

        Args:
            doc: Open fitz.Document.
            sampled: List of page indices to analyze.

        Returns:
            Populated LayoutFeatures.
        """
        widths: list[float] = []
        heights: list[float] = []
        text_area_ratios: list[float] = []
        column_counts: list[int] = []
        margins_left: list[float] = []
        margins_right: list[float] = []
        margins_top: list[float] = []
        margins_bottom: list[float] = []

        for page_idx in sampled:
            page = doc[page_idx]
            rect = page.rect
            w, h = rect.width, rect.height
            widths.append(w)
            heights.append(h)

            ml, mt, mr, mb = self._compute_margins(page)
            margins_left.append(ml)
            margins_right.append(mr)
            margins_top.append(mt)
            margins_bottom.append(mb)

            # Text area ratio
            page_area = w * h
            text_w = w - ml - mr
            text_h = h - mt - mb
            if page_area > 0 and text_w > 0 and text_h > 0:
                text_area_ratios.append((text_w * text_h) / page_area)
            else:
                text_area_ratios.append(0.0)

            # Column detection
            cols = self._detect_columns(page)
            column_counts.append(cols)

        if not widths:
            return LayoutFeatures()

        w_arr = np.array(widths)
        h_arr = np.array(heights)

        # Consistency: 1.0 - (std / mean) clamped to [0, 1]
        w_consistency = 1.0
        h_consistency = 1.0
        if len(widths) > 1:
            w_mean = float(np.mean(w_arr))
            h_mean = float(np.mean(h_arr))
            if w_mean > 0:
                w_consistency = max(0.0, 1.0 - float(np.std(w_arr)) / w_mean)
            if h_mean > 0:
                h_consistency = max(0.0, 1.0 - float(np.std(h_arr)) / h_mean)

        # Median column count (most common)
        col_counter = Counter(column_counts)
        estimated_columns = col_counter.most_common(1)[0][0] if col_counter else 1

        return LayoutFeatures(
            page_width=float(np.median(w_arr)),
            page_height=float(np.median(h_arr)),
            width_consistency=w_consistency,
            height_consistency=h_consistency,
            margin_left=float(np.median(margins_left)) if margins_left else 0.0,
            margin_right=float(np.median(margins_right)) if margins_right else 0.0,
            margin_top=float(np.median(margins_top)) if margins_top else 0.0,
            margin_bottom=float(np.median(margins_bottom)) if margins_bottom else 0.0,
            text_area_ratio=float(np.mean(text_area_ratios)),
            estimated_columns=estimated_columns,
        )

    @staticmethod
    def _compute_margins(page: fitz.Page) -> tuple[float, float, float, float]:
        """Estimate page margins from text bounding boxes.

        Args:
            page: A fitz.Page object.

        Returns:
            Tuple of (left, top, right, bottom) margins in points.
        """
        blocks = page.get_text("blocks")
        if not blocks:
            return (0.0, 0.0, 0.0, 0.0)

        rect = page.rect
        # blocks: (x0, y0, x1, y1, text, block_no, block_type)
        text_blocks = [b for b in blocks if b[6] == 0]  # type 0 = text
        if not text_blocks:
            return (0.0, 0.0, 0.0, 0.0)

        x0_min = min(b[0] for b in text_blocks)
        y0_min = min(b[1] for b in text_blocks)
        x1_max = max(b[2] for b in text_blocks)
        y1_max = max(b[3] for b in text_blocks)

        left = max(0.0, x0_min)
        top = max(0.0, y0_min)
        right = max(0.0, rect.width - x1_max)
        bottom = max(0.0, rect.height - y1_max)

        return (left, top, right, bottom)

    @staticmethod
    def _detect_columns(page: fitz.Page) -> int:
        """Estimate number of text columns using x-coordinate clustering.

        Looks for gaps in the horizontal distribution of text blocks
        that would indicate column boundaries.

        Args:
            page: A fitz.Page object.

        Returns:
            Estimated number of columns (1, 2, or 3).
        """
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0]
        if len(text_blocks) < 2:
            return 1

        rect = page.rect
        page_width = rect.width

        # Collect x-center coordinates of each text block
        centers = sorted((b[0] + b[2]) / 2 for b in text_blocks)

        if len(centers) < 2:
            return 1

        # Look for a gap in the middle third of the page
        mid_start = page_width * 0.3
        mid_end = page_width * 0.7

        left_blocks = [c for c in centers if c < mid_start]
        right_blocks = [c for c in centers if c > mid_end]

        if left_blocks and right_blocks:
            # Check for actual gap (no blocks in the middle)
            mid_blocks = [c for c in centers if mid_start <= c <= mid_end]
            if len(mid_blocks) <= len(text_blocks) * 0.2:
                return 2

        return 1

    # ------------------------------------------------------------------
    # Character analysis (cycle 6)
    # ------------------------------------------------------------------

    def _analyze_character_features(
        self, doc: fitz.Document, sampled: list[int]
    ) -> CharacterFeatures:
        """Analyze character class distributions across sampled pages.

        Args:
            doc: Open fitz.Document.
            sampled: List of page indices to analyze.

        Returns:
            Populated CharacterFeatures.
        """
        all_text: list[str] = []
        for page_idx in sampled:
            page = doc[page_idx]
            text = page.get_text("text")
            if text:
                all_text.append(text)

        combined = "".join(all_text)
        total = len(combined)

        if total == 0:
            return CharacterFeatures()

        alpha_count = sum(1 for c in combined if c.isalpha())
        numeric_count = sum(1 for c in combined if c.isdigit())
        whitespace_count = sum(1 for c in combined if c.isspace())
        punct_count = sum(1 for c in combined if c in string.punctuation)
        special_count = total - alpha_count - numeric_count - whitespace_count - punct_count

        uppercase_count = sum(1 for c in combined if c.isupper())
        uppercase_ratio = uppercase_count / alpha_count if alpha_count > 0 else 0.0

        return CharacterFeatures(
            alpha_ratio=alpha_count / total,
            numeric_ratio=numeric_count / total,
            punctuation_ratio=punct_count / total,
            whitespace_ratio=whitespace_count / total,
            special_ratio=max(0.0, special_count / total),
            uppercase_ratio=uppercase_ratio,
            total_chars=total,
        )

    # ------------------------------------------------------------------
    # Repetition analysis (cycle 7)
    # ------------------------------------------------------------------

    def _analyze_repetition_features(
        self, doc: fitz.Document, sampled: list[int]
    ) -> RepetitionFeatures:
        """Detect cross-page repetition patterns.

        Checks for repeated headers, footers, and sequential page numbers
        across sampled pages.

        Args:
            doc: Open fitz.Document.
            sampled: List of page indices to analyze.

        Returns:
            Populated RepetitionFeatures.
        """
        if len(sampled) < 2:
            return RepetitionFeatures(first_line_diversity=1.0)

        top_texts: list[str] = []
        bottom_texts: list[str] = []
        first_lines: list[str] = []

        for page_idx in sampled:
            page = doc[page_idx]
            rect = page.rect
            page_height = rect.height

            blocks = page.get_text("blocks")
            text_blocks = sorted(
                [b for b in blocks if b[6] == 0],
                key=lambda b: b[1],
            )

            if not text_blocks:
                continue

            # Top region (top 8% of page)
            top_threshold = page_height * 0.08
            top_block_texts = [b[4].strip() for b in text_blocks if b[1] < top_threshold]
            if top_block_texts:
                top_texts.append(top_block_texts[0])

            # Bottom region (bottom 8% of page)
            bottom_threshold = page_height * 0.92
            bottom_block_texts = [b[4].strip() for b in text_blocks if b[3] > bottom_threshold]
            if bottom_block_texts:
                bottom_texts.append(bottom_block_texts[-1])

            # First meaningful line of content (skip header region)
            content_blocks = [b for b in text_blocks if b[1] >= top_threshold]
            if content_blocks:
                first_lines.append(content_blocks[0][4].strip())

        # Detect repeated headers
        has_headers = self._has_repeated_text(top_texts)

        # Detect footers / page numbers
        has_footers = self._has_repeated_text(bottom_texts)
        has_page_numbers = self._has_page_numbers(bottom_texts)

        # Repetition ratio: fraction of pages with any repeated element
        pages_with_repetition = 0
        for i, page_idx in enumerate(sampled):
            if i < len(top_texts) and top_texts.count(top_texts[i]) > 1:
                pages_with_repetition += 1
            elif i < len(bottom_texts) and self._looks_like_page_number(
                bottom_texts[i] if i < len(bottom_texts) else ""
            ):
                pages_with_repetition += 1
        repetition_ratio = pages_with_repetition / len(sampled)

        # First-line diversity
        unique_first = len(set(first_lines)) if first_lines else 0
        total_first = len(first_lines) if first_lines else 1
        first_line_diversity = unique_first / total_first

        return RepetitionFeatures(
            has_page_numbers=has_page_numbers,
            has_headers=has_headers,
            has_footers=has_footers,
            repetition_ratio=repetition_ratio,
            first_line_diversity=first_line_diversity,
        )

    @staticmethod
    def _has_repeated_text(texts: list[str]) -> bool:
        """Check if any text appears on more than half of pages.

        Args:
            texts: List of text strings from same region across pages.

        Returns:
            True if a repeated pattern found.
        """
        if len(texts) < 2:
            return False
        counter = Counter(texts)
        most_common_count = counter.most_common(1)[0][1]
        return most_common_count >= len(texts) * 0.5

    @staticmethod
    def _has_page_numbers(bottom_texts: list[str]) -> bool:
        """Check for sequential page numbers in bottom text.

        Args:
            bottom_texts: Text from bottom region of each page.

        Returns:
            True if sequential numbers found.
        """
        if len(bottom_texts) < 2:
            return False

        numbers = []
        for text in bottom_texts:
            # Extract any number from the text
            nums = re.findall(r"\d+", text.strip())
            if nums:
                numbers.append(int(nums[0]))

        if len(numbers) < 2:
            return False

        # Check for sequential pattern
        sequential = sum(1 for i in range(1, len(numbers)) if numbers[i] == numbers[i - 1] + 1)
        return sequential >= len(numbers) * 0.5

    @staticmethod
    def _looks_like_page_number(text: str) -> bool:
        """Check if text looks like a page number.

        Args:
            text: Text to check.

        Returns:
            True if text appears to be a page number.
        """
        stripped = text.strip()
        if not stripped:
            return False
        # Simple page number: just digits optionally with dashes/dots
        return bool(re.match(r"^[\d\s\-\.]+$", stripped))

    # ------------------------------------------------------------------
    # Structural rhythm analysis (cycle 8)
    # ------------------------------------------------------------------

    def _analyze_structural_rhythm(
        self, doc: fitz.Document, sampled: list[int]
    ) -> StructuralRhythmFeatures:
        """Analyze document structural patterns and element densities.

        Args:
            doc: Open fitz.Document.
            sampled: List of page indices to analyze.

        Returns:
            Populated StructuralRhythmFeatures.
        """
        # TOC analysis (uses full document, not sampled)
        toc = doc.get_toc()
        has_toc = len(toc) > 0
        toc_depth = max((entry[0] for entry in toc), default=0) if toc else 0

        # Link count across sampled pages
        link_count = 0
        for page_idx in sampled:
            page = doc[page_idx]
            links = page.get_links()
            link_count += len([lnk for lnk in links if lnk.get("kind") == fitz.LINK_URI])

        # Scale link count estimate to full document
        if sampled and len(sampled) < len(doc):
            link_count = int(link_count * len(doc) / len(sampled))

        # Analyze font-based structural elements
        heading_count = 0
        image_count = 0
        table_count = 0
        heading_sizes: set[float] = set()

        for page_idx in sampled:
            page = doc[page_idx]

            # Detect images
            image_count += len(page.get_images(full=True))

            # Detect drawings that look like tables (rectangles)
            drawings = page.get_drawings()
            rect_count = sum(
                1
                for d in drawings
                if d.get("type") == "re"
                or (d.get("items") and any(item[0] == "re" for item in d.get("items", [])))
            )
            if rect_count >= 4:  # At least 4 rectangles suggests a table
                table_count += 1

            # Detect headings via font size analysis
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            page_sizes: list[float] = []
            for block in blocks.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            page_sizes.append(span.get("size", 0.0))

            if page_sizes:
                median_size = float(np.median(page_sizes))
                # Spans with size > 1.2x median are likely headings
                threshold = median_size * 1.2
                for size in page_sizes:
                    if size > threshold:
                        heading_count += 1
                        heading_sizes.add(round(size, 1))

        n_pages = len(sampled) if sampled else 1

        return StructuralRhythmFeatures(
            heading_density=heading_count / n_pages,
            table_density=table_count / n_pages,
            image_density=image_count / n_pages,
            list_density=0.0,  # List detection requires content parsing
            has_toc=has_toc,
            toc_depth=toc_depth,
            link_count=link_count,
            heading_size_levels=len(heading_sizes),
        )
