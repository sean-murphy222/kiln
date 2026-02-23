"""Tier 1 structural document fingerprinting.

Analyzes raw PDF structure statistically — no content parsing — to produce
a feature vector for ML document-type classification.

Data flow::

    PDF File → DocumentFingerprinter.extract(path) → DocumentFingerprint
        → .to_feature_vector() → list[float]  (for T-002 ML classifier)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    structural_rhythm: StructuralRhythmFeatures = field(
        default_factory=StructuralRhythmFeatures
    )

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
            "byte", "font", "layout", "char", "rep", "rhythm",
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
