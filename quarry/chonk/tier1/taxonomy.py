"""Tier 1 document type taxonomy and structural profiles.

Defines the document types the ML classifier can predict and the
structural feature profiles used to generate synthetic training data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DocumentType(str, Enum):
    """Enumeration of all classifiable document types.

    UNKNOWN is a sentinel returned when confidence is below threshold;
    it is not a trainable class.
    """

    TECHNICAL_MANUAL = "technical_manual"
    MAINTENANCE_PROCEDURE = "maintenance_procedure"
    PARTS_CATALOG = "parts_catalog"
    WIRING_DIAGRAM = "wiring_diagram"
    REGULATION = "regulation"
    SPECIFICATION = "specification"
    TRAINING_MATERIAL = "training_material"
    FORM = "form"
    REPORT = "report"
    REFERENCE_CARD = "reference_card"
    ACADEMIC_PAPER = "academic_paper"
    PRESENTATION = "presentation"
    DATASHEET = "datasheet"
    CONTRACT = "contract"
    UNKNOWN = "unknown"


TRAINABLE_TYPES: list[DocumentType] = [t for t in DocumentType if t != DocumentType.UNKNOWN]


@dataclass
class DocumentTypeProfile:
    """Structural feature profile for one document type.

    Used by the synthetic training data generator to sample realistic
    feature vectors for each document type.

    Args:
        label: The document type this profile describes.
        description: Human-readable description.
        feature_means: Expected mean value per feature name.
        feature_stds: Expected standard deviation per feature name.
            Features not listed default to std=1.0.
    """

    label: DocumentType
    description: str
    feature_means: dict[str, float] = field(default_factory=dict)
    feature_stds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "label": self.label.value,
            "description": self.description,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentTypeProfile:
        """Deserialize from dictionary."""
        return cls(
            label=DocumentType(data["label"]),
            description=data["description"],
            feature_means=data.get("feature_means", {}),
            feature_stds=data.get("feature_stds", {}),
        )


# ---------------------------------------------------------------------------
# Default feature values (used when a profile doesn't specify a feature)
# ---------------------------------------------------------------------------

_DEFAULT_MEANS: dict[str, float] = {
    "byte_file_size": 500_000.0,
    "byte_pdf_version": 1.7,
    "byte_object_count": 100.0,
    "byte_stream_count": 50.0,
    "byte_has_metadata": 0.5,
    "byte_has_xmp_metadata": 0.3,
    "byte_page_count": 10.0,
    "byte_encrypted": 0.0,
    "byte_has_acroform": 0.0,
    "font_font_count": 3.0,
    "font_size_min": 8.0,
    "font_size_max": 14.0,
    "font_size_mean": 10.5,
    "font_size_std": 1.5,
    "font_size_median": 10.0,
    "font_bold_ratio": 0.05,
    "font_italic_ratio": 0.02,
    "font_monospace_ratio": 0.0,
    "font_distinct_sizes": 3.0,
    "layout_page_width": 612.0,
    "layout_page_height": 792.0,
    "layout_width_consistency": 1.0,
    "layout_height_consistency": 1.0,
    "layout_margin_left": 72.0,
    "layout_margin_right": 72.0,
    "layout_margin_top": 72.0,
    "layout_margin_bottom": 72.0,
    "layout_text_area_ratio": 0.65,
    "layout_estimated_columns": 1.0,
    "char_alpha_ratio": 0.65,
    "char_numeric_ratio": 0.05,
    "char_punctuation_ratio": 0.05,
    "char_whitespace_ratio": 0.20,
    "char_special_ratio": 0.05,
    "char_uppercase_ratio": 0.10,
    "char_total_chars": 5000.0,
    "rep_has_page_numbers": 0.5,
    "rep_has_headers": 0.3,
    "rep_has_footers": 0.2,
    "rep_repetition_ratio": 0.3,
    "rep_first_line_diversity": 0.7,
    "rhythm_heading_density": 1.0,
    "rhythm_table_density": 0.1,
    "rhythm_image_density": 0.1,
    "rhythm_list_density": 0.0,
    "rhythm_has_toc": 0.0,
    "rhythm_toc_depth": 0.0,
    "rhythm_link_count": 0.0,
    "rhythm_heading_size_levels": 2.0,
}

_DEFAULT_STD = 1.0

# ---------------------------------------------------------------------------
# Document type profiles
# ---------------------------------------------------------------------------

DOCUMENT_TYPE_PROFILES: dict[DocumentType, DocumentTypeProfile] = {
    DocumentType.TECHNICAL_MANUAL: DocumentTypeProfile(
        label=DocumentType.TECHNICAL_MANUAL,
        description="Military technical manuals, TMs, FMs — long, structured, TOC",
        feature_means={
            "byte_page_count": 120.0,
            "byte_file_size": 8_000_000.0,
            "byte_object_count": 800.0,
            "byte_stream_count": 400.0,
            "byte_has_metadata": 1.0,
            "byte_has_xmp_metadata": 0.5,
            "font_font_count": 4.0,
            "font_size_mean": 10.0,
            "font_size_std": 3.0,
            "font_bold_ratio": 0.08,
            "font_distinct_sizes": 5.0,
            "layout_text_area_ratio": 0.70,
            "char_alpha_ratio": 0.68,
            "char_numeric_ratio": 0.06,
            "rep_has_page_numbers": 1.0,
            "rep_has_headers": 1.0,
            "rep_repetition_ratio": 0.8,
            "rhythm_has_toc": 1.0,
            "rhythm_toc_depth": 3.0,
            "rhythm_heading_density": 2.5,
            "rhythm_table_density": 0.3,
            "rhythm_image_density": 0.4,
            "rhythm_heading_size_levels": 4.0,
        },
        feature_stds={
            "byte_page_count": 60.0,
            "byte_file_size": 4_000_000.0,
            "byte_object_count": 300.0,
            "byte_stream_count": 150.0,
            "rhythm_heading_density": 1.0,
        },
    ),
    DocumentType.MAINTENANCE_PROCEDURE: DocumentTypeProfile(
        label=DocumentType.MAINTENANCE_PROCEDURE,
        description="Step-by-step maintenance procedures, checklists, work orders",
        feature_means={
            "byte_page_count": 30.0,
            "byte_file_size": 2_000_000.0,
            "byte_object_count": 200.0,
            "byte_stream_count": 100.0,
            "font_font_count": 3.0,
            "font_size_mean": 10.0,
            "font_bold_ratio": 0.12,
            "font_monospace_ratio": 0.05,
            "layout_text_area_ratio": 0.65,
            "char_alpha_ratio": 0.60,
            "char_numeric_ratio": 0.10,
            "rep_has_page_numbers": 1.0,
            "rep_has_headers": 0.8,
            "rhythm_heading_density": 1.5,
            "rhythm_table_density": 0.4,
            "rhythm_list_density": 0.5,
            "rhythm_has_toc": 0.3,
            "rhythm_toc_depth": 1.0,
        },
        feature_stds={
            "byte_page_count": 20.0,
            "byte_file_size": 1_000_000.0,
            "rhythm_list_density": 0.3,
        },
    ),
    DocumentType.PARTS_CATALOG: DocumentTypeProfile(
        label=DocumentType.PARTS_CATALOG,
        description="Parts lists, illustrated parts breakdowns, IPBs",
        feature_means={
            "byte_page_count": 80.0,
            "byte_file_size": 6_000_000.0,
            "byte_object_count": 600.0,
            "byte_stream_count": 300.0,
            "font_font_count": 3.0,
            "font_size_mean": 8.0,
            "font_monospace_ratio": 0.10,
            "layout_text_area_ratio": 0.75,
            "char_alpha_ratio": 0.45,
            "char_numeric_ratio": 0.20,
            "rep_has_page_numbers": 1.0,
            "rep_has_headers": 1.0,
            "rhythm_table_density": 0.8,
            "rhythm_image_density": 0.5,
            "rhythm_has_toc": 1.0,
            "rhythm_toc_depth": 2.0,
            "rhythm_heading_density": 0.8,
        },
        feature_stds={
            "byte_page_count": 50.0,
            "byte_file_size": 3_000_000.0,
            "char_numeric_ratio": 0.05,
        },
    ),
    DocumentType.WIRING_DIAGRAM: DocumentTypeProfile(
        label=DocumentType.WIRING_DIAGRAM,
        description="Electrical wiring diagrams, schematics, circuit layouts",
        feature_means={
            "byte_page_count": 15.0,
            "byte_file_size": 3_000_000.0,
            "byte_object_count": 300.0,
            "byte_stream_count": 200.0,
            "font_font_count": 2.0,
            "font_size_mean": 7.0,
            "layout_text_area_ratio": 0.30,
            "char_alpha_ratio": 0.25,
            "char_numeric_ratio": 0.15,
            "char_special_ratio": 0.15,
            "rep_has_page_numbers": 0.8,
            "rhythm_image_density": 1.8,
            "rhythm_table_density": 0.05,
            "rhythm_heading_density": 0.3,
        },
        feature_stds={
            "byte_page_count": 10.0,
            "byte_file_size": 2_000_000.0,
            "rhythm_image_density": 0.8,
        },
    ),
    DocumentType.REGULATION: DocumentTypeProfile(
        label=DocumentType.REGULATION,
        description="Regulations, directives, policies, legal documents",
        feature_means={
            "byte_page_count": 50.0,
            "byte_file_size": 1_500_000.0,
            "byte_object_count": 200.0,
            "font_font_count": 3.0,
            "font_size_mean": 11.0,
            "font_bold_ratio": 0.06,
            "layout_text_area_ratio": 0.72,
            "char_alpha_ratio": 0.78,
            "char_numeric_ratio": 0.04,
            "rep_has_page_numbers": 1.0,
            "rep_has_headers": 0.9,
            "rhythm_heading_density": 1.8,
            "rhythm_table_density": 0.05,
            "rhythm_has_toc": 1.0,
            "rhythm_toc_depth": 3.0,
            "rhythm_heading_size_levels": 4.0,
        },
        feature_stds={
            "byte_page_count": 30.0,
            "byte_file_size": 800_000.0,
        },
    ),
    DocumentType.SPECIFICATION: DocumentTypeProfile(
        label=DocumentType.SPECIFICATION,
        description="Technical specifications, MIL-SPECs, standards",
        feature_means={
            "byte_page_count": 40.0,
            "byte_file_size": 2_500_000.0,
            "byte_object_count": 250.0,
            "font_font_count": 4.0,
            "font_size_mean": 10.0,
            "font_distinct_sizes": 5.0,
            "layout_text_area_ratio": 0.68,
            "char_alpha_ratio": 0.62,
            "char_numeric_ratio": 0.12,
            "rep_has_page_numbers": 1.0,
            "rhythm_heading_density": 1.5,
            "rhythm_table_density": 0.6,
            "rhythm_has_toc": 0.8,
            "rhythm_toc_depth": 2.0,
        },
        feature_stds={
            "byte_page_count": 25.0,
            "byte_file_size": 1_500_000.0,
            "rhythm_table_density": 0.3,
        },
    ),
    DocumentType.TRAINING_MATERIAL: DocumentTypeProfile(
        label=DocumentType.TRAINING_MATERIAL,
        description="Training manuals, courseware, instructional materials",
        feature_means={
            "byte_page_count": 60.0,
            "byte_file_size": 4_000_000.0,
            "byte_object_count": 400.0,
            "font_font_count": 5.0,
            "font_size_mean": 11.0,
            "font_bold_ratio": 0.12,
            "font_distinct_sizes": 6.0,
            "layout_text_area_ratio": 0.60,
            "char_alpha_ratio": 0.70,
            "rep_has_page_numbers": 1.0,
            "rhythm_heading_density": 2.5,
            "rhythm_image_density": 0.6,
            "rhythm_has_toc": 1.0,
            "rhythm_toc_depth": 2.0,
            "rhythm_heading_size_levels": 4.0,
        },
        feature_stds={
            "byte_page_count": 30.0,
            "byte_file_size": 2_000_000.0,
            "rhythm_image_density": 0.3,
        },
    ),
    DocumentType.FORM: DocumentTypeProfile(
        label=DocumentType.FORM,
        description="Fillable forms, applications, checklists with fields",
        feature_means={
            "byte_page_count": 4.0,
            "byte_file_size": 200_000.0,
            "byte_object_count": 80.0,
            "byte_has_acroform": 1.0,
            "font_font_count": 3.0,
            "font_size_mean": 10.0,
            "layout_text_area_ratio": 0.55,
            "char_alpha_ratio": 0.55,
            "char_numeric_ratio": 0.10,
            "char_punctuation_ratio": 0.08,
            "rep_has_page_numbers": 0.3,
            "rhythm_table_density": 0.4,
            "rhythm_heading_density": 0.5,
            "rhythm_has_toc": 0.0,
        },
        feature_stds={
            "byte_page_count": 3.0,
            "byte_file_size": 150_000.0,
        },
    ),
    DocumentType.REPORT: DocumentTypeProfile(
        label=DocumentType.REPORT,
        description="Incident reports, inspection reports, narrative + data",
        feature_means={
            "byte_page_count": 15.0,
            "byte_file_size": 800_000.0,
            "byte_object_count": 120.0,
            "font_font_count": 3.0,
            "font_size_mean": 11.0,
            "font_bold_ratio": 0.06,
            "layout_text_area_ratio": 0.70,
            "char_alpha_ratio": 0.72,
            "char_numeric_ratio": 0.06,
            "rep_has_page_numbers": 0.6,
            "rhythm_heading_density": 1.2,
            "rhythm_table_density": 0.15,
            "rhythm_has_toc": 0.1,
        },
        feature_stds={
            "byte_page_count": 10.0,
            "byte_file_size": 500_000.0,
        },
    ),
    DocumentType.REFERENCE_CARD: DocumentTypeProfile(
        label=DocumentType.REFERENCE_CARD,
        description="Quick-reference guides, laminated cards, dense short docs",
        feature_means={
            "byte_page_count": 2.0,
            "byte_file_size": 100_000.0,
            "byte_object_count": 30.0,
            "font_font_count": 3.0,
            "font_size_mean": 8.0,
            "font_bold_ratio": 0.18,
            "layout_text_area_ratio": 0.85,
            "char_alpha_ratio": 0.60,
            "char_numeric_ratio": 0.08,
            "rep_has_page_numbers": 0.0,
            "rhythm_heading_density": 3.5,
            "rhythm_table_density": 0.3,
            "rhythm_has_toc": 0.0,
        },
        feature_stds={
            "byte_page_count": 1.5,
            "byte_file_size": 80_000.0,
            "rhythm_heading_density": 1.5,
        },
    ),
    DocumentType.ACADEMIC_PAPER: DocumentTypeProfile(
        label=DocumentType.ACADEMIC_PAPER,
        description="Research papers, 2-column, abstract structure",
        feature_means={
            "byte_page_count": 12.0,
            "byte_file_size": 600_000.0,
            "byte_object_count": 100.0,
            "font_font_count": 4.0,
            "font_size_mean": 10.0,
            "font_italic_ratio": 0.08,
            "layout_estimated_columns": 2.0,
            "layout_text_area_ratio": 0.75,
            "char_alpha_ratio": 0.72,
            "char_numeric_ratio": 0.05,
            "rep_has_page_numbers": 1.0,
            "rhythm_heading_density": 1.8,
            "rhythm_has_toc": 0.0,
            "rhythm_link_count": 10.0,
            "rhythm_heading_size_levels": 3.0,
        },
        feature_stds={
            "byte_page_count": 6.0,
            "byte_file_size": 400_000.0,
            "rhythm_link_count": 8.0,
        },
    ),
    DocumentType.PRESENTATION: DocumentTypeProfile(
        label=DocumentType.PRESENTATION,
        description="Slide decks exported to PDF, landscape, sparse text",
        feature_means={
            "byte_page_count": 25.0,
            "byte_file_size": 3_000_000.0,
            "byte_object_count": 300.0,
            "font_font_count": 4.0,
            "font_size_mean": 18.0,
            "font_bold_ratio": 0.20,
            "font_distinct_sizes": 5.0,
            "layout_page_width": 792.0,
            "layout_page_height": 612.0,
            "layout_text_area_ratio": 0.35,
            "char_alpha_ratio": 0.55,
            "char_total_chars": 2000.0,
            "rep_has_page_numbers": 0.5,
            "rhythm_image_density": 0.8,
            "rhythm_heading_density": 1.0,
            "rhythm_has_toc": 0.0,
        },
        feature_stds={
            "byte_page_count": 15.0,
            "byte_file_size": 2_000_000.0,
            "font_size_mean": 4.0,
        },
    ),
    DocumentType.DATASHEET: DocumentTypeProfile(
        label=DocumentType.DATASHEET,
        description="Component datasheets, spec tables, dense numerical",
        feature_means={
            "byte_page_count": 8.0,
            "byte_file_size": 400_000.0,
            "byte_object_count": 80.0,
            "font_font_count": 3.0,
            "font_size_mean": 8.5,
            "font_monospace_ratio": 0.08,
            "layout_text_area_ratio": 0.78,
            "char_alpha_ratio": 0.50,
            "char_numeric_ratio": 0.20,
            "rep_has_page_numbers": 0.7,
            "rhythm_table_density": 0.7,
            "rhythm_heading_density": 1.2,
            "rhythm_image_density": 0.3,
            "rhythm_has_toc": 0.0,
        },
        feature_stds={
            "byte_page_count": 5.0,
            "byte_file_size": 300_000.0,
            "char_numeric_ratio": 0.05,
        },
    ),
    DocumentType.CONTRACT: DocumentTypeProfile(
        label=DocumentType.CONTRACT,
        description="Legal/procurement contracts, numbered clauses, dense text",
        feature_means={
            "byte_page_count": 25.0,
            "byte_file_size": 800_000.0,
            "byte_object_count": 150.0,
            "font_font_count": 2.0,
            "font_size_mean": 11.0,
            "font_bold_ratio": 0.04,
            "font_italic_ratio": 0.03,
            "layout_text_area_ratio": 0.75,
            "char_alpha_ratio": 0.82,
            "char_numeric_ratio": 0.03,
            "rep_has_page_numbers": 1.0,
            "rhythm_heading_density": 0.6,
            "rhythm_table_density": 0.05,
            "rhythm_has_toc": 0.2,
            "rhythm_heading_size_levels": 2.0,
        },
        feature_stds={
            "byte_page_count": 15.0,
            "byte_file_size": 500_000.0,
        },
    ),
}
