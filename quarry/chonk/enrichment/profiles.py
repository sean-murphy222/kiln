"""Metadata extraction profiles for document types.

Each profile defines a set of extraction rules appropriate for a
particular document type. The registry maps DocumentType to profile.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from chonk.enrichment.rules import ExtractionRule, ExtractionSource
from chonk.tier1.taxonomy import DocumentType


@dataclass
class MetadataProfile:
    """A collection of extraction rules for one document type.

    Args:
        name: Profile identifier.
        document_type: The document type this profile targets.
        rules: Extraction rules to apply.
        description: Human-readable description.
    """

    name: str
    document_type: DocumentType
    rules: list[ExtractionRule] = field(default_factory=list)
    description: str = ""

    @property
    def required_fields(self) -> list[str]:
        """Return names of required fields in this profile."""
        return [r.field_name for r in self.rules if r.required]

    @property
    def optional_fields(self) -> list[str]:
        """Return names of optional fields in this profile."""
        return [r.field_name for r in self.rules if not r.required]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "document_type": self.document_type.value,
            "rules": [r.to_dict() for r in self.rules],
            "description": self.description,
        }


# --- Shared extraction rules ---

_TM_NUMBER_RULE = ExtractionRule(
    field_name="tm_number",
    pattern=re.compile(r"(TM\s+\d+-\d+(?:-\d+)*)", re.IGNORECASE),
    source=ExtractionSource.BOTH,
    required=True,
    description="Technical Manual number (e.g., TM 9-2320-272-20)",
)

_NSN_RULE = ExtractionRule(
    field_name="nsn",
    pattern=re.compile(r"(\d{4}-\d{2}-\d{3}-\d{4})"),
    source=ExtractionSource.CONTENT,
    required=False,
    description="National Stock Number (e.g., 2320-01-107-7155)",
)

_MAINTENANCE_LEVEL_RULE = ExtractionRule(
    field_name="maintenance_level",
    pattern=re.compile(
        r"(organizational|direct support|general support|depot"
        r"|unit|intermediate|field)\s+(?:level\s+)?maintenance",
        re.IGNORECASE,
    ),
    source=ExtractionSource.CONTENT,
    required=False,
    description="Maintenance level (organizational, DS, GS, depot)",
)

_EQUIPMENT_SYSTEM_RULE = ExtractionRule(
    field_name="equipment_system",
    pattern=re.compile(
        r"(?:for|of|on)\s+(?:the\s+)?([A-Z][A-Za-z0-9\s/-]{3,40}?)(?:\.|,|\n|$)",
    ),
    source=ExtractionSource.CONTENT,
    required=False,
    description="Equipment system name or designation",
)

_WORK_PACKAGE_RULE = ExtractionRule(
    field_name="work_package",
    pattern=re.compile(r"(WP\s+\d{4}\s+\d{2})", re.IGNORECASE),
    source=ExtractionSource.CONTENT,
    required=False,
    description="Work package identifier (e.g., WP 0001 00)",
)

_LIN_RULE = ExtractionRule(
    field_name="lin",
    pattern=re.compile(r"LIN:\s*([A-Z]\d{5})"),
    source=ExtractionSource.CONTENT,
    required=False,
    description="Line Item Number (e.g., T51687)",
)

_SMR_CODE_RULE = ExtractionRule(
    field_name="smr_code",
    pattern=re.compile(r"SMR(?:\s+CODE)?:\s*([A-Z]{4,6})"),
    source=ExtractionSource.CONTENT,
    required=False,
    description="Source, Maintenance, and Recoverability code",
)

_SECTION_NUMBER_RULE = ExtractionRule(
    field_name="section_number",
    pattern=re.compile(
        r"(?:Section|Chapter|Part)\s+(\d+(?:\.\d+)*)",
        re.IGNORECASE,
    ),
    source=ExtractionSource.BOTH,
    required=False,
    description="Section or chapter number from hierarchy",
)

_PARAGRAPH_REF_RULE = ExtractionRule(
    field_name="paragraph_ref",
    pattern=re.compile(r"(?:para(?:graph)?\.?\s*)(\d+(?:-\d+)*(?:\.\d+)*)", re.IGNORECASE),
    source=ExtractionSource.CONTENT,
    required=False,
    description="Paragraph reference number",
)

_FIGURE_REF_RULE = ExtractionRule(
    field_name="figure_ref",
    pattern=re.compile(r"(?:Fig(?:ure)?\.?\s*)(\d+(?:-\d+)*)", re.IGNORECASE),
    source=ExtractionSource.CONTENT,
    required=False,
    description="Figure reference number",
)

_TABLE_REF_RULE = ExtractionRule(
    field_name="table_ref",
    pattern=re.compile(r"(?:Table\s+)(\d+(?:-\d+)*)", re.IGNORECASE),
    source=ExtractionSource.CONTENT,
    required=False,
    description="Table reference number",
)

_REGULATION_NUMBER_RULE = ExtractionRule(
    field_name="regulation_number",
    pattern=re.compile(
        r"((?:AR|DA PAM|FM|ATP|ADP|TC|STP|MIL-STD|MIL-SPEC)\s+" r"\d+(?:-\d+)*(?:\.\d+)*)",
        re.IGNORECASE,
    ),
    source=ExtractionSource.BOTH,
    required=False,
    description="Regulation or standard reference number",
)


# --- Built-in profiles ---

_TECHNICAL_MANUAL_PROFILE = MetadataProfile(
    name="technical_manual",
    document_type=DocumentType.TECHNICAL_MANUAL,
    description="Profile for military technical manuals (TMs, FMs)",
    rules=[
        _TM_NUMBER_RULE,
        _NSN_RULE,
        _MAINTENANCE_LEVEL_RULE,
        _EQUIPMENT_SYSTEM_RULE,
        _WORK_PACKAGE_RULE,
        _LIN_RULE,
        _SMR_CODE_RULE,
        _SECTION_NUMBER_RULE,
        _FIGURE_REF_RULE,
        _TABLE_REF_RULE,
    ],
)

_MAINTENANCE_PROCEDURE_PROFILE = MetadataProfile(
    name="maintenance_procedure",
    document_type=DocumentType.MAINTENANCE_PROCEDURE,
    description="Profile for maintenance procedures and work orders",
    rules=[
        ExtractionRule(
            field_name="tm_number",
            pattern=re.compile(r"(TM\s+\d+-\d+(?:-\d+)*)", re.IGNORECASE),
            source=ExtractionSource.BOTH,
            required=True,
            description="Source TM number",
        ),
        _NSN_RULE,
        ExtractionRule(
            field_name="maintenance_level",
            pattern=re.compile(
                r"(organizational|direct support|general support|depot"
                r"|unit|intermediate|field)\s+(?:level\s+)?maintenance",
                re.IGNORECASE,
            ),
            source=ExtractionSource.CONTENT,
            required=True,
            description="Required maintenance level",
        ),
        _WORK_PACKAGE_RULE,
        _SMR_CODE_RULE,
        _EQUIPMENT_SYSTEM_RULE,
        _SECTION_NUMBER_RULE,
        _FIGURE_REF_RULE,
    ],
)

_PARTS_CATALOG_PROFILE = MetadataProfile(
    name="parts_catalog",
    document_type=DocumentType.PARTS_CATALOG,
    description="Profile for illustrated parts breakdowns and catalogs",
    rules=[
        _TM_NUMBER_RULE,
        ExtractionRule(
            field_name="nsn",
            pattern=re.compile(r"(\d{4}-\d{2}-\d{3}-\d{4})"),
            source=ExtractionSource.CONTENT,
            required=True,
            description="National Stock Number (required for parts)",
        ),
        _LIN_RULE,
        _SMR_CODE_RULE,
        _FIGURE_REF_RULE,
        ExtractionRule(
            field_name="part_number",
            pattern=re.compile(r"P/N:\s*([A-Za-z0-9-]{3,20})"),
            source=ExtractionSource.CONTENT,
            required=False,
            description="Manufacturer part number",
        ),
        ExtractionRule(
            field_name="cage_code",
            pattern=re.compile(r"CAGE:\s*(\d{5})"),
            source=ExtractionSource.CONTENT,
            required=False,
            description="Commercial and Government Entity code",
        ),
    ],
)

_REGULATION_PROFILE = MetadataProfile(
    name="regulation",
    document_type=DocumentType.REGULATION,
    description="Profile for regulations, directives, and policies",
    rules=[
        _REGULATION_NUMBER_RULE,
        _SECTION_NUMBER_RULE,
        _PARAGRAPH_REF_RULE,
        ExtractionRule(
            field_name="effective_date",
            pattern=re.compile(
                r"(?:effective|dated?)\s+" r"(\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4})",
                re.IGNORECASE,
            ),
            source=ExtractionSource.CONTENT,
            required=False,
            description="Effective or publication date",
        ),
    ],
)

_DEFAULT_PROFILE = MetadataProfile(
    name="default",
    document_type=DocumentType.UNKNOWN,
    description="Minimal fallback profile for unrecognized document types",
    rules=[
        _SECTION_NUMBER_RULE,
        _PARAGRAPH_REF_RULE,
        _FIGURE_REF_RULE,
        _TABLE_REF_RULE,
    ],
)


class MetadataProfileRegistry:
    """Registry mapping document types to metadata profiles.

    Comes pre-loaded with built-in profiles for common military
    document types. Custom profiles can be registered at runtime.

    Example::

        registry = MetadataProfileRegistry()
        profile = registry.get(DocumentType.TECHNICAL_MANUAL)
        print(profile.name)  # "technical_manual"
    """

    def __init__(self) -> None:
        self._profiles: dict[DocumentType, MetadataProfile] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register all built-in profiles."""
        for profile in [
            _TECHNICAL_MANUAL_PROFILE,
            _MAINTENANCE_PROCEDURE_PROFILE,
            _PARTS_CATALOG_PROFILE,
            _REGULATION_PROFILE,
        ]:
            self._profiles[profile.document_type] = profile

    def get(self, document_type: DocumentType) -> MetadataProfile:
        """Get the profile for a document type.

        Falls back to the default profile if no specific profile
        is registered for the given type.

        Args:
            document_type: The document type to look up.

        Returns:
            The matching MetadataProfile, or default if not found.
        """
        return self._profiles.get(document_type, _DEFAULT_PROFILE)

    def register(self, profile: MetadataProfile) -> None:
        """Register a custom profile.

        Args:
            profile: The profile to register. Overwrites any
                existing profile for the same document type.
        """
        self._profiles[profile.document_type] = profile

    @property
    def registered_types(self) -> list[DocumentType]:
        """Return list of document types with registered profiles."""
        return list(self._profiles.keys())
