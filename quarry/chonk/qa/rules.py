"""Configurable filtering rules per document type.

Allows different document types to have different filtering
strategies (e.g., parts catalogs preserve index sections).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chonk.qa.filter_log import FilterCategory


@dataclass
class FilterRule:
    """A single configurable filtering rule.

    Attributes:
        name: Human-readable rule name.
        description: What this rule does.
        categories: Which filter categories this rule targets.
        enabled: Whether this rule is active.
    """

    name: str
    description: str
    categories: list[FilterCategory] = field(default_factory=list)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "categories": [c.value for c in self.categories],
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FilterRule:
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            categories=[FilterCategory(c) for c in data.get("categories", [])],
            enabled=data.get("enabled", True),
        )


@dataclass
class RuleSet:
    """Collection of rules, optionally document-type-specific.

    Attributes:
        name: Human-readable name for this rule set.
        document_type: Document type this applies to, or None for default.
        rules: List of filter rules.
    """

    name: str
    document_type: str | None = None
    rules: list[FilterRule] = field(default_factory=list)

    def get_active_categories(self) -> set[FilterCategory]:
        """Get all filter categories that are enabled.

        Returns:
            Set of active FilterCategory values.
        """
        categories: set[FilterCategory] = set()
        for rule in self.rules:
            if rule.enabled:
                categories.update(rule.categories)
        return categories

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "document_type": self.document_type,
            "rules": [r.to_dict() for r in self.rules],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuleSet:
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            document_type=data.get("document_type"),
            rules=[FilterRule.from_dict(r) for r in data.get("rules", [])],
        )

    @classmethod
    def default(cls) -> RuleSet:
        """Create the default rule set with all categories enabled.

        Returns:
            RuleSet with all filter categories active.
        """
        return cls(
            name="default",
            document_type=None,
            rules=[
                FilterRule(
                    name="toc_filter",
                    description="Remove table of contents entries",
                    categories=[FilterCategory.TOC],
                ),
                FilterRule(
                    name="index_filter",
                    description="Remove index sections",
                    categories=[FilterCategory.INDEX],
                ),
                FilterRule(
                    name="distribution_filter",
                    description="Remove distribution statements",
                    categories=[FilterCategory.DISTRIBUTION],
                ),
                FilterRule(
                    name="boilerplate_filter",
                    description="Remove boilerplate text",
                    categories=[FilterCategory.BOILERPLATE],
                ),
                FilterRule(
                    name="header_filter",
                    description="Remove repeated page headers",
                    categories=[FilterCategory.PAGE_HEADER],
                ),
                FilterRule(
                    name="footer_filter",
                    description="Remove repeated page footers",
                    categories=[FilterCategory.PAGE_FOOTER],
                ),
                FilterRule(
                    name="repetition_filter",
                    description="Remove near-duplicate blocks",
                    categories=[FilterCategory.REPETITION],
                ),
            ],
        )

    @classmethod
    def for_document_type(cls, document_type: str) -> RuleSet:
        """Create a rule set tailored to a specific document type.

        Args:
            document_type: Document type string (e.g., 'TECHNICAL_MANUAL').

        Returns:
            RuleSet with type-specific rule configuration.
        """
        base = cls.default()
        base.document_type = document_type

        if document_type.upper() in (
            "PARTS_CATALOG",
            "PARTS_LIST",
        ):
            base.name = "parts_catalog"
            for rule in base.rules:
                if FilterCategory.INDEX in rule.categories:
                    rule.enabled = False
        elif document_type.upper() in (
            "TECHNICAL_MANUAL",
            "MAINTENANCE_MANUAL",
        ):
            base.name = "technical_manual"

        return base
