"""Field validation for extracted metadata values.

Validates extracted metadata fields against known format patterns
to ensure data quality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class FieldValidationResult:
    """Result of validating a single extracted field.

    Args:
        field_name: Name of the validated field.
        is_valid: Whether the value passed validation.
        error_message: Description of validation failure, if any.
    """

    field_name: str
    is_valid: bool
    error_message: str = ""


# --- Compiled validation patterns ---

_NSN_RE = re.compile(r"^\d{4}-\d{2}-\d{3}-\d{4}$")
_TM_NUMBER_RE = re.compile(r"^TM\s+\d+-\d+(-\d+)*$", re.IGNORECASE)
_LIN_RE = re.compile(r"^[A-Z]\d{5}$")
_SMR_CODE_RE = re.compile(r"^[A-Z]{4,6}$")
_WORK_PACKAGE_RE = re.compile(r"^WP\s+\d{4}\s+\d{2}$", re.IGNORECASE)
_MAINTENANCE_LEVEL_RE = re.compile(
    r"^(organizational|direct support|general support|depot" r"|unit|intermediate|field)$",
    re.IGNORECASE,
)


# --- Validator registry ---

_VALIDATORS: dict[str, re.Pattern[str]] = {
    "nsn": _NSN_RE,
    "tm_number": _TM_NUMBER_RE,
    "lin": _LIN_RE,
    "smr_code": _SMR_CODE_RE,
    "work_package": _WORK_PACKAGE_RE,
    "maintenance_level": _MAINTENANCE_LEVEL_RE,
}


class FieldValidator:
    """Validates extracted metadata field values against format patterns.

    Uses a registry of compiled regex patterns keyed by field name.
    Fields without a registered validator always pass.

    Example::

        validator = FieldValidator()
        result = validator.validate("nsn", "2320-01-107-7155")
        assert result.is_valid
    """

    def __init__(self) -> None:
        self._validators = dict(_VALIDATORS)

    def validate(self, field_name: str, value: str) -> FieldValidationResult:
        """Validate a field value against its format pattern.

        Args:
            field_name: Name of the field to validate.
            value: Extracted value to check.

        Returns:
            FieldValidationResult indicating pass/fail.
        """
        pattern = self._validators.get(field_name)
        if pattern is None:
            return FieldValidationResult(
                field_name=field_name,
                is_valid=True,
            )
        if pattern.match(value.strip()):
            return FieldValidationResult(
                field_name=field_name,
                is_valid=True,
            )
        return FieldValidationResult(
            field_name=field_name,
            is_valid=False,
            error_message=(
                f"Value '{value}' does not match expected format " f"for field '{field_name}'"
            ),
        )

    def register(self, field_name: str, pattern: re.Pattern[str]) -> None:
        """Register a custom validator pattern.

        Args:
            field_name: Field name to validate.
            pattern: Compiled regex the value must match.
        """
        self._validators[field_name] = pattern

    @property
    def registered_fields(self) -> list[str]:
        """Return list of field names with registered validators."""
        return list(self._validators.keys())
