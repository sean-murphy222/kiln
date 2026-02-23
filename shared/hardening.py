"""Production hardening utilities for the Kiln pipeline.

Provides retry logic, resource monitoring, user-friendly error formatting,
input validation with path traversal prevention, and system health checking.
All utilities are designed for local-first deployment on resource-constrained
machines (laptops/workstations).
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Retry Logic
# ---------------------------------------------------------------------------

_DEFAULT_RETRYABLE = (IOError, OSError, TimeoutError, ConnectionError)


@dataclass
class RetryConfig:
    """Configuration for retry-with-backoff behavior.

    Attributes:
        max_attempts: Total number of attempts (including the first).
        base_delay: Initial delay in seconds before first retry.
        max_delay: Upper bound on delay between retries.
        exponential_backoff: Double delay on each retry when True.
        retryable_exceptions: Tuple of exception types that trigger a retry.
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_backoff: bool = True
    retryable_exceptions: tuple[type[BaseException], ...] = _DEFAULT_RETRYABLE


class RetriesExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted.

    Attributes:
        last_error: The final exception that caused the failure.
        attempts: Total number of attempts made.
    """

    def __init__(self, last_error: Exception, attempts: int) -> None:
        self.last_error = last_error
        self.attempts = attempts
        super().__init__(f"All {attempts} attempts failed. Last error: {last_error}")


def _compute_delay(attempt: int, config: RetryConfig) -> float:
    """Compute the delay before the next retry attempt.

    Args:
        attempt: Zero-based attempt index (0 = first retry).
        config: Retry configuration.

    Returns:
        Delay in seconds, capped at config.max_delay.
    """
    if config.exponential_backoff:
        delay = config.base_delay * (2**attempt)
    else:
        delay = config.base_delay
    return min(delay, config.max_delay)


def retry_with_backoff(
    func: Callable[..., Any],
    config: RetryConfig | None = None,
    *args: Any,
    sleep_func: Callable[[float], None] | None = None,
    **kwargs: Any,
) -> Any:
    """Execute *func* with exponential-backoff retry on transient failures.

    Args:
        func: Callable to invoke.
        config: Retry configuration. Uses defaults when None.
        *args: Positional arguments forwarded to *func*.
        sleep_func: Injectable sleep for testing. Defaults to time.sleep.
        **kwargs: Keyword arguments forwarded to *func*.

    Returns:
        Whatever *func* returns on success.

    Raises:
        RetriesExhaustedError: When all attempts fail with retryable errors.
        Exception: Immediately re-raised for non-retryable errors.
    """
    cfg = config or RetryConfig()
    do_sleep = sleep_func or time.sleep
    last_error: Exception | None = None

    for attempt in range(cfg.max_attempts):
        try:
            return func(*args, **kwargs)
        except cfg.retryable_exceptions as exc:
            last_error = exc
            if attempt < cfg.max_attempts - 1:
                delay = _compute_delay(attempt, cfg)
                logger.warning(
                    "Attempt %d/%d failed (%s). Retrying in %.1fs.",
                    attempt + 1,
                    cfg.max_attempts,
                    exc,
                    delay,
                )
                do_sleep(delay)
        except Exception:
            raise

    raise RetriesExhaustedError(last_error, cfg.max_attempts)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. Resource Limiter
# ---------------------------------------------------------------------------


@dataclass
class ResourceLimits:
    """Thresholds for system resource usage.

    Attributes:
        max_memory_mb: Maximum process memory in megabytes.
        max_file_size_mb: Maximum individual file size in megabytes.
        max_concurrent_operations: Parallelism cap.
        operation_timeout_seconds: Per-operation wall-clock limit.
    """

    max_memory_mb: int = 4096
    max_file_size_mb: int = 500
    max_concurrent_operations: int = 4
    operation_timeout_seconds: int = 300


class ResourceLimitExceededError(Exception):
    """Raised when a resource limit check fails."""


class ResourceMonitor:
    """Track and enforce resource limits.

    Args:
        limits: Configuration thresholds.
    """

    def __init__(self, limits: ResourceLimits | None = None) -> None:
        self.limits = limits or ResourceLimits()

    def check_memory(self) -> bool:
        """Return True when process memory is within limits.

        Uses os-level RSS estimation. Returns True when the platform
        does not expose memory info (conservative: assume OK).
        """
        try:
            rss_mb = _get_process_rss_mb()
        except RuntimeError:
            return True
        return rss_mb <= self.limits.max_memory_mb

    def check_file_size(self, path: Path) -> bool:
        """Return True when the file at *path* is within limits.

        Args:
            path: Path to an existing file.

        Returns:
            True if file size is acceptable.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        size_mb = resolved.stat().st_size / (1024 * 1024)
        return size_mb <= self.limits.max_file_size_mb

    def get_usage_report(self) -> dict[str, Any]:
        """Return a snapshot of current resource usage.

        Returns:
            Dictionary with keys: memory_mb, memory_limit_mb,
            memory_ok, file_size_limit_mb.
        """
        try:
            rss = _get_process_rss_mb()
        except RuntimeError:
            rss = -1.0
        return {
            "memory_mb": round(rss, 1),
            "memory_limit_mb": self.limits.max_memory_mb,
            "memory_ok": rss <= self.limits.max_memory_mb if rss >= 0 else True,
            "file_size_limit_mb": self.limits.max_file_size_mb,
            "max_concurrent_operations": self.limits.max_concurrent_operations,
            "operation_timeout_seconds": self.limits.operation_timeout_seconds,
        }


def _get_process_rss_mb() -> float:
    """Return resident set size of current process in MB.

    Raises:
        RuntimeError: When memory info is unavailable.
    """
    try:
        import psutil  # type: ignore[import-untyped]

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    # Fallback for platforms without psutil: /proc on Linux
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except (FileNotFoundError, PermissionError):
        pass
    raise RuntimeError("Cannot determine process memory usage")


# ---------------------------------------------------------------------------
# 3. Error Formatting
# ---------------------------------------------------------------------------


@dataclass
class UserFriendlyError:
    """A structured error designed for end-user consumption.

    Attributes:
        message: Clear description for the user.
        suggestion: Actionable guidance.
        component: Originating subsystem (quarry, forge, foundry, hearth).
        error_code: Machine-readable identifier (e.g. "TRAIN_001").
        technical_detail: Debugging info for logs only -- never shown to users.
    """

    message: str
    suggestion: str
    component: str
    error_code: str
    technical_detail: str = ""

    def to_dict(self) -> dict[str, str]:
        """Serialize for API responses (excludes technical_detail).

        Returns:
            Dictionary safe for sending to end users.
        """
        return {
            "message": self.message,
            "suggestion": self.suggestion,
            "component": self.component,
            "error_code": self.error_code,
        }


class ErrorFormatter:
    """Convert internal exceptions to user-friendly messages.

    All methods return a ``UserFriendlyError`` and never expose internal
    paths, stack traces, or implementation details to the end user.
    """

    def format_training_error(self, error: Exception) -> UserFriendlyError:
        """Format a training-pipeline error.

        Args:
            error: The caught exception.

        Returns:
            User-friendly error with actionable suggestion.
        """
        return self._format(
            error,
            component="foundry",
            code_prefix="TRAIN",
        )

    def format_evaluation_error(self, error: Exception) -> UserFriendlyError:
        """Format an evaluation-pipeline error.

        Args:
            error: The caught exception.

        Returns:
            User-friendly error with actionable suggestion.
        """
        return self._format(
            error,
            component="foundry",
            code_prefix="EVAL",
        )

    def format_retrieval_error(self, error: Exception) -> UserFriendlyError:
        """Format a retrieval-pipeline error.

        Args:
            error: The caught exception.

        Returns:
            User-friendly error with actionable suggestion.
        """
        return self._format(
            error,
            component="quarry",
            code_prefix="RETR",
        )

    def format_storage_error(self, error: Exception) -> UserFriendlyError:
        """Format a data-storage error.

        Args:
            error: The caught exception.

        Returns:
            User-friendly error with actionable suggestion.
        """
        return self._format(
            error,
            component="forge",
            code_prefix="STOR",
        )

    # ------------------------------------------------------------------

    def _format(
        self,
        error: Exception,
        *,
        component: str,
        code_prefix: str,
    ) -> UserFriendlyError:
        """Shared formatting logic.

        Args:
            error: The caught exception.
            component: Subsystem name.
            code_prefix: Short prefix for error code.

        Returns:
            Structured error with safe user message.
        """
        message, suggestion, code_suffix = _classify_error(error)
        return UserFriendlyError(
            message=message,
            suggestion=suggestion,
            component=component,
            error_code=f"{code_prefix}_{code_suffix}",
            technical_detail=repr(error),
        )


def _classify_error(error: Exception) -> tuple[str, str, str]:
    """Map an exception to (message, suggestion, code_suffix).

    Args:
        error: The caught exception.

    Returns:
        Tuple of user message, suggestion text, and error code suffix.
    """
    if isinstance(error, FileNotFoundError):
        return (
            "A required file could not be found.",
            "Check that the file path is correct and the file exists.",
            "001",
        )
    if isinstance(error, PermissionError):
        return (
            "Permission denied when accessing a resource.",
            "Check file permissions and ensure the application has access.",
            "002",
        )
    if isinstance(error, (TimeoutError, ConnectionError)):
        return (
            "The operation timed out or lost its connection.",
            "Try again. If the problem persists, check system resources.",
            "003",
        )
    if isinstance(error, MemoryError):
        return (
            "The system ran out of memory.",
            "Close other applications or reduce the batch size.",
            "004",
        )
    if isinstance(error, json.JSONDecodeError):
        return (
            "A data file contains invalid JSON.",
            "Verify the file format is valid JSON or JSONL.",
            "006",
        )
    if isinstance(error, ValueError):
        return (
            "Invalid input was provided.",
            "Check the input values and try again.",
            "005",
        )
    return (
        "An unexpected error occurred.",
        "If this keeps happening, please report the issue.",
        "999",
    )


# ---------------------------------------------------------------------------
# 4. Input Validation
# ---------------------------------------------------------------------------

# Characters that could be used for path traversal
_TRAVERSAL_PATTERN = re.compile(r"(\.\.[\\/]|[\\/]\.\.)")
# Null bytes in paths
_NULL_BYTE = re.compile(r"\x00")


class ValidationError(Exception):
    """Raised when input validation fails."""


class InputValidator:
    """Validate inputs at system boundaries.

    All methods raise ``ValidationError`` on failure unless
    documented otherwise.
    """

    def validate_file_path(
        self,
        path: str | Path,
        *,
        must_exist: bool = True,
        allowed_extensions: tuple[str, ...] | None = None,
        base_directory: Path | None = None,
    ) -> Path:
        """Validate a file path, preventing traversal attacks.

        Args:
            path: Raw path from user input.
            must_exist: Require the file to exist on disk.
            allowed_extensions: Restrict to these suffixes (e.g. (".pdf",)).
            base_directory: Confine resolved path under this directory.

        Returns:
            Resolved, validated Path.

        Raises:
            ValidationError: On any validation failure.
        """
        raw = str(path)
        self._check_traversal(raw)
        resolved = Path(raw).resolve()

        if base_directory is not None:
            base = base_directory.resolve()
            if not _is_subpath(resolved, base):
                raise ValidationError("Path is outside the allowed directory.")

        if allowed_extensions is not None:
            if resolved.suffix.lower() not in {e.lower() for e in allowed_extensions}:
                allowed = ", ".join(allowed_extensions)
                raise ValidationError(f"File type not allowed. Accepted types: {allowed}")

        if must_exist and not resolved.exists():
            raise ValidationError("File does not exist.")

        return resolved

    def validate_jsonl_file(
        self,
        path: str | Path,
        *,
        max_records: int = 100_000,
    ) -> list[dict[str, Any]]:
        """Read and validate a JSONL file.

        Args:
            path: Path to the JSONL file.
            max_records: Safety cap on number of records.

        Returns:
            List of parsed JSON objects.

        Raises:
            ValidationError: On format or size errors.
        """
        validated_path = self.validate_file_path(
            path, must_exist=True, allowed_extensions=(".jsonl",)
        )
        records: list[dict[str, Any]] = []
        try:
            with open(validated_path, encoding="utf-8") as fh:
                for line_num, line in enumerate(fh, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if len(records) >= max_records:
                        raise ValidationError(f"File exceeds the maximum of {max_records} records.")
                    try:
                        obj = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise ValidationError(f"Invalid JSON on line {line_num}.") from exc
                    if not isinstance(obj, dict):
                        raise ValidationError(f"Line {line_num} is not a JSON object.")
                    records.append(obj)
        except UnicodeDecodeError as exc:
            raise ValidationError("File is not valid UTF-8 text.") from exc
        return records

    def validate_curriculum_format(self, records: list[dict[str, Any]]) -> list[str]:
        """Check that curriculum records have required fields.

        Args:
            records: List of parsed JSON objects (from JSONL).

        Returns:
            List of error strings. Empty list means valid.
        """
        required_fields = {"question", "ideal_answer", "competency_id"}
        errors: list[str] = []
        for idx, record in enumerate(records):
            missing = required_fields - set(record.keys())
            if missing:
                errors.append(f"Record {idx + 1}: missing fields {sorted(missing)}")
            for fld in ("question", "ideal_answer"):
                val = record.get(fld, "")
                if isinstance(val, str) and not val.strip():
                    errors.append(f"Record {idx + 1}: '{fld}' is empty.")
        return errors

    def validate_model_path(self, path: str | Path) -> bool:
        """Check that a model directory or file exists and is non-empty.

        Args:
            path: Path to model weights or directory.

        Returns:
            True when the path is a non-empty file or non-empty directory.
        """
        resolved = Path(path).resolve()
        if resolved.is_file():
            return resolved.stat().st_size > 0
        if resolved.is_dir():
            return any(resolved.iterdir())
        return False

    def sanitize_string(
        self,
        value: str,
        *,
        max_length: int = 1000,
    ) -> str:
        """Sanitize a user-provided string.

        Strips HTML entities, control characters, and truncates.

        Args:
            value: Raw user string.
            max_length: Maximum allowed length after sanitization.

        Returns:
            Cleaned string.
        """
        cleaned = html.escape(value, quote=True)
        cleaned = _strip_control_chars(cleaned)
        cleaned = cleaned.strip()
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]
        return cleaned

    # ------------------------------------------------------------------

    @staticmethod
    def _check_traversal(raw: str) -> None:
        """Reject paths with traversal sequences or null bytes.

        Args:
            raw: Raw path string.

        Raises:
            ValidationError: On dangerous patterns.
        """
        if _NULL_BYTE.search(raw):
            raise ValidationError("Path contains null bytes.")
        if _TRAVERSAL_PATTERN.search(raw):
            raise ValidationError("Path traversal is not allowed.")


def _is_subpath(child: Path, parent: Path) -> bool:
    """Return True when *child* is under *parent*.

    Args:
        child: Resolved candidate path.
        parent: Resolved base directory.

    Returns:
        True if child is equal to or nested inside parent.
    """
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _strip_control_chars(text: str) -> str:
    """Remove ASCII control characters except common whitespace.

    Args:
        text: Input string.

    Returns:
        Cleaned string.
    """
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


# ---------------------------------------------------------------------------
# 5. Graceful Degradation / Health Checks
# ---------------------------------------------------------------------------


@dataclass
class HealthCheck:
    """Result of a single component health check.

    Attributes:
        component: Subsystem name (quarry, forge, foundry).
        status: One of "healthy", "degraded", "unavailable".
        message: Human-readable description.
        checked_at: UTC timestamp of the check.
    """

    component: str
    status: str
    message: str
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, str]:
        """Serialize for API responses.

        Returns:
            Dictionary with component, status, message, checked_at.
        """
        return {
            "component": self.component,
            "status": self.status,
            "message": self.message,
            "checked_at": self.checked_at.isoformat(),
        }


class SystemHealthChecker:
    """Check health of all Kiln subsystems.

    Each check returns a ``HealthCheck`` with status:
      - ``healthy``: Fully operational.
      - ``degraded``: Partially working (missing optional deps).
      - ``unavailable``: Cannot function.

    Args:
        quarry_path: Root path of the quarry package.
        forge_path: Root path of the forge package.
        foundry_path: Root path of the foundry package.
    """

    def __init__(
        self,
        quarry_path: Path | None = None,
        forge_path: Path | None = None,
        foundry_path: Path | None = None,
    ) -> None:
        self.quarry_path = quarry_path
        self.forge_path = forge_path
        self.foundry_path = foundry_path

    def check_quarry(self) -> HealthCheck:
        """Check quarry subsystem health.

        Returns:
            HealthCheck for the quarry component.
        """
        return self._check_component(
            "quarry",
            self.quarry_path,
            optional_import="quarry.chonk",
        )

    def check_forge(self) -> HealthCheck:
        """Check forge subsystem health.

        Returns:
            HealthCheck for the forge component.
        """
        return self._check_component(
            "forge",
            self.forge_path,
            optional_import="forge.src",
        )

    def check_foundry(self) -> HealthCheck:
        """Check foundry subsystem health.

        Returns:
            HealthCheck for the foundry component.
        """
        return self._check_component(
            "foundry",
            self.foundry_path,
            optional_import="foundry.src",
        )

    def full_check(self) -> list[HealthCheck]:
        """Run health checks for all components.

        Returns:
            List of HealthCheck results.
        """
        return [
            self.check_quarry(),
            self.check_forge(),
            self.check_foundry(),
        ]

    # ------------------------------------------------------------------

    def _check_component(
        self,
        name: str,
        path: Path | None,
        *,
        optional_import: str,
    ) -> HealthCheck:
        """Shared component check logic.

        Args:
            name: Component name.
            path: Expected directory path.
            optional_import: Module to attempt importing.

        Returns:
            HealthCheck result.
        """
        if path is not None and not path.is_dir():
            return HealthCheck(
                component=name,
                status="unavailable",
                message=f"{name} directory not found.",
            )

        importable = _try_import(optional_import)
        if path is not None and path.is_dir() and not importable:
            return HealthCheck(
                component=name,
                status="degraded",
                message=f"{name} directory exists but module import failed.",
            )

        if path is None and not importable:
            return HealthCheck(
                component=name,
                status="unavailable",
                message=f"{name} is not configured.",
            )

        return HealthCheck(
            component=name,
            status="healthy",
            message=f"{name} is operational.",
        )


def _try_import(module_name: str) -> bool:
    """Attempt to import *module_name* without side effects.

    Args:
        module_name: Dotted module path.

    Returns:
        True on success, False on ImportError.
    """
    import importlib

    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False
