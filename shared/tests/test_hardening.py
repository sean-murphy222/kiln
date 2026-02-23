"""Tests for shared.hardening production hardening utilities.

Covers all five components: retry logic, resource monitoring,
error formatting, input validation, and system health checking.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from shared.hardening import (
    ErrorFormatter,
    HealthCheck,
    InputValidator,
    ResourceLimits,
    ResourceMonitor,
    RetriesExhaustedError,
    RetryConfig,
    SystemHealthChecker,
    UserFriendlyError,
    ValidationError,
    _classify_error,
    _compute_delay,
    _is_subpath,
    _strip_control_chars,
    retry_with_backoff,
)

# =========================================================================
# 1. Retry Logic
# =========================================================================


class TestRetryConfig:
    """Tests for RetryConfig dataclass defaults and overrides."""

    def test_default_values(self) -> None:
        """Default config has sensible production values."""
        cfg = RetryConfig()
        assert cfg.max_attempts == 3
        assert cfg.base_delay == 1.0
        assert cfg.max_delay == 30.0
        assert cfg.exponential_backoff is True
        assert OSError in cfg.retryable_exceptions

    def test_custom_values(self) -> None:
        """Overriding defaults works correctly."""
        cfg = RetryConfig(
            max_attempts=5,
            base_delay=0.5,
            max_delay=10.0,
            exponential_backoff=False,
            retryable_exceptions=(ValueError,),
        )
        assert cfg.max_attempts == 5
        assert cfg.base_delay == 0.5
        assert cfg.max_delay == 10.0
        assert cfg.exponential_backoff is False
        assert cfg.retryable_exceptions == (ValueError,)


class TestComputeDelay:
    """Tests for the delay computation helper."""

    def test_exponential_backoff_attempt_0(self) -> None:
        """First retry delay equals base_delay."""
        cfg = RetryConfig(base_delay=1.0, exponential_backoff=True)
        assert _compute_delay(0, cfg) == 1.0

    def test_exponential_backoff_attempt_1(self) -> None:
        """Second retry doubles the delay."""
        cfg = RetryConfig(base_delay=1.0, exponential_backoff=True)
        assert _compute_delay(1, cfg) == 2.0

    def test_exponential_backoff_attempt_4(self) -> None:
        """Fourth retry is 2^4 = 16x base."""
        cfg = RetryConfig(base_delay=1.0, max_delay=30.0, exponential_backoff=True)
        assert _compute_delay(4, cfg) == 16.0

    def test_delay_capped_at_max(self) -> None:
        """Delay never exceeds max_delay."""
        cfg = RetryConfig(base_delay=1.0, max_delay=5.0, exponential_backoff=True)
        assert _compute_delay(10, cfg) == 5.0

    def test_linear_backoff(self) -> None:
        """Without exponential, delay stays constant."""
        cfg = RetryConfig(base_delay=2.0, exponential_backoff=False)
        assert _compute_delay(0, cfg) == 2.0
        assert _compute_delay(5, cfg) == 2.0

    def test_linear_capped_at_max(self) -> None:
        """Linear delay respects max_delay when base exceeds it."""
        cfg = RetryConfig(base_delay=50.0, max_delay=10.0, exponential_backoff=False)
        assert _compute_delay(0, cfg) == 10.0


class TestRetryWithBackoff:
    """Tests for the retry_with_backoff function."""

    def test_succeeds_on_first_attempt(self) -> None:
        """No retries needed when func succeeds immediately."""
        result = retry_with_backoff(lambda: 42, sleep_func=lambda _: None)
        assert result == 42

    def test_succeeds_after_transient_failure(self) -> None:
        """Retries on OSError and eventually succeeds."""
        call_count = 0

        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("transient")
            return "ok"

        result = retry_with_backoff(flaky, sleep_func=lambda _: None)
        assert result == "ok"
        assert call_count == 3

    def test_raises_retries_exhausted(self) -> None:
        """Raises RetriesExhaustedError when all attempts fail."""
        cfg = RetryConfig(max_attempts=2)

        with pytest.raises(RetriesExhaustedError) as exc_info:
            retry_with_backoff(
                lambda: (_ for _ in ()).throw(OSError("fail")),
                config=cfg,
                sleep_func=lambda _: None,
            )

        assert exc_info.value.attempts == 2
        assert isinstance(exc_info.value.last_error, OSError)

    def test_non_retryable_exception_raised_immediately(self) -> None:
        """ValueError is not retryable by default and raises at once."""
        call_count = 0

        def bad() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            retry_with_backoff(bad, sleep_func=lambda _: None)

        assert call_count == 1

    def test_custom_retryable_exceptions(self) -> None:
        """Custom retryable exceptions are retried."""
        cfg = RetryConfig(
            max_attempts=3,
            retryable_exceptions=(ValueError,),
        )
        call_count = 0

        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("retry me")
            return "done"

        result = retry_with_backoff(flaky, config=cfg, sleep_func=lambda _: None)
        assert result == "done"
        assert call_count == 2

    def test_sleep_called_between_retries(self) -> None:
        """Sleep function is invoked with correct delays."""
        delays: list[float] = []
        cfg = RetryConfig(max_attempts=3, base_delay=1.0)
        call_count = 0

        def always_fail() -> None:
            nonlocal call_count
            call_count += 1
            raise OSError("fail")

        with pytest.raises(RetriesExhaustedError):
            retry_with_backoff(
                always_fail,
                config=cfg,
                sleep_func=lambda d: delays.append(d),
            )

        # Two sleeps between 3 attempts: 1.0, 2.0
        assert len(delays) == 2
        assert delays[0] == 1.0
        assert delays[1] == 2.0

    def test_single_attempt_no_retry(self) -> None:
        """With max_attempts=1, failure raises immediately."""
        cfg = RetryConfig(max_attempts=1)

        with pytest.raises(RetriesExhaustedError) as exc_info:
            retry_with_backoff(
                lambda: (_ for _ in ()).throw(OSError("once")),
                config=cfg,
                sleep_func=lambda _: None,
            )

        assert exc_info.value.attempts == 1

    def test_passes_args_and_kwargs(self) -> None:
        """Positional and keyword arguments are forwarded."""

        def add(a: int, b: int, extra: int = 0) -> int:
            return a + b + extra

        result = retry_with_backoff(add, None, 2, 3, sleep_func=lambda _: None, extra=10)
        assert result == 15

    def test_retries_exhausted_str(self) -> None:
        """RetriesExhaustedError has a helpful string representation."""
        err = RetriesExhaustedError(OSError("disk full"), 3)
        assert "3 attempts" in str(err)
        assert "disk full" in str(err)


# =========================================================================
# 2. Resource Limiter
# =========================================================================


class TestResourceLimits:
    """Tests for ResourceLimits defaults."""

    def test_defaults(self) -> None:
        """Default limits are reasonable for local deployment."""
        lim = ResourceLimits()
        assert lim.max_memory_mb == 4096
        assert lim.max_file_size_mb == 500
        assert lim.max_concurrent_operations == 4
        assert lim.operation_timeout_seconds == 300

    def test_custom_limits(self) -> None:
        """Custom limits override correctly."""
        lim = ResourceLimits(max_memory_mb=2048, max_file_size_mb=100)
        assert lim.max_memory_mb == 2048
        assert lim.max_file_size_mb == 100


class TestResourceMonitor:
    """Tests for ResourceMonitor checks."""

    def test_check_memory_within_limits(self) -> None:
        """Memory check passes when RSS is below limit."""
        monitor = ResourceMonitor(ResourceLimits(max_memory_mb=99999))
        # Should pass -- actual RSS is way under 99999 MB
        assert monitor.check_memory() is True

    def test_check_memory_fallback_returns_true(self) -> None:
        """When memory cannot be determined, returns True (conservative)."""
        monitor = ResourceMonitor()
        with patch("shared.hardening._get_process_rss_mb", side_effect=RuntimeError):
            assert monitor.check_memory() is True

    def test_check_memory_over_limit(self) -> None:
        """Memory check fails when RSS exceeds limit."""
        monitor = ResourceMonitor(ResourceLimits(max_memory_mb=1))
        with patch("shared.hardening._get_process_rss_mb", return_value=500.0):
            assert monitor.check_memory() is False

    def test_check_file_size_ok(self, tmp_path: Path) -> None:
        """File within size limit passes."""
        small_file = tmp_path / "small.txt"
        small_file.write_text("hello")
        monitor = ResourceMonitor(ResourceLimits(max_file_size_mb=1))
        assert monitor.check_file_size(small_file) is True

    def test_check_file_size_too_large(self, tmp_path: Path) -> None:
        """File exceeding limit fails."""
        big_file = tmp_path / "big.bin"
        # Write 2 MB
        big_file.write_bytes(b"\x00" * (2 * 1024 * 1024))
        monitor = ResourceMonitor(ResourceLimits(max_file_size_mb=1))
        assert monitor.check_file_size(big_file) is False

    def test_check_file_not_found(self, tmp_path: Path) -> None:
        """Missing file raises FileNotFoundError."""
        monitor = ResourceMonitor()
        with pytest.raises(FileNotFoundError):
            monitor.check_file_size(tmp_path / "ghost.txt")

    def test_get_usage_report_keys(self) -> None:
        """Usage report contains all expected keys."""
        monitor = ResourceMonitor()
        report = monitor.get_usage_report()
        assert "memory_mb" in report
        assert "memory_limit_mb" in report
        assert "memory_ok" in report
        assert "file_size_limit_mb" in report
        assert "max_concurrent_operations" in report
        assert "operation_timeout_seconds" in report

    def test_get_usage_report_with_fallback(self) -> None:
        """Usage report when memory is unavailable shows -1."""
        monitor = ResourceMonitor()
        with patch("shared.hardening._get_process_rss_mb", side_effect=RuntimeError):
            report = monitor.get_usage_report()
        assert report["memory_mb"] == -1.0
        assert report["memory_ok"] is True


# =========================================================================
# 3. Error Formatting
# =========================================================================


class TestUserFriendlyError:
    """Tests for UserFriendlyError dataclass."""

    def test_to_dict_excludes_technical_detail(self) -> None:
        """to_dict never includes technical_detail."""
        err = UserFriendlyError(
            message="Something failed.",
            suggestion="Try again.",
            component="foundry",
            error_code="TRAIN_001",
            technical_detail="Traceback at line 42 in /secret/path.py",
        )
        d = err.to_dict()
        assert "technical_detail" not in d
        assert d["message"] == "Something failed."
        assert d["error_code"] == "TRAIN_001"

    def test_to_dict_has_required_keys(self) -> None:
        """All user-facing keys are present."""
        err = UserFriendlyError(
            message="msg", suggestion="sug", component="forge", error_code="STOR_001"
        )
        d = err.to_dict()
        assert set(d.keys()) == {"message", "suggestion", "component", "error_code"}


class TestClassifyError:
    """Tests for the internal _classify_error helper."""

    @pytest.mark.parametrize(
        "exc_class,expected_suffix",
        [
            (FileNotFoundError, "001"),
            (PermissionError, "002"),
            (TimeoutError, "003"),
            (ConnectionError, "003"),
            (MemoryError, "004"),
            (ValueError, "005"),
        ],
    )
    def test_known_exception_types(self, exc_class: type, expected_suffix: str) -> None:
        """Each known exception maps to a specific code suffix."""
        _msg, _sug, suffix = _classify_error(exc_class("test"))
        assert suffix == expected_suffix

    def test_json_decode_error(self) -> None:
        """JSONDecodeError maps to 006."""
        exc = json.JSONDecodeError("bad", "doc", 0)
        _, _, suffix = _classify_error(exc)
        assert suffix == "006"

    def test_unknown_exception(self) -> None:
        """Unexpected exceptions get code 999."""
        _, _, suffix = _classify_error(RuntimeError("surprise"))
        assert suffix == "999"


class TestErrorFormatter:
    """Tests for ErrorFormatter methods."""

    def setup_method(self) -> None:
        """Create a formatter for each test."""
        self.fmt = ErrorFormatter()

    def test_format_training_error_file_not_found(self) -> None:
        """Training error for FileNotFoundError produces clear message."""
        result = self.fmt.format_training_error(FileNotFoundError("x"))
        assert result.component == "foundry"
        assert result.error_code == "TRAIN_001"
        assert "file" in result.message.lower()
        assert "path" in result.suggestion.lower() or "exist" in result.suggestion.lower()

    def test_format_evaluation_error(self) -> None:
        """Evaluation error has correct component and prefix."""
        result = self.fmt.format_evaluation_error(MemoryError())
        assert result.component == "foundry"
        assert result.error_code.startswith("EVAL_")
        assert "memory" in result.message.lower()

    def test_format_retrieval_error(self) -> None:
        """Retrieval error routes to quarry component."""
        result = self.fmt.format_retrieval_error(TimeoutError())
        assert result.component == "quarry"
        assert result.error_code.startswith("RETR_")

    def test_format_storage_error(self) -> None:
        """Storage error routes to forge component."""
        result = self.fmt.format_storage_error(PermissionError("denied"))
        assert result.component == "forge"
        assert result.error_code.startswith("STOR_")

    def test_technical_detail_contains_repr(self) -> None:
        """technical_detail preserves the original exception repr."""
        err = ValueError("bad value 42")
        result = self.fmt.format_training_error(err)
        assert "bad value 42" in result.technical_detail

    def test_user_message_has_no_internal_paths(self) -> None:
        """User-facing message does not contain file system paths."""
        err = FileNotFoundError("/secret/internal/path/model.bin")
        result = self.fmt.format_training_error(err)
        assert "/secret" not in result.message
        assert "/secret" not in result.suggestion

    def test_all_formatters_return_user_friendly_error(self) -> None:
        """Every formatter method returns a UserFriendlyError instance."""
        exc = RuntimeError("test")
        for method in (
            self.fmt.format_training_error,
            self.fmt.format_evaluation_error,
            self.fmt.format_retrieval_error,
            self.fmt.format_storage_error,
        ):
            result = method(exc)
            assert isinstance(result, UserFriendlyError)


# =========================================================================
# 4. Input Validation
# =========================================================================


class TestInputValidator:
    """Tests for InputValidator methods."""

    def setup_method(self) -> None:
        """Create a validator for each test."""
        self.validator = InputValidator()

    # -- validate_file_path --

    def test_valid_existing_file(self, tmp_path: Path) -> None:
        """A normal existing file passes validation."""
        f = tmp_path / "data.txt"
        f.write_text("content")
        result = self.validator.validate_file_path(f)
        assert result.is_file()

    def test_path_traversal_rejected(self) -> None:
        """Paths containing '..' sequences are rejected."""
        with pytest.raises(ValidationError, match="traversal"):
            self.validator.validate_file_path("../../etc/passwd", must_exist=False)

    def test_path_traversal_backslash(self) -> None:
        """Backslash traversal is also rejected."""
        with pytest.raises(ValidationError, match="traversal"):
            self.validator.validate_file_path("..\\..\\windows\\system32", must_exist=False)

    def test_null_byte_rejected(self) -> None:
        """Null bytes in paths are rejected."""
        with pytest.raises(ValidationError, match="null"):
            self.validator.validate_file_path("file\x00.txt", must_exist=False)

    def test_must_exist_fails_for_missing(self, tmp_path: Path) -> None:
        """must_exist=True raises when file is absent."""
        with pytest.raises(ValidationError, match="does not exist"):
            self.validator.validate_file_path(tmp_path / "nope.txt")

    def test_must_exist_false_allows_missing(self, tmp_path: Path) -> None:
        """must_exist=False allows non-existent path."""
        result = self.validator.validate_file_path(tmp_path / "future.txt", must_exist=False)
        assert not result.exists()

    def test_allowed_extensions_pass(self, tmp_path: Path) -> None:
        """File with allowed extension passes."""
        f = tmp_path / "data.pdf"
        f.write_text("fake pdf")
        result = self.validator.validate_file_path(f, allowed_extensions=(".pdf", ".txt"))
        assert result.suffix == ".pdf"

    def test_allowed_extensions_fail(self, tmp_path: Path) -> None:
        """File with wrong extension is rejected."""
        f = tmp_path / "data.exe"
        f.write_text("bad")
        with pytest.raises(ValidationError, match="not allowed"):
            self.validator.validate_file_path(f, allowed_extensions=(".pdf",))

    def test_base_directory_containment(self, tmp_path: Path) -> None:
        """File inside base_directory passes."""
        f = tmp_path / "sub" / "data.txt"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("ok")
        result = self.validator.validate_file_path(f, base_directory=tmp_path)
        assert result.is_file()

    def test_base_directory_escape_rejected(self, tmp_path: Path) -> None:
        """File outside base_directory is rejected."""
        outside = tmp_path.parent / "escape.txt"
        # Create a file that actually exists at a sibling location
        try:
            outside.write_text("escaped")
            with pytest.raises(ValidationError, match="outside"):
                self.validator.validate_file_path(outside, base_directory=tmp_path)
        finally:
            if outside.exists():
                outside.unlink()

    # -- validate_jsonl_file --

    def test_valid_jsonl_file(self, tmp_path: Path) -> None:
        """Well-formed JSONL parses correctly."""
        f = tmp_path / "data.jsonl"
        lines = [
            json.dumps({"question": "Q1", "ideal_answer": "A1"}),
            json.dumps({"question": "Q2", "ideal_answer": "A2"}),
        ]
        f.write_text("\n".join(lines))
        records = self.validator.validate_jsonl_file(f)
        assert len(records) == 2
        assert records[0]["question"] == "Q1"

    def test_jsonl_skips_blank_lines(self, tmp_path: Path) -> None:
        """Blank lines in JSONL are ignored."""
        f = tmp_path / "data.jsonl"
        f.write_text('{"a": 1}\n\n{"b": 2}\n')
        records = self.validator.validate_jsonl_file(f)
        assert len(records) == 2

    def test_jsonl_invalid_json_raises(self, tmp_path: Path) -> None:
        """Malformed JSON line raises ValidationError."""
        f = tmp_path / "bad.jsonl"
        f.write_text('{"ok": 1}\nnot json\n')
        with pytest.raises(ValidationError, match="Invalid JSON on line 2"):
            self.validator.validate_jsonl_file(f)

    def test_jsonl_non_object_raises(self, tmp_path: Path) -> None:
        """A JSON array line raises ValidationError."""
        f = tmp_path / "arr.jsonl"
        f.write_text("[1, 2, 3]\n")
        with pytest.raises(ValidationError, match="not a JSON object"):
            self.validator.validate_jsonl_file(f)

    def test_jsonl_wrong_extension(self, tmp_path: Path) -> None:
        """Non-.jsonl extension is rejected."""
        f = tmp_path / "data.json"
        f.write_text('{"a": 1}\n')
        with pytest.raises(ValidationError, match="not allowed"):
            self.validator.validate_jsonl_file(f)

    def test_jsonl_max_records_exceeded(self, tmp_path: Path) -> None:
        """Exceeding max_records raises ValidationError."""
        f = tmp_path / "big.jsonl"
        lines = ['{"i": ' + str(i) + "}" for i in range(20)]
        f.write_text("\n".join(lines))
        with pytest.raises(ValidationError, match="maximum"):
            self.validator.validate_jsonl_file(f, max_records=10)

    # -- validate_curriculum_format --

    def test_valid_curriculum(self) -> None:
        """Records with all required fields produce no errors."""
        records = [
            {
                "question": "Q1",
                "ideal_answer": "A1",
                "competency_id": "comp_001",
            }
        ]
        errors = self.validator.validate_curriculum_format(records)
        assert errors == []

    def test_curriculum_missing_fields(self) -> None:
        """Missing required fields are reported."""
        records = [{"question": "Q1"}]
        errors = self.validator.validate_curriculum_format(records)
        assert len(errors) >= 1
        assert "competency_id" in errors[0]

    def test_curriculum_empty_question(self) -> None:
        """Empty question string is flagged."""
        records = [
            {
                "question": "   ",
                "ideal_answer": "A1",
                "competency_id": "comp_001",
            }
        ]
        errors = self.validator.validate_curriculum_format(records)
        assert any("question" in e and "empty" in e for e in errors)

    def test_curriculum_empty_answer(self) -> None:
        """Empty ideal_answer is flagged."""
        records = [
            {
                "question": "Q1",
                "ideal_answer": "",
                "competency_id": "comp_001",
            }
        ]
        errors = self.validator.validate_curriculum_format(records)
        assert any("ideal_answer" in e and "empty" in e for e in errors)

    def test_curriculum_multiple_errors(self) -> None:
        """Multiple bad records produce multiple errors."""
        records = [
            {"question": "Q1"},
            {"ideal_answer": "A2"},
        ]
        errors = self.validator.validate_curriculum_format(records)
        assert len(errors) >= 2

    # -- validate_model_path --

    def test_model_path_existing_file(self, tmp_path: Path) -> None:
        """Non-empty model file returns True."""
        model = tmp_path / "model.bin"
        model.write_bytes(b"\x00" * 100)
        assert self.validator.validate_model_path(model) is True

    def test_model_path_empty_file(self, tmp_path: Path) -> None:
        """Empty model file returns False."""
        model = tmp_path / "empty.bin"
        model.write_bytes(b"")
        assert self.validator.validate_model_path(model) is False

    def test_model_path_directory_with_files(self, tmp_path: Path) -> None:
        """Non-empty model directory returns True."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "weights.bin").write_bytes(b"\x01")
        assert self.validator.validate_model_path(model_dir) is True

    def test_model_path_empty_directory(self, tmp_path: Path) -> None:
        """Empty model directory returns False."""
        model_dir = tmp_path / "empty_model"
        model_dir.mkdir()
        assert self.validator.validate_model_path(model_dir) is False

    def test_model_path_nonexistent(self, tmp_path: Path) -> None:
        """Non-existent model path returns False."""
        assert self.validator.validate_model_path(tmp_path / "nope") is False

    # -- sanitize_string --

    def test_sanitize_normal_string(self) -> None:
        """Normal string is returned as-is after stripping."""
        result = self.validator.sanitize_string("  hello world  ")
        assert result == "hello world"

    def test_sanitize_html_tags_escaped(self) -> None:
        """HTML tags are escaped."""
        result = self.validator.sanitize_string("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_sanitize_truncates(self) -> None:
        """Long strings are truncated at max_length."""
        result = self.validator.sanitize_string("a" * 2000, max_length=100)
        assert len(result) == 100

    def test_sanitize_control_characters_removed(self) -> None:
        """Control characters are stripped."""
        result = self.validator.sanitize_string("hello\x00world\x07!")
        assert "\x00" not in result
        assert "\x07" not in result
        assert "helloworld!" in result

    def test_sanitize_preserves_newlines_and_tabs(self) -> None:
        """Common whitespace (newlines, tabs) is preserved."""
        result = self.validator.sanitize_string("line1\nline2\ttab")
        assert "\n" in result
        assert "\t" in result

    def test_sanitize_quotes_escaped(self) -> None:
        """Single and double quotes are HTML-escaped."""
        result = self.validator.sanitize_string("He said \"hi\" & she said 'bye'")
        assert "&quot;" in result
        assert "&#x27;" in result


class TestIsSubpath:
    """Tests for the _is_subpath helper."""

    def test_child_under_parent(self, tmp_path: Path) -> None:
        """Nested path is a subpath."""
        child = tmp_path / "a" / "b"
        assert _is_subpath(child, tmp_path) is True

    def test_same_path(self, tmp_path: Path) -> None:
        """Equal paths count as subpath."""
        assert _is_subpath(tmp_path, tmp_path) is True

    def test_sibling_not_subpath(self, tmp_path: Path) -> None:
        """Sibling directory is not a subpath."""
        sibling = tmp_path.parent / "other"
        assert _is_subpath(sibling, tmp_path) is False


class TestStripControlChars:
    """Tests for the _strip_control_chars helper."""

    def test_removes_null_byte(self) -> None:
        """Null byte is removed."""
        assert _strip_control_chars("a\x00b") == "ab"

    def test_preserves_newline(self) -> None:
        """Newline is preserved."""
        assert _strip_control_chars("a\nb") == "a\nb"

    def test_preserves_tab(self) -> None:
        """Tab is preserved."""
        assert _strip_control_chars("a\tb") == "a\tb"

    def test_removes_bell(self) -> None:
        """Bell character (0x07) is removed."""
        assert _strip_control_chars("a\x07b") == "ab"

    def test_removes_delete(self) -> None:
        """DEL character (0x7f) is removed."""
        assert _strip_control_chars("a\x7fb") == "ab"


# =========================================================================
# 5. Graceful Degradation / Health Checks
# =========================================================================


class TestHealthCheck:
    """Tests for HealthCheck dataclass."""

    def test_to_dict_keys(self) -> None:
        """to_dict has all expected keys."""
        hc = HealthCheck(
            component="quarry",
            status="healthy",
            message="All good.",
        )
        d = hc.to_dict()
        assert set(d.keys()) == {"component", "status", "message", "checked_at"}

    def test_checked_at_is_utc(self) -> None:
        """Default timestamp is UTC."""
        hc = HealthCheck(component="forge", status="healthy", message="ok")
        assert hc.checked_at.tzinfo == timezone.utc

    def test_checked_at_iso_format(self) -> None:
        """Serialized timestamp is ISO-8601."""
        hc = HealthCheck(component="forge", status="healthy", message="ok")
        d = hc.to_dict()
        # Should parse without error
        datetime.fromisoformat(d["checked_at"])


class TestSystemHealthChecker:
    """Tests for SystemHealthChecker component checks."""

    def test_quarry_healthy_with_importable_module(self, tmp_path: Path) -> None:
        """Quarry is healthy when directory exists and module imports."""
        quarry_dir = tmp_path / "quarry"
        quarry_dir.mkdir()
        checker = SystemHealthChecker(quarry_path=quarry_dir)
        with patch("shared.hardening._try_import", return_value=True):
            hc = checker.check_quarry()
        assert hc.status == "healthy"
        assert hc.component == "quarry"

    def test_quarry_unavailable_when_dir_missing(self, tmp_path: Path) -> None:
        """Quarry is unavailable when directory does not exist."""
        checker = SystemHealthChecker(quarry_path=tmp_path / "nope")
        hc = checker.check_quarry()
        assert hc.status == "unavailable"

    def test_quarry_degraded_when_import_fails(self, tmp_path: Path) -> None:
        """Quarry is degraded when directory exists but import fails."""
        quarry_dir = tmp_path / "quarry"
        quarry_dir.mkdir()
        checker = SystemHealthChecker(quarry_path=quarry_dir)
        with patch("shared.hardening._try_import", return_value=False):
            hc = checker.check_quarry()
        assert hc.status == "degraded"

    def test_forge_healthy(self, tmp_path: Path) -> None:
        """Forge is healthy when directory and import succeed."""
        forge_dir = tmp_path / "forge"
        forge_dir.mkdir()
        checker = SystemHealthChecker(forge_path=forge_dir)
        with patch("shared.hardening._try_import", return_value=True):
            hc = checker.check_forge()
        assert hc.status == "healthy"
        assert hc.component == "forge"

    def test_foundry_healthy(self, tmp_path: Path) -> None:
        """Foundry is healthy when directory and import succeed."""
        foundry_dir = tmp_path / "foundry"
        foundry_dir.mkdir()
        checker = SystemHealthChecker(foundry_path=foundry_dir)
        with patch("shared.hardening._try_import", return_value=True):
            hc = checker.check_foundry()
        assert hc.status == "healthy"
        assert hc.component == "foundry"

    def test_foundry_unavailable_not_configured(self) -> None:
        """Foundry is unavailable when path is None and import fails."""
        checker = SystemHealthChecker()
        with patch("shared.hardening._try_import", return_value=False):
            hc = checker.check_foundry()
        assert hc.status == "unavailable"
        assert "not configured" in hc.message

    def test_full_check_returns_three_results(self, tmp_path: Path) -> None:
        """full_check returns exactly three HealthCheck objects."""
        checker = SystemHealthChecker(
            quarry_path=tmp_path,
            forge_path=tmp_path,
            foundry_path=tmp_path,
        )
        with patch("shared.hardening._try_import", return_value=True):
            results = checker.full_check()
        assert len(results) == 3
        components = {r.component for r in results}
        assert components == {"quarry", "forge", "foundry"}

    def test_full_check_mixed_statuses(self, tmp_path: Path) -> None:
        """full_check can report different statuses per component."""
        quarry_dir = tmp_path / "quarry"
        quarry_dir.mkdir()

        checker = SystemHealthChecker(
            quarry_path=quarry_dir,
            forge_path=tmp_path / "nonexistent",
            foundry_path=None,
        )

        def selective_import(name: str) -> bool:
            return name == "quarry.chonk"

        with patch("shared.hardening._try_import", side_effect=selective_import):
            results = checker.full_check()

        status_map = {r.component: r.status for r in results}
        assert status_map["quarry"] == "healthy"
        assert status_map["forge"] == "unavailable"
        assert status_map["foundry"] == "unavailable"

    def test_component_healthy_when_no_path_but_importable(self) -> None:
        """Component is healthy when path is None but import works."""
        checker = SystemHealthChecker()
        with patch("shared.hardening._try_import", return_value=True):
            hc = checker.check_quarry()
        assert hc.status == "healthy"
