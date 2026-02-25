"""Validate that the MVP demo script runs without errors.

Each demo step is tested independently and also as a full
end-to-end sequence. Tests use temporary directories to avoid
any filesystem side effects.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.demo_mvp import (
    demo_forge,
    demo_foundry,
    demo_hearth,
    demo_quarry,
    main,
)


class TestDemoQuarry:
    """Validate Quarry demo step."""

    def test_demo_quarry_returns_pdf_path(self, tmp_path: Path) -> None:
        """demo_quarry should return the path to the created PDF."""
        result = demo_quarry(tmp_path)
        assert isinstance(result, Path)
        assert result.exists()
        assert result.suffix == ".pdf"

    def test_demo_quarry_creates_valid_pdf(self, tmp_path: Path) -> None:
        """The created PDF should be a valid fitz-readable document."""
        import fitz

        pdf_path = demo_quarry(tmp_path)
        doc = fitz.open(str(pdf_path))
        assert doc.page_count == 3
        doc.close()


class TestDemoForge:
    """Validate Forge demo step."""

    def test_demo_forge_returns_curriculum_path(self, tmp_path: Path) -> None:
        """demo_forge should return the path to the curriculum JSONL."""
        result = demo_forge(tmp_path)
        assert isinstance(result, Path)
        assert result.exists()
        assert result.suffix == ".jsonl"

    def test_demo_forge_curriculum_has_records(self, tmp_path: Path) -> None:
        """Exported curriculum should contain valid Alpaca-format records."""
        curriculum_path = demo_forge(tmp_path)
        with open(curriculum_path, encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) >= 3  # at least 3 non-test examples

        for line in lines:
            record = json.loads(line)
            assert "instruction" in record
            assert "output" in record
            assert record["instruction"]
            assert record["output"]

    def test_demo_forge_creates_test_set(self, tmp_path: Path) -> None:
        """demo_forge should also create a test set JSONL."""
        demo_forge(tmp_path)
        test_set_path = tmp_path / "test_set.jsonl"
        assert test_set_path.exists()

        with open(test_set_path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) >= 1


class TestDemoFoundry:
    """Validate Foundry demo step."""

    def test_demo_foundry_runs_without_error(self, tmp_path: Path) -> None:
        """demo_foundry should complete without raising exceptions."""
        curriculum_path = self._create_curriculum(tmp_path)
        demo_foundry(curriculum_path, tmp_path)

    @staticmethod
    def _create_curriculum(tmp_path: Path) -> Path:
        """Create a minimal valid curriculum JSONL for testing.

        Args:
            tmp_path: Temporary directory for the file.

        Returns:
            Path to the created JSONL file.
        """
        path = tmp_path / "test_curriculum.jsonl"
        records = [
            {
                "instruction": f"Question {i}",
                "input": "",
                "output": f"Answer {i}",
                "metadata": {"example_id": f"ex_{i}"},
            }
            for i in range(5)
        ]
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        return path


class TestDemoHearth:
    """Validate Hearth demo step."""

    def test_demo_hearth_runs_without_error(self) -> None:
        """demo_hearth should complete without raising exceptions."""
        demo_hearth()


class TestFullDemo:
    """Validate the complete end-to-end demo."""

    def test_main_returns_zero(self) -> None:
        """main() should return 0 on successful completion."""
        result = main()
        assert result == 0

    def test_full_pipeline_sequence(self, tmp_path: Path) -> None:
        """All four steps should run in sequence without errors."""
        pdf_path = demo_quarry(tmp_path)
        assert pdf_path.exists()

        curriculum_path = demo_forge(tmp_path)
        assert curriculum_path.exists()

        demo_foundry(curriculum_path, tmp_path)
        demo_hearth()
