"""Tests for model merging pipeline.

Covers adapter info, merge config, compatibility checking,
linear/TIES merging, pipeline lifecycle, registry, and
plain-language summaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundry.src.merging import (
    AdapterInfo,
    CompatibilityChecker,
    CompatibilityResult,
    LinearMerger,
    MergeConfig,
    MergeMethod,
    MergePipeline,
    MergeRegistry,
    MergeResult,
    MergeStatus,
    MergingError,
    TIESMerger,
)

# ===================================================================
# TestAdapterInfo
# ===================================================================


class TestAdapterInfo:
    """Tests for AdapterInfo dataclass."""

    def test_construction(self, adapter_info_a: AdapterInfo) -> None:
        """AdapterInfo stores all fields correctly."""
        assert adapter_info_a.discipline_id == "disc_hydraulics"
        assert adapter_info_a.discipline_name == "Hydraulic Systems"
        assert adapter_info_a.base_model == "microsoft/phi-3-mini-4k-instruct"
        assert adapter_info_a.base_model_family == "phi"
        assert adapter_info_a.lora_rank == 16

    def test_to_dict(self, adapter_info_a: AdapterInfo) -> None:
        """to_dict serializes all fields."""
        data = adapter_info_a.to_dict()
        assert data["discipline_id"] == "disc_hydraulics"
        assert data["base_model"] == "microsoft/phi-3-mini-4k-instruct"
        assert data["lora_rank"] == 16
        assert "adapter_path" in data

    def test_from_dict_roundtrip(self, adapter_info_a: AdapterInfo) -> None:
        """from_dict reconstructs an equivalent AdapterInfo."""
        data = adapter_info_a.to_dict()
        restored = AdapterInfo.from_dict(data)
        assert restored.discipline_id == adapter_info_a.discipline_id
        assert restored.base_model == adapter_info_a.base_model
        assert restored.lora_rank == adapter_info_a.lora_rank
        assert restored.adapter_path == adapter_info_a.adapter_path

    def test_optional_training_run_id(self, tmp_path: Path) -> None:
        """training_run_id defaults to None."""
        adapter_dir = tmp_path / "adapter_no_run"
        adapter_dir.mkdir()
        info = AdapterInfo(
            adapter_path=adapter_dir,
            discipline_id="disc_test",
            discipline_name="Test",
            base_model="test-model",
            base_model_family="phi",
            lora_rank=16,
        )
        assert info.training_run_id is None

    def test_with_training_run_id(self, tmp_path: Path) -> None:
        """training_run_id can be explicitly set."""
        adapter_dir = tmp_path / "adapter_with_run"
        adapter_dir.mkdir()
        info = AdapterInfo(
            adapter_path=adapter_dir,
            discipline_id="disc_test",
            discipline_name="Test",
            base_model="test-model",
            base_model_family="phi",
            lora_rank=16,
            training_run_id="run_abc123",
        )
        assert info.training_run_id == "run_abc123"


# ===================================================================
# TestMergeConfig
# ===================================================================


class TestMergeConfig:
    """Tests for MergeConfig dataclass."""

    def test_defaults(self, merge_config: MergeConfig) -> None:
        """Default config uses LINEAR method with equal weights."""
        assert merge_config.method == MergeMethod.LINEAR
        assert merge_config.weights is None
        assert merge_config.ties_density == 0.5
        assert merge_config.validate_compatibility is True

    def test_custom_weights(self) -> None:
        """Custom per-adapter weights are stored."""
        cfg = MergeConfig(weights=[0.7, 0.3])
        assert cfg.weights == [0.7, 0.3]

    def test_ties_config(self) -> None:
        """TIES config stores density and majority sign."""
        cfg = MergeConfig(
            method=MergeMethod.TIES,
            ties_density=0.3,
            ties_majority_sign=False,
        )
        assert cfg.method == MergeMethod.TIES
        assert cfg.ties_density == 0.3
        assert cfg.ties_majority_sign is False

    def test_to_dict(self) -> None:
        """to_dict serializes config fields."""
        cfg = MergeConfig(method=MergeMethod.TIES, ties_density=0.4)
        data = cfg.to_dict()
        assert data["method"] == "ties"
        assert data["ties_density"] == 0.4

    def test_from_dict_roundtrip(self) -> None:
        """from_dict reconstructs equivalent MergeConfig."""
        cfg = MergeConfig(
            method=MergeMethod.TIES,
            weights=[0.6, 0.4],
            ties_density=0.3,
        )
        data = cfg.to_dict()
        restored = MergeConfig.from_dict(data)
        assert restored.method == cfg.method
        assert restored.weights == cfg.weights
        assert restored.ties_density == cfg.ties_density


# ===================================================================
# TestCompatibilityResult
# ===================================================================


class TestCompatibilityResult:
    """Tests for CompatibilityResult dataclass."""

    def test_compatible(self) -> None:
        """Compatible result has no issues."""
        result = CompatibilityResult(
            is_compatible=True,
            issues=[],
            warnings=[],
        )
        assert result.is_compatible is True
        assert len(result.issues) == 0

    def test_incompatible(self) -> None:
        """Incompatible result lists issues."""
        result = CompatibilityResult(
            is_compatible=False,
            issues=["Mismatched base models"],
            warnings=[],
        )
        assert result.is_compatible is False
        assert "Mismatched base models" in result.issues

    def test_with_warnings(self) -> None:
        """Warnings are separate from blocking issues."""
        result = CompatibilityResult(
            is_compatible=True,
            issues=[],
            warnings=["Different LoRA ranks may affect quality"],
        )
        assert result.is_compatible is True
        assert len(result.warnings) == 1

    def test_to_dict(self) -> None:
        """to_dict serializes compatibility result."""
        result = CompatibilityResult(
            is_compatible=False,
            issues=["bad model"],
            warnings=["rank diff"],
        )
        data = result.to_dict()
        assert data["is_compatible"] is False
        assert "bad model" in data["issues"]
        assert "rank diff" in data["warnings"]


# ===================================================================
# TestMergeResult
# ===================================================================


class TestMergeResult:
    """Tests for MergeResult dataclass."""

    def test_construction(self, adapter_info_a: AdapterInfo, adapter_info_b: AdapterInfo) -> None:
        """MergeResult stores all fields correctly."""
        from datetime import datetime

        result = MergeResult(
            merge_id="merge_abc123",
            method=MergeMethod.LINEAR,
            status=MergeStatus.COMPLETED,
            adapters=[adapter_info_a, adapter_info_b],
            weights_used=[0.5, 0.5],
            merged_adapter_path=Path("/tmp/merged"),
            evaluation_results=None,
            started_at=datetime(2026, 2, 20, 10, 0, 0),
            completed_at=datetime(2026, 2, 20, 10, 5, 0),
            error_message=None,
            plain_language_summary="Merged 2 adapters.",
        )
        assert result.merge_id == "merge_abc123"
        assert result.status == MergeStatus.COMPLETED
        assert len(result.adapters) == 2

    def test_to_dict_from_dict_roundtrip(
        self, adapter_info_a: AdapterInfo, adapter_info_b: AdapterInfo
    ) -> None:
        """to_dict/from_dict preserves all data."""
        from datetime import datetime

        original = MergeResult(
            merge_id="merge_xyz",
            method=MergeMethod.LINEAR,
            status=MergeStatus.COMPLETED,
            adapters=[adapter_info_a, adapter_info_b],
            weights_used=[0.5, 0.5],
            merged_adapter_path=Path("/tmp/merged"),
            evaluation_results={"disc_a": 0.85},
            started_at=datetime(2026, 2, 20, 10, 0, 0),
            completed_at=datetime(2026, 2, 20, 10, 5, 0),
            error_message=None,
            plain_language_summary="Test summary.",
        )
        data = original.to_dict()
        restored = MergeResult.from_dict(data)
        assert restored.merge_id == original.merge_id
        assert restored.method == original.method
        assert restored.status == original.status
        assert len(restored.adapters) == 2
        assert restored.evaluation_results == {"disc_a": 0.85}

    def test_failed_status(self, adapter_info_a: AdapterInfo) -> None:
        """Failed merge stores error message."""
        from datetime import datetime

        result = MergeResult(
            merge_id="merge_fail",
            method=MergeMethod.LINEAR,
            status=MergeStatus.FAILED,
            adapters=[adapter_info_a],
            weights_used=[1.0],
            merged_adapter_path=None,
            evaluation_results=None,
            started_at=datetime(2026, 2, 20, 10, 0, 0),
            completed_at=datetime(2026, 2, 20, 10, 0, 5),
            error_message="Incompatible adapters",
            plain_language_summary="Merge failed.",
        )
        assert result.status == MergeStatus.FAILED
        assert result.error_message == "Incompatible adapters"
        assert result.merged_adapter_path is None


# ===================================================================
# TestCompatibilityChecker
# ===================================================================


class TestCompatibilityChecker:
    """Tests for CompatibilityChecker."""

    def test_compatible_adapters(
        self, adapter_info_a: AdapterInfo, adapter_info_b: AdapterInfo
    ) -> None:
        """Two adapters with same base model and rank are compatible."""
        checker = CompatibilityChecker()
        result = checker.check([adapter_info_a, adapter_info_b])
        assert result.is_compatible is True
        assert len(result.issues) == 0

    def test_mismatched_base_model(
        self, adapter_info_a: AdapterInfo, incompatible_adapter: AdapterInfo
    ) -> None:
        """Different base models are incompatible."""
        checker = CompatibilityChecker()
        result = checker.check([adapter_info_a, incompatible_adapter])
        assert result.is_compatible is False
        assert any("base model" in issue.lower() for issue in result.issues)

    def test_mismatched_rank_linear(self, tmp_path: Path) -> None:
        """Different LoRA ranks are incompatible for LINEAR merge."""
        dir_a = tmp_path / "adapter_rank16"
        dir_a.mkdir()
        dir_b = tmp_path / "adapter_rank32"
        dir_b.mkdir()
        info_a = AdapterInfo(
            adapter_path=dir_a,
            discipline_id="disc_a",
            discipline_name="A",
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family="phi",
            lora_rank=16,
        )
        info_b = AdapterInfo(
            adapter_path=dir_b,
            discipline_id="disc_b",
            discipline_name="B",
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family="phi",
            lora_rank=32,
        )
        checker = CompatibilityChecker()
        result = checker.check(
            [info_a, info_b],
            method=MergeMethod.LINEAR,
        )
        assert result.is_compatible is False
        assert any("rank" in issue.lower() for issue in result.issues)

    def test_ties_allows_different_ranks(self, tmp_path: Path) -> None:
        """TIES merge allows different LoRA ranks with a warning."""
        dir_a = tmp_path / "adapter_rank16_ties"
        dir_a.mkdir()
        dir_b = tmp_path / "adapter_rank32_ties"
        dir_b.mkdir()
        info_a = AdapterInfo(
            adapter_path=dir_a,
            discipline_id="disc_a",
            discipline_name="A",
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family="phi",
            lora_rank=16,
        )
        info_b = AdapterInfo(
            adapter_path=dir_b,
            discipline_id="disc_b",
            discipline_name="B",
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family="phi",
            lora_rank=32,
        )
        checker = CompatibilityChecker()
        result = checker.check(
            [info_a, info_b],
            method=MergeMethod.TIES,
        )
        assert result.is_compatible is True
        assert any("rank" in w.lower() for w in result.warnings)

    def test_missing_paths_warning(self, tmp_path: Path) -> None:
        """Missing adapter paths produce warnings."""
        info = AdapterInfo(
            adapter_path=tmp_path / "nonexistent",
            discipline_id="disc_a",
            discipline_name="A",
            base_model="microsoft/phi-3-mini-4k-instruct",
            base_model_family="phi",
            lora_rank=16,
        )
        checker = CompatibilityChecker()
        result = checker.check([info])
        assert any("path" in w.lower() for w in result.warnings)

    def test_single_adapter_compatible(self, adapter_info_a: AdapterInfo) -> None:
        """A single adapter is always compatible with itself."""
        checker = CompatibilityChecker()
        result = checker.check([adapter_info_a])
        assert result.is_compatible is True


# ===================================================================
# TestLinearMerger
# ===================================================================


class TestLinearMerger:
    """Tests for LinearMerger."""

    def test_merge_equal_weights(
        self,
        adapter_info_a: AdapterInfo,
        adapter_info_b: AdapterInfo,
        tmp_path: Path,
    ) -> None:
        """Linear merge with equal weights produces output directory."""
        merger = LinearMerger()
        output = tmp_path / "linear_output"
        result_path = merger.merge(
            [adapter_info_a, adapter_info_b],
            weights=[0.5, 0.5],
            output_path=output,
        )
        assert result_path.exists()
        assert result_path.is_dir()

    def test_merge_custom_weights(
        self,
        adapter_info_a: AdapterInfo,
        adapter_info_b: AdapterInfo,
        tmp_path: Path,
    ) -> None:
        """Linear merge with custom weights stores weights in metadata."""
        merger = LinearMerger()
        output = tmp_path / "linear_custom"
        merger.merge(
            [adapter_info_a, adapter_info_b],
            weights=[0.7, 0.3],
            output_path=output,
        )
        metadata_path = output / "merge_metadata.json"
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["weights"] == [0.7, 0.3]

    def test_weight_normalization(self) -> None:
        """Weights that do not sum to 1 are normalized."""
        merger = LinearMerger()
        normalized = merger._normalize_weights([2.0, 2.0])
        assert abs(sum(normalized) - 1.0) < 1e-9
        assert abs(normalized[0] - 0.5) < 1e-9

    def test_output_metadata_contains_adapters(
        self,
        adapter_info_a: AdapterInfo,
        adapter_info_b: AdapterInfo,
        tmp_path: Path,
    ) -> None:
        """Output metadata lists source adapters."""
        merger = LinearMerger()
        output = tmp_path / "linear_meta"
        merger.merge(
            [adapter_info_a, adapter_info_b],
            weights=[0.5, 0.5],
            output_path=output,
        )
        metadata_path = output / "merge_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert len(metadata["adapters"]) == 2
        assert metadata["method"] == "linear"


# ===================================================================
# TestTIESMerger
# ===================================================================


class TestTIESMerger:
    """Tests for TIESMerger."""

    def test_merge_with_density(
        self,
        adapter_info_a: AdapterInfo,
        adapter_info_b: AdapterInfo,
        tmp_path: Path,
    ) -> None:
        """TIES merge creates output with density parameter."""
        merger = TIESMerger()
        config = MergeConfig(
            method=MergeMethod.TIES,
            ties_density=0.3,
        )
        output = tmp_path / "ties_output"
        result_path = merger.merge(
            [adapter_info_a, adapter_info_b],
            config=config,
            output_path=output,
        )
        assert result_path.exists()

    def test_merge_majority_sign(
        self,
        adapter_info_a: AdapterInfo,
        adapter_info_b: AdapterInfo,
        tmp_path: Path,
    ) -> None:
        """TIES merge stores majority_sign setting in metadata."""
        merger = TIESMerger()
        config = MergeConfig(
            method=MergeMethod.TIES,
            ties_majority_sign=True,
        )
        output = tmp_path / "ties_majority"
        merger.merge(
            [adapter_info_a, adapter_info_b],
            config=config,
            output_path=output,
        )
        metadata_path = output / "merge_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["ties_majority_sign"] is True

    def test_output_metadata(
        self,
        adapter_info_a: AdapterInfo,
        adapter_info_b: AdapterInfo,
        tmp_path: Path,
    ) -> None:
        """TIES output metadata includes density and method."""
        merger = TIESMerger()
        config = MergeConfig(
            method=MergeMethod.TIES,
            ties_density=0.4,
        )
        output = tmp_path / "ties_meta"
        merger.merge(
            [adapter_info_a, adapter_info_b],
            config=config,
            output_path=output,
        )
        metadata_path = output / "merge_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["method"] == "ties"
        assert metadata["ties_density"] == 0.4
        assert len(metadata["adapters"]) == 2


# ===================================================================
# TestMergePipeline
# ===================================================================


class TestMergePipeline:
    """Tests for MergePipeline."""

    def test_full_merge_lifecycle(
        self,
        adapter_info_a: AdapterInfo,
        adapter_info_b: AdapterInfo,
        tmp_path: Path,
    ) -> None:
        """Pipeline validates, merges, and returns completed result."""
        config = MergeConfig(output_dir=tmp_path / "pipeline_out")
        pipeline = MergePipeline(config=config)
        result = pipeline.merge([adapter_info_a, adapter_info_b])
        assert result.status == MergeStatus.COMPLETED
        assert result.merged_adapter_path is not None
        assert result.merged_adapter_path.exists()
        assert len(result.adapters) == 2
        assert len(result.weights_used) == 2

    def test_incompatible_adapters_fail(
        self,
        adapter_info_a: AdapterInfo,
        incompatible_adapter: AdapterInfo,
        tmp_path: Path,
    ) -> None:
        """Pipeline fails if adapters are incompatible."""
        config = MergeConfig(output_dir=tmp_path / "pipeline_fail")
        pipeline = MergePipeline(config=config)
        with pytest.raises(MergingError, match="[Cc]ompatib"):
            pipeline.merge([adapter_info_a, incompatible_adapter])

    def test_custom_method_selection(
        self,
        adapter_info_a: AdapterInfo,
        adapter_info_b: AdapterInfo,
        tmp_path: Path,
    ) -> None:
        """Pipeline uses the configured merge method."""
        config = MergeConfig(
            method=MergeMethod.TIES,
            ties_density=0.4,
            output_dir=tmp_path / "pipeline_ties",
        )
        pipeline = MergePipeline(config=config)
        result = pipeline.merge([adapter_info_a, adapter_info_b])
        assert result.method == MergeMethod.TIES
        assert result.status == MergeStatus.COMPLETED

    def test_status_tracking(
        self,
        adapter_info_a: AdapterInfo,
        adapter_info_b: AdapterInfo,
        tmp_path: Path,
    ) -> None:
        """Pipeline reports PENDING before merge and COMPLETED after."""
        config = MergeConfig(output_dir=tmp_path / "pipeline_status")
        pipeline = MergePipeline(config=config)
        assert pipeline.get_status() == MergeStatus.PENDING
        pipeline.merge([adapter_info_a, adapter_info_b])
        assert pipeline.get_status() == MergeStatus.COMPLETED

    def test_custom_weights(
        self,
        adapter_info_a: AdapterInfo,
        adapter_info_b: AdapterInfo,
        tmp_path: Path,
    ) -> None:
        """Pipeline uses custom weights when provided."""
        config = MergeConfig(
            weights=[0.8, 0.2],
            output_dir=tmp_path / "pipeline_weights",
        )
        pipeline = MergePipeline(config=config)
        result = pipeline.merge([adapter_info_a, adapter_info_b])
        assert abs(result.weights_used[0] - 0.8) < 1e-9
        assert abs(result.weights_used[1] - 0.2) < 1e-9

    def test_skip_validation(
        self,
        adapter_info_a: AdapterInfo,
        incompatible_adapter: AdapterInfo,
        tmp_path: Path,
    ) -> None:
        """Pipeline skips validation when validate_compatibility=False."""
        config = MergeConfig(
            validate_compatibility=False,
            output_dir=tmp_path / "pipeline_skip",
        )
        pipeline = MergePipeline(config=config)
        # Should not raise even with incompatible adapters
        result = pipeline.merge([adapter_info_a, incompatible_adapter])
        assert result.status == MergeStatus.COMPLETED


# ===================================================================
# TestMergeRegistry
# ===================================================================


class TestMergeRegistry:
    """Tests for MergeRegistry."""

    def _make_result(
        self,
        merge_id: str,
        adapter_info_a: AdapterInfo,
        merged_path: Path,
    ) -> MergeResult:
        """Helper to create a MergeResult for registry tests."""
        from datetime import datetime

        return MergeResult(
            merge_id=merge_id,
            method=MergeMethod.LINEAR,
            status=MergeStatus.COMPLETED,
            adapters=[adapter_info_a],
            weights_used=[1.0],
            merged_adapter_path=merged_path,
            evaluation_results=None,
            started_at=datetime(2026, 2, 20, 10, 0, 0),
            completed_at=datetime(2026, 2, 20, 10, 5, 0),
            error_message=None,
            plain_language_summary="Test merge.",
        )

    def test_register_and_get(self, adapter_info_a: AdapterInfo, tmp_path: Path) -> None:
        """Registered merge can be retrieved by ID."""
        registry = MergeRegistry(registry_dir=tmp_path / "registry")
        merged_path = tmp_path / "merged"
        merged_path.mkdir()
        result = self._make_result("merge_001", adapter_info_a, merged_path)
        registry.register(result)
        loaded = registry.get("merge_001")
        assert loaded is not None
        assert loaded.merge_id == "merge_001"

    def test_list_merges(self, adapter_info_a: AdapterInfo, tmp_path: Path) -> None:
        """list_merges returns all registered merges."""
        registry = MergeRegistry(registry_dir=tmp_path / "registry_list")
        merged_path = tmp_path / "merged_list"
        merged_path.mkdir()
        registry.register(self._make_result("merge_a", adapter_info_a, merged_path))
        registry.register(self._make_result("merge_b", adapter_info_a, merged_path))
        merges = registry.list_merges()
        assert len(merges) == 2

    def test_get_nonexistent(self, tmp_path: Path) -> None:
        """get returns None for unknown merge ID."""
        registry = MergeRegistry(registry_dir=tmp_path / "registry_empty")
        assert registry.get("nonexistent") is None

    def test_get_latest(self, adapter_info_a: AdapterInfo, tmp_path: Path) -> None:
        """get_latest returns the most recently registered merge."""
        from datetime import datetime

        registry = MergeRegistry(registry_dir=tmp_path / "registry_latest")
        merged_path = tmp_path / "merged_latest"
        merged_path.mkdir()

        result_old = MergeResult(
            merge_id="merge_old",
            method=MergeMethod.LINEAR,
            status=MergeStatus.COMPLETED,
            adapters=[adapter_info_a],
            weights_used=[1.0],
            merged_adapter_path=merged_path,
            evaluation_results=None,
            started_at=datetime(2026, 2, 19, 10, 0, 0),
            completed_at=datetime(2026, 2, 19, 10, 5, 0),
            error_message=None,
            plain_language_summary="Old merge.",
        )
        result_new = MergeResult(
            merge_id="merge_new",
            method=MergeMethod.LINEAR,
            status=MergeStatus.COMPLETED,
            adapters=[adapter_info_a],
            weights_used=[1.0],
            merged_adapter_path=merged_path,
            evaluation_results=None,
            started_at=datetime(2026, 2, 20, 10, 0, 0),
            completed_at=datetime(2026, 2, 20, 10, 5, 0),
            error_message=None,
            plain_language_summary="New merge.",
        )
        registry.register(result_old)
        registry.register(result_new)
        latest = registry.get_latest()
        assert latest is not None
        assert latest.merge_id == "merge_new"

    def test_empty_registry(self, tmp_path: Path) -> None:
        """Empty registry returns empty list and None for latest."""
        registry = MergeRegistry(registry_dir=tmp_path / "empty_reg")
        assert registry.list_merges() == []
        assert registry.get_latest() is None


# ===================================================================
# TestPlainLanguage
# ===================================================================


class TestPlainLanguage:
    """Tests for plain-language summary generation."""

    def test_summary_describes_merge(
        self,
        adapter_info_a: AdapterInfo,
        adapter_info_b: AdapterInfo,
        tmp_path: Path,
    ) -> None:
        """Summary mentions adapter count and discipline names."""
        config = MergeConfig(output_dir=tmp_path / "plain_lang")
        pipeline = MergePipeline(config=config)
        result = pipeline.merge([adapter_info_a, adapter_info_b])
        summary = result.plain_language_summary
        assert "2" in summary
        assert "Hydraulic Systems" in summary or "Electrical Systems" in summary

    def test_summary_mentions_method(
        self,
        adapter_info_a: AdapterInfo,
        adapter_info_b: AdapterInfo,
        tmp_path: Path,
    ) -> None:
        """Summary mentions the merge method used."""
        config = MergeConfig(
            method=MergeMethod.TIES,
            output_dir=tmp_path / "plain_lang_ties",
        )
        pipeline = MergePipeline(config=config)
        result = pipeline.merge([adapter_info_a, adapter_info_b])
        summary = result.plain_language_summary.lower()
        assert "ties" in summary

    def test_summary_sme_friendly(
        self,
        adapter_info_a: AdapterInfo,
        adapter_info_b: AdapterInfo,
        tmp_path: Path,
    ) -> None:
        """Summary avoids ML jargon."""
        config = MergeConfig(output_dir=tmp_path / "sme_friendly")
        pipeline = MergePipeline(config=config)
        result = pipeline.merge([adapter_info_a, adapter_info_b])
        summary = result.plain_language_summary.lower()
        # Should not contain ML-jargon terms
        assert "loss" not in summary
        assert "perplexity" not in summary
        assert "f1" not in summary


# ===================================================================
# TestIntegration
# ===================================================================


class TestIntegration:
    """Integration tests for the full merge workflow."""

    def test_two_adapters_full_workflow(
        self,
        adapter_info_a: AdapterInfo,
        adapter_info_b: AdapterInfo,
        tmp_path: Path,
    ) -> None:
        """Full workflow: validate, merge, save to registry."""
        # Step 1: Validate compatibility
        checker = CompatibilityChecker()
        compat = checker.check([adapter_info_a, adapter_info_b])
        assert compat.is_compatible is True

        # Step 2: Merge with pipeline
        config = MergeConfig(output_dir=tmp_path / "integration_out")
        pipeline = MergePipeline(config=config)
        result = pipeline.merge([adapter_info_a, adapter_info_b])
        assert result.status == MergeStatus.COMPLETED

        # Step 3: Register result
        registry = MergeRegistry(registry_dir=tmp_path / "integration_reg")
        registry.register(result)

        # Step 4: Retrieve and verify
        loaded = registry.get(result.merge_id)
        assert loaded is not None
        assert loaded.status == MergeStatus.COMPLETED
        assert loaded.merged_adapter_path is not None

    def test_ties_merge_with_evaluation_results(
        self,
        adapter_info_a: AdapterInfo,
        adapter_info_b: AdapterInfo,
        tmp_path: Path,
    ) -> None:
        """TIES merge stores evaluation results when provided."""
        config = MergeConfig(
            method=MergeMethod.TIES,
            ties_density=0.5,
            output_dir=tmp_path / "ties_eval",
        )
        pipeline = MergePipeline(config=config)
        result = pipeline.merge([adapter_info_a, adapter_info_b])
        # Manually set evaluation results (simulating post-merge eval)
        result.evaluation_results = {
            "disc_hydraulics": 0.85,
            "disc_electrical": 0.78,
        }
        # Save and verify
        registry = MergeRegistry(registry_dir=tmp_path / "ties_eval_reg")
        registry.register(result)
        loaded = registry.get(result.merge_id)
        assert loaded is not None
        assert loaded.evaluation_results is not None
        assert loaded.evaluation_results["disc_hydraulics"] == 0.85
