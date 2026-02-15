"""
Extraction strategy management.

Handles selection and availability of different extraction tiers.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from chonk.core.document import Block, DocumentMetadata

# ============================================================================
# GPU Compatibility Check - MUST run before any torch/docling imports
# RTX 50 series (Blackwell, sm_120) is not supported by PyTorch 2.x
# ============================================================================

def _check_gpu_architecture() -> tuple[bool, str | None]:
    """
    Check if GPU is compatible with PyTorch CUDA support.
    Uses nvidia-smi to detect GPU without importing torch.

    Returns:
        Tuple of (is_compatible, upgrade_message)
        - is_compatible: True if compatible or no GPU, False if incompatible architecture
        - upgrade_message: If incompatible, a message about how to fix it
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return True, None  # No nvidia-smi or no GPU

        compute_cap = result.stdout.strip().split("\n")[0]
        if not compute_cap:
            return True, None

        major_version = int(compute_cap.split(".")[0])

        # RTX 50 series (Blackwell, sm_120, compute capability 12.x)
        # requires PyTorch 2.7+ with CUDA 12.8
        if major_version >= 10:
            # Check if PyTorch version supports Blackwell
            try:
                import importlib.metadata
                torch_version = importlib.metadata.version("torch")
                # Parse version: "2.7.0+cu128" -> (2, 7, 0)
                version_parts = torch_version.split("+")[0].split(".")
                major, minor = int(version_parts[0]), int(version_parts[1])

                # PyTorch 2.7+ supports Blackwell with CUDA 12.8
                if major > 2 or (major == 2 and minor >= 7):
                    return True, None  # Compatible!
            except Exception:
                pass  # If we can't check, assume incompatible

            upgrade_msg = (
                f"RTX 50 series GPU detected (compute capability {compute_cap}). "
                "PyTorch 2.7+ with CUDA 12.8 is required for Blackwell support. "
                "Upgrade: pip install torch>=2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128"
            )
            return False, upgrade_msg

        return True, None

    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
        return True, None


# Disable CUDA BEFORE any torch/docling imports if GPU is incompatible
_GPU_INCOMPATIBLE = False
_GPU_UPGRADE_MESSAGE: str | None = None

_gpu_compatible, _upgrade_msg = _check_gpu_architecture()
if not _gpu_compatible:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _GPU_INCOMPATIBLE = True
    _GPU_UPGRADE_MESSAGE = _upgrade_msg


def _patch_torch_cuda() -> None:
    """
    Monkeypatch torch.cuda.is_available() to return False.

    This is necessary because:
    1. CUDA_VISIBLE_DEVICES="" makes device_count() return 0
    2. But is_available() still returns True (CUDA runtime is installed)
    3. Many libraries (Docling, transformers, etc.) check is_available()
       and try to use CUDA, then fail with "no kernel image" error

    This patch forces all CUDA-dependent code to fall back to CPU.
    """
    try:
        import torch
        import torch.cuda

        # Only patch if not already patched
        if hasattr(torch.cuda, '_chonk_patched'):
            return

        # Store original for potential restoration
        torch.cuda._original_is_available = torch.cuda.is_available

        # Patch is_available to always return False
        torch.cuda.is_available = lambda: False

        # Also patch device_count to be safe
        torch.cuda._original_device_count = torch.cuda.device_count
        torch.cuda.device_count = lambda: 0

        # Mark as patched
        torch.cuda._chonk_patched = True

    except ImportError:
        pass  # torch not installed, nothing to patch


def _ensure_cuda_patched() -> None:
    """Ensure CUDA is patched before any GPU-dependent code runs."""
    if _GPU_INCOMPATIBLE:
        _patch_torch_cuda()


class ExtractionTier(Enum):
    """Available extraction tiers."""

    FAST = "fast"  # Tier 1: PyMuPDF + pdfplumber
    ENHANCED = "enhanced"  # Tier 2: Docling
    AI = "ai"  # Tier 3: LayoutParser


@dataclass
class ExtractionResult:
    """Result from an extraction operation."""

    blocks: list[Block]
    metadata: DocumentMetadata
    tier_used: ExtractionTier
    extraction_info: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@runtime_checkable
class Extractor(Protocol):
    """Protocol for document extractors."""

    def extract(self, path: Path) -> ExtractionResult:
        """Extract content from a document."""
        ...

    def is_available(self) -> bool:
        """Check if this extractor is available (dependencies installed)."""
        ...

    @property
    def tier(self) -> ExtractionTier:
        """Get the tier of this extractor."""
        ...


def check_docling_available() -> bool:
    """Check if Docling is installed and available."""
    # Ensure CUDA is patched BEFORE importing docling (which imports torch)
    _ensure_cuda_patched()
    try:
        import docling
        return True
    except ImportError:
        return False


def check_layoutparser_available() -> bool:
    """Check if LayoutParser is installed and available."""
    try:
        import layoutparser
        return True
    except ImportError:
        return False


def get_available_tiers() -> list[ExtractionTier]:
    """Get list of available extraction tiers based on installed dependencies."""
    tiers = [ExtractionTier.FAST]  # Always available

    if check_docling_available():
        tiers.append(ExtractionTier.ENHANCED)

    if check_layoutparser_available():
        tiers.append(ExtractionTier.AI)

    return tiers


def get_extractor(tier: ExtractionTier) -> Extractor:
    """
    Get an extractor for the specified tier.

    Falls back to lower tiers if requested tier is unavailable.
    """
    available = get_available_tiers()

    # If requested tier is available, use it
    if tier in available:
        return _create_extractor(tier)

    # Fall back to highest available tier
    if ExtractionTier.ENHANCED in available and tier == ExtractionTier.AI:
        return _create_extractor(ExtractionTier.ENHANCED)

    # Default to fast
    return _create_extractor(ExtractionTier.FAST)


def _create_extractor(tier: ExtractionTier) -> Extractor:
    """Create an extractor instance for the given tier."""
    if tier == ExtractionTier.FAST:
        from chonk.extraction.fast_extractor import FastExtractor
        return FastExtractor()

    elif tier == ExtractionTier.ENHANCED:
        from chonk.extraction.docling_extractor import DoclingExtractor
        return DoclingExtractor()

    elif tier == ExtractionTier.AI:
        from chonk.extraction.layoutparser_extractor import LayoutParserExtractor
        return LayoutParserExtractor()

    raise ValueError(f"Unknown extraction tier: {tier}")


@dataclass
class ExtractionStrategy:
    """
    Strategy for document extraction.

    Manages tier selection and fallback behavior.
    """

    preferred_tier: ExtractionTier = ExtractionTier.FAST
    auto_fallback: bool = True  # Automatically fall back to lower tiers
    auto_upgrade: bool = False  # Automatically use higher tier for complex docs

    # Thresholds for auto-upgrade (if enabled)
    upgrade_on_tables: int = 5  # Upgrade if document has many tables
    upgrade_on_scanned: bool = True  # Upgrade if document appears scanned
    upgrade_on_multicolumn: bool = True  # Upgrade if multi-column layout detected

    def select_tier(
        self,
        path: Path,
        detected_features: dict[str, Any] | None = None,
    ) -> ExtractionTier:
        """
        Select the best tier for a document.

        Args:
            path: Path to the document
            detected_features: Optional pre-detected document features

        Returns:
            The tier to use for extraction
        """
        available = get_available_tiers()
        tier = self.preferred_tier

        # Check if preferred tier is available
        if tier not in available:
            if self.auto_fallback:
                # Fall back to highest available
                tier = available[-1]
            else:
                raise RuntimeError(
                    f"Extraction tier {tier.value} is not available. "
                    f"Install the required dependencies."
                )

        # Auto-upgrade if enabled and features suggest complex document
        if self.auto_upgrade and detected_features:
            if self._should_upgrade(detected_features, available):
                # Upgrade to next available tier
                current_idx = available.index(tier)
                if current_idx < len(available) - 1:
                    tier = available[current_idx + 1]

        return tier

    def _should_upgrade(
        self,
        features: dict[str, Any],
        available: list[ExtractionTier],
    ) -> bool:
        """Determine if we should upgrade to a higher tier."""
        # Check table count
        if features.get("table_count", 0) >= self.upgrade_on_tables:
            return True

        # Check if scanned
        if self.upgrade_on_scanned and features.get("is_scanned", False):
            return True

        # Check for multi-column
        if self.upgrade_on_multicolumn and features.get("is_multicolumn", False):
            return True

        return False

    def extract(self, path: Path) -> ExtractionResult:
        """
        Extract content from a document using the best available strategy.

        Args:
            path: Path to the document

        Returns:
            ExtractionResult with blocks and metadata
        """
        # Quick feature detection for auto-upgrade
        detected_features = None
        if self.auto_upgrade:
            detected_features = self._quick_detect_features(path)

        # Select tier
        tier = self.select_tier(path, detected_features)

        # Get extractor and run
        extractor = get_extractor(tier)
        return extractor.extract(path)

    def _quick_detect_features(self, path: Path) -> dict[str, Any]:
        """
        Quickly detect document features for tier selection.

        Uses fast heuristics without full parsing.
        """
        features: dict[str, Any] = {
            "table_count": 0,
            "is_scanned": False,
            "is_multicolumn": False,
        }

        # Use PyMuPDF for quick detection
        try:
            import fitz
            doc = fitz.open(path)

            # Sample first few pages
            for page_num in range(min(5, len(doc))):
                page = doc[page_num]

                # Check for text (scanned docs have little/no text)
                text = page.get_text()
                if len(text.strip()) < 100:
                    features["is_scanned"] = True

                # Check for tables (look for grid-like structures)
                # This is a heuristic based on line count
                drawings = page.get_drawings()
                horizontal_lines = sum(
                    1 for d in drawings
                    if d.get("type") == "l" and abs(d.get("y0", 0) - d.get("y1", 0)) < 2
                )
                if horizontal_lines > 10:
                    features["table_count"] += 1

                # Check for multi-column (text blocks at different x positions)
                blocks = page.get_text("dict").get("blocks", [])
                x_positions = set()
                for block in blocks:
                    if block.get("type") == 0:  # Text block
                        x_positions.add(round(block.get("bbox", [0])[0] / 50) * 50)
                if len(x_positions) >= 2:
                    features["is_multicolumn"] = True

            doc.close()

        except Exception:
            pass  # If detection fails, use defaults

        return features
