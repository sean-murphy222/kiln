"""Root pytest configuration for Kiln.

Registers behavior shared across all test suites (quarry, forge, foundry,
hearth, integration, shared). Currently this gates GPU/model integration
tests so they only run when explicitly requested.
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked ``gpu`` unless ``KILN_RUN_GPU_TESTS=1``.

    GPU tests load real model weights and require a CUDA device, so they are
    excluded from the default suite (and CI) to keep it fast and hardware-free.

    Args:
        config: The active pytest configuration.
        items: Collected test items, mutated in place to add skip markers.
    """
    if os.environ.get("KILN_RUN_GPU_TESTS") == "1":
        return
    skip_gpu = pytest.mark.skip(reason="GPU test; set KILN_RUN_GPU_TESTS=1 to run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
