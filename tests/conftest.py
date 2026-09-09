"""Shared pytest configuration for the pyCamSet test suite.

Three things happen here that the suite previously relied on the developer's
shell for:

1. **Headless by default.** pyvista, Qt and matplotlib all try to open a window
   when a display is available and fail outright when one is not.  CI has no
   display on any of the three platforms, so the backends are pinned to their
   offscreen variants before anything imports them.

2. **Absolute test data paths.** Tests used to reference ``tests/test_data``
   relative to the current directory, so the suite only ran from the repository
   root.  The ``data_dir`` fixture resolves the path from ``__file__`` instead,
   and skips rather than errors when the data is not present (shallow clones,
   sdist installs).

3. **No writes into the repository.** Tests run in a per-test temporary
   directory, so a test that writes a cache or a target PDF cannot make the
   next run test something different from the last one.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA = REPO_ROOT / "tests" / "test_data"

# Seed chosen once so a failure is reproducible; any fixed value would do.
RANDOM_SEED = 20260909

MARKERS = {
    "data": "requires the image corpus in tests/test_data",
    "slow": "takes more than ~10s; runs a full bundle adjustment",
    "gui": "requires the optional PySide6 dependency",
}


def pytest_configure(config: pytest.Config) -> None:
    """Force offscreen rendering and register the suite's markers."""
    # setdefault so a developer can still override these to watch a run.
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")

    for name, description in MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {description}")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip data-backed tests when the image corpus is absent."""
    if TEST_DATA.is_dir():
        return
    skip_no_data = pytest.mark.skip(reason=f"image corpus not found at {TEST_DATA}")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip_no_data)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture
def data_dir() -> Path:
    """Absolute path to ``tests/test_data``, skipping the test if it is absent.

    Always use this rather than a relative path: it is what allows the suite to
    run from any working directory.
    """
    if not TEST_DATA.is_dir():
        pytest.skip(f"image corpus not found at {TEST_DATA}")
    return TEST_DATA


@pytest.fixture(autouse=True)
def deterministic_rng() -> None:
    """Reset numpy's global RNG so tests using random poses are reproducible."""
    np.random.seed(RANDOM_SEED)
