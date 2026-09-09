"""Shared pytest configuration for the pyCamSet test suite.

Four things happen here that the suite previously relied on the developer's
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

4. **Optional-environment gates.** Two regression tests depend on optional
   runtime capabilities rather than library code: ``test_aruco2_backend.py``
   imports the optional compiled ``aruco2`` package at module level, and
   ``bundle_correctness_test.py`` calls the legacy ``cv2.aruco`` ChArUco
   calibration API that is absent across the supported ``>=4.8,<5`` band.
   Both are gated here so a fresh checkout collects cleanly on every
   supported platform instead of erroring at collection time.
"""

from __future__ import annotations

import os
import warnings
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

# --- Optional-environment probes --------------------------------------------
# ``aruco2`` is an optional compiled backend package; its regression tests
# import it at module level, so the file cannot be collected (let alone
# skipped via a marker) when the package is absent.  This mirrors the guarded
# import used by ``pyCamSet.calibration_targets.aruco2_detection``.
try:
    import aruco2  # noqa: F401

    ARUCO2_AVAILABLE = True
except ImportError:
    aruco2 = None  # type: ignore[assignment]
    ARUCO2_AVAILABLE = False

# ``cv2.aruco.calibrateCameraCharucoExtended`` is a legacy API that is absent
# across the whole supported ``opencv-python>=4.8,<5`` band.  cv2 is a
# mandatory pyCamSet dependency, so a missing cv2 only matters here for this
# one legacy-API regression test.
try:
    import cv2.aruco  # noqa: F401

    HAS_LEGACY_CHARUCO_CALIBRATION = hasattr(
        cv2.aruco, "calibrateCameraCharucoExtended"
    )
except ImportError:
    HAS_LEGACY_CHARUCO_CALIBRATION = False

# Files that must not be collected because they import optional packages at
# module level: a skip marker can never fire for those (collection aborts
# before items exist), so a visible warning from ``pytest_configure`` carries
# the skip reason instead.
collect_ignore_glob = [] if ARUCO2_AVAILABLE else ["test_aruco2_backend.py"]

# Regression test gated on the legacy ChArUco calibration API availability.
BUNDLE_TEST_BASENAME = "bundle_correctness_test.py"


def pytest_configure(config: pytest.Config) -> None:
    """Force offscreen rendering and register the suite's markers."""
    # setdefault so a developer can still override these to watch a run.
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")

    for name, description in MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {description}")

    if not ARUCO2_AVAILABLE:
        warnings.warn(
            "tests/test_aruco2_backend.py is not collected: it imports the "
            "optional compiled 'aruco2' backend package at module level, "
            "which is not installed in this environment.",
            UserWarning,
            stacklevel=2,
        )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip tests whose runtime environment is missing their prerequisites."""
    skip_no_data = pytest.mark.skip(reason=f"image corpus not found at {TEST_DATA}")
    for item in items:
        if not TEST_DATA.is_dir() and "data" in item.keywords:
            item.add_marker(skip_no_data)

    if not HAS_LEGACY_CHARUCO_CALIBRATION:
        skip_legacy_api = pytest.mark.skip(
            reason=(
                "requires the legacy ChArUco calibration API "
                "(cv2.aruco.calibrateCameraCharucoExtended), which this "
                "environment's opencv-python does not provide; the API is "
                "absent across the supported >=4.8,<5 release band"
            )
        )
        for item in items:
            if item.path.name == BUNDLE_TEST_BASENAME:
                item.add_marker(skip_legacy_api)


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
