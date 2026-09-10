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
    "needs_jit": "asserts on numba's compiled behaviour; invalid with NUMBA_DISABLE_JIT",
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
    """Skip tests whose prerequisites the current environment does not meet."""
    have_data = TEST_DATA.is_dir()
    skip_no_data = pytest.mark.skip(reason=f"image corpus not found at {TEST_DATA}")

    # The coverage job runs with NUMBA_DISABLE_JIT=1 so that the @njit kernel
    # bodies are interpreted, and therefore visible to coverage.  With JIT off,
    # numba's njit returns the plain function rather than a dispatcher, so
    # anything reaching for .py_func (the codegen bounds guard) has nothing to
    # cross-check against and cannot fail the way it is asserted to.
    jit_disabled = os.environ.get("NUMBA_DISABLE_JIT", "") not in ("", "0")
    skip_no_jit = pytest.mark.skip(reason="NUMBA_DISABLE_JIT is set; numba is not compiling")

    for item in items:
        if not have_data and "data" in item.keywords:
            item.add_marker(skip_no_data)
        if jit_disabled and "needs_jit" in item.keywords:
            item.add_marker(skip_no_jit)


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


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run each test in its own directory so nothing writes into the repo.

    Anything a test saves without an absolute path -- a target PDF, a cached
    ``.camset``, a debug plot -- lands in ``tmp_path`` and is discarded.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Synthetic cameras
#
# Phase 1 of the coverage work tests the pure surface -- containers, geometry,
# persistence -- which needs cameras but no images.  These fixtures build a
# small rig analytically so the tests stay in the millisecond range and do not
# depend on the image corpus.
# ---------------------------------------------------------------------------

# A plausible 640x480 pinhole camera; f and principal point are distinct in x
# and y so a test that transposes them fails instead of silently passing.
REF_INTRINSIC = np.array(
    [
        [800.0, 0.0, 320.0],
        [0.0, 750.0, 240.0],
        [0.0, 0.0, 1.0],
    ]
)
REF_RES = [640, 480]


def make_camera(name="cam", translation=(0.0, 0.0, 0.0), distortion=None, res=None):
    """One synthetic pinhole camera at *translation*, looking down +z."""
    from pyCamSet import Camera

    extrinsic = np.eye(4)
    extrinsic[:3, 3] = translation
    return Camera(
        extrinsic=extrinsic,
        intrinsic=REF_INTRINSIC.copy(),
        res=list(REF_RES) if res is None else list(res),
        distortion_coefs=np.zeros(5) if distortion is None else np.asarray(distortion),
        name=name,
    )


@pytest.fixture
def synthetic_camset():
    """A three camera rig, spaced along x, with no distortion.

    Names are deliberately not in sorted order relative to their positions, so
    a test cannot pass by accidentally relying on dict ordering matching name
    ordering.
    """
    from pyCamSet import CameraSet

    cams = {
        "left": make_camera("left", translation=(-0.05, 0.0, 0.0)),
        "centre": make_camera("centre", translation=(0.0, 0.0, 0.0)),
        "right": make_camera("right", translation=(0.05, 0.0, 0.0)),
    }
    return CameraSet(camera_dict=cams)


@pytest.fixture
def world_points():
    """A fixed cloud of world points that all three synthetic cameras image."""
    return np.array(
        [
            [0.0, 0.0, 1.0],
            [0.01, 0.0, 1.0],
            [0.0, 0.01, 1.0],
            [-0.02, 0.015, 1.2],
            [0.03, -0.01, 0.9],
        ]
    )


# ---------------------------------------------------------------------------
# Shared detections
#
# Detecting the image corpus is the expensive part of every end-to-end test --
# roughly 0.8s for the ChArUco set and 4.8s for the Ccube one.  These fixtures
# do it once per session and hand the same TargetDetection to every test that
# needs real data, so the staged tests below the calibration entry point cost
# almost nothing to add.
#
# Deliberately session scoped rather than written to disk: an on-disk cache
# outlives the OpenCV version that produced it, and a stale one would quietly
# hide exactly the detection changes tests/test_detection_consistency.py exists
# to catch.  Nothing here survives the run.
# ---------------------------------------------------------------------------

# The board the checked-in ChArUco images were taken of.
CHARUCO_ARGS = dict(num_squares_x=20, num_squares_y=20, square_size=4, legacy=True)
# ...and the cube for the Ccube images.
CCUBE_ARGS = dict(n_points=10, length=40, border_fraction=0.2)


@pytest.fixture(scope="session")
def session_data_dir() -> Path:
    """Session scoped ``data_dir``, for fixtures that outlive a single test."""
    if not TEST_DATA.is_dir():
        pytest.skip(f"image corpus not found at {TEST_DATA}")
    return TEST_DATA


@pytest.fixture(scope="session")
def charuco_target():
    from pyCamSet import ChArUco

    return ChArUco(**CHARUCO_ARGS)


@pytest.fixture(scope="session")
def ccube_target():
    from cv2 import aruco

    from pyCamSet import Ccube

    return Ccube(aruco_dict=aruco.DICT_6X6_1000, **CCUBE_ARGS)


def _detect(image_dir: Path, target):
    """Detect the corpus once, without touching the on-disk cache."""
    from pyCamSet.calibration.camera_calibrator import detect_datapoints_in_imfile

    return detect_datapoints_in_imfile(
        f_loc=image_dir, caching=False, calibration_target=target, threads=1
    )


@pytest.fixture(scope="session")
def charuco_detections(session_data_dir, charuco_target):
    """(TargetDetection, camera resolutions) for the ChArUco corpus."""
    return _detect(session_data_dir / "calibration_charuco", charuco_target)


@pytest.fixture(scope="session")
def ccube_detections(session_data_dir, ccube_target):
    """(TargetDetection, camera resolutions) for the Ccube corpus."""
    return _detect(session_data_dir / "calibration_ccube", ccube_target)


@pytest.fixture
def charuco_problem(session_data_dir, charuco_target, charuco_detections):
    """A calibrated-enough ChArUco problem: target, detections, initial cameras.

    The initial calibration is per test rather than session scoped because the
    handlers mutate the camera set they are given.
    """
    from pyCamSet.calibration.camera_calibrator import run_initial_calibration

    detections, camera_res = charuco_detections
    cams = run_initial_calibration(detections, charuco_target, camera_res, save=False)
    cams.set_resolutions_from_file(floc=session_data_dir / "calibration_charuco")
    return charuco_target, detections, cams
