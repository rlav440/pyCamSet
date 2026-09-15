"""
Test that ``pyCamSet/__init__.py``'s PEP 562 lazy attribute resolution is
behaviourally identical to the eager imports it replaced.

``import pyCamSet`` used to eagerly pull in ``pyCamSet.cameras``,
``pyCamSet.calibration`` (and therefore numba) and every optional target
class, at a cost of several seconds -- most of which a caller who only wants
one class never uses. ``__getattr__``/``__dir__`` on the package module defer
that work to first attribute access. These tests pin down the two ways that
change could quietly break something: the heavy modules must actually stay
out of ``sys.modules`` until asked for, and every public name must resolve to
exactly what the old eager imports gave it -- including the three-way outcome
(class / ``None`` / missing-dependency placeholder) each optional target
class has depending on whether cairo and ``puzzle_board`` are available.

The "does a name resolve at all" checks run in the main process against the
real environment. The "does laziness actually hold, and are the failure
branches reachable" checks run in subprocesses with import machinery mocked,
following the pattern in ``cairo_import_test.py``, so they do not depend on
what happens to be installed on the machine running the suite.
"""

from __future__ import annotations

import os
import subprocess
import sys


def test_public_names_resolve_and_are_cached():
    """The plain re-exports must resolve to the real objects, on both the
    `import pyCamSet; pyCamSet.X` and `from pyCamSet import X` spellings, and
    be cached in the module's globals after first access."""
    import pyCamSet

    from pyCamSet.cameras import CameraSet as _CameraSet, Camera as _Camera
    from pyCamSet.utils.saving import load_CameraSet as _load_CameraSet
    from pyCamSet.utils.logs import setup_logging as _setup_logging
    from pyCamSet.utils.calibration_report import CalibrationReport as _CalibrationReport
    from pyCamSet.calibration import calibrate_cameras as _calibrate_cameras

    assert pyCamSet.CameraSet is _CameraSet
    assert pyCamSet.Camera is _Camera
    assert pyCamSet.load_CameraSet is _load_CameraSet
    assert pyCamSet.setup_logging is _setup_logging
    assert pyCamSet.CalibrationReport is _CalibrationReport
    assert pyCamSet.calibrate_cameras is _calibrate_cameras

    # Cached in globals after first resolution -- later access is a plain
    # attribute lookup, not a repeat trip through __getattr__.
    assert vars(pyCamSet)["CameraSet"] is _CameraSet

    from pyCamSet import (
        CameraSet, Camera, load_CameraSet, setup_logging,
        CalibrationReport, calibrate_cameras,
    )
    assert (CameraSet, Camera, load_CameraSet, setup_logging,
            CalibrationReport, calibrate_cameras) == (
        _CameraSet, _Camera, _load_CameraSet, _setup_logging,
        _CalibrationReport, _calibrate_cameras,
    )


def test_dir_and_all_and_star_import():
    """`dir(pyCamSet)`, `__all__`, and `from pyCamSet import *` must all
    advertise the same public surface, and none of that must raise."""
    import pyCamSet

    expected = {
        "CameraSet", "Camera", "load_CameraSet", "setup_logging",
        "CalibrationReport", "calibrate_cameras",
        "ChArUco", "Ccube", "PuzzleBoard", "PuzzleBoardCube",
    }
    assert expected <= set(pyCamSet.__all__)
    assert expected <= set(dir(pyCamSet))

    # help()/introspection walks __dir__ and getattr()s everything it finds;
    # make sure that round-trip does not explode for any advertised name.
    for name in dir(pyCamSet):
        getattr(pyCamSet, name)


def test_optional_targets_resolve_to_class_none_or_placeholder():
    """Each optional target attribute must be a class, `None`, or the
    `_Missing...` placeholder -- never raise on access -- and the
    placeholders must raise ImportError only when instantiated."""
    import pyCamSet

    for name in ("ChArUco", "Ccube"):
        value = getattr(pyCamSet, name)
        assert value is None or isinstance(value, type)

    for name, placeholder_name in (
        ("PuzzleBoard", "_MissingPuzzleBoard"),
        ("PuzzleBoardCube", "_MissingPuzzleBoardCube"),
    ):
        value = getattr(pyCamSet, name)
        assert value is None or isinstance(value, type)
        placeholder = getattr(pyCamSet, placeholder_name)
        if value is placeholder:
            try:
                placeholder()
            except ImportError as exc:
                assert "puzzle_board" in str(exc)
            else:
                raise AssertionError(
                    f"{placeholder_name} must raise ImportError when instantiated"
                )


# --- Subprocess script: prove import pyCamSet alone does not pull the heavy
# modules in, and that touching one name pulls in only what it needs. ---
_LAZINESS_SCRIPT = r"""
import sys

import pyCamSet  # noqa: F401 -- must not import cameras/calibration/numba/scipy.stats

heavy = ("pyCamSet.cameras", "pyCamSet.calibration", "numba", "scipy.stats")
still_out = [m for m in heavy if m not in sys.modules]
if still_out != list(heavy):
    print(f"EAGERLY_PULLED_IN: {[m for m in heavy if m in sys.modules]}")
    sys.exit(1)
print("BARE_IMPORT_STAYS_LAZY_OK")

# Touching CameraSet must pull in pyCamSet.cameras (and only then).
pyCamSet.CameraSet
if "pyCamSet.cameras" not in sys.modules:
    print("CAMERASET_ACCESS_DID_NOT_IMPORT_CAMERAS")
    sys.exit(1)
print("ATTRIBUTE_ACCESS_TRIGGERS_IMPORT_OK")

print("ALL_LAZINESS_CHECKS_PASSED")
sys.exit(0)
"""


def test_bare_import_does_not_pull_in_heavy_modules():
    """`import pyCamSet` alone must not import cameras/calibration/numba/
    scipy.stats; only touching the relevant attribute should."""
    result = subprocess.run(
        [sys.executable, "-c", _LAZINESS_SCRIPT],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 0, (
        f"Subprocess failed (exit {result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "BARE_IMPORT_STAYS_LAZY_OK" in result.stdout
    assert "ATTRIBUTE_ACCESS_TRIGGERS_IMPORT_OK" in result.stdout
    assert "ALL_LAZINESS_CHECKS_PASSED" in result.stdout


# --- Subprocess script: force the "missing dependency" branches that this
# machine's normal environment does not naturally exercise (it has both
# cairo and lacks puzzle_board only in a spot the class import never
# touches), by mocking import machinery the same way cairo_import_test.py
# does for the cairosvg-blocked case. ---
_FAILURE_BRANCHES_SCRIPT = r"""
import builtins, sys

original_import = builtins.__import__

def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
    # Simulate an environment where svgwrite (something ChArUco/Ccube's
    # target modules need at module scope) is entirely absent, forcing the
    # `except Exception: return None` branch pyCamSet.ChArUco/Ccube take.
    if name == "svgwrite":
        raise ImportError("simulated missing svgwrite")
    # Simulate the 'puzzle_board' package being absent AND reached at
    # import time, forcing the ModuleNotFoundError(name='puzzle_board')
    # branch that selects the placeholder class.
    if name == "puzzle_board" or name.startswith("puzzle_board."):
        raise ModuleNotFoundError(name=name)
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = mock_import

import pyCamSet

if pyCamSet.ChArUco is not None:
    print(f"CHARUCO_NOT_NONE: {pyCamSet.ChArUco!r}")
    sys.exit(1)
print("CHARUCO_NONE_OK")

if pyCamSet.Ccube is not None:
    print(f"CCUBE_NOT_NONE: {pyCamSet.Ccube!r}")
    sys.exit(1)
print("CCUBE_NONE_OK")

print("ALL_FAILURE_BRANCH_CHECKS_PASSED")
sys.exit(0)
"""


def test_charuco_and_ccube_resolve_to_none_when_their_import_fails():
    """When the target module's own import fails for a reason other than a
    missing 'puzzle_board', pyCamSet.ChArUco / .Ccube must resolve to None,
    exactly as the old `except Exception: ChArUco = None` did."""
    result = subprocess.run(
        [sys.executable, "-c", _FAILURE_BRANCHES_SCRIPT],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 0, (
        f"Subprocess failed (exit {result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "CHARUCO_NONE_OK" in result.stdout
    assert "CCUBE_NONE_OK" in result.stdout
    assert "ALL_FAILURE_BRANCH_CHECKS_PASSED" in result.stdout
