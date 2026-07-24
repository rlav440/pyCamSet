"""
Test that cairo is a LAZY import — importing target modules and calling
find_in_image() succeeds without cairo installed, and the friendly error
only appears when calling export/generation functions (save_to_pdf with
data_format="vector").

All cairo-blocked tests run in SEPARATE SUBPROCESSES to avoid polluting the
main pytest process's sys.modules (which broke pickling in later test files
in a prior version of this test — see the test-isolation regression fix).
"""

import subprocess
import sys
import os


def test_cairo_present_import_succeeds():
    """When cairo IS installed (the normal case on this machine), importing
    target_charuco should succeed with no behavior change."""
    import pyCamSet.calibration_targets.target_charuco
    assert hasattr(pyCamSet.calibration_targets.target_charuco, "ChArUco")


def test_cairo_present_import_ccube_succeeds():
    """target_Ccube should also import normally when cairo is present."""
    import pyCamSet.calibration_targets.target_Ccube
    assert hasattr(pyCamSet.calibration_targets.target_Ccube, "Ccube")


# --- Subprocess script: block cairosvg, then test import + find_in_image ---
# Runs in a fresh Python process with its own sys.modules.
_BLOCK_CAIRO_IMPORT_AND_TEST = r"""
import builtins, sys

original_import = builtins.__import__

def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in ("cairosvg", "cairocffi"):
        raise OSError('no library called "cairo-2" was found')
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = mock_import

# Pre-populate sys.modules with None to block even cached lookups
sys.modules["cairosvg"] = None
sys.modules["cairocffi"] = None

import numpy as np
from cv2 import aruco

# --- Test 1: import succeeds without cairo ---
try:
    import pyCamSet.calibration_targets.target_charuco as tc
    print("IMPORT_CHARUCO_OK")
except OSError as e:
    print(f"IMPORT_CHARUCO_FAILED: {e}")
    sys.exit(1)

try:
    import pyCamSet.calibration_targets.target_Ccube as tcc
    print("IMPORT_CCUBE_OK")
except OSError as e:
    print(f"IMPORT_CCUBE_FAILED: {e}")
    sys.exit(1)

# --- Test 2: construct ChArUco and call find_in_image ---
try:
    target = tc.ChArUco(
        num_squares_x=5, num_squares_y=7, square_size=30,
        marker_fraction=0.75, a_dict=aruco.DICT_4X4_1000,
    )
    # Generate a small test image with a board
    board_img = target.board.generateImage((800, 1120))
    bg = np.zeros((1200, 1920), dtype=np.uint8)
    h, w = board_img.shape[:2]
    bg[(1200-h)//2:(1200+h)//2, (1920-w)//2:(1920+w)//2] = board_img
    detection = target.find_in_image(bg)
    n_points = len(detection.image_points) if detection.has_data else 0
    print(f"FIND_IN_IMAGE_OK: {n_points} points")
except Exception as e:
    print(f"FIND_IN_IMAGE_FAILED: {type(e).__name__}: {e}")
    sys.exit(1)

# --- Test 3: cairosvg was never imported as a side effect ---
if "cairosvg" in sys.modules and sys.modules["cairosvg"] is not None:
    print("CAIROSVG_LEAKED: cairosvg was imported as a side effect!")
    sys.exit(1)
else:
    print("CAIROSVG_NOT_LEAKED: cairosvg was never imported")

# --- Test 4: calling save_to_pdf with data_format="vector" raises friendly error ---
try:
    from pathlib import Path
    target.save_to_pdf(f_out=Path("_test_charuco_vec.pdf"), data_format="vector")
    print("SAVE_TO_PDF_VECTOR_SUCCEEDED_UNEXPECTEDLY")
    sys.exit(1)
except OSError as e:
    msg = str(e)
    if "conda install -c conda-forge cairo" in msg and "pyCamSet" in msg:
        print("FRIENDLY_ERROR_ON_SAVE_TO_PDF_OK")
    else:
        print(f"FRIENDLY_ERROR_MISSING: {msg}")
        sys.exit(1)
except Exception as e:
    print(f"UNEXPECTED_EXCEPTION_ON_SAVE_TO_PDF: {type(e).__name__}: {e}")
    sys.exit(1)

print("ALL_SUBPROCESS_CHECKS_PASSED")
sys.exit(0)
"""


def test_import_and_find_in_image_without_cairo():
    """Gate 1 + Gate 4: importing target_charuco and calling find_in_image()
    succeeds in a subprocess where cairosvg is blocked, and cairosvg is never
    imported as a side effect."""
    result = subprocess.run(
        [sys.executable, "-c", _BLOCK_CAIRO_IMPORT_AND_TEST],
        capture_output=True, text=True, timeout=60,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 0, (
        f"Subprocess failed (exit {result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "IMPORT_CHARUCO_OK" in result.stdout
    assert "IMPORT_CCUBE_OK" in result.stdout
    assert "FIND_IN_IMAGE_OK" in result.stdout
    assert "CAIROSVG_NOT_LEAKED" in result.stdout
    assert "FRIENDLY_ERROR_ON_SAVE_TO_PDF_OK" in result.stdout
    assert "ALL_SUBPROCESS_CHECKS_PASSED" in result.stdout