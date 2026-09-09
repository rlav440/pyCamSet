"""Accuracy regression for self-calibration (free target points) on a Ccube."""

from multiprocessing import cpu_count

import numpy as np
import pytest
from cv2 import aruco

from pyCamSet import Ccube, calibrate_cameras
from pyCamSet.calibration.camera_calibrator import run_bundle_adjustment
from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler

# Freeing the cube's point geometry absorbs most of the printing and assembly
# error, so this is far tighter than the 5.10 px fixed-target Ccube baseline.
MAX_MEAN_REPROJECTION_PX = 0.50


@pytest.mark.data
@pytest.mark.slow
def test_self_calibration_ccube(data_dir):
    """Self-calibration on a Ccube must stay within its baseline.

    This previously loaded a cached ``self_calib_test.camset`` when one was
    present, which meant every run after the first skipped the initial
    calibration it was supposed to be exercising.  The calibration now always
    runs.
    """
    target = Ccube(
        n_points=10, length=40, aruco_dict=aruco.DICT_6X6_1000, border_fraction=0.2
    )

    cams = calibrate_cameras(data_dir / "calibration_ccube", target, save=False)

    param_handler = SelfBundleHandler(
        detection=cams.calibration_handler.detection,
        target=target,
        camset=cams,
        options={"max_nfev": 100},
    )
    param_handler.set_from_templated_camset(cams)

    _, final_cams = run_bundle_adjustment(
        param_handler=param_handler, threads=cpu_count()
    )

    mean_reprojection = np.mean(
        np.linalg.norm(np.reshape(final_cams.calibration_result, (-1, 2)), axis=1)
    )
    assert mean_reprojection < MAX_MEAN_REPROJECTION_PX, (
        f"Ccube self-calibration reprojection error {mean_reprojection:.3f} px "
        f"exceeds the {MAX_MEAN_REPROJECTION_PX} px baseline."
    )
