"""Accuracy regression for self-calibration (free target points) on ChArUco."""

from multiprocessing import cpu_count

import numpy as np
import pytest

from pyCamSet import ChArUco, calibrate_cameras
from pyCamSet.calibration.camera_calibrator import run_bundle_adjustment
from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler

# Self-calibration frees the target geometry, so it should beat the fixed-target
# ChArUco baseline of 1.8 px.
MAX_MEAN_REPROJECTION_PX = 1.07


@pytest.mark.data
@pytest.mark.slow
def test_self_calibration_charuco(data_dir):
    """Self-calibration must improve on, and stay within, its baseline."""
    target = ChArUco(20, 20, 4)

    cams = calibrate_cameras(
        f_loc=data_dir / "calibration_charuco",
        calibration_target=target,
        save=False,
    )

    param_handler = SelfBundleHandler(
        detection=cams.calibration_handler.detection, target=target, camset=cams
    )
    param_handler.set_from_templated_camset(cams)

    _, final_cams = run_bundle_adjustment(
        param_handler=param_handler, threads=cpu_count()
    )

    mean_reprojection = np.mean(
        np.linalg.norm(np.reshape(final_cams.calibration_result, (-1, 2)), axis=1)
    )
    assert mean_reprojection < MAX_MEAN_REPROJECTION_PX, (
        f"ChArUco self-calibration reprojection error {mean_reprojection:.3f} px "
        f"exceeds the {MAX_MEAN_REPROJECTION_PX} px baseline."
    )
