"""Accuracy regression for a multi-camera ChArUco calibration."""

import numpy as np
import pytest

from pyCamSet import ChArUco, calibrate_cameras

# Baseline measured on the checked-in corpus. Tighten it if the solver improves;
# a rise means a regression in detection, initialisation or bundle adjustment.
MAX_MEAN_REPROJECTION_PX = 1.8


@pytest.mark.data
@pytest.mark.slow
def test_calibration_charuco(data_dir):
    """A full ChArUco calibration must stay within its reprojection baseline."""
    target = ChArUco(20, 20, 4, legacy=True)

    cams = calibrate_cameras(
        f_loc=data_dir / "calibration_charuco",
        calibration_target=target,
        save=False,
        problem_options={"outliers": "n"},
    )

    mean_reprojection = np.mean(
        np.linalg.norm(np.reshape(cams.calibration_result, (-1, 2)), axis=1)
    )
    assert mean_reprojection < MAX_MEAN_REPROJECTION_PX, (
        f"ChArUco calibration reprojection error {mean_reprojection:.3f} px "
        f"exceeds the {MAX_MEAN_REPROJECTION_PX} px baseline."
    )
