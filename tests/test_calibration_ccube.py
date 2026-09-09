"""Accuracy regression for a multi-camera Ccube calibration."""

import numpy as np
import pytest
from cv2 import aruco

from pyCamSet import Ccube, calibrate_cameras

# Baseline measured on the checked-in corpus; higher than the ChArUco baseline
# because the cube faces are seen at sharper angles.
MAX_MEAN_REPROJECTION_PX = 5.10


@pytest.mark.data
@pytest.mark.slow
def test_calibration_ccube(data_dir):
    """A full Ccube calibration must stay within its reprojection baseline."""
    target = Ccube(
        n_points=10, length=40, aruco_dict=aruco.DICT_6X6_1000, border_fraction=0.2
    )

    cams = calibrate_cameras(
        data_dir / "calibration_ccube",
        target,
        save=False,
        problem_options={"outliers": "n"},
    )

    mean_reprojection = np.mean(
        np.linalg.norm(np.reshape(cams.calibration_result, (-1, 2)), axis=1)
    )
    assert mean_reprojection < MAX_MEAN_REPROJECTION_PX, (
        f"Ccube calibration reprojection error {mean_reprojection:.3f} px "
        f"exceeds the {MAX_MEAN_REPROJECTION_PX} px baseline."
    )
