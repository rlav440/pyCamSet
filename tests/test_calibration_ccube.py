"""Ccube accuracy and headless target-factory regressions."""

from pathlib import Path

import numpy as np
import pytest
from cv2 import aruco

from pyCamSet import Ccube, calibrate_cameras
from pyCamSet.calibration_targets.create_Ccube import build_ccube, generate_ccube_target

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
    assert mean_reprojection < MAX_MEAN_REPROJECTION_PX


def test_ccube_texture_is_detector_compatible_and_detects(tmp_path: Path) -> None:
    target = build_ccube(n_points=5, length=40)
    assert len(target.boards) == 6
    assert target.point_data.shape[0] == 6
    assert all(texture.ndim == 2 for texture in target.textures)
    detection = target.find_in_image(target.textures[0])
    assert detection.has_data
    assert detection.data_len >= 4

    _, saved = generate_ccube_target(
        n_points=5,
        length=40,
        output_dir=tmp_path,
        file_name="nested/ccube.txt",
        export_kind="svg",
    )
    assert saved == (tmp_path / "nested/ccube.svg").resolve()
    assert saved.exists()
    assert saved.stat().st_size > 0