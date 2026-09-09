"""Self-calibration and target-detection container regressions."""

from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pytest

from pyCamSet import ChArUco, calibrate_cameras
from pyCamSet.calibration.camera_calibrator import run_bundle_adjustment
from pyCamSet.calibration_targets.create_charuco import build_charuco
from pyCamSet.calibration_targets.target_detections import ImageDetection, TargetDetection
from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler

MAX_MEAN_REPROJECTION_PX = 1.07


@pytest.mark.data
@pytest.mark.slow
def test_self_calibration_charuco(data_dir):
    """Self-calibration must stay within its measured baseline."""
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
    _, final_cams = run_bundle_adjustment(param_handler=param_handler, threads=cpu_count())
    mean_reprojection = np.mean(
        np.linalg.norm(np.reshape(final_cams.calibration_result, (-1, 2)), axis=1)
    )
    assert mean_reprojection < MAX_MEAN_REPROJECTION_PX


def test_target_detection_appends_rows_without_flattening() -> None:
    one_point = ImageDetection(keys=[3], image_points=[[12.0, 8.0]])
    detections = TargetDetection(["cam0", "cam1"])
    detections.add_detection("cam0", 0, one_point)
    detections.add_detection("cam1", 0, one_point)
    assert detections.get_data().shape == (2, 5)
    detections.add_detection("cam0", 1, one_point)
    np.testing.assert_array_equal(
        detections.get_data()[:, :2], np.array([[0, 0], [1, 0], [0, 1]])
    )
    np.testing.assert_array_equal(
        detections.features_per_im_per_cam(), np.array([[1.0, 1.0], [1.0, 0.0]])
    )


def test_empty_and_mismatched_detection_inputs_are_explicit() -> None:
    empty = ImageDetection()
    assert not empty.has_data
    assert empty.data_len == 0
    assert TargetDetection(["cam0"]).get_key_list() == []
    with pytest.raises(ValueError, match="same length"):
        ImageDetection(keys=[1, 2], image_points=[[0.0, 0.0]])
    left = TargetDetection(["cam0"])
    right = TargetDetection(["cam1"])
    with pytest.raises(ValueError, match="consistent camera names"):
        _ = left + right


def test_empty_target_folder_raises_instead_of_silently_continuing(tmp_path: Path) -> None:
    target = build_charuco(5, 7, 4)
    with pytest.raises(ValueError, match="No images were found"):
        target.find_in_imfolder(tmp_path / "cam0", ["cam0"], threads=1)