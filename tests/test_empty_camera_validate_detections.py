"""
Regression test: camera_calibrator.validate_detections() must not crash
when one or more cameras in cam_names have zero detections across all images,
or when the entire detection set has zero images (max_ims == 0).

Background: validate_detections() previously indexed
cam_list.get_data()[0, 0] to recover the camera index. When a camera had
no detections, get(cam=name) returns a TargetDetection with data=None, so
get_data()[0, 0] raised TypeError. The fix uses the enumerate index over
get_cam_list() (which preserves cam_names order), so empty cameras report
0% instead of crashing. A second guard was added for the all-empty case
where max_ims == 0 (division by zero) and board_fraction[cam] is absent
(KeyError) — both now report 0% rather than crashing.

UK english, minimal scope — mirrors tests/test_empty_camera_features_per_im_per_cam.py.
"""
import numpy as np

from pyCamSet.calibration_targets.target_detections import (
    TargetDetection,
    ImageDetection,
)
from pyCamSet.calibration.camera_calibrator import validate_detections


class _StubTarget:
    """Minimal stand-in for AbstractTarget: validate_detections only reads
    target.point_data.shape[-2] (corners_per_face)."""

    def __init__(self, corners_per_face: int):
        # point_data's last axis is the per-face corner count.
        self.point_data = np.zeros((corners_per_face, corners_per_face))


def _make_detection(cam_names, per_cam, max_ims=None):
    """
    Build a TargetDetection from a dict of {cam_name: [(global_im_num, keys, points)]}.
    Cameras in cam_names that are absent from per_cam get zero detections.
    If max_ims is given, force it on the resulting detection (all-empty case).
    """
    det = TargetDetection(cam_names=cam_names)
    for cam_name, im_entries in per_cam.items():
        for global_im_num, keys, points in im_entries:
            keys = np.asarray(keys, dtype=float)
            points = np.asarray(points, dtype=float)
            img_det = ImageDetection(keys=keys, image_points=points)
            det.add_detection(cam_name, global_im_num, img_det)
    det.get_data()  # flush the add_detection buffer into _data
    if max_ims is not None:
        det.max_ims = max_ims
    return det


def test_validate_detections_with_empty_camera():
    """One camera with zero detections must report 0%, not crash."""
    cam_names = ["cam_a", "cam_empty", "cam_c"]
    per_cam = {
        "cam_a": [
            (0, [0, 1], [[10.0, 20.0], [30.0, 40.0]]),
            (1, [0, 1], [[11.0, 21.0], [31.0, 41.0]]),
        ],
        # cam_empty intentionally absent — zero detections across all images
        "cam_c": [
            (0, [2, 3], [[50.0, 60.0], [70.0, 80.0]]),
        ],
    }
    det = _make_detection(cam_names, per_cam)
    target = _StubTarget(corners_per_face=4)

    # Must not raise.
    validate_detections(det, target)


def test_validate_detections_all_empty_zero_max_ims():
    """Every camera empty and max_ims == 0 must report 0%, not crash
    with ZeroDivisionError or KeyError."""
    cam_names = ["cam_a", "cam_b"]
    det = TargetDetection(cam_names=cam_names, max_ims=0)
    target = _StubTarget(corners_per_face=4)

    # Must not raise.
    validate_detections(det, target)


def test_validate_detections_no_regression_all_populated():
    """All cameras populated — the fix must not change behaviour."""
    cam_names = ["cam_a", "cam_b"]
    per_cam = {
        "cam_a": [(0, [0, 1], [[10.0, 20.0], [30.0, 40.0]])],
        "cam_b": [(0, [0, 1], [[50.0, 60.0], [70.0, 80.0]]),
                  (1, [2],    [[90.0, 100.0]])],
    }
    det = _make_detection(cam_names, per_cam)
    target = _StubTarget(corners_per_face=4)

    # Must not raise.
    validate_detections(det, target)


if __name__ == "__main__":
    test_validate_detections_with_empty_camera()
    test_validate_detections_all_empty_zero_max_ims()
    test_validate_detections_no_regression_all_populated()
    print("All empty-camera validate_detections tests passed")