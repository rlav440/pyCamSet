"""
Regression test: TargetDetection.features_per_im_per_cam() must not crash
when one or more cameras in cam_names have zero detections across all images.

Background: features_per_im_per_cam() previously indexed
cam_list.get_data()[0, 0] to recover the camera index. When a camera had
no detections, get(cam=name) returns a TargetDetection with data=None, so
get_data()[0, 0] raised TypeError. The fix uses the enumerate index over
get_cam_list() (which preserves cam_names order), so empty cameras report
a zero column instead of crashing.

UK English, minimal scope — mirrors the existing tests/test_*.py style.
"""
import numpy as np

from pyCamSet.calibration_targets.target_detections import (
    TargetDetection,
    ImageDetection,
)


def _make_detection(cam_names, per_cam):
    """
    Build a TargetDetection from a dict of {cam_name: [(global_im_num, keys, points)]}.

    keys is a list of 1-D key arrays; points is a matching list of [x, y] pairs.
    Cameras in cam_names that are absent from per_cam get zero detections.
    """
    det = TargetDetection(cam_names=cam_names)
    for cam_name, im_entries in per_cam.items():
        for global_im_num, keys, points in im_entries:
            keys = np.asarray(keys, dtype=float)
            points = np.asarray(points, dtype=float)
            img_det = ImageDetection(keys=keys, image_points=points)
            det.add_detection(cam_name, global_im_num, img_det)
    # Flush the add_detection buffer into _data so max_ims / get() see it.
    # In the real pipeline this happens during detection building; here we
    # force it so the object is in the post-build state features_per_im_per_cam
    # expects.
    det.get_data()
    return det


def test_features_per_im_per_cam_with_empty_camera():
    """One camera with zero detections must produce a zero column, not a crash."""
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

    block = det.features_per_im_per_cam()

    # Shape: (n_images, n_cams) — the empty camera contributes a zero column.
    assert block.shape[0] == 2, f"expected 2 images, got {block.shape[0]}"
    assert block.shape[1] == 3, f"expected 3 cameras, got {block.shape[1]}"

    # cam_a (index 0): 2 detections in image 0, 2 in image 1.
    assert block[0, 0] == 2, f"cam_a image 0 expected 2, got {block[0, 0]}"
    assert block[1, 0] == 2, f"cam_a image 1 expected 2, got {block[1, 0]}"

    # cam_empty (index 1): all zeros.
    assert np.all(block[:, 1] == 0), f"empty camera column should be zero, got {block[:, 1]}"

    # cam_c (index 2): 2 detections in image 0, 0 in image 1.
    assert block[0, 2] == 2, f"cam_c image 0 expected 2, got {block[0, 2]}"
    assert block[1, 2] == 0, f"cam_c image 1 expected 0, got {block[1, 2]}"


def test_features_per_im_per_cam_all_empty_camera():
    """A TargetDetection where every camera has zero detections must return a zero block."""
    cam_names = ["cam_a", "cam_b"]
    det = TargetDetection(cam_names=cam_names, max_ims=3)

    block = det.features_per_im_per_cam()

    assert block.shape == (3, 2), f"expected (3, 2), got {block.shape}"
    assert np.all(block == 0), f"all-zero block expected, got {block}"


def test_features_per_im_per_cam_no_regression_all_populated():
    """All cameras populated — the fix must not change the pre-fix result."""
    cam_names = ["cam_a", "cam_b"]
    per_cam = {
        "cam_a": [(0, [0, 1], [[10.0, 20.0], [30.0, 40.0]])],
        "cam_b": [(0, [0, 1], [[50.0, 60.0], [70.0, 80.0]]),
                  (1, [2],    [[90.0, 100.0]])],
    }
    det = _make_detection(cam_names, per_cam)

    block = det.features_per_im_per_cam()

    assert block.shape[0] == 2, f"expected 2 images, got {block.shape[0]}"
    assert block.shape[1] == 2, f"expected 2 cameras, got {block.shape[1]}"
    assert block[0, 0] == 2, f"cam_a image 0 expected 2, got {block[0, 0]}"
    assert block[0, 1] == 2, f"cam_b image 0 expected 2, got {block[0, 1]}"
    assert block[1, 1] == 1, f"cam_b image 1 expected 1, got {block[1, 1]}"


if __name__ == "__main__":
    test_features_per_im_per_cam_with_empty_camera()
    test_features_per_im_per_cam_all_empty_camera()
    test_features_per_im_per_cam_no_regression_all_populated()
    print("All empty-camera features_per_im_per_cam tests passed")