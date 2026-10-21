"""Verify opt-in Phase 1 ChArUco trace capture.
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from pyCamSet import ChArUco


def test_synthetic_charuco_trace_is_opt_in_and_does_not_change_detection():
    target = ChArUco(num_squares_x=5, num_squares_y=7, square_size=4)
    image = target._render_board(px_per_mm=20)

    target.phase1_trace_enabled = False
    untraced = target.find_in_image(image)
    assert getattr(target, "_last_phase1_trace", None) is None

    target.phase1_trace_enabled = True
    traced = target.find_in_image(image)
    assert traced.has_data == untraced.has_data
    np.testing.assert_array_equal(traced.keys, untraced.keys)
    np.testing.assert_array_equal(traced.image_points, untraced.image_points)
    assert target._last_phase1_trace
    assert target._last_phase1_trace[-1]["name"] == "charuco_corners"
    assert target._last_phase1_trace[-1]["corner_ids"]


def test_tracked_charuco_trace_is_bit_equal_to_untraced_detection():
    image_path = Path(__file__).parent / "test_data" / "calibration_charuco" / "3" / "cal1.jpg"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image is not None
    target = ChArUco(num_squares_x=20, num_squares_y=20, square_size=4)

    target.phase1_trace_enabled = False
    untraced = target.find_in_image(image)
    target.phase1_trace_enabled = True
    traced = target.find_in_image(image)

    assert traced.has_data == untraced.has_data
    np.testing.assert_array_equal(traced.keys, untraced.keys)
    np.testing.assert_array_equal(traced.image_points, untraced.image_points)
    assert target._last_phase1_trace[-1]["name"] == "charuco_corners"


def test_multiprocessing_trace_is_returned_to_parent_with_camera_image_index(tmp_path):
    target = ChArUco(num_squares_x=5, num_squares_y=7, square_size=4)
    target.phase1_trace_enabled = True
    camera_folder = tmp_path / "cam0"
    camera_folder.mkdir()
    image = target._render_board(px_per_mm=20)

    assert cv2.imwrite(str(camera_folder / "frame0.png"), image)
    assert cv2.imwrite(str(camera_folder / "frame1.png"), image)
    detections = target.find_in_imfolder(
        camera_folder, cam_names=["cam0"], threads=2)

    assert detections.max_ims == 2
    assert [(item["camera"], item["image_index"]) for item in
            target.phase1_trace_records] == [("cam0", 0), ("cam0", 1)]
    assert all(item["stages"][-1]["name"] == "charuco_corners"
               for item in target.phase1_trace_records)


def test_missing_worker_trace_is_recorded_as_missing_not_fabricated(tmp_path, monkeypatch):
    from pyCamSet.calibration_targets.core import abstract_target
    from pyCamSet.calibration_targets.core.target_detections import ImageDetection

    class TraceDroppingPool:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def starmap(self, _function, tasks):
            return [(task[1], task[2], ImageDetection(), None, False, None, None)
                    for task in tasks]

    monkeypatch.setattr(abstract_target.multiprocessing, "Pool", TraceDroppingPool)
    target = ChArUco(num_squares_x=5, num_squares_y=7, square_size=4)
    target.phase1_trace_enabled = True
    camera_folder = tmp_path / "cam0"
    camera_folder.mkdir()
    assert cv2.imwrite(str(camera_folder / "frame0.png"),
                       target._render_board(px_per_mm=20))

    target.find_in_imfolder(camera_folder, cam_names=["cam0"], threads=2)
    assert target.phase1_trace_records == [{
        "camera": "cam0", "image_index": 0,
        "trace_status": "missing", "stages": [],
    }]


def test_serial_unreadable_image_has_missing_trace_and_keeps_following_indices(tmp_path):
    target = ChArUco(num_squares_x=5, num_squares_y=7, square_size=4)
    target.phase1_trace_enabled = True
    camera_folder = tmp_path / "cam0"
    camera_folder.mkdir()
    (camera_folder / "frame0.png").write_bytes(b"not an image")
    assert cv2.imwrite(str(camera_folder / "frame1.png"),
                       target._render_board(px_per_mm=20))

    detections = target.find_in_imfolder(
        camera_folder, cam_names=["cam0"], threads=1)

    assert detections.max_ims == 2
    assert [(item["camera"], item["image_index"])
            for item in target.phase1_trace_records] == [("cam0", 0), ("cam0", 1)]
    assert target.phase1_trace_records[0] == {
        "camera": "cam0", "image_index": 0,
        "trace_status": "missing", "stages": [],
    }
    assert target.phase1_trace_records[1]["trace_status"] == "captured"
    assert target.phase1_trace_records[1]["stages"][-1]["name"] == "charuco_corners"
