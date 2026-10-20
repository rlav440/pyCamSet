"""
Every detector's points are recorded in the one pixel frame they are returned in.

All targets return points in the pixel-centre frame (integer coordinates are
pixel centres) whichever OpenCV is installed.  Detection caches written before
that, under OpenCV 4.x, recorded ChArUco-family corners as 'pixel-corner'
(half a pixel further on); such a cache must be a miss.  A detection read from
a resized image is mapped back by the plain pixel-centre rule.
"""
import json

import numpy as np
import pytest

import pyCamSet.calibration.detection_cache as cache_module
from pyCamSet.calibration_targets import ImageDetection, TargetDetection
from pyCamSet.calibration_targets.core import abstract_target
from pyCamSet.calibration_targets.core.target_registry import build_target
from pyCamSet.calibration_targets.markers import aruco2_gridboard, aruco_opencv


def _write_cache(tmp_path, spec):
    target = build_target(spec)
    path = tmp_path / "detected_datapoints.npz"
    cache_module.save_to_cache(
        TargetDetection(cam_names=["cam0"], data=np.array([[0, 0, 0, 1.0, 2.0]])),
        [(8, 12)], path, target, ["cam0"], None)
    return target, path


def _recorded_identity(path):
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["identity"]))


def _rewrite_identity(path, identity):
    """The same cache with *identity* in place of its own, as an older pyCamSet would have written it."""
    with np.load(path, allow_pickle=False) as archive:
        members = {name: archive[name] for name in archive.files}
    members["identity"] = np.array(json.dumps(identity, sort_keys=True))
    with open(path, "wb") as fh:
        np.savez(fh, **members)


CHARUCO_FAMILY = [
    {"type": "ChArUco", "marker_backend": "aruco1"},
    {"type": "ChArUco", "marker_backend": "aruco2"},
    {"type": "Ccube", "marker_backend": "aruco2"},
    {"type": "ChArUco2"},
]
CHARUCO_FAMILY_IDS = ["ChArUco-aruco1", "ChArUco-aruco2", "Ccube-aruco2", "ChArUco2"]


def test_the_frame_is_pixel_centre():
    assert aruco_opencv.PIXEL_CONVENTION == "pixel-centre"
    assert aruco2_gridboard.PIXEL_CONVENTION == "pixel-centre"


@pytest.mark.parametrize("spec", CHARUCO_FAMILY, ids=CHARUCO_FAMILY_IDS)
def test_a_cache_records_pixel_centre_and_matches(tmp_path, spec):
    target, path = _write_cache(tmp_path, spec)
    assert _recorded_identity(path)["pixel_convention"] == "pixel-centre"
    assert cache_module.cache_matches(path, target, ["cam0"], None) is True


@pytest.mark.parametrize("spec", CHARUCO_FAMILY, ids=CHARUCO_FAMILY_IDS)
def test_a_pixel_corner_cache_is_a_miss(tmp_path, spec):
    target, path = _write_cache(tmp_path, spec)
    identity = _recorded_identity(path)
    identity["pixel_convention"] = "pixel-corner"  # what OpenCV 4.x wrote before the frame was fixed
    _rewrite_identity(path, identity)
    assert cache_module.cache_matches(path, target, ["cam0"], None) is False


def test_a_puzzleboard_cache_records_no_convention(tmp_path):
    target, path = _write_cache(tmp_path, {"type": "PuzzleBoard"})
    assert "pixel_convention" not in _recorded_identity(path)
    assert cache_module.cache_matches(path, target, ["cam0"], None) is True


@pytest.mark.parametrize("version, offset", [("4.11.0", 0.5), ("4.13.0", 0.5), ("5.0.0", 0.0)])
def test_detectboard_corners_are_moved_to_pixel_centre(version, offset):
    native = np.array([[[10.5, 20.5]], [[123.75, 7.5]]], dtype=np.float32)  # as detectBoard returns them
    moved = aruco_opencv.charuco_corners_to_pixel_centre(native, version)
    if offset == 0.0:
        assert moved is native                                       # untouched where OpenCV is already there
    else:
        assert moved.dtype == np.float32 and moved.shape == native.shape
        np.testing.assert_array_equal(moved, native - np.float32(offset))
    assert aruco_opencv.charuco_corners_to_pixel_centre(None, version) is None


@pytest.mark.parametrize("scale", [0.25, 0.5, 1.5])
def test_the_scaled_restore_is_the_pixel_centre_rule(scale):
    centre = np.array([[10.0, 20.0], [123.25, 7.5]])               # native, pixel-centre
    scaled_centre = (centre + 0.5) * scale - 0.5                     # where the resize puts them
    restored = abstract_target._restore_native_coordinates(
        ImageDetection(np.array([0, 1]), scaled_centre), scale)
    np.testing.assert_allclose(restored.image_points, centre, atol=1e-12)
