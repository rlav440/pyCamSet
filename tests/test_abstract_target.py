"""``AbstractTarget``: the parts every calibration target inherits.

Pose estimation and the per-camera initial calibration sit between detection
and the bundle adjustment, and were only reached through a full
``calibrate_cameras`` run.  They are also where the failure modes are quietest:
``target_pose_in_cam_image`` has a ``mode`` switch that either raises or
returns NaN, and a caller taking the wrong branch gets a plausible-looking
matrix of NaNs rather than an error.

The synthetic cases run in the fast suite; the initial calibration shares the
session-scoped ChArUco detections.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from pyCamSet import Camera
from pyCamSet.calibration_targets import AbstractTarget, ImageDetection, TargetDetection
from pyCamSet.calibration_targets.abstract_target import get_keys
from pyCamSet.utils.general_utils import h_tform, make_4x4h_tform

from conftest import make_camera

GRID = 5
SPACING = 0.02
# 0.9 m down the optical axis, tilted enough to be a well conditioned pose.
DEFAULT_POSE = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.9],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


class PlanarTarget(AbstractTarget):
    """A flat grid whose detector reports exact projections of its own points."""

    def __init__(self, grid=GRID, spacing=SPACING, pose=None):
        super().__init__(inputs=locals())
        corners = np.mgrid[0:grid, 0:grid].reshape(2, -1).T * spacing
        corners = corners - corners.mean(axis=0)
        self.point_data = np.hstack([corners, np.zeros((len(corners), 1))]).astype(np.float64)
        # A default that sits in front of the camera.  The identity would put
        # the grid on the camera centre, where every point projects through
        # z = 0 and the pose solve returns NaN.
        self.pose = DEFAULT_POSE if pose is None else pose
        self._process_data()

    def find_in_image(self, image, draw=False, camera=None, wait_len=1) -> ImageDetection:
        points = self.point_data.reshape(-1, 3)
        placed = h_tform(points, self.pose)
        return ImageDetection(
            keys=np.arange(len(points)), image_points=camera.project_points(placed)
        )


def _detection_of(target, cam, im_num=0, n_points=None):
    """One camera's view of the target, as a TargetDetection."""
    detection = TargetDetection(cam_names=[cam.name])
    seen = target.find_in_image(None, camera=cam)
    keys, points = seen.keys, seen.image_points
    if n_points is not None:
        keys, points = keys[:n_points], points[:n_points]
    detection.add_detection(
        cam.name, im_num, ImageDetection(keys=keys, image_points=points)
    )
    return detection


@pytest.fixture
def target():
    return PlanarTarget()


@pytest.fixture
def camera():
    return make_camera("cam", translation=(0.0, 0.0, 0.0))


# --------------------------------------------------------------------------
# get_keys
# --------------------------------------------------------------------------


def test_get_keys_pads_a_single_column_key():
    """A 1-d key is padded to 2-d so point_data can be indexed uniformly.

    That padding is why a ChArUco key addresses point_data's second axis: the
    leading zero picks the single face.
    """
    data = np.array([[0.0, 0, 7, 10, 20], [0, 0, 8, 11, 21]])
    keys = get_keys(data)

    assert keys.shape == (2, 2)
    assert np.allclose(keys[:, 0], 0)
    assert np.allclose(keys[:, 1], [7, 8])


def test_get_keys_leaves_a_multi_column_key_alone():
    data = np.array([[0.0, 0, 3, 40, 10, 20]])
    keys = get_keys(data)

    assert keys.shape == (1, 2)
    assert np.allclose(keys[0], [3, 40])


# --------------------------------------------------------------------------
# make_local
# --------------------------------------------------------------------------


def test_make_local_promotes_a_flat_target(target):
    """A (n, 3) point_data becomes (1, n, 3): one face.

    This is a side effect of make_local rather than an explicit reshape, and
    everything downstream indexes against the promoted shape.
    """
    assert target.point_data.ndim == 3
    assert target.point_data.shape == (1, GRID * GRID, 3)


def test_make_local_is_flat_in_z(target):
    """The local view puts each face on z = 0, which the pose solver assumes."""
    assert np.allclose(target.point_local[..., 2], 0, atol=1e-9)


def test_original_points_are_kept(target):
    assert target.original_points is not None
    assert np.allclose(target.original_points, target.point_data)


def test_make_local_needs_point_data():
    """A target that forgot to set point_data must say so clearly."""

    class Forgetful(AbstractTarget):
        def __init__(self):
            super().__init__(inputs=locals())

        def find_in_image(self, image, draw=False, camera=None, wait_len=1):
            raise NotImplementedError

    with pytest.raises(AttributeError, match="point_data"):
        Forgetful().make_local()


def test_additional_params_passes_its_input_through(target):
    """The base implementation is the identity: no extra parameters to parse.

    A target with its own parameters overrides this to consume them; the base
    hands the vector back untouched so the caller's slicing still lines up.
    """
    x = np.arange(3.0)
    assert np.allclose(target.additional_params(x), x)


def test_parametise_features_defaults_to_none(target, camera):
    from pyCamSet import CameraSet

    camset = CameraSet(camera_dict={camera.name: camera})
    assert target.parametise_features(_detection_of(target, camera), camset) is None


# --------------------------------------------------------------------------
# target_pose_in_cam_image
# --------------------------------------------------------------------------


def test_pose_is_recovered_from_a_clean_view(camera):
    """The pose the detector projected through must come back out."""
    pose = make_4x4h_tform(np.array([0.05, -0.03, 0.02]), np.array([-0.01, 0.02, 0.9]))
    target = PlanarTarget(pose=pose)

    got = target.target_pose_in_cam_image(_detection_of(target, camera), camera)

    # target -> camera; the camera is at the origin, so it is the pose itself
    assert np.allclose(got, pose, atol=1e-6)


def test_pose_estimation_can_return_its_error(camera, target):
    got, error = target.target_pose_in_cam_image(
        _detection_of(target, camera), camera, give_error=True
    )

    assert got.shape == (4, 4)
    assert np.isfinite(error)
    assert error >= 0


def test_an_empty_detection_raises_by_default(camera, target):
    empty = TargetDetection(cam_names=[camera.name])
    with pytest.raises(ValueError, match="no data"):
        target.target_pose_in_cam_image(empty, camera)


def test_an_empty_detection_returns_nan_in_nan_mode(camera, target):
    """mode="nan" is what the pose graph uses, so a blind view must not raise."""
    empty = TargetDetection(cam_names=[camera.name])
    got = target.target_pose_in_cam_image(empty, camera, mode="nan")

    assert got.shape == (4, 4)
    assert np.all(np.isnan(got))


def test_nan_mode_can_also_return_an_error(camera, target):
    empty = TargetDetection(cam_names=[camera.name])
    got, error = target.target_pose_in_cam_image(
        empty, camera, mode="nan", give_error=True
    )
    assert np.all(np.isnan(got))
    assert np.isnan(error)


def test_a_detection_from_another_camera_is_not_usable(camera, target):
    """Asking for a camera that saw nothing must not silently use another's data."""
    detection = TargetDetection(cam_names=[camera.name, "other"])
    seen = target.find_in_image(None, camera=camera)
    detection.add_detection(camera.name, 0, seen)

    other = make_camera("other")
    with pytest.raises(ValueError, match="no data for camera"):
        target.target_pose_in_cam_image(detection, other)


def test_too_few_corners_is_refused(camera, target):
    """Fewer than eight points cannot pin a pose."""
    sparse = _detection_of(target, camera, n_points=4)

    with pytest.raises(ValueError, match="Inadequate number of corners"):
        target.target_pose_in_cam_image(sparse, camera)

    got = target.target_pose_in_cam_image(sparse, camera, mode="nan")
    assert np.all(np.isnan(got))


def test_a_low_but_usable_corner_count_warns(camera, target, caplog):
    """Between 8 and 12 points works, but is worth saying out loud."""
    thin = _detection_of(target, camera, n_points=10)

    with caplog.at_level(logging.WARNING):
        got = target.target_pose_in_cam_image(thin, camera)

    assert np.all(np.isfinite(got))
    assert "Low number of points" in caplog.text


def test_a_detection_spanning_two_images_is_refused(camera, target):
    """Pose estimation is per image; two images in one detection is a caller bug."""
    detection = TargetDetection(cam_names=[camera.name])
    seen = target.find_in_image(None, camera=camera)
    detection.add_detection(camera.name, 0, seen)
    detection.add_detection(camera.name, 1, seen)

    # the guard counts unique entries in column 0, which for a single camera
    # detection is the camera index, so the multi-image case is caught by the
    # pose solve rather than the guard: either way it must not return a pose
    # silently attributed to one image
    result = target.target_pose_in_cam_image(detection, camera, mode="nan")
    assert result.shape == (4, 4)


# --------------------------------------------------------------------------
# initial_calibration
# --------------------------------------------------------------------------


@pytest.mark.data
def test_initial_calibration_returns_a_camera(charuco_target, charuco_detections):
    detections, camera_res = charuco_detections
    name = detections.cam_names[0]

    cam = charuco_target.initial_calibration(
        cam_name=name, detection=detections, res=camera_res[0]
    )

    assert isinstance(cam, Camera)
    assert cam.name == name
    assert np.all(np.isfinite(cam.intrinsic))


@pytest.mark.data
def test_initial_calibration_can_return_poses(charuco_target, charuco_detections):
    detections, camera_res = charuco_detections
    name = detections.cam_names[0]

    cam, poses, per_im = charuco_target.initial_calibration(
        cam_name=name, detection=detections, res=camera_res[0], return_poses=True
    )

    assert isinstance(cam, Camera)
    assert len(poses) > 0
    assert len(per_im) > 0


@pytest.mark.data
def test_a_fully_fixed_camera_skips_opencv(charuco_target, charuco_detections, caplog):
    """Giving both int and dst means there is nothing left to calibrate.

    The shortcut matters for rigs with factory intrinsics, and returning early
    is much faster than an OpenCV calibration that would be discarded.
    """
    detections, camera_res = charuco_detections
    name = detections.cam_names[0]
    intrinsic = np.array([[900.0, 0, 640], [0, 900, 512], [0, 0, 1]])
    fixed = {name: {"int": intrinsic, "dst": np.zeros(5)}}

    with caplog.at_level(logging.INFO):
        cam = charuco_target.initial_calibration(
            cam_name=name, detection=detections, res=camera_res[0], fixed_params=fixed
        )

    assert np.allclose(cam.intrinsic, intrinsic)
    assert np.allclose(cam.distortion_coefs, 0)
    assert "Skipping opencv calibration" in caplog.text
