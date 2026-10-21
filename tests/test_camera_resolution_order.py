"""
``Camera.res`` is ``(width, height)``; detection's ``cam_res`` is an image shape.

Every case uses a non-square sensor with distinct width and height, so an axis
swap anywhere fails rather than coincidentally passing.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pyCamSet import Camera
from pyCamSet.calibration_targets import ImageDetection, TargetDetection
from pyCamSet.cameras.camera import res_from_image_shape
from pyCamSet.cameras.telecentric_camera import TelecentricCamera
from pyCamSet.utils.general_utils import h_tform, make_4x4h_tform
from pyCamSet.utils.saving import load_CameraSet

WIDTH, HEIGHT = 540, 720          # portrait
IMAGE_SHAPE = (HEIGHT, WIDTH)     # what detection reports as cam_res


def _pinhole(name="cam", res=(640, 480), fx=500.0):
    K = np.array([[fx, 0, res[0] / 2], [0, fx, res[1] / 2], [0, 0, 1.0]])
    return Camera(intrinsic=K, res=list(res), name=name)


def _telecentric(name="cam"):
    return TelecentricCamera(
        intrinsic=np.array([[80.0, 0, WIDTH / 2], [0, 80.0, HEIGHT / 2], [0, 0, 1.0]]),
        res=[WIDTH, HEIGHT], distortion_coefs=np.array([0.0]), telecentricity=0.0,
        name=name)


def _cube_detection(cam_name, target, cam, n_images=6):
    """Each image catches two faces of the cube, which a telecentric seed needs."""
    detection = TargetDetection(cam_names=[cam_name])
    points = np.asarray(target.point_data).reshape(6, -1, 3)
    for im in range(n_images):
        pose = make_4x4h_tform(
            Rotation.from_euler("xyz", [0.2 + 0.05 * im, -0.3, 0.1]).as_rotvec(),
            [0.0, 0.0, 200.0])
        for face in range(2):
            placed = h_tform(points[face], pose)
            keys = np.stack([np.full(len(placed), face), np.arange(len(placed))], axis=-1)
            detection.add_detection(
                cam_name, im, ImageDetection(keys=keys, image_points=cam.project_points(placed)))
    return detection


@pytest.fixture
def cube():
    from pyCamSet.calibration_targets.core.target_registry import build_target
    return build_target({"type": "Ccube", "n_points": 6, "length": 10.0})


# --------------------------------------------------------------------------
# Producers: detection's image shape becomes (width, height) exactly once
# --------------------------------------------------------------------------

def test_res_from_image_shape_reverses_the_numpy_shape():
    assert res_from_image_shape((HEIGHT, WIDTH)) == (WIDTH, HEIGHT)
    assert res_from_image_shape((HEIGHT, WIDTH, 3)) == (WIDTH, HEIGHT)


def test_telecentric_seed_centres_the_principal_point_on_a_non_square_sensor(cube):
    """The seed pins the principal point to the sensor centre, (w/2, h/2)."""
    truth = _telecentric()
    cam = cube.initial_calibration(
        cam_name="cam", detection=_cube_detection("cam", cube, truth),
        res=IMAGE_SHAPE, pose_im=0, model="telecentric", min_detections_per_board=12)

    assert list(np.ravel(cam.res)) == [WIDTH, HEIGHT]
    assert cam.principal_point == pytest.approx([WIDTH / 2, HEIGHT / 2])


def test_calibrate_telecentric_reads_camera_res_as_width_height():
    from pyCamSet.calibration.telecentric import calibrate_telecentric
    from pyCamSet.calibration_targets.core.target_registry import build_target

    target = build_target({"type": "Ccube", "n_points": 6, "length": 10.0})
    truth = _telecentric()
    points = np.asarray(target.point_data).reshape(6, -1, 3)[:2].reshape(-1, 3)
    pose = make_4x4h_tform(Rotation.from_euler("xyz", [0.3, -0.3, 0.1]).as_rotvec(), [0, 0, 200.0])
    _, principal, _, _ = calibrate_telecentric(
        [points], [truth.project_points(h_tform(points, pose))], truth.res)
    assert principal == pytest.approx([WIDTH / 2, HEIGHT / 2])


def test_a_fixed_camera_is_built_with_width_height(cube):
    truth = _telecentric()
    K = np.array([[900.0, 0, 270.0], [0, 900.0, 360.0], [0, 0, 1.0]])
    cam = cube.initial_calibration(
        cam_name="cam", detection=_cube_detection("cam", cube, truth), res=IMAGE_SHAPE,
        fixed_params={"cam": {"int": K, "dst": np.zeros(5)}})
    assert list(np.ravel(cam.res)) == [WIDTH, HEIGHT]


def test_make_cam_dict_default_gives_each_camera_a_resolution():
    from pyCamSet.cameras.camera_set import make_cam_dict
    made = make_cam_dict(["a", "b"], [np.eye(4)] * 2, [np.eye(3)] * 2)
    assert all(list(c.res) == [1024, 768] for c in made.values())


# --------------------------------------------------------------------------
# Consumers
# --------------------------------------------------------------------------

def test_bounds_check_is_u_against_width_and_v_against_height():
    cam = _pinhole(res=(640, 480))
    inside, outside = cam._is_in_image(np.array([[600.0, 100.0], [100.0, 600.0]]))
    assert inside and not outside


def test_crop_bounds_are_width_then_height():
    cam = _pinhole(res=(640, 480))
    cam.crop_to_roi([0, 640, 0, 480])
    with pytest.raises(ValueError):
        _pinhole(res=(640, 480)).crop_to_roi([0, 10, 0, 481])


def test_fov_is_across_the_width_with_fx():
    cam = _pinhole(res=(640, 480), fx=500.0)
    assert cam._cam_fov() == pytest.approx(math.degrees(2 * math.atan(320 / 500)))


def test_sensor_map_is_indexed_u_v_and_reaches_the_far_corner():
    cam = _pinhole(res=(64, 48), fx=50.0)
    cam._make_sensormap()
    assert cam.sensor_map.shape == (64, 48, 3)
    assert cam.get_image_cord_sensor_map().shape == (48, 64, 3)
    cam.im_to_world_ray(np.array([[63, 47]]))


def test_telecentric_field_size_and_undistort_follow_the_sensor():
    cam = _telecentric()
    assert cam.field_size == pytest.approx([WIDTH / 80.0, HEIGHT / 80.0])
    image = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    assert cam.undistort(image).shape == IMAGE_SHAPE


# --------------------------------------------------------------------------
# Saved files (the marker, its refusal and the round trip are in test_res_order.py)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name, expected", [
    # calibrate_cameras output, (width, height), decided by its detections
    ("calibration_ccube/self_calib_test.camset", [1920, 1080]),
    # run_initial_calibration output, (height, width), no detections in the file
    ("calibration_charuco/initial_cameras.camset", [1280, 1024]),
    # a workflow Phase 4 file, (height, width), decided by its detections
    ("education/legacy/phase4_325fae4.camset.json", [1280, 1024]),
])
def test_legacy_test_data_reads_as_width_height(data_dir, name, expected):
    if not (data_dir / name).exists():
        pytest.skip(f"{name} is not part of this checkout")
    cams = load_CameraSet(data_dir / name)
    assert all(list(np.ravel(c.res)) == expected for c in cams)
