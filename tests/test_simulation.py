"""
The synthetic capture package, :mod:`pyCamSet.simulation`.

What matters is that its ground truth is the truth: that a point drawn into a
simulated image is detected where the capture says it projects, in the pixel-centre frame
every detector returns; that pasting errors move the
true points and nothing else; and that a camera set equal to the rig scores
zero.  The end-to-end calibration this package exists for is
``scripts/simulate_calibration.py``, documented in
``docs/how-to/simulate.md``; it is too slow for the suite.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from pyCamSet.calibration_targets.ccube import Ccube
from pyCamSet.simulation import (
    PastingErrors,
    SensorModel,
    compare_to_truth,
    held_target_poses,
    load_ground_truth,
    paste_faces,
    ring_rig,
    simulate_capture,
)
from pyCamSet.simulation.capture import GROUND_TRUTH_FILE, TRUE_POINTS_FILE
from pyCamSet.simulation.render import Scene, project_points, render_view
from pyCamSet.simulation.sensor import distort_image, undistort_normalised


@pytest.fixture(scope="module")
def cube():
    return Ccube(n_points=6, length=60.0)


@pytest.fixture(scope="module")
def rig():
    return ring_rig(2, radius=0.3, res=(800, 600), focal=1100.0, seed=4)


def test_no_pasting_error_leaves_the_points_where_the_target_says(cube):
    assert np.allclose(paste_faces(cube, PastingErrors.none()), cube.point_data, atol=1e-12)


def test_pasting_errors_move_points_within_their_face(cube):
    errors = PastingErrors.by_hand(seed=5)
    moved = paste_faces(cube, errors)
    shift = np.linalg.norm(moved - cube.point_data, axis=-1)
    assert 1e-5 < shift.mean() < 2e-3                      # tenths of a millimetre
    for face in range(6):
        # Still in the face's plane: the paper slides on the solid, it does not lift.
        nominal = cube.point_data[face]
        normal = np.cross(nominal[1] - nominal[0], nominal[5] - nominal[0])
        normal /= np.linalg.norm(normal)
        assert np.allclose((moved[face] - nominal[0]) @ normal, 0.0, atol=1e-12)


def test_the_lens_inverse_undoes_opencvs_distortion():
    distortion = np.array([-0.1, 0.04, 0.001, -0.002, 0.0])
    points = np.array([[0.4, 0.3, 1.0], [-0.45, -0.32, 1.0], [0.01, -0.2, 1.0]])
    pixels, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), np.eye(3), distortion)
    x, y = undistort_normalised(pixels[:, 0, 0], pixels[:, 0, 1], distortion)
    assert np.allclose(np.c_[x, y], points[:, :2], atol=1e-10)


def test_distorting_an_image_moves_a_dot_where_opencv_projects_it():
    intrinsic = np.array([[500.0, 0, 199.5], [0, 500.0, 149.5], [0, 0, 1]])
    distortion = np.array([-0.2, 0.05, 0.0, 0.0, 0.0])
    image = np.zeros((300, 400), np.float32)
    image[40, 60] = 1.0                                     # an ideal pinhole dot
    bent = distort_image(cv2.GaussianBlur(image, (0, 0), 1.5), intrinsic, distortion)
    ray = np.array([[(60 - 199.5) / 500.0, (40 - 149.5) / 500.0, 1.0]])
    expected, _ = cv2.projectPoints(ray, np.zeros(3), np.zeros(3), intrinsic, distortion)
    ys, xs = np.mgrid[0:300, 0:400]
    found = np.array([(xs * bent).sum(), (ys * bent).sum()]) / bent.sum()
    assert np.linalg.norm(found - expected.reshape(2)) < 0.15


def test_a_rendered_view_is_detected_where_the_ground_truth_projects(cube, rig):
    """The whole imaging chain, pasting and lens included, against the truth."""
    errors = PastingErrors.by_hand(seed=6)
    true_points = paste_faces(cube, errors)
    pose = held_target_poses(1, translation=0.0, seed=8)[0]
    camera = rig[0]
    radiance = render_view(cube, camera, pose, errors, Scene(supersample=3))
    sensor = SensorModel(blur_sigma=0.6, read_noise=4.0, seed=1)
    image = sensor.apply(radiance, camera.intrinsic, camera.distortion)
    found = cube.find_in_image(image)
    keys = np.asarray(found.keys)
    points_per_face = int(np.prod(cube.point_data.shape[1:-1]))
    assert len(keys) >= points_per_face               # at least one whole face
    world = true_points[keys[:, 0], keys[:, 1]] @ pose[:3, :3].T + pose[:3, 3]
    expected = project_points(world, camera)
    error = np.linalg.norm(found.image_points - expected, axis=1)
    assert np.median(error) < 0.25 and error.max() < 1.0, (np.median(error), error.max())


def test_a_capture_is_written_as_the_pipeline_reads_it(cube, rig, tmp_path):
    poses = held_target_poses(2, seed=9)
    capture = simulate_capture(cube, rig, poses, tmp_path, errors=PastingErrors.by_hand(),
                               sensor=SensorModel(seed=2), log=lambda _line: None)
    for camera in rig:
        files = sorted((tmp_path / camera.name).glob("*.png"))
        assert [f.name for f in files] == ["000.png", "001.png"]
        image = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
        assert image.shape == (camera.res[1], camera.res[0], 3)
    assert (tmp_path / GROUND_TRUTH_FILE).exists() and (tmp_path / TRUE_POINTS_FILE).exists()
    json.loads((tmp_path / GROUND_TRUTH_FILE).read_text(encoding="utf-8"))
    loaded = load_ground_truth(tmp_path)
    assert np.allclose(loaded.true_points, capture.true_points)
    assert np.allclose(loaded.poses, poses)
    assert [c.name for c in loaded.cameras] == [c.name for c in rig]
    assert loaded.target_spec["type"] == "Ccube"


def test_the_rig_itself_scores_perfectly(cube, rig, tmp_path):
    """A camera set equal to the rig scores zero."""
    from pyCamSet.cameras import CameraSet

    capture = simulate_capture(cube, rig, held_target_poses(1, seed=3), tmp_path,
                               sensor=SensorModel(), log=lambda _line: None)
    cams = CameraSet(camera_dict={c.name: c.to_camera() for c in rig})
    report = compare_to_truth(cams, capture)
    summary = report["summary"]
    assert summary["worst_focal_error_pct"] < 1e-9
    assert summary["worst_position_error_mm"] < 1e-6
    assert summary["worst_rotation_error_deg"] < 1e-6
    assert summary["worst_lens_rms_px"] < 1e-6


@pytest.mark.parametrize("spec", [
    {"type": "Ccube", "n_points": 6, "length": 80.0},
    {"type": "Ccube2", "n_points": 6, "length": 80.0},
], ids=["Ccube", "Ccube2"])
def test_every_detector_reports_where_opencv_projects(spec, rig):
    """One frame for every target, whatever OpenCV is installed: a Ccube's
    corners (OpenCV's CharucoDetector) and a Ccube2's (aruco2's grid board)
    land where OpenCV projects them, in the pixel-centre frame.  Under OpenCV 4
    that holds only because detect_board moves its corners half a pixel back."""
    if spec["type"] == "Ccube2":
        pytest.importorskip("aruco2")
    from pyCamSet.calibration_targets.core.target_registry import build_target

    target = build_target(spec)
    errors = PastingErrors.none(6)
    for seed in (11, 12, 13):
        pose = held_target_poses(1, translation=0.0, seed=seed)[0]
        image = SensorModel(blur_sigma=0.6, read_noise=2.0, vignetting=0.0).apply(
            render_view(target, rig[0], pose, errors), rig[0].intrinsic, rig[0].distortion)
        found = target.find_in_image(image)
        if found.keys is None or len(np.asarray(found.keys)) < 8:
            continue
        keys = np.asarray(found.keys)
        world = target.point_data[tuple(keys.T)] @ pose[:3, :3].T + pose[:3, 3]
        offset = np.median(found.image_points - project_points(world, rig[0]), axis=0)
        assert np.allclose(offset, 0.0, atol=0.1), offset
        return
    pytest.fail("no view detected enough points to measure the convention")
