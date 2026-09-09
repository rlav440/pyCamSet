"""Pose estimation for a known target in a calibrated rig.

The docs advertise this ("the find_target_poses function will return a bundle
adjustment based estimate of the positions of the target"), but the module had
imported ``pyCamSet.optimisation.base_optimiser`` and ``derived_handlers``,
both long deleted, so the feature raised ModuleNotFoundError.  It is rebuilt on
``optimisation_handling.run_bundle_adjustment`` and ``TemplateBundleHandler``.

Detection is stubbed rather than rendered: ``StubTarget`` projects its own known
points through whichever camera it is handed, which is what a perfect detector
would return.  That keeps these tests fast and makes them test the thing they
are about -- assembling the detection, pinning the cameras, and reading the
solved pose back out -- rather than re-testing OpenCV's ChArUco detector, which
``test_calibration_charuco.py`` already covers against real images.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyCamSet import CameraSet
from pyCamSet.calibration_targets import AbstractTarget, ImageDetection
from pyCamSet.optimisation.find_target import (
    find_target_pose_at_timestep,
    find_target_poses,
    fix_all_cameras,
)
from pyCamSet.utils.general_utils import h_tform, make_4x4h_tform

from conftest import make_camera

# The pose solver needs at least 8 points in a view, so the grid is 5x5.
GRID = 5
SPACING = 0.02


class StubTarget(AbstractTarget):
    """A planar grid that reports where its points land, ignoring the pixels.

    The image passed to ``find_in_image`` carries only a timestep index, which
    selects which of the target's true poses to project.  That is how a test
    hands a different ground truth to each frame without rendering anything.
    """

    def __init__(self, poses=None, grid=GRID, spacing=SPACING):
        super().__init__(inputs=locals())
        corners = np.mgrid[0:grid, 0:grid].reshape(2, -1).T * spacing
        # centre it on the origin so the target sits in front of the cameras
        corners = corners - corners.mean(axis=0)
        self.point_data = np.hstack([corners, np.zeros((len(corners), 1))]).astype(np.float64)
        self.poses = [np.eye(4)] if poses is None else list(poses)
        # make_local promotes point_data from (n, 3) to (1, n, 3)
        self._process_data()

    def find_in_image(self, image, draw=False, camera=None, wait_len=1) -> ImageDetection:
        timestep = int(np.asarray(image).flat[0])
        points = self.point_data.reshape(-1, 3)
        placed = h_tform(points, self.poses[timestep])
        return ImageDetection(
            keys=np.arange(len(points)), image_points=camera.project_points(placed)
        )


def _frame(timestep=0):
    """A stand-in image that tells StubTarget which timestep it is."""
    return np.full((1, 1), timestep)


@pytest.fixture
def rig():
    """Three calibrated cameras with a baseline, looking down +z."""
    return CameraSet(
        camera_dict={
            name: make_camera(name, translation=offset)
            for name, offset in [
                ("left", (-0.06, 0.0, 0.0)),
                ("centre", (0.0, 0.0, 0.0)),
                ("right", (0.06, 0.0, 0.0)),
            ]
        }
    )


@pytest.fixture
def true_pose():
    return make_4x4h_tform(np.array([0.05, -0.03, 0.02]), np.array([-0.01, 0.02, 0.9]))


# --------------------------------------------------------------------------
# fix_all_cameras
# --------------------------------------------------------------------------


def test_fix_all_cameras_covers_every_camera(rig):
    fixed = fix_all_cameras(rig)
    assert set(fixed) == set(rig.get_names())


def test_fix_all_cameras_uses_the_packed_layout(rig):
    """The handler writes these into its parameter rows, so widths matter.

    Six for an extrinsic (rodrigues then translation), nine for an intrinsic
    (fx, cx, fy, cy, then five distortion coefficients).  Handing over the 4x4
    and 3x3 matrices instead -- which is the obvious thing to try, and what
    this module used to do -- fails to broadcast.
    """
    fixed = fix_all_cameras(rig)
    cam = rig["left"]

    entry = fixed["left"]
    assert entry["ext"].shape == (6,)
    assert entry["int"].shape == (9,)
    assert np.allclose(entry["ext"][3:], cam.extrinsic[:3, 3])
    assert np.allclose(
        entry["int"][:4],
        [cam.intrinsic[0, 0], cam.intrinsic[0, 2], cam.intrinsic[1, 1], cam.intrinsic[1, 2]],
    )
    assert np.allclose(entry["int"][4:], cam.distortion_coefs)


def test_fix_all_cameras_carries_distortion_inside_int(rig):
    """There is no separate 'dst' key: the handler ignores one."""
    rig["left"].set_distortion_coefs(np.array([-0.2, 0.05, 1e-3, -1e-3, 0.01]))
    entry = fix_all_cameras(rig)["left"]

    assert set(entry) == {"ext", "int"}
    assert np.allclose(entry["int"][4:], [-0.2, 0.05, 1e-3, -1e-3, 0.01])


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_an_unknown_camera_is_rejected(rig):
    """A name not in the set means the caller has mismatched their data."""
    target = StubTarget()
    with pytest.raises(ValueError, match="not part of the given CameraSet"):
        find_target_pose_at_timestep({"nonexistent": _frame()}, target, rig)

    with pytest.raises(ValueError, match="not part of the given CameraSet"):
        find_target_poses({"nonexistent": [_frame()]}, target, rig)


def test_a_target_detected_nowhere_is_reported(rig):
    """An empty detection cannot be solved, and must say so plainly."""

    class BlindTarget(StubTarget):
        def find_in_image(self, image, draw=False, camera=None, wait_len=1):
            return ImageDetection()

    with pytest.raises(ValueError, match="not detected in any"):
        find_target_pose_at_timestep({"left": _frame()}, BlindTarget(), rig)


def test_too_few_points_for_a_pose_is_reported(rig):
    """Pose estimation needs eight points in some single view."""

    class SparseTarget(StubTarget):
        def find_in_image(self, image, draw=False, camera=None, wait_len=1):
            full = super().find_in_image(image, camera=camera)
            return ImageDetection(keys=full.keys[:4], image_points=full.image_points[:4])

    with pytest.raises(ValueError, match="Could not estimate a target pose"):
        find_target_pose_at_timestep(
            {name: _frame() for name in rig.get_names()}, SparseTarget(), rig
        )


# --------------------------------------------------------------------------
# Single timestep
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_a_single_pose_is_recovered(rig, true_pose):
    """The detections are exact, so the solved pose must be the true one."""
    target = StubTarget(poses=[true_pose])
    images = {name: _frame() for name in rig.get_names()}

    got = find_target_pose_at_timestep(images, target, rig)

    assert got.shape == (4, 4)
    assert np.allclose(got, true_pose, atol=1e-8)


@pytest.mark.slow
def test_the_returned_pose_is_a_valid_transform(rig, true_pose):
    """A 4x4 with an orthonormal rotation and a [0,0,0,1] bottom row."""
    got = find_target_pose_at_timestep(
        {name: _frame() for name in rig.get_names()}, StubTarget(poses=[true_pose]), rig
    )

    assert np.allclose(got[3], [0, 0, 0, 1])
    rot = got[:3, :3]
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(rot), 1.0)


@pytest.mark.slow
def test_a_pose_is_recovered_from_a_single_camera(rig, true_pose):
    """One view is enough when the target is planar and fully seen.

    This is the case the old implementation could not have handled even with
    its imports fixed: the handler's own initial-parameter estimator rejects
    poses by their spread across images, which is undefined for one image.
    """
    got = find_target_pose_at_timestep(
        {"centre": _frame()}, StubTarget(poses=[true_pose]), rig
    )
    assert np.allclose(got, true_pose, atol=1e-8)


@pytest.mark.slow
def test_the_cameras_are_not_moved_by_the_solve(rig, true_pose):
    """Only the target pose is free; the calibration must come back intact."""
    before = {cam.name: cam.extrinsic.copy() for cam in rig}
    intrinsics = {cam.name: cam.intrinsic.copy() for cam in rig}

    find_target_pose_at_timestep(
        {name: _frame() for name in rig.get_names()}, StubTarget(poses=[true_pose]), rig
    )

    for cam in rig:
        assert np.allclose(cam.extrinsic, before[cam.name])
        assert np.allclose(cam.intrinsic, intrinsics[cam.name])


# --------------------------------------------------------------------------
# Sequences
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_a_sequence_of_poses_is_recovered(rig):
    """Each timestep gets its own pose, in the order the lists were given."""
    poses = [
        make_4x4h_tform(np.array([0.05, -0.03, 0.02 * i]), np.array([-0.01, 0.02, 0.9 + 0.05 * i]))
        for i in range(3)
    ]
    target = StubTarget(poses=poses)
    image_seq = {name: [_frame(i) for i in range(3)] for name in rig.get_names()}

    got = find_target_poses(image_seq, target, rig)

    assert got.shape == (3, 4, 4)
    assert np.allclose(got, np.array(poses), atol=1e-8)


@pytest.mark.slow
def test_a_single_frame_sequence_agrees_with_the_single_timestep_call(rig, true_pose):
    """The two entry points must not disagree on the same data."""
    single = find_target_pose_at_timestep(
        {name: _frame() for name in rig.get_names()}, StubTarget(poses=[true_pose]), rig
    )
    sequence = find_target_poses(
        {name: [_frame()] for name in rig.get_names()}, StubTarget(poses=[true_pose]), rig
    )

    assert sequence.shape == (1, 4, 4)
    assert np.allclose(sequence[0], single, atol=1e-8)


@pytest.mark.slow
def test_a_moving_target_is_tracked(rig):
    """A pure translation in z must come back as a pure translation in z."""
    depths = [0.8, 0.9, 1.0, 1.1]
    poses = []
    for depth in depths:
        pose = np.eye(4)
        pose[:3, 3] = [0.0, 0.0, depth]
        poses.append(pose)

    got = find_target_poses(
        {name: [_frame(i) for i in range(len(depths))] for name in rig.get_names()},
        StubTarget(poses=poses),
        rig,
    )

    assert np.allclose(got[:, 2, 3], depths, atol=1e-8)
    for pose in got:
        assert np.allclose(pose[:3, :3], np.eye(3), atol=1e-8)


# --------------------------------------------------------------------------
# The CameraSet methods
# --------------------------------------------------------------------------


def test_the_camera_set_exposes_the_methods(rig):
    assert callable(rig.find_target_pose)
    assert callable(rig.find_target_poses)


def test_importing_camera_set_alone_does_not_pull_in_the_optimiser(repo_root):
    """The method's import is deferred, so the module graph stays acyclic.

    find_target imports the optimisation handlers, which import camera_set; a
    module level import in either direction would deadlock on first import.
    Each order is checked in a fresh interpreter, since an import that only
    succeeds because something else got there first proves nothing.
    """
    import os
    import subprocess
    import sys

    # PYTHONPATH so the subprocess imports this working tree rather than
    # whatever copy of pyCamSet is installed in site-packages.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

    for statement in (
        "import pyCamSet.cameras.camera_set",
        "from pyCamSet.optimisation.find_target import find_target_poses",
        "import pyCamSet",
        "import pyCamSet.optimisation.template_handler, pyCamSet.cameras.camera_set",
    ):
        result = subprocess.run(
            [sys.executable, "-c", statement], capture_output=True, text=True, env=env
        )
        assert result.returncode == 0, f"{statement} failed:\n{result.stderr}"


@pytest.mark.slow
def test_the_method_agrees_with_the_function(rig, true_pose):
    images = {name: _frame() for name in rig.get_names()}

    by_function = find_target_pose_at_timestep(images, StubTarget(poses=[true_pose]), rig)
    by_method = rig.find_target_pose(images, StubTarget(poses=[true_pose]))

    assert np.allclose(by_method, by_function)
    assert np.allclose(by_method, true_pose, atol=1e-8)


@pytest.mark.slow
def test_the_sequence_method_agrees_with_the_function(rig):
    poses = [
        make_4x4h_tform(np.zeros(3), np.array([0.0, 0.0, 0.9 + 0.05 * i])) for i in range(2)
    ]
    image_seq = {name: [_frame(i) for i in range(2)] for name in rig.get_names()}

    by_function = find_target_poses(image_seq, StubTarget(poses=poses), rig)
    by_method = rig.find_target_poses(image_seq, StubTarget(poses=poses))

    assert np.allclose(by_method, by_function)
    assert np.allclose(by_method, np.array(poses), atol=1e-8)
