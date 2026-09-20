"""
Zhang's closed form seed, against cameras whose answer is known.

Everything here is synthetic and projected by hand.  The point of the seed is
that it is exact for a distortion free pinhole camera and degrades predictably
away from one, and neither half of that can be measured against real images,
where the truth is what is being estimated.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pyCamSet.calibration.zhang import (
    MIN_VIEWS,
    calibrate_zhang,
    focals_at_principal,
    homography_of,
    intrinsic_from_conic,
    pose_from_homography,
)

#: (height, width), the order detection reports a resolution in
RES = [480, 640]
INTRINSIC = np.array([[800.0, 0.0, 330.0],
                      [0.0, 750.0, 250.0],
                      [0.0, 0.0, 1.0]])

#: tilts that give the conic something to work with, in radians
TILTS = [(0.30, -0.20, 0.10), (-0.35, 0.15, -0.05), (0.10, 0.40, 0.20),
         (-0.15, -0.30, 0.15), (0.25, 0.25, -0.20), (-0.40, 0.05, 0.05)]


def board(n=9, length=150.0):
    """
    A flat grid of board points, in its own z = 0 plane.

    Sized to fill most of the frame at the distances below.  A board covering
    a small patch of the sensor barely feels a lens distortion at all -- the
    radius it is measured at is what drives it -- so a small board would make
    the distortion cases below look far kinder than a real session is.
    """
    axis = np.linspace(-length / 2, length / 2, n)
    grid = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    return np.concatenate([grid, np.zeros((len(grid), 1))], axis=1)


def pose_of(rotation, translation):
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(np.asarray(rotation, dtype=float))[0]
    pose[:3, 3] = translation
    return pose


def project(points, pose, intrinsic=INTRINSIC, distortion=0.0):
    """Project board points, optionally through a radial distortion."""
    camera_frame = points @ pose[:3, :3].T + pose[:3, 3]
    normalised = camera_frame[:, :2] / camera_frame[:, 2:]
    if distortion:
        r2 = np.sum(normalised ** 2, axis=1, keepdims=True)
        normalised = normalised * (1 + distortion * r2)
    return normalised * [intrinsic[0, 0], intrinsic[1, 1]] + [
        intrinsic[0, 2], intrinsic[1, 2]]


def views(tilts=TILTS, distortion=0.0, noise=0.0, seed=0):
    """Board views at a spread of orientations, and their true poses."""
    points = board()
    rng = np.random.default_rng(seed)
    objects, images, poses = [], [], []
    for index, tilt in enumerate(tilts):
        pose = pose_of(tilt, (5.0 * index - 10, -4.0 * index + 8, 300 + 15 * index))
        pixels = project(points, pose, distortion=distortion)
        if noise:
            pixels = pixels + rng.normal(scale=noise, size=pixels.shape)
        objects.append(points)
        images.append(pixels)
        poses.append(pose)
    return objects, images, poses


# ---------------------------------------------------------------------------
# the exact case
# ---------------------------------------------------------------------------

def test_recovers_a_known_camera():
    """
    A distortion free pinhole camera is what the closed form solves exactly.

    With no distortion and no noise there is no approximation anywhere in the
    derivation, so what is left is how precisely ``cv2.findHomography``
    converges -- about 5e-8 relative, which these bounds sit an order above.
    Anything near them is an error in the algebra rather than a tolerance to
    be loosened.
    """
    objects, images, poses = views()
    intrinsic, got_poses, rms = calibrate_zhang(objects, images, RES)

    assert np.allclose(intrinsic, INTRINSIC, rtol=1e-6)
    assert np.max(rms) < 1e-4
    for got, want in zip(got_poses, poses):
        assert np.allclose(got, want, rtol=0, atol=1e-4)


def test_poses_put_the_board_in_front_of_the_camera():
    """
    The homography fixes the pose only up to a sign, and both signs fit.

    The wrong one puts the board behind the camera, where the pose bootstrap
    that consumes this seed rejects it outright.
    """
    objects, images, _ = views()
    _, poses, _ = calibrate_zhang(objects, images, RES)
    assert all(pose[2, 3] > 0 for pose in poses)


def test_a_single_pose_is_recovered_from_its_homography():
    """The pose step on its own, given optics it does not have to estimate."""
    points = board()
    want = pose_of((0.2, -0.3, 0.1), (12.0, -6.0, 310.0))
    homography = homography_of(points, project(points, want))

    assert np.allclose(pose_from_homography(INTRINSIC, homography),
                       want, rtol=0, atol=1e-4)


# ---------------------------------------------------------------------------
# the cases it is actually handed
# ---------------------------------------------------------------------------

def test_survives_pixel_noise():
    """
    Detection noise moves the seed, but not far enough to matter.

    At 0.2 px the measured cost is 0.26% on the focal lengths and a pixel on
    the principal point, which the bundle adjustment closes easily.
    """
    objects, images, _ = views(noise=0.2)
    intrinsic, _, rms = calibrate_zhang(objects, images, RES)

    focal_error = np.abs([intrinsic[0, 0] / INTRINSIC[0, 0] - 1,
                          intrinsic[1, 1] / INTRINSIC[1, 1] - 1])
    assert np.max(focal_error) < 0.005
    assert np.max(np.abs(intrinsic[:2, 2] - INTRINSIC[:2, 2])) < 3
    assert np.max(rms) < 0.5


@pytest.mark.parametrize("k1", [-0.05, -0.15, -0.30])
def test_distortion_biases_the_seed_by_a_bounded_amount(k1):
    """
    A distorting lens biases the closed form, and this is how much.

    The seed fits no distortion, so a real lens is fit as the pinhole camera
    closest to it -- which is the whole premise: close enough to start a
    bundle adjustment that then solves for the distortion properly.

    The bias runs about 2.5% of ``k1`` over this range, measured as 0.14%,
    0.37% and 0.60% of the focal length at the three values below, against a
    board filling most of the frame. The bound is that measurement with room
    to spare, recorded so that a change making the seed worse is visible here
    rather than absorbed downstream by the optimisation.
    """
    objects, images, _ = views(distortion=k1)
    intrinsic, _, _ = calibrate_zhang(objects, images, RES)

    focal_error = np.max(np.abs([intrinsic[0, 0] / INTRINSIC[0, 0] - 1,
                                 intrinsic[1, 1] / INTRINSIC[1, 1] - 1]))
    assert focal_error < 0.05 * abs(k1)


# ---------------------------------------------------------------------------
# what it refuses, and why
# ---------------------------------------------------------------------------

def test_views_at_one_orientation_are_refused_with_a_reason():
    """
    A board only ever seen square on does not pin an intrinsic matrix.

    This is a property of the geometry: with the rotation fixed, every view
    contributes the same two constraints however far the board is moved, so
    there is nothing for the third and later views to add.
    """
    objects, images, _ = views(tilts=[(0.0, 0.0, 0.0)] * 5)
    with pytest.raises(ValueError, match="ambiguous"):
        calibrate_zhang(objects, images, RES)


def test_too_few_views_are_refused():
    objects, images, _ = views(tilts=TILTS[:MIN_VIEWS - 1])
    with pytest.raises(ValueError, match=f"at least {MIN_VIEWS} board views"):
        calibrate_zhang(objects, images, RES)


def test_a_board_out_of_its_own_plane_is_refused():
    """A homography maps a plane; point_local is what flattens each face."""
    points = board()
    points[0, 2] = 5.0
    with pytest.raises(ValueError, match="departs from its own plane"):
        homography_of(points, project(points, pose_of((0.2, 0.1, 0.0), (0, 0, 300))))


def test_an_impossible_conic_is_refused_with_a_reason():
    """
    A conic that no camera produces is reported as such, not returned.

    Neither sign of this one factorises, so there is no intrinsic matrix to
    extract -- which is how a badly distorting lens fails, rather than by
    returning something plausible looking.
    """
    with pytest.raises(ValueError, match="not positive definite"):
        intrinsic_from_conic(np.array([1.0, 0.0, -1.0, 0.0, 0.0, 1.0]))


# ---------------------------------------------------------------------------
# the fallback, for when the principal point comes out wild
# ---------------------------------------------------------------------------

def test_focals_are_recoverable_with_the_principal_point_held():
    """
    The fallback fits what is left when the unstable half is pinned.

    Exact here -- to the same 5e-8 the homographies are good to -- because
    the camera really does have its principal point where the fallback
    assumes it. On a real camera the offset is absorbed into the focal
    lengths instead, which is the trade being made.
    """
    centred = INTRINSIC.copy()
    centred[0, 2], centred[1, 2] = RES[1] / 2, RES[0] / 2
    points = board()
    homographies = [
        homography_of(points, project(points, pose_of(tilt, (0, 0, 320)), centred))
        for tilt in TILTS
    ]

    focals = focals_at_principal(homographies, np.array([RES[1] / 2, RES[0] / 2]))
    assert np.allclose(focals, [INTRINSIC[0, 0], INTRINSIC[1, 1]], rtol=1e-6)

