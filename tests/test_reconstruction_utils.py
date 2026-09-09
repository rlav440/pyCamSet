"""Smoke tests for the stereo reconstruction helpers.

The module was at 0% and unexported: ``reconstruction/__init__.py`` was empty,
so nothing in the package or the tests referenced it and it was
indistinguishable from dead code.  It is exported now, which makes it a
supported surface, so these check the parts that need neither an image corpus
nor a matlab install: that the geometry helpers agree with the cameras they
are given, and that the masks reject what they claim to.

``matlab_stereo`` is untested by design -- it needs the matlab engine, which is
not a declared dependency.
"""

from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv

from pyCamSet.reconstruction import (
    depth_image_ptcloud_mask,
    disparity_to_ptcld,
    rectify_camera_pair,
    remap_im,
    undistort_im,
)

from conftest import REF_RES, make_camera


@pytest.fixture
def stereo_pair():
    """Two cameras separated along x, the usual rectifiable arrangement."""
    return make_camera("left", translation=(-0.05, 0.0, 0.0)), make_camera(
        "right", translation=(0.05, 0.0, 0.0)
    )


@pytest.fixture
def grey_image():
    """A deterministic image with structure, so a remap is visibly a remap."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(REF_RES[1], REF_RES[0]), dtype=np.uint8)


# --------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------


def test_depth_mask_keeps_points_inside_the_range():
    cloud = np.array(
        [
            [0.0, 0.0, 1.0],  # inside
            [0.0, 0.0, 2.0],  # inside
            [0.0, 0.0, 0.1],  # too near
            [0.0, 0.0, 9.0],  # too far
        ]
    )
    mask = depth_image_ptcloud_mask(cloud, 0.5, 2.5)

    assert mask.tolist() == [True, True, False, False]


def test_depth_mask_rejects_nan_and_inf():
    """Reprojection produces both for pixels with no valid disparity."""
    cloud = np.array(
        [
            [0.0, 0.0, 1.0],
            [np.nan, 0.0, 1.0],
            [0.0, np.inf, 1.0],
            [0.0, 0.0, np.nan],
        ]
    )
    mask = depth_image_ptcloud_mask(cloud, 0.5, 2.5)

    assert mask.tolist() == [True, False, False, False]


def test_depth_mask_bounds_are_inclusive_of_the_interior():
    """A point exactly at the limits is kept: the test is strict on both ends."""
    cloud = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 2.5]])
    assert depth_image_ptcloud_mask(cloud, 0.5, 2.5).tolist() == [True, True]


def test_depth_mask_of_an_empty_range_keeps_nothing():
    cloud = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    assert not depth_image_ptcloud_mask(cloud, 5.0, 6.0).any()


# --------------------------------------------------------------------------
# Undistortion
# --------------------------------------------------------------------------


def test_undistort_is_a_no_op_for_an_undistorted_camera(grey_image):
    """Zero coefficients must leave the image alone."""
    cam = make_camera("a")
    assert np.array_equal(undistort_im(grey_image, cam), grey_image)


def test_undistort_changes_a_distorted_image(grey_image):
    cam = make_camera("a", distortion=[-0.3, 0.1, 0.0, 0.0, 0.02])
    out = undistort_im(grey_image, cam)

    assert out.shape == grey_image.shape
    assert not np.array_equal(out, grey_image)


def test_undistort_preserves_dtype(grey_image):
    cam = make_camera("a", distortion=[-0.3, 0.1, 0.0, 0.0, 0.02])
    assert undistort_im(grey_image, cam).dtype == grey_image.dtype


# --------------------------------------------------------------------------
# Rectification
# --------------------------------------------------------------------------


def test_rectify_camera_pair_returns_well_shaped_matrices(stereo_pair):
    cam_0, cam_1 = stereo_pair
    p0, p1, q, r0, r1, s0 = rectify_camera_pair(cam_0, cam_1, zero_flag=True)

    assert p0.shape == p1.shape == (3, 4)
    assert q.shape == (4, 4)
    assert r0.shape == r1.shape == (3, 3)
    for mat in (p0, p1, q, r0, r1):
        assert np.all(np.isfinite(mat))


def test_rectification_rotations_are_rotations(stereo_pair):
    cam_0, cam_1 = stereo_pair
    _, _, _, r0, r1, _ = rectify_camera_pair(cam_0, cam_1, zero_flag=True)

    for rot in (r0, r1):
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-8)
        assert np.isclose(np.linalg.det(rot), 1.0)


def test_rectifying_a_horizontal_pair_puts_the_baseline_in_q(stereo_pair):
    """Q encodes 1/baseline, so a 0.1 m separation must show up in it."""
    cam_0, cam_1 = stereo_pair
    _, _, q, _, _, _ = rectify_camera_pair(cam_0, cam_1, zero_flag=True)

    # q[3, 2] is -1/Tx for a horizontal pair
    assert np.isclose(abs(1.0 / q[3, 2]), 0.1, rtol=1e-6)


def test_rectify_honours_the_zero_distortion_flag():
    """zero_flag substitutes a zero model, so it must change the result."""
    cam_0 = make_camera("left", translation=(-0.05, 0.0, 0.0), distortion=[-0.3, 0.1, 0.0, 0.0, 0.0])
    cam_1 = make_camera("right", translation=(0.05, 0.0, 0.0), distortion=[-0.3, 0.1, 0.0, 0.0, 0.0])

    zeroed = rectify_camera_pair(cam_0, cam_1, zero_flag=True)[0]
    kept = rectify_camera_pair(cam_0, cam_1, zero_flag=False)[0]

    assert not np.allclose(zeroed, kept)


def test_a_zero_baseline_pair_is_rejected():
    """Rectification needs a real baseline; co-located cameras have none.

    cv2.stereoRectify asserts the translation norm is positive, so this fails
    loudly rather than returning a degenerate Q that would silently place
    every reconstructed point at infinity.
    """
    import cv2

    cam = make_camera("a")
    with pytest.raises(cv2.error):
        rectify_camera_pair(cam, make_camera("b"), zero_flag=True)


# --------------------------------------------------------------------------
# Remapping
# --------------------------------------------------------------------------


def test_remap_im_returns_an_image_of_the_requested_size(grey_image, stereo_pair):
    """Regression guard: remap_im used to plt.show() twice per call.

    Two interactive plots and a commented-out raise were left in the body, so
    the function could not be used non-interactively at all.
    """
    cam_0, cam_1 = stereo_pair
    p0, _, _, r0, _, _ = rectify_camera_pair(cam_0, cam_1, zero_flag=True)

    new_size = (REF_RES[0], REF_RES[1])
    out = remap_im(grey_image, cam_0, r0, p0, new_size)

    assert out.shape[:2] == (new_size[1], new_size[0])
    assert out.dtype == grey_image.dtype


# --------------------------------------------------------------------------
# Disparity to cloud
# --------------------------------------------------------------------------


def test_disparity_to_ptcld_returns_a_cloud_and_a_mask(stereo_pair):
    cam_0, cam_1 = stereo_pair
    _, _, q, _, _, _ = rectify_camera_pair(cam_0, cam_1, zero_flag=True)

    # a uniform disparity: everything lands on one fronto-parallel plane
    disp = np.full((REF_RES[1], REF_RES[0]), 16 * 40, dtype=np.int16)
    cloud, mask = disparity_to_ptcld(disp, q)

    assert isinstance(cloud, pv.PolyData)
    assert mask.shape == (REF_RES[0] * REF_RES[1],)
    assert mask.dtype == bool
    assert cloud.n_points == int(mask.sum())


def test_disparity_to_ptcld_drops_out_of_range_depths(stereo_pair):
    """The helper hard-codes a 0.5 to 2.5 m keep window.

    Zero disparity is at infinity, so every point must be rejected -- the
    cloud comes back empty rather than full of infinities.
    """
    cam_0, cam_1 = stereo_pair
    _, _, q, _, _, _ = rectify_camera_pair(cam_0, cam_1, zero_flag=True)

    disp = np.zeros((32, 32), dtype=np.int16)
    cloud, mask = disparity_to_ptcld(disp, q)

    assert not mask.any()
    assert cloud.n_points == 0


def test_disparity_to_ptcld_points_are_all_finite_and_in_range(stereo_pair):
    cam_0, cam_1 = stereo_pair
    _, _, q, _, _, _ = rectify_camera_pair(cam_0, cam_1, zero_flag=True)

    disp = np.full((64, 64), 16 * 40, dtype=np.int16)
    cloud, _ = disparity_to_ptcld(disp, q)

    points = np.asarray(cloud.points)
    assert np.all(np.isfinite(points))
    assert np.all(points[:, 2] >= 0.5)
    assert np.all(points[:, 2] <= 2.5)
