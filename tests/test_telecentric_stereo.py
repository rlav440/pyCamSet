"""
Rectification and triangulation of telecentric pairs, checked without a matcher.

A learned matcher only ever sees the rectified images, so the geometry is
tested by rendering a known surface into both cameras and asking whether the
rectified pair agrees with itself row for row, and whether the true
disparity triangulates back onto the surface.
"""
import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from pyCamSet.reconstruction.telecentric_stereo import (
    bundle_adjust_points, left_right_consistent, rectify_telecentric_pair, reconstruct_telecentric_pair, triangulate_pair)
from telecentric_scene import height, make_cam, oracle_matcher, render


@pytest.fixture(params=['perfect', 'distorted'])
def pair(request):
    k = 0.0 if request.param == 'perfect' else 0.05
    return (make_cam('a', [0.05, -0.25, 0.1], k=k, pp_shift=(12, -7)),
            make_cam('b', [-0.03, 0.22, -0.05], k=-0.04, pp_shift=(-5, 9)))


def test_rectified_rows_agree_for_any_point(pair):
    rect = rectify_telecentric_pair(*pair)
    pts = np.random.default_rng(0).uniform(-0.006, 0.006, size=(200, 3))
    v_l = rect.source_to_rectified(rect.left.project_points(pts), 'left')[:, 1]
    v_r = rect.source_to_rectified(rect.right.project_points(pts), 'right')[:, 1]
    np.testing.assert_allclose(v_l, v_r, atol=1e-8)
    assert rect.max_row_error == 0.0


def test_disparity_grows_toward_the_cameras(pair):
    rect = rectify_telecentric_pair(*pair)
    toward = rect.centre - 0.005 * rect.depth_axis
    away = rect.centre + 0.005 * rect.depth_axis
    d_toward, d_centre, d_away = rect.expected_disparity(np.stack([toward, rect.centre, away]))
    assert d_toward > d_centre > d_away > 0


def test_pair_order_does_not_change_which_camera_is_left(pair):
    a, b = pair
    ab, ba = rectify_telecentric_pair(a, b), rectify_telecentric_pair(b, a)
    assert ab.left is ba.left and ab.swapped != ba.swapped


def test_parallel_views_are_refused():
    with pytest.raises(ValueError, match="nearly the same direction"):
        rectify_telecentric_pair(make_cam('a', [0, 0, 0]), make_cam('b', [0, 0, 0.3]))


def test_triangulation_inverts_projection(pair):
    pts = np.random.default_rng(1).uniform(-0.006, 0.006, size=(500, 3))
    a, b = pair
    got = triangulate_pair(a, b, a.project_points(pts), b.project_points(pts))
    np.testing.assert_allclose(got, pts, atol=1e-9)


def test_rectified_images_match_along_the_true_disparity(pair):
    """The warp itself: the left image, shifted by the true disparity, is the right."""
    a, b = pair
    rect = rectify_telecentric_pair(a, b)
    rect_l, rect_r = rect.rectify(render(a), render(b))
    disp = oracle_matcher(rect)(rect_l, rect_r)
    _, _, keep = rect.disparity_to_correspondences(disp)
    rows, cols = np.nonzero(keep)
    sampled = map_coordinates(rect_r.astype(float), [rows, cols - disp[rows, cols]], order=1)
    err = np.abs(sampled - rect_l[rows, cols].astype(np.float32))
    # resampling twice blurs a busy texture a little; a misalignment of even a
    # pixel would put the median error near the texture's own spread (~40)
    assert keep.sum() > 0.5 * keep.size
    assert np.median(err) < 6


def test_true_disparity_reconstructs_the_surface(pair):
    a, b = pair
    result = reconstruct_telecentric_pair(a, b, render(a), render(b),
                                          matcher=oracle_matcher(rectify_telecentric_pair(a, b)),
                                          lr_tolerance=None)
    z_err = result.points[:, 2] - height(result.points[:, 0], result.points[:, 1])
    assert len(result.points) > 0.5 * result.keep.size
    assert np.max(np.abs(z_err)) < 1e-6
    assert result.colours.shape[0] == len(result.points)


def test_telecentricity_error_is_reported_and_triangulated_exactly():
    a = make_cam('a', [0.0, -0.25, 0.0], eps=2.0)
    b = make_cam('b', [0.0, 0.25, 0.0], eps=2.0)
    rect = rectify_telecentric_pair(a, b)
    assert rect.max_row_error > 0
    result = reconstruct_telecentric_pair(a, b, render(a), render(b),
                                          matcher=oracle_matcher(rect), rectification=rect,
                                          lr_tolerance=None)
    z_err = result.points[:, 2] - height(result.points[:, 0], result.points[:, 1])
    # the oracle disparity is taken along the rectified row, which with a
    # weakly perspective lens is not quite the epipolar line; the error in
    # 3D is bounded by that row error at the rectified magnification
    assert np.max(np.abs(z_err)) < 5 * rect.max_row_error / rect.magnification + 1e-6


def test_bundle_adjustment_recovers_points_through_distortion_and_telecentricity():
    a = make_cam('a', [0.05, -0.25, 0.1], k=0.05, eps=2.0, pp_shift=(12, -7))
    b = make_cam('b', [-0.03, 0.22, -0.05], k=-0.04, eps=2.0, pp_shift=(-5, 9))
    # the telecentric extrinsic block estimates rotation only, so the kernel
    # never sees a translation; a calibrated camera has none, and with eps
    # nonzero a synthetic one that did would project differently there
    for cam in (a, b):
        cam.extrinsic[:3, 3] = 0.0
        cam._update_state()
    pts = np.random.default_rng(2).uniform(-0.006, 0.006, size=(300, 3))
    uv_a, uv_b = a.project_points(pts), b.project_points(pts)
    start = pts + np.random.default_rng(3).normal(scale=2e-4, size=pts.shape)
    got = bundle_adjust_points((a, b), (uv_a, uv_b), start)
    np.testing.assert_allclose(got, pts, atol=1e-8)
    # the cameras were fixed, so they must come back exactly as they went in
    np.testing.assert_allclose(a.project_points(got), uv_a, atol=1e-5)


def test_left_right_check_keeps_agreeing_matches_and_drops_the_rest():
    im = np.zeros((40, 60), np.uint8)
    calls = []

    def matcher(left, right):
        # the second call is the mirrored pair; answer it 0.4 or 3 px off
        calls.append(len(calls))
        return np.full(left.shape[:2], 10.0 + (offset if len(calls) == 2 else 0.0))

    for offset, expect in ((0.4, True), (3.0, False)):
        calls.clear()
        disp = matcher(im, im)
        ok = left_right_consistent(matcher, im, im, disp, tolerance=1.0)
        # columns under the disparity have no right pixel to come back from
        assert np.all(ok[:, 12:] == expect)
        assert not ok[:, :9].any()
