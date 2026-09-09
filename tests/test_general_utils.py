"""The pure numerics in ``pyCamSet.utils.general_utils``.

At 48% this was the largest block of genuinely untested pure functions.  They
have no dependencies and run instantly, but several carry non-obvious
conventions -- ``plane_fit`` wants points as columns, ``distort_points`` wants
them as rows and returns a tuple, ``h_tform`` squeezes its output -- which are
exactly the kind of thing that a caller gets wrong silently.  Those are pinned
here so the convention is a test failure rather than a debugging session.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyCamSet.utils.general_utils import (
    average_tforms,
    colourmap_to_colour_list,
    distort_points,
    downsample_valid,
    ext_4x4_to_rod,
    flatten_pose_list,
    get_close_square_tuple,
    get_subfolder_names,
    glob_ims,
    grouper,
    h_tform,
    list_dict_to_np_array,
    make_4x4h_tform,
    plane_fit,
    px_array,
    write_colour_ply,
)

# --------------------------------------------------------------------------
# Homogeneous transforms
# --------------------------------------------------------------------------


def test_make_4x4h_tform_builds_a_valid_transform():
    tform = make_4x4h_tform(np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0, 3.0]))

    assert tform.shape == (4, 4)
    assert np.allclose(tform[3], [0, 0, 0, 1])
    assert np.allclose(tform[:3, 3], [1.0, 2.0, 3.0])
    # the rotation block must be a rotation: orthonormal, det +1
    rot = tform[:3, :3]
    assert np.allclose(rot @ rot.T, np.eye(3))
    assert np.isclose(np.linalg.det(rot), 1.0)


def test_make_4x4h_tform_with_zero_rotation_is_a_pure_translation():
    tform = make_4x4h_tform(np.zeros(3), np.array([1.0, 2.0, 3.0]))
    expected = np.eye(4)
    expected[:3, 3] = [1.0, 2.0, 3.0]
    assert np.allclose(tform, expected)


def test_make_4x4h_tform_accepts_a_rotation_matrix():
    """A 2-d ``euler_angles`` is taken as the rotation matrix itself."""
    rot = make_4x4h_tform(np.array([0.3, 0.0, 0.0]), np.zeros(3))[:3, :3]
    tform = make_4x4h_tform(rot, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(tform[:3, :3], rot)


def test_make_4x4h_tform_mvg_mode_negates_the_translation():
    """mvg convention stores -R t where opencv stores t."""
    angles, trans = np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0, 3.0])
    opencv = make_4x4h_tform(angles, trans, mode="opencv")
    mvg = make_4x4h_tform(angles, trans, mode="mvg")

    assert np.allclose(mvg[:3, 3], -(opencv[:3, :3] @ trans))


def test_make_4x4h_tform_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="invalid 4x4 type"):
        make_4x4h_tform(np.zeros(3), np.zeros(3), mode="nonsense")


def test_ext_4x4_to_rod_inverts_make_4x4h_tform():
    """The two are used as a matched pair by the optimiser's parameter vector."""
    angles, trans = np.array([0.1, -0.2, 0.35]), np.array([0.5, -1.0, 2.0])
    tform = make_4x4h_tform(angles, trans)

    rod, recovered_trans = ext_4x4_to_rod(tform)
    assert rod.shape == (3,)
    assert np.allclose(rod, angles)
    assert np.allclose(recovered_trans, trans)

    assert np.allclose(make_4x4h_tform(rod, recovered_trans), tform)


def test_h_tform_translates_points():
    tform = make_4x4h_tform(np.zeros(3), np.array([1.0, 2.0, 3.0]))
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    assert np.allclose(h_tform(points, tform), points + [1.0, 2.0, 3.0])


def test_h_tform_with_fill_zero_ignores_translation():
    """fill=0 marks the input as directions, which do not translate."""
    tform = make_4x4h_tform(np.zeros(3), np.array([1.0, 2.0, 3.0]))
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    assert np.allclose(h_tform(vectors, tform, fill=0), vectors)


def test_h_tform_rotates_directions():
    """A quarter turn about z sends +x to +y."""
    tform = make_4x4h_tform(np.array([0.0, 0.0, np.pi / 2]), np.zeros(3))
    assert np.allclose(h_tform(np.array([[1.0, 0.0, 0.0]]), tform, fill=0), [0.0, 1.0, 0.0])


def test_h_tform_squeezes_a_single_point():
    """A single point comes back as (3,), not (1, 3) -- callers rely on it."""
    tform = make_4x4h_tform(np.zeros(3), np.array([1.0, 0.0, 0.0]))
    assert h_tform(np.array([[0.0, 0.0, 0.0]]), tform).shape == (3,)
    assert h_tform(np.array([0.0, 0.0, 0.0]), tform).shape == (3,)


def test_h_tform_composes():
    a = make_4x4h_tform(np.array([0.1, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    b = make_4x4h_tform(np.array([0.0, 0.2, 0.0]), np.array([0.0, 1.0, 0.0]))
    points = np.array([[0.3, -0.2, 1.0], [0.0, 0.0, 2.0]])

    assert np.allclose(h_tform(h_tform(points, a), b), h_tform(points, b @ a))


def test_h_tform_by_the_identity_is_a_no_op():
    points = np.array([[0.3, -0.2, 1.0], [0.0, 0.0, 2.0]])
    assert np.allclose(h_tform(points, np.eye(4)), points)


# --------------------------------------------------------------------------
# Pose lists
# --------------------------------------------------------------------------


def test_flatten_pose_list_packs_six_parameters_per_pose():
    poses = [
        make_4x4h_tform(np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0, 3.0])),
        make_4x4h_tform(np.zeros(3), np.ones(3)),
    ]
    flat = flatten_pose_list(poses)

    assert flat.shape == (12,)
    assert np.allclose(flat[:3], [0.1, 0.2, 0.3])
    assert np.allclose(flat[3:6], [1.0, 2.0, 3.0])


def test_average_tforms_of_one_pose_is_that_pose():
    tform = make_4x4h_tform(np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0, 3.0]))
    assert np.allclose(average_tforms([tform]), tform)


def test_average_tforms_averages_translations():
    a = make_4x4h_tform(np.zeros(3), np.array([0.0, 0.0, 0.0]))
    b = make_4x4h_tform(np.zeros(3), np.array([2.0, 4.0, 6.0]))
    assert np.allclose(average_tforms([a, b])[:3, 3], [1.0, 2.0, 3.0])


def test_average_tforms_returns_a_rotation():
    a = make_4x4h_tform(np.array([0.1, 0.0, 0.0]), np.zeros(3))
    b = make_4x4h_tform(np.array([0.3, 0.0, 0.0]), np.zeros(3))
    rot = average_tforms([a, b])[:3, :3]

    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(rot), 1.0)


def test_average_tforms_skips_nan_poses():
    """Missing poses arrive as NaN from the optimiser and must be dropped."""
    good = make_4x4h_tform(np.zeros(3), np.array([2.0, 0.0, 0.0]))
    assert np.allclose(average_tforms([good, np.full((4, 4), np.nan)]), good)


def test_average_tforms_of_nothing_is_nan():
    """All-NaN input has no answer, and says so rather than raising."""
    result = average_tforms([np.full((4, 4), np.nan)])
    assert result.shape == (4, 4)
    assert np.all(np.isnan(result))


# --------------------------------------------------------------------------
# Distortion
# --------------------------------------------------------------------------


INTRINSICS = np.array([[800.0, 0.0, 320.0], [0.0, 750.0, 240.0], [0.0, 0.0, 1.0]])


def test_distort_points_takes_one_point_and_returns_two_scalars():
    """The contract is a single (2,) pixel in, an (x, y) pair out.

    ``x, y = (pts - centre)/focal`` unpacks the leading axis, so the input must
    be one flat point -- which is how ``Camera.view_sensor_distortion`` calls
    it, one grid node at a time.  The annotation used to claim ``np.ndarray``;
    it is a tuple.  Note this is the opposite layout to
    ``Camera.project_points``, which returns (n, 2).
    """
    x, y = distort_points(np.array([400.0, 300.0]), INTRINSICS, np.zeros(5))

    assert np.isscalar(x) or np.ndim(x) == 0
    assert np.isclose(x, 400.0)
    assert np.isclose(y, 300.0)


def test_distort_points_is_the_identity_with_zero_coefficients():
    for pixel in ([320.0, 240.0], [400.0, 300.0], [10.0, 470.0]):
        x, y = distort_points(np.array(pixel), INTRINSICS, np.zeros(5))
        assert np.allclose([x, y], pixel)


def test_distort_points_leaves_the_principal_point_alone():
    """The distortion centre is fixed, whatever the coefficients."""
    x, y = distort_points(
        np.array([320.0, 240.0]), INTRINSICS, np.array([-0.3, 0.1, 0.0, 0.0, 0.02])
    )
    assert np.isclose(x, 320.0)
    assert np.isclose(y, 240.0)


def test_radial_distortion_pulls_points_toward_the_centre():
    """Negative k1 is barrel distortion."""
    x, y = distort_points(
        np.array([500.0, 400.0]), INTRINSICS, np.array([-0.3, 0.0, 0.0, 0.0, 0.0])
    )
    assert x < 500.0
    assert y < 400.0


def test_distort_points_matches_opencv():
    """Cross-check the Brown-Conrady implementation against OpenCV.

    This is the same class of check ``test_bundle_correctness.py`` makes for
    the compiled projection path, applied to the pure-Python one: any drift in
    the distortion model shows up as a disagreement with cv2.
    """
    import cv2

    dist = np.array([-0.28, 0.12, 1.5e-3, -2.0e-3, 0.03])
    pixels = np.array([[320.0, 240.0], [400.0, 300.0], [80.0, 420.0], [610.0, 30.0]])

    # take each pixel back to a normalised ray, then let OpenCV distort it
    centre = INTRINSICS[:2, -1]
    focal = np.diag(INTRINSICS)[:2]
    normalised = (pixels - centre) / focal
    rays = np.hstack([normalised, np.ones((len(pixels), 1))])

    expected, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), INTRINSICS, dist)
    expected = expected.reshape(-1, 2)

    ours = np.array([distort_points(px, INTRINSICS, dist) for px in pixels])

    assert np.allclose(ours, expected, atol=1e-6)


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def test_plane_fit_recovers_a_known_plane():
    """Points are columns: the input is (d, n), not (n, d).

    ``plane_fit`` asserts ``shape[0] <= shape[1]``, so an (n, 3) cloud of more
    than three points fails with "there are only 3 points in n dimensions".
    """
    rng = np.random.default_rng(0)
    # a cloud in the z = 2 plane, as (3, n)
    points = np.vstack([rng.normal(size=20), rng.normal(size=20), np.full(20, 2.0)])

    centroid, normal = plane_fit(points)

    assert np.isclose(centroid[2], 2.0)
    assert np.allclose(np.abs(normal), [0.0, 0.0, 1.0], atol=1e-8)


def test_plane_fit_normal_is_a_unit_vector():
    rng = np.random.default_rng(1)
    points = rng.normal(size=(3, 30))
    _, normal = plane_fit(points)
    assert np.isclose(np.linalg.norm(normal), 1.0)


def test_plane_fit_rejects_too_few_points():
    with pytest.raises(AssertionError, match="only 2 points in 3 dimensions"):
        plane_fit(np.zeros((3, 2)))


def test_get_close_square_tuple_covers_n():
    """Used to lay out subplots, so the grid must fit every item."""
    for n in range(1, 40):
        x, y = get_close_square_tuple(n)
        assert x * y >= n
        # and be close to square
        assert abs(x - y) <= 1


def test_px_array_returns_index_grids():
    x, y, h = px_array(res=[4, 6], startZero=True)
    assert x.shape == y.shape == h.shape == (4, 6)
    assert np.allclose(h, 1)
    assert x[0, 0] == 0 and x[-1, 0] == 3
    assert y[0, 0] == 0 and y[0, -1] == 5


def test_px_array_can_centre_on_zero():
    x, y, _ = px_array(res=[4, 4], startZero=False)
    assert 0 in x
    assert 0 in y
    assert x.min() < 0 < x.max()


def test_downsample_valid_averages_blocks():
    arr = np.arange(16, dtype=float).reshape(4, 4)
    result = downsample_valid(arr, 2)

    assert result.shape == (2, 2)
    assert np.isclose(result[0, 0], np.mean(arr[:2, :2]))
    assert np.isclose(result[1, 1], np.mean(arr[2:, 2:]))


def test_downsample_valid_by_one_returns_the_input():
    arr = np.arange(9, dtype=float).reshape(3, 3)
    assert downsample_valid(arr, 1) is arr


def test_downsample_valid_crops_the_remainder():
    """A shape that does not divide is trimmed, not padded."""
    arr = np.ones((5, 7))
    assert downsample_valid(arr, 2).shape == (2, 3)


def test_downsample_valid_takes_no_invalid_argument():
    """The signature no longer advertises masking it never did.

    ``invalid`` was accepted and silently ignored, while the docstring
    promised flagged points were excluded from the average and passed through
    for wholly-invalid blocks.  Neither happened, so the parameter is gone
    rather than left as a trap.
    """
    with pytest.raises(TypeError):
        downsample_valid(np.ones((2, 2)), 2, invalid=np.nan)


def test_downsample_valid_propagates_nan():
    """No value is treated as invalid, so a NaN takes its whole block."""
    arr = np.array([[1.0, 1.0], [1.0, np.nan]])
    assert np.isnan(downsample_valid(arr, 2)).all()


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------


def test_grouper_batches_an_iterable():
    assert list(grouper([1, 2, 3, 4], 2)) == [(1, 2), (3, 4)]


def test_grouper_pads_a_short_final_group():
    assert list(grouper([1, 2, 3], 2)) == [(1, 2), (3, None)]
    assert list(grouper([1, 2, 3], 2, fillvalue=0)) == [(1, 2), (3, 0)]


def test_list_dict_to_np_array_converts_nested_lists():
    payload = {"a": [1, 2, 3], "nested": {"b": [[1, 2], [3, 4]]}, "keep": 5}
    result = list_dict_to_np_array(payload)

    assert isinstance(result["a"], np.ndarray)
    assert isinstance(result["nested"]["b"], np.ndarray)
    assert result["nested"]["b"].shape == (2, 2)
    assert result["keep"] == 5


def test_list_dict_to_np_array_passes_through_non_dicts():
    assert list_dict_to_np_array(7) == 7


def test_colourmap_to_colour_list_length():
    colours = colourmap_to_colour_list(5, __import__("matplotlib").cm.viridis)
    assert len(colours) == 5


def test_get_subfolder_names_is_naturally_sorted(tmp_path):
    """Camera names come from folder names, so cam_10 must follow cam_9."""
    for name in ["cam_10", "cam_2", "cam_1"]:
        (tmp_path / name).mkdir()
    (tmp_path / "a_file.txt").write_text("not a folder")

    assert get_subfolder_names(tmp_path) == ["cam_1", "cam_2", "cam_10"]


def test_get_subfolder_names_can_return_full_paths(tmp_path):
    (tmp_path / "cam_1").mkdir()
    paths = get_subfolder_names(tmp_path, return_full_path=True)
    assert paths == [tmp_path / "cam_1"]


def test_glob_ims_finds_images_of_several_extensions(tmp_path):
    for name in ["a.png", "b.jpg", "c.tiff", "ignored.txt"]:
        (tmp_path / name).write_bytes(b"")

    found = [p.name for p in glob_ims(tmp_path)]

    assert "ignored.txt" not in found
    assert {"a.png", "b.jpg"} <= set(found)


def test_write_colour_ply_writes_a_parsable_header(tmp_path):
    target = tmp_path / "cloud.ply"
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    cols = np.array([[255, 0, 0], [0, 255, 0]])

    write_colour_ply(target, verts, cols)

    lines = target.read_text().splitlines()
    assert lines[0] == "ply"
    assert "element vertex 2" in lines
    body = lines[lines.index("end_header") + 1 :]
    assert len([line for line in body if line.strip()]) == 2
    assert body[1].startswith("1.00000000 2.00000000 3.00000000 0 255 0")
