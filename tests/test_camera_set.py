"""``CameraSet`` as a container and as a geometric object.

At 33% coverage this was the largest genuine gap in the package, and the
uncovered lines held three outright defects: ``make_subset(cam_key=...)`` called
a name-mangled ``__update`` that does not exist, ``transform(in_place=False)``
returned ``None``, and ``__iter__`` returned ``self`` so overlapping loops
shared one cursor.  Each has a named regression test below.

Everything here is synthetic and runs in milliseconds; ``test_calibration_*``
covers the same class against real images.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyCamSet import Camera, CameraSet
from pyCamSet.cameras.camera_set import get_v_vec, make_cam_dict
from pyCamSet.utils.general_utils import h_tform, make_4x4h_tform

from conftest import REF_INTRINSIC, REF_RES, make_camera

# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_make_cam_dict_builds_named_cameras():
    names = ["a", "b"]
    cams = make_cam_dict(
        names,
        [np.eye(4), np.eye(4)],
        [REF_INTRINSIC, REF_INTRINSIC],
        dist_coefs=[np.zeros(5)] * 2,
        res=[REF_RES] * 2,
    )
    assert list(cams) == names
    for name, cam in cams.items():
        assert isinstance(cam, Camera)
        assert cam.name == name


def test_make_cam_dict_defaults_distortion_to_zero():
    """dist_coefs is optional and must default to an undistorted model."""
    cams = make_cam_dict(["a"], [np.eye(4)], [REF_INTRINSIC], res=[REF_RES])
    assert np.allclose(cams["a"].distortion_coefs, 0)


def test_camera_set_from_parameter_lists():
    camset = CameraSet(
        camera_names=["a", "b"],
        extrinsic_matrices=[np.eye(4)] * 2,
        intrinsic_matrices=[REF_INTRINSIC] * 2,
        distortion_coefs=[np.zeros(5)] * 2,
        res=[REF_RES] * 2,
    )
    assert camset.get_names() == ["a", "b"]
    assert len(camset) == 2


def test_empty_camera_set_is_allowed():
    """``make_subset`` builds an empty set then fills it, so this must work."""
    camset = CameraSet()
    assert camset.get_cam_dict() is None


def test_partial_parameters_are_rejected():
    """Half a specification is a mistake, not a set of defaults."""
    with pytest.raises(ValueError, match="requires names"):
        CameraSet(camera_names=["a"], extrinsic_matrices=[np.eye(4)])


# --------------------------------------------------------------------------
# The container protocol
# --------------------------------------------------------------------------


def test_len_and_get_n_cams_agree(synthetic_camset):
    assert len(synthetic_camset) == synthetic_camset.get_n_cams() == 3


def test_getitem_by_name(synthetic_camset):
    cam = synthetic_camset["centre"]
    assert isinstance(cam, Camera)
    assert cam.name == "centre"


def test_getitem_by_index_follows_insertion_order(synthetic_camset):
    assert [synthetic_camset[i].name for i in range(3)] == ["left", "centre", "right"]


def test_getitem_by_negative_index(synthetic_camset):
    assert synthetic_camset[-1].name == "right"


def test_getitem_prefers_a_numeric_key_over_a_positional_index():
    """A set keyed by numbers must index by key, not by position.

    ``__getitem__`` checks the dict before the list precisely so that a set
    whose camera names are ints behaves like a mapping.
    """
    cams = {5: make_camera("five"), 0: make_camera("zero")}
    camset = CameraSet(camera_dict=cams)
    assert camset[5].name == "five"  # the key, not _cam_list[5]
    assert camset[0].name == "zero"


def test_getitem_with_a_slice_returns_a_subset(synthetic_camset):
    subset = synthetic_camset[0:2]
    assert isinstance(subset, CameraSet)
    assert subset.get_names() == ["left", "centre"]


def test_getitem_with_a_list_returns_a_subset(synthetic_camset):
    subset = synthetic_camset[[0, 2]]
    assert isinstance(subset, CameraSet)
    assert subset.get_names() == ["left", "right"]


def test_getitem_with_an_int_array_returns_a_subset(synthetic_camset):
    subset = synthetic_camset[np.array([2, 0])]
    assert isinstance(subset, CameraSet)
    assert subset.get_names() == ["right", "left"]


def test_getitem_with_a_float_array_is_rejected(synthetic_camset):
    """Float indices would silently truncate, so they are refused."""
    with pytest.raises(ValueError, match="int arrays"):
        synthetic_camset[np.array([0.0, 1.0])]


def test_getitem_with_a_missing_name_raises(synthetic_camset):
    with pytest.raises(KeyError):
        synthetic_camset["nonexistent"]


def test_setitem_adds_a_camera_and_refreshes_the_list(synthetic_camset):
    synthetic_camset["extra"] = make_camera("extra", translation=(0.1, 0.0, 0.0))

    assert len(synthetic_camset) == 4
    assert synthetic_camset.get_names()[-1] == "extra"
    # _cam_list must be rebuilt, or index access goes stale
    assert synthetic_camset[3].name == "extra"


def test_iteration_yields_every_camera_in_order(synthetic_camset):
    assert [cam.name for cam in synthetic_camset] == ["left", "centre", "right"]


def test_iteration_can_be_repeated(synthetic_camset):
    first = [cam.name for cam in synthetic_camset]
    second = [cam.name for cam in synthetic_camset]
    assert first == second


def test_nested_iteration_visits_every_pair(synthetic_camset):
    """Regression: overlapping loops must not share a cursor.

    ``__iter__`` returned ``self``, making the set its own iterator with a
    single ``ind``.  The inner loop exhausted it, so the outer loop stopped
    after one step and a pairwise walk over n cameras yielded n pairs instead
    of n**2 -- silently skipping most camera pairs.
    """
    pairs = [(a.name, b.name) for a in synthetic_camset for b in synthetic_camset]
    assert len(pairs) == 9
    assert ("right", "left") in pairs


def test_independent_iterators_do_not_interfere(synthetic_camset):
    first, second = iter(synthetic_camset), iter(synthetic_camset)
    assert next(first).name == "left"
    assert next(second).name == "left"  # not "centre"


# --------------------------------------------------------------------------
# Subsets
# --------------------------------------------------------------------------


def test_make_subset_with_a_list_of_indices(synthetic_camset):
    subset = synthetic_camset.make_subset([0, 2])
    assert subset.get_names() == ["left", "right"]
    assert len(subset) == 2
    assert subset.get_n_cams() == 2


def test_make_subset_with_a_slice(synthetic_camset):
    assert synthetic_camset.make_subset(slice(1, None)).get_names() == ["centre", "right"]


def test_make_subset_with_an_array(synthetic_camset):
    assert synthetic_camset.make_subset(np.array([1])).get_names() == ["centre"]


def test_make_subset_rejects_an_unusable_identifier(synthetic_camset):
    with pytest.raises(ValueError, match="not a valid subset identifier"):
        synthetic_camset.make_subset("centre")


def test_make_subset_with_a_cam_key_filters_by_name():
    """Regression: the cam_key branch called a method that does not exist.

    ``new_camset.__update()`` inside the class mangles to
    ``_CameraSet__update``; the method is ``_update``.  Every cam_key subset
    raised AttributeError, so the whole branch was dead.
    """
    cams = {
        "rig_left": make_camera("rig_left"),
        "rig_right": make_camera("rig_right"),
        "witness": make_camera("witness"),
    }
    camset = CameraSet(camera_dict=cams)

    subset = camset.make_subset(slice(None), cam_key="rig")

    assert subset.get_names() == ["rig_left", "rig_right"]
    # _update must have run, or these are stale
    assert len(subset) == 2
    assert subset[0].name == "rig_left"


def test_make_subset_with_a_cam_key_and_a_list():
    cams = {f"rig_{i}": make_camera(f"rig_{i}") for i in range(3)}
    cams["other"] = make_camera("other")
    camset = CameraSet(camera_dict=cams)

    subset = camset.make_subset([0, 2], cam_key="rig")
    assert subset.get_names() == ["rig_0", "rig_2"]


def test_make_subset_with_an_unmatched_cam_key_raises(synthetic_camset):
    with pytest.raises(ValueError, match="found no matching camera names"):
        synthetic_camset.make_subset(slice(None), cam_key="telescope")


def test_make_subset_shares_camera_objects(synthetic_camset):
    """Subsets are views, not copies: the same Camera instances come back."""
    subset = synthetic_camset.make_subset([0])
    assert subset["left"] is synthetic_camset["left"]


# --------------------------------------------------------------------------
# Combining sets
# --------------------------------------------------------------------------


def test_adding_two_sets_merges_them():
    left = CameraSet(camera_dict={"a": make_camera("a")})
    right = CameraSet(camera_dict={"b": make_camera("b")})

    combined = left + right

    assert combined.get_names() == ["a", "b"]
    assert len(combined) == 2


def test_adding_mutates_the_left_operand():
    """Documenting current behaviour: ``+`` is in-place and returns self.

    This is surprising for an operator -- ``a + b`` modifies ``a`` -- but it is
    the established contract, so it is pinned rather than changed here.  A
    caller relying on ``a`` surviving intact needs ``deepcopy``.
    """
    left = CameraSet(camera_dict={"a": make_camera("a")})
    right = CameraSet(camera_dict={"b": make_camera("b")})

    combined = left + right

    assert combined is left
    assert len(left) == 2


def test_adding_sets_with_a_shared_name_is_refused():
    """Merging would silently drop a camera, so it is an error."""
    left = CameraSet(camera_dict={"a": make_camera("a")})
    right = CameraSet(camera_dict={"a": make_camera("a")})

    with pytest.raises(ValueError, match="share camera names"):
        left + right


def test_adding_a_non_camera_set_is_refused(synthetic_camset):
    with pytest.raises(ValueError, match="Can only add together camera sets"):
        synthetic_camset + "not a camera set"


# --------------------------------------------------------------------------
# Accessors
# --------------------------------------------------------------------------


def test_accessors_agree_with_each_other(synthetic_camset):
    names = synthetic_camset.get_names()
    cam_list = synthetic_camset.get_cam_list()
    cam_dict = synthetic_camset.get_cam_dict()

    assert names == list(cam_dict)
    assert cam_list == [cam_dict[name] for name in names]
    assert [cam.name for cam in cam_list] == names


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def test_transform_in_place_moves_every_camera(synthetic_camset, world_points):
    shift = make_4x4h_tform(np.zeros(3), np.array([0.1, 0.0, 0.0]))
    before = {c.name: c.extrinsic.copy() for c in synthetic_camset}

    result = synthetic_camset.transform(shift)

    assert result is None  # in place, by default
    for cam in synthetic_camset:
        assert not np.allclose(cam.extrinsic, before[cam.name])


def test_transform_not_in_place_returns_a_new_set(synthetic_camset):
    """Regression: this returned ``None``.

    The non-in-place branch deep-copied the set then returned the result of
    ``copy.transform(m)`` -- which runs the in-place branch and returns None.
    """
    shift = make_4x4h_tform(np.zeros(3), np.array([0.1, 0.0, 0.0]))
    before = {c.name: c.extrinsic.copy() for c in synthetic_camset}

    moved = synthetic_camset.transform(shift, in_place=False)

    assert isinstance(moved, CameraSet)
    assert moved is not synthetic_camset
    # the original is untouched
    for cam in synthetic_camset:
        assert np.allclose(cam.extrinsic, before[cam.name])
    # and the copy actually moved
    for cam in moved:
        assert not np.allclose(cam.extrinsic, before[cam.name])


def test_transforming_by_the_identity_changes_nothing(synthetic_camset):
    before = {c.name: c.extrinsic.copy() for c in synthetic_camset}
    synthetic_camset.transform(np.eye(4))
    for cam in synthetic_camset:
        assert np.allclose(cam.extrinsic, before[cam.name])


def test_set_reference_cam_puts_that_camera_at_the_origin(synthetic_camset):
    synthetic_camset.set_reference_cam("right")
    assert np.allclose(synthetic_camset["right"].extrinsic, np.eye(4))


def test_set_reference_cam_preserves_relative_geometry(synthetic_camset, world_points):
    """Rebasing the world frame must not change what the cameras see.

    ``Camera.transform`` right-multiplies (``extrinsic @ T``) and
    ``set_reference_cam`` passes ``T = inv(E_ref)``, so a camera's projection
    becomes ``E_c inv(E_ref) p``.  Expressing the same physical points in the
    new frame therefore means moving them by ``E_ref``, and the two
    cancel: ``E_c inv(E_ref) E_ref p == E_c p``.
    """
    before = [cam.project_points(world_points) for cam in synthetic_camset]

    ref_extrinsic = synthetic_camset["right"].extrinsic.copy()
    synthetic_camset.set_reference_cam("right")
    moved_points = h_tform(world_points, ref_extrinsic)

    after = [cam.project_points(moved_points) for cam in synthetic_camset]
    for expected, actual in zip(before, after):
        assert np.allclose(expected, actual)


def test_scale_set_2n_scales_every_camera(synthetic_camset):
    synthetic_camset.scale_set_2n(1)
    for cam in synthetic_camset:
        assert list(cam.res) == [REF_RES[0] // 2, REF_RES[1] // 2]


def test_project_points_to_all_cams_returns_one_dict_per_point(synthetic_camset, world_points):
    projections = synthetic_camset.project_points_to_all_cams(world_points)

    assert isinstance(projections, list)
    assert len(projections) == len(world_points)
    for entry in projections:
        assert set(entry) == set(synthetic_camset.get_names())
        for uv in entry.values():
            assert np.shape(uv) == (2,)


def test_project_points_to_all_cams_unwraps_a_single_point(synthetic_camset):
    """A bare (3,) point returns the dict itself, not a one-element list."""
    single = synthetic_camset.project_points_to_all_cams(np.array([0.0, 0.0, 1.0]))
    assert isinstance(single, dict)
    assert set(single) == set(synthetic_camset.get_names())


def test_project_points_to_all_cams_accepts_a_list(synthetic_camset):
    as_list = synthetic_camset.project_points_to_all_cams([[0.0, 0.0, 1.0]])
    as_array = synthetic_camset.project_points_to_all_cams(np.array([[0.0, 0.0, 1.0]]))
    for from_list, from_array in zip(as_list, as_array):
        for name in synthetic_camset.get_names():
            assert np.allclose(from_list[name], from_array[name])


def test_projection_agrees_with_the_individual_cameras(synthetic_camset, world_points):
    """The set must not do its own geometry; it delegates to each Camera."""
    projections = synthetic_camset.project_points_to_all_cams(world_points)
    for name in synthetic_camset.get_names():
        direct = synthetic_camset[name].project_points(world_points)
        gathered = np.array([entry[name] for entry in projections])
        assert np.allclose(gathered, direct)


def test_triangulation_recovers_projected_points(synthetic_camset, world_points):
    """Project then triangulate must return the points you started with."""
    projections = synthetic_camset.project_points_to_all_cams(world_points)
    reconstructed = synthetic_camset.multi_cam_triangulate(projections)
    assert np.allclose(reconstructed, world_points, atol=1e-6)


def test_get_similar_angles_excludes_the_query_camera(synthetic_camset):
    """The query camera is at angle 0 to itself, so it must be masked out."""
    closest = synthetic_camset.get_similar_angles(0, 2)
    assert len(closest) == 2
    assert 0 not in closest


def test_get_v_vec_is_the_cameras_optical_axis():
    """An unrotated camera looks down +z."""
    assert np.allclose(get_v_vec(np.eye(4)), [0, 0, 1])

    # a 180 degree turn about y points it the other way
    flipped = make_4x4h_tform(np.array([0.0, np.pi, 0.0]), np.zeros(3))
    assert np.allclose(get_v_vec(flipped), [0, 0, -1], atol=1e-9)


# --------------------------------------------------------------------------
# Calibration history
# --------------------------------------------------------------------------


def test_a_fresh_camera_set_has_no_calibration_history(synthetic_camset):
    assert synthetic_camset.calibration_params is None
    assert synthetic_camset.calibration_result is None
    assert synthetic_camset.calibration_jac is None
    assert synthetic_camset.calibration_handler is None


def test_set_calibration_history_stores_the_optimisation_output(synthetic_camset):
    results = {
        "x": np.arange(4.0),
        "fun": np.ones(6),
        "jac": np.eye(6, 4),
    }
    synthetic_camset.set_calibration_history(results, param_handler="sentinel")

    assert np.array_equal(synthetic_camset.calibration_params, results["x"])
    assert np.array_equal(synthetic_camset.calibration_result, results["fun"])
    assert np.array_equal(synthetic_camset.calibration_jac, results["jac"])
    assert synthetic_camset.calibration_handler == "sentinel"
