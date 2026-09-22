"""The bundle handlers, below the calibration entry point.

``TemplateBundleHandler`` (fixed target) and ``SelfBundleHandler`` (target
geometry solved for as well) were only ever reached through a full
``calibrate_cameras`` run, so a fault in the parameter packing or the gauge
handling surfaced as "the ccube calibration drifted" rather than as a named
function.  ``test_free_point_handler.py`` already does this for the third
handler; this is the same shape for the other two.

The pure pieces -- index layouts, the visibility and misalignment checks, the
detection accessors -- are synthetic and run in the fast suite.  The parts that
genuinely need a real problem share the session-scoped detections from
conftest, so they cost the detection once rather than once per test.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from pyCamSet import CameraSet
from pyCamSet.calibration_targets import ImageDetection, TargetDetection
from pyCamSet.optimisation.numba_schur import ParamGroup, spec_from_groups
from pyCamSet.optimisation.standard_bundle_handler import (
    SelfBundleHandler,
    StandardBundlePrimitive,
    find_gauge_points,
)
from pyCamSet.optimisation.template_handler import (
    DEFAULT_OPTIONS,
    TemplateBundleHandler,
    TemplateBundlePrimitive,
    check_feasiblity_and_update_refpose,
    check_for_target_misalignment,
    estimate_initial_rig,
    per_image_reprojection,
)
from pyCamSet.utils.general_utils import make_4x4h_tform

from conftest import make_camera

# --------------------------------------------------------------------------
# A synthetic fixed-target problem, for everything that does not need images
# --------------------------------------------------------------------------

N_GRID = 4
N_POINTS = N_GRID * N_GRID
N_IMAGES = 3


class GridTarget:
    """The smallest thing the handlers accept: point_data plus a square size."""

    def __init__(self, spacing=0.02):
        corners = np.mgrid[0:N_GRID, 0:N_GRID].reshape(2, -1).T * spacing
        corners = corners - corners.mean(axis=0)
        flat = np.hstack([corners, np.zeros((len(corners), 1))]).astype(np.float64)
        self.point_data = flat[None, ...]  # (1, n, 3), as make_local would leave it
        self.point_local = self.point_data.copy()
        self.original_points = self.point_data.copy()
        self.square_size = spacing
        self.valid_map = True
        self.input_args = {}


@pytest.fixture
def synthetic_problem():
    """Three cameras, three target poses, exact detections."""
    from pyCamSet.utils.general_utils import h_tform

    cams = CameraSet(
        camera_dict={
            name: make_camera(name, translation=offset)
            for name, offset in [
                ("left", (-0.06, 0.0, 0.0)),
                ("centre", (0.0, 0.0, 0.0)),
                ("right", (0.06, 0.0, 0.0)),
            ]
        }
    )
    target = GridTarget()
    poses = [
        make_4x4h_tform(np.array([0.02 * i, -0.01, 0.0]), np.array([0.0, 0.0, 0.9 + 0.04 * i]))
        for i in range(N_IMAGES)
    ]

    detection = TargetDetection(cam_names=cams.get_names())
    points = target.point_data.reshape(-1, 3)
    for im_num, pose in enumerate(poses):
        placed = h_tform(points, pose)
        for name in cams.get_names():
            detection.add_detection(
                name,
                im_num,
                ImageDetection(
                    keys=np.arange(N_POINTS),
                    image_points=cams[name].project_points(placed),
                ),
            )
    return cams, target, detection, poses


@pytest.fixture
def template_handler(synthetic_problem):
    cams, target, detection, _ = synthetic_problem
    return TemplateBundleHandler(
        camset=cams, target=target, detection=detection, options={"outliers": "n"}
    )


# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------


def test_options_do_not_leak_between_handlers(synthetic_problem):
    """DEFAULT_OPTIONS is module level, so it must be copied, not updated."""
    cams, target, detection, _ = synthetic_problem
    before = dict(DEFAULT_OPTIONS)

    TemplateBundleHandler(
        camset=cams, target=target, detection=detection,
        options={"verbosity": 99, "outliers": "n"},
    )
    second = TemplateBundleHandler(camset=cams, target=target, detection=detection)

    assert DEFAULT_OPTIONS == before
    assert second.problem_opts["verbosity"] == before["verbosity"]


def test_given_options_override_the_defaults(template_handler):
    assert template_handler.problem_opts["outliers"] == "n"
    assert template_handler.problem_opts["solver"] == DEFAULT_OPTIONS["solver"]


# --------------------------------------------------------------------------
# TemplateBundlePrimitive
# --------------------------------------------------------------------------


def test_template_primitive_index_layout():
    prim = TemplateBundlePrimitive(
        poses=np.zeros((4, 6)), extr=np.zeros((2, 6)), intr=np.zeros((2, 9))
    )
    assert prim.intr_end == 9 * 2
    assert prim.extr_end == 9 * 2 + 6 * 2
    assert prim.pose_end == 9 * 2 + 6 * 2 + 6 * 4


def test_template_primitive_round_trips_parameters():
    prim = TemplateBundlePrimitive(
        poses=np.zeros((1, 6)), extr=np.zeros((1, 6)), intr=np.zeros((1, 9))
    )
    intr, extr, pose = np.arange(9.0), np.arange(9.0, 15.0), np.arange(15.0, 21.0)

    got_intr, got_extr, got_pose = prim.return_bundle_primitives(
        np.concatenate([intr, extr, pose])
    )

    assert np.allclose(got_intr[0], intr)
    assert np.allclose(got_extr[0], extr)
    assert np.allclose(got_pose[0], pose)


def test_template_primitive_honours_fixed_poses():
    """A fixed pose keeps its preset value and costs no parameters."""
    poses = np.zeros((3, 6))
    poses[0] = 7.0
    prim = TemplateBundlePrimitive(
        poses=poses, extr=np.zeros((1, 6)), intr=np.zeros((1, 9)),
        poses_unfixed=np.array([False, True, True]),
    )

    assert prim.free_poses == 2
    assert prim.pose_end == 9 + 6 + 12

    _, _, got = prim.return_bundle_primitives(np.zeros(prim.pose_end))
    assert np.allclose(got[0], 7.0)


# --------------------------------------------------------------------------
# StandardBundlePrimitive
# --------------------------------------------------------------------------


def test_standard_primitive_adds_the_point_block():
    prim = StandardBundlePrimitive(
        poses=np.zeros((2, 6)),
        bundle_points=np.zeros(9),
        extr=np.zeros((2, 6)),
        intr=np.zeros((2, 9)),
    )
    assert prim.intr_end == 18
    assert prim.extr_end == 18 + 12
    assert prim.pose_end == 18 + 12 + 12
    assert prim.bdpt_end == 18 + 12 + 12 + 9


def test_standard_primitive_round_trips_points():
    prim = StandardBundlePrimitive(
        poses=np.zeros((1, 6)),
        bundle_points=np.zeros(6),
        extr=np.zeros((1, 6)),
        intr=np.zeros((1, 9)),
    )
    points = np.arange(21.0, 27.0)
    params = np.concatenate([np.zeros(21), points])

    _, _, _, got_points = prim.return_bundle_primitives(params)
    assert np.allclose(got_points, points.reshape(-1, 3))


def test_standard_primitive_fixed_points_shrink_the_vector():
    """Gauge fixing works by marking individual coordinates fixed."""
    unfixed = np.ones(9, dtype=bool)
    unfixed[:7] = False  # the 7 gauge freedoms
    prim = StandardBundlePrimitive(
        poses=np.zeros((1, 6)),
        bundle_points=np.zeros(9),
        extr=np.zeros((1, 6)),
        intr=np.zeros((1, 9)),
        bundle_points_unfixed=unfixed,
    )
    assert prim.free_bdpt == 2
    assert prim.bdpt_end == prim.pose_end + 2


# --------------------------------------------------------------------------
# find_gauge_points: what picks the gauge
# --------------------------------------------------------------------------


def test_find_gauge_points_picks_a_spanning_triple():
    points = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]])
    (i0, i1, i2), axis = find_gauge_points(points)

    a = points[i0] - points[i1]
    b = points[i0] - points[i2]
    assert np.linalg.norm(np.cross(a, b)) > 1e-8


def test_the_held_coordinate_is_the_one_a_rotation_moves():
    """The last freedom is a rotation about AB, which carries C along the
    normal of ABC: a coordinate in the plane of ABC would not see it."""
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    (i0, i1, i2), axis = find_gauge_points(points)

    ab = points[i1] - points[i0]
    tangent = np.cross(ab / np.linalg.norm(ab), points[i2] - points[i0])
    assert abs(tangent[axis]) == pytest.approx(np.linalg.norm(tangent))


def test_find_gauge_points_fixes_only_points_it_is_offered():
    """A point no camera saw is not a parameter, so holding it fixed removes
    no freedom: the gauge has to be taken from the points that were seen."""
    points = np.array(
        [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [9, 0, 0], [9, 9, 0], [-9, 0, 0]]
    )
    seen = [0, 1, 2]

    assert set(find_gauge_points(points, candidates=seen)[0]) <= set(seen)
    # left to the whole target, the wider spread of the unseen points wins
    assert set(find_gauge_points(points)[0]) - set(seen)


def test_find_gauge_points_rejects_colinear_candidates():
    """A degenerate choice cannot fix a gauge, and must say so -- even when
    the target as a whole is not degenerate."""
    points = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match="colinear"):
        find_gauge_points(points, candidates=[0, 1, 2])


def test_find_gauge_points_needs_three_points():
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match="three points"):
        find_gauge_points(points, candidates=[0, 1])


# --------------------------------------------------------------------------
# Visibility and misalignment checks
# --------------------------------------------------------------------------


def _visible_stack(n_cams=3, n_poses=4, missing=()):
    """A (cams, poses, 4, 4) transform stack, NaN where a pose is unseen."""
    stack = np.tile(np.eye(4), (n_cams, n_poses, 1, 1))
    for cam, pose in missing:
        stack[cam, pose] = np.nan
    return stack


def test_refpose_is_kept_when_every_camera_sees_it():
    ref_pose, try_graph = check_feasiblity_and_update_refpose(_visible_stack(), ref_pose=0)
    assert ref_pose == 0
    assert try_graph is False


def test_refpose_moves_to_a_pose_everyone_can_see():
    """Pose 0 is unseen by camera 1, so pose 1 should be chosen instead."""
    stack = _visible_stack(missing=[(1, 0)])
    ref_pose, try_graph = check_feasiblity_and_update_refpose(stack, ref_pose=0)

    assert ref_pose == 1
    assert try_graph is False


def test_no_commonly_visible_pose_asks_for_the_graph_method():
    """Every pose is missed by someone, so no single reference works."""
    stack = _visible_stack(n_cams=2, n_poses=2, missing=[(0, 0), (1, 1)])
    ref_pose, try_graph = check_feasiblity_and_update_refpose(stack, ref_pose=0)

    assert ref_pose == -1
    assert try_graph is True


def test_consistent_transforms_raise_no_misalignment_warning():
    """A rigid rig produces identical relative transforms, so nothing is flagged."""
    stack = _visible_stack(n_cams=2, n_poses=5)
    stack[1] = np.tile(make_4x4h_tform(np.zeros(3), np.array([0.1, 0, 0])), (5, 1, 1))

    report = check_for_target_misalignment(stack, ref_cam=0)

    assert report.flags == []
    assert report.per_camera[0].translation_stdev_mm == pytest.approx(0.0)


def test_inconsistent_translations_are_reported():
    """A camera that appears to move between images is the classic symptom
    of misordered or temporally misaligned images, and must be flagged."""
    stack = _visible_stack(n_cams=2, n_poses=5)
    for pose in range(5):
        # a metre of drift per image: far past the 10mm threshold
        stack[1, pose] = make_4x4h_tform(np.zeros(3), np.array([pose * 1.0, 0, 0]))

    report = check_for_target_misalignment(stack, ref_cam=0)

    assert any("mm relative to camera" in f for f in report.flags)
    assert report.per_camera[0].translation_stdev_mm > 10.0


def test_the_reference_camera_is_not_measured_against_itself():
    """It is the datum, so it has no relative scatter to report."""
    stack = _visible_stack(n_cams=3, n_poses=4)

    report = check_for_target_misalignment(
        stack, ref_cam=1, cam_names=["a", "b", "c"])

    assert report.reference_camera == "b"
    assert [c.name for c in report.per_camera] == ["a", "c"]


def test_the_consistency_block_is_logged(caplog):
    stack = _visible_stack(n_cams=2, n_poses=4)

    with caplog.at_level(logging.INFO):
        check_for_target_misalignment(stack, ref_cam=0)

    assert "Rig consistency" in caplog.text


# --------------------------------------------------------------------------
# TemplateBundleHandler
# --------------------------------------------------------------------------


def test_handler_reports_its_problem_size(template_handler):
    assert template_handler.problem_maximums["max_cams"] == 3
    assert template_handler.problem_maximums["max_imgs"] == N_IMAGES


def test_handler_copies_the_detection_it_is_given(synthetic_problem):
    """The handler mutates its detection, so it must not be the caller's."""
    cams, target, detection, _ = synthetic_problem
    handler = TemplateBundleHandler(camset=cams, target=target, detection=detection)

    assert handler.detection is not detection
    assert np.array_equal(handler.detection.get_data(), detection.get_data())


def test_get_detection_round_trips_the_data(template_handler):
    detection = template_handler.get_detection()

    assert isinstance(detection, TargetDetection)
    assert detection.cam_names == template_handler.cam_names
    assert np.array_equal(detection.get_data(), template_handler.get_detection_data())


def test_get_detection_data_drops_missing_poses(template_handler):
    """A pose marked missing must not reach the optimiser."""
    template_handler.missing_poses = np.zeros(N_IMAGES, dtype=bool)
    full = template_handler.get_detection_data()

    template_handler.missing_poses[1] = True
    trimmed = template_handler.get_detection_data()

    assert len(trimmed) < len(full)
    assert 1 not in np.unique(trimmed[:, 1])


def test_get_detection_data_with_no_missing_poses_keeps_everything(template_handler):
    template_handler.missing_poses = None
    assert len(template_handler.get_detection_data()) == len(
        template_handler.detection.get_data()
    )


def test_gauge_fixes_is_none_for_the_template_handler(template_handler):
    """The fixed target already fixes the gauge, so there are no multipliers."""
    assert template_handler.gauge_fixes() is None


def test_the_template_handler_can_build_its_jacobian(template_handler):
    assert template_handler.can_make_jac()


def test_parameter_groups_cover_the_free_parameters(template_handler):
    groups = template_handler.parameter_groups()

    assert [g.name for g in groups] == ["intr", "extr", "pose"]
    assert all(isinstance(g, ParamGroup) for g in groups)
    assert sum(g.n_free for g in groups) == template_handler.bundlePrimitive.pose_end


def test_the_eliminated_group_is_the_per_image_pose(template_handler):
    """Poses are the block diagonal group for a fixed target problem."""
    groups = template_handler.parameter_groups()
    spec = spec_from_groups(groups)

    assert groups[-1].name == "pose"
    assert spec.elim_size == 6


# --------------------------------------------------------------------------
# Outlier exclusion, which used to prompt on stdin
# --------------------------------------------------------------------------


def test_outlier_exclusion_requires_missing_poses_to_be_set(template_handler):
    template_handler.missing_poses = None
    with pytest.raises(ValueError, match="missing poses should be initialised"):
        template_handler.find_and_exclude_transform_outliers(np.zeros(N_IMAGES))


def test_outlier_exclusion_keeps_everything_when_errors_are_even(template_handler):
    template_handler.missing_poses = np.zeros(N_IMAGES, dtype=bool)
    template_handler.find_and_exclude_transform_outliers(np.ones(N_IMAGES))

    assert not np.any(template_handler.missing_poses)


def test_outlier_exclusion_answers_itself_when_told_to(synthetic_problem):
    """options={'outliers': 'n'} must skip the prompt entirely.

    The default is 'ask', which called input(); reached from calc_initial_params
    on every calibration, that hangs any unattended run that finds an outlier.
    """
    cams, target, detection, _ = synthetic_problem
    handler = TemplateBundleHandler(
        camset=cams, target=target, detection=detection, options={"outliers": "n"}
    )
    handler.missing_poses = np.zeros(N_IMAGES, dtype=bool)

    errors = np.ones(N_IMAGES)
    errors[0] = 1000.0  # an unmistakable outlier

    handler.find_and_exclude_transform_outliers(errors)

    # answered 'n', so the outlier is reported but nothing is removed
    assert not np.any(handler.missing_poses)


def test_outlier_exclusion_removes_when_told_to(synthetic_problem):
    cams, target, detection, _ = synthetic_problem
    handler = TemplateBundleHandler(
        camset=cams, target=target, detection=detection, options={"outliers": "y"}
    )
    handler.missing_poses = np.zeros(N_IMAGES, dtype=bool)

    errors = np.ones(N_IMAGES)
    errors[2] = 1000.0

    handler.find_and_exclude_transform_outliers(errors)

    assert handler.missing_poses[2]


def test_outlier_exclusion_does_not_read_stdin(synthetic_problem, monkeypatch):
    """Belt and braces: no code path here may block on input().

    With the default 'ask' and no terminal, ask_yes_no falls back rather than
    calling input at all.
    """
    def explode(*args, **kwargs):
        raise AssertionError("input() was called from an automated run")

    monkeypatch.setattr("builtins.input", explode)

    cams, target, detection, _ = synthetic_problem
    handler = TemplateBundleHandler(camset=cams, target=target, detection=detection)
    handler.missing_poses = np.zeros(N_IMAGES, dtype=bool)

    errors = np.ones(N_IMAGES)
    errors[1] = 1000.0
    handler.find_and_exclude_transform_outliers(errors)


# find_and_exclude_transform_outliers reads only missing_poses and the error
# array it is handed, so these size them for the statistic rather than for the
# three image fixture: MAD has nothing to say about two surviving samples.
N_MANY = 12


def _even_errors():
    """A believable spread of per image error, with no outlier in it."""
    return np.linspace(0.9, 1.1, N_MANY)


def test_outlier_exclusion_still_fires_when_an_image_has_no_error(synthetic_problem):
    """An image with no recoverable pose has a NaN error, and a NaN used to
    turn the whole detector off: the median went NaN, every comparison
    against it read False, and nothing was ever removed -- not the unposed
    image, and not the genuinely bad one beside it."""
    cams, target, detection, _ = synthetic_problem
    handler = TemplateBundleHandler(
        camset=cams, target=target, detection=detection, options={"outliers": "y"}
    )
    handler.missing_poses = np.zeros(N_MANY, dtype=bool)

    errors = _even_errors()
    errors[1] = 1000.0
    errors[2] = np.nan

    handler.find_and_exclude_transform_outliers(errors)

    assert handler.missing_poses[1], "the bad image was hidden by the NaN"
    assert handler.missing_poses[2], "the image with no error at all was kept"
    assert int(np.sum(handler.missing_poses)) == 2


def test_outlier_exclusion_records_what_it_removed(synthetic_problem):
    """Phase 3 reports D3.1 and D3.2 from these two attributes.  Never set,
    both getattr calls fell back to empty and every run reported that no
    poses were missing and no outliers were removed."""
    cams, target, detection, _ = synthetic_problem
    handler = TemplateBundleHandler(
        camset=cams, target=target, detection=detection, options={"outliers": "y"}
    )
    handler.missing_poses = np.zeros(N_MANY, dtype=bool)
    handler.missing_poses[0] = True  # already missing before rejection ran

    errors = _even_errors()
    errors[2] = 1000.0

    handler.find_and_exclude_transform_outliers(errors)

    before = handler.missing_poses_before_outlier_rejection
    after = handler.missing_poses_after_outlier_rejection
    assert list(np.where(before)[0]) == [0]
    assert list(np.where(after)[0]) == [0, 2]
    assert int(np.sum(after)) - int(np.sum(before)) == 1


# --------------------------------------------------------------------------
# The initial rig estimate, and what it says about an image it could not place
# --------------------------------------------------------------------------


def _cam_relative_poses(cams, poses):
    """``target_in_camera[cam, image]``: the target pose as each camera sees it."""
    return np.array([[cams[name].extrinsic @ pose for pose in poses]
                     for name in cams.get_names()])


def test_an_unplaced_image_comes_back_as_a_nan_pose(synthetic_problem):
    """calc_initial_params decides an image is unposed with isnan(pose[0, 0]).

    An image no camera could place must not come back as a finite pose, or
    that check never fires and the image enters the solve with a pose that is
    not a pose.
    """
    cams, target, detection, poses = synthetic_problem
    target_in_camera = _cam_relative_poses(cams, poses)
    target_in_camera[:, 1] = np.nan   # no camera placed image 1

    _, target_poses, per_im = estimate_initial_rig(
        target_in_camera, cams, 0, target, detection)

    unposed = np.array([np.isnan(t[0, 0]) for t in target_poses])
    assert list(np.where(unposed)[0]) == [1]
    assert np.isnan(per_im[1])
    assert np.all(np.isfinite(per_im[[0, 2]]))


def test_a_fully_observed_problem_leaves_every_pose_finite(synthetic_problem):
    cams, target, detection, poses = synthetic_problem

    _, target_poses, per_im = estimate_initial_rig(
        _cam_relative_poses(cams, poses), cams, 0, target, detection)

    assert np.all(np.isfinite(target_poses))
    assert np.all(np.isfinite(per_im))


def test_the_estimate_recovers_the_rig_it_was_generated_from(synthetic_problem):
    """Exact detections must give back the cameras that made them."""
    cams, target, detection, poses = synthetic_problem

    extrinsics, target_poses, per_im = estimate_initial_rig(
        _cam_relative_poses(cams, poses), cams, 0, target, detection)

    truth = np.array([cams[name].extrinsic for name in cams.get_names()])
    relative_est = np.array([e @ np.linalg.inv(extrinsics[0]) for e in extrinsics])
    relative_truth = np.array([t @ np.linalg.inv(truth[0]) for t in truth])

    assert np.allclose(relative_est, relative_truth, atol=1e-6)
    assert np.all(per_im < 1e-6)


def test_the_world_frame_is_anchored_on_the_reference_image(synthetic_problem):
    """The solve holds one target pose fixed, so the gauge has to match it."""
    cams, target, detection, poses = synthetic_problem

    for reference in range(N_IMAGES):
        _, target_poses, _ = estimate_initial_rig(
            _cam_relative_poses(cams, poses), cams, reference, target, detection)
        assert np.allclose(target_poses[reference], np.eye(4), atol=1e-6)


def test_per_image_reprojection_is_a_mean_not_a_sum():
    """A sum grows with how many points were detected, so a half detected
    image scored better than a fully detected one and MAD ranked on the
    spread of point counts rather than the spread of error."""
    # image 0: four points at 2 px.  image 1: one point, also at 2 px.
    detection_data = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0],
                               [0, 0, 2, 0, 0], [0, 0, 3, 0, 0],
                               [0, 1, 0, 0, 0]], dtype=float)
    costs = np.full(5, 2.0)

    per_image = per_image_reprojection(
        costs, detection_data, 2, np.ones(2, dtype=bool))

    assert np.allclose(per_image, [2.0, 2.0])


def test_per_image_reprojection_is_nan_where_there_is_nothing_to_average():
    detection_data = np.array([[0, 0, 0, 0, 0]], dtype=float)
    costs = np.array([2.0])

    unreachable = per_image_reprojection(
        costs, detection_data, 2, np.array([False, True]))
    assert np.isnan(unreachable[0])   # viable says no
    assert np.isnan(unreachable[1])   # viable says yes, but nothing detected


def test_per_image_reprojection_can_pick_out_one_camera():
    detection_data = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=float)
    costs = np.array([2.0, 8.0])
    viable = np.ones(1, dtype=bool)

    assert per_image_reprojection(costs, detection_data, 1, viable, 0)[0] == 2.0
    assert per_image_reprojection(costs, detection_data, 1, viable, 1)[0] == 8.0
    assert per_image_reprojection(costs, detection_data, 1, viable)[0] == 5.0


# --------------------------------------------------------------------------
# SelfBundleHandler, on the real corpus
# --------------------------------------------------------------------------


@pytest.fixture
def self_handler(charuco_problem):
    target, detections, cams = charuco_problem
    return SelfBundleHandler(
        camset=cams, target=target, detection=detections, options={"outliers": "n"}
    )


@pytest.mark.data
def test_self_handler_fixes_seven_gauge_freedoms(self_handler):
    """Three points pin the frame: 3 + 3 + 1 coordinates held fixed.

    Without that the target geometry could translate, rotate and scale freely
    and the normal equations would be singular.  The seven are held on top of
    the unseen points, which are fixed for a different reason.
    """
    fixed_from_gauge = 7
    unseen_coords = 3 * int((~self_handler.visible_feature_mask).sum())

    assert int((~self_handler.feat_unfixed).sum()) == unseen_coords + fixed_from_gauge


@pytest.mark.data
def test_the_gauge_is_fixed_on_points_that_were_seen(self_handler):
    """Holding a point no camera saw removes nothing: it was never free."""
    for index in self_handler.fixed_inds:
        assert self_handler.visible_feature_mask[index]


@pytest.mark.data
def test_self_handler_fixes_unseen_points(self_handler):
    """A point no camera saw cannot be solved for, so it must be held fixed."""
    unseen = ~self_handler.visible_feature_mask
    if not unseen.any():
        pytest.skip("every target point was detected in this corpus")

    for index in np.where(unseen)[0]:
        assert not self_handler.feat_unfixed[3 * index : 3 * index + 3].any()


@pytest.mark.data
def test_self_handler_parameter_groups(self_handler):
    groups = self_handler.parameter_groups()

    assert [g.name for g in groups] == ["intr", "extr", "pose", "point"]
    assert sum(g.n_free for g in groups) == self_handler.bundlePrimitive.bdpt_end
    # the free target points are the eliminated block
    assert groups[-1].name == "point"


@pytest.mark.data
def test_self_handler_bundle_inputs_have_the_right_shapes(self_handler):
    x = self_handler.get_initial_params()
    proj, extr, poses, points = self_handler.get_bundle_adjustment_inputs(x)

    n_cams = len(self_handler.cam_names)
    assert proj.shape == (n_cams, 9)
    assert extr.shape == (n_cams, 6)
    assert poses.shape[1] == 6
    assert points.shape[1] == 3


@pytest.mark.data
def test_self_handler_make_points_places_the_target_per_image(self_handler):
    """make_points applies each image's pose to the target geometry."""
    x = self_handler.get_initial_params()
    placed = self_handler.get_bundle_adjustment_inputs(x, make_points=True)

    n_points = int(np.prod(self_handler.point_data.shape[:-1]))
    assert placed.shape[0] == self_handler.detection.max_ims
    assert placed.shape[1:] == (n_points, 3)
    assert np.all(np.isfinite(placed))


@pytest.mark.data
def test_self_handler_get_camset_returns_the_rig(self_handler):
    x = self_handler.get_initial_params()
    cams = self_handler.get_camset(x)

    assert isinstance(cams, CameraSet)
    assert cams.get_names() == self_handler.cam_names


@pytest.mark.data
def test_self_handler_get_camset_can_return_poses(self_handler):
    x = self_handler.get_initial_params()
    cams, poses = self_handler.get_camset(x, return_pose=True)

    assert isinstance(cams, CameraSet)
    assert len(poses) == self_handler.detection.max_ims


@pytest.mark.data
def test_get_updated_target_returns_point_geometry(self_handler):
    """The self-calibration's output: the refined target, gauge corrected."""
    x = self_handler.get_initial_params()
    points = self_handler.get_updated_target(x)

    assert points.shape[-1] == 3
    assert np.all(np.isfinite(points))


@pytest.mark.data
def test_set_from_templated_camset_requires_a_templated_calibration(self_handler, charuco_problem):
    """Seeding from a self-calibration would put the wrong block layout in."""
    _, _, cams = charuco_problem
    cams.calibration_handler = None

    with pytest.raises(ValueError, match="not a templated adjustment"):
        self_handler.set_from_templated_camset(cams)


# --------------------------------------------------------------------------
# Seeding a self-calibration from a previous solve that fixed parameters
# --------------------------------------------------------------------------


def _solved_camset(synthetic_problem, fixed_params=None):
    """The synthetic problem as a camset carrying a finished templated solve.

    The parameters themselves are arbitrary -- the seeding only moves them
    around -- but their length is not: it is whatever that solve left free.
    """
    cams, target, detection, _ = synthetic_problem
    handler = TemplateBundleHandler(
        camset=cams, target=target, detection=detection,
        fixed_params=fixed_params, options={"outliers": "n"},
    )
    rng = np.random.default_rng(0)
    cams.calibration_handler = handler
    cams.calibration_params = rng.normal(size=handler.bundlePrimitive.pose_end)
    return cams, target, detection, handler


def test_set_from_templated_camset_reads_a_solve_that_fixed_parameters(
        synthetic_problem):
    """The previous vector omits what that solve fixed, so it cannot be copied.

    Fixing one camera's intrinsics leaves nine fewer parameters in the
    previous vector than this problem's camera blocks take, and copying it
    straight in either raises or shifts every later parameter by nine.
    """
    fixed = {"left": {"int": np.arange(9, dtype=float)}}
    cams, target, detection, prev = _solved_camset(synthetic_problem, fixed)

    handler = SelfBundleHandler(
        camset=cams, target=target, detection=detection, options={"outliers": "n"})
    assert len(cams.calibration_params) < handler.bundlePrimitive.pose_end

    handler.set_from_templated_camset(cams)

    assert len(handler.initial_params) == handler.bundlePrimitive.bdpt_end
    # the round trip through this problem's own layout returns the previous
    # solve's cameras and poses, in full
    intr, extr, poses, _ = handler.get_bundle_adjustment_inputs(
        handler.initial_params)
    assert np.allclose(intr, prev.bundlePrimitive.intr)
    assert np.allclose(extr, prev.bundlePrimitive.extr)
    assert np.allclose(poses, prev.bundlePrimitive.poses)


def test_set_from_templated_camset_carries_a_previously_fixed_value_over(
        synthetic_problem):
    """A parameter pinned there and free here starts from the value it was pinned at."""
    pinned = np.arange(9, dtype=float)
    cams, target, detection, _ = _solved_camset(
        synthetic_problem, {"left": {"int": pinned}})

    handler = SelfBundleHandler(
        camset=cams, target=target, detection=detection, options={"outliers": "n"})
    handler.set_from_templated_camset(cams)

    assert np.allclose(handler.bundlePrimitive.intr[0], pinned)
    assert handler.bundlePrimitive.intr_unfixed[0]


def test_set_from_templated_camset_keeps_this_solves_fixed_values(
        synthetic_problem):
    """What this solve fixes stays where fixed_params put it, not where the last solve left it."""
    cams, target, detection, _ = _solved_camset(
        synthetic_problem, {"left": {"int": np.arange(9, dtype=float)}})

    pinned_now = np.full(9, 7.0)
    handler = SelfBundleHandler(
        camset=cams, target=target, detection=detection,
        fixed_params={"left": {"int": pinned_now}}, options={"outliers": "n"},
    )
    handler.set_from_templated_camset(cams)

    assert not handler.bundlePrimitive.intr_unfixed[0]
    assert np.allclose(handler.bundlePrimitive.intr[0], pinned_now)
    intr, _, _, _ = handler.get_bundle_adjustment_inputs(handler.initial_params)
    assert np.allclose(intr[0], pinned_now)


def test_set_from_templated_camset_refuses_a_self_calibration_vector(
        synthetic_problem):
    """A self-calibration's vector carries a geometry block a templated read would misplace."""
    cams, target, detection, _ = _solved_camset(synthetic_problem)

    handler = SelfBundleHandler(
        camset=cams, target=target, detection=detection, options={"outliers": "n"})
    handler.set_from_templated_camset(cams)
    # a SelfBundleHandler is a TemplateBundleHandler, so the isinstance check
    # passes and only the block lengths give it away
    cams.calibration_handler = handler
    cams.calibration_params = handler.initial_params

    with pytest.raises(ValueError, match="self calibration"):
        SelfBundleHandler(
            camset=cams, target=target, detection=detection,
            options={"outliers": "n"},
        ).set_from_templated_camset(cams)


def test_set_from_templated_camset_needs_parameters_to_read(synthetic_problem):
    cams, target, detection, _ = _solved_camset(synthetic_problem)
    cams.calibration_params = None

    handler = SelfBundleHandler(
        camset=cams, target=target, detection=detection, options={"outliers": "n"})
    with pytest.raises(ValueError, match="no calibration parameters"):
        handler.set_from_templated_camset(cams)


def test_an_image_with_no_detections_does_not_make_the_problem_degenerate(
        synthetic_problem, caplog):
    """A frame nothing was seen in costs that frame, not the calibration.

    Its six pose parameters have no residual mentioning them, so leaving them
    free puts six all-zero columns in the jacobian and the degeneracy check
    refuses to solve at all.
    """
    cams, target, detection, poses = synthetic_problem
    blank = 1
    without = detection.delete_row(global_im_num=blank)

    full = TemplateBundleHandler(
        camset=cams, target=target, detection=detection, options={"outliers": "n"})
    with caplog.at_level(logging.WARNING):
        gapped = TemplateBundleHandler(
            camset=cams, target=target, detection=without, options={"outliers": "n"})

    free_full = np.asarray(full.bundlePrimitive.poses_unfixed)
    free_gapped = np.asarray(gapped.bundlePrimitive.poses_unfixed)

    assert not free_gapped[blank], "the pose nothing was seen in is held fixed"
    assert free_gapped.sum() == free_full.sum() - 1, "six fewer free parameters"
    assert "no detections" in caplog.text

    # and nothing else changed: the blank pose is the only one newly held
    assert np.flatnonzero(free_gapped != free_full).tolist() == [blank]


def test_the_per_image_initial_error_is_kept_not_just_consumed():
    """Phase 3 and phase 4 report this as the per-image initial reprojection,
    and the diagnostics tab builds its threshold and its remove-these-images
    control on it. Computed and dropped, all of them read an empty array and
    drew nothing.

    A real target, because estimating the poses this measures needs one.
    """
    from scipy.spatial.transform import Rotation

    from pyCamSet.calibration_targets.core.target_registry import build_target
    from pyCamSet.utils.general_utils import h_tform

    target = build_target({"type": "Ccube", "n_points": 6, "length": 10.0})
    cams = CameraSet(camera_dict={
        name: make_camera(name=name, translation=pos)
        for name, pos in (("c0", (0.0, 0.0, 120.0)),
                          ("c1", (40.0, 0.0, 120.0)))})

    n_ims = 4
    points = np.asarray(target.point_data).reshape(6, -1, 3)
    detection = TargetDetection(cam_names=cams.get_names())
    for im in range(n_ims):
        pose = make_4x4h_tform(
            Rotation.from_euler("xyz", [0.15 * im, -0.2, 0.1]).as_rotvec(),
            [0.0, 0.0, 0.0])
        for face in (0, 1):
            placed = h_tform(points[face], pose)
            keys = np.stack([np.full(len(placed), face),
                             np.arange(len(placed))], axis=-1)
            for name in cams.get_names():
                detection.add_detection(
                    name, im,
                    ImageDetection(keys=keys,
                                   image_points=cams[name].project_points(placed)))

    handler = TemplateBundleHandler(
        camset=cams, target=target, detection=detection, options={"outliers": "n"})
    handler.get_initial_params()

    per_im = np.asarray(getattr(handler, "initial_per_im_error", []), dtype=float)
    assert per_im.size == n_ims, "one error per image, not an empty array"
    assert np.all(np.isfinite(per_im)), "a finite error for every image"
