"""``FreePointBundleHandler``: bundle adjustment over free point geometry.

This is a third optimisation mode alongside the fixed-target
``TemplateBundleHandler`` and the self-calibrating ``SelfBundleHandler``.  It
composes ``projection() + extrinsic3D() + free_point()``, with no per-image
target pose at all: the points themselves are the parameters.

Nothing inside pyCamSet imports it, so it had no coverage and no caller to
catch interface drift -- but it is used by applications outside this
repository, which makes it a public surface that has to keep working.  When
``parameter_groups()`` was added for the Schur solver, there was nothing to
run it.  These tests give it that.

Everything is synthetic: cameras and points are built analytically, so the
whole file runs in about a second and needs no image corpus.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyCamSet import CameraSet
from pyCamSet.calibration_targets import ImageDetection, TargetDetection
from pyCamSet.optimisation.free_point_handler import (
    FreePointBundleHandler,
    FreePointPrimitive,
    FreePointTarget,
)
from pyCamSet.optimisation.numba_schur import ParamGroup, SchurSolver, spec_from_groups

from conftest import make_camera

N_POINTS = 6


def _flat_intrinsic(cam):
    """The 9-wide intrinsic row the primitive stores: fx, cx, fy, cy, k1..k5."""
    k = cam.intrinsic
    return np.concatenate([[k[0, 0], k[0, 2], k[1, 1], k[1, 2]], cam.distortion_coefs])


def _flat_extrinsic(cam):
    """The 6-wide extrinsic row: rodrigues rotation then translation."""
    from pyCamSet.utils.general_utils import ext_4x4_to_rod

    rot, trans = ext_4x4_to_rod(cam.extrinsic)
    return np.concatenate([rot, trans])


def _true_camera_params(cams):
    """The intrinsic block then the extrinsic block, as the loss expects them."""
    intr = np.concatenate([_flat_intrinsic(cams[n]) for n in cams.get_names()])
    extr = np.concatenate([_flat_extrinsic(cams[n]) for n in cams.get_names()])
    return np.concatenate([intr, extr])


def _world_points():
    """A small non-planar cloud, in the (1, n, 3) layout point_data wants."""
    return np.array(
        [
            [
                [0.000, 0.000, 1.00],
                [0.010, 0.000, 1.00],
                [0.000, 0.010, 1.00],
                [0.010, 0.010, 1.02],
                [-0.008, 0.006, 0.98],
                [0.006, -0.009, 1.05],
            ]
        ]
    )


@pytest.fixture
def free_point_problem():
    """A three camera rig imaging one cloud of free points, once each.

    The detections are the exact projections of the points, so the residual at
    the true parameters is zero and any drift in the kernels or the parameter
    packing shows up as a non-zero loss.
    """
    cams = CameraSet(
        camera_dict={
            name: make_camera(name, translation=offset)
            for name, offset in [
                ("left", (-0.05, 0.0, 0.0)),
                ("centre", (0.0, 0.0, 0.0)),
                ("right", (0.05, 0.0, 0.0)),
            ]
        }
    )
    points = _world_points()
    target = FreePointTarget(points)

    detection = TargetDetection(cam_names=cams.get_names())
    for name in cams.get_names():
        uv = cams[name].project_points(points.reshape(-1, 3))
        detection.add_detection(
            name, 0, ImageDetection(keys=np.arange(N_POINTS), image_points=uv)
        )

    return cams, target, detection, points


@pytest.fixture
def handler(free_point_problem):
    cams, target, detection, _ = free_point_problem
    return FreePointBundleHandler(cams, target, detection)


# --------------------------------------------------------------------------
# FreePointPrimitive: the parameter packing
# --------------------------------------------------------------------------


def test_primitive_index_layout():
    """Offsets must be cumulative: 9 per intrinsic, 6 per extrinsic, 1 per coord."""
    prim = FreePointPrimitive(
        bundle_points=np.zeros(3 * 4), extr=np.zeros((2, 6)), intr=np.zeros((2, 9))
    )

    assert prim.intr_end == 9 * 2
    assert prim.extr_end == 9 * 2 + 6 * 2
    assert prim.bdpt_end == 9 * 2 + 6 * 2 + 12


def test_primitive_defaults_everything_to_free():
    prim = FreePointPrimitive(
        bundle_points=np.zeros(9), extr=np.zeros((2, 6)), intr=np.zeros((2, 9))
    )

    assert prim.free_intr == 2
    assert prim.free_extr == 2
    assert prim.free_bdpt == 9


def test_primitive_honours_fixed_flags():
    """Fixing a camera must shrink the parameter vector, not just mask it."""
    prim = FreePointPrimitive(
        bundle_points=np.zeros(9),
        extr=np.zeros((3, 6)),
        intr=np.zeros((3, 9)),
        intr_unfixed=np.array([True, False, True]),
        extr_unfixed=np.array([True, True, False]),
        bundle_points_unfixed=np.array([True] * 6 + [False] * 3),
    )

    assert prim.free_intr == 2
    assert prim.free_extr == 2
    assert prim.free_bdpt == 6
    assert prim.bdpt_end == 9 * 2 + 6 * 2 + 6


def test_return_bundle_primitives_round_trips_the_parameters():
    """Packing then unpacking must give back what went in."""
    prim = FreePointPrimitive(
        bundle_points=np.zeros(6), extr=np.zeros((1, 6)), intr=np.zeros((1, 9))
    )
    intr = np.arange(9, dtype=float)
    extr = np.arange(9, 15, dtype=float)
    points = np.arange(15, 21, dtype=float)
    params = np.concatenate([intr, extr, points])

    got_intr, got_extr, got_points = prim.return_bundle_primitives(params)

    assert np.allclose(got_intr[0], intr)
    assert np.allclose(got_extr[0], extr)
    assert np.allclose(got_points, points.reshape(-1, 3))


def test_fixed_parameters_keep_their_preset_value():
    """A fixed camera's values come from the primitive, not from x."""
    intr = np.tile(np.arange(9, dtype=float), (2, 1))
    prim = FreePointPrimitive(
        bundle_points=np.zeros(3),
        extr=np.zeros((2, 6)),
        intr=intr.copy(),
        intr_unfixed=np.array([False, True]),
    )
    params = np.concatenate([np.full(9, 99.0), np.zeros(12), np.zeros(3)])

    got_intr, _, _ = prim.return_bundle_primitives(params)

    assert np.allclose(got_intr[0], np.arange(9))  # untouched
    assert np.allclose(got_intr[1], 99.0)  # taken from x


# --------------------------------------------------------------------------
# The target
# --------------------------------------------------------------------------


def test_free_point_target_holds_its_points():
    points = _world_points()
    target = FreePointTarget(points)

    assert np.allclose(target.point_data, points)
    assert target.original_points is not None
    assert np.allclose(target.original_points, points)


def test_free_point_target_cannot_detect_itself():
    """Free points come from elsewhere; there is nothing to find in an image."""
    target = FreePointTarget(_world_points())
    with pytest.raises(NotImplementedError):
        target.find_in_image(np.zeros((10, 10)))


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_handler_builds_a_free_point_primitive(handler):
    assert isinstance(handler.bundlePrimitive, FreePointPrimitive)
    assert handler.bundlePrimitive.bdpt_end == 9 * 3 + 6 * 3 + 3 * N_POINTS


def test_handler_composes_the_free_point_kernel(handler):
    """The op function must be the free point composition, not the template one."""
    assert "free_point" in str(handler.op_fun).lower() or handler.can_make_jac()
    assert handler.can_make_jac()


def test_handler_has_no_pose_block(handler):
    """The distinguishing feature: no per-image target pose is parameterised."""
    assert not hasattr(handler.bundlePrimitive, "pose_end")
    assert [g.name for g in handler.parameter_groups()] == ["intr", "extr", "point"]


# --------------------------------------------------------------------------
# parameter_groups: the Schur seam, added in 692a593 and never executed
# --------------------------------------------------------------------------


def test_parameter_groups_shapes(handler):
    groups = handler.parameter_groups()

    by_name = {g.name: g for g in groups}
    assert by_name["intr"].base.shape == (3, 9)
    assert by_name["extr"].base.shape == (3, 6)
    assert by_name["point"].base.shape == (N_POINTS, 3)
    for group in groups:
        assert isinstance(group, ParamGroup)


def test_parameter_groups_free_count_matches_the_parameter_vector(handler):
    """The groups must describe exactly the vector the loss is handed."""
    groups = handler.parameter_groups()
    assert sum(g.n_free for g in groups) == handler.bundlePrimitive.bdpt_end


def test_parameter_groups_puts_the_eliminated_block_last(handler):
    """The Schur solver eliminates the final group, so it must be the points.

    The points are the block-diagonal, per-index parameters here; eliminating
    the cameras instead would be wrong and much slower.
    """
    groups = handler.parameter_groups()
    assert groups[-1].name == "point"

    spec = spec_from_groups(groups)
    assert spec.n_elim_blocks == groups[-1].n_rows
    assert spec.elim_size == groups[-1].n_par == 3


def test_parameter_group_indices_address_the_detections(handler):
    """Each group's index must map detections onto its own rows."""
    groups = {g.name: g for g in handler.parameter_groups()}
    n_det = len(handler.detection.get_data())

    for name, expected_rows in [("intr", 3), ("extr", 3), ("point", N_POINTS)]:
        index = groups[name].index
        assert index.shape == (n_det,)
        assert index.min() >= 0
        assert index.max() < expected_rows


def test_parameter_groups_respect_fixed_cameras(free_point_problem):
    """A fixed intrinsic must drop out of the group's free count.

    Note the format: fixed_params[cam]["int"] is written straight into the
    primitive's 9-wide intrinsic row, so it is the flat
    (fx, cx, fy, cy, k1..k5) vector rather than the 3x3 matrix.
    """
    cams, target, detection, _ = free_point_problem
    fixed = {"left": {"int": _flat_intrinsic(cams["left"])}}
    handler = FreePointBundleHandler(cams, target, detection, fixed_params=fixed)

    by_name = {g.name: g for g in handler.parameter_groups()}
    assert by_name["intr"].n_free == 9 * 2  # one of three cameras fixed
    assert sum(g.n_free for g in handler.parameter_groups()) == handler.bundlePrimitive.bdpt_end


def test_the_schur_solver_accepts_these_groups(handler):
    """The groups must actually drive a solve, not merely have valid shapes."""
    groups = handler.parameter_groups()
    spec = spec_from_groups(groups)
    solver = SchurSolver(spec, threads=1)

    rng = np.random.default_rng(0)
    n_det = len(handler.detection.get_data())
    row_width = sum(g.n_par for g in groups)
    blocks = rng.normal(size=(n_det, 2, row_width))
    resid = rng.normal(size=(n_det, 2))

    step = solver.step(blocks, resid, 1e-3)

    assert step.shape == (sum(g.n_free for g in groups),)
    assert np.all(np.isfinite(step))


# --------------------------------------------------------------------------
# Parameter vector plumbing
# --------------------------------------------------------------------------


def test_get_bundle_adjustment_inputs_shapes(handler):
    x = np.zeros(handler.bundlePrimitive.bdpt_end)
    proj, extr, points = handler.get_bundle_adjustment_inputs(x)

    assert proj.shape == (3, 9)
    assert extr.shape == (3, 6)
    assert points.shape == (N_POINTS, 3)


def test_get_updated_points_reads_the_geometry_back(handler):
    """Regression: this was ``def get_updated_points():`` -- no self, free x.

    It raised TypeError before it could even reach the undefined name, so the
    only way to read the solved geometry was unusable.
    """
    prim = handler.bundlePrimitive
    wanted = np.arange(3 * N_POINTS, dtype=float)
    x = np.concatenate([np.zeros(prim.extr_end), wanted])

    points = handler.get_updated_points(x)

    assert points.shape == (N_POINTS, 3)
    assert np.allclose(points, wanted.reshape(-1, 3))


def test_set_initial_params_is_returned_by_get_initial_params(handler):
    x = np.arange(handler.bundlePrimitive.bdpt_end, dtype=float)
    handler.set_initial_params(x)
    assert np.allclose(handler.get_initial_params(), x)


def test_set_from_camset_packs_cameras_then_points(free_point_problem):
    """Regression: the slices were bounded by bdpt_end, the whole length.

    That consumed every slot for the camera parameters and left the points an
    empty slice, so this raised ValueError on any real input.  Cameras belong
    in [:extr_end] and the points in the remainder.
    """
    cams, target, detection, points = free_point_problem
    handler = FreePointBundleHandler(cams, target, detection)
    prim = handler.bundlePrimitive

    camera_params = np.arange(prim.extr_end, dtype=float)
    cams.calibration_params = camera_params

    handler.set_from_camset(cams, points)

    assert handler.initial_params.shape == (prim.bdpt_end,)
    assert np.allclose(handler.initial_params[: prim.extr_end], camera_params)
    assert np.allclose(handler.initial_params[prim.extr_end :], points.flatten())


# --------------------------------------------------------------------------
# get_camset
# --------------------------------------------------------------------------


def test_get_camset_rebuilds_the_cameras(handler, free_point_problem):
    """Round trip: pack the true cameras into x, and get them back out."""
    cams, _, _, _ = free_point_problem
    prim = handler.bundlePrimitive

    x = np.concatenate(
        [_true_camera_params(cams), np.zeros(prim.bdpt_end - prim.extr_end)]
    )
    rebuilt = handler.get_camset(x)

    assert isinstance(rebuilt, CameraSet)
    assert rebuilt.get_names() == cams.get_names()
    for name in cams.get_names():
        assert np.allclose(rebuilt[name].intrinsic, cams[name].intrinsic)
        assert np.allclose(rebuilt[name].extrinsic, cams[name].extrinsic)


def test_get_camset_accepts_return_pose_but_refuses_it(handler):
    """Regression: the base handler takes return_pose and this did not.

    A polymorphic caller got TypeError.  The parameter is accepted now, but a
    free point problem has no per-image target pose, so asking for one is an
    explicit error rather than point geometry handed back under the wrong name.
    """
    x = np.zeros(handler.bundlePrimitive.bdpt_end)

    with pytest.raises(NotImplementedError, match="get_updated_points"):
        handler.get_camset(x, return_pose=True)


# --------------------------------------------------------------------------
# The loss
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_the_loss_is_zero_at_the_true_parameters(handler, free_point_problem):
    """The detections are exact projections, so the residual must vanish.

    This is the end-to-end check that the parameter packing, the composed
    kernel and the detection ordering all agree.
    """
    cams, _, _, points = free_point_problem
    prim = handler.bundlePrimitive

    x = np.concatenate([_true_camera_params(cams), points.flatten()])
    assert x.size == prim.bdpt_end

    loss = handler.make_loss_fun(threads=1)
    residual = loss(x)

    assert np.all(np.isfinite(residual))
    assert np.max(np.abs(residual)) < 1e-6
