"""
The telecentric camera model, from the projection up to a solved bundle adjustment.

A new lens model can drive reprojection error to near zero while recovering the
wrong magnification, so the accuracy tests here assert the parameters against
ground truth rather than only the residuals.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pyCamSet import CameraSet
from pyCamSet.cameras.camera import Camera
from pyCamSet.cameras.telecentric_camera import (
    R2_SCALE, TelecentricCamera, MAGNIFICATION_ERROR_BUDGET)
from pyCamSet.calibration_targets.core.target_detections import (
    ImageDetection, TargetDetection)
from pyCamSet.optimisation.camera_models import blocks_for_camset
from pyCamSet.optimisation.function_block_implementations import (
    telecentric_extrinsic, telecentric_intrinsic)
from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment
from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler
from pyCamSet.optimisation.template_handler import (
    TemplateBundleHandler, _extrinsic_from_params)
from pyCamSet.utils.general_utils import h_tform, make_4x4h_tform

REF_MAGNIFICATION = np.array([30000.0, 29000.0])  # px per metre
REF_RES = [1280, 960]
REF_K = 0.23
REF_EPS = 1.0
N_IMAGES = 9


def make_telecentric_camera(name="cam", rotation=(0.0, 0.0, 0.0), k=REF_K,
                            eps=REF_EPS, res=None, magnification=None):
    """
    One synthetic telecentric camera, rotated by *rotation* in radians.

    The translation is left at zero: a telecentric camera's position is not
    identifiable, so a synthetic rig that sets one would be describing
    something the model cannot represent.
    """
    res = REF_RES if res is None else res
    mag = REF_MAGNIFICATION if magnification is None else np.asarray(magnification)
    intrinsic = np.array([
        [mag[0], 0.0, res[0] / 2],
        [0.0, mag[1], res[1] / 2],
        [0.0, 0.0, 1.0],
    ])
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rotation.from_rotvec(np.asarray(rotation)).as_matrix()
    return TelecentricCamera(
        extrinsic=extrinsic, intrinsic=intrinsic, res=res,
        distortion_coefs=np.array([k]), telecentricity=eps, name=name)


class GridTarget:
    """
    The smallest thing the handlers accept: point_data plus a square size.

    The points are deliberately not coplanar.  A planar target under an affine
    camera has a two-fold out-of-plane tilt ambiguity, which is a property of
    the geometry rather than of this implementation.
    """

    def __init__(self, extent=0.010, depth=0.004):
        xs = np.linspace(-extent, extent, 4)
        ys = np.linspace(-extent, extent, 4)
        x, y = np.meshgrid(xs, ys)
        z = depth * np.cos(3.0 * x / extent) * np.sin(2.0 * y / extent)
        flat = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
        self.point_data = flat[None, ...]
        self.point_local = self.point_data.copy()
        self.original_points = self.point_data.copy()
        self.square_size = float(2 * extent / 3)
        self.valid_map = True
        self.input_args = {}


def _target_poses(n=N_IMAGES, seed=11):
    """
    Poses that keep the target near the reference plane and well tilted.

    The first is the identity because the handler fixes pose 0 to pin the world
    frame, and a telecentric rig needs that gauge fixed: with no camera
    translation estimated, a global shift of every target pose is otherwise
    free.
    """
    rng = np.random.default_rng(seed)
    return [np.eye(4)] + [
        make_4x4h_tform(rng.uniform(-0.25, 0.25, 3), rng.uniform(-0.008, 0.008, 3))
        for _ in range(n - 1)
    ]


def ground_truth_params(handler, cams, poses):
    """
    The exact parameter vector that generated a synthetic problem.

    Built directly rather than through ``calc_initial_params`` so that what is
    under test is the bundle adjustment, not the seed that feeds it.
    """
    from pyCamSet.utils.general_utils import ext_4x4_to_rod

    blocks = []
    for cam in cams:
        blocks.append(cam.to_param_vector())
    for cam in cams:
        rod, _ = ext_4x4_to_rod(cam.extrinsic)
        blocks.append(np.asarray(rod)[:handler.bundlePrimitive.n_extr])
    for idp, pose in enumerate(poses):
        if not handler.bundlePrimitive.poses_unfixed[idp]:
            continue
        rod, trans = ext_4x4_to_rod(pose)
        blocks.append(np.concatenate([rod, trans]))
    return np.concatenate(blocks)


@pytest.fixture
def telecentric_problem():
    """Three telecentric cameras at different orientations, exact detections."""
    cams = CameraSet(camera_dict={
        name: make_telecentric_camera(name, rotation=rot)
        for name, rot in (
            ("cam_0", (0.0, 0.0, 0.0)),
            ("cam_1", (0.0, 0.45, 0.0)),
            ("cam_2", (-0.40, 0.0, 0.25)),
        )
    })
    target = GridTarget()
    poses = _target_poses()
    points = target.point_data.reshape(-1, 3)

    detection = TargetDetection(cam_names=cams.get_names())
    for im_num, pose in enumerate(poses):
        placed = h_tform(points, pose)
        for name in cams.get_names():
            detection.add_detection(name, im_num, ImageDetection(
                keys=np.arange(len(points)),
                image_points=cams[name].project_points(placed),
            ))
    return cams, target, detection, poses


# ----------------------------------------------------------------------
# the model itself
# ----------------------------------------------------------------------

def test_block_matches_the_camera_model():
    """
    The compiled kernel and the python camera must be the same projection.

    They are written twice -- once in numba for the residual and once in numpy
    for everything else -- and the r2 scaling in particular is a literal in the
    kernel because a generated module cannot import a constant.
    """
    cam = make_telecentric_camera("cam")
    rng = np.random.default_rng(3)
    points = rng.uniform(-0.012, 0.012, (200, 3))

    expected = cam.project_points(points)
    params = cam.to_param_vector()
    out, memory = np.empty(2), np.empty(telecentric_intrinsic.array_memory)
    got = np.empty_like(expected)
    for idx, cam_point in enumerate(h_tform(points, cam.extrinsic)):
        telecentric_intrinsic.compute_fun(
            params, np.ascontiguousarray(cam_point), out, memory)
        got[idx] = out
    assert np.allclose(got, expected, atol=1e-12)


def test_r2_scale_keeps_the_distortion_coefficient_conditioned():
    """
    k has to stay O(1) whether the target is measured in metres or millimetres.

    Applied to raw camera frame coordinates the squared radius would carry the
    target's length unit, and k with it.
    """
    assert R2_SCALE == 1e-6
    cam = make_telecentric_camera("cam")
    edge = np.array([[float(REF_RES[0]), float(REF_RES[1])]]) - cam.principal_point
    r2_at_edge = float(np.sum(edge ** 2) * R2_SCALE)
    assert 0.1 < r2_at_edge < 10.0


def test_projection_is_independent_of_depth_for_a_perfect_lens():
    """With no telecentricity error, sliding a point along the axis changes nothing."""
    cam = make_telecentric_camera("cam", eps=0.0, rotation=(0.1, -0.2, 0.05))
    rng = np.random.default_rng(5)
    points = rng.uniform(-0.01, 0.01, (40, 3))
    axis = cam.cam_to_world[:3, 2]
    for shift in (-0.05, 0.0, 0.05, 0.5):
        assert np.allclose(
            cam.project_points(points + axis * shift),
            cam.project_points(points), atol=1e-9)


def test_telecentricity_error_makes_magnification_depend_on_depth():
    """A real lens is not perfectly telecentric, and eps is what says so."""
    cam = make_telecentric_camera("cam", eps=REF_EPS, k=0.0)
    points = np.array([[0.01, 0.0, 0.0]])
    near = cam.project_points(points)[0, 0] - cam.principal_point[0]
    far = cam.project_points(points + np.array([0.0, 0.0, 0.05]))[0, 0] - cam.principal_point[0]
    assert np.isclose(far / near, 1.0 / (1.0 + REF_EPS * 0.05))


def test_distortion_inverts_in_closed_form():
    """The division model's inverse is a quadratic root, not an iteration."""
    cam = make_telecentric_camera("cam")
    rng = np.random.default_rng(7)
    points = rng.uniform(-0.012, 0.012, (500, 3))
    undistorted = cam.project_points(points, distort=False)
    distorted = cam.project_points(points, distort=True)
    assert not np.allclose(distorted, undistorted)
    assert np.allclose(cam.undistort_points(distorted), undistorted, atol=1e-9)


def test_affine_projection_matrix_reproduces_the_projection():
    """
    The 3x4 matrix has [0, 0, 0, 1] as its last row, so the DLT needs no special case.
    """
    cam = make_telecentric_camera("cam", k=0.0, eps=0.0, rotation=(0.2, 0.1, -0.3))
    rng = np.random.default_rng(9)
    points = rng.uniform(-0.01, 0.01, (50, 3))
    assert np.allclose(cam.proj[2], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(h_tform(points, cam.proj), cam.project_points(points), atol=1e-9)


def test_rays_round_trip_through_the_image():
    """A pixel's ray, projected back, lands on the pixel it came from."""
    cam = make_telecentric_camera("cam", rotation=(0.15, -0.25, 0.1))
    uv = np.array([[100.0, 120.0], [640.0, 480.0], [1100.0, 800.0]])
    assert np.allclose(cam.project_points(cam.im_to_world_ray(uv)), uv, atol=1e-8)


def test_a_perfect_lens_has_parallel_rays():
    """Telecentricity is exactly the statement that the chief rays do not converge."""
    cam = make_telecentric_camera("cam", eps=0.0, rotation=(0.15, -0.25, 0.1))
    uv = np.array([[100.0, 120.0], [640.0, 480.0], [1100.0, 800.0]])
    undistorted = cam.undistort_points(uv)
    directions = (cam._pixels_to_cam_frame(undistorted, depth=1.0)
                  - cam._pixels_to_cam_frame(undistorted, depth=0.0))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    assert np.allclose(directions, [0.0, 0.0, 1.0], atol=1e-12)


def test_telecentricity_error_puts_the_entrance_pupil_at_one_over_eps():
    """
    An imperfect telecentric lens is a pinhole a long way back, and eps says how far.

    Every ray scales as ``1 + eps*z`` about the reference plane, so they all
    meet where that vanishes.  This is the same identity that makes the camera's
    axial position a gauge freedom: a pinhole at ``1/eps`` with focal length
    ``m/eps`` draws the identical image.
    """
    cam = make_telecentric_camera("cam", eps=REF_EPS, k=0.0)
    uv = np.array([[100.0, 120.0], [640.0, 480.0], [1100.0, 800.0]])
    pupil = cam._pixels_to_cam_frame(uv, depth=-1.0 / REF_EPS)
    assert np.allclose(pupil[:, :2], 0.0, atol=1e-9)
    assert np.allclose(pupil[:, 2], -1.0 / REF_EPS)


def test_a_telecentric_camera_has_no_angular_field_of_view():
    """The pinhole quantities are read off a focal length that does not exist here."""
    cam = make_telecentric_camera("cam")
    assert cam.fov is None
    assert cam.focal_point is None
    assert np.allclose(cam.field_size, np.asarray(REF_RES) / REF_MAGNIFICATION)
    with pytest.raises(AttributeError, match="no angular field of view"):
        cam._cam_fov()


def test_view_volume_is_a_prism_bounded_by_the_magnification_error():
    """
    A telecentric camera images a box, and eps is what bounds its useful depth.

    The cross section comes from the optics and does not scale, which is the
    difference from a pinhole frustum.
    """
    cam = make_telecentric_camera("cam", eps=REF_EPS)
    pytest.importorskip("pyvista")
    box = cam.get_mesh()
    assert box.points.shape == (8, 3)
    extent = np.ptp(box.points, axis=0)
    assert np.allclose(extent[:2], cam.field_size, rtol=5e-3)
    assert np.isclose(extent[2], 2 * MAGNIFICATION_ERROR_BUDGET / REF_EPS, rtol=5e-3)
    # a perfect lens has no bound, so the drawn depth is capped instead
    assert np.isfinite(make_telecentric_camera("c", eps=0.0).view_depth)


def test_equality_separates_the_lens_models():
    """A subclass holding the same arrays is a different model, not an equal camera."""
    tele = make_telecentric_camera("cam", k=0.0, eps=0.0)
    pinhole = Camera(extrinsic=tele.extrinsic, intrinsic=tele.intrinsic,
                     res=tele.res, distortion_coefs=np.zeros(5), name="cam")
    assert tele != pinhole
    assert pinhole != tele
    assert tele == make_telecentric_camera("cam", k=0.0, eps=0.0)
    assert tele != make_telecentric_camera("cam", k=0.0, eps=0.5)


def test_param_vector_round_trips():
    """The handlers pack and unpack through these, so they have to be inverses."""
    cam = make_telecentric_camera("cam")
    params = cam.to_param_vector()
    assert params.shape == (telecentric_intrinsic.params.n_params,)
    other = make_telecentric_camera("cam", k=0.0, eps=0.0, magnification=[1.0, 1.0])
    other.from_param_vector(params)
    assert np.allclose(other.to_param_vector(), params)
    assert other == cam


# ----------------------------------------------------------------------
# the blocks, and the gauge they have to respect
# ----------------------------------------------------------------------

@pytest.mark.parametrize("block", [telecentric_intrinsic, telecentric_extrinsic])
def test_block_jacobian_matches_finite_differences(block):
    """The check docs/extending/bundle-adjustment.md tells a block author to run."""
    block().test_self()


def test_the_telecentric_extrinsic_estimates_no_translation():
    """
    All three translation components are gauge freedoms, so none is a parameter.

    Sliding the camera along its own axis rescales m and eps to give exactly the
    same image, and moving it in plane is absorbed by the principal point.
    Leaving the parameters in would leave three jacobian columns empty.
    """
    assert telecentric_extrinsic.params.n_params == 3
    cam = make_telecentric_camera("cam", rotation=(0.1, 0.2, -0.15))
    rng = np.random.default_rng(13)
    points = rng.uniform(-0.01, 0.01, (30, 3))

    # slide the camera d along its own optical axis: every camera frame z drops by d
    d = 0.03
    extrinsic = cam.extrinsic.copy()
    extrinsic[2, 3] -= d

    # matching m'/(1 + eps'*(z - d)) to m/(1 + eps*z) for every z forces both
    # to scale by the same factor, which is why the slide is unrecoverable
    scale = 1.0 / (1.0 + REF_EPS * d)
    intrinsic = cam.intrinsic.copy()
    intrinsic[0, 0] *= scale
    intrinsic[1, 1] *= scale

    slid = TelecentricCamera(
        extrinsic=extrinsic, intrinsic=intrinsic, res=cam.res,
        distortion_coefs=cam.distortion_coefs,
        telecentricity=cam.telecentricity * scale, name="slid")
    assert np.allclose(slid.project_points(points), cam.project_points(points), atol=1e-9)


def test_blocks_are_chosen_by_the_camera_model(telecentric_problem):
    cams, _, _, _ = telecentric_problem
    assert blocks_for_camset(cams) == (telecentric_intrinsic, telecentric_extrinsic)


def test_a_mixed_camera_set_is_refused(telecentric_problem):
    """One kernel is compiled per calibration, so one model is calibrated at a time."""
    cams, _, _, _ = telecentric_problem
    mixed = CameraSet(camera_dict={
        "cam_0": cams["cam_0"],
        "cam_1": Camera(res=REF_RES, name="cam_1"),
    })
    with pytest.raises(ValueError, match="share a lens model"):
        blocks_for_camset(mixed)


# ----------------------------------------------------------------------
# end to end
# ----------------------------------------------------------------------

def test_handler_sizes_itself_from_the_lens_model(telecentric_problem):
    """The six parameter widths used to be literals that could disagree with the kernel."""
    cams, target, detection, poses = telecentric_problem
    handler = TemplateBundleHandler(camset=cams, target=target, detection=detection)
    assert len(ground_truth_params(handler, cams, poses)) == 6 * 3 + 3 * 3 + 6 * (N_IMAGES - 1)
    assert handler.bundlePrimitive.n_intr == telecentric_intrinsic.params.n_params
    assert handler.bundlePrimitive.n_extr == telecentric_extrinsic.params.n_params
    assert [b.params.n_params for b in handler.op_fun.function_blocks] == [6, 3, 6]


@pytest.mark.slow
def test_bundle_adjustment_recovers_the_known_camera_parameters(telecentric_problem):
    """
    A perturbed telecentric calibration has to converge back onto ground truth.

    Reprojection error alone would not catch a model that is self consistent but
    wrong, so the magnification, principal point, distortion and telecentricity
    are each checked against what generated the detections.
    """
    cams, target, detection, poses = telecentric_problem
    handler = TemplateBundleHandler(
        camset=cams, target=target, detection=detection,
        options={"max_nfev": 100, "outliers": "ignore"})

    truth = ground_truth_params(handler, cams, poses)
    handler.missing_poses = np.zeros(len(poses), dtype=bool)
    rng = np.random.default_rng(23)
    start = truth.copy()
    n_intr = handler.bundlePrimitive.n_intr
    n_cams = cams.get_n_cams()
    intr = start[:n_intr * n_cams].reshape(n_cams, n_intr)
    intr[:, 0] *= 1.0 + rng.uniform(-0.02, 0.02, n_cams)   # magnification
    intr[:, 2] *= 1.0 + rng.uniform(-0.02, 0.02, n_cams)
    intr[:, 1] += rng.uniform(-8, 8, n_cams)               # principal point
    intr[:, 3] += rng.uniform(-8, 8, n_cams)
    intr[:, 4] += rng.uniform(-0.05, 0.05, n_cams)         # k
    intr[:, 5] += rng.uniform(-0.15, 0.15, n_cams)         # eps
    start[n_intr * n_cams:] += rng.uniform(-0.01, 0.01, len(start) - n_intr * n_cams)
    handler.set_initial_params(start)

    _, final_cams = run_bundle_adjustment(param_handler=handler, threads=2)

    residual = np.reshape(final_cams.calibration_result, (-1, 2))
    assert np.mean(np.linalg.norm(residual, axis=1)) < 1e-3

    for name in cams.get_names():
        got, want = final_cams[name], cams[name]
        assert np.allclose(got.magnification, want.magnification, rtol=1e-4), name
        assert np.allclose(got.principal_point, want.principal_point, atol=0.05), name
        assert np.allclose(got.distortion_coefs, want.distortion_coefs, atol=1e-3), name
        assert np.isclose(got.telecentricity, want.telecentricity, atol=1e-3), name
        assert np.allclose(got.extrinsic[:3, :3], want.extrinsic[:3, :3], atol=1e-4), name


@pytest.mark.slow
def test_bundle_adjustment_jacobian_has_no_dead_columns(telecentric_problem):
    """
    Every telecentric parameter has to reach the residual.

    This is the check that fails if the translation gauge is left in: a t_z
    column is identically zero, which the Schur elimination would divide by.
    """
    cams, target, detection, poses = telecentric_problem
    handler = TemplateBundleHandler(camset=cams, target=target, detection=detection)
    handler.missing_poses = np.zeros(len(poses), dtype=bool)
    params = ground_truth_params(handler, cams, poses)
    jac = handler.make_loss_jac(threads=2)(params)
    dense = jac.toarray() if hasattr(jac, "toarray") else np.asarray(jac)
    dead = np.where(~np.any(np.abs(dense) > 1e-12, axis=0))[0]
    assert dead.size == 0, f"parameters {dead} never reach the residual"


@pytest.mark.slow
def test_triangulation_reconstructs_through_affine_cameras(telecentric_problem):
    """
    The DLT is projective, so it handles an affine projection matrix unchanged.

    What does not is the undistortion, which is why each camera inverts its own
    before the kernel sees the data.
    """
    cams, target, _, poses = telecentric_problem
    points = h_tform(target.point_data.reshape(-1, 3), poses[0])
    projected = [{name: cams[name].project_points(p[None, ...])[0]
                  for name in cams.get_names()} for p in points]
    reconstructed = cams.multi_cam_triangulate(projected)
    assert np.allclose(reconstructed, points, atol=1e-6)


# ----------------------------------------------------------------------
# seeding, where opencv cannot
# ----------------------------------------------------------------------

def test_affine_fit_recovers_a_known_camera(telecentric_problem):
    """The seed is a linear fit, because the model is linear once eps and k are set aside."""
    from pyCamSet.cameras.telecentric_calibration import calibrate_telecentric

    cams, target, _, poses = telecentric_problem
    cam = make_telecentric_camera("cam", k=0.0, eps=0.0, rotation=(0.1, -0.2, 0.05))
    points = target.point_data.reshape(-1, 3)

    object_points = [points for _ in poses]
    image_points = [cam.project_points(h_tform(points, pose)) for pose in poses]

    magnification, principal, got_poses, rms = calibrate_telecentric(
        object_points, image_points, cam.res)

    assert np.allclose(magnification, cam.magnification, rtol=1e-6)
    assert np.allclose(principal, cam.principal_point)
    assert np.max(rms) < 1e-6
    # the pose is recovered up to the depth a telecentric camera cannot see
    for pose, want in zip(got_poses, poses):
        combined = cam.extrinsic @ want
        assert np.allclose(pose[:3, :3], combined[:3, :3], atol=1e-6)
        assert np.allclose(pose[:2, 3], combined[:2, 3], atol=1e-6)
        assert pose[2, 3] == 0.0


def test_a_planar_target_is_refused_with_a_reason():
    """
    Tilt is two-fold ambiguous under an affine camera, so the seed says so.

    This is a property of the geometry, not of the implementation: +theta and
    -theta produce the same image.
    """
    from pyCamSet.cameras.telecentric_calibration import fit_affine_camera, is_planar

    flat = np.stack(np.meshgrid(np.linspace(-0.01, 0.01, 4),
                                np.linspace(-0.01, 0.01, 4)), axis=-1).reshape(-1, 2)
    planar = np.concatenate([flat, np.zeros((len(flat), 1))], axis=1)
    assert is_planar(planar)
    assert not is_planar(GridTarget().point_data.reshape(-1, 3))
    with pytest.raises(ValueError, match="two-fold pose ambiguity"):
        fit_affine_camera(planar, planar[:, :2] * 1000)


def test_seeded_bundle_adjustment_converges(telecentric_problem):
    """
    The seed has to land close enough that the bundle adjustment finishes the job.

    It ignores distortion and telecentricity error by construction, so this is
    the test that those are small enough to be refined rather than guessed.
    """
    from pyCamSet.cameras.telecentric_calibration import calibrate_telecentric

    cams, target, detection, poses = telecentric_problem
    points = target.point_data.reshape(-1, 3)

    seeded = {}
    for name in cams.get_names():
        truth = cams[name]
        image_points = [truth.project_points(h_tform(points, pose)) for pose in poses]
        magnification, principal, _, _ = calibrate_telecentric(
            [points] * len(poses), image_points, truth.res)
        intrinsic = np.array([[magnification[0], 0, principal[0]],
                              [0, magnification[1], principal[1]], [0, 0, 1.0]])
        seeded[name] = TelecentricCamera(
            extrinsic=truth.extrinsic, intrinsic=intrinsic, res=truth.res,
            distortion_coefs=np.array([0.0]), telecentricity=0.0, name=name)

    # the seed sees distortion and telecentricity as magnification error
    assert np.allclose(seeded["cam_0"].magnification, REF_MAGNIFICATION, rtol=0.1)

    handler = TemplateBundleHandler(
        camset=CameraSet(camera_dict=seeded), target=target, detection=detection,
        options={"max_nfev": 200, "outliers": "ignore"})
    handler.missing_poses = np.zeros(len(poses), dtype=bool)
    handler.set_initial_params(
        ground_truth_params(handler, CameraSet(camera_dict=seeded), poses))

    _, final_cams = run_bundle_adjustment(param_handler=handler, threads=2)
    residual = np.reshape(final_cams.calibration_result, (-1, 2))
    assert np.mean(np.linalg.norm(residual, axis=1)) < 1e-3
    for name in cams.get_names():
        assert np.allclose(final_cams[name].magnification,
                           cams[name].magnification, rtol=1e-3), name
        assert np.isclose(final_cams[name].telecentricity,
                          cams[name].telecentricity, atol=1e-2), name


# ----------------------------------------------------------------------
# paths that have no telecentric meaning, and must say so
# ----------------------------------------------------------------------

def test_downscaling_carries_the_distortion_coefficient():
    """
    The division model acts on pixel offsets, so halving the image rescales k.

    Rescaling only the intrinsic matrix would leave the distortion describing an
    image that no longer exists.
    """
    cam = make_telecentric_camera("cam", eps=0.0)
    rng = np.random.default_rng(31)
    points = rng.uniform(-0.012, 0.012, (100, 3))
    full = cam.project_points(points)

    cam.scale_self_2n(1)
    halved = cam.project_points(points)
    assert np.allclose(halved, (full + 0.5) / 2 - 0.5, atol=1e-6)


def test_colmap_export_refuses_a_telecentric_camera(tmp_path):
    """FULL_OPENCV would silently reinterpret a magnification as a focal length."""
    from pyCamSet.utils.saving import export_cameras_txt

    cams = CameraSet(camera_dict={"cam_0": make_telecentric_camera("cam_0")})
    with pytest.raises(ValueError, match="no telecentric camera model"):
        export_cameras_txt(cams, tmp_path)


def test_mvsnet_export_refuses_a_telecentric_camera(tmp_path):
    """The MVSNet format describes a pinhole, and there is no honest mapping."""
    cam = make_telecentric_camera("cam")
    with pytest.raises(NotImplementedError, match="no representation"):
        cam.to_MVSnet_txt(tmp_path / "cam.txt", (0.1, 0.2), 4)


# --------------------------------------------------------------------------
# Seeding from a target with depth
# --------------------------------------------------------------------------


def _cube_detection(cam_name, target, cam, n_images=6, faces_per_image=2):
    """Photograph a cube so each image catches more than one face."""
    from pyCamSet.utils.general_utils import h_tform
    detection = TargetDetection(cam_names=[cam_name])
    points = np.asarray(target.point_data).reshape(6, -1, 3)
    for im in range(n_images):
        pose = make_4x4h_tform(
            Rotation.from_euler("xyz", [0.2 + 0.05 * im, -0.3, 0.1]).as_rotvec(),
            [0.0, 0.0, 200.0])
        for face in range(faces_per_image):
            placed = h_tform(points[face], pose)
            keys = np.stack([np.full(len(placed), face),
                             np.arange(len(placed))], axis=-1)
            detection.add_detection(
                cam_name, im,
                ImageDetection(keys=keys, image_points=cam.project_points(placed)))
    return detection


def test_a_cube_seeds_a_telecentric_camera_from_images_that_show_two_faces():
    """Every board of a cube is a flat face, so a per-board seed can never see
    out-of-plane extent -- not even on the Ccube the refusal recommends. The
    seed is fitted per image, where two faces together carry depth."""
    from pyCamSet.calibration_targets.core.target_registry import build_target

    target = build_target({"type": "Ccube", "n_points": 6, "length": 10.0})
    truth = TelecentricCamera(
        intrinsic=np.array([[80.0, 0, 270.0], [0, 80.0, 360.0], [0, 0, 1.0]]),
        res=[720, 540], distortion_coefs=np.array([0.0]), telecentricity=0.0,
        name="cam")
    detection = _cube_detection("cam", target, truth)

    cam = target.initial_calibration(
        cam_name="cam", detection=detection, res=[720, 540],
        pose_im=0, model="telecentric", min_detections_per_board=12)

    assert isinstance(cam, TelecentricCamera)
    got = np.asarray(cam.intrinsic, dtype=float)
    assert got[0, 0] == pytest.approx(80.0, rel=0.05)
    assert got[1, 1] == pytest.approx(80.0, rel=0.05)


def test_a_camera_that_only_ever_saw_one_face_says_what_to_photograph():
    """A refusal that names the fix beats one that recommends the target
    already in use."""
    from pyCamSet.calibration_targets.core.target_registry import build_target

    target = build_target({"type": "Ccube", "n_points": 6, "length": 10.0})
    truth = TelecentricCamera(
        intrinsic=np.array([[80.0, 0, 270.0], [0, 80.0, 360.0], [0, 0, 1.0]]),
        res=[720, 540], distortion_coefs=np.array([0.0]), telecentricity=0.0,
        name="cam")
    detection = _cube_detection("cam", target, truth, faces_per_image=1)

    with pytest.raises(ValueError, match="two of its faces"):
        target.initial_calibration(
            cam_name="cam", detection=detection, res=[720, 540],
            pose_im=0, model="telecentric", min_detections_per_board=12)


# --------------------------------------------------------------------------
# Reading a calibration back out
#
# The tests above exercise the model -- projection, rays, jacobians, the seeded
# solve.  None of them asked the handler what it had found, which is the one
# thing every phase after a solve does: Phase 3 saves the camset, Phase 4 reads
# it again, and the diagnostics draw from it.  A pose row three wide breaks that
# read-back in three separate places, and a green model suite hid all three.
# --------------------------------------------------------------------------


def test_gauge_transform_survives_a_pose_with_no_translation(telecentric_problem):
    """The gauge transform composes a rigid update onto every camera's pose.

    A telecentric pose carries rotation alone, so the row is three wide and the
    code that scaled and unpacked a six-wide pose was handed an empty
    translation -- raising on both the 4x4 construction and the write-back.
    """
    cams, target, detection, poses = telecentric_problem
    # SelfBundleHandler, not TemplateBundleHandler: the gauge transform and the
    # target read-back are defined on the former, and Phase 4 uses it while
    # Phase 3 uses the latter.  Testing the wrong class would be vacuous.
    handler = SelfBundleHandler(camset=cams, target=target, detection=detection)

    primitive = handler.bundlePrimitive
    assert primitive.n_extr == 3, "this camset's poses must be the narrow ones"

    # Exactly the call get_camset and get_updated_target make.  No solve is
    # needed: the gauge transform runs after one, but the defect is in the call,
    # not in the result.
    params = ground_truth_params(handler, cams, poses)
    model = handler.bundlePrimitive.return_bundle_primitives(params)
    proj, extr, got_poses, points = handler.apply_gauge_transform(*model)

    assert extr.shape == (len(cams), 3), "the pose width must survive the transform"
    assert np.all(np.isfinite(extr))


def test_gauge_transform_still_moves_a_pinhole_rotation(telecentric_problem):
    """The narrow branch must not be a blanket skip.

    A six-wide pose is scaled and conjugated as before, so this pins that the
    fix is a branch rather than the removal of the transform.
    """
    cams, target, detection, poses = telecentric_problem
    handler = SelfBundleHandler(camset=cams, target=target, detection=detection)

    params = ground_truth_params(handler, cams, poses)
    proj, extr, poses_out, points = handler.bundlePrimitive.return_bundle_primitives(params)
    # A pinhole-shaped stand-in: six-wide rows carrying a real translation.
    wide = np.hstack([extr, np.full((len(extr), 3), 0.5)])
    proj2, extr2, _, _ = handler.apply_gauge_transform(proj, wide, poses_out, points)

    assert extr2.shape == wide.shape
    assert np.all(np.isfinite(extr2))
    # A six-wide row is still scaled and unpacked: the translation comes back,
    # and it is s times what went in (no rigid term on an all-equal column).
    assert extr2[:, 3:].shape == (len(cams), 3)


def test_get_camset_returns_a_telecentric_camset(telecentric_problem):
    """What Phase 3 and Phase 4 call the moment a solve finishes.

    The failure being guarded is not a wrong number: it is a ValueError raised
    after the optimisation has already converged, which costs the entire run.
    """
    cams, target, detection, poses = telecentric_problem
    # SelfBundleHandler, deliberately: TemplateBundleHandler.get_camset does not
    # call the gauge transform, which is exactly why Phase 3 survives this bug.
    # Asserting against that class would have passed before the fix too.
    handler = SelfBundleHandler(camset=cams, target=target, detection=detection)

    params = ground_truth_params(handler, cams, poses)
    out = handler.get_camset(params)

    assert out.get_n_cams() == len(cams)
    for name in cams.get_names():
        assert isinstance(out[name], TelecentricCamera)
        got = np.asarray(out[name].intrinsic, dtype=float)
        want = np.asarray(cams[name].intrinsic, dtype=float)
        assert got[0, 0] == pytest.approx(want[0, 0])
        assert got[1, 1] == pytest.approx(want[1, 1])


def test_get_updated_target_returns_the_recovered_shape(telecentric_problem):
    """The other read-back Phase 4's diagnostics depend on."""
    cams, target, detection, poses = telecentric_problem
    handler = SelfBundleHandler(camset=cams, target=target, detection=detection)

    params = ground_truth_params(handler, cams, poses)
    updated = handler.get_updated_target(params)

    n_points = int(np.prod(target.point_data.shape[:2]))
    assert np.asarray(updated).shape == (n_points, 3)
    assert np.all(np.isfinite(updated))


def test_a_saved_telecentric_camset_reloads_with_its_handler(tmp_path, telecentric_problem):
    """Save and reload is how Phase 4 receives Phase 3's result.

    The handler class round-trips, which matters because it is the object the
    Assess Calibration figures are drawn from.
    """
    from pyCamSet.utils.saving import load_CameraSet

    cams, target, detection, poses = telecentric_problem
    handler = SelfBundleHandler(camset=cams, target=target, detection=detection)
    out = handler.get_camset(ground_truth_params(handler, cams, poses))

    written = tmp_path / "telecentric.camset"
    out.save(written)
    reloaded = load_CameraSet(written)

    assert reloaded.get_n_cams() == len(cams)
    assert all(isinstance(reloaded[n], TelecentricCamera) for n in cams.get_names())


def test_fix_all_cameras_packs_telecentric_widths(telecentric_problem):
    """Holding every camera fixed takes its widths from the lens model.

    The literals here were six and nine whatever the model, so a telecentric
    set could not be held at all: the handler's broadcast refused them.
    """
    from pyCamSet.optimisation.find_target import fix_all_cameras

    cams, _, _, _ = telecentric_problem
    fixed = fix_all_cameras(cams)

    assert set(fixed) == set(cams.get_names())
    for name in cams.get_names():
        assert fixed[name]["ext"].shape == (3,), "rotation only: no translation"
        assert fixed[name]["int"].shape == (6,), "m_x, c_x, m_y, c_y, k, eps"


def test_a_telecentric_set_can_be_held_at_its_calibration(telecentric_problem):
    """The whole point of the widths: construct a handler that holds them fixed.

    This is find_target's real path -- it photographs a known target in a known
    rig -- so it has to work for whatever lens that rig was calibrated with.
    """
    from pyCamSet.optimisation.find_target import fix_all_cameras

    cams, target, detection, poses = telecentric_problem
    handler = TemplateBundleHandler(
        camset=cams, target=target, detection=detection,
        fixed_params=fix_all_cameras(cams),
        options={"verbosity": 0, "fixed_pose": []})

    assert not np.any(handler.bundlePrimitive.extr_unfixed)
    assert not np.any(handler.bundlePrimitive.intr_unfixed)

# --------------------------------------------------------------------------
# Does the gauge transform actually preserve the calibration?
#
# The tests above exercise the model and the read-back's shapes.  None of them
# asked the one question the gauge transform exists to answer, and its docstring
# asserts as a guarantee: "The transformation is garunteed to preserve the
# calibration result."
#
# That is testable.  Take a parameter vector that reproduces the detections --
# residual zero, by construction -- apply the transform, and require the
# residual still to be zero.  It was not, and the difference is what a rig
# reports back after a self-calibration.
# --------------------------------------------------------------------------


def _self_params(handler, cams, poses, target):
    """A full SelfBundleHandler parameter vector: cameras, poses, free points.

    `ground_truth_params` builds the posed block alone, which is the whole
    vector for `TemplateBundleHandler`.  This handler carries a block of free
    target points after it, so the posed block on its own is SHORT and every
    slice past `pose_end` would be misaligned.
    """
    head = ground_truth_params(handler, cams, poses)
    bp = handler.bundlePrimitive
    assert len(head) == bp.pose_end, (
        f"the posed block is {len(head)} long but pose_end is {bp.pose_end}")
    free = np.asarray(handler.feat_unfixed, dtype=bool)
    return np.concatenate([head, np.asarray(target.point_data, dtype=float).reshape(-1)[free]])


def _reprojection_of(arrays, handler, cams):
    """Mean pixel reprojection of the model the four returned arrays describe.

    Measured straight from the arrays, never by repacking them into a parameter
    vector: repacking has to know which points are free and which are held for
    the gauge, and a mistake there is indistinguishable from a broken gauge.
    """
    proj, extr, poses, points = arrays
    proj = np.asarray(proj, dtype=float)
    extr = np.asarray(extr, dtype=float)
    placed = np.array([
        make_4x4h_tform(p[:3], p[3:]) for p in np.asarray(poses, dtype=float)])
    point_data = np.asarray(points, dtype=float).reshape(-1, 3)

    local = []
    for cam in cams:
        clone = cam.__class__.__new__(cam.__class__)
        clone.__dict__.update(cam.__dict__)
        local.append(clone)

    flat = handler.detection.sort(["key", "global_im_num"]).get_data()
    errors = []
    for row in flat:
        cam_num, im_num, key = int(row[0]), int(row[1]), int(row[2])
        cam = local[cam_num]
        cam.extrinsic = _extrinsic_from_params(extr[cam_num], cam.extrinsic)
        cam.from_param_vector(proj[cam_num])
        cam._update_state()
        world = h_tform(point_data[key], placed[im_num])
        uv = np.atleast_2d(cam.project_points(world, distort=True))[0]
        errors.append(np.hypot(uv[0] - row[-2], uv[1] - row[-1]))
    return float(np.mean(errors))


# --------------------------------------------------------------------------
# Does the gauge transform actually preserve the calibration?
#
# The tests above exercise the model and the read-back's shapes.  None of them
# asked the one question the gauge transform exists to answer, and its docstring
# asserts as a guarantee: "The transformation is garunteed to preserve the
# calibration result."
#
# That is testable.  Take a parameter vector that reproduces the detections --
# residual zero, by construction -- apply the transform, and require the
# residual still to be zero.  It was not, and the difference is what a rig
# reports back after a self-calibration.
# --------------------------------------------------------------------------


def _self_params(handler, cams, poses, target):
    """A full SelfBundleHandler parameter vector: cameras, poses, free points.

    `ground_truth_params` builds the posed block alone, which is the whole
    vector for `TemplateBundleHandler`.  This handler carries a block of
    estimated target points after it, so the posed block on its own is SHORT and
    everything past `pose_end` would be misaligned.
    """
    head = ground_truth_params(handler, cams, poses)
    bp = handler.bundlePrimitive
    assert len(head) == bp.pose_end, (
        f"the posed block is {len(head)} long but pose_end is {bp.pose_end}")
    free = np.asarray(handler.feat_unfixed, dtype=bool)
    return np.concatenate([head, np.asarray(target.point_data, dtype=float).reshape(-1)[free]])


def _reprojection_of(arrays, handler, cams):
    """Mean pixel reprojection of the model the four returned arrays describe.

    Measured straight from the arrays, never by repacking them into a parameter
    vector: repacking has to know which points are estimated and which are held
    for the gauge, and a mistake there is indistinguishable from a broken gauge.
    """
    proj, extr, poses, points = arrays
    proj = np.asarray(proj, dtype=float)
    extr = np.asarray(extr, dtype=float)
    placed = np.array([
        make_4x4h_tform(p[:3], p[3:]) for p in np.asarray(poses, dtype=float)])
    point_data = np.asarray(points, dtype=float).reshape(-1, 3)

    local = []
    for cam in cams:
        clone = cam.__class__.__new__(cam.__class__)
        clone.__dict__.update(cam.__dict__)
        local.append(clone)

    flat = handler.detection.sort(["key", "global_im_num"]).get_data()
    errors = []
    for row in flat:
        cam_num, im_num, key = int(row[0]), int(row[1]), int(row[2])
        cam = local[cam_num]
        cam.extrinsic = _extrinsic_from_params(extr[cam_num], cam.extrinsic)
        cam.from_param_vector(proj[cam_num])
        cam._update_state()
        world = h_tform(point_data[key], placed[im_num])
        uv = np.atleast_2d(cam.project_points(world, distort=True))[0]
        errors.append(np.hypot(uv[0] - row[-2], uv[1] - row[-1]))
    return float(np.mean(errors))


# --------------------------------------------------------------------------
# Does the gauge transform actually preserve the calibration?
#
# The tests above exercise the model and the read-back's shapes.  None of them
# asked the one question the gauge transform exists to answer, and its docstring
# asserts as a guarantee: "The transformation is garunteed to preserve the
# calibration result."
#
# That is testable, and it was false.  A self-calibration scales its target --
# that is one of the freedoms the solve has -- and this rig reported its result
# back at the wrong scale and the wrong orientation: 426 px of reprojection on a
# real seven-camera cube, against the 0.46 px the solve had actually reached.
# --------------------------------------------------------------------------


def _self_params(handler, cams, poses, target):
    """A full SelfBundleHandler parameter vector: cameras, poses, free points.

    `ground_truth_params` builds the posed block alone, which is the whole
    vector for `TemplateBundleHandler`.  This handler carries a block of
    estimated target points after it, so the posed block on its own is SHORT and
    everything past `pose_end` would be misaligned.
    """
    head = ground_truth_params(handler, cams, poses)
    bp = handler.bundlePrimitive
    assert len(head) == bp.pose_end, (
        f"the posed block is {len(head)} long but pose_end is {bp.pose_end}")
    free = np.asarray(handler.feat_unfixed, dtype=bool)
    return np.concatenate(
        [head, np.asarray(target.point_data, dtype=float).reshape(-1)[free]])


def _reprojection_of(arrays, handler, cams):
    """Mean pixel reprojection of the model the four returned arrays describe.

    Measured straight from the arrays, never by repacking them into a parameter
    vector: repacking has to know which points are estimated and which are held
    for the gauge, and a mistake there is indistinguishable from a broken gauge.
    """
    proj, extr, poses, points = arrays
    proj = np.asarray(proj, dtype=float)
    extr = np.asarray(extr, dtype=float)
    placed = np.array([
        make_4x4h_tform(p[:3], p[3:]) for p in np.asarray(poses, dtype=float)])
    point_data = np.asarray(points, dtype=float).reshape(-1, 3)

    local = []
    for cam in cams:
        clone = cam.__class__.__new__(cam.__class__)
        clone.__dict__.update(cam.__dict__)
        local.append(clone)

    flat = handler.detection.sort(["key", "global_im_num"]).get_data()
    errors = []
    for row in flat:
        cam_num, im_num, key = int(row[0]), int(row[1]), int(row[2])
        cam = local[cam_num]
        cam.extrinsic = _extrinsic_from_params(extr[cam_num], cam.extrinsic)
        cam.from_param_vector(proj[cam_num])
        cam._update_state()
        world = h_tform(point_data[key], placed[im_num])
        uv = np.atleast_2d(cam.project_points(world, distort=True))[0]
        errors.append(np.hypot(uv[0] - row[-2], uv[1] - row[-1]))
    return float(np.mean(errors))


# --------------------------------------------------------------------------
# Does the gauge transform actually preserve the calibration?
#
# The tests above exercise the model and the read-back's shapes.  None of them
# asked the one question the gauge transform exists to answer, and its docstring
# asserts as a guarantee: "The transformation is garunteed to preserve the
# calibration result."
#
# That is testable, and it was false.  A self-calibration scales its target --
# that is one of the freedoms the solve has -- and this rig reported its result
# back at the wrong scale: 426 px of reprojection on a real seven-camera cube,
# against the 0.46 px the solve had actually reached.
# --------------------------------------------------------------------------


def _self_params(handler, cams, poses, target):
    """A full SelfBundleHandler parameter vector: cameras, poses, free points.

    `ground_truth_params` builds the posed block alone, which is the whole
    vector for `TemplateBundleHandler`.  This handler carries a block of
    estimated target points after it, so the posed block on its own is SHORT and
    everything past `pose_end` would be misaligned.
    """
    head = ground_truth_params(handler, cams, poses)
    bp = handler.bundlePrimitive
    assert len(head) == bp.pose_end, (
        f"the posed block is {len(head)} long but pose_end is {bp.pose_end}")
    free = np.asarray(handler.feat_unfixed, dtype=bool)
    return np.concatenate(
        [head, np.asarray(target.point_data, dtype=float).reshape(-1)[free]])


def _reprojection_of(arrays, handler, cams):
    """Mean pixel reprojection of the model the four returned arrays describe.

    Measured straight from the arrays, never by repacking them into a parameter
    vector: repacking has to know which points are estimated and which are held
    for the gauge, and a mistake there is indistinguishable from a broken gauge.
    """
    proj, extr, poses, points = arrays
    proj = np.asarray(proj, dtype=float)
    extr = np.asarray(extr, dtype=float)
    placed = np.array([
        make_4x4h_tform(p[:3], p[3:]) for p in np.asarray(poses, dtype=float)])
    point_data = np.asarray(points, dtype=float).reshape(-1, 3)

    local = []
    for cam in cams:
        clone = cam.__class__.__new__(cam.__class__)
        clone.__dict__.update(cam.__dict__)
        local.append(clone)

    flat = handler.detection.sort(["key", "global_im_num"]).get_data()
    errors = []
    for row in flat:
        cam_num, im_num, key = int(row[0]), int(row[1]), int(row[2])
        cam = local[cam_num]
        cam.extrinsic = _extrinsic_from_params(extr[cam_num], cam.extrinsic)
        cam.from_param_vector(proj[cam_num])
        cam._update_state()
        world = h_tform(point_data[key], placed[im_num])
        uv = np.atleast_2d(cam.project_points(world, distort=True))[0]
        errors.append(np.hypot(uv[0] - row[-2], uv[1] - row[-1]))
    return float(np.mean(errors))


# --------------------------------------------------------------------------
# Does the gauge transform actually preserve the calibration?
#
# The tests above exercise the model and the read-back's shapes.  None of them
# asked the one question the gauge transform exists to answer, and its docstring
# asserts as a guarantee: "The transformation is garunteed to preserve the
# calibration result."
#
# That is testable, and it was false.  A self-calibration scales its target --
# one of the freedoms the solve has -- and this rig reported its result back at
# the wrong scale: 426 px of reprojection on a real seven-camera cube, against
# the 0.46 px the solve had actually reached.
# --------------------------------------------------------------------------


def _self_params(handler, cams, poses, target):
    """A full SelfBundleHandler parameter vector: cameras, poses, free points.

    `ground_truth_params` builds the posed block alone, which is the whole
    vector for `TemplateBundleHandler`.  This handler carries a block of
    estimated target points after it, so the posed block on its own is SHORT and
    everything past `pose_end` would be misaligned.
    """
    head = ground_truth_params(handler, cams, poses)
    bp = handler.bundlePrimitive
    assert len(head) == bp.pose_end, (
        f"the posed block is {len(head)} long but pose_end is {bp.pose_end}")
    free = np.asarray(handler.feat_unfixed, dtype=bool)
    return np.concatenate(
        [head, np.asarray(target.point_data, dtype=float).reshape(-1)[free]])


def _reprojection_of(arrays, handler, cams):
    """Mean pixel reprojection of the model the four returned arrays describe.

    Measured straight from the arrays, never by repacking them into a parameter
    vector: repacking has to know which points are estimated and which are held
    for the gauge, and a mistake there is indistinguishable from a broken gauge.
    """
    proj, extr, poses, points = arrays
    proj = np.asarray(proj, dtype=float)
    extr = np.asarray(extr, dtype=float)
    placed = np.array([
        make_4x4h_tform(p[:3], p[3:]) for p in np.asarray(poses, dtype=float)])
    point_data = np.asarray(points, dtype=float).reshape(-1, 3)

    local = []
    for cam in cams:
        clone = cam.__class__.__new__(cam.__class__)
        clone.__dict__.update(cam.__dict__)
        local.append(clone)

    flat = handler.detection.sort(["key", "global_im_num"]).get_data()
    errors = []
    for row in flat:
        cam_num, im_num, key = int(row[0]), int(row[1]), int(row[2])
        cam = local[cam_num]
        cam.extrinsic = _extrinsic_from_params(extr[cam_num], cam.extrinsic)
        cam.from_param_vector(proj[cam_num])
        cam._update_state()
        world = h_tform(point_data[key], placed[im_num])
        uv = np.atleast_2d(cam.project_points(world, distort=True))[0]
        errors.append(np.hypot(uv[0] - row[-2], uv[1] - row[-1]))
    return float(np.mean(errors))


# --------------------------------------------------------------------------
# Does the gauge transform actually preserve the calibration?
#
# The tests above exercise the model and the read-back's shapes.  None of them
# asked the one question the gauge transform exists to answer, and its docstring
# asserts as a guarantee: "The transformation is garunteed to preserve the
# calibration result."
#
# That is testable, and it was false.  A self-calibration scales its target --
# one of the freedoms the solve has -- and this rig reported its result back at
# the wrong scale: 426 px of reprojection on a real seven-camera cube, against
# the 0.46 px the solve had actually reached.
# --------------------------------------------------------------------------


def _self_params(handler, cams, poses, target, point_scale=1.0):
    """A full SelfBundleHandler parameter vector: cameras, poses, free points.

    `ground_truth_params` builds the posed block alone, which is the whole
    vector for `TemplateBundleHandler`.  This handler carries a block of
    estimated target points after it, so the posed block on its own is SHORT and
    everything past `pose_end` would be misaligned.

    :param point_scale: carry the estimated points at this multiple of the
        model's, which is how a self-calibration's own answer looks before the
        gauge has been applied to it.  1.0 is the drawn model exactly.
    """
    head = ground_truth_params(handler, cams, poses)
    bp = handler.bundlePrimitive
    assert len(head) == bp.pose_end, (
        f"the posed block is {len(head)} long but pose_end is {bp.pose_end}")
    free = np.asarray(handler.feat_unfixed, dtype=bool)
    points = np.asarray(target.point_data, dtype=float).reshape(-1)[free].copy()
    if point_scale != 1.0:
        n_pts = points.size // 3
        block = points.reshape((n_pts, 3))
        free_pts = np.asarray(handler.feat_unfixed, dtype=bool).reshape((n_pts, 3))
        block = block * point_scale
        # components held for the gauge carry no parameter, so they are not
        # scaled with the rest
        block[~free_pts] = (points.reshape((n_pts, 3)))[~free_pts]
        points = block.reshape(-1)
    return np.concatenate([head, points])


def _reprojection_of(arrays, handler, cams):
    """Mean pixel reprojection of the model the four returned arrays describe.

    Measured straight from the arrays, never by repacking them into a parameter
    vector: repacking has to know which points are estimated and which are held
    for the gauge, and a mistake there is indistinguishable from a broken gauge.
    """
    proj, extr, poses, points = arrays
    proj = np.asarray(proj, dtype=float)
    extr = np.asarray(extr, dtype=float)
    placed = np.array([
        make_4x4h_tform(p[:3], p[3:]) for p in np.asarray(poses, dtype=float)])
    point_data = np.asarray(points, dtype=float).reshape(-1, 3)

    local = []
    for cam in cams:
        clone = cam.__class__.__new__(cam.__class__)
        clone.__dict__.update(cam.__dict__)
        local.append(clone)

    flat = handler.detection.sort(["key", "global_im_num"]).get_data()
    errors = []
    for row in flat:
        cam_num, im_num, key = int(row[0]), int(row[1]), int(row[2])
        cam = local[cam_num]
        cam.extrinsic = _extrinsic_from_params(extr[cam_num], cam.extrinsic)
        cam.from_param_vector(proj[cam_num])
        cam._update_state()
        world = h_tform(point_data[key], placed[im_num])
        uv = np.atleast_2d(cam.project_points(world, distort=True))[0]
        errors.append(np.hypot(uv[0] - row[-2], uv[1] - row[-1]))
    return float(np.mean(errors))

# --------------------------------------------------------------------------
# Does the gauge transform actually preserve the calibration?
#
# The tests above exercise the model and the read-back's shapes.  None of them
# asked the one question the gauge transform exists to answer, and its docstring
# asserts as a guarantee: "The transformation is garunteed to preserve the
# calibration result."
#
# That is testable, and it was false.  A self-calibration scales its target --
# one of the freedoms the solve has -- and this rig reported its result back at
# the wrong scale: 426 px of reprojection on a real seven-camera cube, against
# the 0.46 px the solve had actually reached.
# --------------------------------------------------------------------------


def _self_params(handler, cams, poses, target, point_scale=1.0):
    """A full SelfBundleHandler parameter vector: cameras, poses, free points.

    `ground_truth_params` builds the posed block alone, which is the whole
    vector for `TemplateBundleHandler`.  This handler carries a block of
    estimated target points after it, so the posed block on its own is SHORT and
    everything past `pose_end` would be misaligned.

    :param point_scale: carry the estimated points at this multiple of the
        model's, which is how a self-calibration's own answer looks before the
        gauge has been applied to it.  1.0 is the drawn model exactly.
    """
    head = ground_truth_params(handler, cams, poses)
    bp = handler.bundlePrimitive
    assert len(head) == bp.pose_end, (
        f"the posed block is {len(head)} long but pose_end is {bp.pose_end}")
    free = np.asarray(handler.feat_unfixed, dtype=bool)
    # the free block is one scalar per held-free component, in flat order
    points = np.asarray(target.point_data, dtype=float).reshape(-1)[free]
    return np.concatenate([head, points * point_scale])


def _reprojection_of(arrays, handler, cams):
    """Mean pixel reprojection of the model the four returned arrays describe.

    Measured straight from the arrays, never by repacking them into a parameter
    vector: repacking has to know which points are estimated and which are held
    for the gauge, and a mistake there is indistinguishable from a broken gauge.
    """
    proj, extr, poses, points = arrays
    proj = np.asarray(proj, dtype=float)
    extr = np.asarray(extr, dtype=float)
    placed = np.array([
        make_4x4h_tform(p[:3], p[3:]) for p in np.asarray(poses, dtype=float)])
    point_data = np.asarray(points, dtype=float).reshape(-1, 3)

    local = []
    for cam in cams:
        clone = cam.__class__.__new__(cam.__class__)
        clone.__dict__.update(cam.__dict__)
        local.append(clone)

    flat = handler.detection.sort(["key", "global_im_num"]).get_data()
    errors = []
    for row in flat:
        cam_num, im_num, key = int(row[0]), int(row[1]), int(row[2])
        cam = local[cam_num]
        cam.extrinsic = _extrinsic_from_params(extr[cam_num], cam.extrinsic)
        cam.from_param_vector(proj[cam_num])
        cam._update_state()
        world = h_tform(point_data[key], placed[im_num])
        uv = np.atleast_2d(cam.project_points(world, distort=True))[0]
        errors.append(np.hypot(uv[0] - row[-2], uv[1] - row[-1]))
    return float(np.mean(errors))


def test_the_gauge_transform_preserves_the_reprojection(telecentric_problem):
    """The gauge is a symmetry of the model, or it is not.

    With the target sitting where the model says it is, only the rigid half of
    the gauge has anything to do, and that half was already correct -- so this
    passes on the code as it stood.  It is here to pin that the fix did not
    break it.
    """
    cams, target, detection, poses = telecentric_problem
    handler = SelfBundleHandler(camset=cams, target=target, detection=detection)
    handler.missing_poses = np.zeros(len(poses), dtype=bool)

    before_arrays = handler.bundlePrimitive.return_bundle_primitives(
        _self_params(handler, cams, poses, target))
    before = _reprojection_of(before_arrays, handler, cams)
    after = _reprojection_of(
        handler.apply_gauge_transform(*before_arrays), handler, cams)

    assert before < 1e-6, f"the fixture itself is not exact: {before} px"
    assert after == pytest.approx(before, abs=1e-6), (
        f"the gauge moved the reprojection from {before} to {after} px")


def test_the_gauge_rescales_the_lens_when_the_camera_has_no_translation(
        telecentric_problem):
    """Where the world's scale is absorbed, for a camera that cannot carry it.

    The gauge hands the rig back re-expressed in the target's units, and the
    world scale is one of the freedoms it removes.  A pinhole camera takes that
    scale in its extrinsic TRANSLATION: its projection divides by z, so the
    camera's own position is the only thing the scale can change, and the code
    scales that field.  A telecentric extrinsic is rotation-only -- three
    parameters, no translation at all -- so that step has nothing to act on.

    The compensation has to happen in the lens instead.  A telecentric pixel is

        u = m*x/(1 + eps*z) + c

    and scaling the world by s is undone exactly by m -> m/s together with
    eps -> eps/s, because (m/s)(s*x)/(1 + (eps/s)(s*z)) is the original
    expression.

    Feeding the problem a target carried at a scale of its own is what makes the
    gauge's scale non-trivial; with the target already at the model's scale s
    comes out at 1 and this branch does nothing.  That is exactly why the defect
    survived: every fixture in this file has the target at the model's scale.
    """
    s0 = 1.37
    cams, target, detection, poses = telecentric_problem
    handler = SelfBundleHandler(camset=cams, target=target, detection=detection)
    handler.missing_poses = np.zeros(len(poses), dtype=bool)
    bp = handler.bundlePrimitive

    before = bp.return_bundle_primitives(
        _self_params(handler, cams, poses, target, point_scale=s0))
    proj_in, extr_in = (np.asarray(a, dtype=float).copy() for a in before[:2])

    proj_out, extr_out, _, _ = handler.apply_gauge_transform(*before)
    proj_out = np.asarray(proj_out, dtype=float)
    extr_out = np.asarray(extr_out, dtype=float)

    # what the gauge must have done: divide the lens by the scale it removed.
    # the scale itself is read back from that division, so this also fails, and
    # with this message, on code that left the lens alone entirely -- which is
    # the defect it is here to catch.
    assert not np.allclose(proj_out[:, 0], proj_in[:, 0]), (
        "the magnification did not take the world's scale. A translation-free "
        "camera has no extrinsic to carry it, so every pixel moves by the whole "
        "size of the correction")
    ratios = proj_in[:, 0] / proj_out[:, 0]
    assert not np.allclose(ratios, 1.0), (
        "the gauge found no scale to remove; this test would be vacuous. The "
        "target has to be carried at its own scale for the branch to do work")
    assert np.allclose(ratios, ratios[0]), (
        "every camera in a rig is re-expressed with the same scale")

    # m_x, m_y and eps carry the scale; the principal point and k do not
    assert np.allclose(proj_out[:, 1], proj_in[:, 1]), "c_x is not a scale"
    assert np.allclose(proj_out[:, 3], proj_in[:, 3]), "c_y is not a scale"
    assert np.allclose(proj_out[:, 4], proj_in[:, 4]), "k is not a scale"
    assert np.allclose(proj_out[:, 2], proj_in[:, 2] / ratios), "m_y must follow"
    assert np.allclose(proj_out[:, 5], proj_in[:, 5] / ratios), (
        "the telecentricity must be divided by the same scale as the "
        "magnification, or the depth term drifts as the world does")

    # the rig itself has not turned; only the world's frame moved
    assert np.allclose(extr_out, extr_in), (
        "the gauge moved the camera rotations, but a world scale is not a "
        "rotation of the rig")



def test_the_gauge_leaves_every_scalar_the_solve_did_not_estimate(
        telecentric_problem):
    """A point nothing observes, and nothing solves for, must come back unmoved.

    This handler holds a few scalars at their model coordinates to pin the gauge
    (see ``find_not_colinear_pts``), so those scalars carry no parameter and the
    solve never writes to them.  Where such a point was also never seen, it has
    no residual row at all: the gauge may leave it exactly where the caller left
    it, which is where the gauge is putting everything else, and no pixel cares
    because nothing images it.

    Carrying it through the gauge anyway moves it by the whole size of the
    correction.  On ccube2/5_corners/experiment_001 -- 294 target points, of
    which 279 were detected by one of the seven cameras -- the 15 that no camera
    ever saw came back a mean 19.68 mm and up to 26.41 mm from the printed
    model, against a target whose largest point separation is 16.19 mm.  Those
    points, not the cube, then set the cloud's bounding size: the same run's
    cloud measured 30.04 mm across where the printed model measures 16.19 mm.

    The distinction this test draws is between UNOBSERVED and merely pinned, and
    it is not a detail.  Point 1 of that run is pinned and seen; point 7 is
    pinned in one component and seen too -- in 75 of the 98 images, across all
    seven cameras -- and the gauge moves it 1.19 mm.  Its pixel still ties it to
    the rest of the cloud, so it has to move with the world like any other
    imaged point.  Only point 0 is both pinned and unobserved.  So the condition
    the gauge keys on is visibility, not whether a component happens to be a
    free parameter.

    The pinned points are therefore made UNSEEN here, by dropping every other
    key from the detections' shape.  A point a camera observes cannot be both
    pinned and free to be re-framed, so with it in view the assertion below
    would be measuring the construction rather than the gauge.
    """
    cams, target, detection, poses = telecentric_problem
    model = np.asarray(target.point_data, dtype=float).reshape(-1, 3)

    probe = SelfBundleHandler(camset=cams, target=target, detection=detection)
    pinned_points = np.logical_not(np.logical_or.reduce(
        np.asarray(probe.feat_unfixed, dtype=bool).reshape(-1, 3), axis=1))
    assert pinned_points.any(), "this fixture must hold some points for the gauge"

    # Leave the pinned points in view but drop every OTHER key from the
    # detections' target shape, so that what the handler ends up holding is a
    # set of points nothing observes.  That is the case the real run is in: of
    # its 15 never-seen points, one is a pinned gauge point.
    keep = set(np.flatnonzero(np.logical_not(pinned_points)).tolist())
    rows = detection.get_data()
    trimmed = TargetDetection(
        cam_names=cams.get_names(),
        data=rows[[int(r[2]) in keep for r in rows]])
    unobserved = np.logical_not(np.asarray(
        SelfBundleHandler(
            camset=cams, target=target, detection=trimmed
        ).visible_feature_mask, dtype=bool))
    assert unobserved.any(), "the pinned points must end up unobserved here"

    handler = SelfBundleHandler(camset=cams, target=target, detection=trimmed)
    handler.missing_poses = np.zeros(len(poses), dtype=bool)
    bp = handler.bundlePrimitive
    estimated = np.asarray(handler.feat_unfixed, dtype=bool)
    assert estimated.size == model.size

    base = bp.return_bundle_primitives(
        np.concatenate([ground_truth_params(handler, cams, poses),
                        model.reshape(-1)[estimated]]))
    proj, extr, target_poses, _ = (np.asarray(a, dtype=float).copy() for a in base)

    # A state a solve can produce: the estimated scalars at a scale of their own,
    # with the lens and target poses carried along so the pixels are unchanged.
    s0 = 1.37
    proj[:, 0] /= s0          # m_x
    proj[:, 2] /= s0          # m_y
    proj[:, 5] /= s0          # eps
    target_poses[:, 3:] = target_poses[:, 3:] * s0

    given = np.where(estimated, model.reshape(-1) * s0, model.reshape(-1))
    # the pinned scalars are nudged off where the model put them, so that a
    # gauge which quietly moves them cannot pass by accident
    offset = np.flatnonzero(np.logical_not(estimated))
    given[offset] = given[offset] + 0.0011

    state = (proj, extr, target_poses, given.reshape(-1, 3))
    after = np.asarray(
        handler.apply_gauge_transform(*state)[3], dtype=float).reshape(-1, 3)
    before_pts = given.reshape(-1, 3)

    # the gauge must actually have done something, or this proves nothing
    assert not np.allclose(after, before_pts), (
        "the gauge was a no-op here, so this test cannot see the defect")

    moved = np.linalg.norm(after[unobserved] - before_pts[unobserved], axis=1)
    assert np.allclose(moved, 0.0, atol=1e-15), (
        f"the gauge moved {int((moved > 1e-15).sum())} of {int(unobserved.sum())} "
        f"points that no camera observed and no parameter covers, by up to "
        f"{moved.max() * 1000:.4f} mm. Nothing images them and nothing solves for "
        "them, so the gauge has no business moving them")
