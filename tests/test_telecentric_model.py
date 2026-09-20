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
from pyCamSet.optimisation.template_handler import TemplateBundleHandler
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
    from pyCamSet.calibration.telecentric import calibrate_telecentric

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
    from pyCamSet.calibration.telecentric import fit_affine_camera, is_planar

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
    from pyCamSet.calibration.telecentric import calibrate_telecentric

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


# ----------------------------------------------------------------------
# the self-calibration gauge
# ----------------------------------------------------------------------


def _telecentric_pixels(proj, extr, poses, points):
    """Every (camera, pose, point) pixel, straight from the block's formula."""
    from pyCamSet.utils.general_utils import make_4x4h_tform

    out = []
    for ci in range(len(proj)):
        m_x, c_x, m_y, c_y, k, eps = (float(v) for v in proj[ci][:6])
        cam_t = make_4x4h_tform(
            np.asarray(extr[ci][:3], dtype=float),
            np.asarray(extr[ci][3:6], dtype=float)
            if len(extr[ci]) >= 6 else np.zeros(3))
        for pi in range(len(poses)):
            pose_t = make_4x4h_tform(np.asarray(poses[pi][:3], dtype=float),
                                     np.asarray(poses[pi][3:6], dtype=float))
            y = h_tform(h_tform(points, pose_t), cam_t)
            w = 1.0 / (1.0 + eps * y[:, 2])
            xs, ys = m_x * y[:, 0] * w, m_y * y[:, 1] * w
            r2 = (xs * xs + ys * ys) * 1e-6
            den = 1.0 / (1.0 + k * r2)
            out.append(np.stack([xs * den + c_x, ys * den + c_y], axis=-1))
    return np.concatenate(out, axis=0)


def test_the_gauge_transform_leaves_a_telecentric_projection_alone(
        telecentric_problem):
    """The gauge transform re-expresses a solve in the reference frame. It is
    allowed to move every parameter; it is not allowed to move a pixel.

    A telecentric camera's extrinsic is rotation only, so it cannot absorb the
    gauge the way a pinhole's translation does. The poses take the translation
    and the magnification takes the scale. If any part of that bookkeeping is
    wrong the calibration silently changes, which is worse than the crash this
    replaced -- so the check is on the pixels, not on the parameters.

    The points are pushed off the reference deliberately. Left where they are
    the gauge is the identity and this would pass without testing anything.
    """
    from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler
    from pyCamSet.utils.general_utils import ext_4x4_to_rod

    cams, target, detection, poses = telecentric_problem
    handler = SelfBundleHandler(camset=cams, target=target, detection=detection,
                                options={"outliers": "n"})

    proj = np.array([cam.to_param_vector() for cam in cams], dtype=float)
    extr = np.array([np.asarray(ext_4x4_to_rod(cam.extrinsic)[0], dtype=float)
                     for cam in cams], dtype=float)
    assert extr.shape[1] == 3, "a telecentric extrinsic is rotation only"

    pose_block = np.array(
        [np.concatenate(ext_4x4_to_rod(p)) for p in poses], dtype=float)

    # A solve that has drifted: the points come back scaled, turned and moved.
    drift = make_4x4h_tform(
        Rotation.from_euler("xyz", [0.03, -0.02, 0.05]).as_rotvec(),
        [0.0004, -0.0007, 0.0011])
    points = h_tform(target.point_data.reshape(-1, 3) * 1.037, drift)

    before = _telecentric_pixels(proj, extr, pose_block, points)
    new_proj, new_extr, new_poses, new_points = handler.apply_gauge_transform(
        proj.copy(), extr.copy(), pose_block.copy(), points.copy())
    after = _telecentric_pixels(np.asarray(new_proj), np.asarray(new_extr),
                               np.asarray(new_poses), np.asarray(new_points))

    assert np.isfinite(before).all() and np.isfinite(after).all()
    moved = np.linalg.norm(after - before, axis=1)
    assert np.nanmax(moved) < 1e-6, (
        f"the gauge transform moved a pixel by {np.nanmax(moved):.3e} px")

    # and it did do something: the scale left the extrinsic, which has nowhere
    # to put it, and landed in the magnification.
    assert not np.allclose(np.asarray(new_proj)[:, 0], proj[:, 0]), \
        "magnification should carry the gauge scale for a telecentric lens"
    assert np.allclose(np.asarray(new_proj)[:, 1], proj[:, 1]), \
        "there is no in-plane shift left for the principal point to absorb"
