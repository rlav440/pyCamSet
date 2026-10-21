"""
The telecentric camera model, from the projection up to a solved bundle adjustment.

A new lens model can drive reprojection error to near zero while recovering the
wrong magnification, so the accuracy tests here assert the parameters against
ground truth rather than only the residuals.
"""
import cv2
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


def _division_remap(image, cam):
    """The undistortion written out by hand: each output pixel reads the source pixel it came from."""
    rows, cols = image.shape[:2]
    u, v = np.meshgrid(np.arange(cols, dtype=float), np.arange(rows, dtype=float))
    c_x, c_y = cam.principal_point
    den = 1.0 / (1.0 + cam.distortion_coefs[0] * ((u - c_x) ** 2 + (v - c_y) ** 2) * 1e-6)
    return cv2.remap(image, ((u - c_x) * den + c_x).astype(np.float32),
                     ((v - c_y) * den + c_y).astype(np.float32), cv2.INTER_LINEAR)


def test_undistort_keeps_a_non_square_image_shape():
    """A 960 x 1280 (rows x cols) image comes back 960 x 1280, undistorted by the division model."""
    cam = make_telecentric_camera("cam", k=0.23)
    image = np.random.default_rng(0).random((REF_RES[1], REF_RES[0])).astype(np.float32)

    out = cam.undistort(image)

    assert out.shape == image.shape
    assert np.array_equal(out, _division_remap(image, cam))


def test_undistort_ignores_a_res_saved_height_first():
    """Camsets written before the fix carry ``res`` as ``(height, width)``; the image decides the grid."""
    cam = make_telecentric_camera("cam", k=0.23)
    image = np.random.default_rng(1).random((REF_RES[1], REF_RES[0])).astype(np.float32)
    swapped = TelecentricCamera(
        extrinsic=cam.extrinsic, intrinsic=cam.intrinsic, res=[REF_RES[1], REF_RES[0]],
        distortion_coefs=cam.distortion_coefs, telecentricity=cam.telecentricity, name="cam")

    assert np.array_equal(swapped.undistort(image), cam.undistort(image))


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


# --------------------------------------------------------------------------
# reading a self calibration back out, and its gauge
# --------------------------------------------------------------------------


def _self_params(handler, cams, poses, target, point_scale=1.0):
    """A full SelfBundleHandler parameter vector: cameras, poses, then free points.

    :param point_scale: the multiple of the model the free points are carried at
    """
    head = ground_truth_params(handler, cams, poses)
    assert len(head) == handler.bundlePrimitive.pose_end
    free = np.asarray(handler.feat_unfixed, dtype=bool)
    points = np.asarray(target.point_data, dtype=float).reshape(-1)[free]
    return np.concatenate([head, points * point_scale])


def test_gauge_transform_keeps_a_pose_with_no_translation(telecentric_problem):
    """A telecentric extrinsic is rotation only, and stays three wide."""
    cams, target, detection, poses = telecentric_problem
    handler = SelfBundleHandler(camset=cams, target=target, detection=detection)
    assert handler.bundlePrimitive.n_extr == 3

    model = handler.bundlePrimitive.return_bundle_primitives(
        _self_params(handler, cams, poses, target))
    _, extr, _, _ = handler.apply_gauge_transform(*model)

    assert extr.shape == (len(cams), 3)
    assert np.all(np.isfinite(extr))


def test_get_camset_returns_a_telecentric_camset(telecentric_problem):
    """The solved rig reads back as telecentric cameras, and the input is untouched."""
    cams, target, detection, poses = telecentric_problem
    handler = SelfBundleHandler(camset=cams, target=target, detection=detection)
    wanted = {name: np.array(cams[name].intrinsic, dtype=float)
              for name in cams.get_names()}

    out = handler.get_camset(_self_params(handler, cams, poses, target))

    assert out.get_n_cams() == len(cams)
    for name in cams.get_names():
        assert isinstance(out[name], TelecentricCamera)
        got = np.asarray(out[name].intrinsic, dtype=float)
        assert got[0, 0] == pytest.approx(wanted[name][0, 0])
        assert got[1, 1] == pytest.approx(wanted[name][1, 1])


def test_get_updated_target_returns_the_recovered_shape(telecentric_problem):
    cams, target, detection, poses = telecentric_problem
    handler = SelfBundleHandler(camset=cams, target=target, detection=detection)

    updated = handler.get_updated_target(_self_params(handler, cams, poses, target))

    model = np.asarray(target.point_data, dtype=float).reshape(-1, 3)
    assert np.asarray(updated).shape == model.shape
    assert np.allclose(updated, model, atol=1e-9)


def test_a_saved_telecentric_camset_reloads_with_its_handler(tmp_path, telecentric_problem):
    from pyCamSet.utils.saving import load_CameraSet

    cams, target, detection, poses = telecentric_problem
    handler = SelfBundleHandler(camset=cams, target=target, detection=detection)
    out = handler.get_camset(_self_params(handler, cams, poses, target))

    written = tmp_path / "telecentric.camset"
    out.save(written)
    reloaded = load_CameraSet(written)

    assert reloaded.get_n_cams() == len(cams)
    assert all(isinstance(reloaded[n], TelecentricCamera) for n in cams.get_names())


def test_the_gauge_rescales_the_lens_when_the_camera_has_no_translation(
        telecentric_problem):
    """Scaling the world by s is undone by m -> m/s and eps -> eps/s.

    A telecentric pixel is ``u = m*x/(1 + eps*z) + c``, and its extrinsic has no
    translation to carry the scale, so the gauge puts it in the lens.
    """
    s0 = 1.37
    cams, target, detection, poses = telecentric_problem
    handler = SelfBundleHandler(camset=cams, target=target, detection=detection)
    handler.missing_poses = np.zeros(len(poses), dtype=bool)

    before = handler.bundlePrimitive.return_bundle_primitives(
        _self_params(handler, cams, poses, target, point_scale=s0))
    proj_in, extr_in = (np.asarray(a, dtype=float).copy() for a in before[:2])

    proj_out, extr_out, _, _ = handler.apply_gauge_transform(*before)
    proj_out = np.asarray(proj_out, dtype=float)
    extr_out = np.asarray(extr_out, dtype=float)

    ratios = proj_in[:, 0] / proj_out[:, 0]
    assert not np.allclose(ratios, 1.0)
    assert np.allclose(ratios, ratios[0])
    assert np.allclose(proj_out[:, 1], proj_in[:, 1])
    assert np.allclose(proj_out[:, 3], proj_in[:, 3])
    assert np.allclose(proj_out[:, 4], proj_in[:, 4])
    assert np.allclose(proj_out[:, 2], proj_in[:, 2] / ratios)
    assert np.allclose(proj_out[:, 5], proj_in[:, 5] / ratios)
    # the points held for the gauge do not carry point_scale, so the rigid fit
    # picks up a small rotation alongside the scale
    assert np.allclose(extr_out, extr_in, atol=0.02)


def test_the_gauge_returns_unobserved_points_at_the_model(telecentric_problem):
    """A point no camera observes comes back at the model's coordinates,
    whatever the solve left in it."""
    cams, target, detection, poses = telecentric_problem
    model = np.asarray(target.point_data, dtype=float).reshape(-1, 3)

    probe = SelfBundleHandler(camset=cams, target=target, detection=detection)
    pinned_points = np.logical_not(np.logical_or.reduce(
        np.asarray(probe.feat_unfixed, dtype=bool).reshape(-1, 3), axis=1))
    assert pinned_points.any()

    # drop the pinned points from the detections, so they end up unobserved
    keep = set(np.flatnonzero(np.logical_not(pinned_points)).tolist())
    rows = detection.get_data()
    trimmed = TargetDetection(
        cam_names=cams.get_names(),
        data=rows[[int(r[2]) in keep for r in rows]])

    handler = SelfBundleHandler(camset=cams, target=target, detection=trimmed)
    handler.missing_poses = np.zeros(len(poses), dtype=bool)
    unobserved = np.logical_not(np.asarray(handler.visible_feature_mask, dtype=bool))
    assert unobserved.any()
    estimated = np.asarray(handler.feat_unfixed, dtype=bool)

    base = handler.bundlePrimitive.return_bundle_primitives(
        np.concatenate([ground_truth_params(handler, cams, poses),
                        model.reshape(-1)[estimated]]))
    proj, extr, target_poses, _ = (np.asarray(a, dtype=float).copy() for a in base)

    s0 = 1.37
    proj[:, [0, 2, 5]] /= s0
    target_poses[:, 3:] *= s0
    given = np.where(estimated, model.reshape(-1) * s0, model.reshape(-1))
    held = np.flatnonzero(np.logical_not(estimated))
    given[held] += 0.0011

    after = np.asarray(handler.apply_gauge_transform(
        proj, extr, target_poses, given.reshape(-1, 3))[3], dtype=float)

    assert not np.allclose(after, given.reshape(-1, 3))
    assert np.allclose(after[unobserved], model[unobserved], atol=1e-15)


def _telecentric_pixels(proj, extr, poses, points):
    """Every (camera, pose, point) pixel, straight from the block's formula."""
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
    """The gauge may move every parameter but no pixel.

    The points are scaled, turned and moved off the model so the gauge has work
    to do.
    """
    from pyCamSet.utils.general_utils import ext_4x4_to_rod

    cams, target, detection, poses = telecentric_problem
    handler = SelfBundleHandler(camset=cams, target=target, detection=detection,
                                options={"outliers": "n"})

    proj = np.array([cam.to_param_vector() for cam in cams], dtype=float)
    extr = np.array([np.asarray(ext_4x4_to_rod(cam.extrinsic)[0], dtype=float)
                     for cam in cams], dtype=float)
    assert extr.shape[1] == 3

    pose_block = np.array(
        [np.concatenate(ext_4x4_to_rod(p)) for p in poses], dtype=float)
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
    assert np.max(np.linalg.norm(after - before, axis=1)) < 1e-6
    assert not np.allclose(np.asarray(new_proj)[:, 0], proj[:, 0])
    assert np.allclose(np.asarray(new_proj)[:, 1], proj[:, 1])


@pytest.mark.parametrize("declared,spacing", [
    (2.0, 2.0),
    (0.02, 0.02),
    (10.0, 0.01),
    (0.020, 0.0195),
])
def test_the_gauge_reads_the_square_size_from_point_data(declared, spacing):
    """The spacing is measured from the points, whatever the declaration says."""
    from types import SimpleNamespace

    from pyCamSet.optimisation.standard_bundle_handler import _gauge_square_size

    grid = np.array([[x, y, 0.0] for y in range(4) for x in range(4)], dtype=float) * spacing
    target = SimpleNamespace(square_size=declared, point_data=grid)
    assert _gauge_square_size(target) == pytest.approx(spacing)


def test_the_gauge_spacing_ignores_pairs_across_faces():
    """Two faces whose nearest corners sit closer than a square still read one square."""
    from types import SimpleNamespace

    from pyCamSet.optimisation.standard_bundle_handler import _gauge_square_size

    face = np.array([[x, y, 0.0] for y in range(5) for x in range(5)], dtype=float) * 0.01
    # the second face starts beside the first one's last corner
    other = np.array([[0.0405 + 0.01 * x, 0.04 - 0.01 * y, 0.002]
                      for y in range(5) for x in range(5)])
    target = SimpleNamespace(square_size=0.01, point_data=np.vstack([face, other]))
    assert _gauge_square_size(target) == pytest.approx(0.01)


# --------------------------------------------------------------------------
# Tests carried over from isolated/education/integration.  Where a test
# shares a name with one of ours and differs, the education version keeps
# the ``_education_line`` suffix; ours keeps the original name.
# --------------------------------------------------------------------------


def test_get_camset_returns_a_telecentric_camset_education_line(telecentric_problem):
    """What Phase 3 and Phase 4 call the moment a solve finishes.

    The failure being guarded is not a wrong number: it is a ValueError raised
    after the optimisation has already converged, which costs the entire run.
    """
    cams, target, detection, poses = telecentric_problem
    # SelfBundleHandler, deliberately: TemplateBundleHandler.get_camset does not
    # call the gauge transform, which is exactly why Phase 3 survives this bug.
    # Asserting against that class would have passed before the fix too.
    handler = SelfBundleHandler(camset=cams, target=target, detection=detection)

    # SelfBundleHandler's vector carries the free points too (see _self_params).
    params = _self_params(handler, cams, poses, target)
    out = handler.get_camset(params)

    assert out.get_n_cams() == len(cams)
    for name in cams.get_names():
        assert isinstance(out[name], TelecentricCamera)
        got = np.asarray(out[name].intrinsic, dtype=float)
        want = np.asarray(cams[name].intrinsic, dtype=float)
        assert got[0, 0] == pytest.approx(want[0, 0])
        assert got[1, 1] == pytest.approx(want[1, 1])


def test_get_updated_target_returns_the_recovered_shape_education_line(telecentric_problem):
    """The other read-back Phase 4's diagnostics depend on."""
    cams, target, detection, poses = telecentric_problem
    handler = SelfBundleHandler(camset=cams, target=target, detection=detection)

    params = ground_truth_params(handler, cams, poses)
    updated = handler.get_updated_target(params)

    n_points = int(np.prod(target.point_data.shape[:2]))
    assert np.asarray(updated).shape == (n_points, 3)
    assert np.all(np.isfinite(updated))


def test_a_saved_telecentric_camset_reloads_with_its_handler_education_line(tmp_path, telecentric_problem):
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


def test_the_gauge_rescales_the_lens_when_the_camera_has_no_translation_education_line(
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

    # the rig itself has not turned; only the world's frame moved.
    #
    # This is not exact: two of this handler's sixteen points are held at the
    # model's coordinates to pin the gauge (find_gauge_points spreads them
    # apart for a well-conditioned solve, here the two opposite corners of the
    # grid), and they do not carry point_scale with the rest. The rigid fit
    # against the target is a least-squares match over all sixteen, so those
    # two un-scaled points pull in a small spurious rotation alongside the
    # genuine scale change -- a real solve would show the same thing. The
    # tolerance below is generous against that expected residual (~0.011 rad
    # measured here) while still catching a rotation of the size a real
    # gauge defect would produce.
    assert np.allclose(extr_out, extr_in, atol=0.02), (
        "the gauge moved the camera rotations by more than the held gauge "
        "points can explain; a world scale is not a rotation of the rig")


def test_the_gauge_transform_leaves_a_telecentric_projection_alone_education_line(
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


def _offset_rig(k=0.0, eps=0.0):
    """
    A telecentric rig whose seed puts each camera off the world origin in plane.

    The cameras that generated the detections carry the offset in their
    translation, as the rig estimate reports it; the camera set handed to the
    handler is what Phase 2 produces: translation zero, principal point at
    the image centre.
    """
    rotations = ((0.0, 0.0, 0.0), (0.0, 0.45, 0.0), (-0.40, 0.0, 0.25))
    offsets = ((0.0021, -0.0013, 0.0004), (-0.0034, 0.0019, 0.0), (0.0009, 0.0027, -0.002))
    truth, handed = {}, {}
    for idx, (rot, offset) in enumerate(zip(rotations, offsets)):
        name = f"cam_{idx}"
        cam = make_telecentric_camera(name, rotation=rot, k=k, eps=eps)
        extrinsic = cam.extrinsic.copy()
        extrinsic[:3, 3] = offset
        truth[name] = TelecentricCamera(
            extrinsic=extrinsic, intrinsic=cam.intrinsic, res=cam.res,
            distortion_coefs=cam.distortion_coefs, telecentricity=eps, name=name)
        handed[name] = make_telecentric_camera(name, k=k, eps=eps)
    truth, handed = CameraSet(camera_dict=truth), CameraSet(camera_dict=handed)
    target = GridTarget()
    poses = _target_poses()
    points = target.point_data.reshape(-1, 3)
    detection = TargetDetection(cam_names=truth.get_names())
    for im_num, pose in enumerate(poses):
        for name in truth.get_names():
            detection.add_detection(name, im_num, ImageDetection(
                keys=np.arange(len(points)),
                image_points=truth[name].project_points(h_tform(points, pose))))
    return truth, handed, target, detection, np.array(poses)


def _seeded_start(monkeypatch, truth, handed, target, detection, poses, **handler_kwargs):
    """The handler's starting residuals when the seed is the exact rig."""
    from pyCamSet.optimisation import template_handler

    extrinsics = np.array([truth[name].extrinsic for name in truth.get_names()])
    monkeypatch.setattr(template_handler, "estimate_camera_relative_poses",
                        lambda **kwargs: (extrinsics, poses, np.zeros(len(poses))))
    handler = TemplateBundleHandler(camset=handed, target=target, detection=detection,
                                    options={"outliers": "n"}, **handler_kwargs)
    params = handler.get_initial_params()
    residuals = np.reshape(handler.make_loss_fun(threads=1)(params), (-1, 2))
    return handler, params, np.linalg.norm(residuals, axis=1)


def test_the_seeds_in_plane_offset_reaches_the_principal_point(monkeypatch):
    """
    The solver starts where the seed was scored, not m * t pixels away from it.

    A rotation-only extrinsic has no slot for the seed's translation, so it
    used to be dropped: a real telecentric rig scored its seed at 34.9 px and
    started the solve at 121.9 px. The model keeps the in-plane position in
    the principal point, so that is where the offset goes.
    """
    truth, handed, target, detection, poses = _offset_rig()
    handler, params, error = _seeded_start(monkeypatch, truth, handed, target, detection, poses)
    assert np.max(error) < 1e-6, f"the start is {np.max(error):.3g} px off the seed"

    n_intr = handler.bundlePrimitive.n_intr
    for idx, name in enumerate(truth.get_names()):
        intr = params[idx * n_intr:(idx + 1) * n_intr]
        moved = handed[name].principal_point + REF_MAGNIFICATION * truth[name].extrinsic[:2, 3]
        assert np.allclose(intr[[1, 3]], moved, atol=1e-9), name
        # the camera set the handler was given is not edited
        assert np.allclose(handed[name].principal_point, np.array(REF_RES) / 2), name


def test_a_distorted_seed_offset_is_carried_far_closer_than_dropped(monkeypatch):
    """
    With k and eps set the move is exact at the world origin only.

    Elsewhere the distortion and the depth term differ at second order, which
    is still a small fraction of the first order m * t that dropping it costs.
    """
    from pyCamSet.optimisation import template_handler

    truth, handed, target, detection, poses = _offset_rig(k=REF_K, eps=REF_EPS)
    _, _, carried = _seeded_start(monkeypatch, truth, handed, target, detection, poses)
    monkeypatch.setattr(template_handler, "_seed_offset_in_principal_point", lambda cam, _: cam)
    _, _, dropped = _seeded_start(monkeypatch, truth, handed, target, detection, poses)
    assert np.mean(dropped) > 50.0
    assert np.mean(carried) < 0.1 * np.mean(dropped)


def test_fixed_intrinsics_keep_the_camera_as_given(monkeypatch):
    """With nowhere to put the offset, the intrinsics stay exactly as fixed."""
    truth, handed, target, detection, poses = _offset_rig()
    fixed = {"cam_1": {"int": handed["cam_1"].to_param_vector()}}
    handler, params, _ = _seeded_start(monkeypatch, truth, handed, target, detection, poses,
                                       fixed_params=fixed)
    n_intr = handler.bundlePrimitive.n_intr
    assert handler.bundlePrimitive.free_intr == 2
    # cam_0 and cam_2 moved; cam_1's fixed intrinsics are not in the vector at all
    for slot, name in enumerate(("cam_0", "cam_2")):
        intr = params[slot * n_intr:(slot + 1) * n_intr]
        assert not np.allclose(intr[[1, 3]], handed[name].principal_point), name
    assert np.allclose(handler.bundlePrimitive.intr[1], handed["cam_1"].to_param_vector())


def test_a_pinhole_seed_is_handed_over_unchanged(monkeypatch):
    """A pinhole extrinsic carries its own translation; its intrinsics are copied as they are."""
    from pyCamSet.optimisation import template_handler

    _, _, target, _, poses = _offset_rig()
    cams = CameraSet(camera_dict={
        f"cam_{i}": Camera(res=REF_RES, name=f"cam_{i}") for i in range(2)})
    points = target.point_data.reshape(-1, 3)
    extrinsics = np.array([make_4x4h_tform(np.array([0.0, 0.1 * i, 0.0]),
                                           np.array([0.01 * i, -0.02, 0.5])) for i in range(2)])
    detection = TargetDetection(cam_names=cams.get_names())
    for im_num, pose in enumerate(poses):
        for i, name in enumerate(cams.get_names()):
            posed = Camera(extrinsic=extrinsics[i], intrinsic=cams[name].intrinsic,
                           res=REF_RES, name=name)
            detection.add_detection(name, im_num, ImageDetection(
                keys=np.arange(len(points)), image_points=posed.project_points(h_tform(points, pose))))
    monkeypatch.setattr(template_handler, "estimate_camera_relative_poses",
                        lambda **kwargs: (extrinsics, poses, np.zeros(len(poses))))
    called = []
    monkeypatch.setattr(template_handler, "_seed_offset_in_principal_point",
                        lambda *a: called.append(a))
    handler = TemplateBundleHandler(camset=cams, target=target, detection=detection,
                                    options={"outliers": "n"})
    params = handler.get_initial_params()
    assert called == []
    n_intr = handler.bundlePrimitive.n_intr
    for i, name in enumerate(cams.get_names()):
        assert np.array_equal(params[i * n_intr:(i + 1) * n_intr], cams[name].to_param_vector())


def _rotation_only_seed(monkeypatch, cams, poses):
    """Seed with each camera's rotation, as the affine seed does once the offset is in the principal point."""
    from pyCamSet.optimisation import template_handler

    extrinsics = np.array([cams[name].extrinsic for name in cams.get_names()])
    extrinsics[:, :3, 3] = 0.0
    monkeypatch.setattr(template_handler, "estimate_camera_relative_poses",
                        lambda **kwargs: (extrinsics, np.array(poses), np.zeros(len(poses))))


def _saved_camera_residuals(handler, params, floc):
    """
    The residuals of what get_camset returns, saved and loaded, at the solver's own poses.

    Laid out as the kernel's: one row per flattened detection, projected minus detected.
    """
    from pyCamSet.utils.saving import load_CameraSet

    handler.get_camset(params).save(floc)
    loaded = load_CameraSet(floc)
    _, _, pose_params = handler.bundlePrimitive.return_bundle_primitives(params)
    points = handler.target.point_data.reshape(-1, 3)
    rows = handler._flat_detections()
    projected = np.array([
        loaded[handler.cam_names[int(cam)]].project_points(h_tform(
            points[int(key)][None], make_4x4h_tform(pose_params[int(im)][:3],
                                                    pose_params[int(im)][3:])))[0]
        for cam, im, key in rows[:, :3]])
    return loaded, projected - rows[:, -2:]


@pytest.mark.parametrize("k, eps", [(0.0, 0.0), (REF_K, REF_EPS)], ids=["perfect", "distorted"])
def test_a_translated_camera_set_saves_the_cameras_the_solver_fitted(monkeypatch, tmp_path, k, eps):
    """
    The kernel projects a telecentric camera with no translation, so the saved one has none.

    get_camset used to keep the translation of the camera set it was handed. Phase 2
    hands over none, but a camera set that arrived with one saved cameras that
    projected m * t away from the solve: 68-165 px at a real rig's offsets.
    """
    truth, _, target, detection, poses = _offset_rig(k=k, eps=eps)
    given = {name: truth[name].extrinsic.copy() for name in truth.get_names()}
    _rotation_only_seed(monkeypatch, truth, poses)
    handler = TemplateBundleHandler(camset=truth, target=target, detection=detection,
                                    options={"outliers": "n"})
    params = handler.get_initial_params()
    solver = np.reshape(handler.make_loss_fun(threads=1)(params), (-1, 2))

    loaded, saved = _saved_camera_residuals(handler, params, tmp_path / "rig.camset")
    assert np.max(np.abs(saved - solver)) < 1e-6, \
        f"the saved cameras project {np.max(np.abs(saved - solver)):.3g} px from the solver's"
    for name in truth.get_names():
        assert np.array_equal(loaded[name].extrinsic[:3, 3], np.zeros(3)), name
        # the camera set the handler was given is not edited
        assert np.array_equal(truth[name].extrinsic, given[name]), name


def test_a_translated_camera_set_starts_where_it_was_given(monkeypatch):
    """
    For k = eps = 0 the translation moves into the principal point exactly.

    The handler's cameras project as the given ones did, so the solve starts at
    zero error on detections those cameras made.
    """
    truth, _, target, detection, poses = _offset_rig()
    _rotation_only_seed(monkeypatch, truth, poses)
    handler = TemplateBundleHandler(camset=truth, target=target, detection=detection,
                                    options={"outliers": "n"})
    params = handler.get_initial_params()
    error = np.linalg.norm(np.reshape(handler.make_loss_fun(threads=1)(params), (-1, 2)), axis=1)
    assert np.max(error) < 1e-6, f"the start is {np.max(error):.3g} px off the given cameras"
    for name in truth.get_names():
        cam = handler.camset[name]
        assert np.array_equal(cam.extrinsic[:3, 3], np.zeros(3)), name
        moved = truth[name].principal_point + REF_MAGNIFICATION * truth[name].extrinsic[:2, 3]
        assert np.allclose(cam.principal_point, moved, atol=1e-9), name


def test_a_translated_camera_set_round_trips_through_a_solve(monkeypatch, tmp_path):
    """After a solve the cameras run_bundle_adjustment returns, saved and loaded, are the fitted ones."""
    truth, _, target, detection, poses = _offset_rig(k=REF_K, eps=REF_EPS)
    _rotation_only_seed(monkeypatch, truth, poses)
    handler = TemplateBundleHandler(camset=truth, target=target, detection=detection,
                                    options={"outliers": "n", "max_nfev": 5})
    handler.get_initial_params()
    optimisation, final_cams = run_bundle_adjustment(param_handler=handler, threads=1)

    solver = np.reshape(handler.make_loss_fun(threads=1)(optimisation.x), (-1, 2))
    _, saved = _saved_camera_residuals(handler, optimisation.x, tmp_path / "rig.camset")
    assert np.max(np.abs(saved - solver)) < 1e-6
    for name in truth.get_names():
        assert np.array_equal(final_cams[name].extrinsic[:3, 3], np.zeros(3)), name
