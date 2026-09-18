"""
Seeding a pinhole calibration in closed form, without fitting distortion.

``cv2.calibrateCamera`` seeds the intrinsics and the lens distortion together,
by iteration.  The bundle adjustment that follows solves for both anyway --
``projection`` carries nine parameters per camera, four intrinsic and five
Brown-Conrady -- so the distortion the seed fits is thrown away and refit a few
seconds later.

What the bundle adjustment cannot do is start from nowhere, and Zhang's method
is enough to start it: a homography per board view, the image of the absolute
conic from those, and the intrinsic matrix out of that by a Cholesky
factorisation.  It is closed form, it fits no distortion at all, and the camera
it produces is handed on with its distortion at zero.

The price is that everything here assumes a distortion free camera.  A lens
with real distortion still yields an intrinsic matrix, but a biased one, and
the bias grows with the distortion -- so this seeds a bundle adjustment, and is
not a calibration on its own.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: how far out of its own plane a board may sit, relative to its extent, before
#: it is not a plane and a homography does not describe it
FLATNESS_TOLERANCE = 1e-3

#: how small the conic system's second smallest singular value may get,
#: against its largest, before its null space is wider than the one vector a
#: single intrinsic matrix needs it to be.  Measured on synthetic board views:
#: 0.07 for a well tilted set, and unmoved by noise or distortion; 4e-3 at a
#: 0.1 radian tilt, which is a poor seed but a seed; 2e-4 at 0.02 radians and
#: for views that do not tilt at all, which are not.
AMBIGUITY_TOLERANCE = 1e-3

#: the fewest board views Zhang's method is solved from.  Two suffice with the
#: skew pinned, but the third is what makes the system worth trusting.
MIN_VIEWS = 3


def _require_planar(object_points: np.ndarray) -> None:
    """A homography maps a plane, so the board had better be one."""
    points = np.asarray(object_points, dtype=float).reshape(-1, 3)
    extent = np.ptp(points[:, :2])
    if extent <= 0:
        raise ValueError("A board view has no extent in its own plane")
    if np.max(np.abs(points[:, 2])) > FLATNESS_TOLERANCE * extent:
        raise ValueError(
            "Zhang's method seeds from planar board views, and this board "
            "departs from its own plane by more than "
            f"{FLATNESS_TOLERANCE:.0e} of its extent. Board geometry reaches "
            "this through the target's point_local, which is what flattens "
            "each face; a board that is not flat there is a target geometry "
            "problem, not a calibration one."
        )


def homography_of(object_points: np.ndarray, image_points: np.ndarray) -> np.ndarray:
    """
    The homography taking one board view's plane to its pixels.

    :param object_points: (n, 3) board points, flat in their own z = 0 plane
    :param image_points: (n, 2) their detected pixels
    :return: the 3x3 homography
    """
    obj = np.asarray(object_points, dtype=float).reshape(-1, 3)
    img = np.asarray(image_points, dtype=float).reshape(-1, 2)
    if len(obj) < 4:
        raise ValueError(
            f"A homography needs at least 4 correspondences to fit, got {len(obj)}")
    _require_planar(obj)
    homography, _ = cv2.findHomography(obj[:, :2], img, method=0)
    if homography is None:
        raise ValueError(
            "A board view's homography could not be fit, which usually means "
            "its detections are collinear or repeated")
    return np.asarray(homography, dtype=float)


def _conic_row(homography: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    The row that ``h_i^T B h_j`` contributes, for ``B`` as its six free terms.

    ``B`` is the image of the absolute conic, symmetric, and carried here as
    ``[B11, B12, B22, B13, B23, B33]``.
    """
    h = homography
    return np.array([
        h[0, i] * h[0, j],
        h[0, i] * h[1, j] + h[1, i] * h[0, j],
        h[1, i] * h[1, j],
        h[2, i] * h[0, j] + h[0, i] * h[2, j],
        h[2, i] * h[1, j] + h[1, i] * h[2, j],
        h[2, i] * h[2, j],
    ])


def intrinsic_from_conic(conic: np.ndarray) -> np.ndarray:
    """
    The intrinsic matrix whose absolute conic is *conic*.

    ``B = K^-T K^-1`` with ``K`` upper triangular, so ``K^-T`` is the lower
    triangular Cholesky factor of ``B`` and there is nothing else to solve.
    Zhang's paper extracts the same matrix term by term; this is the same
    answer without the five expressions that can each go negative under the
    square root.

    :param conic: ``[B11, B12, B22, B13, B23, B33]``
    :return: the 3x3 intrinsic matrix, normalised to 1 in its corner
    """
    b = np.asarray(conic, dtype=float)
    matrix = np.array([[b[0], b[1], b[3]],
                       [b[1], b[2], b[4]],
                       [b[3], b[4], b[5]]])
    # the conic is a null vector, so its sign is not determined; only one of
    # the two is a positive definite conic and so an actual camera
    for sign in (1.0, -1.0):
        try:
            factor = np.linalg.cholesky(sign * matrix)
        except np.linalg.LinAlgError:
            continue
        intrinsic = np.linalg.inv(factor.T)
        return intrinsic / intrinsic[2, 2]
    raise ValueError(
        "The estimated absolute conic is not positive definite, so no camera "
        "produces it. Zhang's method needs board views at several distinct "
        "orientations; views that are all square on to the camera, or a "
        "strongly distorting lens, both land here."
    )


def _unit_columns(homography: np.ndarray) -> np.ndarray:
    """
    A homography's first two columns brought to unit length, together.

    Only those two columns reach the conic rows, and both constraints they
    carry are quadratic in them, so scaling the pair by one factor leaves the
    conic untouched.  What it does change is the system that solves for it:
    with object points in millimetres those columns run a thousandth of the
    third, which spreads the six columns of that system over six orders of
    magnitude and buries the smallest singular value -- the one the
    conditioning below is read from -- in the scaling rather than the geometry.

    :param homography: the 3x3 homography of one board view
    :return: the same homography, with its first two columns rescaled
    """
    scaled = np.asarray(homography, dtype=float).copy()
    lengths = np.linalg.norm(scaled[:, :2], axis=0)
    if not np.all(lengths > 0):
        raise ValueError("A board view's homography is degenerate")
    scaled[:, :2] *= 2.0 / np.sum(lengths)
    return scaled


def intrinsics_from_homographies(homographies: list[np.ndarray]) -> np.ndarray:
    """
    The intrinsic matrix consistent with a set of board view homographies.

    Each view contributes the two constraints a rotation implies -- its first
    two columns are orthogonal, and equally long -- and a third row pins the
    skew at zero, which is a statement about the sensor rather than about the
    views.

    :param homographies: the 3x3 homography of each board view
    :return: the 3x3 intrinsic matrix
    """
    if len(homographies) < MIN_VIEWS:
        raise ValueError(
            f"Zhang's method is solved from at least {MIN_VIEWS} board views, "
            f"got {len(homographies)}")

    rows = []
    for homography in homographies:
        scaled = _unit_columns(homography)
        rows.append(_conic_row(scaled, 0, 1))
        rows.append(_conic_row(scaled, 0, 0) - _conic_row(scaled, 1, 1))
    rows.append(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]))  # zero skew

    _, singular, right = np.linalg.svd(np.asarray(rows, dtype=float))
    # the conic is the system's null vector, so it is one answer only while the
    # null space is one dimensional.  Views that repeat a constraint rather than
    # adding one leave the second smallest singular value at zero too, and then
    # every vector in that space is an equally good conic -- which is not a
    # near miss to be reported as a ratio between the two smallest, because
    # both are zero and their ratio is noise.
    conditioning = singular[-2] / singular[0]
    logger.debug("absolute conic solved at a conditioning of %.3g", conditioning)
    if conditioning < AMBIGUITY_TOLERANCE:
        raise ValueError(
            f"The absolute conic is ambiguous ({conditioning:.2g} against a "
            f"{AMBIGUITY_TOLERANCE:.0e} tolerance): these board views do not "
            "pin one intrinsic matrix between them. Views at a single "
            "orientation are the usual cause -- every one of them contributes "
            "the same constraint, however far the board is moved, so Zhang's "
            "method needs the board tilted several different ways."
        )

    intrinsic = intrinsic_from_conic(right[-1])
    intrinsic[0, 1] = 0.0
    return intrinsic


def focals_at_principal(homographies: list[np.ndarray],
                        principal: np.ndarray) -> np.ndarray:
    """
    The focal lengths alone, for a principal point that is not being solved for.

    The closed form's principal point is the unstable half of it, and pinning
    the principal point leaves the same two rotation constraints per view
    linear in ``1/fx^2`` and ``1/fy^2`` -- which is what OpenCV's
    ``initCameraMatrix2D`` fits, and what this falls back on.

    :param homographies: the 3x3 homography of each board view
    :param principal: the principal point to hold, in pixels
    :return: (fx, fy)
    """
    rows, values = [], []
    for homography in homographies:
        centred = np.asarray(homography, dtype=float).copy()
        centred[0] -= principal[0] * centred[2]
        centred[1] -= principal[1] * centred[2]
        h1, h2 = centred[:, 0], centred[:, 1]
        rows.append([h1[0] * h2[0], h1[1] * h2[1]])
        values.append(-h1[2] * h2[2])
        rows.append([h1[0] ** 2 - h2[0] ** 2, h1[1] ** 2 - h2[1] ** 2])
        values.append(-(h1[2] ** 2 - h2[2] ** 2))

    inverse_squares, *_ = np.linalg.lstsq(
        np.asarray(rows), np.asarray(values), rcond=None)
    if np.any(inverse_squares <= 0):
        raise ValueError(
            "No positive focal length fits these board views with the "
            "principal point held at the image centre")
    return 1.0 / np.sqrt(inverse_squares)


def _nearest_rotation(columns: np.ndarray) -> np.ndarray:
    """The closest rotation to a noisy 3x3, by orthogonal Procrustes."""
    u, _, vt = np.linalg.svd(np.asarray(columns, dtype=float))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def pose_from_homography(intrinsic: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """
    The board pose a homography implies, for known optics.

    ``K^-1 H`` is the first two columns of a rotation and a translation, up to
    one scale, and the columns of a rotation have unit length -- which fixes
    the scale, and then the third column is a cross product.

    :param intrinsic: the camera's 3x3 intrinsic matrix
    :param homography: the board view's 3x3 homography
    :return: the 4x4 board pose, in camera coordinates
    """
    metric = np.linalg.solve(np.asarray(intrinsic, dtype=float),
                             np.asarray(homography, dtype=float))
    lengths = np.linalg.norm(metric[:, :2], axis=0)
    if not np.all(lengths > 0):
        raise ValueError("A board view's homography is degenerate")
    # both columns estimate the same scale, so average them, and take the sign
    # that puts the board in front of the camera rather than behind it
    scale = 2.0 / np.sum(lengths)
    if metric[2, 2] < 0:
        scale = -scale

    r1, r2, translation = scale * metric[:, 0], scale * metric[:, 1], scale * metric[:, 2]
    pose = np.eye(4)
    pose[:3, :3] = _nearest_rotation(np.stack([r1, r2, np.cross(r1, r2)], axis=1))
    pose[:3, 3] = translation
    return pose


def reprojection_rms(intrinsic: np.ndarray, pose: np.ndarray,
                     object_points: np.ndarray, image_points: np.ndarray) -> float:
    """
    What a pose and an intrinsic matrix leave behind, with no distortion.

    :param intrinsic: the camera's 3x3 intrinsic matrix
    :param pose: the 4x4 board pose
    :param object_points: (n, 3) board points
    :param image_points: (n, 2) their detected pixels
    :return: the rms reprojection error in pixels
    """
    obj = np.asarray(object_points, dtype=float).reshape(-1, 3)
    img = np.asarray(image_points, dtype=float).reshape(-1, 2)
    camera_frame = obj @ pose[:3, :3].T + pose[:3, 3]
    projected = camera_frame @ np.asarray(intrinsic, dtype=float).T
    projected = projected[:, :2] / projected[:, 2:]
    return float(np.sqrt(np.mean(np.sum((projected - img) ** 2, axis=1))))


def calibrate_zhang(object_points: list[np.ndarray],
                    image_points: list[np.ndarray],
                    res: list[int],
                    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """
    A pinhole camera's intrinsics and per view poses, from correspondences.

    No distortion is fit, and none is assumed: the camera this seeds is handed
    to the bundle adjustment with its distortion at zero, for the bundle
    adjustment to solve along with everything else.

    The image points are normalised to the sensor before the absolute conic is
    solved and the result is carried back afterwards.  In pixels the conic's
    terms span six orders of magnitude and the system that solves for it is
    badly conditioned; over a unit sensor they do not.

    :param object_points: per view, (n, 3) board points flat in their own plane
    :param image_points: per view, (n, 2) detected pixels
    :param res: the camera resolution, ``(height, width)``, as detection
        reports it and as ``initial_calibration`` passes it on
    :return: the intrinsic matrix, per view 4x4 poses, and per view rms. The
        poses come straight out of the closed form, and the rms with them is
        what this model leaves behind rather than what the camera can reach:
        the projection onto the rotations discards whatever the intrinsics'
        distortion bias was being absorbed by, and only a pose solved against
        the pixels takes it back. That solve is the bundle adjustment's.
    """
    if len(object_points) < MIN_VIEWS:
        raise ValueError(
            f"Zhang's method is solved from at least {MIN_VIEWS} board views, "
            f"got {len(object_points)}")

    homographies = [homography_of(obj, img)
                    for obj, img in zip(object_points, image_points)]

    centre = np.array([res[1], res[0]], dtype=float) / 2
    scale = float(np.max(res))
    to_sensor = np.array([[1 / scale, 0, -centre[0] / scale],
                          [0, 1 / scale, -centre[1] / scale],
                          [0, 0, 1.0]])
    normalised = [to_sensor @ homography for homography in homographies]

    intrinsic = np.linalg.inv(to_sensor) @ intrinsics_from_homographies(normalised)
    intrinsic /= intrinsic[2, 2]

    # the principal point is what the closed form estimates worst, and a wild
    # one is worse than no estimate: the pose bootstrap that consumes this
    # camera drops any image it cannot solve to 20 pixels, and those
    # observations never reach the bundle adjustment to be recovered.
    if not (0 <= intrinsic[0, 2] <= res[1] and 0 <= intrinsic[1, 2] <= res[0]):
        logger.warning(
            "Zhang's closed form put the principal point at "
            "(%.0f, %.0f), outside a %dx%d sensor. Holding it at the image "
            "centre and fitting the focal lengths alone; the bundle "
            "adjustment refines it from there.",
            intrinsic[0, 2], intrinsic[1, 2], res[1], res[0])
        focals = focals_at_principal(homographies, centre)
        intrinsic = np.array([[focals[0], 0, centre[0]],
                              [0, focals[1], centre[1]],
                              [0, 0, 1.0]])

    poses, errors = [], []
    for homography, obj, img in zip(homographies, object_points, image_points):
        pose = pose_from_homography(intrinsic, homography)
        poses.append(pose)
        errors.append(reprojection_rms(intrinsic, pose, obj, img))
    return intrinsic, poses, np.asarray(errors, dtype=float)
