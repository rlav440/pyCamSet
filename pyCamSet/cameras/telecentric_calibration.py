"""
Seeding a telecentric calibration, where OpenCV cannot.

``cv2.calibrateCamera`` and ``cv2.solvePnP`` both assume a perspective camera,
and ``solvePnP`` additionally filters poses on the sign of their depth, which a
telecentric camera does not have.  With the telecentricity error and the
distortion set aside -- both are small, and the bundle adjustment refines them
-- the model is linear, so the seed is a direct linear transform.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: how flat a point cloud may be, relative to its own extent, before it counts
#: as planar and the pose it implies becomes two-fold ambiguous
PLANARITY_TOLERANCE = 1e-3


def is_planar(object_points: np.ndarray, tolerance: float = PLANARITY_TOLERANCE) -> bool:
    """
    Whether a set of points is flat enough for an affine pose to be ambiguous.

    :param object_points: (n, 3) target geometry
    :param tolerance: smallest singular value, relative to the largest, that
        still counts as out-of-plane extent
    :return: True if the points are coplanar
    """
    centred = np.asarray(object_points, dtype=float).reshape(-1, 3)
    centred = centred - centred.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    if singular[0] <= 0:
        return True
    return bool(singular[2] / singular[0] < tolerance)


def _require_non_planar(object_points: np.ndarray) -> None:
    if is_planar(object_points):
        raise ValueError(
            "A planar target seen through a telecentric lens has a two-fold "
            "pose ambiguity: tilting the plane by +theta and -theta produce "
            "the same image, and an affine camera cannot tell them apart. "
            "Seed a telecentric calibration from a target with out-of-plane "
            "extent, such as a Ccube."
        )


def fit_affine_camera(object_points: np.ndarray, image_points: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """
    Least squares fit of ``uv = A @ P + b`` for one view.

    This is the whole telecentric model once distortion and telecentricity
    error are set aside, and it is linear, so there is nothing to iterate.

    :param object_points: (n, 3) target points
    :param image_points: (n, 2) their detected pixels
    :return: the (2, 3) matrix A and the (2,) offset b
    """
    P = np.asarray(object_points, dtype=float).reshape(-1, 3)
    uv = np.asarray(image_points, dtype=float).reshape(-1, 2)
    if len(P) < 4:
        raise ValueError(
            f"An affine camera needs at least 4 correspondences to fit, got {len(P)}")
    _require_non_planar(P)
    design = np.concatenate([P, np.ones((len(P), 1))], axis=1)
    solution, *_ = np.linalg.lstsq(design, uv, rcond=None)
    return solution[:3].T, solution[3]


def _nearest_rotation_rows(rows: np.ndarray) -> np.ndarray:
    """
    The closest pair of orthonormal rows to a noisy (2, 3) estimate.

    A least squares fit does not know the rows came from a rotation, so this
    projects them back onto the manifold -- the standard orthogonal Procrustes
    solution, restricted to two rows.
    """
    u, _, vt = np.linalg.svd(np.asarray(rows, dtype=float), full_matrices=False)
    return u @ vt


def pose_from_affine(object_points: np.ndarray, image_points: np.ndarray,
                     magnification: np.ndarray, principal_point: np.ndarray,
                     ) -> tuple[np.ndarray, float]:
    """
    The target pose that puts *object_points* at *image_points*, for known optics.

    The depth of the pose is not recoverable -- a telecentric camera cannot see
    how far away anything is -- so the translation along the optical axis is
    reported as zero and left for the bundle adjustment, which recovers it from
    the other cameras in the rig.

    :param object_points: (n, 3) target points
    :param image_points: (n, 2) their detected pixels, already undistorted
    :param magnification: the camera's (m_x, m_y), in pixels per world unit
    :param principal_point: the camera's (c_x, c_y), in pixels
    :return: the 4x4 target pose, and the rms reprojection error in pixels
    """
    P = np.asarray(object_points, dtype=float).reshape(-1, 3)
    uv = np.asarray(image_points, dtype=float).reshape(-1, 2)
    metric = (uv - np.asarray(principal_point)) / np.asarray(magnification)

    A, b = fit_affine_camera(P, metric)
    rows = _nearest_rotation_rows(A)

    pose = np.eye(4)
    pose[:3, :3] = np.stack([rows[0], rows[1], np.cross(rows[0], rows[1])])
    pose[:2, 3] = b

    predicted = (P @ pose[:3, :3][:2].T + b) * np.asarray(magnification) + np.asarray(principal_point)
    rms = float(np.sqrt(np.mean(np.sum((predicted - uv) ** 2, axis=1))))
    return pose, rms


def calibrate_telecentric(object_points: list[np.ndarray],
                          image_points: list[np.ndarray],
                          res: list[int],
                          ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """
    A telecentric camera's intrinsics and per-view poses, from correspondences.

    The principal point is pinned to the image centre rather than fitted.  For
    a telecentric camera it is degenerate with the in-plane translation -- both
    shift every pixel by the same amount -- and only the distortion centre
    separates them, which this seed does not yet estimate.  The bundle
    adjustment refines it once distortion is in the model.

    :param object_points: per view, (n, 3) target points
    :param image_points: per view, (n, 2) detected pixels
    :param res: the camera resolution
    :return: magnification, principal point, per-view 4x4 poses, per-view rms
    """
    if not object_points:
        raise ValueError("A telecentric calibration needs at least one view")

    magnifications = []
    for P, uv in zip(object_points, image_points):
        A, _ = fit_affine_camera(P, uv)
        # A = diag(m) @ R[:2], and the rows of a rotation are unit length
        magnifications.append(np.linalg.norm(A, axis=1))
    magnification = np.median(np.stack(magnifications), axis=0)

    # TODO res reaches initial_calibration as (height, width), so this reads
    # the two the wrong way round and only agrees with itself on a square
    # sensor. Fixing it means auditing every caller: Camera.res is (width,
    # height) once set_resolutions_from_file has run, so the two orders are
    # live in the same codebase.
    principal_point = np.asarray(res, dtype=float) / 2
    poses, errors = [], []
    for P, uv in zip(object_points, image_points):
        pose, rms = pose_from_affine(P, uv, magnification, principal_point)
        poses.append(pose)
        errors.append(rms)

    spread = np.ptp(np.stack(magnifications), axis=0) / magnification
    if np.any(spread > 0.05):
        logger.warning(
            "Per view magnification varies by %.1f%% across the calibration, "
            "which is more than a telecentric lens should. Check that the "
            "detections belong to the camera being calibrated.",
            100 * float(np.max(spread)),
        )
    return magnification, principal_point, poses, np.asarray(errors)
