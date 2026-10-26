"""
A calibration compared against the capture that made its images.
"""

from __future__ import annotations

import cv2
import numpy as np

from .capture import Capture
from .render import readable_pixels


def compare_to_truth(cams, capture: Capture, solved_points: np.ndarray | None = None,
                     extrinsics: bool = True) -> dict:
    """
    How far a calibrated camera set is from the rig that took the images.

    A calibration's world frame is its own choice -- usually the target's pose
    in one shot -- so cameras are compared relative to the first camera:
    each camera's pose in the first camera's frame, truth against estimate.

    :param cams: the calibrated :class:`~pyCamSet.cameras.CameraSet`.
    :param capture: the capture's ground truth.
    :param solved_points: the target's points as a target-refining solve
        (Phase 4) estimated them, shaped like ``point_data``; compared to the
        true, pasted points after the best rigid fit.
    :param extrinsics: compare the cameras' poses too; off for a camera set
        whose poses are not yet solved (Phase 2 calibrates each camera alone).
    :return: a JSON-ready dictionary.  Lens errors are measured only where
        the target's points fell in each camera's images: elsewhere a
        calibration has no data, and its distortion is extrapolation.
    """
    names = list(cams.get_names())
    truth = {camera.name: camera for camera in capture.cameras}
    first_true = truth[names[0]].extrinsic
    first_est = np.asarray(cams[names[0]].extrinsic, dtype=float)

    cameras = {}
    for name in names:
        est, true = cams[name], truth[name]
        k_est = np.asarray(est.intrinsic, float)
        k_true = true.intrinsic
        d_est = np.reshape(np.asarray(est.distortion_coefs, float), -1)
        relative_true = true.extrinsic @ np.linalg.inv(first_true)
        relative_est = np.asarray(est.extrinsic, float) @ np.linalg.inv(first_est)
        delta = relative_est @ np.linalg.inv(relative_true)
        cameras[name] = {
            "focal_error_px": float(np.mean([k_est[0, 0] - k_true[0, 0],
                                             k_est[1, 1] - k_true[1, 1]])),
            "focal_error_pct": float(100.0 * (k_est[0, 0] - k_true[0, 0]) / k_true[0, 0]),
            "principal_point_error_px": float(np.hypot(k_est[0, 2] - k_true[0, 2],
                                                       k_est[1, 2] - k_true[1, 2])),
            "k1_error": float(d_est[0] - true.distortion[0]),
            "k2_error": float(d_est[1] - true.distortion[1]),
            **_lens_error_px(true, k_est, d_est, _observed_pixels(capture, true)),
        }
        if extrinsics:
            cameras[name]["position_error_mm"] = float(1000.0 * np.linalg.norm(
                np.linalg.inv(relative_est)[:3, 3] - np.linalg.inv(relative_true)[:3, 3]))
            cameras[name]["rotation_error_deg"] = float(np.degrees(np.linalg.norm(
                cv2.Rodrigues(delta[:3, :3])[0])))

    report = {"cameras": cameras}
    result = getattr(cams, "calibration_result", None)
    if result is not None:
        residuals = np.linalg.norm(np.reshape(np.asarray(result, float), (-1, 2)), axis=1)
        report["reprojection_rms_px"] = float(np.sqrt(np.mean(residuals ** 2)))
        report["reprojection_median_px"] = float(np.median(residuals))

    nominal = np.asarray(_nominal_points(capture), float).reshape(-1, 3)
    true_points = np.asarray(capture.true_points, float).reshape(-1, 3)
    report["pasting_rms_mm"] = float(1000.0 * _rigid_rms(nominal, true_points))
    if solved_points is not None:
        solved = np.asarray(solved_points, float).reshape(-1, 3)
        report["target_shape_rms_mm"] = float(1000.0 * _rigid_rms(solved, true_points))
    report["summary"] = {
        "worst_focal_error_pct": max(abs(c["focal_error_pct"]) for c in cameras.values()),
        "worst_position_error_mm": max((c["position_error_mm"] for c in cameras.values()),
                                       default=float("nan")) if extrinsics else float("nan"),
        "worst_rotation_error_deg": max((c["rotation_error_deg"] for c in cameras.values()),
                                        default=float("nan")) if extrinsics else float("nan"),
        "worst_lens_rms_px": max(c["lens_rms_px"] for c in cameras.values()),
        "worst_lens_p99_px": max(c["lens_p99_px"] for c in cameras.values()),
        "worst_lens_shape_rms_px": max(c["lens_shape_rms_px"] for c in cameras.values()),
    }
    return report


def _nominal_points(capture: Capture) -> np.ndarray:
    from pyCamSet.calibration_targets.core.target_registry import build_target

    return build_target(capture.target_spec).point_data


def _observed_pixels(capture: Capture, camera, most: int = 4000) -> np.ndarray:
    """
    Where the target's points fell in a camera's images, over the whole
    capture: the points on faces turned readably to the camera, inside the
    frame (the test :func:`~.render.readable_pixels` shares with the capture's
    own visible-point count).
    """
    n_faces = len(capture.errors.rotation_deg)
    seen = [readable_pixels(capture.true_points, n_faces, camera, pose)
            for pose in capture.poses]
    pixels = np.concatenate(seen) if seen else np.zeros((0, 2))
    if len(pixels) > most:
        pixels = pixels[np.linspace(0, len(pixels) - 1, most).astype(int)]
    return pixels


def _lens_error_px(true, k_est, d_est, pixels: np.ndarray) -> dict:
    """
    How far apart the true and estimated lenses put the same rays, in pixels,
    over the image positions the target was seen at -- the error that
    matters, where focal length, principal point and coefficient errors trade
    off against each other.  The rays are each true pixel's, and each is sent
    back through the estimated lens.
    """
    if not len(pixels):
        return {"lens_rms_px": float("nan"), "lens_p99_px": float("nan"),
                "lens_shape_rms_px": float("nan")}
    rays = cv2.undistortPoints(pixels[:, None], true.intrinsic, true.distortion).reshape(-1, 2)
    points = np.c_[rays, np.ones(len(rays))]
    estimate, _ = cv2.projectPoints(points[:, None], np.zeros(3), np.zeros(3), k_est, d_est)
    truth = pixels
    error = np.linalg.norm(estimate.reshape(-1, 2) - truth, axis=1)
    # The same comparison with the best common rotation of the camera taken out first: a
    # tenth of a degree of camera pose moves every ray by a few pixels at these focal
    # lengths, and that is a pose error, which the position and rotation errors report.
    true_rays = np.c_[rays, np.ones(len(rays))]
    est_rays = cv2.undistortPoints(truth[:, None], k_est, d_est).reshape(-1, 2)
    est_rays = np.c_[est_rays, np.ones(len(est_rays))]
    a = est_rays / np.linalg.norm(est_rays, axis=1, keepdims=True)
    b = true_rays / np.linalg.norm(true_rays, axis=1, keepdims=True)
    u, _, vt = np.linalg.svd(a.T @ b)
    rotation = vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    turned = np.arccos(np.clip(np.sum((a @ rotation.T) * b, axis=1), -1.0, 1.0))
    shape = turned * float(true.intrinsic[0, 0])           # radians to pixels
    return {"lens_rms_px": float(np.sqrt(np.mean(error ** 2))),
            "lens_p99_px": float(np.percentile(error, 99)),
            "lens_shape_rms_px": float(np.sqrt(np.mean(shape ** 2)))}


def _rigid_rms(source: np.ndarray, target: np.ndarray) -> float:
    """RMS distance after the best rotation and translation (Kabsch)."""
    a = source - source.mean(axis=0)
    b = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    rotation = vt.T @ flip @ u.T
    return float(np.sqrt(np.mean(np.sum((a @ rotation.T - b) ** 2, axis=1))))
