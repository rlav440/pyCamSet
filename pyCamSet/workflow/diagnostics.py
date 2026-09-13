"""
The numbers a phase reports about its own solve.

Everything here reads a finished optimisation and returns plain Python, so a
run record stays JSON and the figures drawn from it are drawn somewhere else.
"""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from pyCamSet.calibration_targets.abstract_target import get_keys

from pyCamSet.workflow.logs import LogFn, discard

logger = logging.getLogger(__name__)


def observation_residual_xy(residuals: Any, handler: Any) -> np.ndarray:
    """
    Return only the two-scalar-per-observation reprojection residuals.

    The optimisation backend may append one-dimensional lockbox-prior
    residuals after the reprojection ones.  Reshaping those into pixel pairs
    crashes on an odd camera count and quietly contaminates the per-camera
    reprojection metric on an even one.

    :param residuals: the solver's residual vector
    :param handler: the bundle handler that produced it
    :return: an ``(n, 2)`` array of per-observation residuals
    """
    values = np.asarray(residuals, dtype=float).reshape(-1)
    base_count = int(getattr(handler, "get_base_residual_count", lambda: 0)())
    if base_count <= 0:
        base_count = values.size
    if base_count > values.size or base_count % 2:
        raise ValueError(
            "Invalid reprojection residual segment length: "
            f"base_count={base_count}, total_count={values.size}"
        )
    return values[:base_count].reshape(-1, 2)


def per_camera_mean_reprojection(
    optimisation: Any,
    handler: Any,
    log: LogFn = discard,
    label: str = "per-camera reprojection",
) -> tuple[dict[str, float], np.ndarray]:
    """
    Mean reprojection error per camera, and the residual scatter behind it.

    A diagnostic must not fail a solve that has already succeeded, so a
    handler whose detection data does not line up with its residuals is
    reported and skipped rather than raised.

    :param optimisation: the finished ``OptimizeResult``
    :param handler: the bundle handler that was solved
    :param log: what to call with each line of output
    :param label: what to call this diagnostic if it has to be skipped
    :return: the per-camera means, and the ``(n, 2)`` residual scatter
    """
    try:
        detection_data = np.asarray(handler.get_detection_data(flatten=True))
        residual_xy = observation_residual_xy(optimisation.fun, handler)
    except Exception as exc:
        log(f"Warning: {label} skipped due to diagnostics error: {exc}")
        return {}, np.empty((0, 2), dtype=float)

    if detection_data.ndim != 2 or detection_data.shape[1] < 1:
        log(f"Warning: {label} skipped (unexpected detection-data shape).")
        return {}, residual_xy

    residual_norm = np.linalg.norm(residual_xy, axis=1)
    cam_index = detection_data[:, 0].astype(int)
    if cam_index.size != residual_norm.size:
        log(
            f"Warning: {label} skipped due to diagnostics error: "
            f"alignment mismatch: cam_idx={cam_index.size}, "
            f"residuals={residual_norm.size}"
        )
        return {}, residual_xy

    per_camera: dict[str, float] = {}
    for index, name in enumerate(getattr(handler, "cam_names", [])):
        mask = cam_index == index
        per_camera[name] = (
            float(np.mean(residual_norm[mask])) if np.any(mask) else float("nan"))
    return per_camera, residual_xy


def per_view_reprojection(
        detections, calibration_target, cams) -> tuple[dict[str, dict], dict[str, float]]:
    """
    True per-image RMS reprojection error, per camera.

    Each image is posed against the target on its own and its points
    reprojected, rather than reading a number the calibration reported, so an
    image that contributed badly is visible as itself.

    :return: the per-view series per camera, and each camera's pooled RMS
    """
    per_view: dict[str, dict] = {}
    overall_rms: dict[str, float] = {}
    max_ims = int(detections.max_ims)
    pose_failures = 0

    for cam_name in cams.get_names():
        cam = cams[cam_name]
        cam_detection = detections.get(cam=cam_name)
        cam_has_any = cam_detection.has_data()

        image_indices: list[int] = []
        rms_px: list[float] = []
        n_points: list[int] = []
        has_detection: list[bool] = []
        valid_pose: list[bool] = []

        weighted_sq_sum = 0.0
        total_points = 0

        for im_idx in range(max_ims):
            image_indices.append(im_idx)
            if not cam_has_any:
                rms_px.append(float("nan"))
                n_points.append(0)
                has_detection.append(False)
                valid_pose.append(False)
                continue

            im_detect = cam_detection.get(global_im_num=im_idx)
            data = im_detect.get_data()
            if data is None or len(data) == 0:
                rms_px.append(float("nan"))
                n_points.append(0)
                has_detection.append(False)
                valid_pose.append(False)
                continue

            has_detection.append(True)
            n_points.append(int(data.shape[0]))

            try:
                pose = calibration_target.target_pose_in_cam_image(
                    im_detect, cam, mode="nan")
            except Exception as exc:
                # A view with no pose is a NaN by design.  A target that
                # raised is also a NaN, but it is not the same thing, so it
                # is counted and said out loud once at the end.
                logger.debug(
                    "target_pose_in_cam_image raised for cam=%s im=%d: %s",
                    cam_name, im_idx, exc)
                pose_failures += 1
                pose = np.ones((4, 4), dtype=float) * np.nan

            pose_arr = np.asarray(pose, dtype=float)
            if pose_arr.shape != (4, 4) or np.any(np.isnan(pose_arr)):
                rms_px.append(float("nan"))
                valid_pose.append(False)
                continue

            valid_pose.append(True)
            keys = get_keys(data).astype(int)
            object_points = np.asarray(
                calibration_target.point_data[tuple(keys.T)],
                dtype=np.float32).reshape(-1, 3)
            image_points = np.asarray(
                data[:, -2:], dtype=np.float32).reshape(-1, 2)
            rvec, _ = cv2.Rodrigues(pose_arr[:3, :3].astype(np.float64))
            tvec = pose_arr[:3, 3].astype(np.float64)
            projected, _ = cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                np.asarray(cam.intrinsic, dtype=np.float64),
                np.asarray(cam.distortion_coefs, dtype=np.float64).reshape(-1),
            )
            projected = projected.reshape(-1, 2).astype(np.float32)
            sq_err = np.sum((projected - image_points) ** 2, axis=1)
            rms_px.append(
                float(np.sqrt(np.mean(sq_err))) if sq_err.size else float("nan"))
            if sq_err.size:
                weighted_sq_sum += float(np.sum(sq_err))
                total_points += int(sq_err.size)

        per_view[cam_name] = {
            "image_indices": image_indices,
            "rms_px": rms_px,
            "n_points": n_points,
            "has_detection": has_detection,
            "valid_pose": valid_pose,
        }
        overall_rms[cam_name] = (
            float(np.sqrt(weighted_sq_sum / total_points))
            if total_points else float("nan"))

    if pose_failures:
        logger.warning(
            "per_view_reprojection: %d target_pose_in_cam_image call(s) raised "
            "exceptions (converted to NaN poses). Check debug logs for details.",
            pose_failures,
        )

    return per_view, overall_rms
