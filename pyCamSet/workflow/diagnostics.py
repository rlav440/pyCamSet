"""
The numbers a phase reports about its own solve.

Everything here reads a finished optimisation and returns plain Python, so a
run record stays JSON and the figures drawn from it are drawn somewhere else.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pyCamSet.utils.calibration_report import reprojection_residuals
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
    values, _ = reprojection_residuals(residuals, handler)
    if values.size % 2:
        raise ValueError(
            "Invalid reprojection residual segment length: "
            f"reprojection_count={values.size} is not a whole number of "
            f"two-scalar observations"
        )
    return values.reshape(-1, 2)


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
