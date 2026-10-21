"""Generate reproducible multi-camera synthetic detections for A0.

"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pyCamSet.calibration_targets.core.target_detections import TargetDetection


@dataclass(frozen=True)
class SyntheticRig:
    """Known pinhole rig and the projected rows consumed by calibration."""

    camera_names: tuple[str, ...]
    intrinsic: np.ndarray
    camera_translations: np.ndarray
    image_res: tuple[int, int]
    target: object
    target_poses: tuple[tuple[np.ndarray, np.ndarray], ...]
    detections: TargetDetection


def make_synthetic_rig(
    target: object,
    *,
    camera_count: int = 3,
    image_count: int = 6,
    image_res: tuple[int, int] = (640, 480),
) -> SyntheticRig:
    """Project target point_data into a deterministic translated camera rig.

    The camera and target transforms are known ground truth.  Rows are emitted
    directly as TargetDetection records so this exercises multi-view geometry
    independently of any detector backend.
    """
    if camera_count < 2 or image_count < 3:
        raise ValueError("a synthetic rig needs at least 2 cameras and 3 poses")
    height, width = image_res
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")

    points = np.asarray(target.point_data, dtype=np.float64).reshape(-1, 3)
    if len(points) < 4 or not np.all(np.isfinite(points)):
        raise ValueError("target must expose at least four finite 3D points")
    key_shape = tuple(np.asarray(target.point_data).shape[:-1])
    key_width = len(key_shape)
    if key_width == 0:
        key_shape = (len(points),)
        key_width = 1

    # Intrinsics are deliberately non-square and off-centre to expose axis swaps.
    intrinsic = np.array(
        [[0.82 * width, 0.0, 0.49 * width],
         [0.0, 0.79 * width, 0.51 * height],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    names = tuple(f"cam_{index}" for index in range(camera_count))
    camera_centres = np.zeros((camera_count, 3), dtype=np.float64)
    camera_centres[:, 0] = np.linspace(-0.12, 0.12, camera_count)
    rows: list[list[float]] = []
    poses: list[tuple[np.ndarray, np.ndarray]] = []

    # Target poses rotate about two axes and vary depth and lateral position.
    for image_index in range(image_count):
        phase = image_index / max(1, image_count - 1)
        rvec = np.array([0.14 * (phase - 0.5), 0.21 * (phase - 0.5),
                         0.08 * np.sin(phase * np.pi)], dtype=np.float64)
        tvec = np.array([-0.035 + 0.07 * phase, 0.02 * np.cos(phase * np.pi),
                         0.85 + 0.08 * phase], dtype=np.float64)
        poses.append((rvec.copy(), tvec.copy()))
        for camera_index, centre in enumerate(camera_centres):
            camera_tvec = tvec - centre
            pixels, _ = cv2.projectPoints(
                points, rvec, camera_tvec, intrinsic, np.zeros(5, dtype=np.float64)
            )
            pixels = pixels.reshape(-1, 2)
            keys = np.column_stack(np.unravel_index(np.arange(len(points)), key_shape))
            for point_index, (key, pixel) in enumerate(zip(keys, pixels)):
                if 0 <= pixel[0] < width and 0 <= pixel[1] < height:
                    rows.append([
                        float(camera_index), float(image_index),
                        *map(float, key), float(pixel[0]), float(pixel[1]),
                    ])

    detection = TargetDetection(
        cam_names=list(names), data=np.asarray(rows, dtype=np.float64), max_ims=image_count
    )
    return SyntheticRig(
        camera_names=names,
        intrinsic=intrinsic,
        camera_translations=camera_centres,
        image_res=image_res,
        target=target,
        target_poses=tuple(poses),
        detections=detection,
    )
