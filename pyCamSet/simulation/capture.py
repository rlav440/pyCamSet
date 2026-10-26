"""
A whole synthetic capture, written as the image folder the pipeline reads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .imperfection import PastingErrors, paste_faces
from .render import Scene, readable_pixels, render_view
from .rig import SimCamera
from .sensor import SensorModel

#: The file a capture's ground truth is written to, beside its camera folders.
#: A dot-free name that is not an image, so ``calibrate_cameras`` skips it.
GROUND_TRUTH_FILE = "ground_truth.json"

#: The true target points, as ``np.save`` writes them.
TRUE_POINTS_FILE = "ground_truth_points.npy"


@dataclass
class Capture:
    """
    Everything that made a synthetic capture.

    :param folder: where the images are, ``<folder>/<camera>/<shot>.png``.
    :param target_spec: the target, as
        :func:`pyCamSet.calibration_targets.core.target_registry.spec_of`
        writes it -- the *nominal* target the calibration is told about.
    :param cameras: the cameras, in folder order.
    :param poses: the target-to-world transform of every shot.
    :param errors: how the target's faces were pasted.
    :param true_points: where the target's points really were, shaped like
        its ``point_data``, in its own frame.
    :param sensor: the camera degradations.
    :param scene: the lighting and the background.
    :param visible_points: how many target points fell inside each camera's
        image in each shot, ``(n_cameras, n_shots)``, on faces turned within
        :data:`~pyCamSet.simulation.render.READABLE_ANGLE_DEG` of the camera --
        roughly what a perfect detector could read.
    """

    folder: Path
    target_spec: dict
    cameras: list[SimCamera]
    poses: np.ndarray
    errors: PastingErrors
    true_points: np.ndarray
    sensor: SensorModel = field(default_factory=SensorModel)
    scene: Scene = field(default_factory=Scene)
    visible_points: np.ndarray | None = None


def simulate_capture(target, cameras: list[SimCamera], poses: np.ndarray, folder,
                     errors: PastingErrors | None = None,
                     sensor: SensorModel | None = None, scene: Scene | None = None,
                     log=print) -> Capture:
    """
    Photograph ``target`` with every camera at every pose, and write it out.

    The images go to ``<folder>/<camera name>/<shot:03d>.png``, the layout
    :func:`~pyCamSet.calibrate_cameras` and Phase 1 read; the ground truth
    goes beside them in :data:`GROUND_TRUTH_FILE` and :data:`TRUE_POINTS_FILE`.

    :param target: the target, nominal -- a cube target with ``textures``.
    :param cameras: the rig.
    :param poses: ``(n_shots, 4, 4)`` target-to-world transforms.
    :param folder: where to write; created if missing.
    :param errors: how the faces were pasted; none when not given.
    :param sensor: the camera degradations; the defaults when not given.
    :param scene: the lighting and the background; the defaults when not given.
    :param log: what to call with each line of progress.
    :return: the capture, as written.
    """
    from pyCamSet.calibration_targets.core.target_registry import spec_of

    sensor = SensorModel() if sensor is None else sensor
    scene = Scene() if scene is None else scene
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    n_faces = len(target.faceData.ur_ftform)
    errors = PastingErrors.none(n_faces) if errors is None else errors
    true_points = paste_faces(target, errors)
    rng = np.random.default_rng(sensor.seed)
    visible = np.zeros((len(cameras), len(poses)), dtype=int)

    for c, camera in enumerate(cameras):
        out = folder / camera.name
        out.mkdir(parents=True, exist_ok=True)
        for shot, pose in enumerate(poses):
            radiance = render_view(target, camera, pose, errors, scene, rng)
            image = sensor.apply(radiance, camera.intrinsic, camera.distortion, rng)
            # imencode + write_bytes: cv2.imwrite fails on non-ASCII Windows paths.
            encoded, buffer = cv2.imencode(".png", image)
            if not encoded:
                raise RuntimeError(f"could not encode shot {shot} of {camera.name} as PNG")
            (out / f"{shot:03d}.png").write_bytes(buffer.tobytes())
            visible[c, shot] = len(readable_pixels(true_points, n_faces, camera, pose))
        log(f"{camera.name}: {len(poses)} shots, "
            f"{visible[c].sum()} target points in view")

    capture = Capture(folder=folder, target_spec=spec_of(target), cameras=list(cameras),
                      poses=np.asarray(poses), errors=errors, true_points=true_points,
                      sensor=sensor, scene=scene, visible_points=visible)
    _write_ground_truth(capture)
    return capture


def load_ground_truth(folder) -> Capture:
    """Read back a capture written by :func:`simulate_capture`."""
    folder = Path(folder)
    with open(folder / GROUND_TRUTH_FILE, encoding="utf-8") as handle:
        data = json.load(handle)
    return Capture(
        folder=folder, target_spec=data["target"],
        cameras=[SimCamera.from_dict(c) for c in data["cameras"]],
        poses=np.asarray(data["poses"], float),
        errors=PastingErrors.from_dict(data["pasting_errors"]),
        true_points=np.load(folder / TRUE_POINTS_FILE),
        sensor=SensorModel(**data["sensor"]), scene=Scene(**data["scene"]),
        visible_points=np.asarray(data.get("visible_points"), int)
        if data.get("visible_points") is not None else None)


def _write_ground_truth(capture: Capture) -> None:
    from dataclasses import asdict

    data = {
        "target": capture.target_spec,
        "cameras": [c.to_dict() for c in capture.cameras],
        "poses": capture.poses.tolist(),
        "pasting_errors": capture.errors.to_dict(),
        "sensor": capture.sensor.to_dict(),
        "scene": asdict(capture.scene),
        "visible_points": None if capture.visible_points is None
        else capture.visible_points.tolist(),
    }
    with open(capture.folder / GROUND_TRUTH_FILE, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=1)
    np.save(capture.folder / TRUE_POINTS_FILE, capture.true_points)

