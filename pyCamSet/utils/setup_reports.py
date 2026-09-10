"""
What the input data looks like, reported before the optimiser runs.

A calibration can only be as good as what went into it, and the two ways it
goes wrong before the solver ever starts are visible in the data:

* the target was not detected well enough, in enough images, by every camera;
* the cameras do not agree on where the target was, which means they are not
  seeing the same instant -- misordered files, an unsynchronised trigger, or
  detections bad enough to be meaningless.

Both were already checked and both reported a line at a time, interleaved with
everything else. These turn each into a block a person can read at a glance:
the detection rates are graded by colour so the table itself shows which
cameras are weak, and anything that stops a camera being calibrated at all
still says so in words.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from pyCamSet.utils import report_format as fmt

logger = logging.getLogger(__name__)

#: The target should not appear to move relative to the reference camera. Past
#: this much scatter in that relative position the images do not line up.
MAX_TRANSLATION_STDEV_MM = 10.0

#: The same, for the relative angle.
MAX_ANGLE_STDEV_DEG = 5.0


@dataclass
class CameraDetectionStats:
    """
    How well one camera saw the target.

    :param n_images_seen: images in which this camera detected anything
    :param detection_rate: that count over the images in the run
    :param completeness: mean fraction of a board resolved, when seen
    :param n_features: total detected features
    """
    name: str
    index: int
    n_images_seen: int
    detection_rate: float
    completeness: float
    n_features: int


@dataclass
class DetectionReport:
    """
    What the detection stage found, per camera.
    """
    camera_names: list[str]
    n_images: int
    n_features: int
    per_camera: list[CameraDetectionStats] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @classmethod
    def from_detection(cls, detected, target) -> DetectionReport:
        """
        Measure a detection against the calibration target it was found with.

        :param detected: the TargetDetection to describe
        :param target: the calibration target, for its features per face
        """
        corners_per_face = _features_per_face(target)
        n_images = int(detected.max_ims)
        per_camera = []

        for index, name in enumerate(detected.cam_names):
            per_camera.append(_camera_detection_stats(
                detected, index, name, corners_per_face, n_images))

        report = cls(
            camera_names=list(detected.cam_names),
            n_images=n_images,
            n_features=sum(c.n_features for c in per_camera),
            per_camera=per_camera,
        )
        report.flags = report._find_flags()
        return report

    def _find_flags(self) -> list[str]:
        """
        A rate or a completeness that is merely poor is shown by the colour of
        its cell, which says the same thing without a paragraph. Only a camera
        that saw nothing gets prose, because that one cannot be calibrated at
        all and the reason is usually a wrong folder or the wrong target
        rather than anything about the images.
        """
        return [
            f'camera "{cam.name}" detected the target in none of the '
            f"{self.n_images} images, so it cannot be calibrated: check the "
            f"images are in the right folder and that the target matches the "
            f"one being detected"
            for cam in self.per_camera if cam.n_features == 0
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self, colour: bool | None = None) -> str:
        """
        The report as a block of text that fits an 80 column terminal.

        The detection rate and the board completeness are graded by colour --
        green at or above 90%, orange down to 60%, red below that -- so the
        table itself says which cameras are weak.

        :param colour: force colour on or off, defaulting to auto detection
        """
        seen = [c for c in self.per_camera if c.n_features]
        mean_rate = float(np.mean([c.detection_rate for c in seen])) if seen else 0.0

        lines = fmt.title("Detection summary")
        lines += fmt.kv_rows([
            ("cameras", str(len(self.camera_names))),
            ("images", str(self.n_images)),
            ("features", str(self.n_features)),
            ("mean detected", fmt.percent(mean_rate)),
        ])
        lines += [""]
        lines += fmt.table(
            ["camera", "images", "detected", "complete", "features"],
            [[c.name, f"{c.n_images_seen}/{self.n_images}",
              fmt.quality_cell(c.detection_rate),
              fmt.quality_cell(c.completeness),
              c.n_features]
             for c in self.per_camera],
            widths=[18, 12, 12, 12, 12],
            colour=colour,
        )
        lines += fmt.flag_lines(self.flags)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _features_per_face(target) -> int:
    """
    How many features one face of the target carries.

    This is the denominator of the completeness figure, so it has to be the
    count that is actually printed.  For every target but the flat
    PuzzleBoard that is what ``point_data`` holds.  A PuzzleBoard's
    ``point_data`` spans the whole 501x501 virtual code field rather than the
    printed window, so using it would divide by 251,001 and report a
    completeness of a fraction of a percent for a board that was seen whole.

    :param target: the calibration target the detection was found with
    :return: features on one face
    """
    if target.__class__.__name__ == "PuzzleBoard":
        return int(target.num_squares_x * target.num_squares_y)
    return int(target.point_data.shape[-2])


def _camera_detection_stats(detected, index: int, name: str,
                            corners_per_face: int,
                            n_images: int) -> CameraDetectionStats:
    """
    The detection statistics for a single camera.

    A camera that saw nothing is the most important thing this stage can
    report, so it produces a row of zeros rather than an exception.

    :param detected: the whole detection
    :param index: the camera's index in the detection
    :param name: the camera's name
    :param corners_per_face: features on one face of the target
    :param n_images: images in the run, the detection rate denominator
    """
    empty = CameraDetectionStats(name, index, 0, 0.0, 0.0, 0)
    cam_detection = detected.get(cam=name)
    if not cam_detection.has_data():
        return empty

    completeness, n_features, n_seen = [], 0, 0
    for im_detection in cam_detection.get_image_list():
        datum = im_detection.get_data()
        if datum is None:
            continue
        n_seen += 1
        n_features += int(datum.shape[0])
        n_keys = datum.shape[1] - 4
        if n_keys == 1:
            completeness.append(datum.shape[0] / corners_per_face)
        else:
            n_boards = len(np.unique(datum[:, 2:-3], axis=0))
            completeness.append(datum.shape[0] / corners_per_face / n_boards)

    if not n_seen:
        return empty
    return CameraDetectionStats(
        name=name, index=index, n_images_seen=n_seen,
        detection_rate=n_seen / n_images if n_images else 0.0,
        completeness=float(np.mean(completeness)),
        n_features=n_features,
    )


@dataclass
class CameraConsistencyStats:
    """
    How steady one camera's view of the target is, relative to the reference.

    :param translation_stdev_mm: scatter in the relative target position
    :param angle_stdev_deg: scatter in the relative target angle
    """
    name: str
    index: int
    translation_stdev_mm: float
    angle_stdev_deg: float


@dataclass
class RigConsistencyReport:
    """
    Whether the cameras agree on where the target was in each image.

    A rigid rig looking at one target at one instant produces the same
    camera-to-camera transform in every image. Scatter in that transform means
    the images being compared are not of the same instant.
    """
    reference_camera: str
    per_camera: list[CameraConsistencyStats] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @classmethod
    def from_transforms(cls, tforms: np.ndarray, ref_cam: int = 0,
                        cam_names: list[str] | None = None
                        ) -> RigConsistencyReport:
        """
        Measure the per camera scatter in the transform to the reference.

        :param tforms: c x p x 4 x 4 target poses, per camera per image
        :param ref_cam: the camera to measure against
        :param cam_names: the camera names, defaulting to their indices
        """
        names = (list(cam_names) if cam_names is not None
                 else [str(i) for i in range(len(tforms))])

        to_reference = [np.linalg.inv(pose) for pose in tforms[ref_cam]]
        relative = np.array([
            [(cam_pose @ reference) for reference, cam_pose
             in zip(to_reference, per_camera)]
            for per_camera in tforms
        ])

        per_camera = []
        for index, transforms in enumerate(relative):
            if index == ref_cam:
                continue
            angles = np.array([
                np.arccos((np.trace(t[:3, :3]) - 1) / 2) for t in transforms])
            magnitudes = [np.linalg.norm(t[:3, -1]) for t in transforms]
            per_camera.append(CameraConsistencyStats(
                name=names[index] if index < len(names) else str(index),
                index=index,
                translation_stdev_mm=float(np.nanstd(magnitudes)) * 1000,
                angle_stdev_deg=float(np.degrees(np.nanstd(angles))),
            ))

        report = cls(
            reference_camera=(names[ref_cam] if ref_cam < len(names)
                              else str(ref_cam)),
            per_camera=per_camera,
        )
        report.flags = report._find_flags()
        return report

    def _find_flags(self) -> list[str]:
        cause = ("which usually means misordered images, a trigger that is "
                 "not synchronised, or detections bad enough to be "
                 "meaningless")
        flags = []
        for cam in self.per_camera:
            if cam.translation_stdev_mm > MAX_TRANSLATION_STDEV_MM:
                flags.append(
                    f'camera "{cam.name}" moves '
                    f"{cam.translation_stdev_mm:.1f} mm relative to camera "
                    f'"{self.reference_camera}" across the images, over the '
                    f"{MAX_TRANSLATION_STDEV_MM:.0f} mm expected of a rigid "
                    f"rig, {cause}")
            if cam.angle_stdev_deg > MAX_ANGLE_STDEV_DEG:
                flags.append(
                    f'camera "{cam.name}" rotates '
                    f"{cam.angle_stdev_deg:.1f} degrees relative to camera "
                    f'"{self.reference_camera}" across the images, over the '
                    f"{MAX_ANGLE_STDEV_DEG:.0f} degrees expected of a rigid "
                    f"rig, {cause}")
        return flags

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self, colour: bool | None = None) -> str:
        """
        The report as a block of text that fits an 80 column terminal.

        Both columns are graded on the same bands -- green under 1, orange
        under 5, red above -- since a millimetre of shift and a degree of
        rotation are about equally bad for a rig that is supposed to be rigid.

        :param colour: force colour on or off, defaulting to auto detection
        """
        lines = fmt.title("Rig consistency")
        lines += fmt.wrap(
            f'Scatter in each camera\'s view of the target, relative to '
            f'camera "{self.reference_camera}". A rigid rig holds these near '
            f'zero.', lead="  ")
        lines += [""]
        lines += fmt.table(
            ["camera", "shift (mm)", "rotation (deg)"],
            [[c.name, fmt.deviation_cell(c.translation_stdev_mm),
              fmt.deviation_cell(c.angle_stdev_deg)]
             for c in self.per_camera],
            widths=[22, 16, 18],
            colour=colour,
        )
        lines += fmt.flag_lines(self.flags)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
