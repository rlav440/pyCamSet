"""
What each camera's own calibration came out as.

Between the detections and the bundle adjustment every camera is solved on
its own, and that stage reported nothing: a camera whose intrinsics are
already wrong here is not rescued by solving them all together, but the first
number saying so arrived pages later, after the bundle adjustment.

This is that stage's block, laid out by the same rules as the detection and
calibration summaries either side of it.  The per view reprojection it is
measured from lives here too, because the report and the phase diagnostics
both want it and neither should compute it twice.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from pyCamSet.calibration_targets.abstract_target import get_keys
from pyCamSet.utils import report_format as fmt

logger = logging.getLogger(__name__)

#: Above this per camera RMS the initial intrinsics are not a usable starting
#: point for the bundle adjustment, rather than merely rough. The same band
#: the calibration summary calls a poor result.
HIGH_INITIAL_RMS_PX = fmt.ERROR_FAIR_PX


@dataclass
class CameraIntrinsicStats:
    """
    What one camera's own calibration came out as.

    :param n_views: images the camera was posed in, and so measured over
    :param n_points: features those views contributed
    :param rms_px: the pooled reprojection RMS over them
    :param distortion_l2: the norm of the distortion coefficients, as one
        number for how much lens model the solve asked for
    """
    name: str
    index: int
    n_views: int
    n_points: int
    rms_px: float
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_l2: float


@dataclass
class IntrinsicsReport:
    """
    The per camera calibration, as data rather than as log lines.
    """
    camera_names: list[str]
    n_images: int
    per_camera: list[CameraIntrinsicStats] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    blocking_flags: list[str] = field(default_factory=list)

    #: The per image series each camera's RMS was pooled from. Kept because
    #: the report is computed from it and a phase records it as a diagnostic,
    #: and left out of :meth:`to_dict` because it is the working, not the
    #: report: it is one entry per image per camera.
    per_view: dict[str, dict] = field(default_factory=dict, repr=False)

    @classmethod
    def from_calibration(cls, cams, detection, target) -> IntrinsicsReport:
        """
        Measure a per camera calibration against the detections it solved.

        :param cams: the calibrated camera set
        :param detection: the detections it was solved from
        :param target: the calibration target they were found with
        """
        per_view, pooled_rms = per_view_reprojection(detection, target, cams)

        per_camera = []
        for index, name in enumerate(cams.get_names()):
            series = per_view.get(name, {})
            valid = np.array(series.get("valid_pose", []), dtype=bool)
            points = np.array(series.get("n_points", []), dtype=int)
            matrix = np.array(cams[name].intrinsic, dtype=float)
            coefficients = np.array(
                cams[name].distortion_coefs, dtype=float).reshape(-1)
            per_camera.append(CameraIntrinsicStats(
                name=name,
                index=index,
                n_views=int(np.count_nonzero(valid)),
                n_points=int(np.sum(points[valid])) if valid.any() else 0,
                rms_px=float(pooled_rms.get(name, float("nan"))),
                fx=float(matrix[0, 0]), fy=float(matrix[1, 1]),
                cx=float(matrix[0, 2]), cy=float(matrix[1, 2]),
                distortion_l2=float(np.linalg.norm(coefficients)),
            ))

        report = cls(
            camera_names=list(cams.get_names()),
            n_images=int(detection.max_ims),
            per_camera=per_camera,
            per_view=per_view,
        )
        report.flags, report.blocking_flags = report._find_flags()
        return report

    def _find_flags(self) -> tuple[list[str], list[str]]:
        """
        The concerns worth raising, each naming its threshold and its value.

        A camera with no pose at all blocks what follows, because the bundle
        adjustment would start it from nothing. A merely poor reprojection is
        a warning: it is the reader's call whether to solve from it.

        :return: every concern, and the subset that blocks the next phase
        """
        flags, blocking = [], []
        for cam in self.per_camera:
            if cam.n_views == 0 or np.isnan(cam.rms_px):
                flags.append(
                    f'camera "{cam.name}" was posed in none of the '
                    f"{self.n_images} images, so its intrinsics rest on "
                    f"nothing and the bundle adjustment has no sound place "
                    f"to start it from")
                blocking.append(flags[-1])
            elif cam.rms_px > HIGH_INITIAL_RMS_PX:
                flags.append(
                    f'camera "{cam.name}" reprojects at {cam.rms_px:.2f} px, '
                    f"above the {HIGH_INITIAL_RMS_PX:.0f} px this check "
                    f"expects of an initial calibration: check its detections "
                    f"before reading anything into the solve that follows")
        return flags, blocking

    def to_dict(self) -> dict[str, Any]:
        """
        The report as plain types, for a run record or a diff.
        """
        return {
            "camera_names": list(self.camera_names),
            "n_images": self.n_images,
            "per_camera": [vars(c) for c in self.per_camera],
            "flags": list(self.flags),
            "blocking_flags": list(self.blocking_flags),
        }

    def summary(self, colour: bool | None = None) -> str:
        """
        The report as a block of text that fits an 80 column terminal.

        The RMS is graded on the same bands as every other reprojection error
        in these blocks, so a glance down the column means what it means
        everywhere else.

        :param colour: force colour on or off, defaulting to auto detection
        """
        measured = [c for c in self.per_camera if not np.isnan(c.rms_px)]
        mean_rms = (float(np.mean([c.rms_px for c in measured]))
                    if measured else float("nan"))

        lines = fmt.title("Initial intrinsics")
        lines += fmt.kv_rows([
            ("cameras", str(len(self.camera_names))),
            ("images", str(self.n_images)),
            ("views posed", str(sum(c.n_views for c in self.per_camera))),
            ("mean rms, px", f"{mean_rms:.2f}"),
        ])
        lines += [""]
        lines += fmt.table(
            ["camera", "views", "rms", "fx", "fy", "|k|"],
            [[c.name, c.n_views, fmt.error_cell(c.rms_px),
              f"{c.fx:.1f}", f"{c.fy:.1f}", f"{c.distortion_l2:.3f}"]
             for c in self.per_camera],
            widths=[16, 7, 9, 11, 11, 9],
            colour=colour,
        )
        lines += fmt.flag_lines(self.flags)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


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
