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

It is measured under the model the calibration was solved under -- every face
of the target an independent board, posed on its own -- so the number here is
the one OpenCV achieved.  Holding the target rigid instead measures something
real, but it is the target's build error rather than the camera's, and it
belongs to the bundle adjustment's summary rather than to this one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from pyCamSet.calibration_targets.core.abstract_target import get_keys
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
    :param n_points: features those views contributed, over the boards that
        met the per board minimum
    :param rms_px: the reprojection RMS its own calibration achieved, pooled
        over every board view
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
    def from_calibration(cls, cams, detection, target,
                         min_detections_per_board: int = 12) -> IntrinsicsReport:
        """
        Measure a per camera calibration against the detections it solved.

        :param cams: the calibrated camera set
        :param detection: the detections it was solved from
        :param target: the calibration target they were found with
        :param min_detections_per_board: the per board minimum it was solved under
        """
        per_view, pooled_rms = per_view_reprojection(
            detection, target, cams, min_detections_per_board)

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


#: Every intrinsic parameter held, so the solve moves only the board poses.
#: The initial calibration has already chosen the intrinsics; what is wanted
#: back is the reprojection they achieved, not a second opinion on them.
_POSE_ONLY = (
    cv2.CALIB_USE_INTRINSIC_GUESS
    | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT
    | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_TANGENT_DIST
    | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
    | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
)


def _board_views(detections, calibration_target, cam_name,
                 min_detections_per_board):
    """
    The planar board views a camera's own calibration was solved from.

    One view per face per image, in that face's own frame, behind the same
    per board minimum the calibration applied -- so what is measured is what
    was solved.

    :param detections: the detections the camera was calibrated from
    :param calibration_target: the target they were found with
    :param cam_name: the camera to collect for
    :param min_detections_per_board: the per board minimum the calibration used
    :return: the object points, the image points, the image each view came
        from, and which images held any detection at all
    """
    max_ims = int(detections.max_ims)
    n_boards = int(np.prod(calibration_target.point_local.shape[:-2]))
    cam_detection = detections.get(cam=cam_name)

    objects, images, of_image = [], [], []
    has_detection = [False] * max_ims
    if not cam_detection.has_data():
        return objects, images, of_image, has_detection

    for im_idx in range(max_ims):
        data = cam_detection.get(global_im_num=im_idx).get_data()
        if data is None or len(data) == 0:
            continue
        has_detection[im_idx] = True
        keys = get_keys(data)
        for board in np.unique(keys[:, :-1]):
            if board >= n_boards:
                continue
            mask = np.squeeze(keys[:, :-1] == board)
            if np.sum(mask) < min_detections_per_board:
                continue
            objects.append(calibration_target.point_local[
                tuple(keys[mask].astype(int).T)][None, ...].astype("float32"))
            images.append(data[mask, -2:][None, ...].astype("float32"))
            of_image.append(im_idx)

    return objects, images, of_image, has_detection


def _board_errors(objects, images, cam):
    """
    Each board view's RMS, with OpenCV solving the poses at fixed intrinsics.

    Its own solver rather than a pose per board of ours, because the number
    being reproduced is the one its calibration returned, and a weaker pose
    solve shows up as error the calibration never had.

    :param objects: the object points of each board view
    :param images: the image points of each board view
    :param cam: the calibrated camera to measure
    :return: the RMS of each view, and how many points each holds
    """
    counts = np.array([o.shape[1] for o in objects], dtype=int)
    if not objects:
        return np.array([], dtype=float), counts

    matrix = np.asarray(cam.intrinsic, dtype=float).copy()
    coefficients = np.asarray(
        cam.distortion_coefs, dtype=float).reshape(-1).copy()
    size = tuple(int(v) for v in np.asarray(cam.res).reshape(-1)[:2])
    try:
        *_, per_view = cv2.calibrateCameraExtended(
            objects, images, size, matrix, coefficients, flags=_POSE_ONLY)
    except cv2.error as exc:
        logger.debug("pose only solve failed for %s: %s", cam.name, exc)
        return np.full(len(objects), float("nan")), counts
    return np.asarray(per_view, dtype=float).reshape(-1), counts


def _pool(rms, counts):
    """
    Several views' RMS as one, weighted by the points each was measured over.

    Pooling this way is associative, so an image pooled from its boards and
    then pooled with other images gives what pooling every board at once
    would: the per camera figure is OpenCV's own, however it is grouped.

    :param rms: the RMS of each view
    :param counts: how many points each view holds
    """
    total = int(np.sum(counts))
    if total == 0:
        return float("nan")
    return float(np.sqrt(np.sum(np.asarray(rms) ** 2 * counts) / total))


def per_view_reprojection(
        detections, calibration_target, cams,
        min_detections_per_board: int = 12,
) -> tuple[dict[str, dict], dict[str, float]]:
    """
    The reprojection each camera's own calibration achieved, per image.

    Each face of the target is an independent board with a pose of its own,
    which is the model the calibration was solved under: ``cv2.calibrateCamera``
    is handed one planar board per face per image and never asks the faces to
    agree about where the target is. Measuring the result against a single
    rigid pose for the whole target instead charges the camera for the
    target's build error, which on a cube is most of the number.

    :param detections: the detections the cameras were calibrated from
    :param calibration_target: the target they were found with
    :param cams: the calibrated cameras
    :param min_detections_per_board: the per board minimum the calibration used
    :return: the per-view series per camera, and each camera's pooled RMS
    """
    per_view: dict[str, dict] = {}
    overall_rms: dict[str, float] = {}
    max_ims = int(detections.max_ims)

    for cam_name in cams.get_names():
        objects, images, of_image, has_detection = _board_views(
            detections, calibration_target, cam_name, min_detections_per_board)
        board_rms, board_points = _board_errors(objects, images, cams[cam_name])
        of_image = np.array(of_image, dtype=int)
        solved = ~np.isnan(board_rms) if board_rms.size else np.zeros(0, bool)

        rms_px, n_points, valid_pose = [], [], []
        for im_idx in range(max_ims):
            here = solved & (of_image == im_idx) if solved.size else solved
            if not np.any(here):
                rms_px.append(float("nan"))
                n_points.append(0)
                valid_pose.append(False)
                continue
            rms_px.append(_pool(board_rms[here], board_points[here]))
            n_points.append(int(np.sum(board_points[here])))
            valid_pose.append(True)

        per_view[cam_name] = {
            "image_indices": list(range(max_ims)),
            "rms_px": rms_px,
            "n_points": n_points,
            "has_detection": has_detection,
            "valid_pose": valid_pose,
        }
        overall_rms[cam_name] = _pool(board_rms[solved], board_points[solved])

    return per_view, overall_rms
