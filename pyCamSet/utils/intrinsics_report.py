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

It is the residual the seed itself leaves: every face of the target an
independent board, posed from its own homography at the camera's intrinsics
by Zhang's closed form (:mod:`pyCamSet.calibration.zhang`).  The seed hands
those numbers over (:class:`~pyCamSet.calibration.seed_residuals.SeedResiduals`)
and they are restated as they are; only a camera that was not seeded here --
loaded from a saved calibration, or handed in -- is measured, the same way.  Holding the target rigid instead measures
something real, but it is the target's build error rather than the camera's,
and it belongs to the bundle adjustment's summary rather than to this one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from pyCamSet.calibration.seed_residuals import SeedResiduals
from pyCamSet.calibration.zhang import (
    homography_of,
    pose_from_homography,
    reprojection_rms,
)
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
                         min_detections_per_board: int = 12,
                         seeded: dict[str, SeedResiduals] | None = None,
                         ) -> IntrinsicsReport:
        """
        Measure a per camera calibration against the detections it solved.

        :param cams: the calibrated camera set
        :param detection: the detections it was solved from
        :param target: the calibration target they were found with
        :param min_detections_per_board: the per board minimum it was solved under
        :param seeded: the residuals the seed left, per camera, which are
            restated as they are; a camera without them is measured here
        """
        per_view, pooled_rms = per_view_reprojection(
            detection, target, cams, min_detections_per_board, seeded)

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


def telecentric_project(cam, points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Project board-frame points through a telecentric camera at a given board pose.

    The camera's own model (``TelecentricCamera.project_points``) with the pose given
    explicitly: ``u = m_x * x * w * d + c_x`` with ``w = 1 / (1 + eps * z)`` and the
    camera's division distortion ``d``.

    :param cam: the telecentric camera
    :param points: N x 3 board-frame points
    :param rvec: board-to-camera Rodrigues rotation
    :param tvec: board-to-camera translation (its z matters only when eps != 0)
    :return: N x 2 pixel coordinates
    """
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3))
    in_camera = np.asarray(points, dtype=float).reshape(-1, 3) @ rotation.T + np.asarray(tvec, dtype=float)
    w = 1.0 / (1.0 + float(cam.telecentricity) * in_camera[:, 2])
    m_x, m_y = cam.magnification
    c_x, c_y = cam.principal_point
    xs, ys = cam._distort(m_x * in_camera[:, 0] * w, m_y * in_camera[:, 1] * w)
    return np.stack([xs + c_x, ys + c_y], axis=-1)


def _orthographic_starts(points: np.ndarray, pixels: np.ndarray, cam) -> list[np.ndarray]:
    """Closed-form starting poses (rvec, tx, ty) for an orthographic view of a board.

    A least-squares affine fit gives the first two rows of the rotation scaled by the
    magnification. A planar board leaves the third column of those rows free; it is fixed
    by unit length and orthogonality up to a sign - the classic two-fold tilt ambiguity of
    an orthographic view - so both signs are returned.
    """
    m = np.asarray(cam.magnification, dtype=float)
    c = np.asarray(cam.principal_point, dtype=float)
    centred = points - points.mean(axis=0)
    _, spread, frame = np.linalg.svd(centred, full_matrices=False)
    if frame.shape == (3, 3) and np.linalg.det(frame) < 0:
        # SVD may return a reflection; rotation_local @ frame then is no rotation and its
        # Rodrigues vector is garbage. Flipping the normal leaves a planar board's local x, y
        # alone. (Found teaching the single-face recovery: both tilts' starts fitted their own
        # camera at 224 px on exact synthetic data; only the joint refinement hid it.)
        frame = frame * np.array([1.0, 1.0, -1.0])[:, None]
    planar = spread[-1] <= 1e-9 * max(spread[0], 1e-300)
    # work in the board's own principal frame so a planar board has z = 0
    local = centred @ frame.T
    design = np.column_stack([local[:, :2] if planar else local, np.ones(len(local))])
    coefficients, *_ = np.linalg.lstsq(design, pixels - c, rcond=None)
    rows = (coefficients[:-1].T / m[:, None])  # 2 x (2 or 3)
    candidates = []
    if planar:
        r1, r2 = rows[0], rows[1]
        a = max(0.0, 1.0 - r1 @ r1)
        b = max(0.0, 1.0 - r2 @ r2)
        third = (np.sqrt(a), -(r1 @ r2) / np.sqrt(a)) if a > 1e-12 else (0.0, np.sqrt(b))
        for sign in (1.0, -1.0):
            candidates.append(np.array([[r1[0], r1[1], sign * third[0]],
                                        [r2[0], r2[1], sign * third[1]]]))
    else:
        candidates.append(rows)
    starts = []
    for top in candidates:
        full = np.vstack([top, np.cross(top[0], top[1])])
        u, _, vt = np.linalg.svd(full)
        rotation_local = u @ vt
        if np.linalg.det(rotation_local) < 0:
            rotation_local = u @ np.diag([1.0, 1.0, -1.0]) @ vt
        rotation = rotation_local @ frame  # board frame -> camera
        offset = coefficients[-1] / m - (rotation @ points.mean(axis=0))[:2]
        rvec, _ = cv2.Rodrigues(rotation)
        starts.append(np.concatenate([rvec.reshape(3), offset]))
    return starts


def telecentric_view_rms(points, pixels, cam) -> float:
    """One board view's RMS under the telecentric model, its pose solved at fixed intrinsics.

    The telecentric counterpart of OpenCV's pose-only solve used for pinhole cameras:
    rotation and in-plane translation (plus depth when the lens has a telecentricity
    error) are fitted with the camera's own projection and distortion, from both
    orthographic starting poses. (Before this, D2.6 for telecentric cameras ran OpenCV's
    pinhole solve, a different model; found while testing on real rigs.)

    :param points: the board view's object points, (1, N, 3) or (N, 3)
    :param pixels: its image points, (1, N, 2) or (N, 2)
    :param cam: the calibrated telecentric camera
    :return: the RMS reprojection error in pixels, or NaN if no pose could be fitted
    """
    from scipy.optimize import least_squares

    points = np.asarray(points, dtype=float).reshape(-1, 3)
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    if len(points) < 3:
        return float("nan")
    with_depth = float(cam.telecentricity) != 0.0
    best = float("nan")
    for start in _orthographic_starts(points, pixels, cam):
        x0 = np.concatenate([start, [0.0]]) if with_depth else start

        def residual(x):
            t = np.array([x[3], x[4], x[5] if with_depth else 0.0])
            return (telecentric_project(cam, points, x[:3], t) - pixels).reshape(-1)
        try:
            fit = least_squares(residual, x0, method="lm" if len(points) * 2 >= len(x0) else "trf")
        except (ValueError, np.linalg.LinAlgError):
            continue
        errors = fit.fun.reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1))))
        if np.isfinite(rms) and not (rms >= best):
            best = rms
    return best


def _board_errors(objects, images, cam):
    """
    Each board view's RMS, as Zhang's closed form leaves it.

    The view is posed from its own homography at the camera's intrinsics and
    measured with no distortion, which is how ``initial_calibration`` seeded
    it and the residual it logged -- the same numbers, not a second solve.  A
    camera given with distortion (``fixed_params``) is measured on its
    undistorted pixels, so the lens model it was given is not charged as
    error.

    :param objects: the object points of each board view
    :param images: the image points of each board view
    :param cam: the calibrated camera to measure
    :return: the RMS of each view, and how many points each holds
    """
    counts = np.array([o.shape[1] for o in objects], dtype=int)
    if not objects:
        return np.array([], dtype=float), counts
    from pyCamSet.cameras.telecentric_camera import TelecentricCamera
    if isinstance(cam, TelecentricCamera):
        return np.array([telecentric_view_rms(o, i, cam) for o, i in zip(objects, images)],
                        dtype=float), counts

    intrinsic = np.asarray(cam.intrinsic, dtype=float)
    coefficients = np.asarray(cam.distortion_coefs, dtype=float).reshape(-1)
    per_view = np.full(len(objects), float("nan"))
    for k, (obj, img) in enumerate(zip(objects, images)):
        obj = np.asarray(obj, dtype=float).reshape(-1, 3)
        img = np.asarray(img, dtype=float).reshape(-1, 2)
        if np.any(coefficients):
            img = cv2.undistortPoints(img.reshape(-1, 1, 2), intrinsic, coefficients,
                                      P=intrinsic).reshape(-1, 2)
        try:
            pose = pose_from_homography(intrinsic, homography_of(obj, img))
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.debug("no closed form pose for %s, view %d: %s", cam.name, k, exc)
            continue
        per_view[k] = reprojection_rms(intrinsic, pose, obj, img)
    return per_view, counts


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
        seeded: dict[str, SeedResiduals] | None = None,
) -> tuple[dict[str, dict], dict[str, float]]:
    """
    The reprojection each camera's own calibration achieved, per image.

    Each face of the target is an independent board with a pose of its own,
    which is the model the calibration was solved under: Zhang's closed form
    is handed one planar board per face per image and never asks the faces to
    agree about where the target is. Measuring the result against a single
    rigid pose for the whole target instead charges the camera for the
    target's build error, which on a cube is most of the number.

    :param detections: the detections the cameras were calibrated from
    :param calibration_target: the target they were found with
    :param cams: the calibrated cameras
    :param min_detections_per_board: the per board minimum the calibration used
    :param seeded: the residuals the seed left, per camera, restated as they
        are.  A camera without them -- loaded from a saved calibration, handed
        in, or given whole by ``fixed_params`` -- has its views collected and
        measured here, the same way the seed measured them.
    :return: the per-view series per camera, and each camera's pooled RMS
    """
    per_view: dict[str, dict] = {}
    overall_rms: dict[str, float] = {}
    max_ims = int(detections.max_ims)
    seeded = seeded or {}

    for cam_name in cams.get_names():
        seed = seeded.get(cam_name)
        if seed is not None:
            board_rms = np.asarray(seed.rms, dtype=float)
            board_points = np.asarray(seed.points, dtype=int)
            of_image = np.asarray(seed.image, dtype=int)
            has_detection = [bool(h) for h in seed.has_detection]
        else:
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

        from pyCamSet.cameras.telecentric_camera import TelecentricCamera
        per_view[cam_name] = {
            "pose_model": ("telecentric" if isinstance(cams[cam_name], TelecentricCamera)
                           else "pinhole"),
            "image_indices": list(range(max_ims)),
            "rms_px": rms_px,
            "n_points": n_points,
            "has_detection": has_detection,
            "valid_pose": valid_pose,
        }
        overall_rms[cam_name] = _pool(board_rms[solved], board_points[solved])

    return per_view, overall_rms


def report_initial_calibration(cams, detection, target,
                               min_detections_per_board: int = 12,
                               seeded: dict[str, SeedResiduals] | None = None,
                               ) -> IntrinsicsReport:
    """
    Reports what each camera's own calibration came out as.

    :param cams: the per camera calibration to describe
    :param detection: the detections it was solved from
    :param target: the calibration target they were found with
    :param min_detections_per_board: the per board minimum it was solved under
    :param seeded: the residuals the seed left, per camera; see
        :func:`per_view_reprojection`
    :return: the report, which is also logged
    """
    report = IntrinsicsReport.from_calibration(
        cams, detection, target, min_detections_per_board, seeded)
    logger.info("\n" + report.summary())
    return report
