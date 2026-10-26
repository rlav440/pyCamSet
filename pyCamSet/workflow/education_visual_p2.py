"""Qt-free data for the Phase 2 (per-camera intrinsics) education figures.
saved camset and used detections, and checked against the values the run saved.

What it prepares, per camera:

* :func:`board_views` - the board views the camera was measured on (one view per face per image,
  behind the run's per-board minimum), with the board id kept; checked against
  ``intrinsics_report._board_views`` so it is the same list the saved per-image error used.
* :func:`fit_views` - each view's pose and reprojection at the *saved* lens, solved the way Phase 2
  solved it for the saved per-image error (D2.6): the closed-form pose from the board's own
  homography, measured with no distortion, for pinhole lenses (the seed's own residual, not a
  second solve), and the telecentric model's own least squares for telecentric ones. The
  per-view RMS is the same number, and :func:`check_per_image` pools it per image and compares
  it with the saved per-image values.
* :func:`seed_views` / :func:`seed_view_rms` - the views the closed-form first estimate itself
  solved, in the order the per-view trace records them. Pinhole: one view per board. Telecentric:
  one view per *image* (all kept faces together, in the target frame), images whose kept points
  are flat skipped, as ``AbstractTarget.initial_calibration`` does. (The old
  walkthrough indexed telecentric trace records with board views and reported 270 px
  "differences" on a 2.5 px seed.)
* :func:`lens_summary`, :func:`distortion_grid` - numbers and a grid for the lens-model figure.
* :func:`photo_source`, :func:`load_photo` - photos only when Phase 1's manifest verifies them.
* Teaching helpers (round-1 review): :func:`face_warp` (one printed face warped onto its photo:
  affine for telecentric, homography for pinhole), :func:`pattern_test` (do a photo's
  residuals follow a smooth, possibly curved pattern beyond what detection noise gives; round
  2, SCI2-11; :func:`systematic_share` is its older linear-only form), :func:`centre_check` (saved principal point
  vs the real image centre; older runs saved it transposed), :func:`radial_profile` (does the
  error grow towards the edges), :func:`error_map`, :func:`typical_corner`.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

#: Below this many metres the target is drawn in mm (pyCamSet targets are stored in metres).
METRES = 1.0


@dataclass
class BoardView:
    """One planar board view: what the camera's own calibration was measured on."""

    index: int                 # position in the camera's board-view list (0-based)
    image_index: int           # the image (pose) it came from
    board: int                 # board / face id within the target
    object_points: np.ndarray  # N x 3, board-local frame
    image_points: np.ndarray   # N x 2, pixels


@dataclass
class ViewFit:
    """A board view's pose and reprojection at the saved lens."""

    view: BoardView
    R: np.ndarray              # board -> camera rotation (3 x 3)
    t: np.ndarray              # board -> camera translation (3,)
    projected: np.ndarray      # N x 2 pixels
    rms: float                 # RMS of |projected - observed|, px

    @property
    def residuals(self) -> np.ndarray:
        return self.projected - self.view.image_points


@dataclass
class DroppedBoard:
    """A board seen in an image with fewer points than the per-board minimum."""

    image_index: int
    board: int
    image_points: np.ndarray


@dataclass
class CameraViews:
    """All board views of one camera plus the boards left out."""

    camera: str
    views: list[BoardView] = field(default_factory=list)
    dropped: list[DroppedBoard] = field(default_factory=list)
    images_with_detections: list[int] = field(default_factory=list)


@dataclass
class PerImageCheck:
    """The recomputed per-image error compared with the saved per-image error."""

    per_image: dict[int, float]
    compared: int
    matched: int
    worst_difference: float
    saved_available: bool

    @property
    def all_match(self) -> bool:
        return self.saved_available and self.compared > 0 and self.matched == self.compared


# --------------------------------------------------------------------------------------
# Board views


def _keys(data: np.ndarray) -> np.ndarray:
    from pyCamSet.calibration_targets.core.abstract_target import get_keys
    return get_keys(data)


def board_views(detection: Any, target: Any, camera: str, minimum: int) -> CameraViews:
    """The camera's board views (same order as ``intrinsics_report._board_views``) and the
    boards dropped under the per-board minimum.

    Raises ``ValueError`` if the list disagrees with ``_board_views`` (it then is not the list
    the saved error was measured on, and nothing derived from it may be shown).
    """
    from pyCamSet.utils.intrinsics_report import _board_views

    out = CameraViews(str(camera))
    n_boards = int(np.prod(target.point_local.shape[:-2]))
    cam_detection = detection.get(cam=camera)
    if not cam_detection.has_data():
        return out
    for im_idx in range(int(detection.max_ims)):
        data = cam_detection.get(global_im_num=im_idx).get_data()
        if data is None or len(data) == 0:
            continue
        out.images_with_detections.append(im_idx)
        keys = _keys(data)
        for board in np.unique(keys[:, :-1]):
            if board >= n_boards:
                continue
            mask = np.squeeze(keys[:, :-1] == board)
            # float32 first, as the measured list holds them, so every recomputed number
            # is the saved one to the last bit rather than to 1e-5 px.
            pixels = np.asarray(data[mask, -2:], dtype=np.float32).astype(float).reshape(-1, 2)
            if int(np.sum(mask)) < minimum:
                out.dropped.append(DroppedBoard(im_idx, int(board), pixels))
                continue
            objects = np.asarray(target.point_local[tuple(keys[mask].astype(int).T)],
                                 dtype=np.float32).astype(float).reshape(-1, 3)
            out.views.append(BoardView(len(out.views), im_idx, int(board), objects, pixels))
    objects, images, of_image, _ = _board_views(detection, target, camera, minimum)
    same = (len(objects) == len(out.views)
            and list(of_image) == [v.image_index for v in out.views]
            and all(np.allclose(np.asarray(o).reshape(-1, 3), v.object_points.astype("float32"),
                                atol=1e-7)
                    and np.allclose(np.asarray(i).reshape(-1, 2),
                                    v.image_points.astype("float32"), atol=1e-4)
                    for o, i, v in zip(objects, images, out.views)))
    if not same:
        raise ValueError("the board views could not be matched to the measured list")
    return out


def is_telecentric(camera: Any) -> bool:
    from pyCamSet.cameras.telecentric_camera import TelecentricCamera
    return isinstance(camera, TelecentricCamera)


def _telecentric_fit(cam: Any, points: np.ndarray, pixels: np.ndarray
                     ) -> tuple[float, np.ndarray | None]:
    """``intrinsics_report.telecentric_view_rms`` with its best pose kept (same algorithm)."""
    from scipy.optimize import least_squares
    from pyCamSet.utils.intrinsics_report import _orthographic_starts, telecentric_project

    points = np.asarray(points, dtype=float).reshape(-1, 3)
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    if len(points) < 3:
        return float("nan"), None
    with_depth = float(cam.telecentricity) != 0.0
    best, best_x = float("nan"), None
    for start in _orthographic_starts(points, pixels, cam):
        x0 = np.concatenate([start, [0.0]]) if with_depth else start

        def residual(x):
            t = np.array([x[3], x[4], x[5] if with_depth else 0.0])
            return (telecentric_project(cam, points, x[:3], t) - pixels).reshape(-1)
        try:
            fit = least_squares(residual, x0,
                                method="lm" if len(points) * 2 >= len(x0) else "trf")
        except (ValueError, np.linalg.LinAlgError):
            continue
        errors = fit.fun.reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1))))
        if np.isfinite(rms) and not (rms >= best):
            best, best_x = rms, np.asarray(fit.x, dtype=float)
    return best, best_x


def fit_views(cam: Any, views: Sequence[BoardView], *,
             pose_only: bool = False) -> list[ViewFit | None]:
    """Pose and reprojection of every view at the saved lens, solved the way Phase 2 solved the
    saved per-image error (closed-form pose from the board's homography for pinhole lenses).
    ``None`` for a view whose pose could not be solved. ``pose_only`` solves a pinhole view the
    way runs saved before the closed-form seed did (OpenCV's pose-only refit at the saved lens),
    so such a run still verifies."""
    import cv2
    if not views:
        return []
    if is_telecentric(cam):
        from pyCamSet.utils.intrinsics_report import telecentric_project
        fits: list[ViewFit | None] = []
        for view in views:
            rms, x = _telecentric_fit(cam, view.object_points, view.image_points)
            if x is None or not np.isfinite(rms):
                fits.append(None)
                continue
            t = np.array([x[3], x[4], x[5] if len(x) > 5 else 0.0])
            R, _ = cv2.Rodrigues(x[:3])
            projected = telecentric_project(cam, view.object_points, x[:3], t)
            fits.append(ViewFit(view, R, t, projected, rms))
        return fits
    if pose_only:
        # every intrinsic held, so the solve moves only the board poses
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS
                 | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT
                 | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_TANGENT_DIST
                 | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
                 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
        legacy_matrix = np.asarray(cam.intrinsic, dtype=float).copy()
        legacy_coefficients = np.asarray(cam.distortion_coefs, dtype=float).reshape(-1).copy()
        size = tuple(int(v) for v in np.asarray(cam.res).reshape(-1)[:2])
        objects = [v.object_points[None].astype("float32") for v in views]
        images = [v.image_points[None].astype("float32") for v in views]
        try:
            _, _, _, rvecs, tvecs, *_rest, per_view = cv2.calibrateCameraExtended(
                objects, images, size, legacy_matrix.copy(), legacy_coefficients.copy(),
                flags=flags)
        except cv2.error:
            return [None] * len(views)
        per_view = np.asarray(per_view, dtype=float).reshape(-1)
        legacy_fits: list[ViewFit | None] = []
        for view, rvec, tvec, rms in zip(views, rvecs, tvecs, per_view):
            projected, _ = cv2.projectPoints(view.object_points, rvec, tvec, legacy_matrix,
                                             legacy_coefficients)
            R, _ = cv2.Rodrigues(rvec)
            legacy_fits.append(ViewFit(view, R, np.asarray(tvec, dtype=float).reshape(3),
                                       projected.reshape(-1, 2), float(rms)))
        return legacy_fits
    # Phase 2 saves the seed's own residual: each view is posed from its own homography at the
    # camera's intrinsics and measured with no distortion (``intrinsics_report._board_errors``),
    # so this is the same closed-form pose, not a second solve with the lens refitted.
    from pyCamSet.calibration.zhang import homography_of, pose_from_homography
    matrix = np.asarray(cam.intrinsic, dtype=float).copy()
    coefficients = np.asarray(cam.distortion_coefs, dtype=float).reshape(-1).copy()
    fits = []
    for view in views:
        objects = np.asarray(view.object_points, dtype=float).reshape(-1, 3)
        pixels = np.asarray(view.image_points, dtype=float).reshape(-1, 2)
        measured = pixels
        if np.any(coefficients):
            measured = cv2.undistortPoints(pixels.reshape(-1, 1, 2), matrix, coefficients,
                                           P=matrix).reshape(-1, 2)
        try:
            pose = pose_from_homography(matrix, homography_of(objects, measured))
        except (ValueError, np.linalg.LinAlgError):
            fits.append(None)
            continue
        camera_frame = objects @ pose[:3, :3].T + pose[:3, 3]
        ideal = camera_frame @ matrix.T
        ideal = ideal[:, :2] / ideal[:, 2:3]
        rms = float(np.sqrt(np.mean(np.sum((ideal - measured) ** 2, axis=1))))
        # drawn against the detected pixels: with a lens that has distortion the arrows keep it
        projected = ideal
        if np.any(coefficients):
            rvec, _ = cv2.Rodrigues(pose[:3, :3])
            projected, _ = cv2.projectPoints(objects, rvec, pose[:3, 3], matrix, coefficients)
            projected = projected.reshape(-1, 2)
        fits.append(ViewFit(view, np.asarray(pose[:3, :3], dtype=float),
                            np.asarray(pose[:3, 3], dtype=float).reshape(3), projected, rms))
    return fits


def pool_by_image(fits: Sequence[ViewFit | None]) -> dict[int, float]:
    """Per-image RMS pooled over that image's boards, weighted by points (as the saved error)."""
    from pyCamSet.utils.intrinsics_report import _pool
    grouped: dict[int, tuple[list[float], list[int]]] = {}
    for fit in fits:
        if fit is None or not np.isfinite(fit.rms):
            continue
        rms, counts = grouped.setdefault(fit.view.image_index, ([], []))
        rms.append(fit.rms)
        counts.append(len(fit.view.object_points))
    return {image: _pool(np.asarray(r), np.asarray(c)) for image, (r, c) in sorted(grouped.items())}


def pooled_rms(fits: Sequence[ViewFit | None]) -> float:
    """One camera's RMS over all its solved views (the saved per-camera fit error)."""
    from pyCamSet.utils.intrinsics_report import _pool
    good = [f for f in fits if f is not None and np.isfinite(f.rms)]
    if not good:
        return float("nan")
    return _pool(np.asarray([f.rms for f in good]),
                 np.asarray([len(f.view.object_points) for f in good]))


def saved_series(metadata: Mapping[str, Any], camera: str) -> dict[int, float]:
    """The saved per-image error of one camera: ``{image index: px}`` (finite values only)."""
    diagnostics = metadata.get("diagnostics") or {}
    per_view = diagnostics.get("D2.6_per_view_reprojection") if isinstance(diagnostics, Mapping) else None
    series = per_view.get(camera) if isinstance(per_view, Mapping) else None
    if not isinstance(series, Mapping):
        return {}
    values = series.get("rms_px") or ()
    indices = series.get("image_indices") or range(len(values))
    out = {}
    for image, value in zip(indices, values):
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out[int(image)] = number
    return out


def saved_pose_model(metadata: Mapping[str, Any], camera: str) -> str | None:
    diagnostics = metadata.get("diagnostics") or {}
    per_view = diagnostics.get("D2.6_per_view_reprojection") if isinstance(diagnostics, Mapping) else None
    series = per_view.get(camera) if isinstance(per_view, Mapping) else None
    return series.get("pose_model") if isinstance(series, Mapping) else None


def check_per_image(fits: Sequence[ViewFit | None], saved: Mapping[int, float]) -> PerImageCheck:
    """Compare the recomputed per-image error with the saved one (1e-6 relative)."""
    recomputed = pool_by_image(fits)
    compared = matched = 0
    worst = 0.0
    for image in sorted(set(recomputed) | set(saved)):
        a, b = recomputed.get(image), saved.get(image)
        compared += 1
        if a is None or b is None:
            worst = float("inf")
            continue
        difference = abs(a - b)
        worst = max(worst, difference)
        if difference <= max(1e-6, 1e-6 * abs(b)):
            matched += 1
    return PerImageCheck(recomputed, compared, matched, worst, bool(saved))


# --------------------------------------------------------------------------------------
# The first estimate's own views (trace order)


@dataclass
class SeedView:
    """One view the closed-form first estimate solved (trace ``view_index`` order)."""

    index: int
    image_index: int
    boards: tuple[int, ...]
    object_points: np.ndarray   # N x 3 (board frame for pinhole, target frame for telecentric)
    image_points: np.ndarray    # N x 2


def seed_views(detection: Any, target: Any, camera: str, minimum: int,
               telecentric: bool) -> list[SeedView]:
    """The views ``AbstractTarget.initial_calibration`` solved, in its order.

    Pinhole: one per board (as :func:`board_views`). Telecentric: one per image, every kept
    face together in the target frame, flat images skipped (an affine pose of a plane is
    two-fold ambiguous, so the seed does not use them).
    """
    from pyCamSet.calibration.telecentric import is_planar
    out: list[SeedView] = []
    n_boards = int(np.prod(target.point_local.shape[:-2]))
    cam_detection = detection.get(cam=camera)
    if not cam_detection.has_data():
        return out
    for im_idx in range(int(detection.max_ims)):
        data = cam_detection.get(global_im_num=im_idx).get_data()
        if data is None or len(data) == 0:
            continue
        keys = _keys(data)
        boards = [int(b) for b in np.unique(keys[:, :-1]) if b < n_boards]
        kept = np.zeros(len(keys), dtype=bool)
        kept_boards = []
        for board in boards:
            mask = np.squeeze(keys[:, :-1] == board)
            if int(np.sum(mask)) >= minimum:
                kept_boards.append(board)
                kept |= mask
                if not telecentric:
                    out.append(SeedView(
                        len(out), im_idx, (board,),
                        np.asarray(target.point_local[tuple(keys[mask].astype(int).T)],
                                   dtype=float).reshape(-1, 3),
                        np.asarray(data[mask, -2:], dtype=float).reshape(-1, 2)))
        if telecentric and kept.any():
            points = np.asarray(target.point_data[tuple(keys[kept].astype(int).T)],
                                dtype=float).reshape(-1, 3)
            if is_planar(points):
                continue
            out.append(SeedView(len(out), im_idx, tuple(kept_boards), points,
                                np.asarray(data[kept, -2:], dtype=float).reshape(-1, 2)))
    return out


def seed_view_rms(cam: Any, view: SeedView, pose: Any) -> float:
    """RMS of a recorded first-estimate pose reprojected through the saved lens."""
    import cv2
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("the recorded pose is not a valid 4x4 transform")
    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    if is_telecentric(cam):
        from pyCamSet.utils.intrinsics_report import telecentric_project
        projected = telecentric_project(cam, view.object_points, rvec.reshape(3), pose[:3, 3])
    else:
        projected, _ = cv2.projectPoints(
            view.object_points, rvec, pose[:3, 3], np.asarray(cam.intrinsic, dtype=float),
            np.asarray(cam.distortion_coefs, dtype=float).reshape(-1))
        projected = projected.reshape(-1, 2)
    residual = projected - view.image_points
    return float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))


# --------------------------------------------------------------------------------------
# Lens model


@dataclass
class LensSummary:
    """Plain numbers for the lens-model figure of one camera."""

    model: str                       # "telecentric" | "pinhole"
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    coefficients: tuple[float, ...]
    fitted: bool                     # any non-zero distortion coefficient
    um_per_px: tuple[float, float] | None = None     # telecentric: object size of one pixel
    field_mm: tuple[float, float] | None = None      # telecentric: field of view
    fov_deg: tuple[float, float] | None = None       # pinhole: field-of-view angles


def lens_summary(cam: Any, image_size: tuple[int, int],
                 telecentric: bool | None = None) -> LensSummary:
    """Numbers of one camera's saved lens; ``image_size`` is (width, height) in pixels.
    ``telecentric`` overrides the camera type (for a lens rebuilt from the run record)."""
    matrix = np.asarray(cam.intrinsic, dtype=float)
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    coefficients = tuple(float(v) for v in np.asarray(cam.distortion_coefs, dtype=float).reshape(-1))
    width, height = int(image_size[0]), int(image_size[1])
    fitted = any(v != 0.0 for v in coefficients)
    if is_telecentric(cam) if telecentric is None else telecentric:
        # magnification is pixels per metre of the target
        um = (1e6 / fx if fx else float("nan"), 1e6 / fy if fy else float("nan"))
        field_mm = (width * um[0] / 1000.0, height * um[1] / 1000.0)
        return LensSummary("telecentric", width, height, fx, fy, cx, cy, coefficients, fitted,
                           um_per_px=um, field_mm=field_mm)
    fov = (math.degrees(2 * math.atan2(width / 2, fx)) if fx else float("nan"),
           math.degrees(2 * math.atan2(height / 2, fy)) if fy else float("nan"))
    return LensSummary("pinhole", width, height, fx, fy, cx, cy, coefficients, fitted,
                       fov_deg=fov)


def distortion_grid(cam: Any, image_size: tuple[int, int], exaggerate: float = 1.0,
                    lines: int = 9, samples: int = 60, telecentric: bool | None = None
                    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Straight pixel grid lines and the same lines bent by the camera's own distortion.

    The distortion is applied about the principal point with the coefficients multiplied by
    ``exaggerate``. Returns (straight, bent), lists of M x 2 pixel polylines.
    """
    width, height = image_size
    xs = np.linspace(0, width - 1, lines)
    ys = np.linspace(0, height - 1, lines)
    straight: list[np.ndarray] = []
    t = np.linspace(0, 1, samples)
    for x in xs:
        straight.append(np.column_stack([np.full_like(t, x), t * (height - 1)]))
    for y in ys:
        straight.append(np.column_stack([t * (width - 1), np.full_like(t, y)]))
    tele = is_telecentric(cam) if telecentric is None else telecentric
    bent = [_distort_pixels(cam, line, exaggerate, tele) for line in straight]
    return straight, bent


def _distort_pixels(cam: Any, pixels: np.ndarray, exaggerate: float,
                    telecentric: bool) -> np.ndarray:
    matrix = np.asarray(cam.intrinsic, dtype=float)
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    k = np.asarray(cam.distortion_coefs, dtype=float).reshape(-1) * float(exaggerate)
    if not np.any(k):
        return pixels.copy()
    dx, dy = pixels[:, 0] - cx, pixels[:, 1] - cy
    if telecentric:
        from pyCamSet.cameras.telecentric_camera import R2_SCALE
        den = 1.0 / (1.0 + k[0] * (dx * dx + dy * dy) * R2_SCALE)
        return np.column_stack([dx * den + cx, dy * den + cy])
    x, y = dx / fx, dy / fy
    k = np.concatenate([k, np.zeros(max(0, 5 - len(k)))])
    k1, k2, p1, p2, k3 = k[:5]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.column_stack([xd * fx + cx, yd * fy + cy])


# --------------------------------------------------------------------------------------
# Photos


@dataclass
class PhotoSource:
    """Where this run's photos are, and whether they may be shown."""

    status: str
    reason: str | None
    root: Path | None = None
    subfolder: str | None = None
    names: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.status.startswith("verified") and self.root is not None and bool(self.names)

    def path(self, camera: str, image_index: int) -> Path | None:
        names = self.names.get(str(camera)) or ()
        if not self.available or not 0 <= int(image_index) < len(names):
            return None
        base = (self.root / camera / self.subfolder) if self.subfolder else (self.root / camera)
        base = base.resolve()
        candidate = (base / names[int(image_index)]).resolve()
        try:
            candidate.relative_to(base)
        except ValueError:
            return None
        return candidate


def plain(value: Any) -> Any:
    """Thaw a frozen run record (MappingProxy / tuples) into dicts and lists."""
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [plain(item) for item in value]
    return value


def photo_source(context: Any) -> PhotoSource:
    """Photos of the Phase 1 run whose detections this run used, only when its manifest
    verifies them against today's folder (never a guessed file listing)."""
    try:
        from pyCamSet.workflow.education_trace import RunContext, resolve_exact_parent
        metadata = plain(context.metadata)
        run_context = RunContext.from_run(
            context.manager, {**metadata, "_run_dir": context.run_dir}, "phase2")
        result = run_context.phase1_source_check()
    except Exception:
        return PhotoSource("could not be checked", "the photo listing could not be checked")
    if not str(result.status).startswith("verified"):
        return PhotoSource(str(result.status), result.reason)
    try:
        parent = resolve_exact_parent(context.manager, metadata, "phase1") or {}
        params = parent.get("params") or {}
        root = Path(str(params.get("f_loc")))
    except Exception:
        return PhotoSource("could not be checked", "the Phase 1 image folder is unavailable")
    return PhotoSource(str(result.status), result.reason, root,
                       params.get("subfolder_string") or None,
                       {str(k): tuple(v) for k, v in result.photo_names.items()})


def load_photo(path: Path | None, max_side: int | None = None) -> np.ndarray | None:
    """A photo as 8-bit grey (1st-99.7th percentile stretch); ``None`` when it cannot be read.

    Full resolution unless ``max_side`` is given: then the photo is decoded at 1/2, 1/4 or 1/8
    size where the header allows, and resized so its longest side is at most ``max_side``.
    Phase 2 and 3 draw overlays in the photo's own pixel frame (``photo.shape`` is the extent),
    so their callers keep the default; only a caller that maps coordinates itself may pass
    ``max_side``.
    """
    if path is None:
        return None
    try:
        import cv2
        from pyCamSet.utils.paths import long_path
        factor = 1
        if max_side:
            from pyCamSet.workflow.education_visual_p1 import _reduction_for, photo_size
            factor = _reduction_for(photo_size(path), int(max_side))
        data = np.fromfile(str(long_path(path)), dtype=np.uint8)
        if factor > 1:
            image = cv2.imdecode(data, getattr(cv2, f"IMREAD_REDUCED_COLOR_{factor}"))
        else:
            image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None
    if image is None:
        return None
    if image.ndim == 3:
        image = image[..., :3].mean(axis=2)
    if max_side and max(image.shape[:2]) > int(max_side):
        scale = int(max_side) / float(max(image.shape[:2]))
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, [1.0, 99.7])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((image - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def shrink(image: np.ndarray, max_side: int) -> tuple[np.ndarray, int]:
    """Integer-stride shrink (fast, no new values) and the stride used."""
    stride = max(1, int(math.ceil(max(image.shape[:2]) / float(max_side))))
    return image[::stride, ::stride], stride


def image_size_of(cam: Any, photo: np.ndarray | None, points: np.ndarray | None = None
                  ) -> tuple[int, int]:
    """(width, height) of a camera's images: the photo's, else the larger of the saved resolution
    read either way round that holds every detected point."""
    if photo is not None:
        return int(photo.shape[1]), int(photo.shape[0])
    res = [int(v) for v in np.asarray(getattr(cam, "res", (0, 0))).reshape(-1)[:2]]
    if len(res) < 2 or min(res) <= 0:
        if points is not None and len(points):
            return int(np.ceil(points[:, 0].max())) + 1, int(np.ceil(points[:, 1].max())) + 1
        return 640, 480
    a, b = res
    if points is not None and len(points):
        xmax, ymax = float(points[:, 0].max()), float(points[:, 1].max())
        if xmax < a and ymax < b:
            return a, b
        if xmax < b and ymax < a:
            return b, a
    return a, b


def display_depths(fits: Sequence[ViewFit | None], telecentric: bool) -> np.ndarray:
    """Depth (camera z, metres) at which to draw each view. Pinhole: its own. Telecentric: the
    lens cannot see depth, so views are spread evenly along the axis for legibility."""
    n = len(fits)
    if not telecentric:
        return np.array([f.t[2] if f is not None else np.nan for f in fits], dtype=float)
    extent = _typical_extent(fits)
    return 3.0 * extent + np.arange(n) * (6.0 * extent / max(n - 1, 1))


def _typical_extent(fits: Sequence[ViewFit | None]) -> float:
    sizes = [float(np.ptp(f.view.object_points, axis=0).max()) for f in fits if f is not None]
    return float(np.median(sizes)) if sizes else 0.01


@dataclass
class SceneItem:
    """One thing to draw in the 3D stage: a posed board (pinhole) or a posed target (telecentric)."""

    image_index: int
    faces: list[np.ndarray]     # face outlines (4 corners each), camera frame, metres
    points: np.ndarray          # the detected corners' 3D positions, camera frame
    pixels: np.ndarray          # their detected pixels
    error: float                # px: the view's fit error (pinhole) / the photo's (telecentric)
    fit_index: int | None = None


def _face_corners(points: np.ndarray) -> np.ndarray:
    """The 4 extreme corners of a (nearly) planar grid of points, in order around it."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    centred = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    uv = centred @ vt[:2].T
    picks = [int(np.argmax(uv @ d)) for d in
             (np.array([-1, -1]), np.array([1, -1]), np.array([1, 1]), np.array([-1, 1]))]
    return pts[picks]


def scene_items(cam: Any, detection: Any, target: Any, camera: str, minimum: int,
                fits: Sequence[ViewFit | None], per_image: Mapping[int, float]
                ) -> list[SceneItem]:
    """What the 3D stage draws, all from this run's saved lens and detections.

    Pinhole: every board view (the part of the board it saw) at the pose of its fit (``fits``),
    coloured by that view's error.
    Telecentric: the whole target in every photo that showed it from more than one side, posed
    as the first estimate posed it (``calibration.telecentric.pose_from_affine`` at the saved
    magnification: the same deterministic step, so the same pose the run recorded), coloured by
    the photo's fit error. A telecentric pose has no depth (``t_z = 0``).
    """
    items: list[SceneItem] = []
    if is_telecentric(cam):
        from pyCamSet.calibration.telecentric import pose_from_affine
        faces_local = np.asarray(target.point_data, dtype=float)
        faces_local = faces_local.reshape(-1, faces_local.shape[-2], 3)
        m = np.asarray(cam.magnification, dtype=float)
        c = np.asarray(cam.principal_point, dtype=float)
        for view in seed_views(detection, target, camera, minimum, True):
            try:
                pose, _ = pose_from_affine(view.object_points, view.image_points, m, c)
            except (ValueError, np.linalg.LinAlgError):
                continue
            R, t = pose[:3, :3], pose[:3, 3]
            faces = [_face_corners(face) @ R.T + t for face in faces_local]
            items.append(SceneItem(view.image_index, faces, view.object_points @ R.T + t,
                                   view.image_points,
                                   float(per_image.get(view.image_index, float("nan")))))
        return items
    for k, fit in enumerate(fits):
        if fit is None:
            continue
        # the part of the board this view saw (a partly seen board would otherwise dwarf
        # the camera's field of view)
        outline = _face_corners(fit.view.object_points)
        items.append(SceneItem(fit.view.image_index, [outline @ fit.R.T + fit.t],
                               fit.view.object_points @ fit.R.T + fit.t, fit.view.image_points,
                               fit.rms, k))
    return items


def board_outline(target: Any, board: int) -> np.ndarray:
    """All of a board's corners in its own frame (the whole face, not only the detected ones)."""
    pts = np.asarray(target.point_local, dtype=float)
    index = np.unravel_index(int(board), pts.shape[:-2])
    return pts[index].reshape(-1, 3)


# --------------------------------------------------------------------------------------
# Teaching helpers (round-1 review: ED-3, ED-15, SCI-6)


@dataclass
class FaceWarp:
    """How one printed face lands on its photo (ED-15: pose from one view).

    ``kind`` is ``"affine"`` (telecentric: parallel lines stay parallel, so the warp holds the
    tilt and turn but no distance) or ``"homography"`` (pinhole: a perspective warp that also
    holds the distance). ``printed`` are every corner of the face in its own frame (metres,
    x/y), ``outline`` its four extreme corners, ``detected`` the view's own corners (printed
    frame), ``pixels`` their detected pixels, and ``warp(xy)`` maps printed x/y to pixels.
    """

    kind: str
    board: int
    printed: np.ndarray
    outline: np.ndarray
    detected: np.ndarray
    pixels: np.ndarray
    matrix: np.ndarray

    def warp(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        h = np.column_stack([xy, np.ones(len(xy))]) @ self.matrix.T
        if self.kind == "affine":
            return h[:, :2]
        return h[:, :2] / h[:, 2:3]

    @property
    def rms(self) -> float:
        """RMS (px) between the warped printed corners and the detected ones."""
        r = self.warp(self.detected) - self.pixels
        return float(np.sqrt(np.mean(np.sum(r * r, axis=1)))) if len(r) else float("nan")


def face_warp(view: BoardView, target: Any, telecentric: bool) -> FaceWarp | None:
    """Fit the warp from one face's printed corners to its detected pixels (None if < 4 points).

    Telecentric: a least-squares 2D affine map (the planar part of the affine camera the
    first estimate fits). Pinhole: the homography (``calibration.zhang.homography_of``, the
    first step of Zhang's method)."""
    obj = np.asarray(view.object_points, dtype=float).reshape(-1, 3)
    pix = np.asarray(view.image_points, dtype=float).reshape(-1, 2)
    if len(obj) < 4:
        return None
    try:
        printed = board_outline(target, view.board)
    except Exception:  # noqa: BLE001 - a target without per-face points: use the view's own
        printed = obj
    if telecentric:
        design = np.column_stack([obj[:, :2], np.ones(len(obj))])
        solution, *_ = np.linalg.lstsq(design, pix, rcond=None)
        matrix = np.vstack([solution.T, [0.0, 0.0, 1.0]])
        kind = "affine"
    else:
        try:
            from pyCamSet.calibration.zhang import homography_of
            matrix = np.asarray(homography_of(obj, pix), dtype=float)
        except Exception:  # noqa: BLE001
            return None
        kind = "homography"
    return FaceWarp(kind, int(view.board), printed[:, :2].copy(),
                    _face_corners(printed)[:, :2].copy(), obj[:, :2].copy(), pix, matrix)


def systematic_share(groups: Sequence[tuple[np.ndarray, np.ndarray]]) -> float:
    """How much of a photo's residuals follow a smooth pattern (0..1) rather than noise.

    ``groups`` are (pixels N x 2, residuals N x 2) per face. Each face's residuals are fitted
    by an affine field of position (a shift plus a linear change across the face); the share
    is the fitted energy over the total. Pure detection noise gives about 6 / (2 N) per face
    (about 0.15 for 20 corners); a pattern that is the same across a face gives close to 1.
    """
    fitted = total = 0.0
    for pixels, residuals in groups:
        pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
        residuals = np.asarray(residuals, dtype=float).reshape(-1, 2)
        if len(pixels) < 6:
            continue
        centred = pixels - pixels.mean(axis=0)
        design = np.column_stack([np.ones(len(centred)), centred])
        solution, *_ = np.linalg.lstsq(design, residuals, rcond=None)
        model = design @ solution
        fitted += float(np.sum(model * model))
        total += float(np.sum(residuals * residuals))
    return fitted / total if total > 0 else float("nan")


#: Columns of the smooth field fitted per face and per coordinate by :func:`pattern_test`:
#: a shift, a linear change across the face (x, y) and a curve (x², xy, y²).
PATTERN_TERMS = 6
#: Verdicts of :func:`pattern_test`, in the words the stage uses.
PATTERN_VERDICTS = ("noise-like", "some pattern", "pattern")


@dataclass
class PatternTest:
    """How much of a photo's errors a smooth field explains, against pure detection noise
    (SCI2-11, round 2).

    ``share`` is the fitted energy over the total (0..1). ``noise`` is what pure, independent
    detection noise would give on average (the fitted columns over the corners, pooled over
    the faces), ``noise_high`` its 99.9th percentile (a Beta distribution). The page uses
    :func:`simulated_pattern_test`, which replaces both with values simulated through the pose
    fit (SCI8-1). The verdict is
    ``"pattern"`` (well above noise and at least half the energy), ``"some pattern"`` (above
    noise) or ``"noise-like"``."""

    share: float
    noise: float
    noise_high: float
    corners: int
    #: sd of the 99.9 % bar across draw sets (SCI9-1); NaN when the bar is a formula value
    bar_sd: float = float("nan")
    #: SCI9-7: why the simulated baseline was not used (or lost draws), "" when none; the
    #: GUI writes it in the Numbers drawer instead of letting the formula numbers pass silently
    fallback: str = ""

    @property
    def testable(self) -> bool:
        return bool(self.corners) and bool(np.isfinite(self.share))

    @property
    def borderline(self) -> bool:
        """True when the share lies within two sd of the 99.9 % bar (SCI9-1): the simulated
        bar moves by a few points with the draws, so such a photo is not called either way."""
        return (self.testable and bool(np.isfinite(self.bar_sd))
                and abs(self.share - self.noise_high) <= 3.0 * self.bar_sd)

    @property
    def verdict(self) -> str:
        """One of :data:`PATTERN_VERDICTS`, ``"borderline"`` (see :attr:`borderline`), or
        ``"too few corners"`` when no face has enough."""
        if not self.testable:
            return "too few corners"
        if self.borderline:
            return "borderline"
        if self.share <= self.noise_high:
            return PATTERN_VERDICTS[0]
        return PATTERN_VERDICTS[2] if self.share >= 0.5 else PATTERN_VERDICTS[1]


def pattern_test(groups: Sequence[tuple[np.ndarray, np.ndarray]],
                 terms: int = PATTERN_TERMS) -> PatternTest:
    """Do a photo's residuals follow a smooth pattern across each face, beyond noise?

    ``groups`` are (pixels N x 2, residuals N x 2) per face. Each face's residuals are fitted
    by a smooth field of position: a shift and a linear change (``terms=3``), plus a curve
    (``terms=6``, the default: a bent lens or a bent target bends the errors, which a linear
    field misses; found on a real photo, whose arrows curve while their linear share is
    noise-like). Under pure noise each face's fitted share is ``terms / N``; pooled, the share
    follows Beta(sum terms, sum (N - terms)), whose 99.9th percentile is the bar for "above
    noise". Faces with fewer than ``2 * terms`` corners are skipped (too few to tell)."""
    from scipy.stats import beta

    fitted = total = 0.0
    fitted_dof = residual_dof = 0
    corners = 0
    for pixels, residuals in groups:
        pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
        residuals = np.asarray(residuals, dtype=float).reshape(-1, 2)
        n = len(pixels)
        if n < 2 * terms:
            continue
        centred = pixels - pixels.mean(axis=0)
        centred = centred / max(float(np.abs(centred).max()), 1e-12)
        x, y = centred[:, 0], centred[:, 1]
        columns = [np.ones(n), x, y, x * x, x * y, y * y][:terms]
        design = np.column_stack(columns)
        solution, *_ = np.linalg.lstsq(design, residuals, rcond=None)
        model = design @ solution
        fitted += float(np.sum(model * model))
        total += float(np.sum(residuals * residuals))
        fitted_dof += terms
        residual_dof += n - terms
        corners += n
    if total <= 0 or fitted_dof == 0:
        return PatternTest(float("nan"), float("nan"), float("nan"), corners)
    noise = fitted_dof / float(fitted_dof + residual_dof)
    high = float(beta.ppf(0.999, fitted_dof, residual_dof))
    return PatternTest(fitted / total, noise, high, corners)


def _beta_bar(shares: np.ndarray) -> float:
    """The 99.9 % bar of a Beta with these shares' mean and spread (method of moments), or NaN
    when the moments do not give a proper Beta."""
    from scipy.stats import beta
    mean = float(np.mean(shares))
    var = max(float(np.var(shares, ddof=1)), 1e-12)
    k = mean * (1.0 - mean) / var - 1.0
    if not (0.0 < mean < 1.0 and k > 0):
        return float("nan")
    return float(beta.ppf(0.999, mean * k, (1.0 - mean) * k))


def simulated_pattern_test(cam: Any, faces: Sequence[Any], draws: int = 600,
                           seed: int = 2026) -> PatternTest:
    """The pattern test of :func:`pattern_test`, with its noise baseline simulated through the
    page's own pose-only fit (SCI8-1).

    :func:`pattern_test` assumes all six columns per face are free. The pose fit takes up part
    of that, so pure noise gives a lower share than the Beta formula (a real photo, 13 and 35
    corners: about 16 % on average, not 25 %). Each draw adds iid Gaussian noise, with the
    photo's own residual RMS per coordinate, to the model's noiseless projection, refits every
    view with :func:`fit_views` and takes the pattern share. The noise mean is the average of
    the draws. The 99.9 % bar is a Beta with the draws' mean and spread (a method of moments),
    because 1 in 1,000 cannot be read from a few thousand draws.

    The bar itself moves with the draws (SCI9-1: a median of 4.8 points between seeds at 200
    draws). ``bar_sd`` is its bootstrap sd over the draws, and a photo within three of those sd of
    the bar is ``borderline``. The seed is fixed, so the value is the same on every call.

    ``faces`` are the photo's fits (``.view`` and ``.projected``, ``.residuals``). The real
    share comes from the real residuals; the baseline needs ``draws`` pose fits."""
    groups = [(f.view.image_points, f.residuals) for f in faces]
    real = pattern_test(groups)
    if not real.testable or not faces:
        return real
    sigma = float(np.sqrt(np.mean(np.concatenate([np.ravel(f.residuals) for f in faces]) ** 2)))
    rng = np.random.default_rng(seed)
    shares = []
    skipped = 0
    for _ in range(draws):
        views = []
        for f in faces:
            noisy = np.asarray(f.projected, dtype=float) + rng.normal(0.0, sigma,
                                                                    np.shape(f.projected))
            views.append(BoardView(f.view.index, f.view.image_index, f.view.board,
                                   f.view.object_points, noisy))
        fits = fit_views(cam, views)
        sim = pattern_test([(fit.view.image_points, fit.residuals)
                            for fit in fits if fit is not None])
        if sim.testable:
            shares.append(sim.share)
        else:
            skipped += 1
    s = np.asarray(shares, dtype=float)
    # SCI9-7: each fallback to the formula baseline says so in .fallback (shown in Numbers)
    if s.size < 2:
        return replace(real, fallback=f"simulated baseline not used: {s.size} of {draws} "
                                      "draws testable; the formula baseline is shown")
    mean = float(np.mean(s))
    high = _beta_bar(s)
    if not np.isfinite(high):
        return replace(real, fallback="simulated baseline not used: its 99.9 % bar is not "
                                      "finite; the formula baseline is shown")
    # bootstrap sd of the bar over resamples of these draws (seeded, so deterministic)
    boot_rng = np.random.default_rng(seed + 1)
    boot = [_beta_bar(boot_rng.choice(s, size=s.size, replace=True)) for _ in range(200)]
    boot = [b for b in boot if np.isfinite(b)]
    bar_sd = float(np.std(boot, ddof=1)) if len(boot) > 1 else float("nan")
    note = (f"{skipped} of {draws} simulated draws not testable and left out"
            if skipped else "")
    return PatternTest(real.share, mean, high, real.corners, bar_sd, note)


@dataclass
class CentreCheck:
    """The saved principal point against this camera's real image centre (SCI-6)."""

    saved: tuple[float, float]
    centre: tuple[float, float]
    transposed: bool        # saved == (height/2, width/2): an older run read the size swapped
    offset_px: float

    @property
    def matches(self) -> bool:
        return self.offset_px <= 1.0


def centre_check(cx: float, cy: float, image_size: tuple[int, int]) -> CentreCheck:
    """Compare a saved principal point with (width/2, height/2) of the real images.

    Runs before the 2026-10-08 ``Camera.res`` fix saved a telecentric seed's principal point
    at half the *transposed* size (e.g. (224, 187.5) for 375 x 448 photos); new runs save
    (width/2, height/2). For a fitted pinhole principal point an offset is normal."""
    width, height = float(image_size[0]), float(image_size[1])
    centre = (width / 2.0, height / 2.0)
    offset = math.hypot(cx - centre[0], cy - centre[1])
    swapped = math.hypot(cx - height / 2.0, cy - width / 2.0)
    transposed = abs(width - height) > 1.0 and swapped <= 1.0 and offset > 1.0
    return CentreCheck((float(cx), float(cy)), centre, transposed, offset)


@dataclass
class RadialProfile:
    """Outward part of every error against its distance from a centre (lens bending check)."""

    radius: np.ndarray
    outward: np.ndarray
    bin_mid: np.ndarray
    bin_median: np.ndarray
    rise_px: float           # change of the binned median from the inner to the outer bins
    bends: bool              # the rise is large next to the typical error
    #: ED3-12: each bin median's standard error (1.2533 sigma / sqrt(n), the median of a
    #: normal sample), so a plot can shade the band that noise alone gives (+/- 2 SE)
    bin_se: np.ndarray = field(default_factory=lambda: np.zeros(0))


def radial_profile(points: np.ndarray, residuals: np.ndarray, centre: Sequence[float],
                   bins: int = 8) -> RadialProfile:
    """Binned outward error vs distance from ``centre`` (pixels): barrel or pincushion
    distortion makes the outward error grow (or fall) steadily with the distance."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    residuals = np.asarray(residuals, dtype=float).reshape(-1, 2)
    offset = points - np.asarray(centre, dtype=float)
    radius = np.hypot(*offset.T)
    unit = offset / np.maximum(radius[:, None], 1e-9)
    outward = np.sum(residuals * unit, axis=1)
    edges = np.linspace(0, float(radius.max()) if len(radius) else 1.0, bins + 1)
    which = np.digitize(radius, edges)
    mids, med, se = [], [], []
    for k in range(1, len(edges)):
        sel = which == k
        if sel.sum() >= 10:
            mids.append((edges[k - 1] + edges[k]) / 2)
            med.append(float(np.median(outward[sel])))
            # standard error of a median: 1.2533 sigma / sqrt(n) for normal noise
            se.append(1.2533 * float(np.std(outward[sel], ddof=1)) / math.sqrt(int(sel.sum())))
    mids_a, med_a = np.asarray(mids), np.asarray(med)
    rise = 0.0
    if len(mids_a) >= 3:
        slope = np.polyfit(mids_a, med_a, 1)[0]
        rise = float(slope * (mids_a.max() - mids_a.min()))
    typical = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))) if len(residuals) else 1.0
    return RadialProfile(radius, outward, mids_a, med_a, rise, abs(rise) > 0.5 * typical,
                         np.asarray(se))


def error_map(points: np.ndarray, errors: np.ndarray, image_size: tuple[int, int],
              cells: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean error per image cell: (map rows x cols with NaN where empty, x edges, y edges)."""
    width, height = image_size
    cols = max(2, int(cells))
    rows = max(2, int(round(cells * height / max(width, 1))))
    xe = np.linspace(0, width, cols + 1)
    ye = np.linspace(0, height, rows + 1)
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    total, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=[ye, xe], weights=errors)
    count, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=[ye, xe])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return mean, xe, ye


def typical_corner(observed: np.ndarray, projected: np.ndarray,
                   photo: np.ndarray | None = None) -> int:
    """Index of a corner of typical error (ED-3 example): among the corners whose error is
    within 30 % of the photo's median error, the one whose neighbourhood in ``photo`` has the
    most contrast (a crisp black/white crossing reads best); without a photo, the corner
    closest to the median."""
    observed = np.asarray(observed, float).reshape(-1, 2)
    size = np.hypot(*(np.asarray(projected, float).reshape(-1, 2) - observed).T)
    if not len(size):
        return -1
    median = float(np.median(size))
    nearest = int(np.argmin(np.abs(size - median)))
    if photo is None:
        return nearest
    near = np.flatnonzero(np.abs(size - median) <= 0.3 * max(median, 1e-9))
    best, best_contrast = nearest, -1.0
    h, w = photo.shape[:2]
    for k in near:
        u, v = int(round(observed[k, 0])), int(round(observed[k, 1]))
        patch = photo[max(0, v - 4):min(h, v + 5), max(0, u - 4):min(w, u + 5)]
        if patch.size < 25:
            continue
        contrast = float(np.std(patch))
        if contrast > best_contrast:
            best, best_contrast = int(k), contrast
    return best


def pose_angles(R: np.ndarray) -> tuple[float, float]:
    """Tilt of a board towards the camera (degrees between its normal and the view axis) and
    its in-plane rotation (degrees)."""
    normal = R[:, 2]
    tilt = math.degrees(math.acos(min(1.0, abs(float(normal[2])))))
    spin = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return tilt, spin


__all__ = [
    "BoardView", "ViewFit", "DroppedBoard", "CameraViews", "PerImageCheck", "SeedView",
    "LensSummary", "PhotoSource", "board_views", "fit_views", "pool_by_image", "pooled_rms",
    "saved_series", "saved_pose_model", "check_per_image", "seed_views", "seed_view_rms",
    "lens_summary", "distortion_grid", "photo_source", "load_photo", "shrink", "image_size_of",
    "display_depths", "board_outline", "SceneItem", "scene_items", "pose_angles", "plain", "is_telecentric",
    "FaceWarp", "face_warp", "systematic_share", "PatternTest", "pattern_test", "CentreCheck", "centre_check", "RadialProfile",
    "radial_profile", "error_map", "typical_corner",
]
