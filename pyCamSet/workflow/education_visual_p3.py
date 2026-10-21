'''Qt-free preparation of Phase 3 education visuals from a saved run's own data.

Everything is evaluated on a *private, reloaded* handler that
:func:`pyCamSet.workflow.education_playback.prepare_saved_phase3_handler` has prepared:

* :func:`build_phase3_visual` evaluates the solver's own residual function at each recorded
  accepted step (or only at the saved start and result when no steps were recorded) and keeps,
  per state, the camera rotations (and centres for pinhole cameras), the target pose of every
  photo, every observed target corner rebuilt in 3D, and the pixel error of every observation.
* :func:`scene_frame` turns one state into a frame dict for ``education_scene3d`` (cameras,
  rebuilt corners coloured by error, optional rays for one camera).
* :func:`schur_structure`, :func:`reduced_camera_system` and :func:`block_updates` give the
  Schur picture: which parameter blocks each (camera, photo) pair touches, the reduced camera
  system the Schur solver hands to its linear solve, and how large each block's update was.

Conventions (as the solver stores them): camera extrinsics are world-to-camera
``x_cam = R x_world + t``; a telecentric camera stores a rotation only (no centre exists), so it
is *drawn* at a fixed display distance from the scene centre along its viewing axis. The world
frame is the reference photo's target frame. Camera and image indices are 0-based.
'''
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Display distance of a telecentric camera from the scene centre, in scene radii.
TELECENTRIC_DISPLAY_DISTANCE = 2.6
#: Most points / rays a static (matplotlib) 3D view is given; PyVista takes all points.
STATIC_POINT_LIMIT = 2500
RAY_LIMIT = 160


def _rotations(rvecs: np.ndarray) -> np.ndarray:
    """Rodrigues vectors (n, 3) to rotation matrices (n, 3, 3)."""
    from scipy.spatial.transform import Rotation
    rvecs = np.asarray(rvecs, dtype=np.float64).reshape(-1, 3)
    if not len(rvecs):
        return np.zeros((0, 3, 3))
    return Rotation.from_rotvec(rvecs).as_matrix()


def _angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Geodesic angle (degrees) between rotation stacks."""
    rel = np.einsum("nij,nkj->nik", Rb, Ra)
    trace = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


@dataclass
class StepState:
    """One solver state (the start, a recorded accepted step, or the saved result)."""

    nit: int
    cost: float
    rotations: np.ndarray            # (n_cam, 3, 3) world-to-camera
    translations: np.ndarray | None  # (n_cam, 3); None for telecentric cameras
    intrinsics: np.ndarray           # (n_cam, n_intr)
    pose_params: np.ndarray          # (n_images, 6) target-to-world rvec, t
    point_xyz: np.ndarray            # (n_points, 3) rebuilt observed corners, world frame
    point_error_px: np.ndarray       # (n_points,) mean error over the cameras that saw it
    row_error_px: np.ndarray         # (n_rows,) error of every observation
    camera_mean_px: np.ndarray       # (n_cam,) mean error per camera (nan without rows)
    mean_px: float                   # mean error over every observation
    #: (n_rows, 2) projected - observed of every observation, px (the solver's own residual
    #: sign: the predicted corner is ``observed + residual``); None for hand-built states.
    row_residual_xy: np.ndarray | None = None

    @property
    def rms_px(self) -> float:
        """Root-mean-square of the per-observation errors (the statistic Phase 2 reports)."""
        err = np.asarray(self.row_error_px, dtype=float)
        return float(np.sqrt(np.mean(err ** 2))) if err.size else float("nan")

    def centres(self, scene_centre: np.ndarray, distance: float) -> np.ndarray:
        """Camera centres; telecentric cameras at ``distance`` along their viewing axis."""
        if self.translations is not None:
            return -np.einsum("nji,nj->ni", self.rotations, self.translations)
        return scene_centre[None, :] - distance * self.rotations[:, 2, :]


@dataclass
class Phase3Visual:
    """Every Phase 3 picture's data, recomputed from the saved run."""

    camera_names: tuple[str, ...]
    telecentric: bool
    image_sizes: tuple[tuple[int, int], ...]
    rows_camera: np.ndarray          # (n_rows,) camera of each optimised observation
    rows_image: np.ndarray           # (n_rows,) photo of each optimised observation
    rows_xy: np.ndarray              # (n_rows, 2) observed pixel
    row_point: np.ndarray            # (n_rows,) index into the rebuilt points
    point_image: np.ndarray          # (n_points,) photo of each rebuilt point
    n_images: int
    states: list[StepState]
    source: str                      # "recorded steps" | "start and result only"
    loss: str = "linear"
    f_scale: float = 1.0
    solver: str = ""
    scene_centre: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scene_radius: float = 1.0
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    final_residual_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    #: The target's own points (n_keys, 3) in its frame, and each observation's point index.
    template: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    row_key: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    #: The photo whose target pose the solver fixes as a parameter (the world frame; the gauge).
    fixed_pose: int = 0

    @property
    def n_cameras(self) -> int:
        return len(self.camera_names)

    @property
    def camera_distance(self) -> float:
        """Display distance of telecentric cameras (scene units)."""
        return TELECENTRIC_DISPLAY_DISTANCE * self.scene_radius

    @property
    def posed_images(self) -> np.ndarray:
        """Photos with at least one optimised observation, ascending."""
        return np.unique(self.rows_image)

    def camera_image_counts(self) -> np.ndarray:
        """(n_cam, n_images) number of optimised observations per camera and photo."""
        counts = np.zeros((self.n_cameras, self.n_images), dtype=int)
        np.add.at(counts, (self.rows_camera, self.rows_image), 1)
        return counts

    def shared_photos(self) -> np.ndarray:
        """(n_cam, n_cam) number of photos both cameras observed in the solve."""
        seen = (self.camera_image_counts() > 0).astype(int)
        return seen @ seen.T

    def per_camera_image_error(self, state: int) -> dict[int, np.ndarray]:
        """Per camera: the mean error of each photo it observed (poster strip plot data)."""
        err = self.states[state].row_error_px
        result = {}
        for cam in range(self.n_cameras):
            mask = self.rows_camera == cam
            if not np.any(mask):
                result[cam] = np.array([])
                continue
            images = self.rows_image[mask]
            sums = np.bincount(images, weights=err[mask], minlength=self.n_images)
            counts = np.bincount(images, minlength=self.n_images)
            keep = counts > 0
            result[cam] = sums[keep] / counts[keep]
        return result

    def per_camera_image_ids(self, cam: int) -> np.ndarray:
        """The photos of :meth:`per_camera_image_error`'s values for ``cam``, in order."""
        return np.unique(self.rows_image[self.rows_camera == cam])


def _residual_vectors(loss_fn: Any, x: np.ndarray, n_rows: int) -> np.ndarray:
    """(n_rows, 2) projected - observed, px, from the solver's own residual function."""
    residuals = np.asarray(loss_fn(np.asarray(x, dtype=np.float64).copy()), dtype=np.float64)
    if residuals.size < 2 * n_rows:
        raise ValueError("residual vector is shorter than the optimised observations")
    return residuals[:2 * n_rows].reshape(-1, 2)


def _residual_magnitudes(loss_fn: Any, x: np.ndarray, n_rows: int) -> np.ndarray:
    return np.linalg.norm(_residual_vectors(loss_fn, x, n_rows), axis=1)


def _state(handler: Any, loss_fn: Any, x: np.ndarray, nit: int, cost: float,
           rows: np.ndarray, point_image: np.ndarray,
           point_key: np.ndarray, template: np.ndarray, row_point: np.ndarray,
           n_cams: int) -> StepState:
    primitive = handler.bundlePrimitive
    names = ("intr", "extr", "poses")
    snapshots = {name: getattr(primitive, name).copy() for name in names}
    try:
        intr, extr, poses = (np.asarray(v, dtype=np.float64).copy()
                             for v in primitive.return_bundle_primitives(
                                 np.asarray(x, dtype=np.float64).copy())[:3])
    finally:
        for name in names:
            getattr(primitive, name)[...] = snapshots[name]
    rotations = _rotations(extr[:, :3])
    translations = extr[:, 3:6].copy() if extr.shape[1] >= 6 else None
    pose_R = _rotations(poses[:, :3])
    local = template[point_key]
    xyz = np.einsum("nij,nj->ni", pose_R[point_image], local) + poses[point_image, 3:6]
    row_xy = _residual_vectors(loss_fn, x, len(rows))
    row_err = np.linalg.norm(row_xy, axis=1)
    sums = np.bincount(row_point, weights=row_err, minlength=len(point_image))
    counts = np.bincount(row_point, minlength=len(point_image))
    point_err = sums / np.maximum(counts, 1)
    cams = rows[:, 0].astype(int)
    cam_sum = np.bincount(cams, weights=row_err, minlength=n_cams)
    cam_n = np.bincount(cams, minlength=n_cams)
    with np.errstate(invalid="ignore", divide="ignore"):
        cam_mean = np.where(cam_n > 0, cam_sum / np.maximum(cam_n, 1), np.nan)
    return StepState(int(nit), float(cost), rotations, translations, intr, poses, xyz,
                     point_err, row_err, cam_mean, float(row_err.mean()) if len(row_err) else np.nan,
                     row_residual_xy=row_xy.astype(np.float32))


def build_phase3_visual(handler: Any, params: Sequence[np.ndarray], nits: Sequence[int],
                        costs: Sequence[float], *, source: str = "recorded steps",
                        loss: str = "linear", f_scale: float = 1.0,
                        solver: str = "") -> Phase3Visual:
    """Evaluate every given solver vector on a prepared, private handler.

    ``params`` are the recorded accepted-step vectors (first = the solver start, last = the
    saved result), or just ``[start, result]`` with ``source="start and result only"``.
    """
    if not len(params):
        raise ValueError("no solver states to show")
    rows = np.asarray(handler._flat_detections(), dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] < 5 or not len(rows):
        raise ValueError("no optimised observations")
    if rows.shape[1] != 5:
        raise ValueError("observation keys are not flat point indices")
    template = np.asarray(handler.target.point_data, dtype=np.float64).reshape(-1, 3)
    cams = rows[:, 0].astype(int)
    images = rows[:, 1].astype(int)
    keys = rows[:, 2].astype(int)
    if np.any(keys < 0) or np.any(keys >= len(template)):
        raise ValueError("observation key outside the target")
    n_images = int(handler.bundlePrimitive.poses.shape[0])
    pair = images.astype(np.int64) * len(template) + keys
    unique_pairs, row_point = np.unique(pair, return_inverse=True)
    point_image = (unique_pairs // len(template)).astype(int)
    point_key = (unique_pairs % len(template)).astype(int)
    names = tuple(str(name) for name in getattr(handler, "cam_names", ()))
    n_cams = int(handler.bundlePrimitive.extr.shape[0])
    if len(names) != n_cams:
        names = tuple(f"camera {index + 1}" for index in range(n_cams))
    loss_fn = handler.make_loss_fun(1)
    states = [_state(handler, loss_fn, x, nit, cost, rows, point_image, point_key,
                     template, row_point.reshape(-1), n_cams)
              for x, nit, cost in zip(params, nits, costs)]
    sizes = []
    camset = getattr(handler, "camset", None)
    for index in range(n_cams):
        try:
            res = camset[index].res
            sizes.append((int(res[0]), int(res[1])))
        except Exception:
            sizes.append((0, 0))
    visual = Phase3Visual(
        camera_names=names, telecentric=bool(handler.bundlePrimitive.extr.shape[1] < 6),
        image_sizes=tuple(sizes), rows_camera=cams, rows_image=images,
        rows_xy=rows[:, 3:5].copy(), row_point=row_point.reshape(-1),
        point_image=point_image, n_images=n_images, states=states, source=source,
        loss=str(loss), f_scale=float(f_scale), solver=str(solver),
        template=template.copy(), row_key=keys.copy(), fixed_pose=_fixed_pose(handler))
    final = np.asarray(loss_fn(np.asarray(params[-1], dtype=np.float64).copy()), dtype=np.float64)
    visual.final_residual_xy = final[:2 * len(rows)].reshape(-1, 2).copy()
    _fit_scene(visual)
    return visual


def _fit_scene(visual: Phase3Visual) -> None:
    """Scene centre, radius and 'up' from the saved result (fixed for every frame)."""
    final = visual.states[-1]
    pts = final.point_xyz[np.all(np.isfinite(final.point_xyz), axis=1)]
    if len(pts):
        centre = np.median(pts, axis=0)
        radius = float(np.percentile(np.linalg.norm(pts - centre, axis=1), 95))
    else:
        centre, radius = np.zeros(3), 1.0
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    visual.scene_centre = centre
    visual.scene_radius = radius
    # Image 'up' is -y in each camera; the rig's up is their average, when they agree.
    up = -final.rotations[:, 1, :].mean(axis=0)
    norm = np.linalg.norm(up)
    visual.up = up / norm if norm > 0.3 else np.array([0.0, 0.0, 1.0])


def _fixed_pose(handler: Any) -> int:
    options = getattr(handler, "problem_opts", {}) or {}
    try:
        value = options.get("fixed_pose", 0) if isinstance(options, Mapping) else 0
        return int(value) if value is not None and not isinstance(value, bool) else 0
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------------------
# One target in the rig (UX-2, ED-2): the target drawn ONCE, in the frame of the photo whose
# pose the solver fixes as a parameter (the world frame), with the cameras around it. Each photo's own
# target pose is then a ghost of that target. Telecentric cameras are drawn at the same
# display distance as Phase 4 (SCI-18): 1.35 target sizes from the target's centre.

#: Telecentric display distance in target sizes (bounding-box diagonals), as Phase 4.
TELECENTRIC_RIG_DISTANCE = 1.35
#: Exaggeration factors offered for the solver's motion (labelled "×N" wherever used).
MOTION_FACTORS = (1, 2, 5, 10, 20)


def _pose_matrix(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Target-to-world (R, t) from one stored pose row (rvec, t)."""
    R = _rotations(np.asarray(params[:3], dtype=float))[0]
    return R, np.asarray(params[3:6], dtype=float)


def reference_pose(visual: Phase3Visual, state: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Target-to-world (R, t) of the fixed photo (identity when it was not posed)."""
    posed = set(int(i) for i in visual.posed_images)
    if visual.fixed_pose in posed and 0 <= visual.fixed_pose < len(visual.states[state].pose_params):
        return _pose_matrix(visual.states[state].pose_params[visual.fixed_pose])
    return np.eye(3), np.zeros(3)


def target_world_points(visual: Phase3Visual) -> np.ndarray:
    """The target's points where the fixed photo put them (world frame), (n_keys, 3)."""
    R, t = reference_pose(visual)
    return np.asarray(visual.template, dtype=float) @ R.T + t


def target_centre_size(visual: Phase3Visual) -> tuple[np.ndarray, float]:
    """Centre (world) and size (bounding-box diagonal) of the drawn target."""
    pts = target_world_points(visual)
    pts = pts[np.all(np.isfinite(pts), axis=1)] if len(pts) else pts
    if not len(pts):
        return visual.scene_centre, max(2.0 * visual.scene_radius, 1e-9)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    size = float(np.linalg.norm(hi - lo))
    return (lo + hi) / 2, size if size > 0 else 1.0


def target_edges(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Outline of a target in its own frame: the 12 edges of its box (a cube target), or the
    4 edges of its rectangle when it is flat (a board). Returns (origins, ends)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if not len(pts):
        return np.zeros((0, 3)), np.zeros((0, 3))
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = hi - lo
    flat = span <= 1e-6 * max(float(span.max()), 1e-12)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])])
    # Corner k = 4·x_bit + 2·y_bit + z_bit, so an edge along axis j flips bit (2 - j).
    pairs = [(a, b) for a in range(8) for b in range(a + 1, 8)
             if bin(a ^ b).count("1") == 1
             and not flat[2 - int(np.log2(a ^ b))]]
    if flat.any():  # drop the duplicated face of a flat target
        keep = {k for k in range(8) if all(((k >> (2 - axis)) & 1) == 0
                                           for axis in np.flatnonzero(flat))}
        pairs = [(a, b) for a, b in pairs if a in keep and b in keep]
    return corners[[a for a, _ in pairs]], corners[[b for _, b in pairs]]


def corner_errors(visual: Phase3Visual, state: int) -> np.ndarray:
    """Each target point's mean error over every photo and camera that saw it (px; nan for a
    point never seen). The colour of the one drawn target (as Phase 4 stage 1)."""
    n = len(visual.template)
    err = visual.states[state].row_error_px
    keys = np.asarray(visual.row_key, dtype=int)
    if not n or len(keys) != len(err):
        return np.full(n, np.nan)
    sums = np.bincount(keys, weights=err, minlength=n)
    counts = np.bincount(keys, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def photo_errors(visual: Phase3Visual, state: int) -> np.ndarray:
    """Each photo's mean error over every observation in it (nan for photos not in the solve)."""
    err = visual.states[state].row_error_px
    sums = np.bincount(visual.rows_image, weights=err, minlength=visual.n_images)
    counts = np.bincount(visual.rows_image, minlength=visual.n_images)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def _exaggerate_rotation(R: np.ndarray, R_final: np.ndarray, factor: float,
                         limit_deg: float = 75.0) -> np.ndarray:
    """``R`` moved ``factor`` times further from ``R_final`` (``R = D @ R_final``, shown as
    ``D**factor @ R_final``), capped at ``limit_deg`` from ``R_final``."""
    from scipy.spatial.transform import Rotation
    delta = (Rotation.from_matrix(R) * Rotation.from_matrix(R_final).inv()).as_rotvec()
    angle = float(np.linalg.norm(delta))
    scaled = delta * factor
    cap = np.radians(limit_deg)
    if angle * factor > cap and angle > 0:
        scaled = delta / angle * cap
    return (Rotation.from_rotvec(scaled) * Rotation.from_matrix(R_final)).as_matrix()


def rig_motion(visual: Phase3Visual, factor: float = 1.0) -> dict[str, np.ndarray]:
    """Drawn rig of every state with the solver's motion exaggerated ``factor``×.

    Every state is shown relative to the saved result: a camera's (or photo pose's) change
    from the result is multiplied by ``factor`` (rotation and centre), so the final state is
    exactly the result and the start sits ``factor`` times further away. Returns
    ``R`` (n_states, n_cam, 3, 3) world-to-camera, ``C`` (n_states, n_cam, 3) drawn centres,
    ``pose_R`` (n_states, n_img, 3, 3) and ``pose_C`` (n_states, n_img, 3) target-to-world
    rotation and drawn centre of each photo's target (nan for photos without a pose).
    """
    centre, size = target_centre_size(visual)
    local_centre = np.nanmean(np.asarray(visual.template, dtype=float), axis=0) \
        if len(visual.template) else np.zeros(3)
    final = visual.states[-1]
    n_s, n_c, n_i = len(visual.states), visual.n_cameras, len(final.pose_params)
    R = np.zeros((n_s, n_c, 3, 3))
    C = np.zeros((n_s, n_c, 3))
    pR = np.full((n_s, n_i, 3, 3), np.nan)
    pC = np.full((n_s, n_i, 3), np.nan)
    posed = np.zeros(n_i, dtype=bool)
    posed[np.asarray(visual.posed_images, dtype=int)] = True

    def cam_centre(s: StepState, cam: int) -> np.ndarray:
        if s.translations is not None:
            return -s.rotations[cam].T @ s.translations[cam]
        return centre - TELECENTRIC_RIG_DISTANCE * size * s.rotations[cam][2]

    final_C = np.array([cam_centre(final, c) for c in range(n_c)])
    final_pose = [_pose_matrix(p) for p in final.pose_params]
    final_pC = np.array([Rp @ local_centre + tp for Rp, tp in final_pose])
    for k, s in enumerate(visual.states):
        for c in range(n_c):
            Rk = _exaggerate_rotation(s.rotations[c], final.rotations[c], factor)
            R[k, c] = Rk
            if s.translations is not None:
                C[k, c] = final_C[c] + factor * (cam_centre(s, c) - final_C[c])
            else:
                C[k, c] = centre - TELECENTRIC_RIG_DISTANCE * size * Rk[2]
        for i in np.flatnonzero(posed):
            Rp, tp = _pose_matrix(s.pose_params[i])
            pR[k, i] = _exaggerate_rotation(Rp, final_pose[i][0], factor)
            pC[k, i] = final_pC[i] + factor * ((Rp @ local_centre + tp) - final_pC[i])
    return {"R": R, "C": C, "pose_R": pR, "pose_C": pC, "posed": posed,
            "local_centre": local_centre}


def motion_factor(visual: Phase3Visual, share: float = 0.5) -> int:
    """The exaggeration that makes the cameras' and photo poses' typical travel (start to
    result) about ``share`` of the target size: one of :data:`MOTION_FACTORS`."""
    if len(visual.states) < 2:
        return 1
    _centre, size = target_centre_size(visual)
    plain = rig_motion(visual, 1.0)
    cam = np.linalg.norm(plain["C"] - plain["C"][-1][None], axis=2).max(axis=0)
    # A rotation also turns the target's corners: count it at the target's radius.
    from scipy.spatial.transform import Rotation
    angles = np.array([[np.linalg.norm((Rotation.from_matrix(plain["R"][k, c])
                                        * Rotation.from_matrix(plain["R"][-1, c]).inv()
                                        ).as_rotvec()) for c in range(visual.n_cameras)]
                       for k in range(len(visual.states))]).max(axis=0)
    travel = np.maximum(cam, angles * 0.5 * size)
    poses = np.linalg.norm(plain["pose_C"] - plain["pose_C"][-1][None], axis=2)
    posed = plain["posed"]
    pose_travel = poses[:, posed].max(axis=0) if posed.any() else np.zeros(0)
    # Every camera must stay near the rig (its largest travel counts), photo poses by their
    # typical travel; never more than x20.
    pose_typical = (float(np.nanmedian(pose_travel[np.isfinite(pose_travel)]))
                    if np.isfinite(pose_travel).any() else 0.0)
    typical = max(float(np.max(travel)) if len(travel) else 0.0, pose_typical)
    if not np.isfinite(typical) or typical <= 0:
        return 1
    best = 1
    for f in MOTION_FACTORS:
        if f * typical <= share * size:
            best = f
    return int(best)


def posed_outline(visual: Phase3Visual, R: np.ndarray, centre: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """A photo's target outline (edges) for target-to-world rotation ``R`` with its centre
    drawn at ``centre``."""
    origins, ends = target_edges(visual.template)
    local = np.nanmean(np.asarray(visual.template, dtype=float), axis=0)
    return ((origins - local) @ R.T + centre, (ends - local) @ R.T + centre)


# --------------------------------------------------------------------------------------
# One camera's photo: detected corners and where the model predicts them (ED-1, ED-21)


def photo_rows(visual: Phase3Visual, camera: int, photo: int) -> np.ndarray:
    """Observation rows of ``camera`` in ``photo`` (indices into the optimised rows)."""
    return np.flatnonzero((visual.rows_camera == int(camera)) & (visual.rows_image == int(photo)))


def photo_points(visual: Phase3Visual, state: float, rows: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(observed, predicted, error) for ``rows`` at a (possibly fractional) state: between
    two recorded states the residual is interpolated linearly (display only)."""
    observed = visual.rows_xy[rows].astype(float)
    n = len(visual.states)
    s = float(np.clip(state, 0, n - 1))
    lo = int(np.floor(s))
    hi = min(lo + 1, n - 1)
    a = visual.states[lo].row_residual_xy
    b = visual.states[hi].row_residual_xy
    if a is None or b is None:
        return observed, observed.copy(), np.zeros(len(rows))
    w = s - lo
    residual = (1 - w) * np.asarray(a[rows], dtype=float) + w * np.asarray(b[rows], dtype=float)
    return observed, observed + residual, np.linalg.norm(residual, axis=1)


def camera_photo_errors(visual: Phase3Visual, state: int, camera: int) -> np.ndarray:
    """``camera``'s mean error in each photo at ``state`` (nan where it saw nothing)."""
    mask = visual.rows_camera == int(camera)
    err = visual.states[state].row_error_px[mask]
    images = visual.rows_image[mask]
    sums = np.bincount(images, weights=err, minlength=visual.n_images)
    counts = np.bincount(images, minlength=visual.n_images)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def most_improved_camera(visual: Phase3Visual) -> int:
    """The camera whose mean error the solve reduced most (the clearest one to watch)."""
    start = np.asarray(visual.states[0].camera_mean_px, dtype=float)
    end = np.asarray(visual.states[-1].camera_mean_px, dtype=float)
    gain = np.where(np.isfinite(start) & np.isfinite(end), start - end, -np.inf)
    return int(np.argmax(gain)) if np.isfinite(gain).any() else 0


def photo_choices(visual: Phase3Visual, camera: int) -> dict[int, int]:
    """Photos ``camera`` observed in the solve -> number of corners it saw in each."""
    mask = visual.rows_camera == int(camera)
    images, counts = np.unique(visual.rows_image[mask], return_counts=True)
    return {int(i): int(c) for i, c in zip(images, counts)}


def typical_photo(visual: Phase3Visual, camera: int) -> int | None:
    """A representative photo of ``camera``: among the photos where it saw at least half its
    usual number of corners, the one whose start error is the median (not the best or worst)."""
    choices = photo_choices(visual, camera)
    if not choices:
        return None
    counts = np.array(list(choices.values()), dtype=float)
    images = np.array(list(choices), dtype=int)
    keep = counts >= 0.5 * np.median(counts)
    images = images[keep]
    start = camera_photo_errors(visual, 0, camera)[images]
    if not np.isfinite(start).any():
        return int(images[0])
    order = np.argsort(np.where(np.isfinite(start), start, np.inf))
    finite = order[: int(np.isfinite(start).sum())]
    return int(images[finite[len(finite) // 2]])


def photo_crop(observed: np.ndarray, predicted: Sequence[np.ndarray],
               size: tuple[int, int], margin: float = 0.18) -> tuple[float, float, float, float]:
    """A crop (x0, x1, y0, y1) around the detected and every predicted corner, with a margin,
    clipped to the photo of ``size`` (width, height)."""
    pts = np.vstack([np.asarray(observed, float)] + [np.asarray(p, float) for p in predicted])
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    w, h = size
    if not len(pts):
        return -0.5, w - 0.5, h - 0.5, -0.5
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = margin * max(float((hi - lo).max()), 20.0)
    x0, x1 = max(lo[0] - pad, -0.5), min(hi[0] + pad, w - 0.5)
    y0, y1 = max(lo[1] - pad, -0.5), min(hi[1] + pad, h - 0.5)
    return float(x0), float(x1), float(y1), float(y0)


# --------------------------------------------------------------------------------------
# Why the cameras must agree (ED-4): each camera's own pose of one photo vs the shared one


@dataclass
class CameraAgreement:
    """For one photo: where each camera, alone, put the target (its Phase 2 lens, its own
    pose), placed with the first guess's rig, and the one pose the first guess shared
    out (their consensus) and Phase 3 solved. All target-to-world 4x4 in the world frame of
    the fixed photo."""

    photo: int
    alone: dict[int, np.ndarray]
    shared: np.ndarray
    solved: np.ndarray | None
    axes: dict[int, np.ndarray]           # each camera's viewing direction (world)
    offset: dict[int, float]              # ghost centre to shared centre (target units)
    angle_deg: dict[int, float]           # ghost turn relative to the shared pose
    telecentric: bool
    depth_removed: bool                   # telecentric: offset along each axis not shown


def _invert(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def camera_agreement(target_in_camera: np.ndarray, extrinsics: np.ndarray, poses: np.ndarray,
                     template: np.ndarray, *, telecentric: bool, photo: int | None = None,
                     solved_pose: np.ndarray | None = None) -> CameraAgreement | None:
    """Ghost poses of one photo (see :class:`CameraAgreement`) from the first guess's own
    arrays (``education_derive._seed_from_inputs``): ``target_in_camera`` (c, p, 4, 4),
    ``extrinsics`` (c, 4, 4) world to camera, ``poses`` (p, 4, 4) target to world.

    ``photo`` None picks the photo that the most cameras placed on their own. A telecentric
    camera cannot see depth along its own axis, so that component of its ghost's offset is
    removed (``depth_removed``) rather than drawn as if it were measured.
    """
    tic = np.asarray(target_in_camera, dtype=float)
    ext = np.asarray(extrinsics, dtype=float)
    pos = np.asarray(poses, dtype=float)
    if tic.ndim != 4 or ext.ndim != 3 or pos.ndim != 3:
        return None
    seen = np.isfinite(tic[:, :, 0, 0]) & np.isfinite(ext[:, 0, 0])[:, None]
    placed = np.isfinite(pos[:, 0, 0])
    counts = seen.sum(axis=0) * placed
    if photo is None or not (0 <= photo < len(counts)) or counts[photo] < 2:
        if counts.max(initial=0) < 2:
            return None
        photo = int(np.argmax(counts))
    shared = pos[photo]
    local = np.nanmean(np.asarray(template, dtype=float).reshape(-1, 3), axis=0)
    shared_centre = shared[:3, :3] @ local + shared[:3, 3]
    alone, axes, offset, angle = {}, {}, {}, {}
    from scipy.spatial.transform import Rotation
    for cam in np.flatnonzero(seen[:, photo]):
        world = _invert(ext[cam]) @ tic[cam, photo]
        axis = _invert(ext[cam])[:3, :3] @ np.array([0.0, 0.0, 1.0])
        centre = world[:3, :3] @ local + world[:3, 3]
        delta = centre - shared_centre
        if telecentric:
            delta = delta - (delta @ axis) * axis
            world = world.copy()
            world[:3, 3] = shared_centre + delta - world[:3, :3] @ local
        alone[int(cam)] = world
        axes[int(cam)] = axis
        offset[int(cam)] = float(np.linalg.norm(delta))
        angle[int(cam)] = float(np.degrees(np.linalg.norm(
            (Rotation.from_matrix(world[:3, :3])
             * Rotation.from_matrix(shared[:3, :3]).inv()).as_rotvec())))
    return CameraAgreement(int(photo), alone, shared, solved_pose, axes, offset, angle,
                           bool(telecentric), bool(telecentric))


def seed_arrays(phase3_set: Any, phase2_set: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The first guess's per-camera target poses, rig and consensus poses, re-run read-only
    from the run's own inputs (the same call education_derive checks against D3.3)."""
    from pyCamSet.workflow.education_derive import _seed_from_inputs
    target_in_camera, extrinsics, poses, _errors = _seed_from_inputs(phase3_set, phase2_set)
    return (np.asarray(target_in_camera, dtype=float), np.asarray(extrinsics, dtype=float),
            np.asarray(poses, dtype=float))


# --------------------------------------------------------------------------------------
# What Phase 3 fits beyond Phase 2 (ED-5, SCI-5)


@dataclass
class LensChange:
    """One camera's lens, Phase 2 (the input) against Phase 3 (the result)."""

    camera: int
    telecentric: bool
    scale_before: tuple[float, float]       # magnification (px per unit) or focal length (px)
    scale_after: tuple[float, float]
    centre_before: tuple[float, float]       # principal point (px)
    centre_after: tuple[float, float]
    distortion_before: tuple[float, ...]
    distortion_after: tuple[float, ...]
    distortion_shift_px: float               # largest shift the fitted distortion makes
    eps_before: float                        # telecentric depth term (per target unit)
    eps_after: float
    image_size: tuple[int, int]

    @property
    def centre_shift_px(self) -> float:
        return float(np.hypot(self.centre_after[0] - self.centre_before[0],
                              self.centre_after[1] - self.centre_before[1]))

    @property
    def scale_change(self) -> float:
        """Relative change of the mean magnification / focal length."""
        before = float(np.mean(self.scale_before))
        return float(np.mean(self.scale_after)) / before - 1.0 if before else float("nan")

    @property
    def distortion_fitted(self) -> bool:
        return any(v != 0.0 for v in self.distortion_after)


def _distortion_shift(cam: Any, size: tuple[int, int], telecentric: bool) -> float:
    from pyCamSet.workflow.education_visual_p2 import distortion_grid
    straight, bent = distortion_grid(cam, size, 1.0, lines=7, samples=25,
                                     telecentric=telecentric)
    shifts = [np.linalg.norm(b - s, axis=1).max() for s, b in zip(straight, bent) if len(s)]
    return float(max(shifts)) if shifts else 0.0


def lens_changes(phase2_cams: Any, phase3_cams: Any,
                 image_sizes: Sequence[tuple[int, int]]) -> list[LensChange]:
    """Per camera: magnification / focal length, principal point, distortion and (telecentric)
    the depth term eps, before (Phase 2) and after (Phase 3), from the saved camsets."""
    from pyCamSet.cameras.telecentric_camera import TelecentricCamera
    result = []
    names = list(phase3_cams.get_names())
    for index, name in enumerate(names):
        after = phase3_cams[name]
        try:
            before = phase2_cams[name]
        except Exception:
            continue
        tele = isinstance(after, TelecentricCamera)
        ka, kb = np.asarray(after.intrinsic, float), np.asarray(before.intrinsic, float)
        size = tuple(int(v) for v in image_sizes[index]) if index < len(image_sizes) else (0, 0)
        if not all(size):
            size = tuple(int(v) for v in np.asarray(after.res, dtype=float)[:2])
        result.append(LensChange(
            index, tele, (float(kb[0, 0]), float(kb[1, 1])), (float(ka[0, 0]), float(ka[1, 1])),
            (float(kb[0, 2]), float(kb[1, 2])), (float(ka[0, 2]), float(ka[1, 2])),
            tuple(float(v) for v in np.asarray(before.distortion_coefs, float).ravel()),
            tuple(float(v) for v in np.asarray(after.distortion_coefs, float).ravel()),
            _distortion_shift(after, size, tele),
            float(getattr(before, "telecentricity", 0.0) or 0.0) if tele else 0.0,
            float(getattr(after, "telecentricity", 0.0) or 0.0) if tele else 0.0,
            size))
    return result


#: In a camera's intrinsic row (``Camera.to_param_vector``: m_x/f_x, c_x, m_y/f_y, c_y, then
#: the lens bending), the first column of the bending: distortion coefficients and, for a
#: telecentric camera, its depth term eps.
BENDING_FIRST_COLUMN = 4


@dataclass
class LensEffect:
    """What Phase 3's lens bending (distortion and, telecentric, eps) does to the corners: every
    observation's residual with the bending as solved (``on_xy``) and with it switched off and
    nothing re-fitted (``off_xy``), px. ``moved_xy`` = on - off: how far the bending moves each
    predicted corner (SCI2-1 / ED2-2: the effect, not the parameter)."""

    on_xy: np.ndarray
    off_xy: np.ndarray
    rows_camera: np.ndarray

    @property
    def moved_xy(self) -> np.ndarray:
        return self.on_xy - self.off_xy

    @property
    def moved_px(self) -> np.ndarray:
        return np.linalg.norm(self.moved_xy, axis=1)

    def camera_mean(self, which: str) -> dict[int, float]:
        err = np.linalg.norm(self.on_xy if which == "on" else self.off_xy, axis=1)
        return {int(c): float(err[self.rows_camera == c].mean())
                for c in np.unique(self.rows_camera)}

    def camera_largest_move(self) -> dict[int, float]:
        moved = self.moved_px
        return {int(c): float(moved[self.rows_camera == c].max())
                for c in np.unique(self.rows_camera)}


def lens_bending_effect(handler: Any, x: np.ndarray) -> LensEffect:
    """Evaluate the solver's own residual function at ``x`` with and without every camera's
    lens bending (intrinsic columns from :data:`BENDING_FIRST_COLUMN` set to 0, free and fixed
    cameras alike; the primitive is restored afterwards)."""
    primitive = handler.bundlePrimitive
    rows = np.asarray(handler._flat_detections(), dtype=np.float64)
    loss_fn = handler.make_loss_fun(1)
    x = np.asarray(x, dtype=np.float64).copy()
    on = _residual_vectors(loss_fn, x, len(rows))
    n_intr = int(primitive.n_intr)
    if n_intr <= BENDING_FIRST_COLUMN:
        return LensEffect(on, on.copy(), rows[:, 0].astype(int))
    free = np.asarray(primitive.intr_unfixed, dtype=bool)
    off_x = x.copy()
    block = off_x[:int(primitive.intr_end)].reshape(int(free.sum()), n_intr)
    block[:, BENDING_FIRST_COLUMN:] = 0.0
    saved = np.asarray(primitive.intr).copy()
    try:
        primitive.intr[:, BENDING_FIRST_COLUMN:] = 0.0
        off = _residual_vectors(loss_fn, off_x, len(rows))
    finally:
        primitive.intr[...] = saved
    return LensEffect(on, off, rows[:, 0].astype(int))


@dataclass
class LaterLens:
    """The same cameras' lens bending in a later run on the same photos (Phase 4): the
    largest distortion shift in the frame (px, as :class:`LensChange`) and eps (1/m)."""

    run_id: str
    distortion_shift_px: dict[int, float]
    eps_per_m: dict[int, float]


def eps_per_metre(eps: float, mm_per_unit: float | None) -> float | None:
    """A telecentric depth term in 1/m (Phase 4's unit), from eps per target unit. 1/m reads
    as % size change per cm of depth: w = 1/(1 + eps z), so eps = -3 /m draws a point 1 cm
    further away about 3 % larger."""
    if not mm_per_unit or not np.isfinite(eps):
        return None
    return float(eps) * 1000.0 / float(mm_per_unit)


def later_lens(phase4_metadata: Mapping | None, visual: Phase3Visual,
               mm_per_unit: float | None) -> LaterLens | None:
    """Phase 4's lenses (the run that continued from this one), matched by camera name."""
    if not isinstance(phase4_metadata, Mapping):
        return None
    try:
        from pyCamSet.cameras.telecentric_camera import TelecentricCamera
        from pyCamSet.utils.saving import load_CameraSet
        artifacts = phase4_metadata.get("artifacts") or {}
        path = artifacts.get("self_calibrated_camset") or artifacts.get("optimised_camset")
        if not path or not Path(str(path)).is_file():
            return None
        camset = load_CameraSet(Path(str(path)))
        names = list(camset.get_names())
        shift, eps = {}, {}
        for index, name in enumerate(visual.camera_names):
            if name not in names:
                continue
            cam = camset[name]
            tele = isinstance(cam, TelecentricCamera)
            size = visual.image_sizes[index]
            if not all(size):
                size = tuple(int(v) for v in np.asarray(cam.res, dtype=float)[:2])
            shift[index] = _distortion_shift(cam, size, tele)
            if tele:
                value = eps_per_metre(float(getattr(cam, "telecentricity", 0.0) or 0.0),
                                      mm_per_unit)
                if value is not None:
                    eps[index] = value
        return LaterLens(str(phase4_metadata.get("run_id") or ""), shift, eps) if shift else None
    except Exception:  # noqa: BLE001 - an unreadable later run just shows nothing
        return None


# --------------------------------------------------------------------------------------
# The join order's criterion (ED2-11): every link the grow step could take, and its score


@dataclass
class JoinCandidates:
    """For each recorded join: the disagreement (target units) of every link from a placed
    camera to an unplaced one at that moment (``options``) and of the one taken (``chosen``,
    always the smallest). ``reproduced``: the recomputed choices equal the recorded events."""

    chosen: list[float]
    options: list[list[float]]
    reproduced: bool


def join_candidates(target_in_camera: np.ndarray, length_scale: float,
                    events: Sequence[Mapping]) -> JoinCandidates | None:
    """Recompute the seed's camera links (``region_growing.camera_links``, the same seed and
    sampling) and replay the grow order; None when the links cannot be rebuilt."""
    if not events:
        return None
    try:
        from pyCamSet.optimisation.region_growing import camera_links
        tic = np.asarray(target_in_camera, dtype=float)
        observed = np.isfinite(tic[:, :, 0, 0])
        _links, spread = camera_links(tic, observed, float(length_scale),
                                      np.random.default_rng(0))
    except Exception:  # noqa: BLE001
        return None
    n = len(spread)
    placed = np.zeros(n, dtype=bool)
    first = int(events[0].get("joined_from", 0))
    placed[first] = True
    chosen, options, ok = [], [], True
    for event in events:
        a, b = int(event.get("joined_from", -1)), int(event.get("joining", -1))
        if not (0 <= a < n and 0 <= b < n):
            return None
        here = spread[np.ix_(placed, ~placed)]
        options.append(sorted(float(v) for v in here[np.isfinite(here)]))
        chosen.append(float(spread[a, b]))
        recorded = event.get("spread")
        if recorded is None or not np.isclose(float(recorded), spread[a, b], rtol=1e-6,
                                              atol=1e-12):
            ok = False
        if not np.isclose(spread[a, b], np.nanmin(np.where(np.isfinite(here), here, np.inf))):
            ok = False
        placed[b] = True
    return JoinCandidates(chosen, options, ok)


# --------------------------------------------------------------------------------------
# Single-face photos at L1 (ED2-10): the two candidate tilts drawn into a second camera


@dataclass
class TiltDemo:
    """One recovered single-face photo: camera ``proposer`` fits the face alone (two mirror
    tilts); each tilt, its turn held and only its position fitted over every camera that saw
    the photo, is projected into ``viewer``'s photo. ``rms`` per tilt is that fit's RMS over
    all cameras (equal to the saved D3.14 'tilt kept' value when ``matches_saved``)."""

    image: int
    proposer: int
    viewer: int
    found: np.ndarray                 # viewer's detected corners (n, 2)
    predicted: dict[str, np.ndarray]  # tilt "+"/"-" -> viewer's predicted corners (n, 2)
    rms: dict[str, float]
    kept: str                         # the tilt with the smaller rms
    matches_saved: bool


def tilt_demo(phase3_set: Any, phase2_set: Any, image: int,
              teaching: Sequence[Mapping] | None = None, *,
              seed: tuple | None = None) -> TiltDemo | None:
    """Rebuild one recovered photo's two candidate tilts from the run's own inputs (the
    Phase 2 lenses and the first guess's rig), as ``recover_single_face_telecentric_poses``
    does, read-only. None when the photo is not a single-face photo of telecentric cameras."""
    try:
        import cv2
        from scipy.optimize import least_squares

        from pyCamSet.optimisation.template_handler import _is_planar
        from pyCamSet.utils import general_utils as gu
        from pyCamSet.utils.intrinsics_report import _orthographic_starts, telecentric_project
        from pyCamSet.workflow.education_derive import _seed_from_inputs
        extrinsics = (seed if seed is not None else _seed_from_inputs(phase3_set, phase2_set))[1]
        handler = phase3_set.calibration_handler
        target, detection = handler.target, handler.detection
        names = list(phase2_set.get_names())
        cams = [phase2_set[name] for name in names]
        flat = detection.return_flattened_keys(target.point_data.shape[:-1]).get_data()
        points = np.asarray(target.point_data).reshape((-1, 3))
        rows = flat[flat[:, 1].astype(int) == int(image)]
        views = []
        for c in sorted(set(rows[:, 0].astype(int).tolist())):
            here = rows[rows[:, 0].astype(int) == c]
            obj = points[here[:, 2].astype(int)]
            if len(here) < 4 or not np.all(np.isfinite(extrinsics[c])):
                continue
            if not _is_planar(obj):
                return None
            views.append((c, obj, here[:, 3:5].astype(float)))
        if len(views) < 2:
            return None
        order = [str(e.get("camera")) for e in (teaching or [])]
        views.sort(key=lambda v: order.index(names[v[0]]) if names[v[0]] in order else 99)
        proposer = views[0]
        viewer = max(views[1:], key=lambda v: len(v[2]))

        def project(c, obj, x):
            in_camera = extrinsics[c] @ gu.make_4x4h_tform(x[:3], x[3:6])
            rvec, _ = cv2.Rodrigues(in_camera[:3, :3])
            return telecentric_project(cams[c], obj, rvec.reshape(3), in_camera[:3, 3])

        def residual(x):
            return np.concatenate([(project(c, obj, x) - pix).reshape(-1)
                                   for c, obj, pix in views])

        c, obj, pix = proposer
        starts = _orthographic_starts(obj, cams[c].undistort_points(pix), cams[c])
        predicted, rms = {}, {}
        for tilt, start in zip("+-", starts):
            in_camera = gu.make_4x4h_tform(start[:3], np.array([start[3], start[4], 0.0]))
            world = np.linalg.inv(extrinsics[c]) @ in_camera
            rvec, _ = cv2.Rodrigues(world[:3, :3])
            x0 = np.concatenate([rvec.reshape(3), world[:3, 3]])
            kept = least_squares(lambda t: residual(np.concatenate([x0[:3], t])), x0[3:])
            x = np.concatenate([x0[:3], kept.x])
            rms[tilt] = float(np.sqrt(np.mean(np.sum(kept.fun.reshape(-1, 2) ** 2, axis=1))))
            predicted[tilt] = project(viewer[0], viewer[1], x)
        saved = {str(e.get("tilt")).replace("−", "-"): e.get("all_cameras_tilt_kept_rms_px")
                 for e in (teaching or []) if str(e.get("camera")) == names[c]}
        matches = bool(saved) and all(
            tilt in saved and saved[tilt] is not None
            and np.isclose(rms[tilt], float(saved[tilt]), rtol=1e-3, atol=1e-3) for tilt in rms)
        return TiltDemo(int(image), int(c), int(viewer[0]), viewer[2], predicted, rms,
                        min(rms, key=rms.get), matches)
    except Exception:  # noqa: BLE001 - the stage falls back to the bar chart
        return None


def capped_turns(visual: Phase3Visual, factor: float, limit_deg: float = 75.0) -> int:
    """How many photo poses' exaggerated turns reach :func:`_exaggerate_rotation`'s cap at
    ``factor`` (SCI2-20: the subtitle says so when any do)."""
    if factor <= 1 or len(visual.states) < 2:
        return 0
    from scipy.spatial.transform import Rotation
    final = visual.states[-1]
    count = 0
    for i in np.asarray(visual.posed_images, dtype=int):
        Rf = _pose_matrix(final.pose_params[i])[0]
        worst = max(float(np.linalg.norm((Rotation.from_matrix(_pose_matrix(s.pose_params[i])[0])
                                          * Rotation.from_matrix(Rf).inv()).as_rotvec()))
                    for s in visual.states)
        if np.degrees(worst) * factor > limit_deg:
            count += 1
    return count


# --------------------------------------------------------------------------------------
# The error ladder (ED-4): each camera alone (Phase 2) -> one rig (Phase 3) -> refined target


@dataclass
class ErrorLadder:
    """Per-camera RMS error alone (Phase 2, saved D2.1), in the shared rig (Phase 3, the
    same statistic recomputed from the solve's own residuals) and, when a Phase 4 run continued
    from this one, after the target was refined (Phase 4's per-camera RMS, replayed from its
    saved result; SCI2-4: one statistic on every rung). ``phase4_mean`` is that run's saved
    mean (D4.3), kept for the Numbers drawer only: a mean is not drawn on the RMS axis."""

    phase2_rms: dict[int, float]
    phase3_rms: dict[int, float]
    phase3_mean: float
    phase4_mean: float | None = None
    phase4_run: str | None = None
    phase4_rms: dict[int, float] = field(default_factory=dict)


def phase4_camera_rms(phase4_metadata: Mapping | None) -> dict[str, float]:
    """Per-camera RMS error (by camera name) of a saved Phase 4 result, replayed through the handler Phase 4's
    own derivation rebuilds (``education_derive._phase4_rebuild``); empty unless the replayed
    mean reproduces the saved D4.3 final error (never a guessed value)."""
    if not isinstance(phase4_metadata, Mapping):
        return {}
    try:
        from pyCamSet.workflow.education_derive import _load_camset, _phase4_handler
        artifacts = phase4_metadata.get("artifacts") or {}
        path = artifacts.get("self_calibrated_camset") or artifacts.get("optimised_camset")
        diagnostics = phase4_metadata.get("diagnostics") or {}
        saved_mean = float(diagnostics.get("D4.3_final_euclid_px"))
        if not path or not Path(str(path)).is_file() or not np.isfinite(saved_mean):
            return {}
        camset = _load_camset(Path(str(path)))
        handler = _phase4_handler(Path(str(path)), camset, diagnostics)
        rows = np.asarray(handler._flat_detections(), dtype=np.float64)
        x = np.asarray(camset.calibration_params, dtype=np.float64).copy()
        residuals = np.asarray(handler.make_loss_fun(1)(x), dtype=np.float64)
        err = np.linalg.norm(residuals[:2 * len(rows)].reshape(-1, 2), axis=1)
        if not np.isclose(float(err.mean()), saved_mean, rtol=1e-6, atol=1e-9):
            return {}
        cams = rows[:, 0].astype(int)
        names = [str(n) for n in handler.cam_names]
        return {names[c]: float(np.sqrt(np.mean(err[cams == c] ** 2))) for c in np.unique(cams)}
    except Exception:  # noqa: BLE001 - an unreadable Phase 4 run just shows no rung
        return {}


def error_ladder(visual: Phase3Visual, phase2_metadata: Mapping | None,
                 phase4_metadata: Mapping | None = None,
                 phase4_rms: Mapping[str, float] | None = None) -> ErrorLadder:
    """Build the ladder; Phase 2 values are matched to cameras by name. ``phase4_rms`` is
    :func:`phase4_camera_rms` of ``phase4_metadata`` (passed in, it is slow to replay)."""
    final = visual.states[-1]
    p3 = {}
    for cam in range(visual.n_cameras):
        err = final.row_error_px[visual.rows_camera == cam]
        if err.size:
            p3[cam] = float(np.sqrt(np.mean(err ** 2)))
    p2 = {}
    saved = ((phase2_metadata or {}).get("diagnostics") or {}).get(
        "D2.1_per_camera_rms_reprojection") or {}
    if isinstance(saved, Mapping):
        for cam, name in enumerate(visual.camera_names):
            try:
                value = float(saved[name])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                p2[cam] = value
    p4, p4_run = None, None
    if isinstance(phase4_metadata, Mapping):
        try:
            value = float((phase4_metadata.get("diagnostics") or {}).get("D4.3_final_euclid_px"))
            if np.isfinite(value):
                p4, p4_run = value, str(phase4_metadata.get("run_id") or "")
        except (TypeError, ValueError):
            pass
    given = dict(phase4_rms or {})
    p4_rms = {cam: float(given[name]) for cam, name in enumerate(visual.camera_names)
              if name in given and np.isfinite(given[name])}
    return ErrorLadder(p2, p3, float(final.mean_px), p4, p4_run, p4_rms)


def photo_source(run_context: Any) -> Any:
    """Photos of the Phase 1 run behind this Phase 3 run, only when the used-input chain and
    Phase 1's manifest verify them (``education_visual_p2.PhotoSource``; never guessed)."""
    from pyCamSet.workflow.education_visual_p2 import PhotoSource
    try:
        from pyCamSet.workflow.education_trace import resolve_exact_parent
        result = run_context.phase1_source_check()
    except Exception:
        return PhotoSource("could not be checked", "the photo listing could not be checked")
    if not str(result.status).startswith("verified"):
        return PhotoSource(str(result.status), result.reason)
    try:
        phase2 = resolve_exact_parent(run_context.manager, run_context.metadata, "phase2") or {}
        phase1 = resolve_exact_parent(run_context.manager, phase2, "phase1") or {}
        params = phase1.get("params") or {}
        root = Path(str(params.get("f_loc")))
    except Exception:
        return PhotoSource("could not be checked", "the Phase 1 image folder is unavailable")
    return PhotoSource(str(result.status), result.reason, root,
                       params.get("subfolder_string") or None,
                       {str(k): tuple(v) for k, v in result.photo_names.items()})


# --------------------------------------------------------------------------------------
# Pose graph


def join_events(trace: Mapping | None, diagnostics: Mapping | None) -> list[dict]:
    """The seed's camera join order (trace record, else the saved diagnostics list)."""
    for record in (trace or {}).get("records", []) or []:
        if isinstance(record, Mapping) and record.get("kind") == "seed_join_order":
            events = (record.get("payload") or {}).get("events")
            if isinstance(events, (list, tuple)):
                return [dict(e) for e in events if isinstance(e, Mapping)]
    saved = (diagnostics or {}).get("seed_join_order")
    if isinstance(saved, (list, tuple)):
        return [dict(e) for e in saved if isinstance(e, Mapping)]
    return []


def graph_layout(visual: Phase3Visual | None, n_cams: int) -> np.ndarray:
    """2D node positions for the pose graph, laid out as the rig is seen from above.

    Each camera sits opposite its viewing direction, projected onto the plane normal to the
    rig's 'up'; steeper (more downward/upward looking) cameras sit nearer the middle. Falls back
    to a circle in camera order.
    """
    angles = 2 * np.pi * np.arange(n_cams) / max(n_cams, 1) + np.pi / 2
    circle = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    if visual is None or visual.n_cameras != n_cams or n_cams < 2:
        return circle
    up = visual.up
    ref = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = ref - up * (ref @ up)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    # Point 'front' (camera 1's side) towards the bottom of the picture.
    look = -visual.states[-1].rotations[:, 2, :]
    flat = np.stack([look @ e1, look @ e2], axis=1)
    length = np.linalg.norm(flat, axis=1)
    if np.any(length < 0.15):
        return circle
    theta = np.arctan2(flat[:, 1], flat[:, 0])
    theta = theta - theta[0] - np.pi / 2
    radius = 0.55 + 0.45 * (length - length.min()) / max(length.max() - length.min(), 1e-9)
    pos = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    # Push apart nodes that would overlap.
    for _ in range(60):
        moved = False
        for i in range(n_cams):
            for j in range(i + 1, n_cams):
                d = pos[j] - pos[i]
                dist = float(np.linalg.norm(d))
                if dist < 0.36:
                    push = (d / dist if dist > 1e-9 else np.array([1.0, 0])) * (0.36 - dist) / 2
                    pos[i] -= push
                    pos[j] += push
                    moved = True
        if not moved:
            break
    return pos


# --------------------------------------------------------------------------------------
# 3D frames


def _subsample(n: int, limit: int | None, seed: int = 3) -> np.ndarray:
    if limit is None or n <= limit:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=limit, replace=False))


def point_selection(visual: Phase3Visual, limit: int | None) -> np.ndarray:
    """A fixed subset of rebuilt points (the same indices in every frame)."""
    return _subsample(len(visual.point_image), limit)


def ray_selection(visual: Phase3Visual, camera: int, limit: int = RAY_LIMIT) -> np.ndarray:
    """A fixed subset of ``camera``'s observation rows for residual rays."""
    rows = np.flatnonzero(visual.rows_camera == camera)
    return rows[_subsample(len(rows), limit, seed=11)]


def camera_pose(visual: Phase3Visual, state: int, camera: int) -> dict[str, np.ndarray]:
    """World-to-camera (R, t) as drawn (telecentric: at the display distance)."""
    s = visual.states[state]
    R = s.rotations[camera]
    centre = s.centres(visual.scene_centre, visual.camera_distance)[camera]
    return {"R": R, "t": -R @ centre}


def viewing_axis(visual: Phase3Visual, state: int, camera: int,
                 reach: float = 0.72) -> tuple[np.ndarray, np.ndarray]:
    """A camera's viewing axis as a segment from its (drawn) centre towards the scene centre."""
    s = visual.states[state]
    R = s.rotations[camera]
    centre = s.centres(visual.scene_centre, visual.camera_distance)[camera]
    depth = float((visual.scene_centre - centre) @ R[2])
    return centre, centre + reach * max(depth, 0.0) * R[2]


def scene_frame(visual: Phase3Visual, state: int, *, points: np.ndarray | None = None,
                ray_camera: int | None = None, rays: np.ndarray | None = None,
                points_name: str = "corners", axes: bool = True) -> dict[str, dict[str, Any]]:
    """Frame dict for ``education_scene3d`` (cameras ``cam<i>``, points, viewing axes
    ``axis<i>``, optional residual rays ``rays``)."""
    s = visual.states[state]
    sel = point_selection(visual, None) if points is None else points
    frame: dict[str, dict[str, Any]] = {
        "cameras": {f"cam{i}": camera_pose(visual, state, i) for i in range(visual.n_cameras)},
        "points": {points_name: {"xyz": s.point_xyz[sel], "scalars": s.point_error_px[sel]}},
    }
    if axes:
        frame["rays"] = {}
        for i in range(visual.n_cameras):
            start, end = viewing_axis(visual, state, i)
            frame["rays"][f"axis{i}"] = {"origins": start[None, :], "ends": end[None, :]}
    if ray_camera is not None and rays is not None and len(rays):
        ends = s.point_xyz[visual.row_point[rays]]
        R = s.rotations[ray_camera]
        centre = s.centres(visual.scene_centre, visual.camera_distance)[ray_camera]
        if visual.telecentric:
            # Parallel rays: start each on the camera's image plane, straight behind the point.
            depth = (ends - centre) @ R[2]
            origins = ends - depth[:, None] * R[2][None, :]
        else:
            origins = np.repeat(centre[None, :], len(ends), axis=0)
        frame.setdefault("rays", {})["rays"] = {"origins": origins, "ends": ends,
                                                "scalars": s.row_error_px[rays]}
    return frame


# --------------------------------------------------------------------------------------
# Schur picture


@dataclass
class SchurStructure:
    """Which parameter blocks each (camera, photo) pair of observations touches."""

    pair_camera: np.ndarray       # (n_pairs,) sorted by photo then camera
    pair_image: np.ndarray
    pair_points: np.ndarray       # observations (corners) in each pair
    pose_order: np.ndarray        # posed photos, in pose-column order
    n_intr: int
    n_extr: int
    n_cameras: int
    n_rows: int                   # residual rows (2 per observation)
    n_params: int
    n_camera_params: int          # the retained (reduced system) size
    n_pose_params: int            # the eliminated size


def schur_structure(visual: Phase3Visual, n_intr: int, n_extr: int, n_params: int,
                    fixed_pose: int | None) -> SchurStructure:
    """The block structure of the Jacobian, from the optimised observations alone."""
    pairs = visual.rows_image.astype(np.int64) * visual.n_cameras + visual.rows_camera
    unique, counts = np.unique(pairs, return_counts=True)
    images = (unique // visual.n_cameras).astype(int)
    cams = (unique % visual.n_cameras).astype(int)
    posed = np.unique(images)
    if fixed_pose is not None:
        posed = posed[posed != int(fixed_pose)]
    n_cam_params = visual.n_cameras * (n_intr + n_extr)
    return SchurStructure(cams, images, counts.astype(int), posed, int(n_intr), int(n_extr),
                          visual.n_cameras, 2 * len(visual.rows_camera), int(n_params),
                          int(n_cam_params), int(n_params) - int(n_cam_params))


def reduced_camera_system(handler: Any, x: np.ndarray, *, loss: str = "linear",
                          f_scale: float = 1.0) -> dict[str, Any]:
    """The undamped reduced camera system ``S = B - E C^-1 E^T`` the Schur solver forms.

    Uses the solver's own assembly (``numba_schur.SchurSolver``) on the handler's parameter
    groups at ``x``; rows and columns are reordered so each camera's intrinsics and
    extrinsics sit together. Returns ``S`` (reordered), ``|g|`` per camera block and the
    camera slices. The Levenberg-Marquardt damping the solver adds is not recorded per step,
    so it is left out (lam = 0).
    """
    from pyCamSet.optimisation.numba_schur import SchurSolver, spec_from_groups
    groups = handler.parameter_groups()
    spec = spec_from_groups(groups)
    solver = SchurSolver(spec, threads=1)
    blocks_fn = handler.make_loss_blocks(1)
    loss_fn = handler.make_loss_fun(1)
    n_base = 2 * len(handler._flat_detections())

    def residual_fn(params):
        return np.asarray(loss_fn(params), dtype=np.float64)[:n_base]

    if loss != "linear":
        from pyCamSet.optimisation import robust_loss
        problem = robust_loss.RobustProblem(residual_fn, blocks_fn, loss, f_scale)
        residual_fn, blocks_fn = problem.residuals, problem.blocks
    x = np.asarray(x, dtype=np.float64).copy()
    B, E, C, v, w = solver.normal_equations(blocks_fn(x), residual_fn(x).reshape(-1, 2))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        S, g, _ = solver.reduced_system(B, E, C, v, w, 0.0)
    intr_group, extr_group = groups[0], groups[1]
    n_cam = intr_group.n_rows
    ni, ne = intr_group.n_par, extr_group.n_par
    order = []
    slices = []
    for cam in range(n_cam):
        start = len(order)
        order.extend(range(cam * ni, (cam + 1) * ni))
        order.extend(range(n_cam * ni + cam * ne, n_cam * ni + (cam + 1) * ne))
        slices.append((start, len(order)))
    order = np.asarray(order)
    S = S[np.ix_(order, order)]
    pinned = spec.keep_fixed[order]
    S = np.where(pinned[:, None] | pinned[None, :], 0.0, S)
    g = g[order]
    return {"S": S, "slices": slices, "g": g, "pinned": pinned,
            "n_eliminated": int(spec.elim_free.sum()), "n_kept": int(spec.keep_free.sum())}


def normal_matrix_blocks(structure: SchurStructure) -> dict[str, Any]:
    """Which blocks of the normal matrix JᵀJ are non-zero, parameter by parameter.

    Order: each camera's numbers (its lens, then its orientation/position), then each photo's
    6 pose numbers. Codes: -1 zero; ``0..n_cam-1`` a camera's own block (B, block-diagonal);
    ``n_cam`` a camera-photo coupling (E, non-zero when the camera saw the photo);
    ``n_cam + 1`` a photo's own 6×6 block (C, block-diagonal, so it inverts photo by photo).
    """
    per_cam = structure.n_intr + structure.n_extr
    n_cam = structure.n_cameras
    poses = [int(p) for p in structure.pose_order]
    col = {p: k for k, p in enumerate(poses)}
    n_keep = n_cam * per_cam
    n = n_keep + 6 * len(poses)
    codes = np.full((n, n), -1, dtype=np.int16)
    for c in range(n_cam):
        codes[c * per_cam:(c + 1) * per_cam, c * per_cam:(c + 1) * per_cam] = c
    for k in range(len(poses)):
        a = n_keep + 6 * k
        codes[a:a + 6, a:a + 6] = n_cam + 1
    for cam, image in zip(structure.pair_camera, structure.pair_image):
        k = col.get(int(image))
        if k is None:
            continue
        r0, r1 = int(cam) * per_cam, (int(cam) + 1) * per_cam
        c0 = n_keep + 6 * k
        codes[r0:r1, c0:c0 + 6] = n_cam
        codes[c0:c0 + 6, r0:r1] = n_cam
    return {"codes": codes, "n_kept": n_keep, "n_eliminated": 6 * len(poses),
            "n_cameras": n_cam, "per_camera": per_cam, "n_poses": len(poses)}


def coupling_matrix(system: Mapping[str, Any]) -> np.ndarray:
    """How strongly the reduced camera system ties each pair of cameras together, 0..1.

    S is first Jacobi-scaled, ``D^-1/2 S D^-1/2`` with ``D = diag(S)`` (SCI2-19): every number
    of every camera (lens and orientation alike) then counts on one unit-free footing, so the
    tie is not just the rotation numbers' (whose raw entries are ~100x the lens numbers').
    Then |S_ij| / sqrt(|S_ii| |S_jj|) over the camera blocks (Frobenius norms). The diagonal
    is 1; a pinned (zero) parameter is left out."""
    S = np.asarray(system["S"], dtype=float)
    d = np.abs(np.diag(S))
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    S = S * inv[:, None] * inv[None, :]
    slices = system["slices"]
    norms = np.array([[np.linalg.norm(S[a0:a1, b0:b1]) for b0, b1 in slices]
                      for a0, a1 in slices])
    diag = np.sqrt(np.maximum(np.diag(norms), 1e-300))
    return np.clip(norms / np.outer(diag, diag), 0.0, 1.0)


def block_updates(visual: Phase3Visual, state: int) -> dict[str, np.ndarray]:
    """How far each block moved in the step that reached ``state`` (rotation, degrees).

    ``camera_deg`` (n_cam,), ``pose_deg`` (n_images,; nan for photos without a pose), and
    ``intrinsic_rel`` (n_cam,): the relative change of each camera's intrinsic block.
    State 0 (the start) has no update: everything is zero.
    """
    s = visual.states[state]
    n_img = len(s.pose_params)
    if state <= 0:
        return {"camera_deg": np.zeros(visual.n_cameras), "pose_deg": np.zeros(n_img),
                "intrinsic_rel": np.zeros(visual.n_cameras)}
    p = visual.states[state - 1]
    cam = _angle_deg(p.rotations, s.rotations)
    pose = _angle_deg(_rotations(p.pose_params[:, :3]), _rotations(s.pose_params[:, :3]))
    used = np.zeros(n_img, dtype=bool)
    used[visual.posed_images] = True
    pose = np.where(used, pose, np.nan)
    scale = np.maximum(np.linalg.norm(p.intrinsics, axis=1), 1e-12)
    intr = np.linalg.norm(s.intrinsics - p.intrinsics, axis=1) / scale
    return {"camera_deg": cam, "pose_deg": pose, "intrinsic_rel": intr}


# --------------------------------------------------------------------------------------
# Consistency with the saved diagnostics


def consistency(visual: Phase3Visual, derivation: Any, diagnostics: Mapping | None) -> dict:
    """Check the recomputed start/final errors against the saved run (honest numbers).

    ``final_matches`` compares the final per-camera means with the derivation's checked D3.12
    values; ``start_matches`` / ``final_mean_matches`` compare the overall means with D3.5 /
    D3.6. Each is True, False or None (nothing saved to compare with).
    """
    result: dict[str, bool | None] = {"final_matches": None, "start_matches": None,
                                      "final_mean_matches": None}
    final = visual.states[-1]
    means = getattr(derivation, "per_camera_residual_mean_px", None)
    if means and len(means) == visual.n_cameras:
        saved = np.array([np.nan if v is None else float(v) for _, v in means])
        result["final_matches"] = bool(np.allclose(saved, final.camera_mean_px, rtol=1e-6,
                                                   atol=1e-9, equal_nan=True))
    diag = diagnostics or {}
    try:
        d35 = float(diag.get("D3.5_initial_euclid_px"))
        if visual.source in ("recorded steps", "start and result only") and np.isfinite(d35):
            result["start_matches"] = bool(np.isclose(visual.states[0].mean_px, d35,
                                                      rtol=1e-6, atol=1e-9))
    except (TypeError, ValueError):
        pass
    try:
        d36 = float(diag.get("D3.6_final_euclid_px"))
        if np.isfinite(d36):
            result["final_mean_matches"] = bool(np.isclose(final.mean_px, d36, rtol=1e-6,
                                                           atol=1e-9))
    except (TypeError, ValueError):
        pass
    return result


__all__ = [
    "StepState", "Phase3Visual", "build_phase3_visual", "join_events", "graph_layout",
    "point_selection", "ray_selection", "camera_pose", "scene_frame", "viewing_axis", "SchurStructure",
    "schur_structure", "reduced_camera_system", "block_updates", "consistency",
    "TELECENTRIC_DISPLAY_DISTANCE", "STATIC_POINT_LIMIT", "RAY_LIMIT",
    "TELECENTRIC_RIG_DISTANCE", "MOTION_FACTORS", "reference_pose", "target_world_points",
    "target_centre_size", "target_edges", "corner_errors", "photo_errors", "rig_motion",
    "motion_factor", "posed_outline", "photo_rows", "photo_points", "photo_choices",
    "typical_photo", "photo_crop", "camera_photo_errors", "most_improved_camera",
    "CameraAgreement", "camera_agreement", "seed_arrays",
    "LensChange", "lens_changes", "ErrorLadder", "error_ladder", "photo_source",
    "normal_matrix_blocks", "coupling_matrix", "phase4_camera_rms", "BENDING_FIRST_COLUMN",
    "LensEffect", "lens_bending_effect", "LaterLens", "later_lens", "eps_per_metre",
    "JoinCandidates", "join_candidates", "TiltDemo", "tilt_demo", "capped_turns", "full_seed",
]


def full_seed(phase3_set: Any, phase2_set: Any, *, phase3_path: Any = None,
              phase2_path: Any = None) -> tuple:
    """``education_derive._seed_from_inputs``: (target_in_camera, extrinsics, poses, errors),
    the first guess re-run read-only from the run's own inputs. With the two files' paths the
    result is shared with the derivation (``_seed_for_files``), so it is computed once."""
    from pyCamSet.workflow.education_derive import _seed_for_files, _seed_from_inputs
    if phase3_path is None or phase2_path is None:
        return _seed_from_inputs(phase3_set, phase2_set)
    return _seed_for_files(phase3_path, phase2_path, phase3_set, phase2_set)
