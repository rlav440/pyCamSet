'''Recalculate Phase 3 evidence from persisted, provenance-checked camsets.
'''
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import copy
import hashlib
import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Mapping
from zipfile import BadZipFile

import numpy as np
from scipy.spatial.transform import Rotation

from pyCamSet.utils.paths import long_path
from pyCamSet.utils.saving import load_CameraSet
from pyCamSet.utils.target_units import (
    LEGACY_UNVERIFIED,
    millimetres_per_point_data_unit,
    saved_solution_units,
    square_size_in_point_data_units,
    target_point_data_units,
)
from pyCamSet.workflow.education_trace import (
    RunContext,
    Verification,
    read_trace,
)

POSITION_LEVEL_MM = 10.0
ROTATION_LEVEL_DEG = 5.0
MIN_OUTLIER_N = 8
OUTLIER_MULTIPLIER = 5.0
OUTLIER_DECLARED_N = (8, 80)
# Phase 4 scale-consistency ratio tolerance behind metric_qualifier_required.
SCALE_RATIO_TOLERANCE = 0.01


@dataclass(frozen=True)
class DerivationUnavailable:
    """A fail-closed result with a stable learner-facing disposition."""

    reason: str


@dataclass(frozen=True)
class ResidualObservation:
    """An optimised detection and its attributed saved residual vector."""

    camera: int
    image: int
    key: tuple[int, ...]
    observed_xy: tuple[float, float]
    residual_xy: tuple[float, float]
    magnitude_px: float


@dataclass(frozen=True)
class CameraRigSpread:
    """Seed consistency for one camera relative to camera index zero."""

    camera: int
    position_scatter_units: float
    position_scatter_mm: float | None
    position_scatter_squares: float | None
    rotation_scatter_deg: float
    per_image_position_units: tuple[float | None, ...]
    per_image_rotation_deg: tuple[float | None, ...]
    worst_position_image: int | None
    worst_rotation_image: int | None
    position_photo_outliers: tuple[int, ...]
    rotation_photo_outliers: tuple[int, ...]
    usable_image_count: int
    outlier_rule_status: str


@dataclass(frozen=True)
class Phase3Derivation:
    """Verified values derived from a persisted Phase 3 run."""

    status: str
    reason: str | None
    run_id: str
    observations: tuple[ResidualObservation, ...]
    per_image_observation_count: tuple[int, ...]
    residual_mean_px: float | None
    residual_xy: tuple[tuple[float, float], ...]
    per_camera_residual_mean_px: tuple[tuple[str, float | None], ...]
    seed_mean_by_image_px: tuple[float | None, ...]
    seed_weighted_mean_px: float | None
    d35_initial_px: float | None
    seed_matches_saved: bool
    weighted_seed_matches_d35: bool | None
    d35_convention: str
    seed_camset_origin: str
    input_verification: str
    saved_point_units: str
    rebuilt_point_units: str
    fixed_pose: int | None
    fixed_pose_valid: bool | None
    rig_spreads: tuple[CameraRigSpread, ...]
    unit_disposition: str
    seed_observations: tuple[ResidualObservation, ...] | None = None
    seed_observation_status: str = "unavailable: keyed starting-state residuals not recorded"
    seed_detection_count_by_image: tuple[int, ...] = ()
    #: cameras sharing fewer than two posed photos with camera 0: no spread measured
    rig_unmeasured_cameras: tuple[int, ...] = ()


@dataclass(frozen=True)
class Phase4PointClasses:
    """Observation and ray-direction classes for the visible target points."""

    unseen: tuple[int, ...]
    one_direction: tuple[int, ...]
    several_directions: tuple[int, ...]
    ray_separation_deg: tuple[float | None, ...]


@dataclass(frozen=True)
class Phase4Frame:
    """One accepted solver state projected into shape and rig display frames."""

    nit: int
    cost: float
    shape_points: np.ndarray | None
    rig_cameras: Any | None
    target_poses: np.ndarray | None
    point_classes: Phase4PointClasses
    status: str
    reason: str | None = None
    recorded_start: bool = False
    state_source_verified: bool | None = None
    held_parameter_difference: bool = False


@dataclass(frozen=True)
class Phase4Derivation:
    """Verified Phase 4 measurements and optional recorded-step projections."""

    status: str
    reason: str | None
    run_id: str
    d41_visible_points: int | None
    d43_final_px: float | None
    d45_gauge_scale: float | None
    d47_mean_displacement_mm: float | None
    point_classes: Phase4PointClasses | None
    display_refit_points: np.ndarray | None
    display_refit_scale: float | None
    scale_consistency_ratio: float | None
    metric_qualifier_required: bool
    frames: tuple[Phase4Frame, ...]
    #: Mean distance from the design over the several-direction points only,
    #: after the display alignment (A1 section 4: the measured,
    #: well-constrained-point headline). None when there is no such point or
    #: no display alignment.
    several_directions_mean_displacement_mm: float | None = None
    #: True when the saved camset's length units are legacy/unverified (the
    #: units half of metric_qualifier_required). None when not recorded.
    saved_units_unverified: bool | None = None
    #: The scale a size-fitted (similarity) alignment would apply. Not used for display:
    #: it would shrink every face (SCI-1); reported so the learner can see what it hides.
    display_similarity_scale: float | None = None


def _unavailable(context: RunContext, reason: str) -> DerivationUnavailable:
    """Return a single consistent refusal for missing or unverifiable evidence."""
    del context
    return DerivationUnavailable(reason)


def _finite_vector(value: Any) -> np.ndarray | None:
    """Return a finite one-dimensional numeric array, otherwise no evidence."""
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None
    return result


def _observation_rows(handler: Any) -> np.ndarray:
    """Read only the rows the solver actually optimised, in stable row order."""
    rows = np.asarray(handler._flat_detections(), dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] < 5 or not np.all(np.isfinite(rows)):
        raise ValueError("optimised detection rows are malformed or non-finite")
    return rows


def attribute_residuals(camera_set: Any) -> tuple[tuple[ResidualObservation, ...], tuple[int, ...], float, tuple[tuple[float, float], ...]]:
    """Pair the saved reprojection segment with the reloaded solver rows.

    Any residual tail (for example lockbox priors) is intentionally excluded.
    A count/layout mismatch is an unavailable derivation, never guessed around.
    """
    handler = getattr(camera_set, "calibration_handler", None)
    if handler is None:
        raise ValueError("saved camset has no reloadable calibration handler")
    rows = _observation_rows(handler)
    values = _finite_vector(getattr(camera_set, "calibration_result", None))
    if values is None or len(values) < 2 * len(rows):
        raise ValueError("saved residual result is shorter than the optimised rows")
    segment = values[:2 * len(rows)].reshape(-1, 2)
    if not np.all(np.isfinite(segment)):
        raise ValueError("saved reprojection segment contains non-finite values")
    norms = np.linalg.norm(segment, axis=1)
    row_count = len(rows)
    image_count = int(getattr(handler.detection, "max_ims", 0))
    if image_count < 0 or np.any(rows[:, 1] < 0) or np.any(rows[:, 1] >= image_count):
        raise ValueError("optimised rows contain an invalid image index")
    counts = np.bincount(rows[:, 1].astype(int), minlength=image_count)
    attributed = tuple(
        ResidualObservation(
            camera=int(row[0]), image=int(row[1]),
            key=tuple(int(value) for value in row[2:-2]),
            observed_xy=(float(row[-2]), float(row[-1])),
            residual_xy=(float(residual[0]), float(residual[1])),
            magnitude_px=float(norm),
        )
        for row, residual, norm in zip(rows, segment, norms)
    )
    # The persisted D3.11 is residual-vector data; return its exact segment.
    residual_xy = tuple((float(x), float(y)) for x, y in segment)
    return attributed, tuple(int(value) for value in counts), float(norms.mean()), residual_xy


def _keyed_seed_residuals(context: RunContext, handler: Any,
                          final: tuple[ResidualObservation, ...],
                          d35: float) -> tuple[tuple[ResidualObservation, ...] | None, str]:
    """Replay the saved start on the exact final solver rows, or refuse."""
    raw = (context.metadata.get("artifacts") or {}).get("initial_state")
    if not isinstance(raw, str) or not raw:
        return None, "unavailable: the run has no saved Phase 3 starting vector"
    path = Path(raw)
    if not path.is_absolute():
        path = context.run_dir / path
    try:
        path = path.resolve(strict=True)
        if not path.is_relative_to(context.run_dir.resolve(strict=True)):
            return None, "unavailable: starting vector is outside this run"
    except OSError:
        return None, "unavailable: the saved Phase 3 starting vector is missing"
    expected_digest = (context.metadata.get("artifacts") or {}).get("initial_state_sha256")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        return None, "unavailable: saved Phase 3 starting vector has no file digest"
    try:
        if path.stat().st_size > 8 * 1024 * 1024:
            return None, "unavailable: saved Phase 3 starting vector exceeds its bound"
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest:
            return None, "unavailable: saved Phase 3 starting vector digest mismatch"
    except OSError:
        return None, "unavailable: saved Phase 3 starting vector could not be checked"
    try:
        with np.load(path, allow_pickle=False) as snapshot:
            x = np.asarray(snapshot["calibration_params"], dtype=np.float64).reshape(-1)
            missing = np.asarray(snapshot["missing_poses"], dtype=bool).reshape(-1)
            stored_mean = float(snapshot["initial_error_px"])
    except (OSError, BadZipFile, KeyError, ValueError, TypeError):
        return None, "unavailable: saved Phase 3 starting vector is corrupt"
    signature = [(item.camera, item.image, item.key) for item in final]
    if len(signature) != len(set(signature)):
        return None, "unavailable: duplicate optimised observation keys"
    rows = _observation_rows(handler)
    row_signature = [(int(row[0]), int(row[1]), tuple(int(v) for v in row[2:-2]))
                     for row in rows]
    if signature != row_signature:
        return None, "unavailable: saved final row keys differ from starting-state rows"
    reference_missing = np.asarray(getattr(handler, "missing_poses", []), dtype=bool).reshape(-1)
    if (not len(final) or not len(x) or not np.all(np.isfinite(x))
            or missing.shape != reference_missing.shape
            or not np.array_equal(missing, reference_missing)
            or not math.isfinite(stored_mean) or not math.isfinite(d35)
            or not math.isclose(stored_mean, d35, rel_tol=1e-12, abs_tol=1e-9)):
        return None, "unavailable: starting vector, pose mask or D3.5 does not reconcile"
    # Reconcile the private reloaded handler's outlier pose slots with the
    # vector actually supplied to the solver, without modifying the run.
    try:
        from pyCamSet.workflow.education_playback import prepare_saved_phase3_handler
        prepare_saved_phase3_handler(handler, x)
        values = np.asarray(handler.make_loss_fun(1)(x), dtype=np.float64).reshape(-1)
    except (ValueError, TypeError, AttributeError, IndexError, RuntimeError):
        return None, "unavailable: saved starting vector cannot be replayed on final rows"
    if values.size < 2 * len(final) or not np.all(np.isfinite(values[:2 * len(final)])):
        return None, "unavailable: starting residual segment is missing or non-finite"
    segment = values[:2 * len(final)].reshape(-1, 2)
    magnitudes = np.linalg.norm(segment, axis=1)
    if not math.isclose(float(magnitudes.mean()), d35, rel_tol=1e-9, abs_tol=1e-9):
        return None, "unavailable: starting residuals do not reproduce D3.5"
    attributed = tuple(
        ResidualObservation(camera=old.camera, image=old.image, key=old.key,
                            observed_xy=old.observed_xy,
                            residual_xy=(float(residual[0]), float(residual[1])),
                            magnitude_px=float(magnitude))
        for old, residual, magnitude in zip(final, segment, magnitudes)
    )
    return attributed, "verified on the same ordered optimised rows"


def _seed_from_inputs(
    phase3_set: Any, phase2_set: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Re-run the real, read-only Phase 3 initialiser from the recorded input.

    The Phase 2 cameras are reloaded independently. Only their intrinsics and
    camera models seed the estimate; no optimiser is run and no saved object is
    passed to a mutating solve path.
    """
    from pyCamSet.optimisation.template_handler import estimate_camera_relative_poses

    phase3_handler = getattr(phase3_set, "calibration_handler", None)
    if phase3_handler is None:
        raise ValueError("Phase 3 camset cannot rebuild its target/handler")
    target = phase3_handler.target
    detection = phase3_handler.detection
    input_cams = phase2_set
    if list(input_cams.get_names()) != list(detection.cam_names):
        raise ValueError("used Phase 2 camset camera order does not match embedded detections")
    target_in_camera = np.asarray([
        [target.target_pose_in_cam_image(image, cam, mode="nan")
         for image in detection.get_image_list()]
        for cam in input_cams
    ], dtype=np.float64)
    # The run's own single-face setting, from its saved options, or the rebuilt D3.3 differs.
    options = getattr(phase3_handler, "problem_opts", {}) or {}
    seed_result = estimate_camera_relative_poses(
        target, detection, input_cams,
        recover_single_face_telecentric=bool(
            # A run saved without the key predates the option. Single-face recovery did not
            # exist before eaf4ae8 (2026-10-08), so False is right for those runs. The solver's
            # current default (True) would place images such runs never placed. Runs from
            # eaf4ae8 to c63b968 placed them unrecorded; they cannot be told apart.
            options.get("recover_single_face_telecentric", False)
            if isinstance(options, Mapping) else False))
    return target_in_camera, *seed_result


def _fixed_pose_valid(handler: Any, rows: np.ndarray) -> tuple[int | None, bool]:
    """Check fixed-pose evidence by rank of its observed design points."""
    options = getattr(handler, "problem_opts", {})
    fixed_pose = options.get("fixed_pose", 0) if isinstance(options, Mapping) else 0
    if isinstance(fixed_pose, bool) or not isinstance(fixed_pose, (int, np.integer)):
        return None, False
    fixed_pose = int(fixed_pose)
    missing = getattr(handler, "missing_poses", None)
    if missing is not None:
        missing_array = np.asarray(missing, dtype=bool).reshape(-1)
        if fixed_pose < 0 or fixed_pose >= len(missing_array) or missing_array[fixed_pose]:
            return fixed_pose, False
    target_points = np.asarray(handler.target.point_data, dtype=np.float64).reshape(-1, 3)
    image_rows = rows[rows[:, 1].astype(int) == fixed_pose]
    if len(image_rows) < 3:
        return fixed_pose, False
    raw_keys = image_rows[:, 2:-2]
    if not np.all(np.isfinite(raw_keys)) or not np.array_equal(raw_keys, np.floor(raw_keys)):
        return fixed_pose, False
    indices = raw_keys.astype(int)
    keydims = np.asarray(handler.target.point_data).shape[:-1]
    try:
        if indices.shape[1] == 1:
            # TemplateHandler._flat_detections produces one linear point key.
            point_indices = indices[:, 0]
            if np.any((point_indices < 0) | (point_indices >= len(target_points))):
                return fixed_pose, False
        elif indices.shape[1] == len(keydims):
            # Also accept unflattened, multi-axis detection rows.
            point_indices = np.ravel_multi_index(indices.T, keydims)
        else:
            return fixed_pose, False
    except (ValueError, TypeError):
        return fixed_pose, False
    design = target_points[np.unique(point_indices)]
    if len(design) < 3:
        return fixed_pose, False
    singular = np.linalg.svd(design - design.mean(axis=0), compute_uv=False)
    return fixed_pose, bool(len(singular) >= 2 and singular[0] > 0 and singular[1] > 1e-3 * singular[0])


def _relative_transforms(target_in_camera: np.ndarray) -> np.ndarray:
    """Form camera-i to camera-0 transforms exactly as the report does."""
    poses = np.asarray(target_in_camera, dtype=np.float64)
    if poses.ndim != 4 or poses.shape[0] < 2 or poses.shape[-2:] != (4, 4):
        raise ValueError("seed pose stack has an unexpected shape")
    ref = poses[0]
    result = np.full_like(poses, np.nan)
    for camera_index in range(poses.shape[0]):
        for image_index in range(poses.shape[1]):
            if not (np.all(np.isfinite(ref[image_index]))
                    and np.all(np.isfinite(poses[camera_index, image_index]))):
                continue
            result[camera_index, image_index] = poses[camera_index, image_index] @ np.linalg.inv(ref[image_index])
    return result


def _photo_outlier_indices(deviations: np.ndarray, level: float, *, minimum_n: int,
                           multiplier: float, declared_range: tuple[int, int]) -> tuple[tuple[int, ...], str]:
    """Apply the candidate per-photo rule only in its declared N range."""
    finite = deviations[np.isfinite(deviations)]
    n = len(finite)
    low, high = declared_range
    if n < minimum_n:
        return (), f"not applicable: fewer than {minimum_n} finite image deviations"
    if not low <= n <= high:
        return (), f"not applicable: N={n} usable images outside declared range {low}–{high}"
    median = float(np.median(finite))
    candidates = np.flatnonzero(np.isfinite(deviations) & (deviations > level)
                                & (deviations > multiplier * median))
    return tuple(int(index) for index in candidates), "applied"


def derive_rig_spreads(target_in_camera: np.ndarray, target: Any, *,
                       minimum_n: int = MIN_OUTLIER_N,
                       multiplier: float = OUTLIER_MULTIPLIER,
                       declared_range: tuple[int, int] = OUTLIER_DECLARED_N,
                       metric_units_verified: bool = True,
                       unmeasured: list[int] | None = None,
                       ) -> tuple[CameraRigSpread, ...]:
    """Compute camera-0-relative camera-centre and chordal-mean rotation scatters.

    Position is the RMS distance of camera i's centre (-R^T t, expressed in
    camera 0's frame) from its vector mean; rotation is RMS geodesic
    distance in degrees from SciPy's chordal L2 rotation mean. Unverified
    saved units cannot produce an mm value or a 10-mm position outlier flag.
    """
    from pyCamSet.utils.target_units import square_size_in_point_data_units

    relative = _relative_transforms(target_in_camera)
    mm_scale = millimetres_per_point_data_unit(target) if metric_units_verified else None
    square_size = square_size_in_point_data_units(target)
    result: list[CameraRigSpread] = []
    for camera_index in range(1, relative.shape[0]):
        transforms = relative[camera_index]
        valid = np.all(np.isfinite(transforms), axis=(1, 2))
        indices = np.flatnonzero(valid)
        rotations = transforms[valid, :3, :3]
        # T_{i<-0} = [R | t] holds camera 0's origin in camera i's frame.
        # Camera i's centre in camera 0's frame is C = -R^T t (Hartley &
        # Zisserman, MVG 2nd ed., sec. 6.1); scattering t itself would turn
        # rotation jitter into a fake position spread of about baseline x angle.
        centres = -np.einsum("nji,nj->ni", rotations, transforms[valid, :3, 3])
        per_pos: list[float | None] = [None] * relative.shape[1]
        per_rot: list[float | None] = [None] * relative.shape[1]
        if len(indices) < 2:
            # One sparse camera must not cost the whole Phase 3 explanation: with a list to
            # report into, it is left out and named. (On a real PuzzleBoard cube set, one sparse
            # PuzzleBoard camera lost the first guess, residuals and agreement views.)
            if unmeasured is not None:
                unmeasured.append(int(camera_index))
                continue
            raise ValueError(f"camera {camera_index} has fewer than two common seed poses")
        centre = centres.mean(axis=0)
        deviations = np.linalg.norm(centres - centre, axis=1)
        pos_scatter = float(np.sqrt(np.mean(deviations ** 2)))
        mean_rotation = Rotation.from_matrix(rotations).mean()
        angles = np.degrees((mean_rotation.inv() * Rotation.from_matrix(rotations)).magnitude())
        rot_scatter = float(np.sqrt(np.mean(angles ** 2)))
        for image, deviation, angle in zip(indices, deviations, angles):
            per_pos[int(image)] = float(deviation)
            per_rot[int(image)] = float(angle)
        position_level_units = POSITION_LEVEL_MM / mm_scale if mm_scale else math.nan
        position_bad, rule_status = _photo_outlier_indices(
            np.asarray(per_pos, dtype=float), position_level_units,
            minimum_n=minimum_n, multiplier=multiplier,
            declared_range=declared_range)
        rotation_bad, rotation_status = _photo_outlier_indices(
            np.asarray(per_rot, dtype=float), ROTATION_LEVEL_DEG,
            minimum_n=minimum_n, multiplier=multiplier,
            declared_range=declared_range)
        if rotation_status != "applied":
            rule_status = rotation_status
        elif mm_scale is None:
            rule_status = "rotation only: saved position units unverified"
        mm_value = pos_scatter * mm_scale if mm_scale is not None else None
        squares = pos_scatter / square_size if math.isfinite(square_size) and square_size > 0 else None
        result.append(CameraRigSpread(
            camera=camera_index,
            position_scatter_units=pos_scatter,
            position_scatter_mm=mm_value,
            position_scatter_squares=squares,
            rotation_scatter_deg=rot_scatter,
            per_image_position_units=tuple(per_pos),
            per_image_rotation_deg=tuple(per_rot),
            worst_position_image=(int(indices[int(np.argmax(deviations))]) if len(indices) else None),
            worst_rotation_image=(int(indices[int(np.argmax(angles))]) if len(indices) else None),
            position_photo_outliers=position_bad,
            rotation_photo_outliers=rotation_bad,
            usable_image_count=len(indices),
            outlier_rule_status=rule_status,
        ))
    return tuple(result)


# Small caches for the education derivations. A CameraSet JSON parse and a seeded
# Phase 4 handler are the two expensive, repeated setup steps when a run is opened
# (the derivation, the replay and the visual each needed one). Entries are keyed by the
# file's identity (resolved path, mtime, size) and the handler also by the saved
# diagnostics, so a rewritten file or a changed record is never served from the cache.
# Every hit hands out a deep copy: the handlers mutate their camera objects while they
# replay frames, so a shared object would let one consumer change another's numbers.
_CACHE_SLOTS = 4
_CACHE_LOCK = threading.RLock()
_CAMSET_CACHE: "OrderedDict[tuple, Any]" = OrderedDict()
_PHASE4_HANDLER_CACHE: "OrderedDict[tuple, Any]" = OrderedDict()
_SEED_CACHE: "OrderedDict[tuple, Any]" = OrderedDict()


def _file_identity(path: Path | str) -> tuple[str, int, int] | None:
    """(resolved path, mtime_ns, size) of *path*, or None when it cannot be read."""
    try:
        target = long_path(path)
        stat = os.stat(target)
    except (OSError, TypeError, ValueError):
        return None
    return (str(target), int(stat.st_mtime_ns), int(stat.st_size))


def _private_copy(value: Any) -> Any:
    """A deep copy to keep in a cache, or None when the object cannot be copied (for example a
    rebuilt handler holding an OpenCV ArUco dictionary). Uncopyable objects are not cached."""
    try:
        return copy.deepcopy(value)
    except Exception:  # noqa: BLE001 - any copy failure means "do not cache"
        return None


def _cache_put(cache: "OrderedDict[tuple, Any]", key: tuple, value: Any) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _CACHE_SLOTS:
        cache.popitem(last=False)


def _load_camset(path: Path) -> Any:
    """Load a camset using the project's long-path-aware loader.

    Repeated loads of an unchanged file return a deep copy of one parsed object.
    """
    identity = _file_identity(path)
    if identity is None:
        return load_CameraSet(long_path(path))
    with _CACHE_LOCK:
        cached = _CAMSET_CACHE.get(identity)
        if cached is not None:
            _CAMSET_CACHE.move_to_end(identity)
            return copy.deepcopy(cached)
    loaded = load_CameraSet(long_path(path))
    kept = _private_copy(loaded)
    if kept is not None:
        with _CACHE_LOCK:
            _cache_put(_CAMSET_CACHE, identity, kept)
    return loaded


def _seed_for_files(phase3_path: Any, phase2_path: Any, phase3_set: Any, phase2_set: Any) -> tuple:
    """``_seed_from_inputs`` for two saved files, computed once per file pair.

    The first guess re-estimates the relative poses of the whole rig; the derivation and the
    Phase 3 first-guess stage both need it for the same two files. A hit returns a deep copy.
    """
    first, second = _file_identity(phase3_path), _file_identity(phase2_path)
    if first is None or second is None:
        return _seed_from_inputs(phase3_set, phase2_set)
    key = (first, second)
    with _CACHE_LOCK:
        cached = _SEED_CACHE.get(key)
        if cached is not None:
            _SEED_CACHE.move_to_end(key)
            return copy.deepcopy(cached)
    result = _seed_from_inputs(phase3_set, phase2_set)
    kept = _private_copy(result)
    if kept is not None:
        with _CACHE_LOCK:
            _cache_put(_SEED_CACHE, key, kept)
    return result


def derive_phase3(context: RunContext) -> Phase3Derivation | DerivationUnavailable:
    """Recalculate and provenance-check one saved Phase 3 result.

    The function never writes to the workspace. A missing file, failed input
    edge, unknown unit contract or scalar mismatch returns a refusal instead
    of partial evidence that could be mistaken for a verified derivation.
    """
    if context.phase != "phase3":
        return _unavailable(context, "unavailable: the selected record is not Phase 3")
    if context.quality_check != "checked" or context.status not in {"complete", "incomplete"}:
        return _unavailable(context, "unavailable: the run has no checkable Phase 3 result")
    output_path = context.artifact("phase3")
    used_path_value = (context.metadata.get("artifacts") or {}).get("phase2_initial_camset_used")
    used_path = Path(str(used_path_value)) if used_path_value else None
    if output_path is None or used_path is None or not used_path.is_file():
        return _unavailable(context, "unavailable: nothing saved to check it against")
    edge = context.verify_input("phase2")
    if edge.verification is not Verification.VERIFIED:
        return _unavailable(context, "cannot be recalculated for this run: Phase 2 input is not verified")
    try:
        phase3_set = _load_camset(output_path)
        phase2_set = _load_camset(used_path)
        handler = getattr(phase3_set, "calibration_handler", None)
        if handler is None:
            raise ValueError("saved Phase 3 camset has no reloadable handler")
        observations, counts, residual_mean, residual_xy = attribute_residuals(phase3_set)
        diagnostics = context.metadata.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise ValueError("saved Phase 3 diagnostics are unavailable")
        stored_xy = np.asarray(diagnostics.get("D3.11_residual_xy_scatter"), dtype=float)
        if stored_xy.shape != (len(residual_xy), 2) or not np.array_equal(stored_xy, np.asarray(residual_xy)):
            raise ValueError("saved D3.11 does not match the attributed residual segment")
        stored_mean = diagnostics.get("D3.6_final_euclid_px")
        if not math.isfinite(float(stored_mean)) or abs(residual_mean - float(stored_mean)) > 1e-12 * abs(float(stored_mean)):
            raise ValueError("recalculated final residual mean does not match D3.6")
        # Reproduce D3.12 over the same ordered, optimised residual rows.
        camera_names = list(getattr(handler, "cam_names", ()))
        camera_means: list[tuple[str, float | None]] = []
        stored_per_camera = diagnostics.get("D3.12_per_camera_mean_reprojection")
        if not isinstance(stored_per_camera, Mapping):
            raise ValueError("saved D3.12 per-camera means are unavailable")
        row_cameras = _observation_rows(handler)[:, 0].astype(int)
        norms = np.linalg.norm(np.asarray(residual_xy, dtype=np.float64), axis=1)
        if len(row_cameras) != len(norms):
            raise ValueError("D3.12 camera rows do not align with D3.11 residuals")
        for camera_index, camera_name in enumerate(camera_names):
            selected = norms[row_cameras == camera_index]
            mean = float(selected.mean()) if len(selected) else None
            camera_means.append((str(camera_name), mean))
            try:
                saved_camera_mean = float(stored_per_camera[camera_name])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"saved D3.12 mean is unavailable for {camera_name}")
            if mean is None or not math.isfinite(saved_camera_mean) or abs(mean - saved_camera_mean) > 1e-12 * abs(saved_camera_mean):
                raise ValueError(f"recalculated per-camera mean does not match D3.12 for {camera_name}")
        # Recompute seed only from the actual Phase 2 artifact and saved detections.
        target_in_camera, _, _, seed_errors = _seed_for_files(
            output_path, used_path, phase3_set, phase2_set)
        seed_values = np.asarray(seed_errors, dtype=np.float64).reshape(-1)
        stored_seed = np.asarray(diagnostics.get("D3.3_per_image_initial_reprojection"), dtype=float)
        if seed_values.shape != stored_seed.shape:
            raise ValueError("recalculated seed image layout does not match D3.3")
        if not np.array_equal(np.isnan(seed_values), np.isnan(stored_seed)):
            raise ValueError("recalculated seed has a different NaN pattern from D3.3")
        if np.any(np.isinf(seed_values)) or np.any(np.isinf(stored_seed)):
            raise ValueError("D3.3 seed values contain infinity")
        finite = np.isfinite(seed_values)
        if np.any(finite) and not np.allclose(seed_values[finite], stored_seed[finite], rtol=1e-9, atol=0.0):
            raise ValueError("recalculated seed does not match D3.3 within tolerance")
        # D3.3 counts come from the exact flattened detections scored by
        # estimate_initial_rig, before the solver's outlier rejection.
        flatten = getattr(handler.detection, "return_flattened_keys", None)
        if callable(flatten):
            seed_detection = flatten(handler.target.point_data.shape[:-1]).get_data()
        else:
            # Older detection adapters expose their already flattened rows
            # directly; retain that read-only contract while still validating.
            get_data = getattr(handler.detection, "get_data", None)
            if not callable(get_data):
                raise ValueError("D3.3 source detections cannot be enumerated")
            seed_detection = get_data()
        seed_detection = np.asarray(seed_detection, dtype=np.float64)
        image_count = len(seed_values)
        if (seed_detection.ndim != 2 or seed_detection.shape[1] < 5
                or not np.all(np.isfinite(seed_detection))
                or np.any(seed_detection[:, 1] != np.floor(seed_detection[:, 1]))
                or np.any(seed_detection[:, 1] < 0)
                or np.any(seed_detection[:, 1] >= image_count)):
            raise ValueError("D3.3 source detection rows are malformed or outside the seed image layout")
        per_image_counts = np.bincount(
            seed_detection[:, 1].astype(int), minlength=image_count)
        # Per-image seed means are NaN exactly where the initialiser had no
        # solved pose; do not count detections for those excluded images.
        per_image_counts[~finite] = 0
        if np.any(finite & (per_image_counts == 0)):
            raise ValueError("finite D3.3 seed image has no source detections")
        if per_image_counts.shape != seed_values.shape:
            raise ValueError("D3.3 detection counts do not align with seed image means")
        weighted = (float(np.average(seed_values[finite], weights=per_image_counts[finite]))
                    if np.any(finite) and np.sum(per_image_counts[finite]) else None)
        stored_initial = diagnostics.get("D3.5_initial_euclid_px")
        d35 = float(stored_initial) if stored_initial is not None else None
        if weighted is None or d35 is None or not math.isfinite(d35):
            raise ValueError("D3.5 or finite seed observations are unavailable")
        d35_matches = (abs(weighted - d35) <= max(1e-9 * abs(d35), 1e-9)
                       if d35 >= 1e-6 else True)
        d35_convention = (
            "convention irrelevant (no initial error)" if d35 < 1e-6
            else "weighted mean" if d35_matches
            else "unknown: weighted mean does not reconcile with D3.5"
        )
        handler = phase3_set.calibration_handler
        fixed_pose, fixed_valid = _fixed_pose_valid(handler, _observation_rows(handler))
        target = handler.target
        saved_units = saved_solution_units(getattr(phase3_set, "target_config", None))
        # load_CameraSet exposes the authoritative marker separately; legacy is
        # deliberately not inferred from rebuilt target coordinates.
        saved_units = getattr(phase3_set, "_saved_point_data_units", saved_units)
        rebuilt_units = target_point_data_units(target)
        unit_source = getattr(phase3_set, "_saved_point_data_units", LEGACY_UNVERIFIED)
        if saved_units not in {"m", "mm"} or rebuilt_units not in {"m", "mm"}:
            unit_disposition = "position in squares only; saved/rebuilt unit unverified"
        elif saved_units != rebuilt_units:
            unit_disposition = "cannot be recalculated for this run: saved and rebuilt point units differ"
            raise ValueError(unit_disposition)
        else:
            unit_disposition = f"verified in {saved_units}; square spacing from rebuilt target"
        # Unit marker may be absent in old JSON; use saved config from serialized
        # input only when it has explicit marker, never from target class alone.
        if unit_source == LEGACY_UNVERIFIED:
            unit_disposition = "position in squares only; legacy saved unit unverified"
        rig_unmeasured: list[int] = []
        spreads = derive_rig_spreads(
            target_in_camera, target, unmeasured=rig_unmeasured,
            metric_units_verified=(
                unit_source != LEGACY_UNVERIFIED
                and saved_units in {"m", "mm"} and saved_units == rebuilt_units
            ),
        )
        seed_attributed, seed_status = _keyed_seed_residuals(
            context, handler, observations, d35)
        origin = "Phase 2 run's own artifact" if edge.reason is None else edge.reason
        return Phase3Derivation(
            status="available", reason=None, run_id=context.run_id,
            observations=observations, per_image_observation_count=counts,
            residual_mean_px=residual_mean, residual_xy=residual_xy,
            per_camera_residual_mean_px=tuple(camera_means),
            seed_mean_by_image_px=tuple(None if not math.isfinite(value) else float(value)
                                        for value in seed_values),
            seed_detection_count_by_image=tuple(int(value) for value in per_image_counts),
            seed_weighted_mean_px=weighted, d35_initial_px=d35,
            seed_matches_saved=True, seed_camset_origin=origin,
            weighted_seed_matches_d35=d35_matches, d35_convention=d35_convention,
            input_verification=edge.verification.value,
            saved_point_units=str(saved_units), rebuilt_point_units=str(rebuilt_units),
            fixed_pose=fixed_pose, fixed_pose_valid=fixed_valid,
            rig_spreads=spreads, unit_disposition=unit_disposition,
            rig_unmeasured_cameras=tuple(rig_unmeasured),
            seed_observations=seed_attributed,
            seed_observation_status=seed_status,
        )
    except (OSError, ValueError, TypeError, AttributeError, IndexError, np.linalg.LinAlgError) as exc:
        return _unavailable(context, f"cannot be recalculated for this run: {exc}")


def phase4_scale_consistency_ratio(
    nominal_points: np.ndarray,
    solved_points: np.ndarray,
    measured_points: np.ndarray,
    square_size: float,
    *,
    pairs: np.ndarray | None = None,
) -> float:
    """Compare nominal/solved one-square edges whose endpoints are measured."""
    nominal = np.asarray(nominal_points, dtype=np.float64).reshape(-1, 3)
    solved = np.asarray(solved_points, dtype=np.float64).reshape(-1, 3)
    measured = np.asarray(measured_points, dtype=bool).reshape(-1)
    if nominal.shape != solved.shape or measured.shape != (len(nominal),):
        raise ValueError("point arrays and measured mask do not have matching layouts")
    if not math.isfinite(square_size) or square_size <= 0:
        raise ValueError("square size must be positive and finite")
    if pairs is None:
        indices = np.triu_indices(len(nominal), k=1)
        pair_indices = np.column_stack(indices)
    else:
        pair_indices = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if np.any(pair_indices < 0) or np.any(pair_indices >= len(nominal)):
        raise ValueError("one-square edge pair index is outside the point layout")
    pair_indices = pair_indices[
        measured[pair_indices[:, 0]] & measured[pair_indices[:, 1]]]
    if pairs is None:
        distances = np.linalg.norm(
            nominal[pair_indices[:, 0]] - nominal[pair_indices[:, 1]], axis=1)
        pair_indices = pair_indices[np.isclose(distances, square_size)]
    if not len(pair_indices):
        raise ValueError("no measured one-square edges are available")
    nominal_lengths = np.linalg.norm(
        nominal[pair_indices[:, 0]] - nominal[pair_indices[:, 1]], axis=1)
    solved_lengths = np.linalg.norm(
        solved[pair_indices[:, 0]] - solved[pair_indices[:, 1]], axis=1)
    if np.any(solved_lengths <= 0) or not np.all(np.isfinite(solved_lengths)):
        raise ValueError("solved one-square edge lengths are invalid")
    return float(np.mean(nominal_lengths / solved_lengths))


def _phase4_indices(diagnostics: Mapping[str, Any]) -> tuple[int, ...]:
    """Read the immutable gauge-index record from D4.2."""
    value = diagnostics.get("D4.2_gauge_fixed_points")
    indices = value.get("indices") if isinstance(value, Mapping) else None
    if not isinstance(indices, (list, tuple)) or len(indices) != 3:
        raise ValueError("saved D4.2 gauge indices are unavailable")
    if any(isinstance(index, bool) or not isinstance(index, (int, np.integer))
           for index in indices):
        raise ValueError("saved D4.2 gauge indices are malformed")
    return tuple(int(index) for index in indices)


def _phase4_rebuild(camset: Any, diagnostics: Mapping[str, Any]) -> Any:
    """Rebuild a fresh E1-compatible handler with recorded gauge and scale rules."""
    from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler

    saved_handler = getattr(camset, "calibration_handler", None)
    if saved_handler is None:
        raise ValueError("saved Phase 4 camset has no rebuildable handler")
    options = dict(getattr(saved_handler, "problem_opts", {}) or {})
    options["recorded_gauge_indices"] = list(_phase4_indices(diagnostics))
    # E1's saved scale-rule marker is authoritative: its absence means a
    # pre-repair solve, so select the retained legacy normaliser locally.
    if "target_scale_rule" not in options:
        options["target_scale_rule"] = "legacy_mean_all_edges"
    saved_units = getattr(camset, "_saved_point_data_units", None)
    rebuilt_units = target_point_data_units(saved_handler.target)
    if saved_units in {"m", "mm"} and rebuilt_units in {"m", "mm"} and saved_units != rebuilt_units:
        raise ValueError("saved and rebuilt point units differ")
    def build(candidate_options: dict[str, Any]) -> Any:
        return SelfBundleHandler(
            camset=camset,
            target=saved_handler.target,
            detection=saved_handler.detection,
            fixed_params=getattr(saved_handler, "fixed_params", None),
            options=candidate_options,
            missing_poses=getattr(saved_handler, "missing_poses", None),
        )

    try:
        handler = build(options)
    except ValueError as exc:
        # A legacy gauge index may no longer be visible under the rebuilt
        # detection layout. Build a comparison-only handler to measure the
        # current layout; derive_phase4 will fail closed on its index check.
        if "Recorded gauge indices" not in str(exc):
            raise
        comparison_options = dict(options)
        comparison_options.pop("recorded_gauge_indices", None)
        handler = build(comparison_options)
    # Seeding applies the solve's missing-pose layout; the loss is only the solve's own
    # once it has run. (With 17 unposed images, a recorded step replayed through
    # a handler nobody had seeded gave nan or ~5e8 instead of the stored cost, and the
    # whole Phase 4 playback was shown as 'no recorded solver steps'.)
    handler.get_initial_params()
    return handler


def _phase4_handler(path: Path | str | None, camset: Any, diagnostics: Mapping[str, Any]) -> Any:
    """A freshly seeded Phase 4 handler, built once per saved run and copied thereafter.

    Seeding a handler (``_phase4_rebuild``) re-estimates the whole rig and dominates
    opening a Phase 4 run. The first call for a file builds and seeds the handler; the
    cache keeps a private deep copy and every caller, including the first, gets its own
    copy, so no consumer sees another's frame state. Without a path (or an unreadable
    file) this rebuilds every time, as before.
    """
    identity = _file_identity(path) if path is not None else None
    if identity is None:
        return _phase4_rebuild(camset, diagnostics)
    digest = hashlib.sha1(json.dumps(diagnostics, sort_keys=True, default=str)
                          .encode("utf-8")).hexdigest()
    key = (identity, digest)
    with _CACHE_LOCK:
        cached = _PHASE4_HANDLER_CACHE.get(key)
        if cached is not None:
            _PHASE4_HANDLER_CACHE.move_to_end(key)
            return copy.deepcopy(cached)
    built = _phase4_rebuild(camset, diagnostics)
    kept = _private_copy(built)
    if kept is not None:
        with _CACHE_LOCK:
            _cache_put(_PHASE4_HANDLER_CACHE, key, kept)
    return built


def _phase4_counts(handler: Any) -> np.ndarray:
    """Count only solver-used detections, excluding missing poses."""
    rows = _observation_rows(handler)
    image = rows[:, 1].astype(int)
    missing_value = getattr(handler, "missing_poses", None)
    missing = np.asarray(missing_value if missing_value is not None else [], dtype=bool)
    if missing.size:
        if np.any(image >= len(missing)):
            raise ValueError("detection row has an image outside missing_poses")
        rows = rows[~missing[image]]
    keys = rows[:, 2:-2]
    if keys.shape[1] != 1:
        raise ValueError("Phase 4 education derivation needs flattened point keys")
    n_points = int(np.prod(handler.target.point_data.shape[:-1]))
    return np.bincount(keys[:, 0].astype(int), minlength=n_points)


def _phase4_start_state(context: Any) -> tuple[bool | None, bool]:
    """Describe whether Phase 3's recorded state source confirms the start."""
    resolver = getattr(context, "phase3_state_source", None)
    if not callable(resolver):
        return None, False
    try:
        state = resolver()
    except Exception:
        return False, False
    verified = bool(getattr(state, "confirmed", False)
                    and getattr(getattr(state, "verification", None), "value", None)
                    == Verification.VERIFIED.value)
    differences = getattr(state, "effective_fixed_parameter_differences", ())
    return verified, bool(differences)


def _phase4_display_refit(
    nominal: np.ndarray, solved: np.ndarray, measured: np.ndarray, *, with_scale: bool = False,
) -> tuple[np.ndarray, float]:
    """Line the solved shape up with the design on the ``measured`` points, for display.

    Default: a least-squares *rigid* fit (turn + shift). The solved shape already has the
    scale Phase 4 hands on (the entered square size, D4.5 edge rule), so its size is kept;
    a free scale would shrink every face and draw corrections the run never estimated
    (round-1 science review SCI-1). ``with_scale=True`` gives the similarity fit (returned
    scale != 1), kept only to report what a size fit *would* remove. Returns
    ``(points, scale)``; ``scale`` is 1.0 for the rigid fit.
    """
    source = np.asarray(solved, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(nominal, dtype=np.float64).reshape(-1, 3)
    mask = np.asarray(measured, dtype=bool).reshape(-1)
    if source.shape != target.shape or mask.shape != (len(source),) or mask.sum() < 3:
        raise ValueError("display refit needs at least three measured points")
    src = source[mask]
    dst = target[mask]
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    covariance = dst_c.T @ src_c / len(src)
    left, singular, right = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right)
    rotation = left @ correction @ right
    variance = float(np.sum(src_c * src_c) / len(src))
    if variance <= 0 or not math.isfinite(variance):
        raise ValueError("display refit source geometry has no spread")
    scale = float(np.sum(singular * np.diag(correction)) / variance) if with_scale else 1.0
    transform_points = scale * (source @ rotation.T) + (dst_mean - scale * src_mean @ rotation.T)
    return transform_points, scale


def _phase4_class_mean_displacement_mm(
    points: np.ndarray | None, nominal: np.ndarray, indices: tuple[int, ...],
    mm_per_unit: float | None,
) -> float | None:
    """Mean point-to-design distance over *indices* only, in millimetres.

    Points seen once or from nearly one direction are left out by the caller:
    their depth along the line of sight is poorly determined, so their motion
    is largely noise rather than shape.
    """
    if points is None or mm_per_unit is None or not indices:
        return None
    chosen = np.asarray(indices, dtype=int)
    source = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    design = np.asarray(nominal, dtype=np.float64).reshape(-1, 3)
    if source.shape != design.shape or chosen.min() < 0 or chosen.max() >= len(design):
        return None
    value = float(np.mean(np.linalg.norm(source[chosen] - design[chosen], axis=1)) * mm_per_unit)
    return value if math.isfinite(value) else None


def _phase4_point_classes(handler: Any, *, theta_min_deg: float = 5.0) -> Phase4PointClasses:
    """Partition visible points by observation count and viewing-ray spread."""
    counts = _phase4_counts(handler)
    visible = np.asarray(handler.visible_feature_mask, dtype=bool).reshape(-1)
    if visible.shape != counts.shape or not np.array_equal(visible, counts > 0):
        raise ValueError("solver visible-feature mask does not match optimised detections")
    separations: list[float | None] = [None] * len(counts)
    one: list[int] = []
    several: list[int] = []
    unseen = np.flatnonzero(~visible).astype(int).tolist()
    # Directional separation is only computed for observed rays. Each pose is
    # expressed in target-local coordinates, so target motion is not mistaken
    # for a change in viewing direction.
    rows = _observation_rows(handler)
    missing_value = getattr(handler, "missing_poses", None)
    missing = np.asarray(missing_value if missing_value is not None else [], dtype=bool)
    if missing.size:
        rows = rows[~missing[rows[:, 1].astype(int)]]
    from pyCamSet.cameras.camera import Camera
    from pyCamSet.cameras.telecentric_camera import TelecentricCamera
    try:
        camera_models = [handler.camset[name] for name in handler.cam_names]
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError("camera model unavailable for viewing-ray classification") from exc
    if not camera_models or not all(isinstance(camera, Camera) for camera in camera_models):
        raise ValueError("camera model unavailable for viewing-ray classification")
    primitive = handler.bundlePrimitive
    params = np.asarray(handler.get_initial_params(), dtype=np.float64)
    values = primitive.return_bundle_primitives(params.copy())
    extr = np.asarray(values[1], dtype=np.float64)
    poses = np.asarray(values[2], dtype=np.float64)
    from pyCamSet.utils.general_utils import make_4x4h_tform
    from pyCamSet.optimisation.template_handler import _extrinsic_from_params
    # A telecentric model estimates only the rotation (3 extrinsic parameters);
    # its translation is the camera's seeded one, as the solver itself rebuilds it.
    # (On a real telecentric rig, extr[:, 3:] was empty.)
    camera_to_world_all = [np.linalg.inv(_extrinsic_from_params(
        extr[index], np.asarray(camera_models[index].extrinsic, dtype=np.float64)))
        for index in range(len(camera_models))]
    # Each image's target pose and inverse are the same for every point it sees, so they
    # are built once per image (identical arithmetic to building them per observation).
    target_to_world_cache: dict[int, np.ndarray] = {}
    world_to_target_cache: dict[int, np.ndarray] = {}
    point_lookup: dict[int, list[np.ndarray]] = {}
    for row in rows:
        point_lookup.setdefault(int(row[2]), []).append(row)
    target_points = handler.target.point_data.reshape(-1, 3)
    for point_index in np.flatnonzero(visible):
        point_rows = point_lookup.get(int(point_index), [])
        rays: list[np.ndarray] = []
        for row in point_rows:
            camera_index, image_index = int(row[0]), int(row[1])
            camera_to_world = camera_to_world_all[camera_index]
            if image_index not in target_to_world_cache:
                target_to_world_cache[image_index] = make_4x4h_tform(
                    poses[image_index, :3], poses[image_index, 3:])
            target_to_world = target_to_world_cache[image_index]
            if isinstance(camera_models[camera_index], TelecentricCamera):
                # There is no pinhole centre. Parallel chief rays acquire a
                # different *origin* as the target translates, not a new angle.
                axis_target = target_to_world[:3, :3].T @ camera_to_world[:3, 2]
                if np.linalg.norm(axis_target) > 0:
                    rays.append(axis_target / np.linalg.norm(axis_target))
                continue
            centre_world = camera_to_world[:3, 3]
            if image_index not in world_to_target_cache:
                world_to_target_cache[image_index] = np.linalg.inv(target_to_world)
            centre_target = (world_to_target_cache[image_index]
                             @ np.append(centre_world, 1.0))[:3]
            centre_target = centre_target - target_points[point_index]
            if np.linalg.norm(centre_target) > 0:
                rays.append(centre_target / np.linalg.norm(centre_target))
        if len(rays) < 2:
            one.append(int(point_index))
            separations[int(point_index)] = 0.0 if len(rays) else None
            continue
        angles = [math.degrees(math.acos(float(np.clip(a @ b, -1.0, 1.0))))
                  for index, a in enumerate(rays) for b in rays[index + 1:]]
        maximum = max(angles, default=0.0)
        separations[int(point_index)] = maximum
        (several if maximum >= theta_min_deg else one).append(int(point_index))
    return Phase4PointClasses(tuple(unseen), tuple(one), tuple(several), tuple(separations))


def derive_phase4(context: RunContext, *, include_playback: bool = True) -> Phase4Derivation | DerivationUnavailable:
    """Recalculate saved Phase 4 evidence and optionally project accepted steps."""
    if context.phase != "phase4":
        return _unavailable(context, "unavailable: the selected record is not Phase 4")
    if context.status not in {"complete", "incomplete"}:
        return _unavailable(context, "unavailable: the run has no checkable Phase 4 result")
    path = context.artifact("phase4")
    diagnostics = context.metadata.get("diagnostics")
    if path is None or not isinstance(diagnostics, Mapping):
        return _unavailable(context, "unavailable: nothing saved to check it against")
    try:
        camset = _load_camset(path)
        params = np.asarray(camset.calibration_params, dtype=np.float64).reshape(-1)
        handler = _phase4_handler(path, camset, diagnostics)
        expected_count = int(handler.get_initial_params().size)
        if params.size != expected_count:
            raise ValueError("saved parameter count does not match rebuilt handler layout")
        if tuple(handler.fixed_inds) != _phase4_indices(diagnostics):
            raise ValueError("rebuilt gauge indices do not match saved D4.2")
        if int(np.sum(handler.visible_feature_mask)) != int(diagnostics["D4.1_n_free_target_points"]):
            raise ValueError("rebuilt visible-point count does not match D4.1")
        classes = _phase4_point_classes(handler)
        visible = np.asarray(handler.visible_feature_mask, dtype=bool)
        nominal = np.asarray(handler.target.point_data, dtype=np.float64).reshape(-1, 3).copy()
        solved = np.asarray(handler.get_updated_target(params.copy()), dtype=np.float64).reshape(-1, 3)
        from pyCamSet.workflow.phase4 import _target_shape_change
        scale, displacement = _target_shape_change(
            handler, type("OptimisationResult", (), {"x": params.copy()})(), visible)
        saved_scale = float(diagnostics["D4.5_gauge_scale_factor"])
        saved_displacement = float(diagnostics["D4.7_mean_target_displacement_mm"])
        if not np.isclose(scale, saved_scale, rtol=1e-9, atol=1e-12):
            raise ValueError("recalculated D4.5 does not match the saved result")
        if not np.isclose(displacement, saved_displacement, rtol=1e-9, atol=1e-12):
            raise ValueError("recalculated D4.7 does not match the saved result")
        rows = _observation_rows(camset.calibration_handler)
        residuals = _finite_vector(getattr(camset, "calibration_result", None))
        final_px = float(np.linalg.norm(residuals[:2 * len(rows)].reshape(-1, 2), axis=1).mean())
        if (not math.isfinite(final_px)
                or not math.isclose(final_px,
                                    float(diagnostics.get("D4.3_final_euclid_px")),
                                    rel_tol=1e-9, abs_tol=1e-9)):
            raise ValueError("recalculated D4.3 final does not match saved residuals")
        counts = _phase4_counts(handler)
        measured = counts >= 2
        # Line up on the several-direction points (the well-determined ones) when there
        # are enough; otherwise on every point seen at least twice.
        align_mask = np.zeros(len(nominal), dtype=bool)
        align_mask[list(classes.several_directions)] = True
        if align_mask.sum() < 3:
            align_mask = measured
        try:
            refit_points, refit_scale = _phase4_display_refit(nominal, solved, align_mask)
            _sized, similarity_scale = _phase4_display_refit(nominal, solved, align_mask,
                                                             with_scale=True)
        except ValueError:
            # The display-only fit is optional; its absence does not invalidate
            # the saved-solution metrics or the recorded-step frames.
            refit_points, refit_scale, similarity_scale = None, None, None
        square_size = square_size_in_point_data_units(handler.target)
        edge_pairs = (handler.target.valid_map
                      if isinstance(handler.target.valid_map, np.ndarray) else None)
        scale_ratio = phase4_scale_consistency_ratio(
            nominal, solved, measured, square_size, pairs=edge_pairs)
        state_verified, fixed_difference = _phase4_start_state(context)
        frames: tuple[Phase4Frame, ...] = ()
        if include_playback:
            from pyCamSet.workflow.education_playback import read_accepted_step_trace
            trace = read_trace(context.run_dir, context.run_id, "phase4")
            first_step = next((record.get("payload", {}).get("steps", [])[0]
                               for record in trace.get("records", [])
                               if record.get("kind") == "accepted_step"
                               and record.get("payload", {}).get("steps")), None)
            initial = (np.asarray(first_step["x"],
                                  dtype=first_step.get("dtype", "float64"))
                       if first_step is not None else np.empty_like(params))
            validated = read_accepted_step_trace(
                trace, expected_start=initial, saved_params=params,
                handler=_phase4_handler(path, camset, diagnostics))
            if validated["status"] == "available":
                import copy
                from pyCamSet.cameras.camera_set import CameraSet
                # One rebuilt handler replays every frame. (Rebuilding two handlers
                # and re-classifying the points per frame took ~3 minutes on a real 8-camera rig.)
                # get_updated_target fills every free parameter from the frame, and get_camset
                # rewrites every estimated camera block, so a shared handler gives the same
                # values as fresh ones; get_camset hands back the handler's own Camera objects,
                # so each frame keeps deep copies.
                frame_handler = _phase4_handler(path, camset, diagnostics)
                built: list[Phase4Frame] = []
                for nit, cost, frame_params in zip(
                        validated["nits"], validated["costs"], validated["arrays"]):
                    try:
                        if np.asarray(frame_params).size != frame_handler.get_initial_params().size:
                            raise ValueError("recorded frame parameter layout differs")
                        point_frame = np.asarray(
                            frame_handler.get_updated_target(np.asarray(frame_params).copy()),
                            dtype=np.float64).reshape(-1, 3).copy()
                        # The classes depend only on the rebuilt handler's seeded start and its
                        # detections, which are the same for every frame.
                        frame_classes = classes
                        shared_set, pose_rows = frame_handler.get_camset(
                            np.asarray(frame_params).copy(), return_pose=True)
                        rig_set = CameraSet(camera_dict={
                            name: copy.deepcopy(shared_set[name])
                            for name in shared_set.get_names()})
                        rig_handler = frame_handler
                        target_poses = np.asarray(pose_rows, dtype=np.float64).reshape(-1, 3, 4)
                        target_poses = np.concatenate((
                            target_poses,
                            np.broadcast_to(np.array([0., 0., 0., 1.]),
                                            (len(target_poses), 1, 4))), axis=1)
                        missing_value = getattr(rig_handler, "missing_poses", None)
                        if missing_value is not None:
                            missing = np.asarray(missing_value, dtype=bool).reshape(-1)
                            target_poses[missing] = np.eye(4)
                        built.append(Phase4Frame(
                            int(nit), float(cost), point_frame, rig_set, target_poses,
                            frame_classes, "available", recorded_start=(len(built) == 0),
                            state_source_verified=(state_verified if not built else None),
                            held_parameter_difference=(fixed_difference if not built else False)))
                    except Exception as frame_exc:
                        built.append(Phase4Frame(
                            int(nit), float(cost), None, None, None, classes,
                            "unavailable", str(frame_exc), recorded_start=(len(built) == 0),
                            state_source_verified=(state_verified if not built else None),
                            held_parameter_difference=(fixed_difference if not built else False)))
                frames = tuple(built)
            else:
                frames = (Phase4Frame(
                    0, math.nan, None, None, None, classes, "unavailable",
                    str(validated.get("reason", "recorded steps unavailable"))),)
        units = getattr(camset, "_saved_point_data_units", LEGACY_UNVERIFIED)
        qualifier = units == LEGACY_UNVERIFIED or abs(scale_ratio - 1.0) > SCALE_RATIO_TOLERANCE
        return Phase4Derivation(
            "available", None, context.run_id, int(visible.sum()), final_px,
            float(scale), float(displacement), classes, refit_points, refit_scale,
            scale_ratio, qualifier, frames,
            _phase4_class_mean_displacement_mm(
                refit_points, nominal, classes.several_directions,
                millimetres_per_point_data_unit(handler.target)),
            saved_units_unverified=bool(units == LEGACY_UNVERIFIED),
            display_similarity_scale=similarity_scale)
    except (OSError, ValueError, TypeError, AttributeError, IndexError, KeyError,
            np.linalg.LinAlgError) as exc:
        return _unavailable(context,
                            f"cannot be recalculated for this run: {exc}")
