"""
Headless data model for the Optimisation tab.

This module is intentionally GUI-free and Optuna-free so it can be unit-tested
without a display or the optional ``optuna`` dependency.  It implements the
spec sections:

- §8: coverage / quality metrics
- §9: validity rules
- §10: objective functions
- §11: successful run definition
- §12: retention policy (best-N successes, default 20)
- §13: metadata saving
- §16: validation rules
- §20: trial result schema
- §24: success ranking semantics

The trial-runner worker (``pyCamSet.optimisation.optimisation_worker``) and
the optional Optuna adapter
(``pyCamSet.optimisation.optuna_adapter``) build on top of this module.
"""
from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Constants from the spec
# ---------------------------------------------------------------------------

OBJECTIVE_DIRECTION = "minimize"
"""Lower is better — applies to both fast and full mode (§10)."""

FAILURE_SCORE: float = 1e9
"""Score assigned to invalid trials (§9)."""

MAX_SUCCESSES_HARD_CAP: int = 20
"""Hard cap for retained successful runs (§4.4, §12)."""

# Full-mode penalty thresholds (§10.1, §23)
FULL_IMAGE_COVERAGE_MIN: float = 0.60
FULL_CAMERA_COVERAGE_MIN: float = 0.90
FULL_MULTICAM_COVERAGE_MIN: float = 0.50
FULL_POINT_RATIO_MIN: float = 0.70

# Fast-mode penalty thresholds (§10.2, §23)
FAST_IMAGE_COVERAGE_MIN: float = 0.50
FAST_CAMERA_COVERAGE_MIN: float = 0.60
FAST_MULTICAM_COVERAGE_MIN: float = 0.40

# Stage offsets used by the full-mode objective (§10.1).
STAGE_OFFSET_PHASE3: float = 0.00
STAGE_OFFSET_PHASE4: float = 0.25

# Score added to phase4-unsuccessful but still-valid trials (§10.1, recommended policy).
UNSUCCESSFUL_FULL_STAGE_OFFSET: float = 1.0


# ---------------------------------------------------------------------------
# Trial result
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """Structured trial outcome (§20).

    Use :meth:`as_dict` for JSON-ready serialisation.
    """

    trial_number: int
    mode: str  # "fast" | "full"
    valid: bool = False
    successful: bool = False
    success_stage: Optional[str] = None  # None | "phase3" | "phase4"
    self_calibration_run: bool = False

    effective_detector_settings: dict[str, Any] = field(default_factory=dict)
    fixed_detector_settings: dict[str, Any] = field(default_factory=dict)
    optimised_keys: list[str] = field(default_factory=list)
    optimised_bounds: dict[str, dict[str, float]] = field(default_factory=dict)
    trial_params: dict[str, Any] = field(default_factory=dict)

    image_coverage: float = 0.0
    camera_coverage: float = 0.0
    multicam_image_coverage: float = 0.0
    point_count: int = 0
    point_ratio: float = 0.0

    phase3_rpe: Optional[float] = None
    phase4_rpe: Optional[float] = None
    final_rpe: Optional[float] = None
    score: float = FAILURE_SCORE

    failure_reason: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    saved_metadata_path: Optional[str] = None

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Coverage metrics (§8)
# ---------------------------------------------------------------------------


def compute_coverage_metrics(
    features_per_im_per_cam: Any,
    *,
    baseline_point_count: Optional[int] = None,
) -> dict[str, float]:
    """Compute the §8 coverage metrics from a per-image-per-camera feature table.

    Parameters
    ----------
    features_per_im_per_cam:
        2-D array-like with shape ``(n_images, n_cameras)`` giving the number
        of detected features per image / camera cell.  Cells with ``0`` are
        treated as unused.
    baseline_point_count:
        Optional reference count for :math:`point\\_ratio` (§8.5).  When
        ``None`` or ``<= 0``, the point ratio is set to ``0.0``.

    Returns a dict with keys ``image_coverage``, ``camera_coverage``,
    ``multicam_image_coverage``, ``point_count``, ``point_ratio``.
    """
    import numpy as np

    arr = np.asarray(features_per_im_per_cam)
    if arr.ndim == 0 or arr.size == 0:
        return {
            "image_coverage": 0.0,
            "camera_coverage": 0.0,
            "multicam_image_coverage": 0.0,
            "point_count": 0,
            "point_ratio": 0.0,
        }
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    used = arr > 0
    n_images, n_cameras = arr.shape

    image_used = used.any(axis=1)
    cam_used = used.any(axis=0)
    multicam_image = used.sum(axis=1) >= 2

    point_count = int(arr.sum())
    if baseline_point_count is not None and baseline_point_count > 0:
        point_ratio = float(point_count) / float(baseline_point_count)
    else:
        point_ratio = 0.0

    return {
        "image_coverage": float(image_used.sum()) / float(n_images) if n_images else 0.0,
        "camera_coverage": float(cam_used.sum()) / float(n_cameras) if n_cameras else 0.0,
        "multicam_image_coverage": (
            float(multicam_image.sum()) / float(n_images) if n_images else 0.0
        ),
        "point_count": point_count,
        "point_ratio": point_ratio,
    }


# ---------------------------------------------------------------------------
# Validity (§9)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidityVerdict:
    valid: bool
    reason: Optional[str] = None


def assess_validity(
    coverage: dict[str, float],
    *,
    n_cameras_with_detections: int,
    n_valid_images: int,
    final_rpe: Optional[float] = None,
) -> ValidityVerdict:
    """Apply the §9 validity rules."""
    if coverage.get("point_count", 0) <= 0:
        return ValidityVerdict(False, "no usable detections")
    if n_cameras_with_detections < 2:
        return ValidityVerdict(False, "fewer than 2 cameras contributed detections")
    if n_valid_images < 3:
        return ValidityVerdict(False, "fewer than 3 valid poses/images")
    if final_rpe is not None and (math.isnan(final_rpe) or math.isinf(final_rpe)):
        return ValidityVerdict(False, "optimisation returned NaN/inf RPE")
    return ValidityVerdict(True)


# ---------------------------------------------------------------------------
# Objective functions (§10)
# ---------------------------------------------------------------------------


def _shortfall_sq(threshold: float, value: float) -> float:
    """``max(0, threshold - value)**2`` — penalty term used by both modes."""
    return max(0.0, threshold - float(value)) ** 2


def compute_full_score(
    *,
    valid: bool,
    success_stage: Optional[str],
    phase3_rpe: Optional[float],
    phase4_rpe: Optional[float],
    image_coverage: float,
    camera_coverage: float,
    multicam_image_coverage: float,
    point_ratio: float,
) -> tuple[float, Optional[float]]:
    """Full-mode objective (§10.1).

    Returns ``(score, final_rpe)``.  ``final_rpe`` is ``None`` for invalid trials.
    """
    if not valid:
        return FAILURE_SCORE, None

    penalties = (
        10.0 * _shortfall_sq(FULL_IMAGE_COVERAGE_MIN, image_coverage)
        + 20.0 * _shortfall_sq(FULL_CAMERA_COVERAGE_MIN, camera_coverage)
        + 10.0 * _shortfall_sq(FULL_MULTICAM_COVERAGE_MIN, multicam_image_coverage)
        + 2.0 * _shortfall_sq(FULL_POINT_RATIO_MIN, point_ratio)
    )

    if success_stage == "phase3" and phase3_rpe is not None:
        return STAGE_OFFSET_PHASE3 + float(phase3_rpe) + penalties, float(phase3_rpe)
    if success_stage == "phase4" and phase4_rpe is not None:
        return STAGE_OFFSET_PHASE4 + float(phase4_rpe) + penalties, float(phase4_rpe)

    # Valid but unsuccessful: keep ranking information per the §10.1
    # "recommended policy".  Prefer the latest available RPE.
    ranking_rpe = phase4_rpe if phase4_rpe is not None else phase3_rpe
    if ranking_rpe is None:
        return FAILURE_SCORE, None
    return UNSUCCESSFUL_FULL_STAGE_OFFSET + float(ranking_rpe) + penalties, float(ranking_rpe)


def compute_fast_score(
    *,
    valid: bool,
    image_coverage: float,
    camera_coverage: float,
    multicam_image_coverage: float,
    point_ratio: float,
) -> float:
    """Fast-mode objective (§10.2).  Lower is better."""
    if not valid:
        return FAILURE_SCORE
    return (
        -float(point_ratio)
        + 2.0 * _shortfall_sq(FAST_CAMERA_COVERAGE_MIN, camera_coverage)
        + 2.0 * _shortfall_sq(FAST_IMAGE_COVERAGE_MIN, image_coverage)
        + 3.0 * _shortfall_sq(FAST_MULTICAM_COVERAGE_MIN, multicam_image_coverage)
    )


# ---------------------------------------------------------------------------
# Retention (§12, §24)
# ---------------------------------------------------------------------------


def _stage_rank(stage: Optional[str]) -> int:
    """Lower is better — phase3 preferred over phase4 (§24)."""
    if stage == "phase3":
        return 0
    if stage == "phase4":
        return 1
    return 2


def _ranking_key(result: TrialResult) -> tuple[float, float, int, float, int]:
    """Tie-breaker key for §24."""
    rpe = result.final_rpe if result.final_rpe is not None else math.inf
    coverage_score = -(
        result.image_coverage + result.camera_coverage + result.multicam_image_coverage
    )
    return (
        result.score,
        rpe,
        _stage_rank(result.success_stage),
        coverage_score,
        result.trial_number,
    )


def clamp_retain_count(value: int, *, hard_cap: int = MAX_SUCCESSES_HARD_CAP) -> int:
    """Clamp the user-supplied "retain successes" value to ``[1, hard_cap]`` (§12)."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return hard_cap
    return max(1, min(hard_cap, v))


class SuccessRetention:
    """Bounded best-N container for successful full-mode trials (§12).

    The container retains the *best* successes by §24 ranking, not the
    earliest.  Operations are O(N log N) per insert which is fine for N <= 20.
    """

    def __init__(self, max_successes: int = MAX_SUCCESSES_HARD_CAP):
        self._max = clamp_retain_count(max_successes)
        self._items: list[TrialResult] = []

    @property
    def max_successes(self) -> int:
        return self._max

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self.ranked())

    def ranked(self) -> list[TrialResult]:
        """Return retained successes in §24 order (best first)."""
        return sorted(self._items, key=_ranking_key)

    def consider(self, result: TrialResult) -> bool:
        """Offer a result for retention.

        Only successful trials are accepted.  Returns ``True`` when the
        container ends up holding *result*, ``False`` otherwise.
        """
        if not result.successful:
            return False
        if len(self._items) < self._max:
            self._items.append(result)
            return True
        worst = max(self._items, key=_ranking_key)
        if _ranking_key(result) < _ranking_key(worst):
            self._items.remove(worst)
            self._items.append(result)
            return True
        return False

    def best(self) -> Optional[TrialResult]:
        if not self._items:
            return None
        return min(self._items, key=_ranking_key)


# ---------------------------------------------------------------------------
# Run-level validation (§16.2)
# ---------------------------------------------------------------------------


def validate_run_settings(
    *,
    f_loc: Path | str,
    n_trials: int,
    target_rpe: float,
    max_nfev_phase3: int,
    max_nfev_phase4: int,
    retain_successes: int,
    target_settings: dict[str, Any],
) -> list[str]:
    """Validate global run settings (§16.2).  Returns a list of error messages."""
    errors: list[str] = []
    p = Path(f_loc) if f_loc else None
    if p is None or not str(p):
        errors.append("Dataset path (f_loc) is required.")
    elif not p.exists():
        errors.append(f"Dataset path does not exist: {p}")
    elif not p.is_dir():
        errors.append(f"Dataset path is not a directory: {p}")

    if not isinstance(n_trials, int) or n_trials <= 0:
        errors.append("Number of trials must be a positive integer.")
    if not isinstance(target_rpe, (int, float)) or float(target_rpe) <= 0.0:
        errors.append("Target RPE must be > 0.")
    if not isinstance(max_nfev_phase3, int) or max_nfev_phase3 <= 0:
        errors.append("Phase 3 max_nfev must be a positive integer.")
    if not isinstance(max_nfev_phase4, int) or max_nfev_phase4 <= 0:
        errors.append("Phase 4 max_nfev must be a positive integer.")
    if not isinstance(retain_successes, int) or not (1 <= retain_successes <= MAX_SUCCESSES_HARD_CAP):
        errors.append(f"Retain successes must be an integer in [1, {MAX_SUCCESSES_HARD_CAP}].")

    # Target validation: just check the required ChArUco fields are present and sane.
    if target_settings is None or not isinstance(target_settings, dict):
        errors.append("Target settings are missing.")
        return errors
    if target_settings.get("target_type", "ChArUco") == "ChArUco":
        for required in ("num_squares_x", "num_squares_y", "square_size"):
            value = target_settings.get(required)
            try:
                if value is None or float(value) <= 0:
                    errors.append(f"Target field '{required}' must be > 0.")
            except (TypeError, ValueError):
                errors.append(f"Target field '{required}' must be numeric.")
    return errors


# ---------------------------------------------------------------------------
# Metadata writer (§13, §14)
# ---------------------------------------------------------------------------


def make_study_id() -> str:
    """Return a deterministic-looking but unique study id."""
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"study_{now}_{uuid.uuid4().hex[:8]}"


def default_output_dir(f_loc: Path | str, study_id: Optional[str] = None) -> Path:
    """Return the spec's recommended output directory (§4.1, §13.1)."""
    base = Path(f_loc) / "optimisation_runs"
    if study_id:
        return base / study_id
    return base


def trial_subdir_name(result: TrialResult) -> str:
    """Return the subdirectory name for one successful trial (§13.1)."""
    stage = result.success_stage or "valid"
    return f"trial_{result.trial_number:06d}_{stage}"


def write_trial_metadata(
    output_dir: Path | str,
    result: TrialResult,
    *,
    study_id: str,
    f_loc: Path | str,
    target_settings: dict[str, Any],
    calibration_controls: dict[str, Any],
    sampler_name: Optional[str] = None,
    seed: Optional[int] = None,
    camset_path: Optional[Path | str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Persist a per-trial metadata record (§13.2, §13.3).

    Returns the path to the written JSON file.  The function also stores the
    written path on ``result.saved_metadata_path`` for convenience.
    """
    out_dir = Path(output_dir) / trial_subdir_name(result)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "identity": {
            "study_id": study_id,
            "trial_number": result.trial_number,
            "timestamp": result.timestamp,
            "mode": result.mode,
            "success_stage": result.success_stage,
        },
        "paths": {
            "f_loc": str(f_loc),
            "output_dir": str(Path(output_dir)),
            "trial_dir": str(out_dir),
            "camset_path": str(camset_path) if camset_path else None,
        },
        "target": dict(target_settings),
        "detector_settings": {
            "effective": dict(result.effective_detector_settings),
            "fixed": dict(result.fixed_detector_settings),
            "optimised_keys": list(result.optimised_keys),
            "bounds": {k: dict(v) for k, v in result.optimised_bounds.items()},
        },
        "calibration_controls": dict(calibration_controls),
        "metrics": {
            "phase3_rpe": result.phase3_rpe,
            "phase4_rpe": result.phase4_rpe,
            "final_rpe": result.final_rpe,
            "score": result.score,
            "image_coverage": result.image_coverage,
            "camera_coverage": result.camera_coverage,
            "multicam_image_coverage": result.multicam_image_coverage,
            "point_count": result.point_count,
            "point_ratio": result.point_ratio,
            "valid": result.valid,
            "self_calibration_run": result.self_calibration_run,
        },
        "optuna": {
            "sampler": sampler_name,
            "seed": seed,
            "trial_params": dict(result.trial_params),
            "trial_state": "successful" if result.successful else "complete",
        },
        "diagnostics": {
            "failure_reason": result.failure_reason,
            "warnings": list(result.warnings),
        },
    }
    if extra:
        metadata["extra"] = dict(extra)
    json_path = out_dir / "metadata.json"
    json_path.write_text(json.dumps(metadata, indent=2, default=_json_default))
    result.saved_metadata_path = str(json_path)
    return json_path


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback for numpy scalars and Path objects."""
    try:
        import numpy as np  # local import; numpy is a hard pyCamSet dep
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def write_study_summary(
    output_dir: Path | str,
    *,
    study_id: str,
    started_at: float,
    finished_at: Optional[float],
    retention: SuccessRetention,
    n_trials_completed: int,
    n_trials_requested: int,
    mode: str,
    sampler_name: Optional[str] = None,
    seed: Optional[int] = None,
) -> Path:
    """Write a top-level ``study_summary.json`` (§13.1)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "study_id": study_id,
        "mode": mode,
        "started_at": datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat(timespec="seconds"),
        "finished_at": (
            datetime.fromtimestamp(finished_at, tz=timezone.utc).isoformat(timespec="seconds")
            if finished_at is not None
            else None
        ),
        "duration_sec": (finished_at - started_at) if finished_at is not None else None,
        "n_trials_completed": n_trials_completed,
        "n_trials_requested": n_trials_requested,
        "n_successes_retained": len(retention),
        "max_successes": retention.max_successes,
        "sampler": sampler_name,
        "seed": seed,
        "successes": [
            {
                "rank": rank + 1,
                "trial_number": r.trial_number,
                "success_stage": r.success_stage,
                "score": r.score,
                "final_rpe": r.final_rpe,
                "phase3_rpe": r.phase3_rpe,
                "phase4_rpe": r.phase4_rpe,
                "image_coverage": r.image_coverage,
                "camera_coverage": r.camera_coverage,
                "multicam_image_coverage": r.multicam_image_coverage,
                "point_ratio": r.point_ratio,
                "metadata_path": r.saved_metadata_path,
            }
            for rank, r in enumerate(retention.ranked())
        ],
    }
    path = out_dir / "study_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=_json_default))
    return path


__all__ = [
    "OBJECTIVE_DIRECTION",
    "FAILURE_SCORE",
    "MAX_SUCCESSES_HARD_CAP",
    "TrialResult",
    "ValidityVerdict",
    "compute_coverage_metrics",
    "assess_validity",
    "compute_full_score",
    "compute_fast_score",
    "SuccessRetention",
    "clamp_retain_count",
    "validate_run_settings",
    "make_study_id",
    "default_output_dir",
    "trial_subdir_name",
    "write_trial_metadata",
    "write_study_summary",
]
