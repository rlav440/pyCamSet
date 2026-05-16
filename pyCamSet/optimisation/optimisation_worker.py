"""Purpose: Headless backend worker for the Optimisation tab.

Status: Active optimisation orchestration used by the GUI and tests.

Future: Keep GUI-specific promotion and target widgets outside this module.

This module orchestrates one trial of the §27 execution contract:

- builds the effective detector settings (fixed + Optuna-sampled overrides),
- runs phase 1 detection on the dataset (or invokes the injected detection callable),
- runs phase 2 initial calibration from that trial's detections,
- runs phase 3 bundle adjustment and optionally phase 4 self-calibration,
- computes the §10 objective,
- writes per-success metadata via :mod:`pyCamSet.optimisation.optimisation_study`.

The worker exposes a synchronous, callable Python API (``run_trial``) and an
``OptimisationStudy`` driver that iterates trials.  GUI integration sits in
``pyCamSet.gui.optimisation_tab`` and runs this driver inside a ``QThread``.

Heavy operations are injected as callables (``detection_fn``,
``phase2_fn``, ``phase3_fn``, ``phase4_fn``) so the worker can be unit-tested without image
fixtures or scipy ``least_squares`` calls.  The default callables wrap the
existing pyCamSet entry points.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from pyCamSet.optimisation.charuco_detector_metadata import (
    CHARUCO_PARAMETER_METADATA,
    assemble_detection_options,
    clamp_to_bounds,
    coerce_value,
    default_fixed_settings,
    metadata_by_key,
    validate_all_rows,
)
from pyCamSet.optimisation.optimisation_study import (
    FAILURE_SCORE,
    SuccessRetention,
    TrialResult,
    assess_validity,
    compute_coverage_metrics,
    compute_fast_score,
    compute_full_score,
    default_output_dir,
    make_study_id,
    validate_run_settings,
    write_study_summary,
    write_trial_metadata,
)

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ParameterRowConfig:
    """User-supplied configuration for one detector parameter row."""

    key: str
    fixed: float
    optimise: bool = False
    lower: Optional[float] = None
    upper: Optional[float] = None


@dataclass
class TargetSettings:
    """Calibration target definition for ChArUco and Ccube trials."""

    target_type: str = "ChArUco"
    num_squares_x: int = 5
    num_squares_y: int = 5
    square_size: float = 30.0  # mm
    marker_fraction: float = 0.8
    a_dict: int = 0  # cv2.aruco.DICT_4X4_1000 numerically; resolved by the GUI
    n_points: int = 6
    length: float = 30.0  # mm
    border_fraction: float = 0.1
    legacy: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_type": self.target_type,
            "num_squares_x": self.num_squares_x,
            "num_squares_y": self.num_squares_y,
            "square_size": self.square_size,
            "marker_fraction": self.marker_fraction,
            "a_dict": self.a_dict,
            "n_points": self.n_points,
            "length": self.length,
            "border_fraction": self.border_fraction,
            "legacy": self.legacy,
        }


@dataclass
class CalibrationControls:
    """Non-detector optimisation settings (§4.4)."""

    outliers: str = "n"  # "ask" | "y" | "n"
    max_nfev_phase3: int = 100
    max_nfev_phase4: int = 100
    target_rpe: float = 1.0
    retain_successes: int = 20

    def as_dict(self) -> dict[str, Any]:
        return {
            "outliers": self.outliers,
            "max_nfev_phase3": self.max_nfev_phase3,
            "max_nfev_phase4": self.max_nfev_phase4,
            "target_rpe": self.target_rpe,
            "retain_successes": self.retain_successes,
        }


@dataclass
class RunConfig:
    """Full configuration for an optimisation study run (§19)."""

    f_loc: Path
    mode: str = "full"  # "fast" | "full"
    n_trials: int = 40
    seed: Optional[int] = None
    sampler_name: Optional[str] = None
    parameter_rows: list[ParameterRowConfig] = field(default_factory=list)
    target: TargetSettings = field(default_factory=TargetSettings)
    controls: CalibrationControls = field(default_factory=CalibrationControls)
    output_dir: Optional[Path] = None  # derived from f_loc when None

    def resolved_output_dir(self, study_id: str) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir) / study_id
        return default_output_dir(self.f_loc, study_id)


# ---------------------------------------------------------------------------
# Effective detector settings
# ---------------------------------------------------------------------------


def build_effective_settings(
    rows: list[ParameterRowConfig],
    *,
    sampled: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Merge fixed and sampled values into a flat ``{key: value}`` dict.

    For each parameter row:

    - if ``optimise`` is true *and* the key is in *sampled*, use the sampled
      value (coerced and clamped to absolute bounds);
    - otherwise use the row's fixed value (coerced and clamped).
    """
    by_key = metadata_by_key()
    out: dict[str, Any] = {}
    # Start from declared defaults so unspecified rows keep deterministic values.
    for entry in CHARUCO_PARAMETER_METADATA:
        out[entry["key"]] = entry["default"]
    sampled = sampled or {}
    for row in rows:
        entry = by_key.get(row.key)
        if entry is None:
            continue
        if row.optimise and row.key in sampled:
            value = sampled[row.key]
        else:
            value = row.fixed
        out[row.key] = clamp_to_bounds(entry, value)
    return out


def fixed_settings_only(rows: list[ParameterRowConfig]) -> dict[str, Any]:
    """Return only the user-defined fixed values (used for baseline run)."""
    return build_effective_settings(rows, sampled=None)


def detection_options_from_settings(values: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Group a flat settings dict back into the OpenCV sub-dict shape."""
    return assemble_detection_options(values)


# ---------------------------------------------------------------------------
# Detection / phase callable defaults
# ---------------------------------------------------------------------------


DetectionResult = dict[str, Any]
"""Shape of a detection-callable result.

Required keys:

- ``features_per_im_per_cam``: 2-D ndarray, shape (n_images, n_cameras).
- ``detections``: opaque payload passed to the phase 3 callable.

Optional keys:

- ``n_cameras_with_detections``: int (default: computed from the table)
- ``n_valid_images``: int (default: computed from the table)
- ``warnings``: list[str]
"""


def default_detection_fn(
    f_loc: Path,
    detection_options: dict[str, dict[str, Any]],
    target_settings: TargetSettings,
) -> DetectionResult:
    """Wrap :func:`pyCamSet.calibration.camera_calibrator.detect_datapoints_in_imfile`.

    Caching is disabled because each trial uses different detector settings.
    """
    from pyCamSet.calibration.camera_calibrator import detect_datapoints_in_imfile
    if target_settings.target_type == "Ccube":
        from pyCamSet.calibration_targets.target_Ccube import Ccube

        target = Ccube(
            n_points=target_settings.n_points,
            length=target_settings.length,
            # Ccube names the dictionary field aruco_dict; ChArUco names it a_dict.
            aruco_dict=target_settings.a_dict,
            border_fraction=target_settings.border_fraction,
            legacy=target_settings.legacy,
            detection_options=detection_options,
        )
    else:
        from pyCamSet.calibration_targets.target_charuco import ChArUco

        target = ChArUco(
            num_squares_x=target_settings.num_squares_x,
            num_squares_y=target_settings.num_squares_y,
            square_size=target_settings.square_size,
            marker_fraction=target_settings.marker_fraction,
            a_dict=target_settings.a_dict,
            legacy=target_settings.legacy,
            detection_options=detection_options,
        )
    detections, cam_res = detect_datapoints_in_imfile(
        f_loc=f_loc,
        calibration_target=target,
        caching=False,
    )
    features = np.asarray(detections.features_per_im_per_cam())
    return {
        "features_per_im_per_cam": features,
        "detections": detections,
        "cam_res": cam_res,
        "target": target,
        "f_loc": f_loc,
        "n_cameras_with_detections": int((features.sum(axis=0) > 0).sum()),
        "n_valid_images": int((features.sum(axis=1) > 0).sum()),
        "warnings": [],
    }


def _bundle_options(max_nfev: int, outliers: str) -> dict[str, Any]:
    """Build backend bundle-adjustment options aligned with the GUI phases."""
    outlier_mode = "y" if outliers == "ask" else (outliers or "n")
    return {
        "verbosity": 0,
        "fixed_pose": 0,
        "ref_cam": 0,
        "ref_pose": 0,
        "outliers": outlier_mode,
        "max_nfev": int(max_nfev),
    }


def default_phase2_fn(
    detection_payload: DetectionResult,
    *,
    controls: CalibrationControls,
) -> dict[str, Any]:
    """Run a fresh phase-2 initial calibration from one trial's detections."""
    from pyCamSet.calibration.camera_calibrator import run_initial_calibration

    detections = detection_payload.get("detections")
    target = detection_payload.get("target")
    cam_res = detection_payload.get("cam_res")
    if detections is None:
        raise RuntimeError("Phase 2 requires trial detections from phase 1.")
    if target is None:
        raise RuntimeError("Phase 2 requires the trial calibration target instance.")
    if cam_res is None:
        raise RuntimeError("Phase 2 requires camera resolutions from the detection pass.")
    cams = run_initial_calibration(
        detection=detections,
        calibration_target=target,
        cam_res=cam_res,
        save=False,
        fixed_params=None,
    )
    return {"camset": cams}


def default_phase3_fn(
    detection_payload: DetectionResult,
    phase2_payload: dict[str, Any],
    *,
    controls: CalibrationControls,
) -> dict[str, Any]:
    """Run phase 3 from this trial's fresh phase-2 initial calibration."""
    from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment_with_stats
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler

    cams = phase2_payload.get("camset")
    detections = detection_payload.get("detections")
    target = detection_payload.get("target")
    if cams is None:
        raise RuntimeError("Phase 3 requires the fresh phase-2 camset.")
    if detections is None:
        raise RuntimeError("Phase 3 requires trial detections from phase 1.")
    if target is None:
        raise RuntimeError("Phase 3 requires the trial calibration target instance.")
    handler = TemplateBundleHandler(
        camset=cams,
        target=target,
        detection=detections,
        fixed_params=None,
        options=_bundle_options(controls.max_nfev_phase3, controls.outliers),
    )
    optimisation, out_cams, stats = run_bundle_adjustment_with_stats(
        handler,
        threads=1,
    )
    final_rpe = float(stats.get("final_euclid", float("nan")))
    return {
        "rpe": final_rpe,
        "camset": out_cams,
        "optimisation": optimisation,
        "stats": dict(stats),
    }


def default_phase4_fn(
    detection_payload: DetectionResult,
    phase3_payload: dict[str, Any],
    *,
    controls: CalibrationControls,
) -> dict[str, Any]:
    """Run a phase-4 self-calibration starting from the phase-3 camset."""
    from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment_with_stats
    from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler

    phase3_cams = phase3_payload.get("camset")
    detections = detection_payload.get("detections")
    target = detection_payload.get("target")
    if phase3_cams is None:
        raise RuntimeError("Phase 4 requires the fresh phase-3 camset.")
    if detections is None:
        raise RuntimeError("Phase 4 requires trial detections from phase 1.")
    if target is None:
        raise RuntimeError("Phase 4 requires the trial calibration target instance.")
    handler = SelfBundleHandler(
        camset=phase3_cams,
        target=target,
        detection=detections,
        fixed_params=None,
        options=_bundle_options(controls.max_nfev_phase4, controls.outliers),
    )
    handler.set_from_templated_camset(phase3_cams)
    optimisation, out_cams, stats = run_bundle_adjustment_with_stats(
        handler,
        threads=1,
    )
    final_rpe = float(stats.get("final_euclid", float("nan")))
    return {
        "rpe": final_rpe,
        "camset": out_cams,
        "optimisation": optimisation,
        "stats": dict(stats),
    }


# ---------------------------------------------------------------------------
# Cancellation token
# ---------------------------------------------------------------------------


class CancelToken:
    """Thread-safe cancellation flag (§17.1).

    The §17.1 contract is: cancellation stops new trials from starting and lets
    the in-flight trial finish cleanly.  Callers should periodically check
    :meth:`is_cancelled`.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


# ---------------------------------------------------------------------------
# Single-trial executor
# ---------------------------------------------------------------------------


def _compute_metrics(
    detection_payload: DetectionResult,
    baseline_point_count: Optional[int],
) -> tuple[dict[str, float], int, int]:
    """Compute coverage and count metrics from a detection payload."""
    features = np.asarray(detection_payload["features_per_im_per_cam"])
    coverage = compute_coverage_metrics(
        features, baseline_point_count=baseline_point_count
    )
    n_cams = int(detection_payload.get(
        "n_cameras_with_detections", int((features.sum(axis=0) > 0).sum())
    ))
    n_imgs = int(detection_payload.get(
        "n_valid_images", int((features.sum(axis=1) > 0).sum())
    ))
    return coverage, n_cams, n_imgs


def run_trial(
    trial_number: int,
    rows: list[ParameterRowConfig],
    *,
    config: RunConfig,
    sampled: Optional[dict[str, Any]] = None,
    baseline_point_count: Optional[int],
    detection_fn: Callable[[Path, dict, TargetSettings], DetectionResult],
    phase2_fn: Optional[Callable[..., dict[str, Any]]] = None,
    phase3_fn: Optional[Callable[..., dict[str, Any]]] = None,
    phase4_fn: Optional[Callable[..., dict[str, Any]]] = None,
    metadata_writer: Optional[Callable[[TrialResult, dict[str, Any]], None]] = None,
) -> tuple[TrialResult, dict[str, Any]]:
    """Run one optimisation trial end-to-end.

    Returns ``(result, payload)`` where ``payload`` contains the trial's
    detection / phase outputs and is suitable for downstream camset saving.
    Failure modes are caught and surfaced via :attr:`TrialResult.failure_reason`
    rather than re-raised; this matches §22's per-trial recoverable failure
    policy.
    """
    effective = build_effective_settings(rows, sampled=sampled)
    grouped = detection_options_from_settings(effective)
    optimised_keys = [r.key for r in rows if r.optimise]
    bounds = {
        r.key: {"lower": float(r.lower), "upper": float(r.upper)}
        for r in rows
        if r.optimise and r.lower is not None and r.upper is not None
    }
    result = TrialResult(
        trial_number=trial_number,
        mode=config.mode,
        effective_detector_settings=effective,
        fixed_detector_settings={r.key: r.fixed for r in rows},
        optimised_keys=optimised_keys,
        optimised_bounds=bounds,
        trial_params=dict(sampled or {}),
    )

    payload: dict[str, Any] = {"detection_options": grouped}

    # ---- Stage A: detection -------------------------------------------------
    try:
        detection_payload = detection_fn(Path(config.f_loc), grouped, config.target)
    except Exception as exc:  # pragma: no cover - defensive; tests inject fakes
        result.failure_reason = f"detection_fn raised: {exc!r}"
        result.score = FAILURE_SCORE
        if metadata_writer is not None:
            metadata_writer(result, payload)
        return result, payload

    coverage, n_cams, n_imgs = _compute_metrics(detection_payload, baseline_point_count)
    result.image_coverage = coverage["image_coverage"]
    result.camera_coverage = coverage["camera_coverage"]
    result.multicam_image_coverage = coverage["multicam_image_coverage"]
    result.point_count = coverage["point_count"]
    result.point_ratio = coverage["point_ratio"]
    payload["detection"] = detection_payload

    verdict = assess_validity(
        coverage,
        n_cameras_with_detections=n_cams,
        n_valid_images=n_imgs,
    )
    result.valid = verdict.valid
    if not verdict.valid:
        result.failure_reason = verdict.reason
        result.score = FAILURE_SCORE
        if metadata_writer is not None:
            metadata_writer(result, payload)
        return result, payload

    # ---- Fast mode short-circuit -------------------------------------------
    if config.mode == "fast":
        result.score = compute_fast_score(
            valid=True,
            image_coverage=result.image_coverage,
            camera_coverage=result.camera_coverage,
            multicam_image_coverage=result.multicam_image_coverage,
            point_ratio=result.point_ratio,
        )
        if metadata_writer is not None:
            metadata_writer(result, payload)
        return result, payload

    # ---- Stage B: phase 2 ---------------------------------------------------
    phase2_runner = phase2_fn or default_phase2_fn
    try:
        phase2 = phase2_runner(detection_payload, controls=config.controls)
    except Exception as exc:
        result.failure_reason = f"phase2_fn raised: {exc!r}"
        result.score = FAILURE_SCORE
        if metadata_writer is not None:
            metadata_writer(result, payload)
        return result, payload
    payload["phase2"] = phase2

    # ---- Stage C: phase 3 ---------------------------------------------------
    phase3_runner = phase3_fn or default_phase3_fn

    try:
        phase3 = phase3_runner(detection_payload, phase2, controls=config.controls)
    except Exception as exc:
        result.failure_reason = f"phase3_fn raised: {exc!r}"
        result.score = FAILURE_SCORE
        if metadata_writer is not None:
            metadata_writer(result, payload)
        return result, payload

    phase3_rpe = float(phase3.get("rpe")) if phase3.get("rpe") is not None else None
    result.phase3_rpe = phase3_rpe
    payload["phase3"] = phase3
    if phase3_rpe is None or math.isnan(phase3_rpe) or math.isinf(phase3_rpe):
        result.valid = False
        result.failure_reason = "phase 3 returned NaN/inf or no RPE"
        result.score = FAILURE_SCORE
        if metadata_writer is not None:
            metadata_writer(result, payload)
        return result, payload

    if phase3_rpe <= config.controls.target_rpe:
        result.success_stage = "phase3"
        result.successful = True
        score, final_rpe = compute_full_score(
            valid=True,
            success_stage="phase3",
            phase3_rpe=phase3_rpe,
            phase4_rpe=None,
            image_coverage=result.image_coverage,
            camera_coverage=result.camera_coverage,
            multicam_image_coverage=result.multicam_image_coverage,
            point_ratio=result.point_ratio,
        )
        result.score = score
        result.final_rpe = final_rpe
        if metadata_writer is not None:
            metadata_writer(result, payload)
        return result, payload

    # ---- Stage D: phase 4 ---------------------------------------------------
    result.self_calibration_run = True
    phase4_runner = phase4_fn or default_phase4_fn
    try:
        phase4 = phase4_runner(detection_payload, phase3, controls=config.controls)
    except Exception as exc:
        result.failure_reason = f"phase4_fn raised: {exc!r}"
        result.score = FAILURE_SCORE
        result.valid = False
        if metadata_writer is not None:
            metadata_writer(result, payload)
        return result, payload

    phase4_rpe = float(phase4.get("rpe")) if phase4.get("rpe") is not None else None
    result.phase4_rpe = phase4_rpe
    payload["phase4"] = phase4
    if phase4_rpe is None or math.isnan(phase4_rpe) or math.isinf(phase4_rpe):
        result.valid = False
        result.failure_reason = "phase 4 returned NaN/inf or no RPE"
        result.score = FAILURE_SCORE
        if metadata_writer is not None:
            metadata_writer(result, payload)
        return result, payload

    if phase4_rpe <= config.controls.target_rpe:
        result.success_stage = "phase4"
        result.successful = True

    score, final_rpe = compute_full_score(
        valid=True,
        success_stage=result.success_stage,
        phase3_rpe=phase3_rpe,
        phase4_rpe=phase4_rpe,
        image_coverage=result.image_coverage,
        camera_coverage=result.camera_coverage,
        multicam_image_coverage=result.multicam_image_coverage,
        point_ratio=result.point_ratio,
    )
    result.score = score
    result.final_rpe = final_rpe
    if metadata_writer is not None:
        metadata_writer(result, payload)
    return result, payload


# ---------------------------------------------------------------------------
# Study driver
# ---------------------------------------------------------------------------


@dataclass
class StudyProgress:
    """Snapshot of in-flight study state, suitable for GUI signalling."""

    trial_number: int
    total_trials: int
    best_score: float
    best_stage: Optional[str]
    best_phase3_rpe: Optional[float]
    best_phase4_rpe: Optional[float]
    n_successes: int
    elapsed_sec: float
    last_result: TrialResult


class OptimisationStudy:
    """Drive a sequence of trials end-to-end.

    The driver is sampler-agnostic: ``sampler`` is any callable
    ``(trial_number, rows) -> dict[str, float]`` that returns Optuna-style
    sampled overrides for the optimisable keys.  The Optuna adapter wires
    Optuna in via that callable; tests pass in a deterministic stub.
    """

    def __init__(
        self,
        config: RunConfig,
        *,
        sampler: Callable[[int, list[ParameterRowConfig]], dict[str, Any]],
        detection_fn: Callable[..., DetectionResult] = default_detection_fn,
        phase2_fn: Optional[Callable[..., dict[str, Any]]] = None,
        phase3_fn: Optional[Callable[..., dict[str, Any]]] = None,
        phase4_fn: Optional[Callable[..., dict[str, Any]]] = None,
        cancel_token: Optional[CancelToken] = None,
        progress_cb: Optional[Callable[[StudyProgress], None]] = None,
        write_metadata: bool = True,
    ):
        self.config = config
        self.sampler = sampler
        self.detection_fn = detection_fn
        self.phase2_fn = phase2_fn
        self.phase3_fn = phase3_fn
        self.phase4_fn = phase4_fn
        self.cancel_token = cancel_token or CancelToken()
        self.progress_cb = progress_cb
        self.write_metadata = write_metadata

        self.study_id = make_study_id()
        self.retention = SuccessRetention(config.controls.retain_successes)
        self.results: list[TrialResult] = []
        self.baseline_point_count: Optional[int] = None
        self._started_at: Optional[float] = None
        self._finished_at: Optional[float] = None

    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Run §16 validation; returns list of error messages."""
        errors = validate_run_settings(
            f_loc=self.config.f_loc,
            n_trials=self.config.n_trials,
            target_rpe=self.config.controls.target_rpe,
            max_nfev_phase3=self.config.controls.max_nfev_phase3,
            max_nfev_phase4=self.config.controls.max_nfev_phase4,
            retain_successes=self.config.controls.retain_successes,
            target_settings=self.config.target.as_dict(),
        )
        errors.extend(
            validate_all_rows(
                {
                    "key": r.key,
                    "fixed": r.fixed,
                    "optimise": r.optimise,
                    "lower": r.lower,
                    "upper": r.upper,
                }
                for r in self.config.parameter_rows
            )
        )
        return errors

    # ------------------------------------------------------------------

    def compute_baseline(self) -> Optional[int]:
        """Run detection once with fixed defaults to anchor :math:`point\\_ratio` (§18)."""
        baseline_settings = fixed_settings_only(self.config.parameter_rows)
        grouped = detection_options_from_settings(baseline_settings)
        try:
            payload = self.detection_fn(Path(self.config.f_loc), grouped, self.config.target)
        except Exception as exc:
            _LOG.warning("Baseline detection failed: %r", exc)
            return None
        features = np.asarray(payload["features_per_im_per_cam"])
        baseline = int(features.sum())
        self.baseline_point_count = baseline if baseline > 0 else None
        return self.baseline_point_count

    # ------------------------------------------------------------------

    def _metadata_writer_for(self, output_dir: Path) -> Callable[[TrialResult, dict[str, Any]], None]:
        cfg = self.config
        study_id = self.study_id

        def _write(result: TrialResult, payload: dict[str, Any]) -> None:
            if not self.write_metadata:
                return
            if not result.successful:
                return  # §13: per-success metadata
            camset_path = None
            detection_pickle_path = None
            phase2_camset_path = None
            phase3_camset_path = None
            phase4_camset_path = None
            phase2_camset = payload.get("phase2", {}).get("camset") if "phase2" in payload else None
            phase4_camset = payload.get("phase4", {}).get("camset") if "phase4" in payload else None
            phase3_camset = payload.get("phase3", {}).get("camset") if "phase3" in payload else None
            saved_camset = phase4_camset if result.success_stage == "phase4" else phase3_camset
            try:
                from pyCamSet.optimisation.optimisation_study import trial_subdir_name
                from pyCamSet.utils.saving import save_pickle
                trial_dir = output_dir / trial_subdir_name(result)
                trial_dir.mkdir(parents=True, exist_ok=True)
                detection_payload = payload.get("detection", {})
                detection_dict = detection_payload if isinstance(detection_payload, dict) else {}
                has_detections = "detections" in detection_dict and detection_dict["detections"] is not None
                if has_detections:
                    detections = detection_dict["detections"]
                    detection_pickle_path = trial_dir / "detected_datapoints.pickle"
                    save_pickle(
                        (detections, detection_dict.get("cam_res"))
                        if detection_dict.get("cam_res") is not None
                        else detections,
                        detection_pickle_path,
                    )
                if phase2_camset is not None and hasattr(phase2_camset, "save"):
                    phase2_camset_path = trial_dir / "camset_phase2.json"
                    phase2_camset.save(str(phase2_camset_path))
                if phase3_camset is not None and hasattr(phase3_camset, "save"):
                    phase3_camset_path = trial_dir / "camset_phase3.json"
                    phase3_camset.save(str(phase3_camset_path))
                if phase4_camset is not None and hasattr(phase4_camset, "save"):
                    phase4_camset_path = trial_dir / "camset_phase4.json"
                    phase4_camset.save(str(phase4_camset_path))
                if saved_camset is not None and hasattr(saved_camset, "save"):
                    camset_path = phase4_camset_path if result.success_stage == "phase4" else phase3_camset_path
            except Exception as exc:  # pragma: no cover
                _LOG.warning("Could not prepare optimisation trial artefacts: %r", exc)
            write_trial_metadata(
                output_dir,
                result,
                study_id=study_id,
                f_loc=cfg.f_loc,
                target_settings=cfg.target.as_dict(),
                calibration_controls=cfg.controls.as_dict(),
                sampler_name=cfg.sampler_name,
                seed=cfg.seed,
                camset_path=camset_path,
                extra={
                    "artifacts": {
                        "detected_datapoints_pickle": str(detection_pickle_path) if detection_pickle_path else None,
                        "phase2_initial_camset": str(phase2_camset_path) if phase2_camset_path else None,
                        "phase3_camset": str(phase3_camset_path) if phase3_camset_path else None,
                        "phase4_camset": str(phase4_camset_path) if phase4_camset_path else None,
                    },
                    "phase_sources": {
                        "phase2_run_id": payload.get("phase3", {}).get("source_phase2_run_id"),
                        "phase2_initial_camset": (
                            str(phase2_camset_path)
                            if phase2_camset_path is not None
                            else payload.get("phase3", {}).get("source_phase2_camset")
                        ),
                    },
                },
            )

        return _write

    # ------------------------------------------------------------------

    def run(self) -> SuccessRetention:
        """Run the study to completion (or cancellation).  Returns the retention set."""
        errors = self.validate()
        if errors:
            raise ValueError("Run configuration invalid: " + "; ".join(errors))

        self._started_at = time.time()
        output_dir = self.config.resolved_output_dir(self.study_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = self._metadata_writer_for(output_dir)

        if self.config.mode == "full" and self.baseline_point_count is None:
            self.compute_baseline()

        completed = 0
        for trial_number in range(self.config.n_trials):
            if self.cancel_token.is_cancelled():
                _LOG.info("Study cancelled before trial %d", trial_number)
                break
            sampled = self.sampler(trial_number, self.config.parameter_rows)
            result, _payload = run_trial(
                trial_number,
                self.config.parameter_rows,
                config=self.config,
                sampled=sampled,
                baseline_point_count=self.baseline_point_count,
                detection_fn=self.detection_fn,
                phase2_fn=self.phase2_fn,
                phase3_fn=self.phase3_fn,
                phase4_fn=self.phase4_fn,
                metadata_writer=writer,
            )
            self.results.append(result)
            self.retention.consider(result)
            completed += 1

            if self.progress_cb is not None:
                best = self.retention.best()
                self.progress_cb(
                    StudyProgress(
                        trial_number=trial_number,
                        total_trials=self.config.n_trials,
                        best_score=best.score if best else float("inf"),
                        best_stage=best.success_stage if best else None,
                        best_phase3_rpe=best.phase3_rpe if best else None,
                        best_phase4_rpe=best.phase4_rpe if best else None,
                        n_successes=len(self.retention),
                        elapsed_sec=time.time() - self._started_at,
                        last_result=result,
                    )
                )

        self._finished_at = time.time()
        if self.write_metadata:
            write_study_summary(
                output_dir,
                study_id=self.study_id,
                started_at=self._started_at,
                finished_at=self._finished_at,
                retention=self.retention,
                n_trials_completed=completed,
                n_trials_requested=self.config.n_trials,
                mode=self.config.mode,
                sampler_name=self.config.sampler_name,
                seed=self.config.seed,
            )
        return self.retention


__all__ = [
    "ParameterRowConfig",
    "TargetSettings",
    "CalibrationControls",
    "RunConfig",
    "CancelToken",
    "StudyProgress",
    "OptimisationStudy",
    "run_trial",
    "build_effective_settings",
    "fixed_settings_only",
    "detection_options_from_settings",
    "default_detection_fn",
    "default_phase2_fn",
    "default_phase3_fn",
    "default_phase4_fn",
]
