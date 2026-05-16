"""Purpose: Unit tests for the headless optimisation data model and worker.

Status: Active regression coverage for the Optimisation tab follow-up work.

Future: Add GUI-level Qt tests when the project test environment includes pytest-qt.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pyCamSet.optimisation.charuco_detector_metadata import (
    CHARUCO_PARAMETER_METADATA,
    assemble_detection_options,
    clamp_to_bounds,
    coerce_value,
    default_fixed_settings,
    metadata_by_key,
    numeric_keys,
    validate_all_rows,
    validate_parameter_row,
)
from pyCamSet.optimisation.optimisation_study import (
    FAILURE_SCORE,
    MAX_SUCCESSES_HARD_CAP,
    SuccessRetention,
    TrialResult,
    assess_validity,
    clamp_retain_count,
    compute_coverage_metrics,
    compute_fast_score,
    compute_full_score,
    default_output_dir,
    make_study_id,
    trial_subdir_name,
    validate_run_settings,
    write_study_summary,
    write_trial_metadata,
)
from pyCamSet.optimisation.optimisation_worker import (
    CalibrationControls,
    OptimisationStudy,
    ParameterRowConfig,
    RunConfig,
    TargetSettings,
    build_effective_settings,
    detection_options_from_settings,
    run_trial,
)
from pyCamSet.optimisation.optimisation_promotion import promote_retained_trial


class _WorkspaceManager:
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        for phase in ("phase1", "phase2", "phase3", "phase4"):
            (self.workspace_path / f"{phase}_runs").mkdir(parents=True, exist_ok=True)

    def save_run(self, phase: str, run_id: str, metadata: dict) -> Path:
        run_dir = self.workspace_path / f"{phase}_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "metadata.json"
        path.write_text(json.dumps(metadata, indent=2, default=str))
        return path


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_table_unique_keys():
    keys = [e["key"] for e in CHARUCO_PARAMETER_METADATA]
    assert len(keys) == len(set(keys))
    assert numeric_keys() == keys


def test_metadata_groups_are_known():
    allowed = {"CharucoParameters", "DetectorParameters", "RefineParameters"}
    for entry in CHARUCO_PARAMETER_METADATA:
        assert entry["group"] in allowed
        assert entry["dtype"] in {"int", "float"}
        assert entry["min"] <= entry["default"] <= entry["max"]


def test_coerce_value_and_clamp():
    entry = metadata_by_key()["minMarkers"]  # int [0, 4], default 2
    assert coerce_value(entry, 2.7) == 3
    assert clamp_to_bounds(entry, 99) == 4
    assert clamp_to_bounds(entry, -3) == 0

    entry_f = metadata_by_key()["errorCorrectionRate"]  # float [0, 1]
    assert clamp_to_bounds(entry_f, 1.5) == pytest.approx(1.0)
    assert clamp_to_bounds(entry_f, -0.5) == pytest.approx(0.0)


def test_default_fixed_settings_groups():
    settings = default_fixed_settings()
    assert "DetectorParameters" in settings
    assert "CharucoParameters" in settings
    # Every numeric key appears in exactly one group.
    flat = {k for grp in settings.values() for k in grp}
    assert flat == set(numeric_keys())


def test_assemble_drops_unknown_keys():
    grouped = assemble_detection_options({"adaptiveThreshConstant": 10.0, "nonsense": 1})
    assert grouped["DetectorParameters"]["adaptiveThreshConstant"] == 10.0
    assert "nonsense" not in grouped.get("DetectorParameters", {})


def test_validate_parameter_row_happy_path():
    entry = metadata_by_key()["adaptiveThreshConstant"]
    errors = validate_parameter_row(
        entry, fixed_value=7.0, optimise=True, lower=1.0, upper=20.0
    )
    assert errors == []


def test_validate_parameter_row_detects_swapped_and_out_of_range_bounds():
    entry = metadata_by_key()["adaptiveThreshConstant"]
    errors = validate_parameter_row(
        entry, fixed_value=7.0, optimise=True, lower=20.0, upper=1.0
    )
    assert any("exceeds upper bound" in e for e in errors)
    errors = validate_parameter_row(
        entry, fixed_value=7.0, optimise=True, lower=-5.0, upper=200.0
    )
    assert any("exceed allowed range" in e for e in errors)


def test_validate_parameter_row_rejects_out_of_range_fixed():
    entry = metadata_by_key()["minMarkers"]  # max=4
    errors = validate_parameter_row(entry, fixed_value=99, optimise=False)
    assert any("outside allowed bounds" in e for e in errors)


def test_validate_all_rows_unknown_key():
    errors = validate_all_rows([{"key": "doesNotExist", "fixed": 1, "optimise": False}])
    assert any("Unknown parameter key" in e for e in errors)


# ---------------------------------------------------------------------------
# Coverage metrics
# ---------------------------------------------------------------------------


def test_coverage_metrics_full_grid():
    arr = np.array([
        [4, 4, 0],
        [3, 3, 2],
        [0, 5, 5],
        [0, 0, 0],
    ])
    m = compute_coverage_metrics(arr, baseline_point_count=20)
    assert m["image_coverage"] == pytest.approx(3 / 4)
    assert m["camera_coverage"] == pytest.approx(3 / 3)
    assert m["multicam_image_coverage"] == pytest.approx(3 / 4)
    assert m["point_count"] == 26
    assert m["point_ratio"] == pytest.approx(26 / 20)


def test_coverage_metrics_empty():
    m = compute_coverage_metrics(np.zeros((0, 0)))
    assert m["point_count"] == 0
    assert m["image_coverage"] == 0


def test_coverage_metrics_no_baseline_yields_zero_ratio():
    arr = np.array([[1, 2], [3, 4]])
    m = compute_coverage_metrics(arr, baseline_point_count=None)
    assert m["point_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------


def test_validity_requires_two_cameras():
    cov = {"point_count": 10}
    verdict = assess_validity(cov, n_cameras_with_detections=1, n_valid_images=5)
    assert not verdict.valid
    assert "2 cameras" in verdict.reason


def test_validity_requires_three_images():
    cov = {"point_count": 10}
    verdict = assess_validity(cov, n_cameras_with_detections=2, n_valid_images=2)
    assert not verdict.valid


def test_validity_rejects_nan_rpe():
    cov = {"point_count": 10}
    verdict = assess_validity(
        cov, n_cameras_with_detections=2, n_valid_images=4, final_rpe=float("nan")
    )
    assert not verdict.valid


def test_validity_passes_normal_case():
    cov = {"point_count": 10}
    verdict = assess_validity(cov, n_cameras_with_detections=3, n_valid_images=10, final_rpe=0.5)
    assert verdict.valid


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------


def test_full_score_phase3_success_uses_phase3_rpe():
    score, final_rpe = compute_full_score(
        valid=True,
        success_stage="phase3",
        phase3_rpe=0.5,
        phase4_rpe=None,
        image_coverage=1.0,
        camera_coverage=1.0,
        multicam_image_coverage=1.0,
        point_ratio=1.0,
    )
    assert score == pytest.approx(0.5)  # stage_offset 0 + rpe 0.5 + no penalties
    assert final_rpe == pytest.approx(0.5)


def test_full_score_phase4_has_stage_offset():
    score, final_rpe = compute_full_score(
        valid=True,
        success_stage="phase4",
        phase3_rpe=2.0,
        phase4_rpe=0.8,
        image_coverage=1.0,
        camera_coverage=1.0,
        multicam_image_coverage=1.0,
        point_ratio=1.0,
    )
    assert score == pytest.approx(0.25 + 0.8)
    assert final_rpe == pytest.approx(0.8)


def test_full_score_penalty_quadratic():
    score, _ = compute_full_score(
        valid=True,
        success_stage="phase3",
        phase3_rpe=0.0,
        phase4_rpe=None,
        image_coverage=0.50,  # shortfall 0.10
        camera_coverage=1.0,
        multicam_image_coverage=1.0,
        point_ratio=1.0,
    )
    # only image-coverage penalty contributes: 10 * 0.1^2 = 0.10
    assert score == pytest.approx(0.10)


def test_full_score_invalid_returns_failure_sentinel():
    score, final_rpe = compute_full_score(
        valid=False,
        success_stage=None,
        phase3_rpe=None,
        phase4_rpe=None,
        image_coverage=0,
        camera_coverage=0,
        multicam_image_coverage=0,
        point_ratio=0,
    )
    assert score == FAILURE_SCORE
    assert final_rpe is None


def test_fast_score_rewards_higher_point_ratio():
    higher = compute_fast_score(
        valid=True,
        image_coverage=1.0,
        camera_coverage=1.0,
        multicam_image_coverage=1.0,
        point_ratio=2.0,
    )
    lower = compute_fast_score(
        valid=True,
        image_coverage=1.0,
        camera_coverage=1.0,
        multicam_image_coverage=1.0,
        point_ratio=1.0,
    )
    assert higher < lower  # higher point ratio => smaller (better) score


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


def _make_success(trial_number: int, score: float, stage: str = "phase3", rpe: float = 0.5) -> TrialResult:
    return TrialResult(
        trial_number=trial_number,
        mode="full",
        valid=True,
        successful=True,
        success_stage=stage,
        phase3_rpe=rpe if stage == "phase3" else 2.0,
        phase4_rpe=rpe if stage == "phase4" else None,
        final_rpe=rpe,
        score=score,
        image_coverage=0.9,
        camera_coverage=0.9,
        multicam_image_coverage=0.9,
        point_ratio=1.0,
    )


def test_clamp_retain_count():
    assert clamp_retain_count(0) == 1
    assert clamp_retain_count(50) == MAX_SUCCESSES_HARD_CAP
    assert clamp_retain_count(7) == 7
    assert clamp_retain_count("oops") == MAX_SUCCESSES_HARD_CAP


def test_retention_keeps_best_when_full():
    ret = SuccessRetention(max_successes=3)
    assert ret.consider(_make_success(1, 5.0))
    assert ret.consider(_make_success(2, 3.0))
    assert ret.consider(_make_success(3, 4.0))
    # Worst (5.0) should be replaced by 2.0
    assert ret.consider(_make_success(4, 2.0))
    scores = [r.score for r in ret.ranked()]
    assert scores == [2.0, 3.0, 4.0]


def test_retention_rejects_worse_than_worst():
    ret = SuccessRetention(max_successes=2)
    ret.consider(_make_success(1, 1.0))
    ret.consider(_make_success(2, 2.0))
    assert not ret.consider(_make_success(3, 5.0))
    assert len(ret) == 2


def test_retention_ignores_non_success():
    ret = SuccessRetention()
    failure = _make_success(1, 1.0)
    failure.successful = False
    assert not ret.consider(failure)


def test_retention_tiebreaker_prefers_phase3_then_lower_trial_number():
    ret = SuccessRetention(max_successes=5)
    ret.consider(_make_success(7, 1.0, stage="phase4"))
    ret.consider(_make_success(3, 1.0, stage="phase3"))
    best = ret.best()
    assert best.success_stage == "phase3"
    assert best.trial_number == 3


# ---------------------------------------------------------------------------
# Metadata writer / summary
# ---------------------------------------------------------------------------


def test_write_trial_metadata_creates_json(tmp_path: Path):
    result = _make_success(42, 0.7, stage="phase4", rpe=0.7)
    result.effective_detector_settings = {"minMarkers": 2}
    result.optimised_keys = ["minMarkers"]
    result.optimised_bounds = {"minMarkers": {"lower": 0, "upper": 4}}
    path = write_trial_metadata(
        tmp_path,
        result,
        study_id="study_test",
        f_loc="/tmp/data",
        target_settings={"target_type": "ChArUco", "num_squares_x": 5, "num_squares_y": 5, "square_size": 30.0},
        calibration_controls={"target_rpe": 1.0, "outliers": "n", "max_nfev_phase3": 100, "max_nfev_phase4": 100, "retain_successes": 20},
    )
    assert path.exists()
    text = path.read_text()
    assert '"study_test"' in text
    assert '"success_stage": "phase4"' in text
    assert result.saved_metadata_path == str(path)
    assert trial_subdir_name(result).startswith("trial_000042_phase4")


def test_write_study_summary_lists_successes(tmp_path: Path):
    ret = SuccessRetention(max_successes=3)
    ret.consider(_make_success(1, 1.0))
    ret.consider(_make_success(2, 0.5))
    path = write_study_summary(
        tmp_path,
        study_id="study_a",
        started_at=1000.0,
        finished_at=1010.0,
        retention=ret,
        n_trials_completed=2,
        n_trials_requested=10,
        mode="full",
        sampler_name="TPESampler",
        seed=42,
    )
    assert path.exists()
    text = path.read_text()
    assert '"study_a"' in text
    assert '"duration_sec": 10.0' in text
    assert '"n_successes_retained": 2' in text


def test_default_output_dir_under_floc():
    result = default_output_dir("/tmp/data", "study_abc")
    assert str(result).endswith("optimisation_runs/study_abc")


def test_make_study_id_format():
    sid = make_study_id()
    assert sid.startswith("study_")
    assert len(sid) > len("study_")


# ---------------------------------------------------------------------------
# Run validation
# ---------------------------------------------------------------------------


def test_validate_run_settings_missing_path(tmp_path: Path):
    target = {"target_type": "ChArUco", "num_squares_x": 5, "num_squares_y": 5, "square_size": 30.0}
    errors = validate_run_settings(
        f_loc=tmp_path / "does_not_exist",
        n_trials=10,
        target_rpe=1.0,
        max_nfev_phase3=100,
        max_nfev_phase4=100,
        retain_successes=5,
        target_settings=target,
    )
    assert any("does not exist" in e for e in errors)


def test_validate_run_settings_clean(tmp_path: Path):
    target = {"target_type": "ChArUco", "num_squares_x": 5, "num_squares_y": 5, "square_size": 30.0}
    errors = validate_run_settings(
        f_loc=tmp_path,
        n_trials=10,
        target_rpe=1.0,
        max_nfev_phase3=100,
        max_nfev_phase4=100,
        retain_successes=5,
        target_settings=target,
    )
    assert errors == []


def test_validate_run_settings_accepts_ccube(tmp_path: Path):
    target = {"target_type": "Ccube", "n_points": 6, "length": 40.0, "border_fraction": 0.1}
    errors = validate_run_settings(
        f_loc=tmp_path,
        n_trials=10,
        target_rpe=1.0,
        max_nfev_phase3=100,
        max_nfev_phase4=100,
        retain_successes=5,
        target_settings=target,
    )
    assert errors == []


def test_validate_run_settings_bad_target(tmp_path: Path):
    errors = validate_run_settings(
        f_loc=tmp_path,
        n_trials=10,
        target_rpe=1.0,
        max_nfev_phase3=100,
        max_nfev_phase4=100,
        retain_successes=5,
        target_settings={"target_type": "ChArUco", "num_squares_x": 0, "num_squares_y": 5, "square_size": 30.0},
    )
    assert any("num_squares_x" in e for e in errors)


# ---------------------------------------------------------------------------
# Effective settings construction
# ---------------------------------------------------------------------------


def test_build_effective_settings_uses_fixed_when_not_optimised():
    rows = [
        ParameterRowConfig(key="minMarkers", fixed=3, optimise=False),
        ParameterRowConfig(key="adaptiveThreshConstant", fixed=12.5, optimise=False),
    ]
    settings = build_effective_settings(rows)
    assert settings["minMarkers"] == 3
    assert settings["adaptiveThreshConstant"] == pytest.approx(12.5)


def test_build_effective_settings_uses_sampled_when_optimised():
    rows = [
        ParameterRowConfig(key="minMarkers", fixed=3, optimise=True, lower=0, upper=4),
    ]
    settings = build_effective_settings(rows, sampled={"minMarkers": 1})
    assert settings["minMarkers"] == 1


def test_build_effective_settings_clamps_to_metadata_bounds():
    rows = [ParameterRowConfig(key="minMarkers", fixed=2, optimise=True, lower=0, upper=4)]
    settings = build_effective_settings(rows, sampled={"minMarkers": 99})
    assert settings["minMarkers"] == 4


# ---------------------------------------------------------------------------
# run_trial / OptimisationStudy with injected fakes
# ---------------------------------------------------------------------------


def _fake_detection_fn(point_count: int = 100):
    arr = np.array([
        [10, 10, 10],
        [10, 10, 10],
        [10, 10, 0],
        [0, 10, 10],
    ])
    if point_count != int(arr.sum()):
        # Scale to hit a specific point count for ratio tests.
        scale = point_count / max(1, int(arr.sum()))
        arr = (arr.astype(float) * scale).astype(int)

    def _fn(f_loc, options, target):
        return {
            "features_per_im_per_cam": arr,
            "detections": object(),
            "warnings": [],
        }

    return _fn


def _fake_phase3(rpe: float):
    def _fn(detection_payload, *, controls):
        return {"rpe": rpe, "camset": None}
    return _fn


class _FakeCamset:
    def __init__(self, label: str):
        self.label = label

    def save(self, path):
        Path(path).write_text(self.label)


def _fake_phase4(rpe: float):
    def _fn(detection_payload, phase3, *, controls):
        return {"rpe": rpe, "camset": None}
    return _fn


def _fake_phase3_camset(rpe: float, label: str = "phase3"):
    def _fn(detection_payload, *, controls):
        return {"rpe": rpe, "camset": _FakeCamset(label)}
    return _fn


def _fake_phase4_camset(rpe: float, label: str = "phase4"):
    def _fn(detection_payload, phase3, *, controls):
        return {"rpe": rpe, "camset": _FakeCamset(label)}
    return _fn


def _make_config(tmp_path: Path, mode: str = "full") -> RunConfig:
    return RunConfig(
        f_loc=tmp_path,
        mode=mode,
        n_trials=3,
        parameter_rows=[
            ParameterRowConfig(key="minMarkers", fixed=2, optimise=False),
        ],
        target=TargetSettings(num_squares_x=5, num_squares_y=5, square_size=30.0),
        controls=CalibrationControls(
            outliers="n",
            max_nfev_phase3=10,
            max_nfev_phase4=10,
            target_rpe=1.0,
            retain_successes=5,
        ),
    )


def test_run_trial_full_phase3_success(tmp_path: Path):
    config = _make_config(tmp_path)
    result, _payload = run_trial(
        0,
        config.parameter_rows,
        config=config,
        sampled=None,
        baseline_point_count=80,
        detection_fn=_fake_detection_fn(80),
        phase3_fn=_fake_phase3(0.4),
    )
    assert result.valid
    assert result.successful
    assert result.success_stage == "phase3"
    assert result.score == pytest.approx(0.4, abs=1e-6)


def test_run_trial_full_phase4_runs_when_phase3_above_threshold(tmp_path: Path):
    config = _make_config(tmp_path)
    result, _payload = run_trial(
        0,
        config.parameter_rows,
        config=config,
        sampled=None,
        baseline_point_count=80,
        detection_fn=_fake_detection_fn(80),
        phase3_fn=_fake_phase3(2.0),
        phase4_fn=_fake_phase4(0.7),
    )
    assert result.self_calibration_run is True
    assert result.success_stage == "phase4"
    assert result.successful


def test_run_trial_full_phase4_failure_marks_unsuccessful(tmp_path: Path):
    config = _make_config(tmp_path)
    result, _payload = run_trial(
        0,
        config.parameter_rows,
        config=config,
        sampled=None,
        baseline_point_count=80,
        detection_fn=_fake_detection_fn(80),
        phase3_fn=_fake_phase3(2.0),
        phase4_fn=_fake_phase4(3.0),
    )
    assert result.self_calibration_run is True
    assert result.success_stage is None
    assert not result.successful
    assert result.score < FAILURE_SCORE  # valid-but-unsuccessful keeps ranking info


def test_run_trial_full_payload_preserves_detection_and_phase_outputs(tmp_path: Path):
    config = _make_config(tmp_path)
    result, payload = run_trial(
        0,
        config.parameter_rows,
        config=config,
        sampled=None,
        baseline_point_count=80,
        detection_fn=_fake_detection_fn(80),
        phase3_fn=_fake_phase3_camset(2.0),
        phase4_fn=_fake_phase4_camset(0.7),
    )
    assert result.success_stage == "phase4"
    assert "detection" in payload
    assert payload["phase3"]["camset"].label == "phase3"
    assert payload["phase4"]["camset"].label == "phase4"


def test_run_trial_invalid_detection_marks_failure(tmp_path: Path):
    config = _make_config(tmp_path)
    # one camera only, two valid images => fails validity (n_cams<2)
    arr = np.array([[10], [10], [10]])

    def _det(f_loc, options, target):
        return {
            "features_per_im_per_cam": arr,
            "detections": None,
            "n_cameras_with_detections": 1,
            "n_valid_images": 3,
        }

    result, _payload = run_trial(
        0,
        config.parameter_rows,
        config=config,
        sampled=None,
        baseline_point_count=30,
        detection_fn=_det,
        phase3_fn=_fake_phase3(0.4),
    )
    assert not result.valid
    assert result.score == FAILURE_SCORE
    assert "2 cameras" in (result.failure_reason or "")


def test_run_trial_fast_mode_skips_calibration(tmp_path: Path):
    config = _make_config(tmp_path, mode="fast")
    called = {"phase3": False, "phase4": False}

    def _phase3(*a, **k):
        called["phase3"] = True
        return {"rpe": 0.5}

    result, _payload = run_trial(
        0,
        config.parameter_rows,
        config=config,
        sampled=None,
        baseline_point_count=80,
        detection_fn=_fake_detection_fn(80),
        phase3_fn=_phase3,
    )
    assert not called["phase3"]
    assert result.valid
    assert result.success_stage is None


def test_optimisation_study_runs_to_completion_with_stub_sampler(tmp_path: Path):
    config = _make_config(tmp_path)
    study = OptimisationStudy(
        config,
        sampler=lambda i, rows: {},
        detection_fn=_fake_detection_fn(80),
        phase3_fn=_fake_phase3(0.4),
    )
    ret = study.run()
    assert len(study.results) == config.n_trials
    assert len(ret) >= 1
    assert (study.config.resolved_output_dir(study.study_id) / "study_summary.json").exists()


def test_optimisation_study_success_metadata_contains_promotion_artifacts(tmp_path: Path):
    config = _make_config(tmp_path)
    study = OptimisationStudy(
        config,
        sampler=lambda i, rows: {},
        detection_fn=_fake_detection_fn(80),
        phase3_fn=_fake_phase3_camset(0.4),
    )
    ret = study.run()
    result = ret.ranked()[0]
    metadata = json.loads(Path(result.saved_metadata_path).read_text())
    artifacts = metadata["extra"]["artifacts"]
    assert Path(artifacts["detected_datapoints_pickle"]).exists()
    assert Path(artifacts["phase3_camset"]).exists()
    assert artifacts["phase4_camset"] is None


def test_optimisation_study_respects_cancel(tmp_path: Path):
    config = _make_config(tmp_path)
    from pyCamSet.optimisation.optimisation_worker import CancelToken
    token = CancelToken()
    config.n_trials = 10

    progress_calls: list[int] = []

    def _cb(p):
        progress_calls.append(p.trial_number)
        if p.trial_number >= 1:
            token.cancel()

    study = OptimisationStudy(
        config,
        sampler=lambda i, rows: {},
        detection_fn=_fake_detection_fn(80),
        phase3_fn=_fake_phase3(0.4),
        cancel_token=token,
        progress_cb=_cb,
    )
    study.run()
    assert len(study.results) < 10
    assert len(study.results) >= 2


def test_optimisation_study_validate_surfaces_errors(tmp_path: Path):
    config = _make_config(tmp_path)
    # Break validation: target_rpe must be > 0
    config.controls.target_rpe = 0.0
    study = OptimisationStudy(config, sampler=lambda i, rows: {})
    errors = study.validate()
    assert any("Target RPE" in e for e in errors)


def test_detection_options_grouping_round_trip():
    settings = build_effective_settings(
        [ParameterRowConfig(key="adaptiveThreshConstant", fixed=11.0, optimise=False)]
    )
    grouped = detection_options_from_settings(settings)
    assert grouped["DetectorParameters"]["adaptiveThreshConstant"] == pytest.approx(11.0)


def _write_retained_metadata(tmp_path: Path, stage: str) -> Path:
    trial_dir = tmp_path / f"trial_{stage}"
    trial_dir.mkdir()
    detection = trial_dir / "detected_datapoints.pickle"
    phase3 = trial_dir / "camset_phase3.json"
    phase4 = trial_dir / "camset_phase4.json"
    detection.write_bytes(b"detections")
    phase3.write_text("phase3")
    if stage == "phase4":
        phase4.write_text("phase4")
    metadata = {
        "identity": {"success_stage": stage, "trial_number": 3},
        "paths": {"f_loc": str(tmp_path), "trial_dir": str(trial_dir)},
        "target": {"target_type": "Ccube", "n_points": 6, "length": 40.0},
        "detector_settings": {"effective": {"minMarkers": 2}},
        "calibration_controls": {"outliers": "n", "max_nfev_phase3": 10, "max_nfev_phase4": 10},
        "metrics": {"phase3_rpe": 0.4, "phase4_rpe": 0.7 if stage == "phase4" else None},
        "extra": {
            "artifacts": {
                "detected_datapoints_pickle": str(detection),
                "phase3_camset": str(phase3),
                "phase4_camset": str(phase4) if stage == "phase4" else None,
            },
            "phase_sources": {"phase2_run_id": "p2", "phase2_initial_camset": "/tmp/p2.camset"},
        },
    }
    path = trial_dir / "metadata.json"
    path.write_text(json.dumps(metadata))
    return path


def test_promote_retained_phase3_writes_only_phases_1_and_3(tmp_path: Path):
    workspace = tmp_path / "workspace"
    mgr = _WorkspaceManager(workspace)
    promoted = promote_retained_trial(mgr, _write_retained_metadata(tmp_path, "phase3"))
    assert set(promoted) == {"phase1", "phase3"}
    assert (workspace / "phase1_runs" / promoted["phase1"] / "detected_datapoints.pickle").exists()
    assert (workspace / "phase3_runs" / promoted["phase3"] / "optimised_cameras.camset").exists()
    assert not any((workspace / "phase4_runs").iterdir())


def test_promote_retained_phase4_writes_phases_1_3_and_4(tmp_path: Path):
    workspace = tmp_path / "workspace"
    mgr = _WorkspaceManager(workspace)
    promoted = promote_retained_trial(mgr, _write_retained_metadata(tmp_path, "phase4"))
    assert set(promoted) == {"phase1", "phase3", "phase4"}
    assert (workspace / "phase4_runs" / promoted["phase4"] / "self_calibrated_cameras.camset").exists()


# ---------------------------------------------------------------------------
# Optuna adapter (skipped when not installed)
# ---------------------------------------------------------------------------


def test_optuna_adapter_import_does_not_raise():
    from pyCamSet.optimisation import optuna_adapter

    # The module always imports; OPTUNA_AVAILABLE may be True or False.
    assert hasattr(optuna_adapter, "OPTUNA_AVAILABLE")


def test_optuna_suggest_for_row_when_available():
    from pyCamSet.optimisation import optuna_adapter
    if not optuna_adapter.OPTUNA_AVAILABLE:
        pytest.skip("optuna not installed")
    import optuna

    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    row = ParameterRowConfig(key="minMarkers", fixed=2, optimise=True, lower=0, upper=4)
    value = optuna_adapter.suggest_for_row(trial, row)
    assert isinstance(value, int)
    assert 0 <= value <= 4
    study.tell(trial, 1.0)


def test_optuna_suggest_for_row_returns_none_when_not_optimised():
    from pyCamSet.optimisation import optuna_adapter
    if not optuna_adapter.OPTUNA_AVAILABLE:
        pytest.skip("optuna not installed")
    import optuna

    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    row = ParameterRowConfig(key="minMarkers", fixed=2, optimise=False)
    assert optuna_adapter.suggest_for_row(trial, row) is None
    study.tell(trial, 1.0)
