"""What the detector-parameter study does, pinned before it is changed.

``tuning/worker.py`` carries ``default_phase{2,3,4}_fn`` -- a fourth copy of
the phase logic, calling the bundle handlers directly rather than the phase
runners next door.  Replacing those with calls into ``pyCamSet.workflow``
means changing what a trial runs, and almost none of these 2,938 lines had a
test to notice.  These are that net.

They are deliberately about the *seam* rather than about a solve.  A trial
takes its four stages as injected callables, so the driver can be exercised
with fakes: what it does with each outcome, what it counts, what it keeps and
what it writes down.  Whatever replaces ``default_phase3_fn`` has to satisfy
the same contract, and that contract is what is asserted here.

One test at the end does run the real thing over the image corpus, because a
contract nothing has ever satisfied is not worth much.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pyCamSet.workflow.tuning.promotion import promote_retained_trial
from pyCamSet.workflow.tuning.study import (
    FAILURE_SCORE,
    SuccessRetention,
    TrialResult,
    make_trial_gating_settings,
)
from pyCamSet.workflow.tuning.worker import (
    CalibrationControls,
    CancelToken,
    OptimisationStudy,
    ParameterRowConfig,
    RunConfig,
    run_trial,
)
from pyCamSet.workflow.workspace import WorkspaceManager, workspace_path_for

# A detection that passes every gate: four cameras, six images, every one of
# them seen by every camera.
GOOD_FEATURES = np.full((6, 4), 40.0)

#: A real detector parameter.  The tuning table keys these bare, with the
#: OpenCV sub-dict in a ``group`` field, where the workflow's own option
#: table keys the same parameters ``DetectorParameters.minMarker...``.  The
#: two tables are the duplication step 4 of the restructuring merges.
PARAMETER = "minMarkerPerimeterRate"


def detection(features=GOOD_FEATURES, **extra) -> dict:
    """A detection payload of the shape ``detection_fn`` must return."""
    payload = {
        "features_per_im_per_cam": np.asarray(features),
        "detections": object(),
        "target": object(),
        "cam_res": [(100, 100)] * np.asarray(features).shape[1],
    }
    payload.update(extra)
    return payload


@pytest.fixture
def dataset(tmp_path):
    """A folder a study will accept: two cameras, with images in them.

    ``validate_run_settings`` insists on this before a study starts, so the
    driver cannot be exercised against a bare temporary directory.  The
    images are never read here -- every stage is faked -- but they have to
    be there and they have to be real files.
    """
    import cv2

    root = tmp_path / "images"
    for camera in ("cam0", "cam1"):
        folder = root / camera
        folder.mkdir(parents=True)
        for index in range(2):
            cv2.imwrite(str(folder / f"im{index}.png"),
                        np.zeros((16, 16, 3), dtype=np.uint8))
    return root


#: A small ChArUco, which is all the faked stages need of a target.
CHARUCO_TARGET = {"type": "ChArUco", "num_squares_x": 5, "num_squares_y": 5,
                  "square_size": 30.0}


def config(tmp_path, **overrides) -> RunConfig:
    settings = dict(
        f_loc=str(tmp_path),
        mode="full",
        n_trials=1,
        parameter_rows=[ParameterRowConfig(key=PARAMETER, fixed=0.03)],
        target_spec=dict(CHARUCO_TARGET),
        controls=CalibrationControls(target_rpe=1.0),
        trial_gating=make_trial_gating_settings("Moderate"),
        output_dir=str(tmp_path / "study"),
    )
    settings.update(overrides)
    return RunConfig(**settings)


def trial(tmp_path, *, detection_fn=None, phase2=None, phase3=None, phase4=None,
          calls=None, baseline=int(GOOD_FEATURES.sum()), **config_overrides):
    """Run one trial with each stage faked, recording which stages ran."""
    calls = calls if calls is not None else []

    def stage(name, outcome):
        def run(*args, **kwargs):
            calls.append(name)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return run

    return run_trial(
        0,
        config(tmp_path, **config_overrides).parameter_rows,
        config=config(tmp_path, **config_overrides),
        baseline_point_count=baseline,
        detection_fn=detection_fn or stage("detection", detection()),
        phase2_fn=stage("phase2", phase2 if phase2 is not None else {"camset": None}),
        phase3_fn=stage("phase3", phase3 if phase3 is not None else {"rpe": 0.5}),
        phase4_fn=stage("phase4", phase4 if phase4 is not None else {"rpe": 0.4}),
    )[0]


# ---------------------------------------------------------------------------
# What a stage has to return, and what the trial does with it
# ---------------------------------------------------------------------------


def test_a_phase_reports_its_error_as_rpe(tmp_path):
    """The whole contract between the driver and a phase function.

    Whatever replaces ``default_phase3_fn`` must return a mapping with an
    ``rpe`` the driver can compare against the target, and may carry a
    ``camset`` for the metadata writer to save.  Nothing else is read.
    """
    calls: list[str] = []
    result = trial(tmp_path, phase3={"rpe": 0.5}, calls=calls)

    assert result.phase3_rpe == 0.5
    assert result.successful is True
    assert result.success_stage == "phase3"
    # Phase 3 met the target, so phase 4 is not run at all.
    assert calls == ["detection", "phase2", "phase3"]


def test_a_phase_3_that_misses_the_target_hands_on_to_phase_4(tmp_path):
    calls: list[str] = []
    result = trial(tmp_path, phase3={"rpe": 4.0}, phase4={"rpe": 0.4}, calls=calls)

    assert calls == ["detection", "phase2", "phase3", "phase4"]
    assert result.self_calibration_run is True
    assert result.success_stage == "phase4"
    assert result.successful is True
    assert (result.phase3_rpe, result.phase4_rpe) == (4.0, 0.4)


def test_a_trial_that_never_reaches_the_target_is_scored_but_not_successful(tmp_path):
    """Still measured, so the sampler can tell a near miss from a disaster."""
    result = trial(tmp_path, phase3={"rpe": 9.0}, phase4={"rpe": 7.0})

    assert result.successful is False
    assert result.success_stage is None
    assert result.failure_stage == "phase4"
    assert "exceeded the target" in result.failure_reason
    assert result.score != FAILURE_SCORE
    assert np.isfinite(result.score)


@pytest.mark.parametrize("rpe", [float("nan"), float("inf"), None])
def test_an_unusable_rpe_invalidates_the_trial(tmp_path, rpe):
    """A solver that diverged must not be ranked as if it had converged."""
    result = trial(tmp_path, phase3={"rpe": rpe})

    assert result.valid is False
    assert result.score == FAILURE_SCORE
    assert result.failure_stage == "phase3"


@pytest.mark.parametrize(
    ("stage", "kwargs"),
    [("detection", {"detection_fn": None}),
     ("phase2", {"phase2": RuntimeError("no cameras")}),
     ("phase3", {"phase3": RuntimeError("singular jacobian")}),
     ("phase4", {"phase4": RuntimeError("gauge collapsed")})],
)
def test_a_stage_that_raises_ends_the_trial_not_the_study(tmp_path, stage, kwargs):
    """One bad parameter set must not take the whole search down with it."""
    if stage == "detection":
        def boom(*_args, **_kwargs):
            raise RuntimeError("detector rejected every marker")
        kwargs = {"detection_fn": boom}
    if stage == "phase4":
        kwargs["phase3"] = {"rpe": 9.0}   # so phase 4 is reached at all

    result = trial(tmp_path, **kwargs)

    assert result.failure_stage == stage
    assert result.score == FAILURE_SCORE
    assert result.failure_reason


def test_without_a_baseline_every_trial_looks_like_it_found_nothing(tmp_path):
    """``point_ratio`` is measured against a baseline detection pass, so with
    no baseline it reads as zero and the gate rejects everything.

    :class:`OptimisationStudy` computes the baseline before its first trial.
    Anything else driving :func:`run_trial` has to do the same.
    """
    result = trial(tmp_path, baseline=None)

    assert result.valid is False
    assert result.failure_stage == "gating"
    assert "point ratio" in result.failure_reason


@pytest.mark.parametrize(
    ("features", "stage"),
    [(np.zeros((6, 4)), "detection"),
     # Two cameras saw something, the other two saw nothing: detected, but
     # not enough of the rig to calibrate.
     (np.pad(np.full((6, 2), 40.0), ((0, 0), (0, 2))), "gating")],
)
def test_a_detection_that_cannot_be_calibrated_never_reaches_a_phase(
        tmp_path, features, stage):
    """Gating exists to not spend a bundle adjustment on a hopeless trial.

    Finding nothing at all and finding too little are separate verdicts, so
    the progress counters can tell a broken detector from a hard dataset.
    """
    calls: list[str] = []
    payload = detection(features=features)
    result = trial(
        tmp_path,
        detection_fn=lambda *a, **k: (calls.append("detection"), payload)[1],
        calls=calls,
    )

    assert result.valid is False
    assert result.failure_stage == stage
    assert result.score == FAILURE_SCORE
    assert calls == ["detection"]


def test_fast_mode_stops_after_detection(tmp_path):
    """Fast mode scores coverage alone, so it never pays for a solve."""
    calls: list[str] = []
    result = trial(tmp_path, mode="fast", calls=calls)

    assert calls == ["detection"]
    assert result.valid is True
    assert result.successful is False
    assert np.isfinite(result.score) and result.score != FAILURE_SCORE


def test_the_settings_a_trial_ran_with_are_recorded_on_it(tmp_path):
    """A retained trial is only useful if it says what produced it."""
    result = trial(tmp_path)

    assert result.effective_detector_settings[PARAMETER] == 0.03
    assert result.fixed_detector_settings[PARAMETER] == 0.03


# ---------------------------------------------------------------------------
# The study driver over many trials
# ---------------------------------------------------------------------------


def study(tmp_path, dataset, *, rpes, n_trials=None, **kwargs):
    """A study whose phase 3 returns *rpes* in turn."""
    sequence = iter(rpes)
    return OptimisationStudy(
        config(tmp_path, f_loc=str(dataset), n_trials=n_trials or len(rpes)),
        sampler=lambda _i, _rows: {},
        detection_fn=lambda *_a, **_k: detection(),
        phase2_fn=lambda *_a, **_k: {"camset": None},
        phase3_fn=lambda *_a, **_k: {"rpe": next(sequence)},
        phase4_fn=lambda *_a, **_k: {"rpe": 9.0},
        write_metadata=False,
        **kwargs,
    )


def test_every_trial_exit_lands_in_exactly_one_counter(tmp_path, dataset):
    """The counters are what someone watches a long study through."""
    driver = study(tmp_path, dataset, rpes=[0.5, 0.5, 9.0])
    driver.run()

    counts = driver.outcome_counts.as_dict()
    assert counts["succeeded_phase3"] == 2
    # The third missed at phase 3, went on to phase 4, and missed there too.
    assert counts["failed_phase4"] == 1
    assert sum(counts.values()) == 3


def test_retention_keeps_the_best_trials_not_the_first(tmp_path, dataset):
    """The cap is a budget for good results, not an order of arrival."""
    driver = study(tmp_path, dataset, rpes=[0.9, 0.8, 0.1, 0.7])
    driver.config.controls.retain_successes = 2
    driver.retention = SuccessRetention(2)

    retention = driver.run()

    kept = sorted(r.phase3_rpe for r in retention.ranked())
    assert kept == [0.1, 0.7]
    assert retention.best().phase3_rpe == 0.1


def test_a_cancelled_study_stops_where_it_was(tmp_path, dataset):
    token = CancelToken()
    driver = study(tmp_path, dataset, rpes=[0.5] * 10, cancel_token=token)
    driver.progress_cb = lambda progress: token.cancel() if progress.trial_number == 2 else None

    driver.run()

    assert len(driver.results) == 3


def test_progress_reports_the_running_totals(tmp_path, dataset):
    seen = []
    driver = study(tmp_path, dataset, rpes=[0.5, 9.0, 0.2], progress_cb=seen.append)
    driver.run()

    assert [p.trial_number for p in seen] == [0, 1, 2]
    assert [p.n_successes for p in seen] == [1, 1, 2]
    assert seen[-1].best_phase3_rpe == 0.2
    assert seen[-1].total_trials == 3


def test_a_study_refuses_a_configuration_it_cannot_run(tmp_path, dataset):
    driver = study(tmp_path, dataset, rpes=[0.5])
    driver.config.n_trials = 0

    with pytest.raises(ValueError, match="Run configuration invalid"):
        driver.run()


def test_a_gating_profile_nobody_has_heard_of_is_caught_before_the_study_runs(tmp_path):
    """``make_trial_gating_settings`` accepts any name and falls back to
    Moderate's thresholds, keeping the name it was given.  The settings then
    look usable and fail only at :meth:`OptimisationStudy.run`.

    Two profile vocabularies exist and read alike: the gating profiles here,
    and the presets a detector offers for its own bounds.  Passing one where
    the other belongs is the mistake this catches.
    """
    from pyCamSet.calibration_targets.charuco_detection import ARUCO_OPENCV_DETECTOR

    settings = make_trial_gating_settings("Balanced")
    assert settings.profile_name == "Balanced"
    assert settings.min_point_ratio == make_trial_gating_settings("Moderate").min_point_ratio
    assert settings.validate()

    # "Balanced" is a detector preset, not a gating profile.
    assert "Balanced" in ARUCO_OPENCV_DETECTOR.profiles()


def test_a_study_writes_a_summary_and_a_record_per_success(tmp_path, dataset):
    """What the tab lists afterwards, and what promotion later reads."""
    driver = study(tmp_path, dataset, rpes=[0.5, 9.0])
    driver.write_metadata = True
    driver.run()

    output_dir = driver.config.resolved_output_dir(driver.study_id)
    summary = json.loads((output_dir / "study_summary.json").read_text())
    assert summary["n_trials_completed"] == 2

    # One success, so one trial record -- failures are not written.
    records = sorted(output_dir.glob("*/metadata.json"))
    assert len(records) == 1
    written = json.loads(records[0].read_text())
    assert written["identity"]["success_stage"] == "phase3"


# ---------------------------------------------------------------------------
# Promoting a retained trial into ordinary workspace runs
# ---------------------------------------------------------------------------


def test_a_retained_trial_becomes_a_chain_of_workspace_runs(tmp_path):
    """The point of a study: its winner becomes a run like any other."""
    workspace = WorkspaceManager(workspace_path_for(tmp_path))
    artefacts = tmp_path / "trial"
    artefacts.mkdir()
    for name in ("detected_datapoints.pickle", "camset_phase2.json",
                 "camset_phase3.json"):
        (artefacts / name).write_bytes(b"stand-in artefact")

    record = artefacts / "metadata.json"
    record.write_text(json.dumps({
        "identity": {"success_stage": "phase3"},
        "paths": {"f_loc": str(tmp_path)},
        "target": {"type": "ChArUco"},
        "detector_settings": {"DetectorParameters.minMarkerPerimeterRate": 0.03},
        "extra": {
            "artifacts": {
                "detected_datapoints_pickle": str(artefacts / "detected_datapoints.pickle"),
                "phase2_initial_camset": str(artefacts / "camset_phase2.json"),
                "phase3_camset": str(artefacts / "camset_phase3.json"),
            },
            "phase_sources": {},
        },
    }))

    promoted = promote_retained_trial(workspace, record)

    assert set(promoted) == {"phase1", "phase2", "phase3"}
    phase3 = workspace.find_run("phase3", promoted["phase3"])
    assert phase3["inputs"]["phase2_run_id"] == promoted["phase2"]
    assert workspace.find_run("phase2", promoted["phase2"])["inputs"][
        "phase1_run_id"] == promoted["phase1"]

    # The chain resolves the same way any other run's does.
    chain = workspace.build_predecessor_chain(phase3)
    assert [run["run_id"] for run in chain] == [promoted["phase1"], promoted["phase2"]]


def test_a_trial_that_did_not_succeed_cannot_be_promoted(tmp_path):
    workspace = WorkspaceManager(workspace_path_for(tmp_path))
    record = tmp_path / "metadata.json"
    record.write_text(json.dumps({"identity": {"success_stage": None}}))

    with pytest.raises(RuntimeError, match="Only successful"):
        promote_retained_trial(workspace, record)


@pytest.mark.slow
def test_the_real_stages_satisfy_the_contract_the_fakes_assume(session_data_dir):
    """Detection and phase 2, for real, against the shape the driver reads.

    Everything above fakes the four stages.  This is the one that says the
    fakes are not fiction -- and it is the baseline the replacement of
    ``default_phase{2,3}_fn`` has to reproduce.
    """
    from pyCamSet.workflow.tuning.worker import (
        default_detection_fn,
        default_phase2_fn,
        fixed_settings_only,
    )
    from pyCamSet.calibration_targets.charuco_detection import ARUCO_OPENCV_DETECTOR

    # a_dict=3 is DICT_4X4_1000, which a 20x20 board needs: DICT_4X4_50 is
    # exhausted by it, and OpenCV asserts rather than saying so.
    target = {"type": "ChArUco", "num_squares_x": 20, "num_squares_y": 20,
              "square_size": 4.0, "a_dict": 3, "legacy": True}
    rows = [ParameterRowConfig(key=PARAMETER, fixed=0.03)]
    options = fixed_settings_only(rows, ARUCO_OPENCV_DETECTOR)

    payload = default_detection_fn(
        session_data_dir / "calibration_charuco", options, target)

    # The fields run_trial reads out of a detection payload.
    features = np.asarray(payload["features_per_im_per_cam"])
    assert features.ndim == 2 and features.sum() > 0
    assert payload["detections"] is not None
    assert payload["target"] is not None
    assert payload["cam_res"] is not None

    phase2 = default_phase2_fn(payload, controls=CalibrationControls())
    assert phase2["camset"] is not None


# ---------------------------------------------------------------------------
# A study sweeps whatever its target says its detection takes
# ---------------------------------------------------------------------------
#
# The worker imported ChArUco's parameters by name, so a study over any
# other target swept ChArUco's settings and passed them to a detector that
# had never heard of them.


def test_the_detector_a_study_sweeps_comes_from_its_target(tmp_path):
    charuco = config(tmp_path)
    puzzleboard = config(tmp_path, target_spec={
        "type": "PuzzleBoard", "num_squares_x": 20, "num_squares_y": 20})

    assert charuco.detector().name == "aruco1"
    assert puzzleboard.detector().name == "puzzle_board"
    assert "min_width" in puzzleboard.detector()
    assert "min_width" not in charuco.detector()


def test_a_trial_sweeps_the_settings_that_targets_detector_takes(tmp_path):
    """``min_width`` is PuzzleBoard's to sweep and nobody else's."""
    from pyCamSet.workflow.tuning.worker import build_effective_settings

    detector = config(tmp_path, target_spec={
        "type": "PuzzleBoard", "num_squares_x": 20,
        "num_squares_y": 20}).detector()
    rows = [ParameterRowConfig(key="min_width", fixed=4, optimise=True,
                               lower=2, upper=12)]

    assert build_effective_settings(rows, detector) == {"min_width": 4}
    assert build_effective_settings(
        rows, detector, sampled={"min_width": 9}) == {"min_width": 9}
    assert detector.validate_rows(rows) == []


def test_a_row_the_targets_detector_does_not_take_is_refused(tmp_path):
    """Sweeping OpenCV's threshold window over a PuzzleBoard is a mistake
    that used to reach the detector."""
    detector = config(tmp_path, target_spec={
        "type": "PuzzleBoard", "num_squares_x": 20,
        "num_squares_y": 20}).detector()
    rows = [ParameterRowConfig(key="adaptiveThreshWinSizeMin", fixed=3)]

    errors = detector.validate_rows(rows)
    assert errors and "Unknown parameter" in errors[0]


def test_a_target_that_cannot_be_built_stops_the_study(tmp_path, dataset):
    """Validation builds the target, which is what a trial does.

    It used to list the fields two named targets need, which said nothing
    about a third and drifted from what their constructors accept.
    """
    driver = OptimisationStudy(
        config(tmp_path, f_loc=str(dataset),
               target_spec={"type": "ChArUco", "num_squares_x": 1,
                            "num_squares_y": 1, "square_size": 30.0}),
        sampler=lambda i, rows: {})

    assert any("Target cannot be built" in e for e in driver.validate())
