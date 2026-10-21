"""F2 integration cells: viewer/playback/export/preference behaviour.

Viewer, playback, export and preference slice of the education mode.  Cells that other
test files already own are deliberately not duplicated here:

* the E2 copy matrix and the render-level playback cells live in
  ``tests/test_education_phase4.py`` (E2-owned, integrated at this HEAD);
* the trace-reader contracts live in ``tests/test_education_playback.py``
  (D1t-owned);
* the preference sidecar unit contract lives in
  ``tests/test_education_preferences.py`` (F1a-owned);
* the diagnostics/phase-contract and whole-run surfaces live in the two files
  ``t_dc59e4a6`` owns.

This file adds what no sibling owns: the producer end of recorded-step
playback through the real ``phase4.run`` route, the Export sentence bound to
the real accepted-only resolver under E1's recorded choice (b), the
education sidecar inside the real per-user config directory beside the strict
``preferences.json``, and the check that testing never mutates the checkout's
tracked sources.

Conventions (plan §4): Qt-touching tests carry the ``gui`` marker and call
``importorskip("PySide6")`` in the function, so collection stays green in the
PySide6-free environment; the Qt-free cells here stay lean-collectible and
lean-runnable.  Windows local only; Linux/macOS are not claimed.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import shutil

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "tests" / "test_data" / "calibration_charuco"

#: Tracked sources whose bytes must survive a producing test run bit for bit.
PROVENANCE_FILES = (
    "pyCamSet/workflow/phase4.py",
    "pyCamSet/workflow/education_trace.py",
    "pyCamSet/workflow/education_manifest.py",
    "pyCamSet/workflow/education_derive.py",
    "pyCamSet/workflow/education_playback.py",
    "pyCamSet/gui/education_phase4.py",
)

_SENTENCE = ("The Export tab still offers this run; "
             "check which Phase 3 camset it started from first.")


def _recorded_hashes() -> dict[str, str]:
    """SHA-256 of the tracked provenance files, read from disk."""
    hashes = {}
    for relative in PROVENANCE_FILES:
        path = REPO_ROOT / relative
        assert path.is_file(), relative
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _tracked_status() -> str:
    """``git status --short`` output for the clone."""
    git = subprocess.run(
        ["git", "status", "--short"], cwd=REPO_ROOT,
        capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert git.returncode == 0, git.stderr
    return git.stdout


def _stage_phases_2_and_3(workspace_root: Path):
    """Run the real Phase 2/3 stages into ``workspace_root``; return the manager.

    Every run directory and artifact lives under ``workspace_root``; the
    repo tree is only read (the image corpus).
    """
    from pyCamSet import ChArUco
    from pyCamSet.calibration.camera_calibrator import (
        detect_datapoints_in_imfile, run_initial_calibration)
    from pyCamSet.workflow import phase3
    from pyCamSet.workflow.workspace import WorkspaceManager

    target = ChArUco(num_squares_x=20, num_squares_y=20, square_size=4.0,
                     legacy=False)
    detections, resolutions = detect_datapoints_in_imfile(
        f_loc=CORPUS, caching=False, calibration_target=target, threads=1)
    phase2_cams = run_initial_calibration(
        detections, target, resolutions, save=False)
    phase2_cams.set_resolutions_from_file(floc=CORPUS)

    ws_path = workspace_root / ".pycamset_workspace"
    manager = WorkspaceManager(ws_path)
    p2_dir = ws_path / "phase2_runs" / "p2"
    p2_dir.mkdir(parents=True, exist_ok=True)
    p2_camset = p2_dir / "initial_cameras.camset"
    phase2_cams.save(str(p2_camset))
    manager.save_run("phase2", "p2", {
        "run_id": "p2", "status": "complete",
        "artifacts": {"initial_camset": str(p2_camset)}})

    _handler, _result, phase3_camset, _stats = phase3.solve(
        phase2_cams, detections, target, fixed_params={},
        options={"outliers": "n", "fixed_pose": 0}, threads=1)
    p3_dir = ws_path / "phase3_runs" / "p3"
    p3_dir.mkdir(parents=True, exist_ok=True)
    p3_camset_path = p3_dir / "optimised_cameras.camset"
    phase3_camset.save(str(p3_camset_path))
    manager.save_run("phase3", "p3", {
        "run_id": "p3", "status": "complete",
        "params": {"problem_options": {"fixed_pose": 0}},
        "artifacts": {"optimised_camset": str(p3_camset_path)}})
    return manager, p3_camset_path


def _loaded_run_dir(loaded: dict, ws_path: Path) -> Path:
    """The run's directory, from the record or its workspace location."""
    return Path(loaded.get("_run_dir") or ws_path / "phase4_runs"
                / str(loaded["run_id"]))


# --------------------------------------------------------------------------
# No source mutation.
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.data
@pytest.mark.skipif(
    shutil.which("git") is None or not (Path(__file__).resolve().parents[1] / ".git").exists(),
    reason="needs a git checkout to compare the tracked sources")
def test_a_test_run_never_mutates_the_tracked_sources(tmp_path):
    """Producer output stays in the workspace; tracked bytes survive bit for bit.

    This is the "no source mutation" acceptance cell, made concrete: a full
    real Phase 2 -> 3 -> 4 production (recording on) writes only into the
    workspace's own run directories under ``tmp_path``, and the tracked
    sources of the tree the clone ships are unchanged afterwards.  A dirty
    start (this worktree carries the card's own uncommitted test files) is
    legitimate; what must hold is that the producing run adds nothing new to
    the tracked set.
    """
    hashes_before = _recorded_hashes()
    tracked_before = _tracked_status()

    from pyCamSet.workflow import phase4

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    manager, phase3_camset_path = _stage_phases_2_and_3(workspace_root)

    params = {
        "f_loc": str(CORPUS),
        "problem_options": {"outliers": "n", "fixed_pose": 0,
                            "max_nfev": 100},
        "record_education": True,
    }
    record = phase4.run(params, manager, lambda _line: None,
                        phase3_run=manager.find_run("phase3", "p3"),
                        phase3_camset=phase3_camset_path)
    assert record["status"] == "complete", record.get("error")
    assert not record.get("error")

    # Everything produced lives under the temp workspace root, outside the
    # checkout.
    produced = [path for path in workspace_root.rglob("*") if path.is_file()]
    assert produced, "the run produced nothing"

    # The tracked sources are byte-identical afterwards, and the producing
    # run added nothing to the tracked set (its own uncommitted test files
    # are legitimate; a new modification to a tracked file is not).
    assert _recorded_hashes() == hashes_before
    assert _tracked_status() == tracked_before

    workspace_path = workspace_root.resolve()
    repo_path = REPO_ROOT.resolve()
    assert not workspace_path.is_relative_to(repo_path)
    assert not repo_path.is_relative_to(workspace_path)


# --------------------------------------------------------------------------
# Education preferences inside the real per-user config directory.
# --------------------------------------------------------------------------


import pytest

@pytest.mark.gui
def test_education_sidecar_beside_preferences_absent_file_means_off(
        isolated_user_config, monkeypatch):
    """Absent sidecar -> defaults; the strict file is not seen or widened.

    F1a's unit tests pin the sidecar contract on explicit paths; this cell
    pins the wiring: the sidecar resolves beside the strict
    ``preferences.json`` in the real per-user config directory (the suite's
    isolated one), an absent file degrades to the documented defaults without
    being created, a version-1 strict document written by an old build
    validates unchanged, and the education sidecar's own write leaves the
    strict file byte-identical.
    """
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from pyCamSet.gui import education_preferences, preferences

    QApplication.instance() or QApplication([])
    config = Path(isolated_user_config)
    config.mkdir(parents=True, exist_ok=True)

    assert Path(preferences.config_directory()) == config
    assert education_preferences.preferences_path() == config / "education_preferences.json"
    strict_path = config / "preferences.json"

    # Absent files: the documented education defaults, and reading the
    # absent sidecar creates no file.
    assert not strict_path.exists()
    assert not education_preferences.preferences_path().exists()
    settings = education_preferences.load_preferences()
    assert settings == {"shortcuts_enabled": True, "reduce_motion": False}

    # An old build's preferences.json (version 1, exactly its schema) is a
    # real, untouched product of a build that has never heard of the
    # education sidecar; loading it must validate and change nothing on disk.
    initial = {
        "schema": "pycamset.gui-preferences", "version": 1,
        "preferences": {"info_enabled": True, "terminal_visible": True,
                        "export_presets": {}},
    }
    original_bytes = json.dumps(initial, indent=2).encode("utf-8")
    strict_path.write_bytes(original_bytes)
    loaded = preferences.Preferences()
    assert not loaded.load_error
    assert loaded.values["info_enabled"] is True
    assert strict_path.read_bytes() == original_bytes
    document = json.loads(strict_path.read_text(encoding="utf-8"))
    assert document["version"] == 1
    assert not any("education" in key for key in document["preferences"])

    # The education sidecar, written beside it, leaves the strict file
    # byte-identical: two contracts, one directory, no coupling.
    education_preferences.save_preferences({"reduce_motion": True})
    assert education_preferences.preferences_path().exists()
    assert strict_path.read_bytes() == original_bytes
    assert json.loads(strict_path.read_text(encoding="utf-8")) == initial
    assert json.loads(
        education_preferences.preferences_path().read_text(encoding="utf-8")
    )["preferences"]["reduce_motion"] is True


# --------------------------------------------------------------------------
# Export sentence <-> real accepted-only resolver, under E1's choice (b).
# --------------------------------------------------------------------------


def _phase4_run_record(tmp_path, status: str):
    """Build a real workspace record whose Phase 4 run has the given status."""
    from pyCamSet.workflow.workspace import WorkspaceManager

    ws_path = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(ws_path)
    run_dir = ws_path / "phase4_runs" / "p4"
    run_dir.mkdir(parents=True, exist_ok=True)
    if status == "complete":
        camset = run_dir / "self_calibrated_cameras.camset"
        camset.write_bytes(b"")
        artifacts = {"self_calibrated_camset": str(camset)}
    else:
        artifacts = {}
    record = {
        "run_id": "p4", "status": status, "phase": "phase4",
        "artifacts": artifacts,
    }
    manager.save_run("phase4", "p4", record)
    return ws_path, manager.find_run("phase4", "p4")


def _rendered_copy(tmp_path, monkeypatch, run, verification: str) -> str:
    """The dialog copy for a record shaped exactly like the resolver's input.

    The record flows through the real ``Phase4View`` with the derivation
    stubbed to the same ``DerivationUnavailable`` the real view produces when
    a derivation fails: the copy paths under test are disposition-driven,
    exactly as for a real unverified-input workspace.
    """
    pytest.importorskip("PySide6")
    from types import SimpleNamespace

    from PySide6.QtWidgets import QApplication

    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(
        education_phase4, "read_trace",
        lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(
        education_phase4, "derive_phase4",
        lambda *_args, **_kwargs: DerivationUnavailable("stub derivation"))

    state = SimpleNamespace(
        path=None, origin_run_id=None,
        verification=SimpleNamespace(value=verification),
        reason="differs" if verification == "differs" else "cannot be traced",
        confirmed=verification == "verified",
        source_check="verified" if verification == "verified"
        else "not checkable",
        phase3_before_px=None, stored_mean_px=None, start_mismatch=None,
        fixed_pose_matches=None, effective_fixed_parameter_differences=(),
        rig_comparison_withheld=False, photo_names=(),
        phase3_fixed_pose_ordinal1=None, phase4_fixed_pose_ordinal1=None,
        loss="linear", f_scale=1.0,
    )
    context = SimpleNamespace(
        run_id=str(run.get("run_id")), phase="phase4",
        run_dir=Path(run.get("_run_dir") or tmp_path),
        status=str(run.get("status")),
        metadata=dict(run),
        phase3_state_source=lambda: state,
        artifact=lambda phase: None,
    )
    view = education_phase4.Phase4View(context)
    try:
        return view.stage_text(0)
    finally:
        view.close()


@pytest.mark.gui
@pytest.mark.parametrize("status,verification,export_offered", [
    # E1's recorded posture under choice (b): Export stays status-gated;
    # the dialog's sentence covers every non-"verified" outcome on a
    # complete run, and failed runs are withheld by the copy.
    ("complete", "verified", False),
    ("complete", "differs", True),
    ("complete", "could not be checked", True),
    ("failed", "differs", False),
])
def test_export_sentence_matches_the_real_accepted_only_resolver(
        tmp_path, monkeypatch, status, verification, export_offered):
    """E2's Export sentence and the gate's resolver, together, for one run.

    E1 recorded Export choice (b): the Export tab is left unchanged and the
    presentation carries the sentence.  This cell holds both ends of that
    recorded contract against the integrated code for the same saved record:
    when the copy offers the run, Export's accepted-only resolver must offer
    it too; when the copy withholds the sentence (failed runs), the resolver
    must refuse it.  (Under choice (a), F1b's implemented gate would refuse
    the complete unverified run; if the recorded choice is ever revised, this
    cell fails and names the divergence.)
    """
    _ws_path, run = _phase4_run_record(tmp_path, status)
    copy_text = _rendered_copy(tmp_path, monkeypatch, run, verification)

    from pyCamSet.gui.assess_calibration import resolve_run_camset_artifact
    resolver_offered = resolve_run_camset_artifact(run, accepted_only=True)

    # The copy side of the pairing: the sentence tracks the verification
    # outcome exactly as E1's table requires.
    assert (_SENTENCE in copy_text) is export_offered
    # The resolver side of the pairing, held for the same record.  Under
    # choice (b) the resolver is status-gated, so a complete run (verified
    # or not) resolves and a non-complete run does not; the pairing asserts
    # that, and that an offered-by-copy run is never refused by Export.
    if status == "complete":
        assert resolver_offered is not None
        if not export_offered:
            assert verification == "verified"
    else:
        assert resolver_offered is None
        assert not export_offered


def test_resolver_refuses_every_noncomplete_phase4_status():
    """The accepted-only gate refuses each status the copy withholds.

    Qt-free: ``assess_calibration`` imports no Qt at module level, so this
    cell also runs in the PySide6-free environment.
    """
    from pyCamSet.gui.assess_calibration import resolve_run_camset_artifact

    for status in ("failed", "incomplete", "unknown", ""):
        run = {
            "run_id": "p4", "phase": "phase4", "status": status,
            "artifacts": {"self_calibrated_camset": "C:/nowhere/run.camset"},
        }
        assert resolve_run_camset_artifact(
            run, accepted_only=True) is None, status


# --------------------------------------------------------------------------
# Producer-route playback, opt-in vs default (end to end).
# --------------------------------------------------------------------------


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.data
def test_recorded_playback_end_to_end_optin_run(tmp_path):
    """Opt-in ticked on a fresh run: trace, endpoints match, nothing plays.

    F2's playback cell, run through the producer's real route — ``phase4.run``
    with the recording parameter on — then a real ``RunContext`` projecting
    the saved record through the integrated view.  The sidecar must exist and
    read available; the first playback frame is the stored starting state and
    the last recorded state equals the saved camera set's
    ``calibration_params`` bit for bit; opening the dialog shows the static
    first stage with no controls, and the playback stage installs controls
    whose timer is idle until the user acts.
    """
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    import numpy as np

    from pyCamSet.gui import education_phase4
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow import phase4
    from pyCamSet.workflow.education_trace import (
        RunContext, read_trace, trace_path)

    QApplication.instance() or QApplication([])

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    manager, phase3_camset_path = _stage_phases_2_and_3(workspace_root)

    params = {
        "f_loc": str(CORPUS),
        "problem_options": {"outliers": "n", "fixed_pose": 0,
                            "max_nfev": 100},
        "record_education": True,
    }
    record = phase4.run(params, manager, lambda _line: None,
                        phase3_run=manager.find_run("phase3", "p3"),
                        phase3_camset=phase3_camset_path)
    assert record["status"] == "complete", record.get("error")
    assert record["params"]["record_education"] is True

    loaded = manager.find_run("phase4", record["run_id"])
    run_dir = _loaded_run_dir(loaded, manager.workspace_path)

    assert trace_path(run_dir).exists(), "the opted-in run wrote no sidecar"
    trace = read_trace(run_dir, record["run_id"], "phase4")
    assert trace.get("status") == "available", trace

    context = RunContext.from_run(manager, loaded, "phase4")
    assert context.metadata["params"]["record_education"] is True

    view = education_phase4.Phase4View(context)
    try:
        # Nothing plays on open: the initial stage has no controls.
        assert view.current_stage() == 0
        assert view.step_controls is None

        derivation = view._derivation
        frames = tuple(derivation.frames)
        assert frames, "the derivation built no playback frames"

        first = frames[0]
        assert first.status == "available"
        assert first.recorded_start is True
        assert int(first.nit) == 0

        steps = next(
            record["payload"]["steps"]
            for record in trace["records"]
            if record["kind"] == "accepted_step")
        assert len(steps) == len(frames)
        saved = load_CameraSet(record["artifacts"]["self_calibrated_camset"])
        saved_params = np.asarray(saved.calibration_params, dtype=np.float64)
        # The last recorded state equals the saved result bit for bit.  The
        # reader already refused any trace whose endpoints or recorded costs
        # do not reproduce against the saved handler, so an available trace
        # is itself evidence the recorded path reproduces the run.
        last_params = np.asarray(steps[-1]["x"], dtype=np.float64)
        np.testing.assert_array_equal(last_params, saved_params)

        # The playback stage installs controls whose timer is idle.
        assert derivation.status == "available"
        view.set_stage(2)  # 'Watching the solver'
        controls = view.step_controls
        assert controls is not None
        assert not controls._timer.isActive()
    finally:
        view.close()


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.data
def test_default_run_records_no_trace_and_says_so(tmp_path):
    """Default run: no sidecar, setting recorded off, the required message."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow import phase4
    from pyCamSet.workflow.education_trace import RunContext, trace_path

    QApplication.instance() or QApplication([])

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    manager, phase3_camset_path = _stage_phases_2_and_3(workspace_root)

    params = {
        "f_loc": str(CORPUS),
        "problem_options": {"outliers": "n", "fixed_pose": 0,
                            "max_nfev": 100},
    }
    record = phase4.run(params, manager, lambda _line: None,
                        phase3_run=manager.find_run("phase3", "p3"),
                        phase3_camset=phase3_camset_path)
    assert record["status"] == "complete", record.get("error")

    # The default is recorded off, explicitly, in the run's own params.
    assert record["params"]["record_education"] is False
    loaded = manager.find_run("phase4", record["run_id"])
    assert not trace_path(_loaded_run_dir(loaded, manager.workspace_path)).exists()

    context = RunContext.from_run(manager, loaded, "phase4")
    view = education_phase4.Phase4View(context)
    try:
        message = view.stage_text(2)  # 'Watching the solver' renders the playback statement
        # The required trace-absence caption, keyed on the recorded setting.
        assert "Steps were not recorded for this run." in message
        assert "Record solver steps" in message
        assert view.step_controls is None
        shape = view.narrative.shape_copy()
        assert shape and shape in view.stage_text(1), "the shape stage must still have copy"
    finally:
        view.close()