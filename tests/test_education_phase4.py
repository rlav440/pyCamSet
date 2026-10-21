'''Verify the Phase 4 education view honours saved evidence (provenance, qualifiers,
recorded steps, the decision) after the visual redesign.
workflow/education_visual_p4.Phase4Narrative and are rendered in each stage's caption, chips and
Numbers drawer. ``_frame`` reads a topic of the earlier text-first storyboard from where the
redesign renders it, and asserts that it is rendered there.
'''
from __future__ import annotations

import dataclasses
from pathlib import Path
import shutil
from types import SimpleNamespace

import cv2
import pytest
import numpy as np


def _app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _state(**overrides):
    values = {
        "path": None, "origin_run_id": None,
        "verification": SimpleNamespace(value="could not be checked"),
        "reason": "cannot be traced", "confirmed": False,
        "source_check": "not checkable (the photos behind the starting camset cannot be traced)",
        "phase3_before_px": None, "stored_mean_px": None,
        "start_mismatch": None, "fixed_pose_matches": None,
        "effective_fixed_parameter_differences": (),
        "rig_comparison_withheld": False, "photo_names": (),
        "phase3_fixed_pose_ordinal1": None, "phase4_fixed_pose_ordinal1": None,
        "loss": "linear", "f_scale": 1.0,
    }
    return SimpleNamespace(**(values | overrides))


def _context(tmp_path: Path, *, status="complete", state=None, metadata=None):
    run_dir = tmp_path / "phase4_runs" / "run-4"
    run_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        run_id="run-4", phase="phase4", run_dir=run_dir, status=status,
        metadata=metadata or {"params": {}, "diagnostics": {}},
        phase3_state_source=lambda: state or _state(),
        artifact=lambda phase: None,
    )


#: Where each topic of the earlier text-first storyboard is rendered in the redesigned view
#: (0-based stage): provenance, start and reference photo on "Start from Phase 3", the shape on
#: "Freeing the shape", the recorded steps on "Watching the solver" and the decision on "Result".
_TOPIC_STAGE = {
    "Input and quality": 0, "Starting settings": 0, "Reference photo and rig": 0,
    "Shape and observations": 1, "Recorded-step playback": 2, "Run status": 2,
    "Error, loss and decision": 4,
}
_PLAYBACK = 2
_TRUST = 3
_RESULT = 4


def _statement(view, title):
    """The topic's L1 statement as the narrative words it."""
    n = view.narrative
    if title == "Recorded-step playback":
        return (n.failed_playback_copy() if view.run_context.status == "failed"
                else n.playback_copy(view.playback.playable))
    return {"Input and quality": n.input_copy, "Starting settings": n.start_copy,
            "Reference photo and rig": n.gauge_copy, "Shape and observations": n.shape_copy,
            "Error, loss and decision": n.decision_copy, "Run status": n.status_copy}[title]()


def _detail(view, title):
    """The topic's L2 addition."""
    n = view.narrative
    return {"Shape and observations": n.shape_detail, "Error, loss and decision": n.loss_detail,
            "Reference photo and rig": n.gauge_detail}.get(title, lambda: "")()


def _topic_rows(title, rows):
    """The Numbers rows belonging to one topic, in the earlier table's order."""
    labels = [row[0] for row in rows]
    if title == "Input and quality":
        return [row for row in rows if row[0] in ("Phase 3 camset used", "Phase 3 ancestry")]
    if title == "Starting settings":
        return [row for row in rows
                if row[0] in ("Reference photo (fixed pose)", "Effective fixed parameters")]
    if title == "Reference photo and rig":
        head = [row for row in rows if row[0] in ("Reference photo (fixed pose)", "Reference photo")]
        start = labels.index("Reference photo and position frame") + 1
        return head + [row for row in rows[start:] if row[0] != "Position frame"]
    if title == "Shape and observations":
        return [row for row in rows[:3]] + [row for row in rows
                                            if row[0] == "Display-fit scale on measured points"]
    if title in ("Recorded-step playback", "Run status"):
        return [row for row in rows if row[0] != "About these steps"]
    if title == "Error, loss and decision":
        return list(rows[:3])
    return list(rows)


def _frame(view, title):
    """One topic of the earlier storyboard, read from the stage that renders it.

    ``copy`` (L1 statement) and ``detail`` (L2 addition) are asserted to be rendered on that
    stage at that level; ``rows`` / ``rows_detail`` are the topic's rendered Numbers rows at
    L1 / L2. The learner's stage and level are restored afterwards.
    """
    level, stage = view.level(), view.current_stage()
    index = _TOPIC_STAGE[title]
    try:
        text1 = view.stage_text(index, level=1)
        rows1 = _topic_rows(title, view.numbers_rows())
        copy = _statement(view, title)
        if title == "Run status":
            # The run's status is shown with the gate; its recorded costs with the steps.
            assert copy in view.stage_text(_TRUST), copy
        else:
            assert copy in text1, (title, copy)
        detail = _detail(view, title)
        text2 = view.stage_text(index, level=2)
        rows2 = _topic_rows(title, view.numbers_rows())
        if detail:
            assert detail in text2, (title, detail)
        playback = (view.narrative.failed_playback_copy()
                    if view.run_context.status == "failed" else "")
    finally:
        view.set_level(level)
        view.set_stage(stage)
    return {"title": title, "copy": copy, "detail": detail, "rows": rows1,
            "rows_detail": rows2, "playback": playback, "text": text1, "text_l2": text2}


def compose(frame, level):
    """A topic's statement as rendered at one detail level (L2 adds its detail)."""
    if level >= 2 and frame.get("detail"):
        return f"{frame['copy']} {frame['detail']}"
    return frame["copy"]


def _all_topics(view):
    return {title: _frame(view, title) for title in (
        "Input and quality", "Starting settings", "Reference photo and rig",
        "Shape and observations", "Recorded-step playback", "Error, loss and decision")}


def _table(view):
    return view.stage_view.numbers.table


def _saved_phase4_context(tmp_path: Path, *, used_run_id="p3", mismatch=None,
                          phase4_fixed_pose=None):
    """Build a saved-camset workspace with persisted legacy residuals.

    This is a saved-state GUI fixture, not a native Phase 4 solver execution.
    """
    from pyCamSet.utils.saving import load_CameraSet, save_camset
    from pyCamSet.workflow.education_trace import reprojection_segment_mean
    from pyCamSet.workflow.workspace import WorkspaceManager

    legacy = Path(__file__).parent / "test_data" / "education" / "legacy"
    phase4_path = legacy / "phase4_325fae4.camset.json"
    source = load_CameraSet(phase4_path)
    image_root = tmp_path / "images"
    for camera_name in source.calibration_handler.detection.cam_names:
        camera_dir = image_root / camera_name
        camera_dir.mkdir(parents=True)
        for index in range(source.calibration_handler.detection.max_ims):
            (camera_dir / f"frame{index}.png").write_bytes(b"")

    workspace = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    p1_dir = workspace / "phase1_runs" / "p1"
    p1_dir.mkdir(parents=True)
    p1_pickle = p1_dir / "detected_datapoints.pickle"
    shutil.copyfile(legacy / "phase1_325fae4.pickle", p1_pickle)
    manager.save_run("phase1", "p1", {
        "run_id": "p1", "params": {"f_loc": str(image_root), "caching": False},
        "artifacts": {"detected_datapoints_pickle": str(p1_pickle)},
    })
    p2_dir = workspace / "phase2_runs" / "p2"
    p2_dir.mkdir(parents=True)
    p2_camset = p2_dir / "initial_cameras.camset"
    shutil.copyfile(phase4_path, p2_camset)
    manager.save_run("phase2", "p2", {
        "run_id": "p2", "inputs": {"phase1_run_id": "p1"},
        "artifacts": {"initial_camset": str(p2_camset),
                      "education_used_detections": {
                          "schema_version": 1, "source": "phase1_pickle",
                          "path": str(p1_pickle), "fresh_pass_reasons": [],
                          "cache_hit": False}},
    })

    p3_dir = workspace / "phase3_runs" / used_run_id
    p3_dir.mkdir(parents=True)
    p3_camset = p3_dir / "optimised_cameras.camset"
    shutil.copyfile(phase4_path, p3_camset)
    if mismatch == "default_equivalent_fixed":
        # Preserve the real default serialisation contrast while leaving every
        # effective fixed-parameter value empty on both sides.
        used = load_CameraSet(p3_camset)
        used.calibration_handler.fixed_params = {name: {} for name in ("1", "2", "3")}
        save_camset(used, p3_camset)
    manager.save_run("phase3", used_run_id, {
        "run_id": used_run_id, "inputs": {"phase1_run_id": "p1", "phase2_run_id": "p2"},
        "params": {"problem_options": {"fixed_pose": 0}},
        "diagnostics": {"D3.6_final_euclid_px": 12.345},
        "artifacts": {"optimised_camset": str(p3_camset),
                      "phase1_detection_pickle_used": str(p1_pickle),
                      "phase2_initial_camset_used": str(p2_camset)},
    })

    p4_dir = workspace / "phase4_runs" / "p4"
    p4_dir.mkdir(parents=True)
    p4_camset = p4_dir / "self_calibrated_cameras.camset"
    p4 = load_CameraSet(phase4_path)
    if phase4_fixed_pose is not None:
        # Persist a genuine Phase 4 start-setting difference in the camset.
        p4.calibration_handler.problem_opts["fixed_pose"] = phase4_fixed_pose
    source_mean = reprojection_segment_mean(
        source.calibration_result, len(source.calibration_handler._flat_detections()))
    start_mean = source_mean
    if mismatch in {"fixed_parameters", "fixed_parameters_matching_residual"}:
        intrinsic = source.get_cam_dict()["1"].intrinsic.reshape(-1).copy()
        intrinsic[0] += 1.0
        p4.calibration_handler.fixed_params = {"1": {"int": intrinsic}}
        if mismatch == "fixed_parameters":
            # The positive control records a residual that disagrees with the
            # Phase 3 mean; the matched-residual control keeps it unchanged.
            start_mean += 1.0
    elif mismatch == "default_equivalent_fixed":
        p4.calibration_handler.fixed_params = {}
    save_camset(p4, p4_camset)
    manager.save_run("phase4", "p4", {
        "run_id": "p4", "status": "complete", "inputs": {"phase3_run_id": "p3-parent"},
        "params": {"f_loc": str(image_root)},
        "diagnostics": {"D4.3_initial_euclid_px": start_mean},
        "report": {"blocking_flags": []},
        "artifacts": {"self_calibrated_camset": str(p4_camset),
                      "phase3_camset_used": str(p3_camset)},
    })
    if used_run_id != "p3-parent":
        parent_dir = workspace / "phase3_runs" / "p3-parent"
        parent_dir.mkdir()
        parent_path = parent_dir / "optimised_cameras.camset"
        shutil.copyfile(phase4_path, parent_path)
        if mismatch == "fixed_parameters":
            # Counterfactual parent differs from the actual used source: the
            # Phase 4 fixed intrinsic equals this parent, not the used camset.
            parent = load_CameraSet(parent_path)
            parent_intrinsic = parent.get_cam_dict()["1"].intrinsic.reshape(-1).copy()
            parent_intrinsic[0] += 1.0
            parent.calibration_handler.fixed_params = {"1": {"int": parent_intrinsic}}
            parent.get_cam_dict()["1"].intrinsic = parent_intrinsic.reshape(3, 3)
            save_camset(parent, parent_path)
        # Preserve identical saved calibration state but make byte identity
        # distinct, so the path/hash control cannot silently treat it as parent.
        p3_camset.write_bytes(p3_camset.read_bytes() + b"\n")
        manager.save_run("phase3", "p3-parent", {
            "run_id": "p3-parent", "artifacts": {"optimised_camset": str(parent_path)},
        })
    return manager, manager.find_run("phase4", "p4"), p3_camset, p4_camset


def _assert_failed_run_borrows_no_phase3_result(view):
    """A failed run never shows Phase 3's values or a Phase 4 result as if it were usable."""
    assert view.visual.status == "failed"
    assert view.visual.start is None and view.visual.final is None
    assert not view.playback.playable
    for stage in (0, _RESULT):
        rows = view.numbers_rows() if view.stage_text(stage) else []
        assert rows == [("Phase 4 result", "failed", view.narrative.status_copy())], (stage, rows)
    assert "Phase 3" not in view.narrative.status_copy()


@pytest.mark.gui
def test_failed_phase4_trace_shows_failure_without_phase3_frames(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {
        "status": "available", "records": [{"kind": "accepted_step", "payload": {
            "steps": [{"nit": 0, "cost": 8.5, "x": [0.0]},
                      {"nit": 3, "cost": 2.25, "x": [1.0]}]}}]})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("failed result"))
    context = _context(tmp_path, status="failed", metadata={"error": "save/reload check failed"})
    view = education_phase4.Phase4View(context)
    failed = _frame(view, "Run status")
    assert "failed" in failed["copy"]
    assert "Phase 3" not in failed["copy"]
    assert failed["rows"][1:] == [
        ("Starting point (not an accepted step)", "8.5", "recorded trace; failed run"),
        ("Step 3", "2.25", "recorded trace; failed run"),
    ]
    assert "not offered" in failed["playback"]
    text = view.stage_text(_PLAYBACK)
    assert "Playback is not offered" in text
    assert "text alternative" in text
    rows = view.numbers_rows()
    assert rows[2][:2] == ("Step 3", "2.25")
    assert view.step_controls is None
    _assert_failed_run_borrows_no_phase3_result(view)
    view.close()


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.data
def test_real_phase4_run_keeps_trace_after_post_trace_failure(tmp_path, monkeypatch):
    """A real producer trace remains readable when a later check fails."""
    _app()
    from pyCamSet import ChArUco
    from pyCamSet.calibration.camera_calibrator import (
        detect_datapoints_in_imfile, run_initial_calibration,
    )
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow import phase3, phase4
    from pyCamSet.workflow.education_trace import read_trace, write_trace
    from pyCamSet.workflow.workspace import WorkspaceManager

    corpus = Path(__file__).parent / "test_data" / "calibration_charuco"
    target = ChArUco(num_squares_x=20, num_squares_y=20,
                     square_size=4.0, legacy=False)
    detections, resolutions = detect_datapoints_in_imfile(
        f_loc=corpus, caching=False, calibration_target=target, threads=1)
    initial = run_initial_calibration(detections, target, resolutions, save=False)
    initial.set_resolutions_from_file(floc=corpus)
    _handler, _result, phase3_camset, _stats = phase3.solve(
        initial, detections, target, fixed_params={},
        options={"outliers": "n", "fixed_pose": 0}, threads=1)

    manager = WorkspaceManager(tmp_path / "workspace")
    p3_dir = manager.run_dir("phase3", "producer-p3")
    p3_path = p3_dir / "optimised_cameras.camset"
    phase3_camset.save(p3_path)
    phase3_record = {
        "run_id": "producer-p3", "phase": "phase3", "status": "complete",
        "diagnostics": {"D3.6_final_euclid_px": 1.0},
        "artifacts": {"optimised_camset": str(p3_path)},
    }
    manager.save_run("phase3", "producer-p3", phase3_record)

    # Fail only after the production writer has persisted its valid trace.
    def write_then_fail(*args, **kwargs):
        write_trace(*args, **kwargs)
        raise RuntimeError("controlled post-trace acceptance failure")

    monkeypatch.setattr(phase4, "write_trace", write_then_fail)
    metadata = phase4.run(
        {"record_education": True,
         "problem_options": {"outliers": "n", "fixed_pose": 0},
         "threads": 1}, manager, phase3_run=phase3_record,
        phase3_camset=p3_path)

    assert metadata["status"] == "failed"
    assert metadata["error"] == "controlled post-trace acceptance failure"
    saved_run = manager.find_run("phase4", metadata["run_id"])
    assert saved_run is not None
    trace = read_trace(manager.run_dir("phase4", metadata["run_id"]),
                       metadata["run_id"], "phase4")
    assert trace["status"] == "available", trace
    accepted = [record for record in trace["records"]
                if record["kind"] == "accepted_step"]
    assert accepted and accepted[0]["payload"]["steps"]
    recorded_steps = accepted[0]["payload"]["steps"]

    context = SimpleNamespace(
        manager=manager, run_dir=manager.run_dir("phase4", metadata["run_id"]),
        run_id=metadata["run_id"], phase="phase4", status="failed",
        metadata=saved_run,
    )
    view = education_phase4.Phase4View(context)
    failed = _frame(view, "Run status")
    assert "failed" in failed["copy"]
    assert "controlled post-trace acceptance failure" in failed["copy"]
    assert "Phase 3" not in failed["copy"]
    assert failed["rows"][1][2] == "recorded trace; failed run"
    assert failed["rows"][1][0] == "Starting point (not an accepted step)"
    assert len(failed["rows"]) == len(recorded_steps) + 1
    view.set_stage(_PLAYBACK)
    shown = view.numbers_rows()
    for row_index, (row, step) in enumerate(zip(failed["rows"][1:], recorded_steps), start=1):
        expected_label = ("Starting point (not an accepted step)" if step["nit"] == 0
                          else f"Step {step['nit']}")
        assert row == (expected_label, f"{float(step['cost']):.4g}",
                       "recorded trace; failed run")
        assert shown[row_index] == row
    assert "not offered" in failed["playback"]
    assert "Playback is not offered" in view.stage_text(_PLAYBACK)
    assert view.step_controls is None
    _assert_failed_run_borrows_no_phase3_result(view)
    view.close()


@pytest.mark.gui
def test_failed_phase4_without_trace_has_no_phase3_result_or_playback(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("failed result"))
    view = education_phase4.Phase4View(
        _context(tmp_path, status="failed", metadata={"error": "save/reload check failed"}))
    failed = _frame(view, "Run status")
    assert failed["rows"][1] == (
        "Recorded steps", "unavailable", "no readable Phase 4 trace")
    assert "not offered" in failed["playback"]
    text = view.stage_text(_PLAYBACK)
    assert "No readable Phase 4 trace" in text
    assert "Phase 3" not in text
    _assert_failed_run_borrows_no_phase3_result(view)
    view.close()


@pytest.mark.gui
def test_phase4_provenance_and_loss_are_read_from_state_source(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("synthetic test"))
    state = _state(origin_run_id="used-phase3", confirmed=True,
                   verification=SimpleNamespace(value="verified"), reason=None,
                   source_check="verified by manifest", photo_names=("capture-01.jpg",),
                   loss="soft_l1", f_scale=0.5, stored_mean_px=1.25)
    view = education_phase4.Phase4View(_context(
        tmp_path, state=state,
        metadata={"params": {"problem_options": {"loss": "huber", "f_scale": 2.0}},
                  "diagnostics": {"D4.3_initial_euclid_px": 1.2}}))
    assert "used-phase3" in _frame(view, "Input and quality")["copy"]
    assert "capture-01.jpg" in _frame(view, "Reference photo and rig")["copy"]
    decision_frame = _frame(view, "Error, loss and decision")
    # Loss names and f_scale are L2 detail; L1 says the rules differ in plain words.
    assert "loss" not in decision_frame["copy"].lower() and "f_scale" not in decision_frame["copy"]
    assert "scored the fit with different rules" in decision_frame["copy"]
    assert "f_scale" not in decision_frame["text"]
    loss_copy = decision_frame["detail"]
    assert "Phase 3's loss: soft_l1 (f_scale=0.5)" in loss_copy
    assert "Phase 4's loss: huber (f_scale=2)" in loss_copy
    assert "different losses affect the comparison" in loss_copy
    view.close()


@pytest.mark.gui
def test_unverified_input_disposition_includes_recovery_from_phase3_run(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("unconfirmed input"))
    state = _state(origin_run_id="phase3-X", confirmed=False,
                   verification=SimpleNamespace(value="differs"), reason="differs",
                   phase3_before_px=44.0, stored_mean_px=44.0)
    context = _context(tmp_path, state=state)
    context.quality_disposition = "quality check not attributable to the camset used"
    view = education_phase4.Phase4View(context)
    copy = _frame(view, "Input and quality")["copy"]
    assert "Quality check not attributable to the camset used" in copy
    assert "other than Phase 3 run phase3-X" in copy
    assert "select that run alone" in copy
    # The same A1 route as the other refusals, with the GUI's own button texts.
    assert "open ‘Diagnostics ▼’ and select that run alone, then press ‘Phase 4 - Self-Calibration ▶’" in copy
    assert view.narrative.status_copy().endswith("; run Phase 4 again.")
    before = _frame(view, "Error, loss and decision")["rows"][0]
    assert before[1] == "unavailable"
    # Every stage leads with the unconfirmed-start warning.
    for stage in range(6):
        assert view.stage_text(stage).startswith(
            "The camset this run started from could not be confirmed"), stage
    view.close()


@pytest.mark.gui
def test_phase4_start_mismatch_withholds_rig_comparison_and_missing_points_not_estimated(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("layout unavailable"))
    state = _state(origin_run_id="phase3-run", confirmed=True,
                   verification=SimpleNamespace(value="verified"), reason=None,
                   start_mismatch="fixed_pose", rig_comparison_withheld=True,
                   photo_names=("ref.jpg",), phase3_before_px=3.5)
    view = education_phase4.Phase4View(_context(
        tmp_path, state=state, metadata={"params": {}, "diagnostics": {
            "D4.3_initial_euclid_px": 1.2}}))
    assert "inconsistent position" in _frame(view, "Starting settings")["copy"]
    assert "withheld" in str(_frame(view, "Reference photo and rig")["rows"])
    assert "not directly comparable in this view" in _frame(view, "Starting settings")["copy"]
    assert "N2" not in _frame(view, "Starting settings")["copy"]
    error = _frame(view, "Error, loss and decision")
    before = error["rows"][0]
    assert before[1] == "3.5"
    assert error["rows"][1][1] == "1.2"
    assert before[2] == "Phase 3's final average error, px"
    assert "Phase 3 D3.6" in error["rows_detail"][0][2]
    assert "unavailable" in _frame(view, "Shape and observations")["copy"]
    view.close()


@pytest.mark.gui
@pytest.mark.parametrize(
    ("status", "disposition", "expected"),
    [("complete", "checked", "Quality check: checked"),
     ("complete", "not checked", "Quality check: not checked"),
     ("failed", "failed", "Quality check: failed"),
     ("incomplete", "checked", "Quality check: incomplete"),
     ("complete", "quality check not attributable to the camset used",
      "Quality check not attributable to the camset used")],
)
def test_visible_quality_disposition_and_export_copy(
        tmp_path, monkeypatch, status, disposition, expected):
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("synthetic test"))
    state = _state(confirmed=True, verification=SimpleNamespace(value="verified"),
                   reason=None, source_check="verified")
    if disposition == "quality check not attributable to the camset used":
        state = _state(confirmed=False, verification=SimpleNamespace(value="differs"),
                       reason="differs", origin_run_id="phase3-X")
    diagnostics = {}
    if status == "incomplete":
        # workflow/phase4.py _quality_gate records its reasons as 'blocking_flags'.
        diagnostics = {"quality_gate": {"status": "incomplete", "blocking_flags": [
            "solver did not report successful termination"]}}
    context = _context(tmp_path / status.replace(" ", "-"), status=status,
                       state=state, metadata={"params": {}, "diagnostics": diagnostics})
    context.quality_disposition = disposition
    view = education_phase4.Phase4View(context)
    # The gate stage always shows the status; the opening stage shows it unless the run failed.
    visible = view.stage_text(_TRUST)
    assert expected in visible
    if status != "failed":
        assert expected in view.stage_text(0)
    if status == "incomplete":
        # A1 section 9: say why the run is held back, then one next step.
        assert "solver did not report successful termination" in visible
        assert "Next: fix what the reasons above name, then run Phase 4 again." in visible
        assert "Quality check: checked" not in visible
        assert "Held back" in visible
    if status != "complete" or disposition == "not checked":
        assert "Next:" in view.narrative.status_copy()
    opening = view.stage_text(0)
    if state.verification.value != "verified" and status != "failed":
        assert "The Export tab still offers this run; check which Phase 3 camset it started from first." in opening
    else:
        assert "The Export tab still offers this run" not in opening
    view.close()


@pytest.mark.gui
@pytest.mark.parametrize(
    ("status", "verification", "reason", "disposition", "export_offered"),
    [
        # E1's recorded Phase 4 input outcomes; choice (b) leaves the
        # accepted-only Export gate unchanged, so complete runs remain offered.
        ("complete", "verified", None, "checked", False),
        ("complete", "verified", "verified (read from another copy of this workspace)",
         "checked", True),
        ("complete", "differs", "differs", "quality check not attributable to the camset used", True),
        ("complete", "could not be checked", "cannot be traced",
         "quality check not attributable to the camset used", True),
        ("complete", "verified", None, "not checked", False),
        # The integrated E1 accepted-only gate refuses every non-complete
        # Phase 4 result, independently of its input-verification outcome.
        ("failed", "differs", "differs", "failed", False),
        ("incomplete", "could not be checked", "cannot be traced", "checked", False),
        ("unknown", "differs", "differs", "checked", False),
    ],
)
def test_export_copy_matches_every_e1_phase4_outcome(
        tmp_path, monkeypatch, status, verification, reason, disposition,
        export_offered):
    """Render E1's full offered/refused table under the recorded choice (b)."""
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("synthetic test"))
    state = _state(
        confirmed=verification == "verified",
        verification=SimpleNamespace(value=verification),
        reason=reason,
        source_check="verified" if verification == "verified" else "not checkable",
        origin_run_id="phase3-X" if verification == "differs" else None,
    )
    context = _context(
        tmp_path / f"{status}-{verification.replace(' ', '-')}",
        status=status, state=state, metadata={"params": {}, "diagnostics": {}})
    context.quality_disposition = disposition

    view = education_phase4.Phase4View(context)
    rendered = "\n".join(view.stage_text(stage) for stage in range(6))
    export_sentence = (
        "The Export tab still offers this run; check which Phase 3 camset it started from first.")
    assert (export_sentence in rendered) is export_offered
    view.close()


@pytest.mark.gui
@pytest.mark.parametrize(("once_seen", "expected"), [
    ((0, 1, 2), "Each of the three points pyCamSet chose to pin the shape was seen from one direction only"),
    ((0, 1), "Two of the three points pyCamSet chose to pin the shape were seen from one direction only"),
    ((0,), "One of the three points pyCamSet chose to pin the shape was seen from one direction only"),
])
def test_gauge_once_seen_qualifier_uses_saved_indices_and_point_classes(
        tmp_path, monkeypatch, once_seen, expected):
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import Phase4Derivation

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    classes = SimpleNamespace(unseen=(), one_direction=once_seen,
                              several_directions=(4,), ray_separation_deg=())
    result = Phase4Derivation(
        status="available", reason=None, run_id="run-4", d41_visible_points=4,
        d43_final_px=1.0, d45_gauge_scale=1.0, d47_mean_displacement_mm=0.2,
        point_classes=classes, display_refit_points=np.zeros((5, 3)),
        display_refit_scale=1.0, scale_consistency_ratio=1.0,
        metric_qualifier_required=False, frames=())
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: result)
    metadata = {"params": {}, "diagnostics": {
        "D4.2_gauge_fixed_points": {"indices": [0, 1, 2]}}}
    view = education_phase4.Phase4View(_context(tmp_path, metadata=metadata))
    assert view.metadata["diagnostics"]["D4.2_gauge_fixed_points"]["indices"] == [0, 1, 2]
    assert view.narrative.once_seen_gauge_count(classes) == len(set(once_seen) & {0, 1, 2})
    assert expected in view.narrative.shape_copy()
    shape = _frame(view, "Shape and observations")
    assert expected in shape["copy"]
    # SCI2-10: L2 names D4.7 and what it averages (no stale "scale effect" sentence).
    assert "D4.7" not in shape["copy"]
    assert "D4.7" in compose(shape, 2) and "scale effect" not in compose(shape, 2)
    assert "D4.7" not in shape["text"]
    view.close()


@pytest.mark.gui
@pytest.mark.parametrize(("several_mm", "several", "expected_l1"), [
    (0.05, (3, 4), "Points seen from several directions differ from the design you entered by "
                   "0.05 mm on average"),
    (None, (3, 4), "0.2 mm on average over all seen points; this average includes the 3 point(s) "
                   "seen from one direction only, whose positions are poorly determined"),
    (None, (), "No point was seen from several directions, so no shape difference is backed"),
])
def test_shape_headline_uses_only_points_seen_from_several_directions(
        tmp_path, monkeypatch, several_mm, several, expected_l1):
    """Once-seen points have unconstrained depth; their drift must not lead the shape headline."""
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import Phase4Derivation

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    classes = SimpleNamespace(unseen=(5,), one_direction=(0, 1, 2),
                              several_directions=several, ray_separation_deg=())
    result = Phase4Derivation(
        status="available", reason=None, run_id="run-4", d41_visible_points=3 + len(several),
        d43_final_px=1.0, d45_gauge_scale=1.0, d47_mean_displacement_mm=0.2,
        point_classes=classes, display_refit_points=np.zeros((6, 3)),
        display_refit_scale=1.0, scale_consistency_ratio=1.0,
        metric_qualifier_required=False, frames=(),
        several_directions_mean_displacement_mm=several_mm)
    monkeypatch.setattr(education_phase4, "derive_phase4", lambda *_args, **_kwargs: result)
    view = education_phase4.Phase4View(_context(tmp_path))
    shape = _frame(view, "Shape and observations")
    assert expected_l1 in shape["copy"]
    assert "D4.7" not in shape["copy"]
    assert "D4.7" not in shape["text"]
    level2 = compose(shape, 2)
    assert "recalculated from saved results" in level2 and "scale effect" not in level2
    if several_mm is not None:
        assert ("D4.7 averages all 5 seen points (0.2 mm, recalculated from saved results), "
                "including the 3 seen from one direction only, whose positions are poorly "
                "determined") in level2
    view.close()


@pytest.mark.gui
@pytest.mark.parametrize(("several_mm", "ratio"), [(0.05, 1.03), (None, 1.03), (0.05, 1.0)])
def test_scale_qualifier_follows_d47_not_the_display_refit_headline(
        tmp_path, monkeypatch, several_mm, ratio):
    """The display alignment is rigid (round-1 SCI-1: the printed square size is kept), so an
    out-of-tolerance ratio qualifies the refit headline as it qualifies D4.7.

    With ratio 1.0 and the qualifier still required, only the unverified-units
    part stays on the refit headline (it still affects its mm conversion).
    """
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import Phase4Derivation

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    classes = SimpleNamespace(unseen=(5,), one_direction=(0, 1, 2),
                              several_directions=(3, 4), ray_separation_deg=())
    result = Phase4Derivation(
        status="available", reason=None, run_id="run-4", d41_visible_points=5,
        d43_final_px=1.0, d45_gauge_scale=1.0, d47_mean_displacement_mm=0.2,
        point_classes=classes, display_refit_points=np.zeros((6, 3)),
        display_refit_scale=1.0, scale_consistency_ratio=ratio,
        metric_qualifier_required=True, frames=(),
        several_directions_mean_displacement_mm=several_mm)
    monkeypatch.setattr(education_phase4, "derive_phase4", lambda *_args, **_kwargs: result)
    view = education_phase4.Phase4View(_context(tmp_path))
    shape = _frame(view, "Shape and observations")
    copy = shape["copy"]
    level2 = compose(shape, 2)
    if several_mm is None:
        # Fallback branch leads with D4.7: the qualifier stays on the L1 value.
        assert "0.2 mm on average over all seen points" in copy
        assert f"The scale-consistency ratio is {ratio:g}; this millimetre value is qualified" in copy
    else:
        assert "0.05 mm on average" in copy
        assert "turned and shifted (not resized) onto the design" in copy
        assert "this millimetre value is qualified because that ratio" not in copy
        qualified = (f"The scale-consistency ratio is {ratio:g}, outside tolerance, so this "
                     "millimetre value is qualified, as are D4.7 and the camera distances.")
        if ratio != 1.0:
            assert qualified in copy
            assert "saved length units are unverified" not in copy
        else:
            assert "saved length units are unverified, so this millimetre value is qualified" in copy
            assert "scale-consistency ratio is" not in copy
        d47 = level2.index("D4.7 averages all")
        qualifier = level2.index(f"The scale-consistency ratio is {ratio:g}; D4.7 is a qualified "
                                 "millimetre value")
        assert qualifier > d47
    view.close()


@pytest.mark.gui
@pytest.mark.parametrize(("ratio", "unverified"), [(1.03, True), (1.03, False), (1.0, True)])
def test_refit_headline_keeps_units_qualifier_and_l2_names_the_reason(
        tmp_path, monkeypatch, ratio, unverified):
    """Unverified units qualify the refit mm headline even when the ratio is also out of
    tolerance, and L2 names which reason(s) qualify D4.7 instead of 'ratio or units'."""
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import Phase4Derivation

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    classes = SimpleNamespace(unseen=(5,), one_direction=(0, 1, 2),
                              several_directions=(3, 4), ray_separation_deg=())
    result = Phase4Derivation(
        status="available", reason=None, run_id="run-4", d41_visible_points=5,
        d43_final_px=1.0, d45_gauge_scale=1.0, d47_mean_displacement_mm=0.2,
        point_classes=classes, display_refit_points=np.zeros((6, 3)),
        display_refit_scale=1.0, scale_consistency_ratio=ratio,
        metric_qualifier_required=True, frames=(),
        several_directions_mean_displacement_mm=0.05, saved_units_unverified=unverified)
    monkeypatch.setattr(education_phase4, "derive_phase4", lambda *_args, **_kwargs: result)
    view = education_phase4.Phase4View(_context(tmp_path))
    shape = _frame(view, "Shape and observations")
    copy = shape["copy"]
    level2 = compose(shape, 2)
    units_line = "saved length units are unverified, so this millimetre value is qualified"
    assert (units_line in copy) is unverified
    assert ("outside tolerance, so this millimetre value is qualified" in copy) is (ratio != 1.0)
    assert " or the saved" not in level2 and "tolerance or" not in level2
    reason = {(1.03, True): "that ratio is outside 1 ± 0.01 and the saved length units are unverified",
              (1.03, False): "that ratio is outside 1 ± 0.01.",
              (1.0, True): "because the saved length units are unverified."}[(ratio, unverified)]
    assert reason in level2
    view.close()


@pytest.mark.gui
@pytest.mark.parametrize("required", [True, False])
def test_step_table_rig_change_is_scale_qualified_like_the_rig_table(
        tmp_path, monkeypatch, required):
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable
    from pyCamSet.workflow.education_visual_p4 import CameraPose

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("synthetic rig"))
    view = education_phase4.Phase4View(_context(tmp_path))
    view._derivation = SimpleNamespace(metric_qualifier_required=required)
    monkeypatch.setattr(view.narrative, "display_unit", lambda: ("mm", 1000.0))

    class _Rig(dict):
        def get_names(self):
            return list(self)

    def frame(x):
        return SimpleNamespace(status="available", rig_cameras=_Rig(
            cam0=SimpleNamespace(position=np.zeros(3)),
            cam1=SimpleNamespace(position=np.array([x, 0.0, 0.0]))))

    change = view._largest_rig_change(frame(0.0), frame(0.002))
    assert change == ("2 mm (scale-qualified)" if required else "2 mm")

    # The replayed-step table says the same for pinhole camera centres; a telecentric
    # camera has no position, so its row names the largest turn instead.
    def state(x, telecentric=False):
        R = np.eye(3)
        cams = (CameraPose("cam0", 0, R, np.zeros(3), telecentric),
                CameraPose("cam1", 1, R, -R @ np.array([x, 0.0, 0.0]), telecentric))
        return SimpleNamespace(cameras=cams)

    assert view._largest_camera_change(state(0.0), state(0.002)) == change
    assert view._largest_camera_change(state(0.0, True), state(0.002, True)).endswith("° turn")
    view.close()


@pytest.mark.gui
@pytest.mark.parametrize("source_check", [
    "not checkable (its detection run may have reused saved detections, so which photo each came from was not recorded)",
    "not checkable (its detections may have come from an earlier photo set)",
    "mismatch (the photo list of its detection run no longer matches the photos in the folder)",
])
def test_uncertain_cache_provenance_never_names_current_photo(tmp_path, monkeypatch, source_check):
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("synthetic provenance"))
    state = _state(confirmed=True, verification=SimpleNamespace(value="verified"),
                   reason=None, source_check=source_check, photo_names=())
    view = education_phase4.Phase4View(_context(tmp_path, state=state))
    copy = _frame(view, "Reference photo and rig")["copy"]
    assert "current file name is assigned" in copy
    assert "capture-" not in copy and "frame" not in copy
    if "may have reused" in source_check:
        assert "may have reused saved detections" in copy
    elif "earlier photo set" in source_check:
        assert "earlier photo set" in copy
    else:
        assert "photo list" in copy
    view.close()


@pytest.mark.gui
def test_recorded_step_text_preserves_solver_step_numbers_and_thinning_gaps(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_visual_p4 import step_rows

    frames = [SimpleNamespace(nit=0, cost=9.0, status="unavailable"),
              SimpleNamespace(nit=2, cost=4.0, status="unavailable"),
              SimpleNamespace(nit=5, cost=1.0, status="unavailable")]
    rows = step_rows(frames)
    assert rows[:3] == [
        ("Starting point (not an accepted step)", "9", "—"),
        ("Step 2", "4", "unavailable"),
        ("Step 5", "1", "unavailable"),
    ]
    assert rows[-1] == ("Thinning", "steps not kept", "1, 3, 4")

    # The same rows reach the learner on 'Watching the solver' when nothing is playable.
    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: SimpleNamespace(
                            frames=tuple(frames), point_classes=None, d43_final_px=None,
                            display_refit_points=None, scale_consistency_ratio=None,
                            metric_qualifier_required=False, d47_mean_displacement_mm=None))
    view = education_phase4.Phase4View(_context(tmp_path))
    assert _frame(view, "Recorded-step playback")["rows"] == rows
    view.close()


@pytest.mark.gui
@pytest.mark.parametrize(("phase3_loss", "phase3_scale", "phase4_loss", "phase4_scale"), [
    ("linear", 1.0, "soft_l1", 1.0),
    ("huber", 0.7, "huber", 1.0),
    ("linear", 1.0, "huber", 0.3),
])
def test_loss_comparison_names_both_phases_and_scales(
        tmp_path, monkeypatch, phase3_loss, phase3_scale, phase4_loss, phase4_scale):
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("synthetic loss"))
    state = _state(confirmed=True, verification=SimpleNamespace(value="verified"),
                   reason=None, loss=phase3_loss, f_scale=phase3_scale)
    metadata = {"params": {"problem_options": {"loss": phase4_loss, "f_scale": phase4_scale}},
                "diagnostics": {}}
    view = education_phase4.Phase4View(_context(tmp_path, state=state, metadata=metadata))
    frame = _frame(view, "Error, loss and decision")
    copy = frame["detail"]
    assert f"Phase 3's loss: {phase3_loss}" in copy
    assert f"Phase 4's loss: {phase4_loss}" in copy
    assert "Phase 3's loss" in copy and "Phase 4's loss" in copy
    if (phase3_loss, phase3_scale) != (phase4_loss, phase4_scale):
        assert "different losses affect the comparison" in copy
        # Only the loss differed here, so the note must not blame the start.
        assert "starting point" not in copy
        assert "scored the fit with different rules" in frame["copy"]
    assert frame["copy"].startswith("Does Phase 4 fit your points better")
    view.close()


@pytest.mark.gui
def test_n2_displays_different_saved_pose_ordinals_without_inventing_photo_names(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("synthetic N2"))
    state = _state(confirmed=True, reason=None, source_check="verified saved detections",
                   start_mismatch="fixed_pose", rig_comparison_withheld=True,
                   phase3_fixed_pose_ordinal1=1, phase4_fixed_pose_ordinal1=6,
                   photo_names=())
    view = education_phase4.Phase4View(_context(tmp_path, state=state))
    start = _frame(view, "Starting settings")["copy"]
    gauge = _frame(view, "Reference photo and rig")["copy"]
    assert "Phase 3 used the 1st photo of the detection run's list" in start
    assert "Phase 4 used the 6th photo of the detection run's list" in start
    assert "Phase 3 used the 1st photo of the detection run's list" in gauge
    assert "Phase 4 used the 6th photo of the detection run's list" in gauge
    assert "withheld" in gauge
    assert "cannot be named from the saved source" in gauge
    assert "capture-" not in gauge and "frame" not in gauge
    view.close()


@pytest.mark.gui
@pytest.mark.parametrize("ordinal", [None, 0, -1, 1.5, True, "6"])
def test_missing_or_invalid_saved_pose_ordinal_is_not_inferred(tmp_path, monkeypatch, ordinal):
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("synthetic unavailable"))
    state = _state(confirmed=True, reason=None, source_check="verified saved detections",
                   phase3_fixed_pose_ordinal1=ordinal, phase4_fixed_pose_ordinal1=None,
                   photo_names=())
    view = education_phase4.Phase4View(_context(tmp_path, state=state))
    copy = _frame(view, "Reference photo and rig")["copy"]
    assert "Phase 3's reference photo cannot be established from the saved data" in copy
    assert "Phase 4's reference photo cannot be established from the saved data" in copy
    assert "cannot be named from the saved source" in copy
    assert "capture-" not in copy and "frame" not in copy
    view.close()


@pytest.mark.gui
def test_cached_detection_provenance_withholds_photo_names_but_keeps_ordinals(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("synthetic cache"))
    state = _state(confirmed=True, reason=None,
                   source_check="not checkable (its detection run may have reused saved detections, so which photo each came from was not recorded)",
                   photo_names=("STALE_FOLDER_FRAME_999.png",), phase3_fixed_pose_ordinal1=1,
                   phase4_fixed_pose_ordinal1=1)
    view = education_phase4.Phase4View(_context(tmp_path, state=state))
    copy = _frame(view, "Reference photo and rig")["copy"]
    assert "Phase 3 used the 1st photo of the detection run's list" in copy
    assert "Phase 4 used the 1st photo of the detection run's list" in copy
    assert "may have reused saved detections" in copy
    assert "which photo each came from was not recorded" in copy
    assert "the 1st photo in the detection run's list" in copy
    assert "not necessarily the 1st file in the folder now" in copy
    assert "STALE_FOLDER_FRAME_999.png" not in copy
    assert "capture-" not in copy and "frame" not in copy
    view.close()


@pytest.mark.gui
def test_real_state_source_used_camset_path_wins_over_recorded_parent(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable
    from pyCamSet.workflow.education_trace import RunContext, Verification

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("fixture GUI only"))
    manager, record, used_path, _ = _saved_phase4_context(
        tmp_path, used_run_id="p3-actual")
    context = RunContext.from_run(manager, record)
    state = context.phase3_state_source()
    assert state.path == used_path
    assert state.origin_run_id == "p3-actual"
    assert state.verification is Verification.VERIFIED
    assert state.confirmed
    assert context.quality_disposition == "quality check not attributable to the camset used"
    view = education_phase4.Phase4View(context)
    copy = _frame(view, "Input and quality")["copy"]
    assert "Phase 3 run p3-actual" in copy
    assert "Quality check not attributable" in copy
    assert "p3-parent" not in copy.split("started from", 1)[1].split(".", 1)[0]
    view.close()


@pytest.mark.gui
def test_changed_used_camset_fails_real_confirmation_and_rendered_attribution(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.utils.saving import load_CameraSet, save_camset
    from pyCamSet.workflow.education_derive import DerivationUnavailable
    from pyCamSet.workflow.education_trace import RunContext, Verification

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("fixture GUI only"))
    manager, record, used_path, _ = _saved_phase4_context(
        tmp_path, used_run_id="p3-actual")
    edited = load_CameraSet(used_path)
    edited.calibration_handler.detection._data = np.asarray(
        edited.calibration_handler.detection._data).copy()
    edited.calibration_handler.detection._data[0, 3] += 0.25
    save_camset(edited, used_path)
    context = RunContext.from_run(manager, manager.find_run("phase4", "p4"))
    state = context.phase3_state_source()
    assert state.path == used_path
    assert state.verification is Verification.VERIFIED
    assert state.detections_match is False
    assert not state.confirmed
    assert "cannot be confirmed" in (state.reason or "")
    view = education_phase4.Phase4View(context)
    copy = _frame(view, "Input and quality")["copy"]
    assert "Phase 3 run p3-actual" in copy
    assert "cannot be confirmed" in copy
    assert "p3-parent" not in copy.split("started from", 1)[1].split(".", 1)[0]
    view.close()


@pytest.mark.gui
@pytest.mark.parametrize(("saved_form", "expected_mismatch", "expected_differences"), [
    ("fixed_parameters", "fixed_parameters", ("1.int",)),
    ("fixed_parameters_matching_residual", None, ("1.int",)),
    ("default_equivalent_fixed", None, ()),
])
def test_real_saved_fixed_parameter_effective_comparison_is_rendered(
        tmp_path, monkeypatch, saved_form, expected_mismatch, expected_differences):
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.education_derive import DerivationUnavailable
    from pyCamSet.workflow.education_trace import RunContext

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("fixture GUI only"))
    manager, record, used_path, p4_path = _saved_phase4_context(tmp_path, mismatch=saved_form)
    context = RunContext.from_run(manager, record)
    state = context.phase3_state_source()
    source_mean = state.stored_mean_px
    assert source_mean is not None and source_mean > 0.0
    assert state.confirmed
    assert state.start_mismatch == expected_mismatch
    assert state.detections_match is True
    assert state.start_metric_px is not None
    if expected_mismatch:
        # The recorded start residual must be the reason the otherwise
        # effective fixed-value difference becomes an explanatory mismatch.
        assert state.start_metric_px != pytest.approx(source_mean, rel=1e-9, abs=1e-9)
        assert state.start_metric_px - source_mean == pytest.approx(1.0)
        assert state.effective_fixed_parameter_differences == expected_differences
        assert state.phase3_before_px == pytest.approx(12.345)
    else:
        assert state.start_metric_px == pytest.approx(source_mean, rel=1e-9, abs=1e-9)
        assert state.effective_fixed_parameter_differences == expected_differences
        used_camset = load_CameraSet(used_path)
        if saved_form == "default_equivalent_fixed":
            assert used_camset.calibration_handler.fixed_params == {"1": {}, "2": {}, "3": {}}
            assert load_CameraSet(p4_path).calibration_handler.fixed_params == {}
        else:
            phase4_fixed = load_CameraSet(p4_path).calibration_handler.fixed_params
            assert set(phase4_fixed) == {"1"}
            assert state.effective_fixed_parameter_differences == ("1.int",)
    if expected_mismatch:
        parent_record = manager.find_run("phase3", "p3-parent")
        parent_path = Path(parent_record["artifacts"]["optimised_camset"])
        phase4_camset = load_CameraSet(p4_path)
        used_camset = load_CameraSet(used_path)
        parent_camset = load_CameraSet(parent_path)
        held_value = np.asarray(
            phase4_camset.calibration_handler.fixed_params["1"]["int"]).reshape(3, 3)
        actual_used_value = np.asarray(used_camset.get_cam_dict()["1"].intrinsic)
        recorded_parent_value = np.asarray(parent_camset.get_cam_dict()["1"].intrinsic)
        assert not np.array_equal(held_value, actual_used_value)
        assert np.array_equal(held_value, recorded_parent_value)
    view = education_phase4.Phase4View(context)
    start = _frame(view, "Starting settings")["copy"]
    if expected_mismatch:
        assert "held 1.int fixed at values different" in start
        assert "not a measure of Phase 3's fit" in start
        assert "The Phase 4 start is confirmed" not in start
        before = _frame(view, "Error, loss and decision")["rows"][0]
        assert before[0:2] == ("Before", "12.35")
        assert before[2] == "Phase 3's final average error, px"
        assert (_frame(view, "Error, loss and decision")["rows_detail"][0]
                == ("Before", "12.35", "Phase 3 D3.6 from its verified state source"))
        identity = _frame(view, "Input and quality")["copy"]
        assert "Phase 3 run p3" in identity
        assert "verified" in identity.lower()
    else:
        assert "Phase 4 start is confirmed" in start
        assert "held 1.int fixed at values different" not in start
        if expected_differences:
            # A genuine effective difference alone must not be explained as a
            # start mismatch when the recorded residual matches Phase 3.
            assert "fixed parameter" not in start.lower()
            assert state.start_metric_px == pytest.approx(state.stored_mean_px, rel=1e-9, abs=1e-9)
    assert start in view.stage_text(0)
    if expected_mismatch:
        view.set_stage(_RESULT)
        assert _table(view).item(0, 1).text() == "12.35"
    view.close()


@pytest.mark.gui
def test_real_saved_fixed_pose_n2_withholds_gauge_but_preserves_phase3_d36(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.education_derive import DerivationUnavailable
    from pyCamSet.workflow.education_trace import RunContext, Verification

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("fixture GUI only"))
    manager, record, used_path, p4_path = _saved_phase4_context(
        tmp_path, used_run_id="p3-fixed-pose-zero", phase4_fixed_pose=5)

    # Verify the two distinct saved authorities before asking the GUI to render.
    phase3_record = manager.find_run("phase3", "p3-fixed-pose-zero")
    phase3_camset = load_CameraSet(used_path)
    phase4_camset = load_CameraSet(p4_path)
    assert phase3_record["params"]["problem_options"]["fixed_pose"] == 0
    assert phase3_camset.calibration_handler.problem_opts["fixed_pose"] == 0
    assert phase4_camset.calibration_handler.problem_opts["fixed_pose"] == 5

    context = RunContext.from_run(manager, record)
    state = context.phase3_state_source()
    assert state.path == used_path
    assert state.verification is Verification.VERIFIED
    assert state.confirmed
    assert state.origin_run_id == "p3-fixed-pose-zero"
    assert state.start_mismatch == "fixed_pose"
    assert state.rig_comparison_withheld
    assert state.phase3_fixed_pose_ordinal1 == 1
    assert state.phase4_fixed_pose_ordinal1 == 6

    view = education_phase4.Phase4View(context)
    start_copy = _frame(view, "Starting settings")["copy"]
    assert "Phase 3 used the 1st photo of the detection run's list" in start_copy
    assert "Phase 4 used the 6th photo of the detection run's list" in start_copy
    assert "not directly comparable in this view" in start_copy
    assert "N2" not in start_copy
    assert "inconsistent position" in start_copy

    gauge = _frame(view, "Reference photo and rig")
    assert "withheld because the two phases used different reference photos" in gauge["copy"]
    assert "cannot be confirmed" not in gauge["copy"]
    comparison = next(row for row in gauge["rows"] if row[0] == "Rig comparison")
    assert comparison[1] == "withheld"
    assert "the two phases used different reference photos" in comparison[2]

    error = _frame(view, "Error, loss and decision")
    before = error["rows"][0]
    assert before[1] == "12.35"
    assert before[2] == "Phase 3's final average error, px"
    assert error["rows"][1][0] == "Phase 4 start"
    assert error["rows"][1][2] == "where Phase 4 began, average error, px"
    assert before[0] == "Before"
    # Diagnostic names arrive at L2 (A1 section 1); the value is unchanged.
    assert error["rows_detail"][0] == ("Before", "12.35", "Phase 3 D3.6 from its verified state source")
    assert error["rows_detail"][1][2] == "D4.3, px"
    view.set_level(1)
    view.set_stage(_RESULT)
    assert "D3.6" not in _table(view).item(0, 2).text()
    view.set_level(2)
    assert _table(view).item(0, 2).text() == "Phase 3 D3.6 from its verified state source"
    assert _table(view).item(0, 1).text() == "12.35"
    view.close()


@pytest.mark.gui
def test_phase4_gui_surfaces_manifest_for_actual_used_y_not_recorded_x(
        tmp_path, monkeypatch):
    _app()
    from pyCamSet.calibration_targets.core.abstract_target import enumerate_detection_images
    from pyCamSet.gui import education_phase4
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.education_derive import DerivationUnavailable
    from pyCamSet.workflow.education_manifest import describe_enumeration
    from pyCamSet.workflow.education_trace import RunContext, reprojection_segment_mean
    from pyCamSet.workflow.workspace import WorkspaceManager

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("fixture GUI only"))
    legacy = Path(__file__).parent / "test_data" / "education" / "legacy"
    source_camset = legacy / "phase4_325fae4.camset.json"
    source_pickle = legacy / "phase1_325fae4.pickle"
    manager = WorkspaceManager(tmp_path / ".pycamset_workspace")
    source = load_CameraSet(source_camset)
    camera_names = tuple(source.calibration_handler.detection.cam_names)
    root_x, root_y = tmp_path / "photos-X", tmp_path / "photos-Y"

    for label, run_id, root in (("X", "p1-x", root_x), ("Y", "p1-y", root_y)):
        camera_paths = {}
        for camera in camera_names:
            folder = root / camera / "nested"
            folder.mkdir(parents=True)
            (folder / "frame.png").write_bytes(f"{label}-{camera}".encode("utf-8"))
            camera_paths[camera] = tuple(enumerate_detection_images(folder))
        manifest = describe_enumeration(camera_paths, root, "nested", include_digest=True)
        p1_dir = manager.workspace_path / "phase1_runs" / run_id
        p1_dir.mkdir(parents=True)
        p1_pickle = p1_dir / "detected_datapoints.pickle"
        shutil.copyfile(source_pickle, p1_pickle)
        manager.save_run("phase1", run_id, {
            "run_id": run_id,
            "params": {"f_loc": str(root), "subfolder_string": "nested",
                       "selected_cameras": list(camera_names), "n_lim": None,
                       "caching": False},
            "diagnostics": {"education_manifest": manifest,
                            "education_cache": {"reused": False}},
            "artifacts": {"detected_datapoints_pickle": str(p1_pickle)},
        })
        p2_id = f"p2-{label.lower()}"
        p2_dir = manager.workspace_path / "phase2_runs" / p2_id
        p2_dir.mkdir(parents=True)
        p2_camset = p2_dir / "initial_cameras.camset"
        shutil.copyfile(source_camset, p2_camset)
        manager.save_run("phase2", p2_id, {
            "run_id": p2_id, "inputs": {"phase1_run_id": run_id},
            "artifacts": {"initial_camset": str(p2_camset),
                          "education_used_detections": {
                              "schema_version": 1, "source": "phase1_pickle",
                              "path": str(p1_pickle), "fresh_pass_reasons": [],
                              "cache_hit": False}},
        })
        p3_id = f"p3-{label.lower()}"
        p3_dir = manager.workspace_path / "phase3_runs" / p3_id
        p3_dir.mkdir(parents=True)
        p3_camset = p3_dir / "optimised_cameras.camset"
        shutil.copyfile(source_camset, p3_camset)
        if label == "X":
            # The recorded X parent has genuinely different calibration state.
            x_camset = load_CameraSet(p3_camset)
            intrinsic = np.asarray(x_camset.get_cam_dict()["1"].intrinsic).copy()
            intrinsic[0, 0] += 2.0
            x_camset.get_cam_dict()["1"].intrinsic = intrinsic
            from pyCamSet.utils.saving import save_camset
            save_camset(x_camset, p3_camset)
        manager.save_run("phase3", p3_id, {
            "run_id": p3_id,
            "inputs": {"phase1_run_id": run_id, "phase2_run_id": p2_id},
            "artifacts": {"optimised_camset": str(p3_camset),
                          "phase1_detection_pickle_used": str(p1_pickle),
                          "phase2_initial_camset_used": str(p2_camset)},
        })

    p4_dir = manager.workspace_path / "phase4_runs" / "p4-xy"
    p4_dir.mkdir(parents=True)
    p4_camset = p4_dir / "self_calibrated_cameras.camset"
    shutil.copyfile(source_camset, p4_camset)
    used_y = manager.workspace_path / "phase3_runs" / "p3-y" / "optimised_cameras.camset"
    manager.save_run("phase4", "p4-xy", {
        "run_id": "p4-xy", "status": "complete",
        "inputs": {"phase3_run_id": "p3-x"},
        "params": {"f_loc": str(root_y)},
        "diagnostics": {"D4.3_initial_euclid_px": reprojection_segment_mean(
            source.calibration_result,
            len(source.calibration_handler._flat_detections()))},
        "report": {"blocking_flags": []},
        "artifacts": {"self_calibrated_camset": str(p4_camset),
                      "phase3_camset_used": str(used_y)},
    })
    context = RunContext.from_run(manager, manager.find_run("phase4", "p4-xy"))

    def assert_gui_status(expected):
        state = context.phase3_state_source()
        assert state.origin_run_id == "p3-y"
        assert state.source_check.startswith(expected)
        view = education_phase4.Phase4View(context)
        copied = view.stage_text(0)
        assert "Phase 3 run p3-y" in copied
        assert expected in copied
        assert "p3-x" not in copied
        view.close()

    assert_gui_status("verified by manifest")
    # An edit to X's distinct source is a negative control for Y attribution.
    (root_x / camera_names[0] / "nested" / "frame.png").write_bytes(b"changed X only")
    assert_gui_status("verified by manifest")
    # A file outside the manifest's selected nested enumeration is immaterial.
    (root_y / camera_names[0] / "unselected.png").write_bytes(b"outside selected set")
    assert_gui_status("verified by manifest")
    # An edit to the actual-used Y photo must be surfaced in the saved GUI copy.
    (root_y / camera_names[0] / "nested" / "frame.png").write_bytes(b"changed Y")
    assert_gui_status("mismatch")


@pytest.mark.gui
def test_rendered_manifest_and_legacy_cache_copy_do_not_substitute_deleted_photo(
        tmp_path, monkeypatch):
    _app()
    from pyCamSet.calibration_targets.core.abstract_target import enumerate_detection_images
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable
    from pyCamSet.workflow.education_manifest import describe_enumeration
    from pyCamSet.workflow.education_trace import RunContext

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("fixture GUI only"))
    manager, record, _, _ = _saved_phase4_context(tmp_path, used_run_id="p3-cache")
    phase1 = manager.find_run("phase1", "p1")
    image_root = Path(phase1["params"]["f_loc"])
    # Derive camera names from the saved Phase 3 detection layout, not a listing.
    state = RunContext.from_run(manager, record).phase3_state_source()
    from pyCamSet.utils.saving import load_CameraSet
    used_camset = load_CameraSet(state.path)
    camera_names = tuple(used_camset.calibration_handler.detection.cam_names)
    camera_paths = {
        camera: tuple(enumerate_detection_images(image_root / camera))
        for camera in camera_names
    }
    manifest = describe_enumeration(camera_paths, image_root, None, include_digest=True)
    phase1["params"]["caching"] = True
    phase1["diagnostics"] = {
        "education_manifest": manifest,
        "education_cache": {"reused": True},
    }
    manager.save_run("phase1", "p1", phase1)

    context = RunContext.from_run(manager, manager.find_run("phase4", "p4"))
    baseline_state = context.phase3_state_source()
    assert baseline_state.confirmed
    assert baseline_state.source_check.startswith("verified by manifest")
    assert "may have reused saved detections" in baseline_state.source_check
    baseline_view = education_phase4.Phase4View(context)
    baseline_copy = baseline_view.stage_text(0)
    baseline_gauge = _frame(baseline_view, "Reference photo and rig")["copy"]
    assert "verified by manifest" in baseline_copy
    assert "may have reused saved detections" in baseline_copy
    assert "Phase 3 used the 1st photo of the detection run's list" in baseline_gauge
    assert "the 1st photo in the detection run's list" in baseline_gauge
    assert "not necessarily the 1st file in the folder now" in baseline_gauge
    baseline_view.close()

    # A deleted saved-list entry is a real manifest mismatch, not permission to
    # attribute the ordinal to whichever file now sorts into that position.
    deleted = camera_paths[camera_names[0]][0]
    deleted.unlink()
    stale_context = RunContext.from_run(manager, manager.find_run("phase4", "p4"))
    stale_state = stale_context.phase3_state_source()
    assert stale_state.confirmed
    assert stale_state.source_check.startswith("mismatch")
    stale_view = education_phase4.Phase4View(stale_context)
    stale_copy = stale_view.stage_text(0)
    stale_gauge = _frame(stale_view, "Reference photo and rig")["copy"]
    assert "mismatch" in stale_copy
    assert "the detection run's photo list no longer matches the current folder" in stale_gauge
    assert "Phase 3 used the 1st photo of the detection run's list" in stale_gauge
    assert "no current file name is assigned to that pose" in stale_gauge
    assert deleted.name not in stale_gauge
    current_names = tuple(path.name for path in (image_root / camera_names[0]).iterdir())
    assert current_names
    assert all(name not in stale_gauge for name in current_names)
    stale_view.close()


def _generated_view(tmp_path, monkeypatch=None, **state_overrides):
    """A view of a real, generated Phase 4 run (A2r4's fixture: real detection + solves)."""
    from pyCamSet.gui import education_phase4
    from education_p4_fixtures import generate_phase4_run

    camset_path, metadata = generate_phase4_run(tmp_path)
    context = SimpleNamespace(
        run_id=metadata["run_id"], phase="phase4", status="complete",
        run_dir=camset_path.parent, metadata=metadata,
        phase3_state_source=lambda: _state(**state_overrides),
        artifact=lambda phase: camset_path if phase == "phase4" else None,
    )
    return education_phase4.Phase4View(context), camset_path, metadata


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.data
def test_successful_recorded_phase4_playback_renders_saved_endpoints(tmp_path, monkeypatch):
    """Exercise a real Phase 4 trace through the offscreen GUI: the replayed start equals the
    solver's own x0, the replayed end equals the saved result, every step is drawn from its own
    recorded state, and nothing plays until the learner acts."""
    _app()
    from education_p4_fixtures import generate_phase4_run
    from pyCamSet.gui import education_phase4
    from pyCamSet.gui.education_phase4 import _camera_display
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.education_derive import _phase4_rebuild
    from pyCamSet.workflow import phase4
    from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler
    from pyCamSet.optimisation.optimisation_handling import reprojection_residuals

    # Capture x0 from the exact Phase 4 inputs immediately before the fixture's
    # real solve; this oracle is independent of the persisted trace and frames.
    solver_start_evidence = {}
    original_solve = phase4.solve

    def capture_solver_start(previous_cams, target, detections, **kwargs):
        options = dict(kwargs.get("options") or {})
        options.setdefault("target_scale_rule", "measured_edges_v1")
        start_handler = SelfBundleHandler(
            camset=previous_cams, target=target, detection=detections,
            fixed_params=kwargs.get("fixed_params"), options=options)
        start_handler.set_from_templated_camset(previous_cams)
        start_params = np.asarray(start_handler.get_initial_params(), dtype=np.float64)
        initial_residuals, _ = reprojection_residuals(
            start_handler.make_loss_fun(kwargs.get("threads", 1))(start_params),
            start_handler)
        solver_start_evidence["params"] = start_params.copy()
        solver_start_evidence["initial_euclid_px"] = float(np.mean(np.linalg.norm(
            np.asarray(initial_residuals, dtype=np.float64).reshape(-1, 2), axis=1)))
        return original_solve(previous_cams, target, detections, **kwargs)

    monkeypatch.setattr(phase4, "solve", capture_solver_start)
    camset_path, metadata = generate_phase4_run(tmp_path)
    run_dir = camset_path.parent
    context = SimpleNamespace(
        run_id=metadata["run_id"], phase="phase4", status="complete",
        run_dir=run_dir, metadata=metadata,
        phase3_state_source=lambda: _state(),
        artifact=lambda phase: camset_path if phase == "phase4" else None,
    )
    view = education_phase4.Phase4View(context)

    assert view._trace["status"] == "available"
    assert view._derivation.frames
    start, end = view._derivation.frames[0], view._derivation.frames[-1]
    assert start.status == end.status == "available"
    assert start.recorded_start is True
    assert start.nit == 0
    saved = load_CameraSet(camset_path)
    start_params = solver_start_evidence["params"]
    assert solver_start_evidence["initial_euclid_px"] == pytest.approx(
        metadata["diagnostics"]["D4.3_initial_euclid_px"], rel=1e-9, abs=1e-12)
    assert view._derivation.d47_mean_displacement_mm == pytest.approx(
        metadata["diagnostics"]["D4.7_mean_target_displacement_mm"], rel=1e-9)

    # The view replays the same recorded steps, checked against their stored costs.
    playback = view.playback
    assert playback.playable
    assert [state.nit for state in playback.states] == [int(frame.nit) for frame in
                                                        view._derivation.frames]
    np.testing.assert_allclose([state.cost for state in playback.states],
                               [frame.cost for frame in view._derivation.frames], rtol=0, atol=0)
    assert playback.states[0].mean_px == pytest.approx(
        metadata["diagnostics"]["D4.3_initial_euclid_px"], rel=1e-9)
    assert playback.states[-1].mean_px == pytest.approx(
        metadata["diagnostics"]["D4.3_final_euclid_px"], rel=1e-9)

    # Opening is static: no controls until the learner reaches the solver stage.
    assert view.current_stage() == 0
    assert view.step_controls is None
    view.set_stage(_PLAYBACK)
    controls = view.step_controls
    assert controls is not None
    assert not controls._timer.isActive()
    assert controls._scrubber.isEnabled()
    assert controls._next.isEnabled()
    assert controls.at_start and controls.current_step is None
    assert controls._status.text().startswith(
        "Starting point — where the solver began (not an accepted step).")
    assert "Step 0" not in controls._status.text()

    # Oracles: the solver's own start and the saved result, rebuilt independently. Each
    # rebuild gets its own freshly loaded camset: get_camset mutates the Camera objects a
    # handler shares with its camset, so two handlers on one camset would alias.
    oracle_handler = _phase4_rebuild(load_CameraSet(camset_path), metadata["diagnostics"])
    oracle_shape = np.asarray(oracle_handler.get_updated_target(start_params.copy())).reshape(-1, 3)
    oracle_rig, _ = oracle_handler.get_camset(start_params.copy(), return_pose=True)
    np.testing.assert_allclose(playback.states[0].shape, oracle_shape, rtol=0.0, atol=1e-9)
    saved_shape = np.asarray(saved.calibration_handler.get_updated_target(
        np.asarray(saved.calibration_params, dtype=np.float64).copy())).reshape(-1, 3)
    np.testing.assert_allclose(playback.states[-1].shape, saved_shape, rtol=0.0, atol=1e-9)
    final_rig, _ = _phase4_rebuild(load_CameraSet(camset_path), metadata["diagnostics"]).get_camset(
        np.asarray(saved.calibration_params, dtype=np.float64).copy(), return_pose=True)
    for rig, state in ((oracle_rig, playback.states[0]), (final_rig, playback.states[-1])):
        for camera in state.cameras:
            extrinsic = np.asarray(rig[camera.name].extrinsic, dtype=np.float64)
            # World-to-camera rotations agree once both are put in the reference-photo frame.
            if not camera.telecentric:
                np.testing.assert_allclose(
                    camera.centre,
                    _world_to_reference(view, state, np.asarray(rig[camera.name].position)),
                    rtol=0, atol=1e-9)
            assert extrinsic.shape == (4, 4)

    # Stepping draws exactly the recorded state named by the controls.
    Q = view._display_frame()
    for _ in range(min(3, len(controls._steps))):
        controls.step_forward()
        view.player.stop_for_context_change()
        index = controls._index
        state = playback.states[index]
        expected = view._solver_frame(state, Q, view.level())
        drawn = view.split.scene._impl.items["shape"]["verts"]
        np.testing.assert_allclose(drawn, expected["points"]["shape"]["xyz"], rtol=0, atol=1e-9)
        for camera in state.cameras:
            R, _t = _camera_display(Q, camera)
            assert R.shape == (3, 3)
        assert controls._status.text().startswith(f"Step {state.nit} of ")
    stepped = controls.current_step.solver_step
    # Leaving the stage and returning keeps the controls' position.
    view.set_stage(_PLAYBACK - 1)
    view.set_stage(_PLAYBACK)
    assert view.step_controls is controls
    assert controls.current_step.solver_step == stepped
    view.close()


def _world_to_reference(view, state, world_point):
    """A world point in the reference photo's target frame, as the replay draws it."""
    replay = view.visual._replay
    _world_to_cam, target_to_world, _pts = replay.geometry(state.params)
    ref = target_to_world[replay.fixed_pose]
    return (np.linalg.inv(ref) @ np.append(np.asarray(world_point, dtype=float).ravel(), 1.0))[:3]


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.data
def test_saved_once_seen_displacements_render_unexaggerated_with_bound_caption(tmp_path):
    """Display corrections from saved Phase 4 evidence, including a one-view point: the
    several-direction corrections are drawn N x, one-direction ones at 1 x, and the caption,
    subtitle and chip all name the same N."""
    _app()
    from pyCamSet.gui.education_phase4 import MORPH_POSITIONS, _to_display

    view, _camset_path, _metadata = _generated_view(tmp_path)
    try:
        derivation = view._derivation
        assert derivation.status == "available"
        classes = derivation.point_classes
        assert classes.one_direction
        assert derivation.display_refit_points is not None
        assert derivation.display_refit_scale == 1.0  # rigid: the printed size is kept
        nominal, units = view.narrative.nominal_points()
        assert units in {"m", "mm"}
        to_mm = 1000.0 if units == "m" else 1.0
        displayed = derivation.display_refit_points
        once = np.asarray(classes.one_direction, dtype=int)
        raw_once = displayed[once] - nominal[once]
        assert np.all(np.isfinite(raw_once))
        assert np.any(np.linalg.norm(raw_once, axis=1) > 0.0)
        assert np.max(np.linalg.norm(raw_once, axis=1)) * to_mm <= 1.292

        view.set_level(2)
        view.set_stage(1)
        factor = view.morph.exaggeration
        assert factor > 1
        view.morph.go_to(MORPH_POSITIONS - 1)
        Q = view._morph_items["Q"]
        items = view.split.scene._impl.items
        several = np.asarray(classes.several_directions, dtype=int)
        np.testing.assert_allclose(
            items["refined"]["verts"],
            _to_display(Q, nominal[several] + factor * (displayed[several] - nominal[several])),
            rtol=0, atol=1e-9)
        # One-direction corners are never exaggerated (negative control: N x would differ).
        np.testing.assert_allclose(items["refined_one"]["verts"], _to_display(Q, displayed[once]),
                                   rtol=0, atol=1e-9)
        assert not np.allclose(items["refined_one"]["verts"], _to_display(
            Q, nominal[once] + factor * raw_once), rtol=1e-8, atol=1e-10)
        text = view.stage_text()
        assert f"×{factor} longer" in text
        assert f"lines ×{factor} longer" in view.split.subtitle.text()
        assert f"Exaggeration | ×{factor}" in text
        assert "grey: one side only (×1)" in view.split.subtitle.text()
        # Sliding back to the printed design puts every corner on the design.
        view.morph.go_to(0)
        np.testing.assert_allclose(items["refined"]["verts"], _to_display(Q, nominal[several]),
                                   rtol=0, atol=1e-12)
    finally:
        view.close()


@pytest.mark.gui
def test_rendered_untraceable_phase3_source_does_not_name_foreign_folder_photos(
        tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable
    from pyCamSet.workflow.education_trace import RunContext

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("fixture GUI only"))
    manager, record, _, _ = _saved_phase4_context(tmp_path, used_run_id="p3-untraceable")
    record["artifacts"]["phase3_camset_used"] = str(
        tmp_path / "foreign-image-folder" / "unlinked-phase3.camset")
    (tmp_path / "foreign-image-folder").mkdir()
    (tmp_path / "foreign-image-folder" / "unrelated-private-photo.png").write_bytes(b"x")
    context = RunContext.from_run(manager, record)
    state = context.phase3_state_source()
    assert not state.confirmed
    assert state.verification.value == "could not be checked"
    view = education_phase4.Phase4View(context)
    copied = "\n".join(view.stage_text(stage, level=level)
                       for level in (1, 2, 3) for stage in range(6))
    assert "cannot be traced" in copied
    gauge = _frame(view, "Reference photo and rig")["copy"]
    assert "Which photo's pose was fixed for the position frame cannot be checked" in gauge
    assert "The position-frame fix cannot be checked" in compose(
        _frame(view, "Reference photo and rig"), 2)
    assert "unrelated-private-photo.png" not in copied
    assert "unrelated-private-photo.png" not in gauge
    assert "foreign-image-folder" not in copied
    view.close()


def _foreign_typed_start_context(tmp_path: Path, *, foreign_kind: str,
                                 marker: str = "CURRENTFOLDER-PHOTO-9f3a.png"):
    """Build the strongest row-5 case: a real current-folder A2m manifest.

    A Phase 4 run whose recorded start is a *typed* camset from another image
    folder, while this run's own Phase 1 chain carries a real, verifying A2m
    manifest for the current folder. Returns the run context inputs, the manifest
    and the foreign start so the caller can prove the manifest would verify here
    and that the refusal is therefore about the start, not a broken manifest.
    """
    from pyCamSet.calibration_targets.core.abstract_target import enumerate_detection_images
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.education_manifest import describe_enumeration

    manager, record, p3_camset, _p4_camset = _saved_phase4_context(
        tmp_path, used_run_id="p3-current")
    image_root = tmp_path / "images"
    used = load_CameraSet(p3_camset)
    camera_names = tuple(used.calibration_handler.detection.cam_names)
    for name in camera_names:
        (image_root / name / marker).write_bytes(b"current folder only")
    camera_paths = {name: tuple(enumerate_detection_images(image_root / name))
                    for name in camera_names}
    manifest = describe_enumeration(camera_paths, image_root, None, include_digest=True)

    phase1 = dict(manager.find_run("phase1", "p1"))
    phase1["params"] = dict(phase1["params"])
    phase1["params"].update({"subfolder_string": None, "n_lim": None})
    phase1["diagnostics"] = {"education_manifest": manifest,
                            "education_cache": {"reused": False}}
    manager.save_run("phase1", "p1", phase1)

    if foreign_kind == "loose":
        foreign = tmp_path / "foreign-image-folder" / "typed-phase3.camset"
    elif foreign_kind == "another_workspace":
        foreign = (tmp_path / "other-project" / ".pycamset_workspace"
                   / "phase3_runs" / "p3-elsewhere" / "optimised_cameras.camset")
    else:
        raise ValueError(f"unknown foreign_kind {foreign_kind!r}")
    foreign.parent.mkdir(parents=True)
    shutil.copyfile(p3_camset, foreign)

    updated = dict(record)
    updated["artifacts"] = dict(record["artifacts"])
    updated["artifacts"]["phase3_camset_used"] = str(foreign)
    updated["params"] = dict(record.get("params") or {})
    updated["params"]["f_loc"] = str(image_root)
    manager.save_run("phase4", "p4", updated)
    return (manager, manager.find_run("phase4", "p4"), manifest, foreign,
            image_root, camera_names)


def _assert_foreign_manifest_isolated(view, manifest, marker, foreign_token) -> bool:
    """Assert the strongest foreign-input properties a control must be able to break.

    Factored out so the control below drives the SAME assertions against a start
    that resolves to this folder's own camset; if that cannot make this raise,
    the foreign-input regression is vacuous.
    """
    state = view._state
    assert state.confirmed is False
    assert str(state.verification.value) == "could not be checked"
    assert str(state.source_check).startswith("not checkable")
    assert tuple(state.photo_names or ()) == ()
    recorded = [str(item["name"]) for entries in manifest["cameras"].values()
                for item in entries]
    assert marker in "\n".join(recorded), "the manifest must really list the marker"
    # This folder's manifest verdict and its photo names must not be attributed
    # to a run that started from an unconfirmed foreign camset, on any topic.
    # The manifest verdicts are the exact SourceCheck values, checked cell by
    # cell so an unrelated "no mismatch" start-setting cell cannot satisfy or
    # trip the assertion.
    for title, frame in _all_topics(view).items():
        cells = [str(frame.get("copy") or "")]
        for row in frame.get("rows", ()):
            cells.extend(str(cell) for cell in row)
        for cell in cells:
            assert "verified by manifest" not in cell, (title, cell)
            assert cell != "mismatch", (title, cell)
            assert marker not in cell, (title, cell)
            for recorded_name in recorded:
                assert recorded_name not in cell, (title, cell, recorded_name)
    # The learner-facing text never names the foreign folder, this folder's
    # photos, or credits this folder's manifest to the run, on any stage or level.
    surfaces = [view.stage_text(stage, level=level) for level in (1, 2, 3) for stage in range(6)]
    surfaces.append(str(_frame(view, "Reference photo and rig")["copy"]))
    for surface in surfaces:
        assert foreign_token not in surface
        assert marker not in surface
        assert "verified by manifest" not in surface
    assert _frame(view, "Input and quality")["rows"][1][2] == "not checkable"
    gauge_row = _frame(view, "Reference photo and rig")["rows"][1]
    assert gauge_row[1] == "cannot be named"
    assert gauge_row[2] == "not checkable"
    return True


@pytest.mark.gui
@pytest.mark.parametrize("foreign_kind", ["loose", "another_workspace"])
def test_foreign_typed_start_with_a_real_manifest_names_no_current_photo(
        tmp_path, monkeypatch, foreign_kind):
    """Strongest row 5: a foreign typed start over a real current-folder manifest.

    A Phase 4 run started from a *typed* camset in another image folder, with a real A2m
    manifest in THIS folder. That manifest must not be attributed to the foreign start —
    Source reads "not checkable", no current-folder photo is named or shown, and no manifest
    verdict appears on any stage.
    """
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable
    from pyCamSet.workflow.education_manifest import verify
    from pyCamSet.workflow.education_trace import RunContext

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_a, **_k: DerivationUnavailable("fixture GUI only"))
    marker = "CURRENTFOLDER-PHOTO-9f3a.png"
    (manager, record, manifest, _foreign, image_root,
     camera_names) = _foreign_typed_start_context(
        tmp_path, foreign_kind=foreign_kind, marker=marker)

    # The manifest is real: it verifies against this folder on its own, so the
    # refusal below is about the foreign start, not a broken manifest.
    assert verify(manifest, image_root, camera_names, None).status == "verified by manifest"

    foreign_token = ("foreign-image-folder" if foreign_kind == "loose"
                     else "other-project")
    view = education_phase4.Phase4View(RunContext.from_run(manager, record))
    try:
        assert _assert_foreign_manifest_isolated(view, manifest, marker, foreign_token)
    finally:
        view.close()


@pytest.mark.gui
def test_foreign_manifest_regression_fails_when_the_start_resolves_to_this_folder(
        tmp_path, monkeypatch):
    """Negative control: a start that resolves here must break the assertions."""
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable
    from pyCamSet.workflow.education_trace import RunContext

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_a, **_k: DerivationUnavailable("fixture GUI only"))
    marker = "CURRENTFOLDER-PHOTO-9f3a.png"
    (manager, record, manifest, _foreign, _image_root,
     _names) = _foreign_typed_start_context(
        tmp_path, foreign_kind="loose", marker=marker)
    local = (manager.workspace_path / "phase3_runs" / "p3-current"
             / "optimised_cameras.camset")
    updated = dict(record)
    updated["artifacts"] = dict(record["artifacts"])
    updated["artifacts"]["phase3_camset_used"] = str(local)
    updated["inputs"] = {"phase3_run_id": "p3-current"}
    manager.save_run("phase4", "p4", updated)
    control = education_phase4.Phase4View(
        RunContext.from_run(manager, manager.find_run("phase4", "p4")))
    try:
        # Sanity: this start really did resolve to this folder's own camset.
        assert control._state.confirmed is True
        assert control._state.source_check == "verified by manifest"
        with pytest.raises(AssertionError):
            _assert_foreign_manifest_isolated(
                control, manifest, marker, "foreign-image-folder")
    finally:
        control.close()


@pytest.mark.gui
def test_starting_state_only_trace_offers_no_playback_and_says_so(tmp_path, monkeypatch):
    """A trace that kept only the starting state must not promise playback."""
    _app()
    from pyCamSet.gui import education_phase4

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {
        "status": "available", "records": [
            {"kind": "accepted_step", "payload": {"steps": [{"nit": 0, "cost": 9.0}]}}]})
    starting_state_only = SimpleNamespace(
        nit=0, cost=9.0, status="available", rig_cameras=None,
        shape_points=None, point_classes=None)
    derivation = SimpleNamespace(
        frames=(starting_state_only,), point_classes=None,
        d43_final_px=None, display_refit_points=None,
        scale_consistency_ratio=None, metric_qualifier_required=False,
        d47_mean_displacement_mm=None)
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: derivation)
    view = education_phase4.Phase4View(_context(
        tmp_path, metadata={"params": {"record_education": True}, "diagnostics": {}}))

    playback = _frame(view, "Recorded-step playback")
    # The learner must not be told that usable steps were read; the copy falls
    # back to the static-endpoint wording instead.
    assert "Static before/after only." in playback["copy"]
    assert "Recorded accepted solver steps" not in playback["copy"]
    assert playback["rows"] == [
        ("Starting point (not an accepted step)", "9", "—")]
    view.set_stage(_PLAYBACK)
    assert view.step_controls is None
    assert view.player is None
    text = view.stage_text()
    assert text.count("Static before/after only.") == 2  # caption and the Numbers row
    view.close()


@pytest.mark.gui
def test_endpoint_mismatch_keeps_static_endpoints_without_fabricating_a_path(
        tmp_path, monkeypatch):
    """A trace whose endpoints do not match gives no playback and no connecting curve."""
    _app()
    from pyCamSet.gui import education_phase4

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {
        "status": "available", "reason": "saved endpoint mismatch"})
    unavailable_endpoint = SimpleNamespace(
        nit=0, cost=float("nan"), status="unavailable", rig_cameras=None,
        shape_points=None, point_classes=None)
    derivation = SimpleNamespace(
        frames=(unavailable_endpoint,), point_classes=None,
        d43_final_px=None, display_refit_points=None,
        scale_consistency_ratio=None, metric_qualifier_required=True,
        d47_mean_displacement_mm=None)
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: derivation)
    view = education_phase4.Phase4View(_context(
        tmp_path, metadata={"params": {"record_education": True}, "diagnostics": {}}))

    playback = _frame(view, "Recorded-step playback")
    assert "Static before/after only." in playback["copy"]
    assert "saved start and result are not connected by an invented path" in playback["copy"]
    assert playback["rows"] == [
        ("Starting point (not an accepted step)", "unavailable", "—")]
    view.set_stage(_PLAYBACK)
    assert view.step_controls is None
    # No curve is drawn between unverified endpoints.
    figure = view.stage_view.figure()
    assert all(not ax.get_lines() for ax in figure.axes)
    view.close()


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.data
def test_entered_design_size_caveat_renders_for_a_scaled_synthetic_target(tmp_path):
    """Keep the “size not checked” caveat on a target scaled by 0.98."""
    _app()
    from pyCamSet import ChArUco
    from pyCamSet.gui import education_phase4

    design = ChArUco(num_squares_x=20, num_squares_y=20,
                     square_size=4.0 * 0.98, legacy=False)
    camset_path, diagnostics = _find_scaled_target_camset(tmp_path, design)
    state = _state(origin_run_id="scaled-phase3", confirmed=True,
                   verification=SimpleNamespace(value="verified"), reason=None,
                   source_check="verified by manifest", fixed_pose_matches=0,
                   photo_names=("frame0.png",), phase3_before_px=1.0)
    context = SimpleNamespace(
        run_id="scaled-phase4", phase="phase4", status="complete",
        run_dir=camset_path.parent,
        metadata={"run_id": "scaled-phase4", "status": "complete",
                  "phase": "phase4", "params": {"record_education": True},
                  "artifacts": {"self_calibrated_camset": str(camset_path)},
                  "diagnostics": diagnostics},
        phase3_state_source=lambda: state,
        artifact=lambda phase: camset_path if phase == "phase4" else None)
    view = education_phase4.Phase4View(context)
    try:
        assert view._derivation.status == "available", view._derivation.reason
        # A 2% smaller print is inside the gauge's own tolerance, so no mm
        # quantity is qualified and no caption credits the print for a change.
        assert abs(view._derivation.scale_consistency_ratio - 1.0) <= 0.01
        assert not view._derivation.metric_qualifier_required
        assert view._derivation.d47_mean_displacement_mm == pytest.approx(
            diagnostics["D4.7_mean_target_displacement_mm"], rel=1e-9)

        gauge = _frame(view, "Reference photo and rig")
        assert "design you entered" in gauge["copy"]
        assert "overall size is not checked" in gauge["copy"]
        assert "callipers" in gauge["copy"].lower()

        joined = "\n".join(view.stage_text(stage) for stage in range(6))
        assert "overall size is not checked" in joined
        assert "printer" not in joined.lower()
        assert "printed size" not in joined.lower().replace(
            "does not measure the printed size", "")
    finally:
        view.close()


@pytest.mark.gui
def test_display_refit_scale_is_shown_only_at_l3_and_never_labelled_d45(tmp_path, monkeypatch):
    """The size-fitted alignment's scale (not used for display, SCI-1) stays off L1/L2 and
    away from the D4.5 key; at L3 it says what it would do to every face."""
    _app()
    from pyCamSet.gui import education_phase4

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    derivation = SimpleNamespace(
        status="available", reason=None, run_id="run-4", d41_visible_points=272,
        d43_final_px=0.095, d45_gauge_scale=1.0000364,
        d47_mean_displacement_mm=0.1515,
        point_classes=SimpleNamespace(several_directions=(0, 1), one_direction=(2,),
                                      unseen=()),
        display_refit_points=np.zeros((3, 3)), display_refit_scale=1.0,
        display_similarity_scale=1.003682,
        scale_consistency_ratio=1.00299, metric_qualifier_required=False,
        frames=())
    monkeypatch.setattr(education_phase4, "derive_phase4", lambda *_a, **_k: derivation)
    view = education_phase4.Phase4View(_context(
        tmp_path, state=_state(confirmed=True, verification=SimpleNamespace(value="verified"),
                               reason=None, source_check="verified"),
        metadata={"params": {}, "diagnostics": {}}))
    shape_index = _TOPIC_STAGE["Shape and observations"]
    try:
        def shape_rows():
            return [row[0] for row in view.numbers_rows()]

        shown = "×1.004"
        # L1 is the view's own default.
        assert view.level() == 1
        assert view.current_stage() == 0
        text = view.stage_text(shape_index)
        assert shown not in text
        assert "Resizing alignment (not used)" not in shape_rows()

        # L2 still withholds it.
        assert shown not in view.stage_text(level=2)
        assert "Resizing alignment (not used)" not in shape_rows()

        # L3 shows it as one expert row with what it would do (ED2-5: not in the caption).
        copy = view.stage_text(level=3)
        assert "would grow the whole solved target by 0.4 %" in copy
        assert shown not in view.stage_view.caption.text()
        assert copy.count(shown) == 1
        assert shape_rows().count("Resizing alignment (not used)") == 1
        # The saved gauge-scale value is a different number and is never substituted.
        assert "1.00004" not in copy

        # Every stage at L3: only the shape stage carries the number; nothing names D4.5
        # as this value.
        for stage in range(6):
            surface = view.stage_text(stage, level=3)
            if stage != shape_index:
                assert shown not in surface, stage
            assert "d4.5" not in surface.lower(), stage
            assert "printed board's size is" not in surface.lower(), stage

        # Returning to L1 removes it again, so the level gate is not one-shot.
        view.set_stage(shape_index)
        view.set_level(1)
        assert shown not in view.stage_text()
        assert "Resizing alignment (not used)" not in shape_rows()

        # A level change never moves the learner off the stage they were reading (stage 0
        # is index 0 and falsy, so a `currentIndex() or index` fallback would move them).
        for target_level in (2, 3):
            view.set_stage(0)
            view.set_level(target_level)
            assert view.current_stage() == 0
        view.set_stage(shape_index)
        view.set_level(1)
        assert view.current_stage() == shape_index
        view.set_level(3)
        assert view.current_stage() == shape_index
        assert shown in view.stage_text()
    finally:
        view.close()


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.data
def test_real_phase4_run_shows_display_refit_scale_at_l3_only(tmp_path):
    """Show a real run's own display-fit scale at L3, never as the gauge scale."""
    _app()
    view, _camset_path, _metadata = _generated_view(tmp_path)
    try:
        derivation = view._derivation
        assert derivation.status == "available", derivation.reason
        assert derivation.display_refit_scale == 1.0  # rigid display alignment (SCI-1)
        scale = float(derivation.display_similarity_scale)
        gauge_scale = float(derivation.d45_gauge_scale)
        assert np.isfinite(scale) and np.isfinite(gauge_scale)
        assert not np.isclose(scale, gauge_scale, rtol=0.0, atol=1e-9)
        shown = f"×{scale:.4g}"
        withheld = f"{gauge_scale:.5g}"
        shape_index = _TOPIC_STAGE["Shape and observations"]
        assert shown not in view.stage_text(shape_index, level=1)
        assert shown not in view.stage_text(level=2)
        rendered = view.stage_text(level=3)
        assert shown in rendered
        assert "Resizing alignment (not used)" in [row[0] for row in view.numbers_rows()]
        assert "d4.5" not in rendered.lower()
        assert withheld not in rendered
        view.set_level(1)
        assert shown not in view.stage_text()
        assert not any("resizing" in row[0].lower() for row in view.numbers_rows())
    finally:
        view.close()


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.data
@pytest.mark.skipif(
    int(cv2.__version__.split(".")[0]) < 5,
    reason=("OpenCV 5 measurement: OpenCV 4 places sub-pixel ChArUco corners slightly "
            "differently, so its results are not numerically comparable."),
)
def test_real_scale_attribution_qualifies_every_mm_quantity(tmp_path):
    """Residual row 2: the over-tolerance ratio must qualify the mm quantities."""
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.education_trace import RunContext

    (workspace, records, y_run_id, scale_path, scale_diagnostics,
     camset_path, out_fx) = _scale_attribution_fixture(tmp_path)
    y_camset = Path(records["Y"]["artifacts"]["optimised_camset"])
    y_fx = {name: float(np.asarray(cam.intrinsic, dtype=np.float64)[0, 0])
            for name, cam in load_CameraSet(y_camset).get_cam_dict().items()}

    context = RunContext.from_run(
        workspace, workspace.find_run("phase4", "p4-y-started-scaled"))
    state = context.phase3_state_source()
    assert state.origin_run_id == y_run_id
    assert state.confirmed
    assert context.metadata["inputs"]["phase3_run_id"] == records["X"]["run_id"]

    view = education_phase4.Phase4View(context)
    try:
        derivation = view._derivation
        assert derivation.status == "available", derivation.reason
        assert abs(float(derivation.scale_consistency_ratio) - 1.033036) < 5e-6
        assert abs(float(derivation.scale_consistency_ratio) - 1.0) > 0.01
        assert derivation.metric_qualifier_required
        assert derivation.d47_mean_displacement_mm == pytest.approx(
            scale_diagnostics["D4.7_mean_target_displacement_mm"], rel=1e-9)

        for name, delta, detail in view.narrative.rig_comparison_rows():
            assert delta.endswith("mm (scale-qualified)"), (name, delta)
            assert "fx: " in detail and detail.endswith("px; same fixed pose, no fitted alignment")
        gauge = _frame(view, "Reference photo and rig")
        rendered_gauge = "\n".join(
            [str(gauge.get("copy") or "")]
            + [str(cell) for row in gauge.get("rows", ()) for cell in row])
        assert "(scale-qualified)" in rendered_gauge
        assert f"fx: {y_fx['1']:.4g} → {out_fx['1']:.4g} px" in rendered_gauge

        shape = _frame(view, "Shape and observations")
        shape_copy = str(shape.get("copy") or "")
        shape_l2 = compose(shape, 2)
        assert "The scale-consistency ratio is 1.033" in shape_l2
        if view.narrative.refit_headline_shown():
            # The display keeps the solved size (rigid alignment, SCI-1), so the ratio
            # qualifies the headline as it does D4.7.
            assert "outside tolerance, so this millimetre value is qualified" in shape_copy
            assert ("same scale and alignment as the several-direction mean. The "
                    "scale-consistency ratio is 1.033; D4.7 is a qualified millimetre value"
                    ) in shape_l2
        else:
            assert "this millimetre value is qualified" in shape_copy
        assert "D4.7" in shape_l2 and "scale effect" not in shape_l2
        assert "freeing the target shape" not in shape_copy

        error = _frame(view, "Error, loss and decision")
        error_surface = "\n".join(
            [str(error.get(key) or "") for key in ("copy", "detail")]
            + [str(cell) for row in (*error.get("rows", ()), *error.get("rows_detail", ()))
               for cell in row])
        assert "max_nfev" in error_surface
        assert "not attributed to the shape alone" in error_surface
    finally:
        view.close()


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.data
def test_scale_qualifier_regression_fails_when_the_ratio_is_inside_tolerance(tmp_path):
    """Negative control: an inside-tolerance ratio must fail the row-2 assertions."""
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_trace import RunContext

    (workspace, _records, _y_run_id, _scale_path, _scale_diagnostics,
     _camset_path, _out_fx) = _scale_attribution_fixture(tmp_path)
    context = RunContext.from_run(
        workspace, workspace.find_run("phase4", "p4-y-started-scaled"))
    view = education_phase4.Phase4View(context)
    try:
        assert _assert_scale_qualifier(view)
        inside = dataclasses.replace(view._derivation, metric_qualifier_required=False)
        view._derivation = inside
        view.narrative.derivation = inside
        with pytest.raises(AssertionError):
            _assert_scale_qualifier(view)
    finally:
        view.close()


def _assert_scale_qualifier(view) -> bool:
    """Assert the SCALE-ATTRIBUTION properties a control must be able to break."""
    derivation = view._derivation
    assert abs(float(derivation.scale_consistency_ratio) - 1.0) > 0.01
    assert derivation.metric_qualifier_required
    rows = view.narrative.rig_comparison_rows()
    assert rows, "a confirmed pinhole rig comparison must render rows"
    for name, delta, detail in rows:
        assert delta.endswith("mm (scale-qualified)"), (name, delta)
        assert "fx: " in detail
    shape = _frame(view, "Shape and observations")
    shape_copy = str(shape.get("copy") or "")
    if view.narrative.refit_headline_shown():
        assert "D4.7 is a qualified millimetre value" in compose(shape, 2)
    else:
        assert "scale-consistency ratio" in shape_copy
        assert "this millimetre value is qualified" in shape_copy
    assert "freeing the target shape" not in shape_copy
    return True


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.data
def test_real_xy_run_reads_y_in_every_frame_not_the_recorded_x_parent(tmp_path):
    """Residual row 1: every rig/K/fixed-pose/N2 topic must read the used Y."""
    _app()
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.education_trace import RunContext, Verification
    from pyCamSet.gui import education_phase4

    (workspace, x_run_id, y_run_id, y_camset, _, records,
     camset_path) = _bound_xy_records(tmp_path)

    assert records["X"]["status"] == "complete"
    assert (records["X"]["diagnostics"] or {}).get("D3.9_nfev") > 4
    assert records["Y"]["status"] == "incomplete"
    assert (records["Y"]["diagnostics"] or {}).get("D3.9_nfev") == 5
    assert records["Y"]["diagnostics"]["D3.6_final_euclid_px"] > \
        records["X"]["diagnostics"]["D3.6_final_euclid_px"]
    assert x_run_id != y_run_id

    y_fx, x_fx, out_fx = {}, {}, {}
    for name, cam in load_CameraSet(y_camset).get_cam_dict().items():
        y_fx[name] = float(np.asarray(cam.intrinsic, dtype=np.float64)[0, 0])
    for name, cam in load_CameraSet(
            records["X"]["artifacts"]["optimised_camset"]).get_cam_dict().items():
        x_fx[name] = float(np.asarray(cam.intrinsic, dtype=np.float64)[0, 0])
    for name, cam in load_CameraSet(camset_path).get_cam_dict().items():
        out_fx[name] = float(np.asarray(cam.intrinsic, dtype=np.float64)[0, 0])

    assert x_fx["1"] - y_fx["1"] > 1.0
    assert out_fx["1"] - y_fx["1"] > 1.0
    assert out_fx["1"] - x_fx["1"] < -1.0

    context = RunContext.from_run(
        workspace, workspace.find_run("phase4", "p4-x-recorded-y-started"))
    assert context.metadata["inputs"]["phase3_run_id"] == x_run_id
    state = context.phase3_state_source()
    assert state.path == y_camset
    assert state.origin_run_id == y_run_id
    assert state.origin_run_id != x_run_id
    assert state.verification is Verification.VERIFIED
    assert state.confirmed
    assert state.start_metric_px == pytest.approx(
        records["Y"]["diagnostics"]["D3.6_final_euclid_px"], rel=1e-9)
    assert state.loss == "linear"
    assert state.phase3_fixed_pose_ordinal1 == 1
    assert state.phase4_fixed_pose_ordinal1 == 1

    view = education_phase4.Phase4View(context)
    try:
        frames = _all_topics(view)

        def surfaces(title):
            frame = frames[title]
            parts = [str(frame.get(key) or "") for key in ("copy", "detail")]
            for row in (*frame.get("rows", ()), *frame.get("rows_detail", ())):
                parts.extend(str(cell) for cell in row)
            return "\n".join(parts)

        x_fx_text = f"{x_fx['1']:.4g}"
        _assert_xy_binding(view, x_run_id, y_run_id, x_fx_text)
        assert "Phase 3 run" in surfaces("Input and quality")

        gauge = surfaces("Reference photo and rig")
        assert f"fx: {y_fx['1']:.4g} → {out_fx['1']:.4g} px" in gauge
        assert "same fixed pose, no fitted alignment" in gauge
        assert out_fx["1"] - y_fx["1"] > 0
        assert out_fx["1"] - x_fx["1"] < 0
        assert "1st" in gauge
        assert state.rig_comparison_withheld is False

        error = surfaces("Error, loss and decision")
        assert f"{state.start_metric_px:.4g}" in error
        assert "max_nfev" in error
        assert "not attributed to the shape alone" in error
        assert f"{records['X']['diagnostics']['D3.6_final_euclid_px']:.4g}" not in error
    finally:
        view.close()


def _assert_xy_binding(view, x_run_id, y_run_id, x_fx_text):
    """Assert the X/Y binding properties a control must be able to break."""
    frames = _all_topics(view)

    def surfaces(title):
        frame = frames[title]
        parts = [str(frame.get(key) or "") for key in ("copy", "detail")]
        for row in (*frame.get("rows", ()), *frame.get("rows_detail", ())):
            parts.extend(str(cell) for cell in row)
        return "\n".join(parts)

    for title in frames:
        surface = surfaces(title)
        assert f"Phase 3 run {x_run_id}" not in surface, title
        assert x_fx_text not in surface, title
    for stage in range(6):
        rendered = view.stage_text(stage, level=3)
        assert f"Phase 3 run {x_run_id}" not in rendered, stage
    assert y_run_id in surfaces("Input and quality")


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.data
def test_xy_binding_regression_fails_when_the_recorded_parent_is_used(tmp_path):
    """Negative control: the recorded X parent must break the binding assertions."""
    _app()
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.education_trace import RunContext
    from pyCamSet.gui import education_phase4

    (workspace, x_run_id, y_run_id, _, _, records,
     camset_path) = _bound_xy_records(tmp_path)
    x_camset = Path(records["X"]["artifacts"]["optimised_camset"])
    record = workspace.find_run("phase4", "p4-x-recorded-y-started")

    good = education_phase4.Phase4View(RunContext.from_run(workspace, record))
    x_intrinsic = np.asarray(load_CameraSet(x_camset).get_cam_dict()["1"].intrinsic,
                             dtype=np.float64)
    x_fx_text = f"{float(x_intrinsic[0, 0]):.4g}"
    try:
        _assert_xy_binding(good, x_run_id, y_run_id, x_fx_text)
    finally:
        good.close()

    record = dict(record)
    record["artifacts"] = dict(record["artifacts"])
    record["artifacts"]["phase3_camset_used"] = str(x_camset)
    record["artifacts"]["self_calibrated_camset"] = str(camset_path)
    record["_run_dir"] = str(record.get("_run_dir", ""))
    control_view = education_phase4.Phase4View(
        RunContext.from_run(workspace, record))
    try:
        with pytest.raises(AssertionError):
            _assert_xy_binding(control_view, x_run_id, y_run_id, x_fx_text)
    finally:
        control_view.close()


def _sentences(text):
    import re
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]




@pytest.mark.gui
@pytest.mark.parametrize("disposition", [None, "quality check not attributable to the camset used"])
def test_phase4_unconfirmed_start_refusals_end_with_the_a1_recovery_step(tmp_path, monkeypatch, disposition):
    """A1 section 9: every refusal about an unconfirmed starting camset ends with the recovery route."""
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("unconfirmed input"))
    context = _context(tmp_path, state=_state())
    context.quality_disposition = disposition
    view = education_phase4.Phase4View(context)
    recovery = ("In Phase 3, open ‘Diagnostics ▼’, then press ‘Phase 4 - Self-Calibration ▶’ "
                "to start Phase 4 from a traceable Phase 3 result; run Phase 4 again.")
    try:
        assert view.narrative.start_copy() == ("Unavailable: the starting camset cannot be confirmed. "
                                               + recovery)
        assert view.narrative.gauge_copy().endswith(recovery)
        status = view.narrative.status_copy()
        assert "confirmed" in status and status.endswith("; run Phase 4 again."), status
        assert "‘Diagnostics ▼’" in status and "‘Phase 4 - Self-Calibration ▶’" in status
        assert recovery in view.stage_text(0)
    finally:
        view.close()


@pytest.mark.gui
@pytest.mark.parametrize(("final_px", "fit_answer", "consistent"),
                         [(0.8, "Fit: yes, the average point error is lower", True),
                          (2.0, "Fit: no, the average point error is higher", False),
                          (1.5, "Fit: no, the average point error is not lower", False)])
def test_phase4_decision_answers_and_flags_an_error_that_did_not_fall(
        tmp_path, monkeypatch, final_px, fit_answer, consistent):
    """The decision states an Answer line, and a higher error under the same rule
    and a matching start is not called consistent with freeing the shape."""
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import Phase4Derivation

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    classes = SimpleNamespace(unseen=(), one_direction=(), several_directions=(),
                              ray_separation_deg=())
    result = Phase4Derivation(
        status="available", reason=None, run_id="run-4", d41_visible_points=3,
        d43_final_px=final_px, d45_gauge_scale=1.0, d47_mean_displacement_mm=0.2,
        point_classes=classes, display_refit_points=np.zeros((3, 3)),
        display_refit_scale=1.0, scale_consistency_ratio=1.0,
        metric_qualifier_required=False, frames=())
    monkeypatch.setattr(education_phase4, "derive_phase4", lambda *_args, **_kwargs: result)
    state = _state(confirmed=True, verification=SimpleNamespace(value="verified"), reason=None,
                   origin_run_id="p3", phase3_before_px=1.5, source_check="verified by manifest",
                   fixed_pose_matches=True)
    view = education_phase4.Phase4View(_context(tmp_path, state=state))
    try:
        decision = compose(_frame(view, "Error, loss and decision"), 1)
        assert fit_answer in decision
        assert "Shape: not backed, no point was seen from several directions." in decision
        assert ("comes from freeing the target shape together with the lenses"
                in decision) is consistent
        chips = {chip.label: chip for chip in view.stage_view.chips.chips()}
        view.stage_text(_RESULT)
        chips = {chip.label: chip for chip in view.stage_view.chips.chips()}
        assert chips["mean error, Phase 3 → 4"].tone == (
            "good" if consistent else "bad" if final_px > 1.5 else "warn")
        if not consistent:
            assert "the error did not go down" in decision
            assert "run Phase 4 again" in decision
            # A complete run records no blocking flags: never send the learner to absent reasons.
            assert "Recorded warnings or blocking flags: none recorded" in _frame(
                view, "Input and quality")["copy"]
            assert "read the reasons" not in decision.lower()
            assert "reason for stopping is: not recorded" in decision
            assert "‘Record solver steps for this run’" in decision
            assert "'Watching the solver'" in decision
            assert decision.rstrip().endswith("at the last recorded step.")
    finally:
        view.close()


@pytest.mark.gui
@pytest.mark.parametrize(("several", "single", "unseen", "expected", "unqualified"),
                         [((0,), tuple(range(1, 120)), (), "Shape: partly backed, only the 1 of 120 "
                           "points seen from several directions back a shape difference; the rest "
                           "are not backed.", False),
                          ((0, 1, 2), (), (), "Shape: backed, all 3 point(s) were seen from several "
                           "directions.", True)])
def test_phase4_shape_answer_is_proportional_to_the_backed_points(
        tmp_path, monkeypatch, several, single, unseen, expected, unqualified):
    """One several-direction point among many must not read as the whole shape being backed."""
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import Phase4Derivation

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    classes = SimpleNamespace(unseen=unseen, one_direction=single, several_directions=several,
                              ray_separation_deg=())
    count = len(several) + len(single) + len(unseen)
    result = Phase4Derivation(
        status="available", reason=None, run_id="run-4", d41_visible_points=count,
        d43_final_px=0.8, d45_gauge_scale=1.0, d47_mean_displacement_mm=0.2,
        point_classes=classes, display_refit_points=np.zeros((count, 3)),
        display_refit_scale=1.0, scale_consistency_ratio=1.0,
        metric_qualifier_required=False, frames=())
    monkeypatch.setattr(education_phase4, "derive_phase4", lambda *_args, **_kwargs: result)
    state = _state(confirmed=True, verification=SimpleNamespace(value="verified"), reason=None,
                   origin_run_id="p3", phase3_before_px=1.5, source_check="verified by manifest",
                   fixed_pose_matches=True)
    view = education_phase4.Phase4View(_context(tmp_path, state=state))
    try:
        decision = compose(_frame(view, "Error, loss and decision"), 1)
        assert expected in decision
        assert ("Shape: backed" in decision) is unqualified
        view.stage_text(_RESULT)
        chip = next(c for c in view.stage_view.chips.chips()
                    if c.label == "corners well seen (back the shape)")
        assert chip.value == f"{len(several)} of {count}"
        assert chip.tone == ("good" if unqualified else "warn")
    finally:
        view.close()


@pytest.mark.gui
def test_phase4_matching_start_uses_the_stored_phase3_end_as_its_baseline(tmp_path, monkeypatch):
    """Found live (mouse1 experiment_001): a confirmed start with no mismatch began exactly at
    the camset's stored mean, so that is the Phase 3 baseline; unconfirmed, it is never shown."""
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("layout unavailable"))
    state = _state(origin_run_id="phase3-run", confirmed=True,
                   verification=SimpleNamespace(value="verified"), reason=None,
                   start_mismatch=None, phase3_before_px=None, stored_mean_px=2.5)
    view = education_phase4.Phase4View(_context(
        tmp_path, state=state, metadata={"params": {}, "diagnostics": {
            "D4.3_initial_euclid_px": 2.5}}))
    assert "Phase 3 ended at 2.5 px" in _frame(view, "Starting settings")["copy"]
    assert _frame(view, "Error, loss and decision")["rows"][0][1] == "2.5"
    state2 = _state(origin_run_id="phase3-run", confirmed=False,
                    verification=SimpleNamespace(value="differs"), reason="differs",
                    start_mismatch=None, phase3_before_px=None, stored_mean_px=2.5)
    view2 = education_phase4.Phase4View(_context(tmp_path, state=state2))
    assert _frame(view2, "Error, loss and decision")["rows"][0][1] == "unavailable"
    view2.stage_text(_RESULT)
    assert view2.stage_view.chips.chips()[0].value.startswith("unavailable")
    view.close()
    view2.close()


def _find_scaled_target_camset(root: Path, design) -> tuple[Path, dict]:
    """Run a real Phase 4 solve against a 0.98-scaled ChArUco design.

    The saved-camset route cannot express this case: scaling a stored design
    without its square size trips the one-square-spacing guard in
    ``standard_bundle_handler``, so the scaled design has to be present when the
    solve runs. This builder therefore performs the real detection, Phase 3 and
    Phase 4 solves against a target whose printed square size is 2% smaller than
    the 4.0 mm the learner entered, and returns the saved Phase 4 camset.
    """
    from pyCamSet.calibration.camera_calibrator import (
        detect_datapoints_in_imfile, run_initial_calibration,
    )
    from pyCamSet.workflow import phase3, phase4

    corpus = Path(__file__).parent / "test_data" / "calibration_charuco"
    detections, resolutions = detect_datapoints_in_imfile(
        f_loc=corpus, caching=False, calibration_target=design, threads=1)
    phase2 = run_initial_calibration(detections, design, resolutions, save=False)
    phase2.set_resolutions_from_file(floc=corpus)
    _handler, _result, phase3_camset, _stats = phase3.solve(
        phase2, detections, design, fixed_params={},
        options={"outliers": "n", "fixed_pose": 0}, threads=1)
    handler, result, solved, stats = phase4.solve(
        phase3_camset, design, detections, fixed_params={},
        options={"outliers": "n", "fixed_pose": 0}, threads=1,
        record_education=True)
    run_dir = root / "phase4_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    camset_path = run_dir / "self_calibrated_cameras.camset"
    solved.save(camset_path)
    diagnostics = phase4._diagnostics(
        result, handler, stats,
        {"run_id": "scaled-phase3", "status": "complete", "diagnostics": {}},
        lambda _message: None, out_cams=solved, previous_cams=phase3_camset,
        previous_camera_snapshot=phase4._snapshot_camera_parameters(phase3_camset),
        params={"fixed_pose": 0})
    return camset_path, diagnostics


# ---------------------------------------------------------------------------
# Residual row 1: X/Y all-frame source binding.
#
# The card's original route (``phase4.run`` recorded against X but started from
# Y's saved camset) is refused by design since ``18cfd7a`` landed the guard at
# ``pyCamSet/workflow/phase4.py:171-180``; round 6 proved that by execution.
# This is the materially new route: X and Y are real ``phase3.run`` producer
# records (X ``complete``, Y ``incomplete`` from ``max_nfev=4``), the Phase 4
# output is a real ``phase4.solve`` started from Y's saved camset, and only the
# Phase 4 run RECORD is hand-assembled to carry the X-recorded / Y-started pair
# the acceptance case describes. The card permits exactly this ("fixtures built
# inside the test ... with hand-set metadata").
# ---------------------------------------------------------------------------


def _bound_xy_records(base_dir: Path):
    """Build real X/Y Phase 3 records and the X-recorded, Y-started Phase 4 run.

    Returns ``(workspace, x_run_id, y_run_id, y_camset, phase4_record)``.
    The chain is producer-built end to end; only the Phase 4 record's
    ``inputs``/``artifacts`` pairing is authored here.
    """
    from pyCamSet import ChArUco
    from pyCamSet.calibration.camera_calibrator import detect_datapoints_in_imfile
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow import phase1, phase2, phase3, phase4
    from pyCamSet.workflow.education_trace import reprojection_segment_mean
    from pyCamSet.workflow.workspace import WorkspaceManager

    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    images = root / "images"
    if not images.exists():
        shutil.copytree(
            Path(__file__).parent / "test_data" / "calibration_charuco", images)
    # The workspace must be named as such: recorded paths are normalised
    # relative to their nearest ``.pycamset_workspace`` ancestor.
    workspace = WorkspaceManager(root / ".pycamset_workspace")
    target = ChArUco(num_squares_x=20, num_squares_y=20, square_size=4.0,
                     legacy=False)
    quiet = lambda _line: None  # noqa: E731
    common = {"f_loc": str(images), "target": {"type": "ChArUco",
              "num_squares_x": 20, "num_squares_y": 20, "square_size": 4,
              "legacy": False}, "selected_cameras": [], "n_lim": None,
              "caching": False}
    first = phase1.run({**common, "upscale_factor": 1,
                        "rescale_and_gamma": False, "manifest_digest": True},
                       workspace, log=quiet)
    second = phase2.run({**common, "high_distortion": False,
                         "fixed_params": None, "min_detections_per_board": 12,
                         "lens_model": "pinhole"},
                        workspace, log=quiet, phase1_run=first)

    records = {}
    for label, options in (("X", {"outliers": "n", "fixed_pose": 0}),
                           ("Y", {"outliers": "n", "fixed_pose": 0,
                                  "max_nfev": 4})):
        records[label] = phase3.run(
            {**common, "fixed_params": {}, "threads": 1,
             "problem_options": dict(options)},
            workspace, log=quiet, phase2_run=second, phase1_run=first)

    x_run_id = records["X"]["run_id"]
    y_run_id = records["Y"]["run_id"]
    y_camset = Path(records["Y"]["artifacts"]["optimised_camset"])

    # A real Phase 4 solve started from Y's real saved camset. The producer
    # refuses to RECORD this as an X-parented run, so the record is authored
    # by hand below; the solve itself and its output camset are genuine.
    detections, _resolutions = detect_datapoints_in_imfile(
        f_loc=images, caching=False, calibration_target=target, threads=1)
    y_cams = load_CameraSet(y_camset)
    handler, result, solved, stats = phase4.solve(
        y_cams, target, detections, fixed_params={},
        options={"outliers": "n", "fixed_pose": 0}, threads=1,
        record_education=True)
    run_id = "p4-x-recorded-y-started"
    run_dir = workspace.run_dir("phase4", run_id)
    camset_path = run_dir / "self_calibrated_cameras.camset"
    solved.save(camset_path)
    diagnostics = phase4._diagnostics(
        result, handler, stats,
        {"run_id": "y-parent", "status": "complete", "diagnostics": {}},
        quiet, out_cams=solved, previous_cams=y_cams,
        previous_camera_snapshot=phase4._snapshot_camera_parameters(y_cams),
        params={"fixed_pose": 0})
    y_rows = y_cams.calibration_handler._flat_detections()
    diagnostics["D4.3_initial_euclid_px"] = reprojection_segment_mean(
        y_cams.calibration_result, len(y_rows))
    workspace.save_run("phase4", run_id, {
        "run_id": run_id, "status": "complete",
        "inputs": {"phase3_run_id": x_run_id},
        "params": {"f_loc": str(images),
                   "problem_options": {"outliers": "n", "fixed_pose": 0}},
        "diagnostics": diagnostics,
        "report": {"blocking_flags": []},
        "artifacts": {"self_calibrated_camset": str(camset_path),
                      "phase3_camset_used": str(y_camset)},
    })
    return (workspace, x_run_id, y_run_id, y_camset, x_run_id,
            records, camset_path)


# ---------------------------------------------------------------------------
# Residual row 2: real out-of-tolerance scale attribution.
#
# The card names A2r4's single-observation-displacement fixture
# (``tests/education_p4_fixtures.py``) as the route and requires the cell to be
# skipped with its stated reason only when the helper's own precondition fails.
# Round 7 measured (reviewer run #223) that the precondition fires only for the
# UNDRIVEN form: ``displace_once_seen=True`` alone raises, while
# ``legacy_scale_rule=True, displace_once_seen=True`` builds and drives the
# scale-consistency ratio to 1.033036, outside the 0.01 tolerance.
#
# The one leg that is still missing is provenance, not a fixture precondition:
# the helper's own recorded Phase 4 start metric is the bundled initial residual
# (1.3053232568515671 px), while the camset it stores as the start has its own
# stored residual mean (0.7696766713402885 px), so A2's start-confirmation leg
# reads ``confirmed=False`` and E2 renders an unconfirmed start instead of the
# scale-qualified attribution. The card permits rebuilding a fixture with
# hand-set metadata, and run #222 already set one measured field for row 1; this
# cell does the same, recording ``D4.3_initial_euclid_px`` from the used camset's
# own measured residual mean.
# ---------------------------------------------------------------------------


def _scale_attribution_fixture(base_dir: Path):
    """Build real X/Y Phase 3 records and a Phase 4 run started from the scaled Y.

    Returns ``(workspace, records, y_run_id, scale_path, scale_diagnostics,
    camset_path, out_fx)``. X is the recorded Phase 3 parent; Y is the Phase 3 run
    whose saved camset the Phase 4 run actually started from, and it carries the
    one extra saved option (``max_nfev: 4``) that makes A2's settings-difference
    clause attributable to a real saved cause.
    """
    from pyCamSet import ChArUco
    from pyCamSet.calibration.camera_calibrator import detect_datapoints_in_imfile
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow import phase1, phase2, phase3, phase4
    from pyCamSet.workflow.education_trace import reprojection_segment_mean
    from pyCamSet.workflow.workspace import WorkspaceManager
    from education_p4_fixtures import generate_phase4_run

    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    images = root / "images"
    if not images.exists():
        shutil.copytree(
            Path(__file__).parent / "test_data" / "calibration_charuco", images)
    workspace = WorkspaceManager(root / ".pycamset_workspace")
    target = ChArUco(num_squares_x=20, num_squares_y=20, square_size=4.0,
                     legacy=False)
    quiet = lambda _line: None  # noqa: E731
    common = {"f_loc": str(images), "target": {"type": "ChArUco",
              "num_squares_x": 20, "num_squares_y": 20, "square_size": 4,
              "legacy": False}, "selected_cameras": [], "n_lim": None,
              "caching": False}
    first = phase1.run({**common, "upscale_factor": 1,
                        "rescale_and_gamma": False, "manifest_digest": True},
                       workspace, log=quiet)
    second = phase2.run({**common, "high_distortion": False,
                         "fixed_params": None, "min_detections_per_board": 12,
                         "lens_model": "pinhole"},
                        workspace, log=quiet, phase1_run=first)

    records = {}
    for label, options in (("X", {"outliers": "n", "fixed_pose": 0}),
                           ("Y", {"outliers": "n", "fixed_pose": 0,
                                  "max_nfev": 4})):
        records[label] = phase3.run(
            {**common, "fixed_params": {}, "threads": 1,
             "problem_options": dict(options)},
            workspace, log=quiet, phase2_run=second, phase1_run=first)
    y_camset = Path(records["Y"]["artifacts"]["optimised_camset"])

    # The over-tolerance shape change, from A2r4's own helper: once-seen points
    # are displaced along their observation rays under the legacy scale rule,
    # which is the pre-E1 configuration E2's gate names.
    scale_path, scale_metadata = generate_phase4_run(
        root / "scaled", legacy_scale_rule=True, displace_once_seen=True)
    scale_diagnostics = dict(scale_metadata.get("diagnostics") or {})
    y_cams = load_CameraSet(y_camset)

    # A real Phase 4 solve is NOT usable as the output here: the derivation gate
    # recomputes D4.5/D4.7 from the OUTPUT camset's own rebuild and requires them
    # within 1e-9 of the saved values, so overlaying the fixture's displaced
    # parameters on a Y-started solve is refused ("recalculated D4.5 does not
    # match the saved result", measured this run). The output is therefore the
    # fixture's own solved camset, copied into the workspace run directory, and
    # the start is recorded from Y's own measured residual mean.
    run_id = "p4-y-started-scaled"
    run_dir = workspace.run_dir("phase4", run_id)
    camset_path = run_dir / "self_calibrated_cameras.camset"
    camset_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(scale_path, camset_path)
    y_rows = y_cams.calibration_handler._flat_detections()
    start_metric = reprojection_segment_mean(
        y_cams.calibration_result, len(y_rows))
    # Only the start metric is authored here, and it is Y's own measured value —
    # the same hand-set-metadata technique run #222 used for residual row 1. The
    # producer's own record of this pairing is refused by the guard at
    # ``pyCamSet/workflow/phase4.py:171-180`` (``18cfd7a``), which is why the
    # record is assembled; every saved artefact it names is producer-built.
    workspace.save_run("phase4", run_id, {
        "run_id": run_id, "status": "complete",
        "inputs": {"phase3_run_id": records["X"]["run_id"]},
        "params": {"f_loc": str(images), "record_education": True,
                   "problem_options": {"outliers": "n", "fixed_pose": 0}},
        "diagnostics": {**scale_diagnostics,
                        "D4.3_initial_euclid_px": start_metric},
        "report": {"blocking_flags": []},
        "artifacts": {"self_calibrated_cameras": str(camset_path),
                      "self_calibrated_camset": str(camset_path),
                      "phase3_camset_used": str(y_camset)},
    })
    out_fx = {}
    for name, cam in load_CameraSet(camset_path).get_cam_dict().items():
        out_fx[name] = float(np.asarray(cam.intrinsic, dtype=np.float64)[0, 0])
    return (workspace, records, records["Y"]["run_id"], scale_path,
            scale_diagnostics, camset_path, out_fx)

