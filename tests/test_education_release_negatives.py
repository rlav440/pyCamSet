'''Exercise release-negative and corruption cases for education mode.
'''
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY = REPO_ROOT / "tests" / "test_data" / "education" / "legacy"


def _app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _phase_context(tmp_path: Path, phase: str, metadata: dict | None = None):
    """Create a minimal run-shaped context with an isolated run directory."""
    run_id = f"{phase}-fixture"
    run_dir = tmp_path / ".pycamset_workspace" / f"{phase}_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    record = dict(metadata or {})
    record.setdefault("run_id", run_id)
    record.setdefault("phase", phase)
    return SimpleNamespace(run_id=run_id, phase=phase, run_dir=run_dir,
                           status="complete", metadata=record,
                           manager=SimpleNamespace(workspace_path=run_dir.parents[1]),
                           artifact=lambda _phase=None: None)


@pytest.mark.gui
def test_no_data_opening_each_phase_explanation_is_safe(tmp_path, monkeypatch):
    """Each phase opens an explanation for an empty legacy record."""
    _app()
    from pyCamSet.gui import education_phase1, education_phase2
    from pyCamSet.gui import education_phase3, education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable

    monkeypatch.setattr(education_phase1, "_load_run_detections",
                        lambda _context: (None, "this run's detections were not saved"))
    monkeypatch.setattr(education_phase1, "_source_state", lambda _context: (
        SimpleNamespace(status="source identity not recorded", reason="no manifest"), {}, "layout"))
    monkeypatch.setattr(education_phase2, "read_trace", lambda *_args: {"status": "unavailable"})
    monkeypatch.setattr(education_phase3, "read_trace", lambda *_args: {"status": "unavailable"})
    monkeypatch.setattr(education_phase3, "derive_phase3",
                        lambda _context: DerivationUnavailable("no saved data"))
    monkeypatch.setattr(education_phase4, "read_trace", lambda *_args: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("no saved data"))

    views = [
        education_phase1.Phase1EducationView(_phase_context(tmp_path, "phase1")),
        education_phase2.Phase2View(_phase_context(tmp_path, "phase2")),
        education_phase3.Phase3View(_phase_context(tmp_path, "phase3")),
        education_phase4.Phase4View(_phase_context(tmp_path, "phase4")),
    ]
    try:
        assert [view.context.phase for view in views] == ["phase1", "phase2", "phase3", "phase4"]
        assert all(view is not None for view in views)
    finally:
        for view in views:
            view.close()


@pytest.mark.parametrize("phase,view_name", [
    ("phase2", "Phase2View"), ("phase3", "Phase3View"), ("phase4", "Phase4View"),
])
@pytest.mark.gui
def test_promoted_run_is_not_presented_as_a_normal_quality_pass(
        tmp_path, monkeypatch, phase, view_name):
    """A promoted run has no normal per-phase quality acceptance."""
    _app()
    from pyCamSet.gui import education_phase2, education_phase3, education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable

    context = _phase_context(tmp_path, phase, {
        "run_id": f"{phase}-promoted", "diagnostics": {"promoted_from": "trial-1"},
        "promoted_from": "trial-1", "status": "complete", "params": {},
    })
    if phase == "phase2":
        module = education_phase2
        view_type = getattr(module, view_name)
        monkeypatch.setattr(module, "read_trace", lambda *_args: {"status": "unavailable"})
    elif phase == "phase3":
        module = education_phase3
        view_type = getattr(module, view_name)
        monkeypatch.setattr(module, "read_trace", lambda *_args: {"status": "unavailable"})
        monkeypatch.setattr(module, "derive_phase3", lambda _context: DerivationUnavailable("promoted"))
    else:
        module = education_phase4
        view_type = getattr(module, view_name)
        monkeypatch.setattr(module, "read_trace", lambda *_args: {"status": "unavailable"})
        monkeypatch.setattr(module, "derive_phase4",
                            lambda *_args, **_kwargs: DerivationUnavailable("promoted"))

    view = view_type(context)
    try:
        rendered = " ".join(str(frame) for frame in getattr(view, "_frames", ()))
        if hasattr(view, "copy"):
            rendered += " " + view.copy.text()
        if hasattr(view, "stage_text"):  # Phase 4's visual view: the stage's rendered text
            rendered += " " + view.stage_text()
        assert "promoted_from" in str(context.metadata)
        if phase == "phase2":
            assert "unmarked Optimisation-tab trial" in rendered
        elif phase == "phase3":
            assert "normal Phase 3 quality check was not run" in rendered
            assert "not shown as a normal quality pass" in rendered
        else:
            # Phase 4's derivation is deliberately unavailable without saved
            # evidence; a promoted status cannot manufacture missing metrics.
            assert "unavailable" in rendered.lower() or "not recorded" in rendered.lower()
    finally:
        view.close()


@pytest.mark.gui
def test_corrupted_legacy_trace_fails_closed_and_shows_copy(tmp_path, monkeypatch):
    """Truncated sidecar JSON degrades to the no-recorded-steps explanation."""
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable
    from pyCamSet.workflow.education_trace import read_trace, trace_path

    context = _phase_context(tmp_path, "phase4", {"params": {}, "diagnostics": {}})
    path = trace_path(context.run_dir)
    path.write_text('{"schema_version": 1, "records": [', encoding="utf-8")
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("legacy evidence unavailable"))
    trace = read_trace(context.run_dir, context.run_id, "phase4")
    assert trace["status"] == "unavailable"
    view = education_phase4.Phase4View(context)
    try:
        # 'Watching the solver' (stage 3) renders the playback statement.
        playback = view.stage_text(2)
        assert "not recorded for this run" in playback or "unavailable" in playback.lower()
    finally:
        view.close()


def test_corrupted_legacy_missing_artifact_fails_closed(tmp_path, monkeypatch):
    """A legacy run with no saved camset cannot be rebuilt or borrowed from elsewhere."""
    from pyCamSet.workflow.education_derive import DerivationUnavailable
    from pyCamSet.workflow.education_derive import derive_phase4
    from pyCamSet.workflow.education_trace import RunContext
    from pyCamSet.workflow.workspace import WorkspaceManager

    workspace = WorkspaceManager(tmp_path / ".pycamset_workspace")
    run_dir = workspace.run_dir("phase4", "legacy-missing")
    run_dir.mkdir(parents=True, exist_ok=True)
    run = {"run_id": "legacy-missing", "status": "complete", "phase": "phase4",
           "artifacts": {}, "diagnostics": {}}
    workspace.save_run("phase4", "legacy-missing", run)
    context = RunContext.from_run(workspace, workspace.find_run("phase4", "legacy-missing"))
    result = derive_phase4(context)
    assert isinstance(result, DerivationUnavailable)
    assert not any(path.name == "self_calibrated_cameras.camset" for path in run_dir.iterdir())


@pytest.mark.gui
@pytest.mark.data
def test_image_mismatch_phase4_uses_y_manifest_not_recorded_x(tmp_path, monkeypatch):
    """Editing X cannot taint Y's frames; editing Y changes their verdict."""
    _app()
    from pyCamSet.calibration_targets.core.abstract_target import enumerate_detection_images
    from pyCamSet.gui import education_phase4
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.education_derive import DerivationUnavailable
    from pyCamSet.workflow.education_manifest import describe_enumeration
    from pyCamSet.workflow.education_trace import RunContext, reprojection_segment_mean
    from pyCamSet.workflow.workspace import WorkspaceManager

    monkeypatch.setattr(education_phase4, "read_trace", lambda *_args: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("GUI provenance only"))
    source = load_CameraSet(LEGACY / "phase4_325fae4.camset.json")
    camera_names = tuple(source.calibration_handler.detection.cam_names)
    manager = WorkspaceManager(tmp_path / ".pycamset_workspace")
    roots = {label: tmp_path / f"photos-{label}" for label in ("X", "Y")}

    for label, root in roots.items():
        camera_paths = {}
        for camera in camera_names:
            folder = root / camera / "nested"
            folder.mkdir(parents=True)
            (folder / "frame.png").write_bytes(f"{label}-{camera}".encode("utf-8"))
            camera_paths[camera] = tuple(enumerate_detection_images(folder))
        manifest = describe_enumeration(camera_paths, root, "nested", include_digest=True)
        p1_id, p2_id, p3_id = (f"p1-{label.lower()}", f"p2-{label.lower()}",
                               f"p3-{label.lower()}")
        p1_dir = manager.run_dir("phase1", p1_id)
        p1_pickle = p1_dir / "detected_datapoints.pickle"
        shutil.copyfile(LEGACY / "phase1_325fae4.pickle", p1_pickle)
        manager.save_run("phase1", p1_id, {
            "run_id": p1_id, "params": {"f_loc": str(root), "subfolder_string": "nested",
                                           "selected_cameras": list(camera_names), "n_lim": None,
                                           "caching": False},
            "diagnostics": {"education_manifest": manifest,
                            "education_cache": {"reused": False}, "cam_names": list(camera_names)},
            "artifacts": {"detected_datapoints_pickle": str(p1_pickle)},
        })
        p2_dir = manager.run_dir("phase2", p2_id)
        p2_camset = p2_dir / "initial_cameras.camset"
        shutil.copyfile(LEGACY / "phase4_325fae4.camset.json", p2_camset)
        manager.save_run("phase2", p2_id, {
            "run_id": p2_id, "inputs": {"phase1_run_id": p1_id},
            "artifacts": {"initial_camset": str(p2_camset),
                          "education_used_detections": {"schema_version": 1,
                              "source": "phase1_pickle", "path": str(p1_pickle),
                              "fresh_pass_reasons": [], "cache_hit": False}},
        })
        p3_dir = manager.run_dir("phase3", p3_id)
        p3_camset = p3_dir / "optimised_cameras.camset"
        shutil.copyfile(LEGACY / "phase4_325fae4.camset.json", p3_camset)
        manager.save_run("phase3", p3_id, {
            "run_id": p3_id, "inputs": {"phase1_run_id": p1_id, "phase2_run_id": p2_id},
            "artifacts": {"optimised_camset": str(p3_camset),
                          "phase1_detection_pickle_used": str(p1_pickle),
                          "phase2_initial_camset_used": str(p2_camset)},
        })

    p4_dir = manager.run_dir("phase4", "p4-xy")
    p4_camset = p4_dir / "self_calibrated_cameras.camset"
    shutil.copyfile(LEGACY / "phase4_325fae4.camset.json", p4_camset)
    used_y = manager.run_dir("phase3", "p3-y") / "optimised_cameras.camset"
    manager.save_run("phase4", "p4-xy", {
        "run_id": "p4-xy", "status": "complete", "inputs": {"phase3_run_id": "p3-X"},
        "params": {"f_loc": str(roots["Y"])},
        "diagnostics": {"D4.3_initial_euclid_px": reprojection_segment_mean(
            source.calibration_result, len(source.calibration_handler._flat_detections()))},
        "report": {"blocking_flags": []},
        "artifacts": {"self_calibrated_camset": str(p4_camset), "phase3_camset_used": str(used_y)},
    })
    context = RunContext.from_run(manager, manager.find_run("phase4", "p4-xy"))

    def check(expected):
        state = context.phase3_state_source()
        assert state.origin_run_id == "p3-y"
        assert state.source_check.startswith(expected)
        view = education_phase4.Phase4View(context)
        try:
            visible = view.stage_text(0)
            assert "Phase 3 run p3-y" in visible and "p3-x" not in visible
            assert expected in visible
        finally:
            view.close()

    check("verified by manifest")
    (roots["X"] / camera_names[0] / "nested" / "frame.png").write_bytes(b"edit X")
    check("verified by manifest")
    (roots["Y"] / camera_names[0] / "nested" / "frame.png").write_bytes(b"edit Y")
    check("mismatch")


@pytest.mark.gui
def test_wrong_frame_selection_never_borrows_neighbouring_camera_step(tmp_path, monkeypatch):
    """A detector trace for one camera/image is not reused for another slot."""
    _app()
    from pyCamSet.gui import education_phase1

    context = _phase_context(tmp_path, "phase1", {"params": {}, "diagnostics": {"cam_names": ["camA", "camB"]}})
    monkeypatch.setattr(education_phase1, "_load_run_detections",
                        lambda _context: (None, "no saved detections"))
    monkeypatch.setattr(education_phase1, "_source_state", lambda _context: (
        SimpleNamespace(status="not checkable", reason="fixture"), {}, "layout"))
    view = education_phase1.Phase1EducationView(context)
    view._trace = {"records": [{"kind": "detector_stage", "payload": {
        "camera": "camA", "image_index": 0, "stage": "accepted"}}]}
    try:
        stages, _caption = view._trace_stages_for_selection("camB", 0)
        assert stages == []
    finally:
        view.close()


def test_wrong_gauge_selection_fails_legacy_layout_validation(tmp_path):
    """A legacy Phase 4 parameter layout with a different gauge is refused."""
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.education_derive import DerivationUnavailable, derive_phase4
    from pyCamSet.workflow.education_trace import RunContext
    from pyCamSet.workflow.workspace import WorkspaceManager

    legacy_camset = LEGACY / "phase4_325fae4.camset.json"
    camera_set = load_CameraSet(legacy_camset)
    run_id = "wrong-gauge"
    workspace = WorkspaceManager(tmp_path / ".pycamset_workspace")
    run_dir = workspace.run_dir("phase4", run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    from pyCamSet.utils.saving import save_camset
    saved = run_dir / "self_calibrated_cameras.camset"
    save_camset(camera_set, saved)
    diagnostics = {"D4.2_gauge_fixed_points": {"count": 3, "indices": [999, 1000, 1001]},
                   "D4.5_gauge_scale_factor": 1.0}
    run = {"run_id": run_id, "phase": "phase4", "status": "complete",
           "diagnostics": diagnostics, "artifacts": {"self_calibrated_camset": str(saved)}}
    context = RunContext.from_run(workspace, {**run, "_run_dir": str(run_dir)})
    result = derive_phase4(context)
    assert isinstance(result, DerivationUnavailable)
    assert "layout" in result.reason.lower() or "gauge" in result.reason.lower()


@pytest.mark.gui
def test_no_source_mutation_hashes_sources_and_saved_run(tmp_path, monkeypatch):
    """Opening a saved explanation leaves source bytes and run bytes unchanged."""
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable

    source_paths = [REPO_ROOT / "pyCamSet" / "workflow" / name
                    for name in ("education_trace.py", "education_manifest.py", "education_derive.py")]
    source_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    context = _phase_context(tmp_path, "phase4", {"params": {}, "diagnostics": {}})
    saved = context.run_dir / "self_calibrated_cameras.camset"
    saved.write_bytes(b"fixture saved-run artifact\x00")
    run_hash = hashlib.sha256(saved.read_bytes()).hexdigest()
    monkeypatch.setattr(education_phase4, "read_trace", lambda *_args: {"status": "unavailable"})
    monkeypatch.setattr(education_phase4, "derive_phase4",
                        lambda *_args, **_kwargs: DerivationUnavailable("read-only fixture"))
    view = education_phase4.Phase4View(context)
    view.close()
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths} == source_hashes
    assert hashlib.sha256(saved.read_bytes()).hexdigest() == run_hash


@pytest.mark.gui
def test_old_build_preferences_loads_untouched_preferences_json(tmp_path, monkeypatch):
    """An old strict preferences document validates without education fields."""
    _app()
    from pyCamSet.gui import preferences

    config = tmp_path / "config"
    config.mkdir()
    strict_path = config / "preferences.json"
    old_document = {"schema": "pycamset.gui-preferences", "version": 1,
                    "preferences": {"info_enabled": True, "terminal_visible": True,
                                    "export_presets": {}}}
    original = json.dumps(old_document, indent=2).encode("utf-8")
    strict_path.write_bytes(original)
    monkeypatch.setattr(preferences, "config_directory", lambda: config)
    loaded = preferences.Preferences()
    assert not loaded.load_error
    assert strict_path.read_bytes() == original


def test_old_build_preferences_absent_education_sidecar_is_off(tmp_path, monkeypatch):
    """The absent user sidecar defaults education mode off and stays absent."""
    from pyCamSet.gui import education_preferences, preferences

    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setattr(preferences, "config_directory", lambda: config)
    sidecar = config / "education_preferences.json"
    assert not sidecar.exists()
    assert education_preferences.education_mode_enabled() is False
    assert not sidecar.exists()
