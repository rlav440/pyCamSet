'''Verify the Phase 3 education storyboard fails closed and follows run evidence.
'''
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


def _app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _context(tmp_path: Path, metadata: dict, *, status: str = "complete"):
    run_dir = tmp_path / "phase3_runs" / "phase3-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        run_id="phase3-run", phase="phase3", run_dir=run_dir,
        metadata=metadata, status=status, quality_check="unchecked",
        artifact=lambda _phase=None: None,
        verify_input=lambda _phase: SimpleNamespace(
            verification=SimpleNamespace(value="could not be checked"), parent_run_id=None),
    )


def _frame(view, title: str):
    """One saved-evidence block (the Numbers drawer, chips and captions read these)."""
    return next(frame for frame in view._evidence if frame["title"] == title)


def _table_cells(view) -> list[str]:
    table = view.stage_view.numbers.table
    return [table.item(row, column).text() for row in range(table.rowCount())
            for column in range(table.columnCount()) if table.item(row, column) is not None]


def compose(frame, level: int) -> str:
    """Render a frame's copy as the view shows it at one detail level."""
    from pyCamSet.gui.education_levels import compose as compose_levels

    return compose_levels(frame, level)


@pytest.mark.gui
def test_phase3_storyboard_has_its_stages_evidence_and_absent_trace_copy(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase3
    from pyCamSet.workflow.education_derive import DerivationUnavailable

    monkeypatch.setattr(education_phase3, "derive_phase3",
                        lambda _context: DerivationUnavailable("input not verified"))
    monkeypatch.setattr(education_phase3, "read_trace",
                        lambda *_args: {"status": "unavailable", "reason": "missing sidecar"})
    view = education_phase3.Phase3View(_context(tmp_path, {"params": {}}))
    assert len(view._evidence) == 6
    assert view.stage_keys() == ["graph", "first", "solver", "result"]
    assert view.stage_view.current_stage() == 0
    assert "current file is not proof of what the run used" in _frame(view, "Input chain")["copy"]
    assert "rerun Phase 3" in _frame(view, "Input chain")["copy"]
    assert "not recorded for this run" in _frame(view, "Recorded-step playback")["copy"]
    # Quote the checkbox exactly as the Phase 3 settings show it.
    assert "tick ‘Record solver steps for this run’" in _frame(view, "Recorded-step playback")["copy"]
    # Stage 1's Numbers drawer holds the input chain and the lens-estimate input's check
    # (no join order was recorded).
    assert view.stage_view.numbers.table.rowCount() == 4
    assert "Lens estimates (Phase 2) input" in _table_cells(view)
    # Every stage still draws a figure (an explanatory empty panel without data).
    for index in range(len(view.stage_keys())):
        view.stage_view.set_current_stage(index)
        assert view.stage_view.figure().axes
    view.close()


@pytest.mark.gui
def test_opted_in_run_uses_production_record_education_param(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase3
    from pyCamSet.workflow.education_derive import DerivationUnavailable
    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _: DerivationUnavailable("missing"))
    monkeypatch.setattr(education_phase3, "read_trace", lambda *_: {"status": "available", "records": []})
    view = education_phase3.Phase3View(_context(tmp_path, {"params": {"record_education": True}}))
    assert view._playback["reason"] == "saved start or result is unavailable"
    view.close()


@pytest.mark.gui
def test_d33_weighted_sum_is_hidden_until_l3(tmp_path, monkeypatch):
    _app()
    from dataclasses import replace
    from pyCamSet.gui import education_phase3
    derivation = replace(_available_derivation(), seed_detection_count_by_image=(3,))
    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _: derivation)
    monkeypatch.setattr(education_phase3, "read_trace", lambda *_: {"status": "unavailable"})
    view = education_phase3.Phase3View(_context(tmp_path, {"params": {}}))
    view.show_stage("first")
    assert not any("summed error" in row[0].lower() or "detections (d3.3 n)" in row[0].lower()
                   for row in view._seed_rows())
    view.stage_view.set_level(3)
    rows = view._seed_rows()
    assert any(row[0] == "D3.3 summed error, px (L3)" for row in rows)
    assert any("detections (D3.3 N)" in row[0] for row in rows)
    # Counts and sums must not sit under a mean-pixel-error heading.
    assert _frame(view, "First guess and solver start")["headers"][1] == "Value"
    view.stage_view.set_level(2)
    assert not any("summed error" in row[0].lower() for row in view._seed_rows())
    view.close()


@pytest.mark.gui
def test_unmarked_promoted_phase2_input_is_named_as_untrusted_trial(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase3
    from pyCamSet.workflow.education_trace import Verification

    context = _context(tmp_path, {"params": {}, "artifacts": {
        "phase2_initial_camset_used": "phase2_runs/trial/initial_cameras.camset"}})
    context.manager = SimpleNamespace(find_run=lambda *_: {
        "run_id": "trial", "diagnostics": {"promoted_from": "study-1"}, "education": {},
    })
    context.verify_input = lambda _phase: SimpleNamespace(
        verification=SimpleNamespace(value=Verification.VERIFIED.value), parent_run_id="trial")
    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _: _available_derivation())
    monkeypatch.setattr(education_phase3, "read_trace", lambda *_: {"status": "unavailable"})
    view = education_phase3.Phase3View(context)
    copy = _frame(view, "Input chain")["copy"]
    assert "unmarked Optimisation-tab trial camset" in copy
    assert "may contain later-stage cameras" in copy
    assert "not confirmed" in copy
    seed_copy = _frame(view, "First guess and solver start")["copy"]
    assert "lens estimates from an unmarked Optimisation-tab trial camset" in seed_copy
    assert "lens estimates from Phase 2" not in seed_copy
    view.close()


@pytest.mark.gui
def test_unmarked_promoted_phase2_with_malformed_nested_metadata_does_not_crash(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase3
    from pyCamSet.workflow.education_trace import Verification

    context = _context(tmp_path, {"params": {}, "artifacts": {
        "phase2_initial_camset_used": "phase2_runs/trial/initial_cameras.camset"}})
    context.manager = SimpleNamespace(find_run=lambda *_: {
        "run_id": "trial", "diagnostics": ["invalid"], "education": ["invalid"], "promoted_from": "study-1",
    })
    context.verify_input = lambda _phase: SimpleNamespace(
        verification=SimpleNamespace(value=Verification.VERIFIED.value), parent_run_id="trial")
    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _: _available_derivation())
    monkeypatch.setattr(education_phase3, "read_trace", lambda *_: {"status": "unavailable"})

    view = education_phase3.Phase3View(context)
    copy = _frame(view, "Input chain")["copy"]
    assert "unmarked Optimisation-tab trial camset" in copy
    view.close()


@pytest.mark.gui
def test_phase3_failed_fixed_pose_withholds_camera_agreement(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase3
    from pyCamSet.workflow.education_derive import Phase3Derivation

    derivation = Phase3Derivation(
        status="available", reason=None, run_id="phase3-run", observations=(),
        per_image_observation_count=(0,), residual_mean_px=0.0, residual_xy=(),
        per_camera_residual_mean_px=(), seed_mean_by_image_px=(0.0,),
        seed_weighted_mean_px=0.0, d35_initial_px=0.0, seed_matches_saved=True,
        weighted_seed_matches_d35=True, d35_convention="D3.5", seed_camset_origin="input",
        input_verification="verified", saved_point_units="mm", rebuilt_point_units="mm",
        fixed_pose=0, fixed_pose_valid=False, rig_spreads=(), unit_disposition="known")
    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _context: derivation)
    monkeypatch.setattr(education_phase3, "read_trace",
                        lambda *_args: {"status": "unavailable", "reason": "missing sidecar"})
    view = education_phase3.Phase3View(_context(tmp_path, {
        "diagnostics": {"reference_pose_disposition": "reference image excluded; world frame not fixed"},
        "params": {"outliers": True},
    }))
    copy = _frame(view, "Did the cameras agree?")["copy"]
    assert "No valid reference photo" in copy
    assert "not tied to a fixed frame" in copy
    assert "yes" not in copy.lower()
    assert "removed" not in str(_frame(view, "Quality and next action")["rows"])
    view.close()


@pytest.mark.gui
def test_outlier_setting_reads_saved_problem_options_not_missing_top_level(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase3
    from pyCamSet.workflow.education_derive import DerivationUnavailable
    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _: DerivationUnavailable("missing"))
    monkeypatch.setattr(education_phase3, "read_trace", lambda *_: {"status": "unavailable"})
    view = education_phase3.Phase3View(_context(tmp_path, {"params": {"problem_options": {"outliers": "n"}}}))
    assert "was off" in view._outlier_copy()
    view.close()


@pytest.mark.gui
def test_outlier_setting_not_recorded_is_not_inferred_from_no_removals(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase3
    from pyCamSet.workflow.education_derive import DerivationUnavailable
    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _: DerivationUnavailable("missing"))
    monkeypatch.setattr(education_phase3, "read_trace", lambda *_: {"status": "unavailable"})
    view = education_phase3.Phase3View(_context(tmp_path, {"params": {}}))
    assert view._outlier_setting() is None
    assert view._outlier_copy() == "Outlier exclusion setting was not recorded for this run."
    view.close()


@pytest.mark.gui
def test_exclusions_use_saved_handler_masks_not_unwritten_diagnostics(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase3
    from pyCamSet.workflow.education_derive import DerivationUnavailable
    from pyCamSet.utils import saving
    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _: DerivationUnavailable("missing"))
    monkeypatch.setattr(education_phase3, "read_trace", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(saving, "load_CameraSet", lambda _: SimpleNamespace(calibration_handler=SimpleNamespace(
        missing_poses_before_outlier_rejection=np.array([True, False, False]),
        missing_poses_after_outlier_rejection=np.array([True, False, True]))))
    metadata = {"params": {"problem_options": {"outliers": "y"}},
                "diagnostics": {"D3.1_n_missing_poses": 1, "D3.2_n_outlier_removed": 1}}
    context = _context(tmp_path, metadata)
    artifact = context.run_dir / "phase3.camset"
    artifact.write_bytes(b"saved")
    context.artifact = lambda _: artifact
    view = education_phase3.Phase3View(context)
    assert view._excluded_images() == [(0, "not posed"), (2, "removed by outlier exclusion")]
    view.metadata["diagnostics"]["D3.2_n_outlier_removed"] = 2
    assert "per-image identities could not be verified" in view._outlier_copy()
    view.close()


@pytest.mark.gui
def test_lockbox_playback_rehydrates_only_digest_checked_source(tmp_path, monkeypatch):
    _app()
    import hashlib
    from pyCamSet.gui import education_phase3
    from pyCamSet.workflow.education_derive import DerivationUnavailable
    from pyCamSet.utils import saving
    source = tmp_path / "locked.camset"
    source.write_bytes(b"verified lockbox source")
    metadata = {"params": {"record_education": True,
                           "lockbox": {"enabled": True, "source_camset": str(source)}},
                "artifacts": {"lockbox_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}
    context = _context(tmp_path, metadata)
    artifact = context.run_dir / "phase3.camset"
    artifact.write_bytes(b"saved")
    context.artifact = lambda _: artifact
    np.savez_compressed(context.run_dir / "phase3_initial_state.npz",
                        calibration_params=np.array([0.]))
    monkeypatch.setattr(education_phase3, "derive_phase3",
                        lambda _: DerivationUnavailable("not needed"))
    monkeypatch.setattr(education_phase3, "read_trace",
                        lambda *_: {"status": "available", "records": []})
    expected_source = SimpleNamespace()
    def fake_load(path):
        if Path(path) == source:
            return expected_source
        return SimpleNamespace(calibration_params=np.array([0.]),
                               calibration_handler=SimpleNamespace())
    monkeypatch.setattr(saving, "load_CameraSet", fake_load)
    def fake_build(factory, _trace, **_kwargs):
        replay = factory()
        assert replay.lockbox_config.enabled is True
        assert replay.lockbox_source_camset is expected_source
        return {"status": "available", "frames": [], "payload": {}}
    monkeypatch.setattr(education_phase3, "build_phase3_frames", fake_build)
    view = education_phase3.Phase3View(context)
    assert view._playback["status"] == "available"
    view.close()
    source.write_bytes(b"edited later")
    changed = education_phase3.Phase3View(context)
    assert "no longer matches" in changed._playback["reason"]
    changed.close()


@pytest.mark.gui
def test_largest_camera_step_measures_camera_centres_in_solver_layout():
    from pyCamSet.gui.education_phase3 import _largest_camera_step

    # Solver layout: (rotation vector, world-to-camera translation) per camera.
    previous = np.zeros((2, 6))
    previous[:, 3:] = (0.0, 0.0, 10.0)
    # A pure rotation about the camera centre changes t but not C = -R^T t.
    rotated = previous.copy()
    rvec = np.array([0.0, 0.3, 0.0])
    rotation, _ = cv2.Rodrigues(rvec)
    centre = -previous[0, 3:]
    rotated[0, :3] = rvec
    rotated[0, 3:] = -rotation @ centre
    assert np.linalg.norm(rotated[0, 3:] - previous[0, 3:]) > 1.0
    assert _largest_camera_step((None, previous), (None, rotated)) == pytest.approx(0.0, abs=1e-9)
    # A real centre move is reported in target units even with identity rotation.
    moved = previous.copy()
    moved[1, 3:] = (-3.0, -4.0, 10.0)
    assert _largest_camera_step((None, previous), (None, moved)) == pytest.approx(5.0)
    assert _largest_camera_step((None, previous), (None, moved[:, :3])) is None


@pytest.mark.gui
def test_largest_camera_step_inverts_homogeneous_extrinsics():
    from pyCamSet.gui.education_phase3 import _largest_camera_step

    previous = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
    current = previous.copy()
    current[0, :3, 3] = (3.0, 4.0, 0.0)
    current[1, :3, 3] = (0.0, 0.0, 2.0)
    assert _largest_camera_step((None, previous), (None, current)) == 5.0
    rotated = previous.copy()
    rotated[0, :3, :3] = cv2.Rodrigues(np.array([0.0, 0.0, 0.5]))[0]
    assert _largest_camera_step((None, previous), (None, rotated)) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.gui
def test_geometry_snapshot_rows_are_named_and_fail_closed():
    from pyCamSet.gui.education_phase3 import _geometry_snapshot_rows

    cameras = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
    poses = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
    cameras[1, :3, 3] = (1.0, 2.0, 3.0)
    rows = _geometry_snapshot_rows((None, cameras, poses), "Saved final result")
    # World-to-camera translation t=(1,2,3) with R=I puts the centre at -t.
    # Cameras are numbered as their badges are (index 1 is camera 2).
    assert rows[1] == ("Saved final result: camera 2 centre", "-1, -2, -3",
                       "target coordinate units; C = -Rᵀt from the recorded extrinsics")
    assert _geometry_snapshot_rows((None, cameras), "Saved final result")[0][1] == "unavailable"


@pytest.mark.gui
def test_real_persisted_phase3_trace_builds_geometry_in_qt(tmp_path, monkeypatch):
    """Exercise the producer trace, disk reader and real frame builder together."""
    _app()
    import hashlib
    from pyCamSet import ChArUco
    from pyCamSet.calibration.camera_calibrator import (
        detect_datapoints_in_imfile, run_initial_calibration,
    )
    from pyCamSet.gui import education_phase3
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow import phase3
    from pyCamSet.workflow.education_trace import read_trace, write_trace

    corpus = Path(__file__).parent / "test_data" / "calibration_charuco"
    target = ChArUco(num_squares_x=20, num_squares_y=20, square_size=4.0, legacy=False)
    detections, resolutions = detect_datapoints_in_imfile(
        f_loc=corpus, caching=False, calibration_target=target, threads=1)
    phase2_set = run_initial_calibration(detections, target, resolutions, save=False)
    phase2_set.set_resolutions_from_file(floc=corpus)
    handler, _result, solved, _stats = phase3.solve(
        phase2_set, detections, target, fixed_params={},
        options={"outliers": "n", "fixed_pose": 0}, threads=1,
        record_education=True)
    payload = handler.education_step_trace
    run_dir = tmp_path / "phase3_runs" / "phase3-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    phase2_path = run_dir / "phase2.camset"
    phase3_path = run_dir / "phase3.camset"
    phase2_set.save(phase2_path)
    solved.save(phase3_path)
    payload.update({"run_id": "phase3-run", "phase": "phase3"})
    write_trace(run_dir, "phase3-run", "phase3", "accepted_step", payload, enabled=True)
    initial = run_dir / "phase3_initial_state.npz"
    np.savez_compressed(initial,
                        calibration_params=np.asarray(payload["steps"][0]["x"], dtype=np.float64))
    # The Qt view's derivation is isolated here; playback still consumes the real,
    # persisted producer trace, saved camset and production frame builder.
    monkeypatch.setattr(education_phase3, "derive_phase3",
                        lambda _context: _available_derivation())
    trace = read_trace(run_dir, "phase3-run", "phase3")
    assert trace["status"] == "available"
    context = _context(tmp_path, {
        "params": {"record_education": True},
        "artifacts": {"phase2_initial_camset_used": str(phase2_path),
                      "initial_state": str(initial),
                      "initial_state_sha256": hashlib.sha256(initial.read_bytes()).hexdigest()},
    })
    context.artifact = lambda _phase=None: phase3_path
    view = education_phase3.Phase3View(context)
    assert view._playback["status"] == "available"
    assert any("Solver starting point: camera" in row[0] for row in view._step_rows())
    assert any("Saved final result: Photo" in row[0] for row in view._step_rows())
    assert view._playback["frames"][-1][1].shape[0] == len(load_CameraSet(phase3_path).get_names())
    # Real frames use the solver's (rvec, t) layout; every accepted step must
    # report a camera-centre displacement checked against Camera.position.
    from pyCamSet.gui.education_phase3 import _largest_camera_step
    from pyCamSet.utils.general_utils import make_4x4h_tform
    frames = view._playback["frames"]
    step_rows = [row for row in view._step_rows() if row[0].startswith("Step ")]
    assert step_rows and all(row[2] != "unavailable" for row in step_rows)

    def centres(frame):
        return np.array([np.linalg.inv(make_4x4h_tform(extr[:3], extr[3:]))[:3, 3]
                         for extr in np.asarray(frame[1], dtype=np.float64)])

    expected = np.max(np.linalg.norm(centres(frames[1]) - centres(frames[0]), axis=1))
    assert _largest_camera_step(frames[0], frames[1]) == pytest.approx(expected)
    final_positions = np.array([camera.position[:3] for camera in load_CameraSet(phase3_path)])
    assert np.allclose(centres(frames[-1]), final_positions, atol=1e-6)
    view.close()


def _available_derivation():
    from pyCamSet.workflow.education_derive import Phase3Derivation

    observation = SimpleNamespace(camera=0)
    spread = SimpleNamespace(
        camera=1, position_scatter_mm=2.5, position_scatter_squares=None,
        rotation_scatter_deg=1.2, usable_image_count=9,
        position_photo_outliers=(), rotation_photo_outliers=(),
        outlier_rule_status="applied")
    return Phase3Derivation(
        status="available", reason=None, run_id="phase3-run",
        observations=(observation,), per_image_observation_count=(1,),
        residual_mean_px=2.0, residual_xy=((1.0, 1.0),),
        per_camera_residual_mean_px=(("camera-A", 2.0),),
        seed_mean_by_image_px=(3.0,), seed_weighted_mean_px=3.0,
        d35_initial_px=3.0, seed_matches_saved=True,
        weighted_seed_matches_d35=True, d35_convention="D3.5",
        seed_camset_origin="verified Phase 2 artifact", input_verification="verified",
        saved_point_units="mm", rebuilt_point_units="mm", fixed_pose=0,
        fixed_pose_valid=True, rig_spreads=(spread,), unit_disposition="known")


@pytest.mark.gui
def test_phase3_same_observation_seed_residual_is_explicitly_unavailable(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase3

    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _context: _available_derivation())
    monkeypatch.setattr(education_phase3, "read_trace",
                        lambda *_args: {"status": "unavailable", "reason": "missing sidecar"})
    view = education_phase3.Phase3View(_context(tmp_path, {"params": {}}))
    frame = _frame(view, "Same observations before and after")
    assert "Same-observation before/after residuals are unavailable" in frame["copy"]
    assert "keyed starting-state residuals not recorded" in frame["copy"]
    assert "unavailable: keyed starting-state residuals not recorded" in frame["rows"][0][2]
    view.close()


@pytest.mark.gui
def test_phase3_same_rows_display_verified_before_and_after(tmp_path, monkeypatch):
    _app()
    from dataclasses import replace
    from pyCamSet.gui import education_phase3
    from pyCamSet.workflow.education_derive import ResidualObservation
    final = ResidualObservation(0, 0, (1,), (10., 12.), (2., 0.), 2.)
    start = ResidualObservation(0, 0, (1,), (10., 12.), (4., 0.), 4.)
    derived = replace(_available_derivation(), observations=(final,),
                      seed_observations=(start,),
                      seed_observation_status="verified on the same ordered optimised rows")
    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _: derived)
    monkeypatch.setattr(education_phase3, "read_trace",
                        lambda *_: {"status": "unavailable", "reason": "missing sidecar"})
    view = education_phase3.Phase3View(_context(tmp_path, {"params": {}}))
    frame = _frame(view, "Same observations before and after")
    assert frame["rows"][0] == ("camera-A", "1", "4", "2", "camera mean")
    # Raw per-observation rows are L3 evidence (A1 section 1); L1 and L2
    # keep the per-camera means so a real rig does not list thousands of rows.
    assert len(frame["rows"]) == 1
    view.stage_view.set_level(2)
    assert len(_frame(view, "Same observations before and after")["rows"]) == 1
    view.stage_view.set_level(3)
    frame = _frame(view, "Same observations before and after")
    assert frame["rows"][0] == ("camera-A", "1", "4", "2", "camera mean")
    assert frame["rows"][1] == ("0 / image 0 / point 1", "1", "4", "2", "(4, 0) → (2, 0)")
    assert "same ordered" in frame["copy"]
    view.close()


@pytest.mark.gui
@pytest.mark.data
@pytest.mark.skipif(
    int(cv2.__version__.split(".")[0]) < 5,
    reason=("OpenCV 5 measurement: OpenCV 4 places sub-pixel ChArUco corners slightly "
            "differently, so its results are not numerically comparable."),
)
def test_persisted_outlier_run_uses_seed_detection_denominator_in_storyboard(tmp_path):
    """Keep D3.3 N tied to initialiser rows when solver rejection removes views."""
    _app()
    import hashlib
    from pyCamSet import ChArUco
    from pyCamSet.calibration.camera_calibrator import (
        detect_datapoints_in_imfile, run_initial_calibration,
    )
    from pyCamSet.gui import education_phase3
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow import phase3
    from pyCamSet.workflow.education_trace import RunContext
    from pyCamSet.workflow.workspace import WorkspaceManager

    corpus = Path(__file__).parent / "test_data" / "calibration_charuco"
    target = ChArUco(num_squares_x=20, num_squares_y=20, square_size=4.0, legacy=False)
    detections, resolutions = detect_datapoints_in_imfile(
        f_loc=corpus, caching=False, calibration_target=target, threads=1)
    phase2 = run_initial_calibration(detections, target, resolutions, save=False)
    phase2.set_resolutions_from_file(floc=corpus)
    workspace = tmp_path / ".pycamset_workspace"
    manager = WorkspaceManager(workspace)
    phase2_dir = workspace / "phase2_runs" / "p2"
    phase2_dir.mkdir(parents=True)
    phase2_path = phase2_dir / "initial_cameras.camset"
    phase2.save(phase2_path)
    manager.save_run("phase2", "p2", {
        "run_id": "p2", "status": "complete",
        "artifacts": {"initial_camset": str(phase2_path)},
    })
    handler, optimisation, solved, stats = phase3.solve(
        load_CameraSet(phase2_path), detections, target, fixed_params={},
        options={"outliers": "y", "fixed_pose": 0}, threads=1)
    phase3_dir = workspace / "phase3_runs" / "p3"
    phase3_dir.mkdir(parents=True)
    initial_path = phase3._write_initial_state(
        phase3_dir, handler.get_initial_params(), handler.missing_poses,
        float(stats["initial_euclid"]))
    phase3_path = phase3_dir / "optimised_cameras.camset"
    solved.save(phase3_path)
    diagnostics = phase3._diagnostics(
        optimisation, handler, stats, float(stats["initial_euclid"]),
        float(stats["final_euclid"]), lambda _message: None)
    manager.save_run("phase3", "p3", {
        "run_id": "p3", "status": "complete",
        "report": {"blocking_flags": []},
        "inputs": {"phase2_run_id": "p2"},
        "artifacts": {
            "optimised_camset": str(phase3_path),
            "initial_state": str(initial_path),
            "initial_state_sha256": hashlib.sha256(initial_path.read_bytes()).hexdigest(),
            "phase2_initial_camset_used": str(phase2_path),
        },
        "diagnostics": diagnostics,
    })
    record = manager.find_run("phase3", "p3")
    context = RunContext.from_run(manager, record)
    view = education_phase3.Phase3View(context)
    result = view._derivation
    assert result.status == "available", getattr(result, "reason", None)
    seed_counts = result.seed_detection_count_by_image
    final_counts = result.per_image_observation_count
    assert len(seed_counts) == len(result.seed_mean_by_image_px)
    assert sum(seed_counts) >= sum(final_counts)
    source_rows = detections.return_flattened_keys(target.point_data.shape[:-1]).get_data()
    expected = np.bincount(source_rows[:, 1].astype(int), minlength=len(seed_counts))
    expected[np.asarray(result.seed_mean_by_image_px, dtype=object) == None] = 0
    assert tuple(expected.tolist()) == seed_counts
    assert any(seed > final for seed, final in zip(seed_counts, final_counts))
    seed_rows = _frame(view, "First guess and solver start")["rows"]
    assert not any(row[0] == "D3.3 summed error, px (L3)" for row in seed_rows)
    view.stage_view.set_level(3)
    seed_rows = _frame(view, "First guess and solver start")["rows"]
    assert any(row[0] == "D3.3 summed error, px (L3)" for row in seed_rows)
    assert f"N={sum(seed_counts)}" in str(seed_rows)
    view.close()


@pytest.mark.gui
def test_phase3_quality_gauge_outlier_and_lockbox_copy_are_run_bound(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase3
    from pyCamSet.utils import saving

    monkeypatch.setattr(saving, "load_CameraSet", lambda _: SimpleNamespace(calibration_handler=SimpleNamespace(
        missing_poses_before_outlier_rejection=np.array([False, True, False]),
        missing_poses_after_outlier_rejection=np.array([False, True, True]))))
    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _context: _available_derivation())
    monkeypatch.setattr(education_phase3, "read_trace",
                        lambda *_args: {"status": "unavailable", "reason": "missing sidecar"})
    context = _context(tmp_path, {
        "params": {"problem_options": {"outliers": "y"}},
        "artifacts": {"lockbox_prior": "prior/lockbox.camset"},
        "report": {"termination_reason": "converged", "blocking_flags": []},
        "diagnostics": {"D3.1_n_missing_poses": 1,
                        "D3.2_n_outlier_removed": 1, "quality_gate": {}},
    })
    context.quality_check = "checked"
    artifact = context.run_dir / "phase3.camset"
    artifact.write_bytes(b"saved")
    context.artifact = lambda _: artifact
    view = education_phase3.Phase3View(context)
    chain = _frame(view, "Input chain")
    assert ("Lockbox prior", "prior/lockbox.camset", "run record") in chain["rows"]
    agreement = _frame(view, "Did the cameras agree?")
    assert "2.5 mm" in agreement["rows"][0][1]
    assert "camera 2 relative to camera 1" in agreement["rows"][0][0]
    assert "Overall answer: Yes" in compose(agreement, 3)
    assert "Spread check (" in compose(agreement, 3) and "Single-photo check:" in compose(agreement, 3)
    assert "10 mm" in compose(agreement, 3) and "5°" in compose(agreement, 3) and "5×" in compose(agreement, 3)
    quality = _frame(view, "Quality and next action")
    assert "Quality check: complete. Termination: converged." in quality["copy"]
    assert "1 image(s) were recorded as newly excluded" in quality["copy"]
    assert ("Photo 3 (index 2)", "0 retained", "not posed / excluded",
            "no after — removed by outlier exclusion", "not applicable") in view._residual_rows()
    view.close()


@pytest.mark.gui
def test_phase3_decision_copy_reports_separate_position_rotation_and_photo_outcomes(tmp_path, monkeypatch):
    _app()
    from dataclasses import replace
    from pyCamSet.gui import education_phase3

    base = _available_derivation()
    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _context: base)
    monkeypatch.setattr(education_phase3, "read_trace",
                        lambda *_args: {"status": "unavailable", "reason": "missing sidecar"})
    view = education_phase3.Phase3View(_context(tmp_path, {"params": {}}))
    copy = view._decision_copy()
    assert "Overall answer: Yes" in copy
    assert "position = Yes" in copy and "rotation = Yes" in copy and "Single-photo check: Yes" in copy

    spread = base.rig_spreads[0]
    def updated_spread(**changes):
        return SimpleNamespace(**{**vars(spread), **changes})

    position_high = updated_spread(position_scatter_mm=10.1)
    rotation_high = updated_spread(rotation_scatter_deg=5.1)
    flagged = updated_spread(rotation_photo_outliers=(3,))
    insufficient = updated_spread(usable_image_count=4, outlier_rule_status="insufficient")
    unknown_units = replace(base, saved_point_units="squares", rebuilt_point_units="squares",
                            unit_disposition="unknown")
    square_spread = updated_spread(position_scatter_mm=None, position_scatter_squares=0.04)

    view._derivation = replace(base, rig_spreads=(position_high,))
    position_copy = view._decision_copy()
    assert "position = No (at least one camera's spread exceeds 10 mm)" in position_copy
    assert "Overall answer: No" in position_copy
    view._derivation = replace(base, rig_spreads=(rotation_high,))
    rotation_copy = view._decision_copy()
    assert "rotation = No (at least one camera's spread exceeds 5°)" in rotation_copy
    assert "Overall answer: No" in rotation_copy
    view._derivation = replace(base, rig_spreads=(flagged,))
    photo_copy = view._decision_copy()
    assert "Single-photo check: Not clearly (the per-photo rule flagged" in photo_copy
    assert "camera 2, Photo 4 (photo file not checkable)" in photo_copy
    assert "rotation = Not clearly" in photo_copy
    assert "Overall answer: Not clearly" in photo_copy
    view._derivation = unknown_units
    unknown_copy = view._decision_copy()
    assert "Overall answer: Not clearly" in unknown_copy
    assert "position = Not clearly (length units are unverified" in unknown_copy
    assert "rotation = Yes" in unknown_copy
    view._derivation = replace(unknown_units, rig_spreads=(square_spread,))
    square_copy = view._decision_copy()
    assert "about 0 squares" not in square_copy
    assert "about 0 squares" not in str(view._spread_rows())
    assert "square-size" not in square_copy
    assert "unit-free square units, not converted to length" in square_copy
    assert "mm position conversion assumes" not in square_copy
    view._derivation = replace(base, rig_spreads=(insufficient,))
    insufficient_copy = view._decision_copy()
    assert "Overall answer: Not clearly" in insufficient_copy
    assert "Single-photo check: Not clearly (insufficient usable-photo evidence" in insufficient_copy
    assert "at least 8 finite deviations" in insufficient_copy
    low_n_flagged = updated_spread(usable_image_count=4, outlier_rule_status="insufficient",
                                   rotation_photo_outliers=(3,))
    view._derivation = replace(base, rig_spreads=(low_n_flagged,))
    low_n_copy = view._decision_copy()
    assert "Overall answer: Not clearly" in low_n_copy
    assert "rotation = Not clearly" in low_n_copy
    assert "Single-photo check: Not clearly" in low_n_copy
    view.close()


@pytest.mark.gui
def test_thinning_reports_gap_before_first_kept_solver_step(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase3

    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _context: _available_derivation())
    trace = {"status": "available", "records": []}
    monkeypatch.setattr(education_phase3, "read_trace", lambda *_args: trace)
    view = education_phase3.Phase3View(_context(tmp_path, {"params": {"record_education": True}}))
    cameras = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
    view._playback = {"status": "available", "payload": {"loss": "linear", "steps": [
        {"nit": 0, "cost": 5.0}, {"nit": 4, "cost": 2.0}, {"nit": 5, "cost": 1.0}]},
        "frames": [(None, cameras), (None, cameras), (None, cameras)]}
    assert ("Thinning", "steps not kept", "1, 2, 3") in view._step_rows()
    view.close()


@pytest.mark.gui
def test_phase3_join_preroll_precedes_recorded_solver_steps(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase3

    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _context: _available_derivation())
    trace = {"status": "available", "records": [{"kind": "seed_join_order", "payload": {
        "events": [{"joining": 2, "joined_from": 0, "spread": 4.0}]}}]}
    monkeypatch.setattr(education_phase3, "read_trace", lambda *_args: trace)
    view = education_phase3.Phase3View(_context(tmp_path, {"params": {"record_education": True}}))
    view._playback = {"status": "available", "payload": {"loss": "linear", "steps": [
        {"nit": 0, "cost": 5.0}, {"nit": 1, "cost": 3.0},
        {"nit": 4, "cost": 2.0}]},
        "frames": [(None, np.eye(4)[None, :, :]),
                   (None, np.eye(4)[None, :, :]),
                   (None, np.eye(4)[None, :, :]) ]}
    rows = view._step_rows()
    assert rows[0] == ("Camera 3 joined via camera 1", "4 target units", "first-guess assembly — not solver steps")
    assert rows[1] == ("Starting point (not an accepted step)", "5", "—")
    assert rows[2] == ("Step 1", "3", "0")
    assert rows[-1] == ("Thinning", "steps not kept", "2, 3")
    copy = view._playback_copy()
    assert "not solver steps" in copy
    view.close()


@pytest.mark.gui
def test_phase3_trace_error_is_reported_as_unavailable_without_interpolation(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase3
    from pyCamSet.workflow.education_derive import DerivationUnavailable

    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _: DerivationUnavailable("missing"))
    monkeypatch.setattr(education_phase3, "read_trace", lambda *_: {"status": "unavailable", "reason": "invalid trace checksum"})
    view = education_phase3.Phase3View(_context(tmp_path, {"params": {"record_education": True}}))
    assert "Static before/after only" in _frame(view, "Recorded-step playback")["copy"]
    assert "invalid trace checksum" in _frame(view, "Recorded-step playback")["copy"]
    rows = view._step_rows()
    assert rows[0] == ("Playback", "unavailable", "invalid trace checksum")
    assert rows[1][0] == "Saved final rig / target poses"
    assert rows[1][1] == "unavailable"
    view.close()


@pytest.mark.parametrize("status", ["incomplete", "failed"])
@pytest.mark.gui
def test_valid_trace_is_inspectable_for_incomplete_and_failed_runs(tmp_path, monkeypatch, status):
    _app()
    import hashlib
    from pyCamSet.gui import education_phase3
    from pyCamSet.utils import saving
    from pyCamSet.workflow.education_derive import DerivationUnavailable

    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _: DerivationUnavailable("not needed"))
    monkeypatch.setattr(education_phase3, "read_trace", lambda *_: {
        "status": "available", "records": [{"kind": "accepted_step", "payload": {"steps": [{"nit": 0}]}}]})
    monkeypatch.setattr(education_phase3, "build_phase3_frames", lambda *_args, **_kwargs: {
        "status": "available", "payload": {"loss": "linear", "steps": [{"nit": 0, "cost": 3.0}]},
        "frames": [(np.array([0.]), np.eye(4)[None, :, :], np.eye(4)[None, :, :])],
    })
    context = _context(tmp_path, {"params": {"record_education": True}, "artifacts": {}}, status=status)
    camset = context.run_dir / "saved.camset"
    camset.write_bytes(b"saved")
    initial = context.run_dir / "phase3_initial_state.npz"
    np.savez_compressed(initial, calibration_params=np.array([0.]))
    context.artifact = lambda _phase=None: camset
    context.metadata["artifacts"].update({
        "initial_state": str(initial),
        "initial_state_sha256": hashlib.sha256(initial.read_bytes()).hexdigest(),
    })
    monkeypatch.setattr(saving, "load_CameraSet", lambda _path: SimpleNamespace(
        calibration_params=np.array([0.]), calibration_handler=SimpleNamespace()))
    view = education_phase3.Phase3View(context)
    assert view._playback["status"] == "available"
    assert view.stage_view.current_stage() == 0
    assert not hasattr(view, "play")
    view.close()


@pytest.mark.gui
def test_a0_anisotropic_six_mm_n40_healthy_case_answers_yes(tmp_path, monkeypatch):
    """Use the integrated A0 rig generator and fixed-seed realised 6 mm RMS case."""
    from dataclasses import replace
    from scipy.spatial.transform import Rotation
    from equivalence.synthetic_rig import make_synthetic_rig
    from pyCamSet import ChArUco
    from pyCamSet.gui import education_phase3
    from pyCamSet.workflow.education_derive import derive_rig_spreads

    _app()
    target = ChArUco(num_squares_x=8, num_squares_y=8, square_size=40.0)
    rig = make_synthetic_rig(target, camera_count=3, image_count=40)
    poses = np.repeat(np.eye(4)[None, None, :, :], 3, axis=0)
    poses = np.repeat(poses, 40, axis=1)
    rng = np.random.default_rng(20260929)
    anisotropy = np.array([4.5, 3.7, 1.0])
    for camera in (1, 2):
        noise = rng.normal(size=(40, 3)) * anisotropy
        noise -= noise.mean(axis=0)
        noise *= 0.006 / np.sqrt(np.mean(np.sum(noise ** 2, axis=1)))
        angles = rng.normal(size=(40, 3))
        angles -= angles.mean(axis=0)
        angles *= np.deg2rad(1.0) / np.sqrt(np.mean(np.sum(angles ** 2, axis=1)))
        rotations = Rotation.from_rotvec(angles).as_matrix()
        poses[camera, :, :3, :3] = rotations
        # The 6 mm RMS scatter is put on the camera centre C (camera 0 is the
        # identity), so each pose carries t = -R C as in a real seed.
        centres = -rig.camera_translations[camera] + noise
        poses[camera, :, :3, 3] = -np.einsum("nij,nj->ni", rotations, centres)
    spreads = derive_rig_spreads(poses, target)
    assert all(abs(item.position_scatter_mm - 6.0) < 1e-10 for item in spreads)
    assert all(item.rotation_scatter_deg < 5.0 for item in spreads)
    assert all(item.outlier_rule_status == "applied" for item in spreads)
    assert all(not item.position_photo_outliers and not item.rotation_photo_outliers for item in spreads)
    base = _available_derivation()
    metric = replace(base, saved_point_units="m", rebuilt_point_units="m",
                     unit_disposition="known", rig_spreads=spreads)
    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _: metric)
    monkeypatch.setattr(education_phase3, "read_trace", lambda *_: {"status": "unavailable"})
    view = education_phase3.Phase3View(_context(tmp_path, {"params": {}}))
    answer = compose(_frame(view, "Did the cameras agree?"), 3)
    assert "Overall answer: Yes" in answer
    assert "position = Yes" in answer and "rotation = Yes" in answer
    assert "Single-photo check: Yes" in answer
    view.close()


@pytest.mark.gui
def test_phase3_levels_are_additive_and_l1_is_plain(tmp_path, monkeypatch):
    """A1 section 1: L1 is plain language; codes and rule parameters arrive at L2/L3.

    Captions are short (brief: <= 2 sentences at L1) and each level only adds to the last.
    """
    _app()
    from test_education_copy_lint import learner_copy_findings
    from pyCamSet.gui import education_phase3

    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _context: _available_derivation())
    monkeypatch.setattr(education_phase3, "read_trace",
                        lambda *_args: {"status": "unavailable", "reason": "missing sidecar"})
    view = education_phase3.Phase3View(_context(tmp_path, {"params": {}}))
    rendered = {}
    for level in (1, 2, 3):
        view.stage_view.set_level(level)
        rendered[level] = []
        for index in range(len(view.stage_keys())):
            view.stage_view.set_current_stage(index)
            text = view.stage_view.caption.text()
            rendered[level].append(text)
            assert learner_copy_findings(text, level) == [], (level, index, text)
    for low, high in ((1, 2), (2, 3)):
        assert all(text_low in text_high for text_low, text_high in zip(rendered[low], rendered[high]))
    # The agreement answer keeps its reviewed copy, now in the Result stage's Numbers drawer.
    decision = compose(_frame(view, "Did the cameras agree?"), 1)
    assert decision.startswith("Did the cameras agree on where the target was, across your photos?")
    assert "Camera 2: its estimated position relative to camera 1 varied by 2.5 mm across photos." in decision
    assert "first-guess position for every photo" in decision
    assert "the solved rig has one position per camera" in decision
    assert "Spread check" not in decision and "Single-photo rule" not in decision
    assert "not the solved Phase 3 extrinsics" in compose(_frame(view, "Did the cameras agree?"), 2)
    assert "Single-photo rule: it checks deviations" in compose(_frame(view, "Did the cameras agree?"), 3)
    assert "Each camera is compared with camera-A (camera 1), the first camera." in decision
    view.show_stage("result")
    assert any("Did the cameras agree on where the target was" in cell for cell in _table_cells(view))
    view.close()


@pytest.mark.gui
def test_phase3_l1_tables_carry_no_diagnostic_codes(tmp_path, monkeypatch):
    """A1 section 1: D-codes and the derive layer's convention text are L2 table content."""
    _app()
    import re
    from pyCamSet.gui import education_phase3

    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _context: _available_derivation())
    monkeypatch.setattr(education_phase3, "read_trace",
                        lambda *_args: {"status": "unavailable", "reason": "missing sidecar"})
    view = education_phase3.Phase3View(_context(tmp_path, {"params": {}}))
    code = re.compile(r"\bD\d\.\d")
    for level, expect_codes in ((1, False), (2, True)):
        view.stage_view.set_level(level)
        cells = []
        for index in range(len(view.stage_keys())):
            view.stage_view.set_current_stage(index)
            cells += _table_cells(view)
        assert bool([cell for cell in cells if code.search(cell)]) is expect_codes, (level, cells)
    seed = _frame(view, "First guess and solver start")["rows"]
    assert ("Solver starting point (px)", "3", "saved D3.5") in seed
    view.stage_view.set_level(1)
    seed = _frame(view, "First guess and solver start")["rows"]
    assert ("Where the solver began, average error (px)", "3", "saved with this run") in seed
    view.close()

@pytest.mark.gui
def test_phase3_named_cameras_and_reference_in_agreement_answer(tmp_path, monkeypatch):
    """The agreement answer names cameras as the learner's folders do, keeping the index."""
    _app()
    from dataclasses import replace
    from pyCamSet.gui import education_phase3

    base = _available_derivation()
    spread = SimpleNamespace(**{**vars(base.rig_spreads[0]), "position_photo_outliers": (4,)})
    derivation = replace(base, per_camera_residual_mean_px=(("cam_a", 2.0), ("cam_b", 2.1)),
                         rig_spreads=(spread,))
    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _context: derivation)
    monkeypatch.setattr(education_phase3, "read_trace", lambda *_: {"status": "unavailable"})
    view = education_phase3.Phase3View(_context(tmp_path, {"params": {}}))
    agreement = _frame(view, "Did the cameras agree?")
    assert "Each camera is compared with cam_a (camera 1), the first camera." in agreement["copy"]
    assert "cam_b (camera 2): its estimated position relative to camera 1 varied by 2.5 mm" in agreement["copy"]
    assert "Photos that stand out: cam_b (camera 2), Photo 5" in agreement["copy"]
    assert agreement["rows"][0][0] == "cam_b (camera 2) relative to camera 1"
    view.close()


@pytest.mark.gui
def test_phase3_rotation_only_no_names_its_cause_and_next_step_at_l1(tmp_path, monkeypatch):
    """A No decided by rotation alone shows the rotation line, the level and what to do next,
    and the quality frame does not then say a bare 'continue to Phase 4'."""
    _app()
    from dataclasses import replace
    from pyCamSet.gui import education_phase3

    base = _available_derivation()
    turned = SimpleNamespace(**{**vars(base.rig_spreads[0]), "rotation_scatter_deg": 6.2})
    monkeypatch.setattr(education_phase3, "derive_phase3",
                        lambda _context: replace(base, rig_spreads=(turned,)))
    monkeypatch.setattr(education_phase3, "read_trace", lambda *_: {"status": "unavailable"})
    context = _context(tmp_path, {"params": {}, "report": {"termination_reason": "converged",
                                                            "blocking_flags": []}}, status="complete")
    context.quality_check = "checked"
    view = education_phase3.Phase3View(context)
    agreement = _frame(view, "Did the cameras agree?")
    copy = compose(agreement, 1)
    assert "Overall answer: No" in copy
    # The position line is kept and says it is within its level ...
    assert "varied by 2.5 mm across photos. That is within 10 mm" in copy
    # ... and the rotation line that decided the No is shown beside it.
    assert ("Camera 2: from photo to photo, its estimated turn relative to camera 1 varied by "
            "about 6.2° (typical deviation from the average), more than 5°") in copy
    assert copy.rstrip().splitlines()[-1].startswith("What to do next: check the photos")
    assert "re-run from Phase 1" in copy
    quality = _frame(view, "Quality and next action")["copy"]
    assert "continue to Phase 4, or export" not in quality
    assert "the cameras did not clearly agree (see the previous stage)" in quality
    view.close()


@pytest.mark.gui
def test_phase3_yes_agreement_has_no_what_to_do_next_line(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_phase3

    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _context: _available_derivation())
    monkeypatch.setattr(education_phase3, "read_trace", lambda *_: {"status": "unavailable"})
    view = education_phase3.Phase3View(_context(tmp_path, {"params": {}}))
    copy = compose(_frame(view, "Did the cameras agree?"), 1)
    assert "Overall answer: Yes" in copy and "What to do next" not in copy
    view.close()


@pytest.mark.gui
@pytest.mark.parametrize(("metadata", "status", "quality", "expected"), [
    ({"report": {"termination_reason": "converged", "blocking_flags": []}}, "complete", "checked",
     "Next: continue to Phase 4, or export this result."),
    ({"report": {"termination_reason": "x", "blocking_flags": ["solver did not report successful termination"]}},
     "incomplete", "checked", "then run Phase 3 again before continuing to Phase 4."),
    ({"error": "solver crashed"}, "failed", "failed", "Next: fix the recorded error above, then run Phase 3 again."),
    ({}, "unknown", "not checked", "Next: run Phase 3 again to get a checkable quality result"),
])
def test_phase3_quality_frame_ends_on_a_status_keyed_next_step(
        tmp_path, monkeypatch, metadata, status, quality, expected):
    """A1 section 9: the 'Quality and next action' frame names what to do next."""
    _app()
    from pyCamSet.gui import education_phase3

    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _context: _available_derivation())
    monkeypatch.setattr(education_phase3, "read_trace", lambda *_: {"status": "unavailable"})
    context = _context(tmp_path, {"params": {}, **metadata}, status=status)
    context.quality_check = quality
    view = education_phase3.Phase3View(context)
    copy = _frame(view, "Quality and next action")["copy"]
    assert copy.rstrip().endswith(".")
    assert expected in copy
    assert copy.index("Next:") > copy.index("Outlier exclusion")
    view.close()


@pytest.mark.gui
def test_phase3_reference_photo_refusal_ends_with_recovery(tmp_path, monkeypatch):
    _app()
    from dataclasses import replace
    from pyCamSet.gui import education_phase3

    derivation = replace(_available_derivation(), fixed_pose_valid=False)
    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _context: derivation)
    monkeypatch.setattr(education_phase3, "read_trace", lambda *_: {"status": "unavailable"})
    for diagnostics in ({"reference_pose_disposition": "reference image excluded; world frame not fixed"}, {}):
        view = education_phase3.Phase3View(_context(tmp_path, {"params": {}, "diagnostics": diagnostics}))
        copy = _frame(view, "Did the cameras agree?")["copy"]
        assert copy.endswith("Re-run Phase 3 with an observed, usable reference photo, "
                             "then run Phase 4 from that result.")
        view.close()


@pytest.mark.gui
def test_phase3_recorded_steps_copy_names_steps_not_motion(tmp_path, monkeypatch):
    """The playback shows recorded solver states; its copy must not claim physical motion."""
    _app()
    from test_education_copy_lint import _rules, lint_copy
    from pyCamSet.gui import education_phase3

    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _context: _available_derivation())
    monkeypatch.setattr(education_phase3, "read_trace", lambda *_: {"status": "available", "records": []})
    view = education_phase3.Phase3View(_context(tmp_path, {"params": {"record_education": True}}))
    view._playback = {"status": "available", "payload": {"loss": "linear", "steps": []}, "frames": []}
    copy = view._playback_copy()
    assert ("Recorded accepted solver steps: they show the estimates changing, not the rig "
            "physically moving.") in copy
    assert "not pixels" in copy and "interpolated" in copy and "only the recorded states are data" in copy
    assert lint_copy(copy, "education_phase3.py", "phase3.playback", _rules()) == []
    view.close()

@pytest.mark.gui
def test_phase3_unit_free_answer_carries_never_zero_anchor(tmp_path, monkeypatch):
    _app()
    from dataclasses import replace
    from pyCamSet.gui import education_phase3

    base = _available_derivation()
    spread = SimpleNamespace(**{**vars(base.rig_spreads[0]), "position_scatter_mm": None,
                                "position_scatter_squares": 0.04})
    derivation = replace(base, saved_point_units="squares", rebuilt_point_units="squares",
                         unit_disposition="unknown", rig_spreads=(spread,))
    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _context: derivation)
    monkeypatch.setattr(education_phase3, "read_trace", lambda *_: {"status": "unavailable"})
    view = education_phase3.Phase3View(_context(tmp_path, {"params": {}}))
    plain = _frame(view, "Did the cameras agree?")["copy"]
    assert "varied between photos by about 0.04 squares" in plain
    assert "This number is never 0" in plain and "There is no fixed good value" in plain
    assert "same board at a similar distance" in plain
    assert "compare it only between runs with the same board" in compose(
        _frame(view, "Did the cameras agree?"), 2)
    view.close()


@pytest.mark.gui
def test_phase3_agreement_levels_are_this_views_not_pycamset_log_check(tmp_path, monkeypatch):
    """The 10 mm / 5 deg levels apply to the RMS centre/turn spread, a larger statistic than
    pyCamSet's own rig check; the copy must not call them pyCamSet's warning level."""
    _app()
    from dataclasses import replace
    from pyCamSet.gui import education_phase3

    base = _available_derivation()
    wide = SimpleNamespace(**{**vars(base.rig_spreads[0]), "position_scatter_mm": 12.0})
    # The "smaller, may stay quiet" comparison holds only for metre-coordinate targets.
    metric = replace(base, saved_point_units="m", rebuilt_point_units="m", rig_spreads=(wide,))
    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _context: metric)
    monkeypatch.setattr(education_phase3, "read_trace", lambda *_: {"status": "unavailable"})
    view = education_phase3.Phase3View(_context(tmp_path, {"params": {}}))
    agreement = _frame(view, "Did the cameras agree?")
    copy = compose(agreement, 1)
    assert "Overall answer: No" in copy
    assert "varied by 12 mm across photos. That is more than 10 mm, this view's warning level." in copy
    assert "pyCamSet's warning level" not in compose(agreement, 3)
    assert "its log may show no warning" in copy
    assert "standard deviation of the relative translation length" in compose(agreement, 3)
    view.close()


@pytest.mark.gui
def test_phase3_log_check_comparison_for_millimetre_target(tmp_path, monkeypatch):
    """RigConsistencyReport reads pose translations as metres; for a millimetre target
    (flat PuzzleBoard) its logged values are 1000x too large, so it is not the quieter check."""
    _app()
    from dataclasses import replace
    from pyCamSet.gui import education_phase3

    base = _available_derivation()
    calm = SimpleNamespace(**{**vars(base.rig_spreads[0]), "position_scatter_mm": 0.02})
    derivation = replace(base, saved_point_units="mm", rebuilt_point_units="mm", rig_spreads=(calm,))
    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _context: derivation)
    monkeypatch.setattr(education_phase3, "read_trace", lambda *_: {"status": "unavailable"})
    view = education_phase3.Phase3View(_context(tmp_path, {"params": {}}))
    agreement = _frame(view, "Did the cameras agree?")
    full = compose(agreement, 3)
    assert "smaller" not in full
    assert "its log may show no warning" not in full
    assert "1000 times too large" in compose(agreement, 1)
    assert "can warn even when this view answers Yes" in compose(agreement, 1)
    assert "as if the target coordinates were metres" in full
    view.close()


def test_phase3_geometry_rows_for_a_telecentric_rig_show_orientations_not_centres():
    """On a real telecentric rig, the solver's camera block is a
    rotation vector only, so every geometry row read 'camera transforms are invalid'."""
    pytest.importorskip("PySide6")
    import numpy as np
    from pyCamSet.gui.education_phase3 import (
        _TELECENTRIC_STEP, _geometry_snapshot_rows, _rotation_only)

    cameras = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
    poses = np.array([[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]])
    frame = (None, cameras, poses)
    assert _rotation_only(frame)
    rows = _geometry_snapshot_rows(frame, "Saved final result")
    assert [r[0] for r in rows] == ["Saved final result: camera 1 orientation",
                                    "Saved final result: camera 2 orientation",
                                    "Saved final result: Photo 1 (index 0) target-pose translation"]
    assert all("invalid" not in r[2] for r in rows)
    assert "no projection centre" in rows[0][2]
    assert "telecentric" in _TELECENTRIC_STEP
    # a pinhole rig keeps the camera-centre rows
    pinhole = (None, np.zeros((2, 6)), poses)
    assert not _rotation_only(pinhole)
    assert "centre" in _geometry_snapshot_rows(pinhole, "S")[0][0]


# --- Single-face images (telecentric): the optional recovery's stage (on a real telecentric
# --- rig, 17 of 99 images were left out; every camera saw one cube face of each) ---

def _single_face_record(enabled: bool = True) -> dict:
    candidates = [
        {"camera": name, "tilt": tilt, "own_camera_rms_px": 0.4,
         "all_cameras_rms_px": 1.1, "all_cameras_tilt_kept_rms_px": kept}
        for name, pair in (("cam0", (1.2, 46.0)), ("cam3", (38.0, 1.3))) for tilt, kept in zip("+-", pair)]
    if not enabled:
        return {"enabled": False, "telecentric": True, "recovered": [], "teaching": {}, "not_recovered": [
            {"image": 7, "reason": "recovery off: every camera saw one face (recoverable in principle)",
             "cameras": ["cam0", "cam1"]},
            {"image": 9, "reason": "seen by one camera only: the tilt cannot be resolved", "cameras": ["cam2"]}]}
    return {"enabled": True, "telecentric": True,
            "recovered": [{"image": 12, "rms_px": 1.1, "cameras": ["cam0", "cam3"]},
                          {"image": 14, "rms_px": 2.5, "cameras": ["cam0", "cam3"]}],
            "not_recovered": [
                {"image": 40, "reason": "seen by one camera only: the tilt cannot be resolved", "cameras": ["cam5"]},
                {"image": 41, "reason": "best fit 33.10 px exceeds the 20 px limit", "cameras": ["cam1", "cam2"]}],
            "teaching": {"12": candidates}}


def _single_face_view(tmp_path, monkeypatch, record):
    from pyCamSet.gui import education_phase3

    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _context: _available_derivation())
    monkeypatch.setattr(education_phase3, "read_trace",
                        lambda *_args: {"status": "unavailable", "reason": "missing sidecar"})
    metadata = {"params": {}, "diagnostics": {"D3.1_n_missing_poses": 2,
                                              "D3.14_single_face_recovery": record}}
    return education_phase3.Phase3View(_context(tmp_path, metadata))


def _stage_texts(view, key):
    view.show_stage(key)
    texts = {}
    for level in (1, 2, 3):
        view.stage_view.set_level(level)
        # Summarise the figure now: the next level redraws (and clears) it.
        texts[level] = (view.stage_view.caption.text(), _table_cells(view),
                        {ax.get_title(loc="left").split(chr(10))[0]: len(ax.patches)
                         for ax in view.stage_view.figure().axes})
    return texts


@pytest.mark.gui
def test_single_face_stage_follows_the_first_guess_with_its_figures_and_the_table(tmp_path, monkeypatch):
    _app()
    from test_education_copy_lint import _rules, learner_copy_findings, lint_copy

    view = _single_face_view(tmp_path, monkeypatch, _single_face_record())
    assert view.stage_keys() == ["graph", "first", "single", "solver", "result"]
    assert view.stage_view.stepper.stage_name(2) == "One face"
    assert view.single_face_image.accessibleName() == "Recovered single-face image"
    assert view.single_face_image.count() == 2
    texts = _stage_texts(view, "single")
    # A/B: the tilt ambiguity, drawn from geometry (one camera, two cameras: UX-17, one letter
    # each); C: this photo's candidate fits (photos numbered from 1: stored index 12 = Photo 13).
    l1_axes, l2_axes = texts[1][2], texts[2][2]
    assert "A  One camera" in l1_axes and "B  Two cameras" in l1_axes
    assert l1_axes["C  Photo 13: both tilts, every camera"] == 2 * 4  # own camera + all, 4 candidates
    assert l2_axes["C  Photo 13: both tilts, every camera"] == 3 * 4  # L2 adds the free refinement
    assert "D  Every single-face photo" in l2_axes
    assert view.single_face_image.itemText(0).startswith("Photo 13")
    view.single_face_image.setCurrentIndex(1)  # image 14 has no recorded candidates
    panel = next(ax for ax in view.stage_view.figure().axes
                 if ax.get_title(loc="left").startswith("C"))
    assert "were not recorded" in panel.texts[0].get_text()
    # The table: placed and left-out photos with their reasons; D3.14 only from L2.
    l1_cells, l2_cells = texts[1][1], texts[2][1]
    assert "Photo 13 (index 12)" in l1_cells and "Photo 42 (index 41)" in l1_cells
    assert "placed" in l1_cells
    assert "best fit 33.10 px exceeds the 20 px limit" in l1_cells
    assert not any("D3." in cell for cell in l1_cells) and any("D3.14" in cell for cell in l2_cells)
    assert any("Hartley and Zisserman" in cell for cell in texts[3][1])
    # Short captions, different per level, additive, plain at L1, clean under the copy lint.
    copies = [texts[level][0] for level in (1, 2, 3)]
    assert len(set(copies)) == 3 and copies[0] in copies[1] and copies[1] in copies[2]
    assert copies[0].count(". ") <= 1
    assert "a fit under 20 px only has to pick the tilt" in copies[1]  # ED2-10
    assert "take photos where a camera sees two faces" in copies[2]
    # UX-18: the method and its citation live in the Numbers drawer, not the caption (SCI-15:
    # section 6.3 for the affine camera, chapter 14 for the reversal).
    assert "Numbers drawer" in copies[2] and "Hartley" not in copies[2]
    assert any("chapter 14" in cell for cell in texts[3][1])
    for level, (copy, cells, _axes) in texts.items():
        assert learner_copy_findings(copy, level) == [], (level, copy)
        assert lint_copy(copy, "education_phase3.py", f"single_face.L{level}", _rules()) == []
        assert lint_copy(" ".join(cells), "education_phase3.py", "single_face.table", _rules()) == []
    view.close()


@pytest.mark.gui
def test_single_face_stage_with_recovery_off_shows_the_note_and_the_left_out_images(tmp_path, monkeypatch):
    _app()
    from test_education_copy_lint import learner_copy_findings

    view = _single_face_view(tmp_path, monkeypatch, _single_face_record(enabled=False))
    texts = _stage_texts(view, "single")
    copy, cells, axes = texts[1]  # axes: {panel title: bar count}
    assert "recovery was off for this run" in copy
    # The quoted control is the Phase 3 checkbox's own label.
    assert "‘Recover images where each camera sees only one cube face (telecentric)’" in copy
    assert "1 of them was seen by two or more cameras" in copy
    assert "Photo 8 (index 7)" in cells and "Photo 10 (index 9)" in cells and "left out" in cells
    assert axes  # the tilt illustration and the (empty) candidate panel are still drawn
    assert not view.single_face_image.isEnabled()
    assert learner_copy_findings(copy, 1) == []
    view.close()


@pytest.mark.gui
def test_no_single_face_stage_without_a_record_or_anything_to_report(tmp_path, monkeypatch):
    _app()
    record = {"enabled": False, "telecentric": True, "recovered": [], "not_recovered": [], "teaching": {}}
    view = _single_face_view(tmp_path, monkeypatch, record)
    assert len(view._evidence) == 6 and "single" not in view.stage_keys()
    view.close()


@pytest.mark.gui
def test_telecentric_rig_does_not_judge_camera_positions_and_small_targets_scale_the_level(
        tmp_path, monkeypatch):
    """SCI2-5 / ED2-12: a telecentric rig's centre spread mixes in unseen depth, so only the
    turn is judged; otherwise the position level is 10 mm or 10 % of the target, the smaller."""
    _app()
    from pyCamSet.gui import education_phase3

    base = _available_derivation()
    monkeypatch.setattr(education_phase3, "derive_phase3", lambda _context: base)
    monkeypatch.setattr(education_phase3, "read_trace",
                        lambda *_args: {"status": "unavailable", "reason": "missing sidecar"})
    view = education_phase3.Phase3View(_context(tmp_path, {"params": {}}))
    try:
        assert view._position_rule()["level_mm"] == 10.0  # nothing known about the target
        view._visual = SimpleNamespace(telecentric=True)
        rule = view._position_rule()
        assert rule["judge"] is False
        frame = view._decision_frame_copy()
        text = compose(frame, 3)
        assert "position = not judged (telecentric cameras do not see" in text
        assert "Positions are not judged" in text and "10 mm" not in frame["copy"]
        assert frame["answer"] == "Yes"  # the turn and the single-photo checks decide
        cube = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float) * 10
        view._visual = SimpleNamespace(telecentric=False)
        monkeypatch.setattr(education_phase3, "_target_geometry",
                            lambda _v: (cube.mean(axis=0), float(np.linalg.norm(np.ptp(cube, 0)))))
        monkeypatch.setattr(view, "_mm_per_unit", lambda: 1.0)
        assert view._position_rule()["level_mm"] == pytest.approx(1.7)
        assert "exceeds 1.7 mm" in compose(view._decision_frame_copy(), 2)
    finally:
        view.close()


def test_playback_step_labels_say_where_a_fractional_position_sits():
    """ED3-8 / P3-1: "the start", "step k", or "between … and step k" (the short form for
    narrow panels keeps the title from eliding), and the interpolated per-camera means are
    log-linear between the two recorded states."""
    pytest.importorskip("PySide6")
    from pyCamSet.gui.education_phase3 import _step_label, interpolated_camera_means

    nits = [0, 3, 7]
    assert _step_label(nits, 0.0) == "the start"
    assert _step_label(nits, 1.0) == "step 3"
    assert _step_label(nits, 0.5) == "between the start and step 3"
    assert _step_label(nits, 0.5, short=True) == "between start & step 3"
    assert _step_label(nits, 1.5) == "between step 3 and step 7"
    assert _step_label(nits, 2.0) == "step 7"
    states = [SimpleNamespace(camera_mean_px=np.array([100.0, 4.0])),
              SimpleNamespace(camera_mean_px=np.array([1.0, 1.0]))]
    half = interpolated_camera_means(states, 0.5)
    assert half[0] == pytest.approx(10.0) and half[1] == pytest.approx(2.0)
    assert np.allclose(interpolated_camera_means(states, 1.0), [1.0, 1.0])
