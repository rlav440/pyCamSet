'''Verify the Phase 2 education storyboard stays tied to saved run evidence.
'''
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def _app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _context(tmp_path: Path, metadata: dict, *, run_id: str = "phase2-run"):
    run_dir = tmp_path / ".pycamset_workspace" / "phase2_runs" / run_id
    run_dir.mkdir(parents=True)
    manager = SimpleNamespace(workspace_path=run_dir.parent.parent)
    return SimpleNamespace(run_id=run_id, run_dir=run_dir, metadata=metadata, manager=manager)


def _frame(view, title: str):
    return next(frame for frame in view._frames if frame["title"] == title)


@pytest.mark.gui
def test_initial_seed_shows_nonzero_distortion_and_telecentric_parameters(tmp_path):
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View

    context = _context(tmp_path, {
        "params": {"lens_model": "telecentric", "record_phase2_per_view": False},
        "diagnostics": {
            "D2.2_intrinsics": {"camA": {"fx": 230.0, "fy": 240.0, "cx": 320.0, "cy": 240.0}},
            "D2.3_distortion": {"camA": {"coeffs": [0.125]}},
            "D2.6_per_view_reprojection": {},
        },
    })
    view = Phase2View(context)
    seed = _frame(view, "Initial camera fit")
    assert "not a later Phase 3/4 solution" in seed["copy"]
    assert seed["rows"] == [("camA", "fx=230, fy=240, cx=320, cy=240", "coefficient 1 = 0.125")]
    assert "multi-camera rig" in seed["copy"]
    assert not any("optimisation step" in row[2] for row in seed["rows"])
    view.close()


@pytest.mark.gui
def test_camera_subset_fresh_pass_never_substitutes_phase1_points_or_photo(tmp_path):
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View

    context = _context(tmp_path, {
        "inputs": {"phase1_run_id": "wrong-to-use"},
        "artifacts": {"education_used_detections": {
            "schema_version": 1, "source": "fresh_pass_not_saved", "path": None,
            "fresh_pass_reasons": ["camera_subset"], "cache_hit": False,
        }},
        "params": {"record_phase2_per_view": False},
    })
    view = Phase2View(context)
    copy = _frame(view, "Input selection")["copy"]
    assert "camera subset" in copy
    assert "Origin: detected during this run (not saved)" in copy
    assert "Raw unavailable (detections made during Phase 2, not saved)" in copy
    assert "Phase 1" not in copy
    view.close()


@pytest.mark.gui
def test_fresh_cache_hit_words_reuse_as_pass_and_does_not_show_photo(tmp_path):
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View

    context = _context(tmp_path, {
        "artifacts": {"education_used_detections": {
            "schema_version": 1, "source": "fresh_pass_not_saved", "path": None,
            "fresh_pass_reasons": ["camera_subset"], "cache_hit": True,
        }},
        "params": {"record_phase2_per_view": True},
    })
    view = Phase2View(context)
    copy = _frame(view, "Input selection")["copy"]
    assert "This pass reused cached detections" in copy
    assert "originating run is unknown" in copy
    assert copy.count("No photo is shown") == 1
    absent = _frame(view, "Per-view walkthrough")
    assert "unavailable" in absent["copy"]
    view.close()


@pytest.mark.gui
def test_input_selection_reports_saved_camera_image_board_point_contributions(tmp_path, monkeypatch):
    _app()
    import numpy as np
    from pyCamSet.gui.education_phase2 import Phase2View
    from pyCamSet.utils import saving

    detection = SimpleNamespace(cam_names=["camA"])
    target = object()
    camset = SimpleNamespace(calibration_handler=SimpleNamespace(detection=detection, target=target))
    monkeypatch.setattr(saving, "load_CameraSet", lambda _path: camset)
    monkeypatch.setattr("pyCamSet.gui.education_phase2.resolve_artifact",
                        lambda *_args: Path("saved.camset"))
    monkeypatch.setattr("pyCamSet.utils.intrinsics_report._board_views",
                        lambda *_args: ([np.zeros((1, 6, 3)), np.zeros((1, 6, 3))],
                                        [], [0, 0], [True]))
    context = _context(tmp_path, {
        "params": {"min_detections_per_board": 6},
        "artifacts": {"education_used_detections": {
            "source": "filtered_pickle", "path": "filtered.pickle"}},
        "threshold_pruning": {"n_removed_observations": 3},
    })
    view = Phase2View(context)
    input_rows = _frame(view, "Input selection")["rows"]
    assert input_rows == [("camA", "2 usable board views / 1 images",
                           "12 used points; 3 points removed")]
    view.close()


@pytest.mark.gui
@pytest.mark.data
@pytest.mark.slow
def test_real_phase2_recorded_views_feed_saved_detection_walkthrough(data_dir, tmp_path):
    import shutil
    from PySide6.QtWidgets import QApplication
    from pyCamSet.workflow import phase1, phase2
    from pyCamSet.workflow.education_trace import RunContext
    from pyCamSet.workflow.workspace import WorkspaceManager
    from pyCamSet.gui.education_phase2 import Phase2View

    images = tmp_path / "images"
    shutil.copytree(data_dir / "calibration_charuco", images)
    workspace = WorkspaceManager(tmp_path / "workspace")
    target = {"type": "ChArUco", "num_squares_x": 20,
              "num_squares_y": 20, "square_size": 4, "legacy": False}
    first = phase1.run({"f_loc": str(images), "target": target,
                        "selected_cameras": [], "n_lim": None,
                        "caching": False, "upscale_factor": 1,
                        "rescale_and_gamma": False, "manifest_digest": True},
                       workspace, log=lambda _line: None)
    assert first["error"] is None
    second = phase2.run({"f_loc": str(images), "target": target,
                         "selected_cameras": [], "n_lim": None,
                         "caching": False, "high_distortion": False,
                         "fixed_params": None, "min_detections_per_board": 12,
                         "lens_model": "pinhole", "record_phase2_per_view": True},
                        workspace, log=lambda _line: None, phase1_run=first)
    assert second["error"] is None
    context = RunContext.from_run(workspace, {
        **second, "_run_dir": str(workspace.run_dir("phase2", second["run_id"]))},
        "phase2")
    app = QApplication.instance() or QApplication([])
    view = Phase2View(context)
    assert view._trace["status"] == "available"
    assert view.current_stage() == 0
    contributions = view._input_rows(
        second["artifacts"]["education_used_detections"])
    assert "threshold-pruned points not recorded" in str(contributions)
    # Phase 2 saves its camset without a calibration handler; the stages must
    # still read the used detections from the run record, not fail.
    assert all("usable board views" in row[1] for row in contributions), contributions
    walk = next(frame for frame in view._frames
                if frame["title"] == "Per-view walkthrough")
    assert any(walk["camera_groups"].values())
    residuals = _frame(view, "Fit error and decision")["rows"]
    assert residuals and all(row[2].startswith("matches D2.6") for row in residuals), residuals
    walk_rows = [row for rows in walk["camera_groups"].values() for row in rows]
    # The recalculated RMS agrees with the saved one to about 1e-6 relative,
    # so every real row must say it matches, never "differs" beside two
    # identical displayed numbers.
    assert all(row[5] == "recalculated; matches the value Phase 2 saved"
               for row in walk_rows), walk_rows[:3]
    assert all(row[1] == row[2] for row in walk_rows), walk_rows[:3]
    cells = [str(cell) for frame in view._frames for row in frame["rows"] for cell in row]
    cells += [str(cell) for row in walk_rows for cell in row]
    assert not any("Error" in cell or "Exception" in cell for cell in cells)
    # The visual stages draw this run: every camera's recomputed per-photo error equals the
    # saved one, and the 3D stage poses every board view.
    view.stage_view.set_current_stage(3)
    for index in range(view.camera_picker.count()):
        view.camera_picker.setCurrentIndex(index)
        chips = {chip.label: chip.value for chip in view.stage_view.chips.chips()}
        assert chips["the saved errors"] == "✓ matches", (index, chips)
    view.stage_view.set_current_stage(1)
    data = view._camera_data(view._selected_camera())
    assert len(view._scene_items(data)) == sum(f is not None for f in data.fits) > 0
    view.close()


@pytest.mark.gui
def test_trace_walkthrough_steps_by_view_and_opens_on_first_view(tmp_path):
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View
    from pyCamSet.workflow.education_trace import write_trace

    context = _context(tmp_path, {"params": {"record_phase2_per_view": True}})
    write_trace(context.run_dir, context.run_id, "phase2", "phase2_per_view", {
        "cost_semantics": "calibrate_zhang/calibrate_telecentric per-view RMS (px)",
        "view_order": "usable-view order returned by the selected seed calibration",
        "cameras": [{"camera": "camA", "views": [
            {"view_index": 0, "pose_target_to_camera": [[1, 0, 0, 0]] * 4, "per_view_rms_px": 0.2},
            {"view_index": 4, "pose_target_to_camera": [[1, 0, 0, 1]] * 4, "per_view_rms_px": 0.4},
        ]}],
    }, enabled=True)
    view = Phase2View(context)
    assert view.current_stage() == 0
    walkthrough = _frame(view, "Per-view walkthrough")
    assert "not optimisation steps" in walkthrough["copy"]
    assert [row[0] for row in walkthrough["camera_groups"]["camA"]] == ["Board view 0", "Board view 4"]
    assert [row[2] for row in walkthrough["camera_groups"]["camA"]] == ["0.2", "0.4"]
    # The caption says the rows are views, not optimisation steps, once; the
    # status cell does not repeat it on every row.
    assert not any("optimisation step" in row[5] for row in walkthrough["camera_groups"]["camA"])
    assert all("unavailable" in row[1] and "unavailable" in row[3]
               for row in walkthrough["camera_groups"]["camA"])
    assert "step through" not in walkthrough["copy"]
    # The stage stepper moves between the five visual stages; the recorded views are the
    # L3 Numbers of "Board in 3D" (here refused: no saved inputs, so no 3D picture).
    view.stage_view.set_level(3)
    view.step_forward()
    assert view.current_stage() == 1
    table = view.stage_view.numbers.table
    assert table.rowCount() == 2
    assert [table.item(r, 0).text() for r in range(2)] == ["Board view 0", "Board view 4"]
    assert not view.camera_picker.isHidden()
    view.step_back()
    assert view.current_stage() == 0
    view.close()


@pytest.mark.gui
def test_trace_pose_and_homography_are_recalculated_from_saved_detections(tmp_path, monkeypatch):
    _app()
    import cv2
    import numpy as np
    from pyCamSet.gui.education_phase2 import Phase2View
    from pyCamSet.utils import intrinsics_report, saving
    from pyCamSet.workflow.education_trace import write_trace

    objects = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                         [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
    pose = np.eye(4)
    pose[2, 3] = 10.0
    intrinsic = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    rotation, _ = cv2.Rodrigues(pose[:3, :3])
    distortion = np.array([0.12, -0.03, 0.001, -0.002, 0.004])
    images, _ = cv2.projectPoints(objects.reshape(-1, 3), rotation, pose[:3, 3], intrinsic, distortion)
    images = images.reshape(1, -1, 2)
    camera = SimpleNamespace(intrinsic=intrinsic, distortion_coefs=distortion)
    real_project_points = cv2.projectPoints
    passed_distortion = []

    def record_projection(points, rotation_vector, translation, matrix, coefficients):
        passed_distortion.append(np.asarray(coefficients).copy())
        return real_project_points(points, rotation_vector, translation, matrix, coefficients)

    monkeypatch.setattr(cv2, "projectPoints", record_projection)

    class CameraSet(SimpleNamespace):
        def __getitem__(self, _name):
            return camera

    camset = CameraSet(calibration_handler=SimpleNamespace(
        detection=object(), target=object()))
    monkeypatch.setattr(saving, "load_CameraSet", lambda _path: camset)
    monkeypatch.setattr(intrinsics_report, "_board_views",
                        lambda *_args: ([objects], [images], [0], [True]))
    monkeypatch.setattr("pyCamSet.gui.education_phase2.resolve_artifact",
                        lambda *_args: Path("saved.camset"))

    context = _context(tmp_path, {"params": {"lens_model": "pinhole"},
                                  "artifacts": {"initial_camset": "saved.camset"}})
    write_trace(context.run_dir, context.run_id, "phase2", "phase2_per_view", {
        "cameras": [{"camera": "camA", "views": [{
            "view_index": 0, "pose_target_to_camera": pose.tolist(), "per_view_rms_px": 0.0,
        }]}],
    }, enabled=True)
    view = Phase2View(context)
    row = _frame(view, "Per-view walkthrough")["camera_groups"]["camA"][0]
    assert row[1] == "0"
    assert row[2] == "0"
    assert row[3].startswith("recalculated:")
    # any: a noise simulation left on a worker thread by an earlier test may also call this
    assert any(np.array_equal(d, distortion) for d in passed_distortion)
    assert row[4].startswith("recorded seed target→camera pose (4×4):")
    assert "0,0,1,10" in row[4]
    assert row[5] == "recalculated; matches the value Phase 2 saved"
    view.close()


@pytest.mark.gui
def test_multiboard_homography_follows_nonzero_board_view_index(tmp_path, monkeypatch):
    _app()
    import numpy as np
    from pyCamSet.gui.education_phase2 import Phase2View
    from pyCamSet.utils import intrinsics_report, saving
    from pyCamSet.workflow.education_trace import write_trace

    board_objects = [np.full((1, 4, 3), value, dtype=np.float32)
                     for value in (0, 1, 2, 3)]
    board_images = [np.full((1, 4, 2), value, dtype=np.float32)
                    for value in (0, 1, 2, 3)]
    image_indices = [0, 0, 5, 5]
    camset = SimpleNamespace(calibration_handler=SimpleNamespace(
        detection=object(), target=object()))
    monkeypatch.setattr(saving, "load_CameraSet", lambda _path: camset)
    monkeypatch.setattr(intrinsics_report, "_board_views",
                        lambda *_args: (board_objects, board_images, image_indices, []))
    monkeypatch.setattr("pyCamSet.gui.education_phase2.resolve_artifact",
                        lambda *_args: Path("saved.camset"))
    fitted = []

    def homography(obj, img):
        fitted.append((int(obj[0, 0, 0]), int(img[0, 0, 0])))
        return np.eye(3) * (int(obj[0, 0, 0]) + 1)

    monkeypatch.setattr("pyCamSet.calibration.zhang.homography_of", homography)
    context = _context(tmp_path, {"params": {"lens_model": "pinhole"},
                                  "artifacts": {"initial_camset": "saved.camset"}})
    write_trace(context.run_dir, context.run_id, "phase2", "phase2_per_view", {
        "cameras": [{"camera": "camA", "views": [{
            "view_index": 1, "pose_target_to_camera": np.eye(4).tolist(),
            "per_view_rms_px": 0.1,
        }]}],
    }, enabled=True)
    view = Phase2View(context)
    row = _frame(view, "Per-view walkthrough")["camera_groups"]["camA"][0]
    assert fitted == [(1, 1)]
    assert row[3].startswith("recalculated:")
    assert "board view 1:" in row[3]
    assert "board view 2:" not in row[3] and "board view 3:" not in row[3]
    view.close()


@pytest.mark.gui
def test_telecentric_walkthrough_does_not_use_pinhole_projection(tmp_path, monkeypatch):
    _app()
    import cv2
    import numpy as np
    from pyCamSet.gui.education_phase2 import Phase2View

    def forbidden_projection(*_args, **_kwargs):
        raise AssertionError("telecentric route must not call cv2.projectPoints")

    monkeypatch.setattr(cv2, "projectPoints", forbidden_projection)
    view = Phase2View(_context(tmp_path, {"params": {"lens_model": "telecentric"}}))
    error, status = view._pose_only_error("camA", 1, np.eye(4).tolist(), 0.0)
    assert error == "unavailable"
    assert status.startswith("not recalculated")  # no saved inputs here; never the pinhole route
    assert view._homography_status("camA", 1) == "not applicable (telecentric affine fit)"
    view.close()


@pytest.mark.gui
def test_walkthrough_uses_camera_selector_and_stays_within_six_frames(tmp_path):
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View
    from pyCamSet.workflow.education_trace import write_trace

    context = _context(tmp_path, {"params": {"high_distortion": True,
                                               "record_phase2_per_view": True},
                                  "artifacts": {"phase3_solved_distortion": {
                                      "run_id": "phase3-9", "path": "solution.camset",
                                      "coefficients": [0.01]}}})
    write_trace(context.run_dir, context.run_id, "phase2", "phase2_per_view", {
        "cameras": [{"camera": f"cam{i}", "views": []} for i in range(3)],
    }, enabled=True)
    artifact = context.manager.workspace_path / "solution.camset"
    artifact.write_bytes(b"linked Phase 3/4 artifact")
    view = Phase2View(context)
    assert len(view._frames) <= 6
    walkthrough = _frame(view, "Per-view walkthrough")
    assert list(walkthrough["camera_groups"]) == ["cam0", "cam1", "cam2"]
    view.stage_view.set_current_stage(1)
    assert view.camera_picker.isHidden() is False
    # UX-4: the shared camera picker keeps the 0-based index as item data, the name in the text.
    assert [view.camera_picker.itemData(i) for i in range(view.camera_picker.count())] == [
        0, 1, 2]
    assert [view.camera_picker.itemText(i) for i in range(view.camera_picker.count())] == [
        "Camera 1 · cam0", "Camera 2 · cam1", "Camera 3 · cam2"]
    assert view._selected_camera() == "cam0"
    assert "Phase 3/4 distortion from run phase3-9" in _frame(view, "Initial camera fit")["copy"]
    view.close()


@pytest.mark.gui
def test_pruned_high_distortion_run_does_not_claim_second_search(tmp_path):
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View

    context = _context(tmp_path, {
        "params": {"high_distortion": True, "n_lim": 20},
        "threshold_pruning": {"n_removed_observations": 2},
        "artifacts": {"filtered_detection_pickle": "filtered.pickle"},
    })
    view = Phase2View(context)
    copy = _frame(view, "High-distortion second pass")["copy"]
    assert "Phase 1's saved detections, with 2 observations removed" in copy
    assert "no second search ran" in copy
    assert "on in settings, not applied" in copy
    assert "ran detection a second time" not in copy
    view.close()


@pytest.mark.gui
def test_high_distortion_copy_requires_c1_second_pass_record(tmp_path):
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View

    missing = Phase2View(_context(tmp_path, {
        "params": {"high_distortion": True, "n_lim": 20},
        "artifacts": {"education_used_detections": {
            "source": "phase1_pickle", "path": "phase1.pickle",
            "fresh_pass_reasons": [], "cache_hit": None,
        }},
    }, run_id="missing-record"))
    missing_copy = _frame(missing, "High-distortion second pass")["copy"]
    assert "on in settings, not applied" in missing_copy
    assert "does not identify a high-distortion second pass" in missing_copy
    assert "ran detection a second time" not in missing_copy
    missing.close()

    recorded = Phase2View(_context(tmp_path, {
        "params": {"high_distortion": True, "n_lim": 20},
        "artifacts": {"education_used_detections": {
            "source": "fresh_pass_not_saved", "path": None,
            "fresh_pass_reasons": ["high_distortion_second_pass"], "cache_hit": False,
        }},
    }, run_id="recorded-pass"))
    recorded_copy = _frame(recorded, "High-distortion second pass")["copy"]
    assert "ran detection a second time (up to 20 images per camera)" in recorded_copy
    assert "using first-pass camera estimates to guide detection" in recorded_copy
    assert "L3-only" not in recorded_copy
    assert "how the detections moved between the two passes is not available" in recorded_copy
    assert "does not use the first camera estimate" not in recorded_copy
    recorded.close()


@pytest.mark.gui
def test_upscaled_high_distortion_view_is_unavailable_and_high_distortion_off_is_absent(tmp_path):
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View

    upscaled = Phase2View(_context(tmp_path, {
        "params": {"high_distortion": True, "upscale_factor": 2},
        "artifacts": {"education_used_detections": {
            "source": "fresh_pass_not_saved", "path": None,
            "fresh_pass_reasons": ["high_distortion_second_pass"], "cache_hit": False,
        }},
    }, run_id="upscaled"))
    unavailable = _frame(upscaled, "High-distortion second pass")["copy"]
    assert "unavailable (reason)" in unavailable
    assert "no Phase 1 points or photo are substituted" in unavailable
    upscaled.close()

    ordinary = Phase2View(_context(tmp_path, {
        "params": {"high_distortion": False},
    }, run_id="ordinary"))
    assert not any("High-distortion" in frame["title"] for frame in ordinary._frames)
    ordinary.close()


@pytest.mark.gui
@pytest.mark.parametrize("high_distortion", [False, True])
def test_phase2_frames_follow_a1_order_and_end_on_the_decision(tmp_path, high_distortion):
    """A1 Phase 2: per-view contribution, then initial fit, then the decision last;
    an optional high-distortion note comes before the decision, never after it."""
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View

    view = Phase2View(_context(tmp_path, {"params": {"high_distortion": high_distortion}}))
    try:
        titles = [frame["title"] for frame in view._frames]
        expected = ["Input selection", "Run status", "Per-view walkthrough", "Initial camera fit",
                    *(["High-distortion second pass"] if high_distortion else []),
                    "Fit error and decision"]
        assert titles == expected
        assert len(titles) <= 6
    finally:
        view.close()


@pytest.mark.gui
def test_unmarked_legacy_and_recording_off_report_contributions_not_recorded(tmp_path):
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View

    context = _context(tmp_path, {"params": {}})
    view = Phase2View(context)
    assert _frame(view, "Input selection")["copy"].startswith(
        f"Run {context.run_id}.")
    assert "Input used not recorded" in _frame(view, "Input selection")["copy"]
    assert _frame(view, "Per-view walkthrough")["copy"] == (
        "Per-view contributions not recorded for this run. To see per-view contributions next "
        "time, tick ‘Record per-view contributions for this run’ in Phase 2 settings before "
        "you run it. This run is not deficient.")
    view.close()


@pytest.mark.gui
def test_absent_status_uses_shared_blocking_reasons_without_inventing_status(tmp_path):
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View

    view = Phase2View(_context(tmp_path, {"error": "calibration failed", "params": {}}))
    copy = _frame(view, "Run status")["copy"]
    assert "status was not recorded" in copy
    assert "calibration failed" in copy
    assert "shared run-quality rule" in copy
    failed_seed = _frame(view, "Initial camera fit")
    assert "failed before producing first lens estimates" in failed_seed["copy"]
    assert failed_seed["rows"] == []
    view.close()


@pytest.mark.gui
def test_absent_status_with_no_blocking_reasons_is_not_called_complete(tmp_path):
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View

    view = Phase2View(_context(tmp_path, {"params": {}}))
    copy = _frame(view, "Run status")["copy"]
    assert "no blocking reasons" in copy
    assert "not a recorded status" in copy
    view.close()


@pytest.mark.gui
def test_unmarked_promoted_phase2_input_suppresses_stage_claims(tmp_path):
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View

    view = Phase2View(_context(tmp_path, {
        "diagnostics": {"promoted_from": {"trial_id": "trial-7"},
                        "D2.2_intrinsics": {"camA": {"fx": 123.0}}},
        "params": {},
    }))
    seed = _frame(view, "Phase 2 input")
    assert "may contain later-stage cameras" in seed["copy"]
    assert seed["rows"] == []
    residuals = _frame(view, "Fit error and decision")["rows"]
    assert residuals[0][2] == "unmarked promoted camset may contain later-stage state"
    view.close()


@pytest.mark.gui
def test_pose_only_recalculation_is_checked_against_d26(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View
    from pyCamSet.utils import intrinsics_report, saving

    expected = {"camA": {"rms_px": [0.25, 0.75]}}
    context = _context(tmp_path, {
        "params": {"min_detections_per_board": 12},
        "diagnostics": {"D2.6_per_view_reprojection": expected},
        "artifacts": {"initial_camset": "saved.camset"},
    })
    monkeypatch.setattr(saving, "load_CameraSet", lambda _path: SimpleNamespace(
        calibration_handler=SimpleNamespace(detection=object(), target=object())))
    monkeypatch.setattr(intrinsics_report, "report_initial_calibration", lambda *args: SimpleNamespace(
        per_view={"camA": {"rms_px": [0.25, 0.75]}}))
    view = Phase2View(context)
    monkeypatch.setattr("pyCamSet.gui.education_phase2.resolve_artifact",
                        lambda *_args: Path("saved.camset"))
    rows = view._residual_rows()
    assert rows == [("camA · Photo 1 (index 0)", "0.25", "matches D2.6"),
                    ("camA · Photo 2 (index 1)", "0.75", "matches D2.6")]
    view.close()


@pytest.mark.gui
def test_image_without_usable_board_is_not_a_d26_mismatch(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View
    from pyCamSet.utils import intrinsics_report, saving

    # JSON stores the producer's NaN as null; the recalculation keeps NaN.
    expected = {"camA": {"rms_px": [0.25, None, 0.5, None]}}
    context = _context(tmp_path, {
        "params": {"min_detections_per_board": 12},
        "diagnostics": {"D2.6_per_view_reprojection": expected},
        "artifacts": {"initial_camset": "saved.camset"},
    })
    monkeypatch.setattr(saving, "load_CameraSet", lambda _path: SimpleNamespace(
        calibration_handler=SimpleNamespace(detection=object(), target=object())))
    monkeypatch.setattr(intrinsics_report, "report_initial_calibration", lambda *args: SimpleNamespace(
        per_view={"camA": {"rms_px": [0.25, float("nan"), float("nan"), 0.3],
                           "image_indices": [0, 1, 2, 3]}}))
    view = Phase2View(context)
    monkeypatch.setattr("pyCamSet.gui.education_phase2.resolve_artifact",
                        lambda *_args: Path("saved.camset"))
    rows = view._residual_rows()
    assert rows == [
        ("camA · Photo 1 (index 0)", "0.25", "matches D2.6"),
        ("camA · Photo 2 (index 1)", "no usable board in this image", "matches D2.6: none recorded"),
        # One-sided gaps are real disagreements and stay flagged.
        ("camA · Photo 3 (index 2)", "unavailable", "does not match D2.6"),
        ("camA · Photo 4 (index 3)", "0.3", "does not match D2.6"),
    ]
    view.close()


@pytest.mark.gui
def test_walkthrough_labels_board_views_with_their_source_image(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View
    from pyCamSet.utils import intrinsics_report, saving

    context = _context(tmp_path, {"params": {"min_detections_per_board": 12},
                                  "artifacts": {"initial_camset": "saved.camset"}})
    monkeypatch.setattr(saving, "load_CameraSet", lambda _path: SimpleNamespace(
        calibration_handler=SimpleNamespace(detection=object(), target=object())))
    # Image 1 is skipped (too few detections); image 2 holds two board faces.
    monkeypatch.setattr(intrinsics_report, "_board_views",
                        lambda *_args: ([0, 0, 0], [0, 0, 0], [0, 2, 2], [True] * 3))
    monkeypatch.setattr("pyCamSet.gui.education_phase2.resolve_artifact",
                        lambda *_args: Path("saved.camset"))
    view = Phase2View(context)
    # Photos are numbered from 1 for the learner (SCI-8); views keep their stored index.
    assert [view._board_view_label("camA", k) for k in (0, 1, 2, 3)] == [
        "Board view 0 · Photo 1", "Board view 1 · Photo 3",
        "Board view 2 · Photo 3", "Board view 3"]
    view.close()


@pytest.mark.gui
def test_phase2_levels_are_additive_and_the_decision_names_the_warning_level(tmp_path):
    """A1 sections 1, 3 and 8: plain L1 copy, orientation, decision and opt-in next step."""
    _app()
    from test_education_copy_lint import learner_copy_findings
    from pyCamSet.gui.education_levels import compose
    from pyCamSet.gui.education_phase2 import Phase2View
    from pyCamSet.utils.intrinsics_report import HIGH_INITIAL_RMS_PX

    context = _context(tmp_path, {
        "params": {"lens_model": "pinhole"},
        "report": {"per_camera": [{"name": "camA", "n_views": 12, "rms_px": 0.42},
                                  {"name": "camB", "n_views": 3, "rms_px": 7.5},
                                  {"name": "camC", "n_views": 0, "rms_px": None}]},
        "diagnostics": {
            "D2.2_intrinsics": {"camA": {"fx": 900.0, "fy": 905.0, "cx": 320.0, "cy": 240.0}},
            "D2.3_distortion": {"camA": {"coeffs": [0.1, -0.02, 0.0, 0.0, 0.001]}},
        },
    })
    view = Phase2View(context)
    try:
        assert view.stage_view.stepper.accessibleName().startswith(
            "Phase 2 — Calibrate individual cameras")
        fit = _frame(view, "Initial camera fit")
        assert fit["rows"][0][2] == "k1 = 0.1, k2 = -0.02, p1 = 0, p2 = 0, k3 = 0.001"
        assert "{" not in compose(fit, 3) and "k1 = 0.1" in compose(fit, 2)
        decision = _frame(view, "Fit error and decision")["copy"]
        assert decision.startswith("Did each camera see the board in enough photos, with a fit error "
                                   "below pyCamSet's warning level?")
        level = f"{float(HIGH_INITIAL_RMS_PX):g} px warning level"
        assert f"camA: board seen in 12 photo(s); fit error 0.42 px, at or below pyCamSet's {level}" in decision
        assert f"camB: board seen in 3 photo(s); fit error 7.5 px, above pyCamSet's {level}" in decision
        assert "camC: no usable board view" in decision
        assert "does not check how varied the board angles were" in decision
        assert "What to do next:" in decision
        # camA saved fitted (nonzero) coefficients, so no unfitted-distortion cause line.
        assert "which Phase 2 does not fit" not in decision
        assert "D2.6" not in decision and "D2.6" in compose(_frame(view, "Fit error and decision"), 2)
        rendered = {}
        for level_index in (1, 2, 3):
            view.stage_view.set_level(level_index)
            rendered[level_index] = []
            for index in range(5):
                view.stage_view.set_current_stage(index)
                text = view.stage_view.caption.text()
                rendered[level_index].append(text)
                assert learner_copy_findings(text, level_index) == [], (level_index, index, text)
                table = view.stage_view.numbers.table
                headers = [table.horizontalHeaderItem(column).text()
                           for column in range(table.columnCount())]
                assert not any(learner_copy_findings(header, 3) for header in headers), headers
        for low, high in ((1, 2), (2, 3)):
            assert all(a in b for a, b in zip(rendered[low], rendered[high]))
        # The decision stage answers the question from the saved summary: camB is above the
        # warning level, camC has no usable view.
        view.stage_view.set_level(1)
        view.stage_view.set_current_stage(4)
        caption = view.stage_view.caption.text()
        assert "2 of 3 cameras" in caption and "Camera 2 is above the" in caption
        table = view.stage_view.numbers.table
        rows = [[table.item(r, c).text() for c in range(5)] for r in range(table.rowCount())]
        assert [row[4] for row in rows] == [
            "goes on", f"above the {float(HIGH_INITIAL_RMS_PX):g} px warning level",
            "dropped: no usable view"]
    finally:
        view.close()


@pytest.mark.gui
def test_zero_distortion_is_named_unfitted_not_an_estimate(tmp_path):
    """Zhang's closed-form seed fits no distortion and hands on zeros (zhang.py).

    The copy must not call those zeros estimates, and the decision must name
    unmodelled distortion as a possible cause of a high fit error.
    """
    _app()
    from pyCamSet.gui.education_levels import compose
    from pyCamSet.gui.education_phase2 import Phase2View

    context = _context(tmp_path, {
        "params": {"lens_model": "pinhole"},
        "report": {"per_camera": [{"name": "camA", "n_views": 12, "rms_px": 7.5}]},
        "diagnostics": {
            "D2.2_intrinsics": {"camA": {"fx": 900.0, "fy": 905.0, "cx": 320.0, "cy": 240.0}},
            "D2.3_distortion": {"camA": {"coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]}},
        },
    })
    view = Phase2View(context)
    try:
        fit = _frame(view, "Initial camera fit")
        assert "Phase 2 does not fit lens distortion" in fit["copy"]
        assert "zero placeholders" in fit["copy"]
        assert "saved first estimates" not in fit["copy"]
        assert "camA: k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0 (saved as zero, not fitted)" in compose(fit, 2)
        decision = _frame(view, "Fit error and decision")["copy"]
        cause = decision.index("lens distortion, which Phase 2 does not fit")
        assert cause < decision.index("What to do next:")
        assert "Phase 3 solves the distortion" in decision
        view.stage_view.set_current_stage(2)
        assert "Phase 2 fits no lens distortion" in view.stage_view.caption.text()
        chips = {chip.label: chip.value for chip in view.stage_view.chips.chips()}
        assert chips["lens distortion"] == "not fitted"
    finally:
        view.close()


@pytest.mark.gui
def test_phase2_recorded_views_are_l3_numbers_without_raw_matrices(tmp_path):
    """The recorded first-estimate views are method detail: the L3 Numbers of "Board in 3D"
    list them with their check, never the raw 3x3/4x4 matrices."""
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View

    view = Phase2View(_context(tmp_path, {"params": {}}))
    try:
        walkthrough = _frame(view, "Per-view walkthrough")
        walkthrough["camera_groups"] = {"camA": [("Board view 0", "0.2", "0.2",
                                                  "recalculated: board view 0: 1,0,0; 0,1,0; 0,0,1",
                                                  "recorded seed target→camera pose (4×4): ...",
                                                  "recalculated; matches the value Phase 2 saved")]}
        assert view._walkthrough_rows("camA", 3) == [
            ("Board view 0", "0.2", "0.2", "recalculated; matches the value Phase 2 saved")]
    finally:
        view.close()


@pytest.mark.gui
@pytest.mark.parametrize(("used", "recovery"), [
    ({"schema_version": 1, "source": "fresh_pass_not_saved", "path": None,
      "fresh_pass_reasons": ["camera_subset"], "cache_hit": False},
     "The fresh detections were not saved; re-run Phase 2 to create a new result."),
    ({"schema_version": 1, "source": "fresh_pass_not_saved", "path": None,
      "fresh_pass_reasons": ["camera_subset"], "cache_hit": True},
     "Untick ‘Use detection cache when fallback-detecting’ in Phase 2 and run Phase 2 again, "
     "then continue from its new result."),
    (None, "Re-run Phase 2 to create a new result that records its input, then continue from that run."),
])
def test_input_not_recorded_refusals_end_with_recovery(tmp_path, used, recovery):
    """A1 section 9: every Phase 2 input refusal ends with an actionable rerun."""
    _app()
    from pyCamSet.gui.education_phase2 import Phase2View

    metadata = {"params": {}}
    if used is not None:
        metadata["artifacts"] = {"education_used_detections": used}
    view = Phase2View(_context(tmp_path, metadata))
    copy = _frame(view, "Input selection")["copy"]
    assert copy.endswith(recovery), copy
    view.close()


@pytest.mark.gui
def test_evidence_table_values_are_readable_not_silently_elided(tmp_path):
    """Shared Phase 2-4 table helper: tooltips carry full text, columns fit, long text wraps."""
    _app()
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QTableWidget
    from pyCamSet.gui.education_levels import fill_evidence_table, setup_evidence_table

    table = QTableWidget()
    setup_evidence_table(table)
    table.setColumnCount(3)
    table.setHorizontalHeaderLabels(["Recalculated fit error (px)", "Source", "Evidence"])
    long_path = "D:/workspace/.pycamset_workspace/phase2_runs/run-123/initial_cameras.camset"
    fill_evidence_table(table, [("0.25", long_path, "could not be checked")])
    assert table.wordWrap()
    assert table.textElideMode() == Qt.TextElideMode.ElideMiddle
    assert table.item(0, 1).toolTip() == long_path
    assert table.horizontalHeaderItem(0).toolTip() == "Recalculated fit error (px)"
    header = table.horizontalHeader()
    # The first column fits its header instead of Qt's default 100 px.
    assert header.sectionSize(0) >= header.sectionSizeFromContents(0).width()
    assert header.sectionSize(1) <= max(320, header.sectionSizeFromContents(1).width())
    table.deleteLater()


@pytest.mark.gui
def test_phase2_l1_tables_carry_no_codes_or_parameter_names(tmp_path, monkeypatch):
    """A1 section 1 and Phase 2 frame 4: D2.6, fx/fy/cx/cy and k1… are L2 table content."""
    _app()
    import re
    from pyCamSet.gui.education_phase2 import Phase2View
    from pyCamSet.utils import intrinsics_report, saving

    context = _context(tmp_path, {
        "params": {"min_detections_per_board": 12, "lens_model": "pinhole"},
        "diagnostics": {
            "D2.6_per_view_reprojection": {"camA": {"rms_px": [0.25, 0.5]}},
            "D2.2_intrinsics": {"camA": {"fx": 900.0, "fy": 905.0, "cx": 320.0, "cy": 240.0},
                                "camB": {"fx": 910.0, "fy": 915.0, "cx": 321.0, "cy": 241.0}},
            "D2.3_distortion": {"camA": {"coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]},
                                "camB": {"coeffs": [0.1, -0.02, 0.0, 0.0, 0.001]}},
        },
        "artifacts": {"initial_camset": "saved.camset"},
    })
    monkeypatch.setattr(saving, "load_CameraSet", lambda _path: SimpleNamespace(
        calibration_handler=SimpleNamespace(detection=object(), target=object())))
    monkeypatch.setattr(intrinsics_report, "report_initial_calibration", lambda *args: SimpleNamespace(
        per_view={"camA": {"rms_px": [0.25, 0.75]}}))
    monkeypatch.setattr("pyCamSet.gui.education_phase2.resolve_artifact",
                        lambda *_args: Path("saved.camset"))
    view = Phase2View(context)
    jargon = re.compile(r"\bD2\.|\bfx=|\bk1\b")

    def cells():
        table = view.stage_view.numbers.table
        headers = [table.horizontalHeaderItem(column).text()
                   for column in range(table.columnCount())]
        return headers, [[table.item(row, column).text()
                          for column in range(table.columnCount())]
                         for row in range(table.rowCount())]

    try:
        view.stage_view.set_level(1)
        view.stage_view.set_current_stage(2)  # Lens model
        headers, rows = cells()
        assert headers == ["Camera", "Saved lens estimate", "Saved lens distortion"]
        assert rows[0] == ["camA", "focal length (x, y): 900, 905 px; image centre (x, y): 320, 240 px",
                           "none fitted (zero)"]
        assert rows[1][2] != "none fitted (zero)"
        for stage in range(5):
            view.stage_view.set_current_stage(stage)
            headers, rows = cells()
            text = " ".join(headers + [cell for row in rows for cell in row]
                            + [view.stage_view.caption.text()])
            assert not jargon.search(text), (stage, text)
        view.stage_view.set_level(2)
        view.stage_view.set_current_stage(2)
        headers, rows = cells()
        assert headers[1] == "Saved intrinsic parameters"
        assert rows[0][1] == "fx=900, fy=905, cx=320, cy=240" and rows[1][2].startswith("k1 = 0.1")
        assert view._residual_rows() == [("camA · Photo 1 (index 0)", "0.25", "matches D2.6"),
                                         ("camA · Photo 2 (index 1)", "0.75", "does not match D2.6")]
    finally:
        view.close()


def test_phase2_recalculated_rms_compares_at_pixel_precision():
    pytest.importorskip("PySide6")  # the helper lives in a Qt module (Qt-import convention: lean env has no PySide6)
    from pyCamSet.gui.education_phase2 import _rms_comparison

    # projectPoints and the seed calibration agree to about 1e-6 relative.
    assert _rms_comparison(6.561 * (1 + 1.7e-6), 6.561) == (
        "recalculated; matches the value Phase 2 saved")
    assert _rms_comparison(0.0, 0.0) == "recalculated; matches the value Phase 2 saved"
    # A real disagreement is named with its size.
    assert _rms_comparison(6.573, 6.561) == (
        "recalculated; differs from the value Phase 2 saved by 0.012 px")
    assert _rms_comparison(6.561, None) == "recalculated; Phase 2 saved no value to compare"
    assert _rms_comparison(float("nan"), 6.561) == (
        "recalculated; differs from the value Phase 2 saved")
