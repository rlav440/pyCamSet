"""Exercise Phase 1 education-trace envelope and index contracts.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pyCamSet.workflow import phase1
from pyCamSet.workflow.education_trace import read_trace, trace_path, write_trace
from pyCamSet.workflow.workspace import WorkspaceManager


def test_phase1_run_persists_opt_in_trace_and_leaves_default_off(tmp_path, monkeypatch):
    payload = {
        "camera": "cam0", "image_index": 0, "trace_status": "captured",
        "stages": [{"name": "charuco_corners", "corner_ids": [1, 2]}],
    }
    monkeypatch.setattr(phase1, "make_run_id", lambda: "run-trace")
    monkeypatch.setattr(phase1, "_detect", lambda _params, _log: (
        object(), [], {"_phase1_trace_records": [payload]},
        {"blocking_flags": []}))
    monkeypatch.setattr(phase1, "save_detections", lambda path, *_args: path)

    enabled_workspace = WorkspaceManager(tmp_path / "enabled")
    enabled = phase1.run(
        {"f_loc": str(tmp_path), "record_phase1_trace": True},
        workspace=enabled_workspace)
    enabled_dir = enabled_workspace.run_dir("phase1", "run-trace")
    assert enabled["artifacts"]["education_trace"] == str(trace_path(enabled_dir))
    assert read_trace(enabled_dir, "run-trace", "phase1")["records"][0]["payload"] == payload

    disabled_workspace = WorkspaceManager(tmp_path / "disabled")
    disabled = phase1.run({"f_loc": str(tmp_path)}, workspace=disabled_workspace)
    disabled_dir = disabled_workspace.run_dir("phase1", "run-trace")
    assert "education_trace" not in disabled.get("artifacts", {})
    assert not trace_path(disabled_dir).exists()


@pytest.mark.gui
@pytest.mark.data
def test_real_charuco_run_trace_reaches_saved_view_without_changing_detections(
        data_dir, tmp_path):
    import shutil
    import numpy as np
    from PySide6.QtWidgets import QApplication
    from pyCamSet.gui.education_phase1 import Phase1EducationView
    from pyCamSet.workflow.education_trace import RunContext
    from pyCamSet.workflow.detections import extract_detection
    from pyCamSet.utils.saving import load_pickle

    images = tmp_path / "images"
    shutil.copytree(data_dir / "calibration_charuco", images)
    workspace = WorkspaceManager(tmp_path / "workspace")
    params = {"f_loc": str(images), "target": {
        "type": "ChArUco", "num_squares_x": 20, "num_squares_y": 20,
        "square_size": 4.0, "marker_fraction": 0.8,
        "marker_backend": "aruco1", "a_dict": 3, "legacy": False},
        "caching": False, "n_lim": None, "upscale_factor": 1,
        "rescale_and_gamma": False, "threads": 2,
        "manifest_digest": True}
    off = phase1.run(dict(params), workspace=workspace)
    on = phase1.run(dict(params, record_phase1_trace=True), workspace=workspace)
    assert off["error"] is None and on["error"] is None
    off_det = extract_detection(load_pickle(Path(off["artifacts"]["detected_datapoints_pickle"])))
    on_det = extract_detection(load_pickle(Path(on["artifacts"]["detected_datapoints_pickle"])))
    np.testing.assert_array_equal(off_det.get_data(), on_det.get_data())
    off_dir = workspace.run_dir("phase1", off["run_id"])
    on_dir = workspace.run_dir("phase1", on["run_id"])
    assert not trace_path(off_dir).exists()
    trace = read_trace(on_dir, on["run_id"], "phase1")
    assert trace["status"] == "available"
    assert len(trace["records"]) == len(on_det.cam_names) * on_det.max_ims
    indexed = [(record["payload"]["camera"], record["payload"]["image_index"])
               for record in trace["records"]]
    assert set(indexed) == {(name, image_index) for name in on_det.cam_names
                            for image_index in range(on_det.max_ims)}
    assert len(indexed) == len(set(indexed))
    assert any(record["payload"]["stages"] for record in trace["records"])
    app = QApplication.instance() or QApplication([])
    view = Phase1EducationView(RunContext.from_run(
        workspace, {**on, "_run_dir": str(on_dir)}, "phase1"))
    assert view._trace["status"] == "available"
    assert view._stage_index == 0
    view.close()



def test_phase1_trace_roundtrip_keeps_camera_image_and_ordered_stages(tmp_path):
    run_dir = tmp_path / "phase1_runs" / "p1"
    run_dir.mkdir(parents=True)
    payload = {
        "camera": "cam0",
        "image_index": 3,
        "stages": [
            {"name": "aruco_markers", "marker_ids": [4, 9]},
            {"name": "charuco_corners", "corner_ids": [2, 3, 7]},
        ],
    }
    written = write_trace(run_dir, "p1", "phase1", "stage", payload,
                          enabled=True)
    assert written == run_dir / "education_trace.json"
    loaded = read_trace(run_dir, "p1", "phase1")
    assert loaded["status"] == "available"
    assert loaded["records"][0]["payload"] == payload


def test_phase1_trace_budget_overflow_is_explicit_not_silent(tmp_path):
    run_dir = tmp_path / "phase1_runs" / "p1"
    run_dir.mkdir(parents=True)
    for index in range(2):
        write_trace(run_dir, "p1", "phase1", "stage",
                    {"camera": "cam0", "image_index": index, "stages": []},
                    enabled=True, max_records=1)
    loaded = read_trace(run_dir, "p1", "phase1")
    assert loaded["status"] == "unavailable"
    assert loaded["reason"] == "trace records were evicted"
    assert loaded["evicted_kinds"] == ["stage"]
