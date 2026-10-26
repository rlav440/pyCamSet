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
