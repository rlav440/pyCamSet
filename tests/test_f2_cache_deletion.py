'''Verify a real Phase 1 cache hit cannot shift photo identity after deletion.
'''
from __future__ import annotations

import shutil
from pathlib import Path

import pytest


@pytest.mark.gui
@pytest.mark.data
@pytest.mark.slow
def test_phase1_cache_hit_after_photo_deletion_never_shifts_overlay(
        data_dir, tmp_path, monkeypatch):
    """A cache-reused run keeps its saved identity and refuses stale imagery."""
    pytest.importorskip("PySide6")

    from PySide6.QtWidgets import QApplication
    from pyCamSet.gui.education_phase1 import Phase1EducationView
    from pyCamSet.workflow import phase1
    from pyCamSet.workflow.education_trace import RunContext
    from pyCamSet.workflow.workspace import WorkspaceManager, workspace_path_for

    images = tmp_path / "calibration_charuco"
    shutil.copytree(data_dir / "calibration_charuco", images)
    manager = WorkspaceManager(workspace_path_for(images))
    params = {
        "f_loc": str(images),
        "target": {
            "type": "ChArUco", "num_squares_x": 20, "num_squares_y": 20,
            "square_size": 4, "legacy": False,
        },
        "selected_cameras": [],
        "n_lim": None,
        "caching": True,
        "upscale_factor": 1,
        "rescale_and_gamma": False,
        "manifest_digest": True,
        "record_phase1_trace": True,
        "threads": 2,
    }

    first = phase1.run(params, manager, log=lambda _line: None)
    assert first["error"] is None
    cached = phase1.run(params, manager, log=lambda _line: None)
    assert cached["error"] is None
    assert cached["diagnostics"]["education_cache"]["reused"] is True

    manifest = cached["diagnostics"]["education_manifest"]
    camera_name, entries = next(iter(manifest["cameras"].items()))
    deleted_name = entries[0]["name"]
    camera_folder = images / camera_name
    (camera_folder / deleted_name).unlink()
    remaining_names = {path.name for path in camera_folder.iterdir() if path.is_file()}
    assert remaining_names, "the test needs a later image that could otherwise be misattributed"

    saved = {
        **cached,
        "_run_dir": manager.run_dir("phase1", cached["run_id"]),
    }
    context = RunContext.from_run(manager, saved, "phase1")
    checked = context.phase1_source_check()
    assert checked.status == "mismatch"

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(phase1, "detect_datapoints_in_imfile",
                        lambda **_kwargs: pytest.fail("opening the view re-ran detection"))
    view = Phase1EducationView(context)
    view._camera.setCurrentText(camera_name)
    view._image.setValue(0)  # the deleted photo's own slot (the view opens on a busy image)
    view._set_stage(3)
    assert not view.figure().axes[0].images
    caption = view._caption.text()
    assert deleted_name in caption
    assert all(name not in caption for name in remaining_names), (
        "a remaining folder photo was substituted for the deleted cache source")
    assert "mismatch" in view.provenance_text()

    view.close()
    view.deleteLater()
    app.processEvents()
