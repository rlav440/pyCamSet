'''Purpose: Regression tests for Phase 4 self-calibration quality gates.
Status: Active; keeps Phase 4 fail-closed and provenance-preserving.
Future: Extend the real-corpus contract as additional good and underconstrained rigs are characterised.
'''

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyCamSet.workflow import phase4
from pyCamSet.workflow.workspace import WorkspaceManager, workspace_path_for


class _Detection:
    cam_names = ["cam0", "cam1"]
    max_ims = 2


class _Handler:
    cam_names = ["cam0", "cam1"]
    fixed_inds = [0, 1, 2]
    visible_feature_mask = np.array([True, True, True])
    problem_opts = {"fixed_pose": 0, "ref_cam": 0, "ref_pose": 0}
    detection = _Detection()

    def get_detection_data(self, flatten=True):
        return np.array([
            [0, 0, 0], [1, 0, 1],
            [0, 1, 0], [1, 1, 1],
        ])


def _optimisation(*, success=True, finite=True):
    values = np.ones(8)
    if not finite:
        values[0] = np.nan
    return SimpleNamespace(
        x=values,
        fun=np.ones(8),
        success=success,
        status=3,
        message="xtol",
        nfev=4,
    )


def test_phase4_quality_gate_records_gauge_and_coverage_contract():
    gate = phase4._quality_gate(
        _optimisation(), _Handler(),
        {"success": True}, np.ones((4, 2)),
        initial_euclid=10.0, final_euclid=2.0, observation_count=4,
        phase3_status="complete", phase3_run_id="p3",
    )

    assert gate["status"] == "complete"
    assert gate["finite_parameters"]
    assert gate["finite_residuals"]
    assert gate["camera_coverage"]
    assert gate["image_coverage"]
    assert gate["gauge"]["fixed_target_point_count"] == 3
    assert gate["gauge"]["fixed_pose"] == 0
    assert gate["gauge"]["ref_cam"] == 0
    assert gate["gauge"]["ref_pose"] == 0


def test_phase4_quality_gate_blocks_nonfinite_or_unsuccessful_solve():
    gate = phase4._quality_gate(
        _optimisation(success=False, finite=False), _Handler(),
        {"success": False}, np.ones((4, 2)),
        initial_euclid=10.0, final_euclid=2.0, observation_count=4,
        phase3_status="complete", phase3_run_id="p3",
    )

    assert gate["status"] == "incomplete"
    assert "optimiser parameters are non-finite" in gate["blocking_flags"]
    assert "solver did not report successful termination" in gate["blocking_flags"]


def test_phase4_quality_gate_blocks_unobserved_image_indices():
    class _SparseDetection(_Detection):
        max_ims = 3

    class _SparseHandler(_Handler):
        detection = _SparseDetection()

    gate = phase4._quality_gate(
        _optimisation(), _SparseHandler(),
        {"success": True}, np.ones((4, 2)),
        initial_euclid=10.0, final_euclid=2.0, observation_count=4,
        phase3_status="complete", phase3_run_id="p3",
    )

    assert gate["image_coverage"] is False
    assert gate["missing_images"] == [2]
    assert "image observation graph has missing image indices" in gate["blocking_flags"]


def test_phase4_initial_per_image_errors_fall_back_to_solver_stats():
    handler = SimpleNamespace(initial_per_im_error=np.array([], dtype=float))
    stats = {
        "per_pose_initial_error_px": [
            {"pose": 0, "initial_error_px": 3.5},
            {"pose": 2, "initial_error_px": 5.25},
        ]
    }

    result = phase4._initial_per_image_errors(handler, stats)

    assert result.tolist() == [3.5, 5.25]


def test_solver_stats_record_reprojection_objective_costs():
    from pyCamSet.optimisation.optimisation_handling import get_bundle_adjustment_stats

    class _ResidualHandler:
        def get_base_residual_count(self):
            return 4

    result = get_bundle_adjustment_stats(
        SimpleNamespace(
            success=True, status=2, message="ok", nfev=3,
            fun=np.zeros(4),
        ),
        np.ones(2), np.array([3.0, 0.0, 0.0, 4.0]), 0.1,
        param_handler=_ResidualHandler(),
    )

    assert result["initial_reprojection_cost"] == 12.5
    assert result["final_reprojection_cost"] == 0.0


def test_phase4_quality_gate_distinguishes_objective_from_mean_error():
    gate = phase4._quality_gate(
        _optimisation(), _Handler(),
        {
            "success": True,
            "initial_reprojection_cost": 10.0,
            "final_reprojection_cost": 5.0,
        }, np.ones((4, 2)),
        initial_euclid=2.0, final_euclid=3.0, observation_count=4,
        phase3_status="complete", phase3_run_id="p3",
    )

    assert gate["objective_cost_reduced"] is True
    assert gate["error_reduced"] is False
    assert "final reprojection error did not improve finitely" in gate["blocking_flags"]


@pytest.mark.parametrize(
    "phase3_status,phase3_run_id",
    [(None, None), (None, "p3"), ("complete", None), ("failed", "p3")],
)
def test_phase4_provenance_must_be_identified_and_complete(
    phase3_status, phase3_run_id
):
    gate = phase4._quality_gate(
        _optimisation(), _Handler(), {"success": True}, np.ones((4, 2)),
        initial_euclid=10.0, final_euclid=2.0, observation_count=4,
        phase3_status=phase3_status, phase3_run_id=phase3_run_id,
    )
    assert gate["status"] == "incomplete"
    assert not gate["phase3_provenance_complete"]
    assert "Phase 3 input lacks an identified run with complete status" in gate["blocking_flags"]


def test_phase4_provenance_accepts_identified_complete_run():
    gate = phase4._quality_gate(
        _optimisation(), _Handler(), {"success": True}, np.ones((4, 2)),
        initial_euclid=10.0, final_euclid=2.0, observation_count=4,
        phase3_status="complete", phase3_run_id="p3",
    )
    assert gate["status"] == "complete"
    assert gate["phase3_provenance_complete"]


def test_incomplete_phase4_alias_is_diagnostic_only(tmp_path):
    from pyCamSet.gui.assess_calibration import resolve_run_camset_artifact

    camset = tmp_path / "incomplete.camset"
    camset.write_text("diagnostic", encoding="utf-8")
    run = {
        "phase": "phase4",
        "status": "incomplete",
        "artifacts": {"optimised_camset": str(camset)},
    }

    assert resolve_run_camset_artifact(run) == camset
    assert resolve_run_camset_artifact(run, accepted_only=True) is None


def test_accepted_phase4_requires_a_phase4_output_artifact(tmp_path):
    from pyCamSet.gui.assess_calibration import resolve_run_camset_artifact

    phase3_input = tmp_path / "initial.camset"
    phase3_input.write_text("input", encoding="utf-8")
    run = {
        "phase": "phase4",
        "status": "complete",
        "artifacts": {"initial_camset": str(phase3_input)},
    }

    # General diagnostic lookup keeps legacy fallback; accepted use must not.
    assert resolve_run_camset_artifact(run) == phase3_input
    assert resolve_run_camset_artifact(run, accepted_only=True) is None


@pytest.mark.parametrize("phase4_status", ["incomplete", None])
def test_exporter_rejects_incomplete_or_unknown_phase4_alias(
    tmp_path, monkeypatch, phase4_status
):
    pytest.importorskip("PySide6")  # the lean install has no GUI toolkit
    from PySide6.QtWidgets import QApplication, QCheckBox, QTabWidget

    from pyCamSet.gui import export_calibration_tab as export_module
    from pyCamSet.gui.export_calibration_tab import ExportCalibrationTab

    QApplication.instance() or QApplication([])
    workspace = WorkspaceManager(workspace_path_for(tmp_path))
    tab = ExportCalibrationTab(QTabWidget(), QCheckBox(), QCheckBox(), workspace)
    messages = []
    tab._terminal = SimpleNamespace(append_line=messages.append)
    run = {
        "phase": "phase4",
        "status": phase4_status,
        "run_id": "p4-incomplete",
        "artifacts": {"optimised_camset": str(tmp_path / "diagnostic.camset")},
    }
    tab._run_selector = SimpleNamespace(get_selected=lambda: [run])
    tab._selected_format = lambda: "colmap"
    exports = []
    monkeypatch.setattr(export_module, "camset_to_colmap", lambda *_args: exports.append(True))
    monkeypatch.setattr(export_module, "camset_to_apde", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(export_module, "load_CameraSet", lambda _path: object())
    monkeypatch.setattr(export_module, "_PYCAMSET_OK", True)

    tab._export_selected()

    assert not exports
    assert any("Phase 4 status is not complete" in message for message in messages)


def test_exporter_rejects_complete_phase4_without_output_artifact(
    tmp_path, monkeypatch
):
    pytest.importorskip("PySide6")  # the lean install has no GUI toolkit
    from PySide6.QtWidgets import QApplication, QCheckBox, QTabWidget

    from pyCamSet.gui import export_calibration_tab as export_module
    from pyCamSet.gui.export_calibration_tab import ExportCalibrationTab

    QApplication.instance() or QApplication([])
    workspace = WorkspaceManager(workspace_path_for(tmp_path))
    tab = ExportCalibrationTab(QTabWidget(), QCheckBox(), QCheckBox(), workspace)
    messages = []
    tab._terminal = SimpleNamespace(append_line=messages.append)
    phase3_input = tmp_path / "initial.camset"
    phase3_input.write_text("input", encoding="utf-8")
    run = {
        "phase": "phase4",
        "status": "complete",
        "run_id": "p4-missing-output",
        "artifacts": {"initial_camset": str(phase3_input)},
    }
    tab._run_selector = SimpleNamespace(get_selected=lambda: [run])
    tab._selected_format = lambda: "colmap"
    exports = []
    monkeypatch.setattr(export_module, "camset_to_colmap", lambda *_args: exports.append(True))
    monkeypatch.setattr(export_module, "camset_to_apde", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(export_module, "load_CameraSet", lambda _path: object())
    monkeypatch.setattr(export_module, "_PYCAMSET_OK", True)

    tab._export_selected()

    assert not exports
    assert any("no camset artifact found" in message for message in messages)


def test_phase4_run_persists_quality_gate_disposition(tmp_path, monkeypatch):
    workspace = WorkspaceManager(workspace_path_for(tmp_path))
    output = Path(tmp_path) / "self_calibrated_cameras.camset"
    output.write_text("placeholder", encoding="utf-8")

    def fake_solve(params, run_dir, phase3_camset, phase3_run, log):
        return output, {"quality_gate": {"status": "incomplete"}}, None

    monkeypatch.setattr(phase4, "_solve", fake_solve)
    metadata = phase4.run(
        {"fixed_params": None, "problem_options": {}, "threads": 1},
        workspace,
        phase3_run={"run_id": "p3"},
        phase3_camset=output,
    )

    assert metadata["status"] == "incomplete"
    assert metadata["inputs"]["phase3_run_id"] == "p3"
    assert workspace.find_run("phase4", metadata["run_id"])["status"] == "incomplete"


@pytest.mark.parametrize("options,expected", [({}, 100), ({"max_nfev": 37}, 37)])
def test_phase4_api_default_is_100_and_explicit_value_is_preserved(
    tmp_path, monkeypatch, options, expected
):
    detection = SimpleNamespace(cam_names=["cam0"])
    previous_handler = SimpleNamespace(target=object(), detection=detection)
    previous_cams = SimpleNamespace(
        calibration_handler=previous_handler,
        get_names=lambda: ["cam0"],
    )
    monkeypatch.setattr(phase4, "load_CameraSet", lambda _path: previous_cams)
    captured = {}

    def stop_after_option_capture(*args, options, **kwargs):
        captured.update(options)
        raise RuntimeError("stop after checking options")

    monkeypatch.setattr(phase4, "solve", stop_after_option_capture)
    with pytest.raises(RuntimeError, match="stop after checking options"):
        phase4._solve(
            {"problem_options": options}, tmp_path, tmp_path / "phase3.camset",
            {"run_id": "p3", "status": "complete"}, lambda _line: None,
        )
    assert captured["max_nfev"] == expected


def test_self_calibration_gauge_uses_target_point_data_units():
    from pyCamSet.optimisation.standard_bundle_handler import _gauge_square_size

    target = SimpleNamespace(
        square_size=4.2857142857,
        point_data=np.array([[[0.0, 0.0, 0.0],
                              [0.0042857143, 0.0, 0.0],
                              [0.0, 0.0042857143, 0.0]]]),
    )

    assert np.isclose(_gauge_square_size(target), 0.0042857143)


