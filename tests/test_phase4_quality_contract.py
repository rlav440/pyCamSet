'''Purpose: Regression tests for Phase 4 self-calibration quality gates.
Status: Active; keeps Phase 4 fail-closed and provenance-preserving.
Future: Extend the real-corpus contract as additional good and underconstrained rigs are characterised.
'''

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

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


def test_self_calibration_gauge_uses_target_point_data_units():
    from pyCamSet.optimisation.standard_bundle_handler import _gauge_square_size

    target = SimpleNamespace(
        square_size=4.2857142857,
        point_data=np.array([[[0.0, 0.0, 0.0],
                              [0.0042857143, 0.0, 0.0],
                              [0.0, 0.0042857143, 0.0]]]),
    )

    assert np.isclose(_gauge_square_size(target), 0.0042857143)
