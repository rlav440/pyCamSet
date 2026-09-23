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
    )

    assert gate["objective_cost_reduced"] is True
    assert gate["error_reduced"] is False
    assert "final reprojection error did not improve finitely" in gate["blocking_flags"]


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


def test_fixed_camera_warm_start_keeps_only_free_parameters():
    from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler, TemplateBundleHandler

    class _Primitive:
        def __init__(self, *, extr_unfixed):
            self.intr = np.array([[10.0, 11.0], [20.0, 21.0]])
            self.extr = np.array([[30.0, 31.0, 32.0], [40.0, 41.0, 42.0]])
            self.poses = np.array([[50.0, 51.0, 52.0, 53.0, 54.0, 55.0]])
            self.bundle_pts = np.array([[60.0, 61.0, 62.0]])
            self.intr_unfixed = np.array([False, False])
            self.extr_unfixed = np.asarray(extr_unfixed, dtype=bool)
            self.poses_unfixed = np.array([True])
            self.bdpt_unfixed = np.array([True])
            self.pose_end = 3 * int(self.extr_unfixed.sum()) + 6
            self.bdpt_end = self.pose_end + 3

        def return_bundle_primitives(self, _params):
            return self.intr, self.extr, self.poses, self.bundle_pts

    previous_handler = TemplateBundleHandler.__new__(TemplateBundleHandler)
    previous_handler.missing_poses = None
    previous_handler.target = SimpleNamespace(point_data=np.array([[[1.0, 2.0, 3.0]]]))
    previous_handler.bundlePrimitive = _Primitive(extr_unfixed=[True, True])
    previous_cams = SimpleNamespace(
        calibration_handler=previous_handler,
        calibration_params=np.arange(15.0),
    )

    current_handler = TemplateBundleHandler.__new__(TemplateBundleHandler)
    current_handler.bundlePrimitive = _Primitive(extr_unfixed=[False, True])
    current_handler.feat_unfixed = np.array([True, True, True])

    SelfBundleHandler.set_from_templated_camset(current_handler, previous_cams)

    assert current_handler.initial_params.shape == (12,)
    assert current_handler.initial_params[:9].tolist() == [40.0, 41.0, 42.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0]


def test_self_calibration_output_does_not_mutate_input_camera_set(monkeypatch):
    from pyCamSet.cameras import Camera, CameraSet
    from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler

    source = CameraSet(camera_dict={"cam0": Camera(name="cam0")})
    set_extrinsic_calls = []
    original_set_extrinsic = Camera.set_extrinsic

    def record_set_extrinsic(self, extrinsic):
        set_extrinsic_calls.append(self)
        return original_set_extrinsic(self, extrinsic)

    monkeypatch.setattr(Camera, "set_extrinsic", record_set_extrinsic)

    class _Primitive:
        def return_bundle_primitives(self, _params):
            return (
                np.array([[1200.0, 500.0, 1200.0, 500.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                np.array([[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]]),
                np.empty((0, 6)),
                np.empty((0, 3)),
            )

    handler = SelfBundleHandler.__new__(SelfBundleHandler)
    handler.camset = source
    handler.cam_names = ["cam0"]
    handler.bundlePrimitive = _Primitive()
    handler.apply_gauge_transform = lambda proj, extr, poses, points: (
        proj, extr, poses, points)

    output = handler.get_camset(np.zeros(1))

    assert output["cam0"] is not source["cam0"]
    assert len(set_extrinsic_calls) == 2
    assert output["cam0"] in set_extrinsic_calls
    assert np.allclose(source["cam0"].extrinsic, np.eye(4))
    assert not np.shares_memory(output["cam0"].intrinsic, source["cam0"].intrinsic)
    assert np.allclose(output["cam0"].position, [-1.0, -2.0, -3.0])
    drift = phase4._camera_drift(source, output)
    assert drift["available"]
    assert drift["cameras"]["cam0"]["extrinsic"] > 0.0
