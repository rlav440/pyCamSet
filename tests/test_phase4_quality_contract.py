'''Purpose: Regression tests for Phase 4 self-calibration quality gates.
Status: Active; keeps Phase 4 fail-closed and provenance-preserving.
Future: Extend the real-corpus contract as additional good and underconstrained rigs are characterised.
'''

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from pyCamSet.workflow import phase4
from pyCamSet.workflow.workspace import WorkspaceManager, workspace_path_for


@pytest.fixture
def stub_jacobian_measurement(monkeypatch):
    """Replace the Jacobian null-space measurement with a clean, constrained result.

    For the quality-gate tests whose fake solver result has no Jacobian to
    measure; the real measurement has its own tests.
    """
    monkeypatch.setattr(
        phase4, "_jacobian_nullity_measurement",
        lambda optimisation, handler: {
            "rank": 8,
            "free_parameters": 8,
            "total_nullity": 0,
            "point_local_nullity": 0,
            "global_nullity": 0,
            "point_local_global_nullity_split": "measured",
            "tolerance": 1e-12,
            "jacobian_shape": [4, 8],
        },
    )


def test_jacobian_nullity_split_separates_local_and_global_modes():
    jacobian = np.array([
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    point_mask = np.array([True, False, False, True, False, False])
    detection_data = np.array([[0, 0, 0], [1, 0, 1]])

    measured = phase4._split_jacobian_nullity(
        jacobian, point_column_start=1,
        point_mask=point_mask, detection_data=detection_data,
    )

    assert measured["rank"] == 1
    assert measured["total_nullity"] == 2
    assert measured["point_local_nullity"] == 1
    assert measured["global_nullity"] == 1
    assert measured["point_local_global_nullity_split"] == "measured"


def test_jacobian_nullity_measurement_uses_solution_jacobian():
    matrix = np.array([
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    handler = SimpleNamespace(
        bundlePrimitive=SimpleNamespace(pose_end=1),
        feat_unfixed=np.array([True, False, False, True, False, False]),
        get_detection_data=lambda flatten=True: np.array([[0, 0, 0], [1, 0, 1]]),
    )
    optimisation = SimpleNamespace(x=np.array([7.0]), jac=sparse.csr_matrix(matrix))

    measured = phase4._jacobian_nullity_measurement(optimisation, handler)

    assert measured["total_nullity"] == 2
    assert measured["point_local_nullity"] == 1
    assert measured["global_nullity"] == 1


def test_jacobian_nullity_measurement_rejects_dense_backend_output():
    handler = SimpleNamespace(
        bundlePrimitive=SimpleNamespace(pose_end=1),
        feat_unfixed=np.array([True, False, False]),
        get_detection_data=lambda flatten=True: np.array([[0, 0, 0]]),
    )
    optimisation = SimpleNamespace(jac=np.ones((2, 2)))

    with pytest.raises(RuntimeError, match="must remain sparse"):
        phase4._jacobian_nullity_measurement(optimisation, handler)


def test_jacobian_nullity_split_fails_closed_over_reduced_memory_budget():
    # the budget is set by the global parameter count (R is globals x globals) ...
    detection_data = np.zeros((2, 3), dtype=int)
    jacobian = sparse.csr_matrix((4, 3000), dtype=float)

    with pytest.raises(ValueError, match="64 MiB measurement budget"):
        phase4._split_jacobian_nullity(
            jacobian, point_column_start=3000,
            point_mask=np.zeros(0, dtype=bool), detection_data=detection_data,
        )


def test_jacobian_nullity_split_measures_many_observations_in_bounded_memory():
    """... not by the observations: 100k residual rows x 100 globals used to exceed the old
    16 MiB dense budget, so every real 8-camera rig was gated without being measured."""
    detection_data = np.zeros((50_000, 3), dtype=int)
    jacobian = sparse.csr_matrix((100_000, 100), dtype=float)

    measured = phase4._split_jacobian_nullity(
        jacobian, point_column_start=100,
        point_mask=np.zeros(3, dtype=bool), detection_data=detection_data,
    )
    assert measured["global_nullity"] == 100 and measured["rank"] == 0


def test_streamed_rank_matches_a_dense_svd_on_a_rank_deficient_jacobian():
    rng = np.random.default_rng(3)
    n_obs, n_glob = 6000, 40
    basis = rng.normal(size=(2 * n_obs, 33))
    global_cols = basis @ rng.normal(size=(33, n_glob))  # rank 33 -> global nullity 7
    jacobian = sparse.csr_matrix(global_cols)
    measured = phase4._split_jacobian_nullity(
        jacobian, point_column_start=n_glob,
        point_mask=np.zeros(3, dtype=bool),
        detection_data=np.zeros((n_obs, 3), dtype=int),
    )
    dense_rank = int(np.linalg.matrix_rank(global_cols, tol=measured["tolerance"]))
    assert measured["rank"] == dense_rank == 33
    assert measured["global_nullity"] == 7

def test_streamed_rank_over_many_chunks_matches_a_dense_svd():
    """Many features and a tiny budget, so the triangular factor is re-folded many times."""
    rng = np.random.default_rng(11)
    n_glob, n_feat, per_feat = 12, 30, 5
    unobserved = 7                       # a feature with no rows: three point-only null modes
    observed = [f for f in range(n_feat) if f != unobserved]
    # Observations in a shuffled order, so grouping by feature is exercised.
    feature_of = rng.permutation(np.repeat(observed, per_feat))
    n_obs = len(feature_of)
    # The globals have rank 9 (three null modes): every row lies in a 9-d subspace.
    global_cols = rng.normal(size=(2 * n_obs, 9)) @ rng.normal(size=(9, n_glob))
    point_cols = np.zeros((2 * n_obs, 3 * n_feat))
    for obs, feature in enumerate(feature_of):
        point_cols[2 * obs:2 * obs + 2, 3 * feature:3 * feature + 3] = rng.normal(size=(2, 3))
    dense = np.hstack((global_cols, point_cols))
    detection_data = np.column_stack((
        np.zeros(n_obs, dtype=int), np.arange(n_obs), feature_of))
    # 40 rows of chunk beyond the R factor: 300 reduced rows need about eight folds.
    budget = (n_glob + 40) * n_glob * np.dtype(float).itemsize

    measured = phase4._split_jacobian_nullity(
        sparse.csr_matrix(dense), point_column_start=n_glob,
        point_mask=np.ones(3 * n_feat, dtype=bool), detection_data=detection_data,
        max_reduced_bytes=budget,
    )

    assert measured["tsqr_chunk_rows"] == 40
    assert measured["reduced_global_shape"][0] > 4 * measured["tsqr_chunk_rows"]
    dense_rank = int(np.linalg.matrix_rank(dense, tol=measured["tolerance"]))
    assert measured["rank"] == dense_rank
    assert measured["total_nullity"] == dense.shape[1] - dense_rank
    assert measured["point_local_nullity"] == 3
    assert measured["global_nullity"] == 3
    # The single-chunk answer (default budget) is the same measurement.
    whole = phase4._split_jacobian_nullity(
        sparse.csr_matrix(dense), point_column_start=n_glob,
        point_mask=np.ones(3 * n_feat, dtype=bool), detection_data=detection_data,
    )
    assert {key: whole[key] for key in ("rank", "global_nullity", "point_local_nullity")} == {
        key: measured[key] for key in ("rank", "global_nullity", "point_local_nullity")}


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


def test_phase4_quality_gate_records_gauge_and_coverage_contract(stub_jacobian_measurement):
    gate = phase4._quality_gate(
        _optimisation(), _Handler(),
        {"success": True}, np.ones((4, 2)),
        initial_euclid=10.0, final_euclid=2.0, observation_count=4,
        phase3_run={"status": "complete", "run_id": "p3"},
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


def test_phase4_quality_gate_lists_but_accepts_images_the_solve_excluded(stub_jacobian_measurement):
    """An image with no usable pose is reported, not treated as a hole."""
    class _SparseDetection(_Detection):
        max_ims = 4

    class _ExcludingHandler(_Handler):
        detection = _SparseDetection()
        missing_poses = np.array([False, False, True, False])

    gate = phase4._quality_gate(
        _optimisation(), _ExcludingHandler(),
        {"success": True}, np.ones((4, 2)),
        initial_euclid=10.0, final_euclid=2.0, observation_count=4,
        phase3_run={"status": "complete", "run_id": "p3"},
    )

    assert gate["excluded_images"] == [2]
    # image 3 was neither observed nor excluded: that still blocks
    assert gate["missing_images"] == [3]
    assert "image observation graph has missing image indices" in gate["blocking_flags"]

    _ExcludingHandler.missing_poses = np.array([False, False, True, True])
    gate = phase4._quality_gate(
        _optimisation(), _ExcludingHandler(),
        {"success": True}, np.ones((4, 2)),
        initial_euclid=10.0, final_euclid=2.0, observation_count=4,
        phase3_run={"status": "complete", "run_id": "p3"},
    )
    assert gate["image_coverage"] is True
    assert gate["excluded_images"] == [2, 3]
    assert gate["status"] == "complete"


def test_phase4_per_image_error_is_skipped_when_residuals_do_not_line_up(monkeypatch):
    monkeypatch.setattr(phase4, "_target_shape_change", lambda *_: (1.0, 0.0))
    monkeypatch.setattr(
        phase4, "per_camera_mean_reprojection",
        lambda *_: ({}, np.ones((3, 2))))
    logged = []

    diagnostics = phase4._diagnostics(
        _optimisation(), _Handler(),
        {"success": True, "initial_euclid": 2.0, "final_euclid": 1.0},
        None, logged.append)

    assert diagnostics["D4.13_per_image_mean_reprojection"] == {}
    assert any("D4.13 skipped" in line for line in logged)


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


def test_solver_stats_record_the_linear_objective_costs():
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

    assert result["initial_objective_cost"] == 12.5
    assert result["final_objective_cost"] == 0.0


def test_solver_stats_record_the_robust_objective_the_solve_minimised():
    from pyCamSet.optimisation.optimisation_handling import get_bundle_adjustment_stats

    class _RobustHandler:
        problem_opts = {"loss": "soft_l1", "f_scale": 1.0}

        def get_base_residual_count(self):
            return 4

    result = get_bundle_adjustment_stats(
        SimpleNamespace(success=True, status=2, message="ok", nfev=3,
                        fun=np.zeros(4)),
        np.ones(2), np.array([3.0, 0.0, 0.0, 4.0]), 0.1,
        param_handler=_RobustHandler(),
    )

    # soft_l1: 0.5 * sum(2 * (sqrt(1 + f^2) - 1)) over 3 and 4
    expected = (np.sqrt(10.0) - 1.0) + (np.sqrt(17.0) - 1.0)
    assert result["initial_objective_cost"] == pytest.approx(expected)


def test_phase4_quality_gate_distinguishes_objective_from_mean_error():
    gate = phase4._quality_gate(
        _optimisation(), _Handler(),
        {
            "success": True,
            "initial_objective_cost": 10.0,
            "final_objective_cost": 5.0,
        }, np.ones((4, 2)),
        initial_euclid=2.0, final_euclid=3.0, observation_count=4,
    )

    assert gate["objective_cost_reduced"] is True
    assert gate["error_reduced"] is False
    assert "final reprojection error did not improve finitely" in gate["blocking_flags"]


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
    target = type("ChArUco", (), {
        "__module__": "pyCamSet.calibration_targets.synthetic",
    })()
    previous_handler = SimpleNamespace(
        target=target, detection=detection, problem_opts={"fixed_pose": 0})
    camera = SimpleNamespace(
        intrinsic=np.eye(3), distortion_coefs=np.zeros(5), extrinsic=np.eye(4))

    class CameraCollection:
        _saved_point_data_units = "m"
        calibration_handler = previous_handler

        def get_names(self):
            return ["cam0"]

        def __getitem__(self, _name):
            return camera

    previous_cams = CameraCollection()
    monkeypatch.setattr(phase4, "load_CameraSet", lambda _path: previous_cams)
    captured = {}

    def stop_after_option_capture(*args, options, **kwargs):
        captured.update(options)
        raise RuntimeError("stop after checking options")

    monkeypatch.setattr(phase4, "solve", stop_after_option_capture)
    camset_path = tmp_path / "phase3.camset"
    with pytest.raises(RuntimeError, match="stop after checking options"):
        phase4._solve(
            {"problem_options": options}, tmp_path, camset_path,
            {"run_id": "p3", "status": "complete",
             "artifacts": {"optimised_camset": str(camset_path)}},
            lambda _line: None,
        )
    assert captured["max_nfev"] == expected


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
            self.intr_end = 0
            self.extr_end = 3 * int(self.extr_unfixed.sum())
            self.pose_end = self.extr_end + 6
            self.bdpt_end = self.pose_end + 3

        def return_bundle_primitives(self, _params):
            return self.intr, self.extr, self.poses, self.bundle_pts

    previous_handler = TemplateBundleHandler.__new__(TemplateBundleHandler)
    previous_handler.missing_poses = None
    previous_handler.target = SimpleNamespace(point_data=np.array([[[1.0, 2.0, 3.0]]]))
    previous_handler.bundlePrimitive = _Primitive(extr_unfixed=[True, True])
    previous_cams = SimpleNamespace(
        calibration_handler=previous_handler,
        calibration_params=np.arange(12.0),
    )

    current_handler = SelfBundleHandler.__new__(SelfBundleHandler)
    current_handler.bundlePrimitive = _Primitive(extr_unfixed=[False, True])
    current_handler.flat_point_data = np.array([1.0, 2.0, 3.0])
    current_handler.feat_unfixed = np.array([True, True, True])

    current_handler.set_from_templated_camset(previous_cams)

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

@pytest.mark.parametrize("phase3_run,reason", [
    ({"status": "complete"}, "Phase 3 input run has no run id"),
    ({"status": "failed", "run_id": "p3"},
     "Phase 3 input disposition was failed; re-run it before hand-off"),
    ({"run_id": "p3"},
     "Phase 3 input disposition was None; re-run it before hand-off"),
])
def test_phase4_provenance_must_be_identified_and_complete(phase3_run, reason):
    gate = phase4._quality_gate(
        _optimisation(), _Handler(), {"success": True}, np.ones((4, 2)),
        initial_euclid=10.0, final_euclid=2.0, observation_count=4,
        phase3_run=phase3_run,
    )
    assert gate["status"] == "incomplete"
    assert gate["phase3_provenance_complete"] is False
    assert [flag for flag in gate["blocking_flags"] if flag.startswith("Phase 3")] == [reason]


def test_phase4_without_a_phase3_run_is_not_blocked_on_provenance(stub_jacobian_measurement):
    """A headless caller that passes only a camset has no Phase 3 run to check."""
    gate = phase4._quality_gate(
        _optimisation(), _Handler(), {"success": True}, np.ones((4, 2)),
        initial_euclid=10.0, final_euclid=2.0, observation_count=4,
    )
    assert gate["status"] == "complete"
    assert gate["phase3_provenance_complete"] is None


def test_phase4_provenance_accepts_identified_complete_run(stub_jacobian_measurement):
    gate = phase4._quality_gate(
        _optimisation(), _Handler(), {"success": True}, np.ones((4, 2)),
        initial_euclid=10.0, final_euclid=2.0, observation_count=4,
        phase3_run={"status": "complete", "run_id": "p3"},
    )
    assert gate["status"] == "complete"
    assert gate["phase3_provenance_complete"]


def test_phase4_quality_gate_marks_measured_global_modes_weak(monkeypatch):
    monkeypatch.setattr(
        phase4, "_jacobian_nullity_measurement",
        lambda optimisation, handler: {
            "rank": 7,
            "free_parameters": 8,
            "total_nullity": 1,
            "point_local_nullity": 0,
            "global_nullity": 1,
            "point_local_global_nullity_split": "measured",
            "tolerance": 1e-12,
            "jacobian_shape": [4, 8],
        },
    )

    gate = phase4._quality_gate(
        _optimisation(), _Handler(), {"success": True}, np.ones((4, 2)),
        initial_euclid=10.0, final_euclid=2.0, observation_count=4,
        phase3_run={"status": "complete", "run_id": "p3"},
    )

    assert gate["constraint_disposition"]["status"] == "weakly_constrained"
    assert gate["constraint_disposition"]["global_nullity"] == 1


def test_phase4_quality_gate_blocks_failed_nullity_measurement(monkeypatch):
    def fail_measurement(optimisation, handler):
        raise RuntimeError("synthetic rank failure")

    monkeypatch.setattr(phase4, "_jacobian_nullity_measurement", fail_measurement)
    gate = phase4._quality_gate(
        _optimisation(), _Handler(), {"success": True}, np.ones((4, 2)),
        initial_euclid=10.0, final_euclid=2.0, observation_count=4,
        phase3_run={"status": "complete", "run_id": "p3"},
    )

    assert gate["status"] == "incomplete"
    assert "Jacobian null-space measurement failed at solution" in gate["blocking_flags"]
    assert gate["constraint_disposition"]["global_nullity"] == "not_verified"


def test_phase4_quality_gate_marks_unobserved_fixed_pose_weakly_constrained():
    class _UnobservedReferenceHandler(_Handler):
        problem_opts = {"fixed_pose": 2, "ref_cam": 0, "ref_pose": 0}

    gate = phase4._quality_gate(
        _optimisation(), _UnobservedReferenceHandler(),
        {"success": True}, np.ones((4, 2)),
        initial_euclid=10.0, final_euclid=2.0, observation_count=4,
        phase3_run={"status": "complete", "run_id": "p3"},
    )

    assert gate["constraint_disposition"]["status"] == "weakly_constrained"
    assert gate["constraint_disposition"][
        "fixed_pose_excluded_or_unobserved"] is True



def test_phase4_refuses_fixed_pose_mismatch_before_solver_start(tmp_path, monkeypatch):
    detection = SimpleNamespace(cam_names=["cam0"])
    target = type("ChArUco", (), {
        "__module__": "pyCamSet.calibration_targets.synthetic",
    })()
    previous_handler = SimpleNamespace(
        target=target, detection=detection, problem_opts={"fixed_pose": 3})
    camera = SimpleNamespace(
        intrinsic=np.eye(3), distortion_coefs=np.zeros(5), extrinsic=np.eye(4))

    class CameraCollection:
        _saved_point_data_units = "m"
        calibration_handler = previous_handler

        def get_names(self):
            return ["cam0"]

        def __getitem__(self, _name):
            return camera

    camset_path = tmp_path / "phase3.camset"
    monkeypatch.setattr(phase4, "load_CameraSet", lambda _path: CameraCollection())

    def unexpected_solve(*_args, **_kwargs):
        raise AssertionError("mismatched fixed_pose must fail before solver start")

    monkeypatch.setattr(phase4, "solve", unexpected_solve)
    with pytest.raises(RuntimeError, match="must match Phase 3's fixed_pose"):
        phase4._solve(
            {"problem_options": {"fixed_pose": 0}}, tmp_path, camset_path,
            {"run_id": "p3", "status": "complete",
             "artifacts": {"optimised_camset": str(camset_path)}},
            lambda _line: None,
        )


@pytest.mark.parametrize("square_size_mm", [2.0, 0.5])
def test_phase4_refuses_puzzleboard_unit_mismatch_on_both_sides_of_1mm(
    tmp_path, monkeypatch, square_size_mm
):
    pytest.importorskip("puzzle_board")
    from pyCamSet import PuzzleBoard

    target = PuzzleBoard(
        num_squares_x=5, num_squares_y=6, square_size=square_size_mm)
    handler = SimpleNamespace(
        target=target,
        detection=SimpleNamespace(cam_names=["cam0"]),
        problem_opts={"fixed_pose": 0},
    )
    camera = SimpleNamespace(
        intrinsic=np.eye(3), distortion_coefs=np.zeros(5), extrinsic=np.eye(4))

    class MetreSavedPuzzleBoardCamset:
        _saved_point_data_units = "m"
        calibration_handler = handler

        def get_names(self):
            return ["cam0"]

        def __getitem__(self, _name):
            return camera

    camset_path = tmp_path / "phase3-puzzleboard.camset"
    monkeypatch.setattr(
        phase4, "load_CameraSet", lambda _path: MetreSavedPuzzleBoardCamset())
    monkeypatch.setattr(
        phase4, "solve",
        lambda *_args, **_kwargs: pytest.fail("unit mismatch must refuse before solve"),
    )

    with pytest.raises(RuntimeError, match="target units changed since Phase 3"):
        phase4._solve(
            {"problem_options": {}}, tmp_path, camset_path,
            {"run_id": "p3", "status": "complete",
             "artifacts": {"optimised_camset": str(camset_path)}},
            lambda _line: None,
        )


class _ReachedSolve(Exception):
    """Raised by a stubbed solve to show the pre-solve checks let the run through."""


def _legacy_unit_camset(saved_units, target_name, handler_module):
    target = type(target_name, (), {"__module__": handler_module})()
    handler = SimpleNamespace(
        target=target,
        detection=SimpleNamespace(cam_names=["cam0"]),
        problem_opts={"fixed_pose": 0},
    )
    camera = SimpleNamespace(
        intrinsic=np.eye(3), distortion_coefs=np.zeros(5), extrinsic=np.eye(4))

    class SavedCameraSet:
        _saved_point_data_units = saved_units
        calibration_handler = handler

        def get_names(self):
            return ["cam0"]

        def __getitem__(self, _name):
            return camera

    return SavedCameraSet()


@pytest.mark.parametrize("target_name", ["ChArUco", "NotInTheUnitTable"])
def test_phase4_proceeds_from_a_camset_saved_before_unit_records(
    tmp_path, monkeypatch, target_name
):
    """A legacy camset cannot be re-saved with units, so it must not be refused."""
    previous_cams = _legacy_unit_camset(
        "legacy/unverified", target_name, "pyCamSet.calibration_targets.synthetic")
    monkeypatch.setattr(phase4, "load_CameraSet", lambda _path: previous_cams)

    def reached(*_args, **_kwargs):
        raise _ReachedSolve

    monkeypatch.setattr(phase4, "solve", reached)
    camset_path = tmp_path / "phase3.camset"
    with pytest.raises(_ReachedSolve):
        phase4._solve(
            {"problem_options": {}}, tmp_path, camset_path,
            {"run_id": "p3", "status": "complete",
             "artifacts": {"optimised_camset": str(camset_path)}},
            lambda _line: None,
        )


def test_phase4_quality_gate_lists_input_notices_without_blocking(
    stub_jacobian_measurement,
):
    gate = phase4._quality_gate(
        _optimisation(), _Handler(), {"success": True}, np.ones((4, 2)),
        initial_euclid=10.0, final_euclid=2.0, observation_count=4,
        input_notices=["the saved Phase 3 camset predates target-unit records"],
    )
    assert gate["status"] == "complete"
    assert any("predates target-unit records" in n for n in gate["notices"])


def test_phase4_without_a_phase3_run_skips_the_run_record_checks(tmp_path, monkeypatch):
    previous_cams = _legacy_unit_camset(
        "m", "ChArUco", "pyCamSet.calibration_targets.synthetic")
    monkeypatch.setattr(phase4, "load_CameraSet", lambda _path: previous_cams)

    def reached(*_args, **_kwargs):
        raise _ReachedSolve

    monkeypatch.setattr(phase4, "solve", reached)
    with pytest.raises(_ReachedSolve):
        phase4._solve(
            {"problem_options": {}}, tmp_path, tmp_path / "phase3.camset",
            None, lambda _line: None,
        )


def test_self_calibration_gauge_uses_target_point_data_units():
    from pyCamSet.optimisation.standard_bundle_handler import _gauge_square_size

    target = type("PuzzleBoardCube", (), {
        "__module__": "pyCamSet.calibration_targets.synthetic",
        "square_size": 4.2857142857,
        "point_data": np.array([[[0.0, 0.0, 0.0],
                                 [0.0042857143, 0.0, 0.0],
                                 [0.0, 0.0042857143, 0.0]]]),
    })()

    assert np.isclose(_gauge_square_size(target), 0.0042857143)


def test_parameter_drift_uses_independent_before_snapshots():
    from pyCamSet.workflow.phase4 import (
        _camera_drift, _snapshot_camera_parameters,
    )

    camera = SimpleNamespace(
        intrinsic=np.eye(3),
        distortion_coefs=np.zeros(5),
        extrinsic=np.eye(4),
    )
    # Python's special-method lookup needs a class-defined __getitem__.
    class CameraCollection:
        def get_names(self):
            return ["cam0"]

        def __getitem__(self, _name):
            return camera

    before = CameraCollection()
    snapshot = _snapshot_camera_parameters(before)
    camera.intrinsic[0, 0] = 2.0
    after = CameraCollection()
    drift = _camera_drift(snapshot, after)

    assert snapshot["cam0"]["intrinsic"][0, 0] == 1.0
    assert drift["cameras"]["cam0"]["intrinsic"] > 0.0
    assert not np.shares_memory(snapshot["cam0"]["intrinsic"], camera.intrinsic)


def test_parameter_drift_matches_reloaded_persisted_camera_values(tmp_path):
    from pyCamSet import Camera, CameraSet
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.phase4 import (
        _camera_drift, _snapshot_camera_parameters,
    )

    cameras = CameraSet(camera_dict={
        name: Camera(
            extrinsic=np.eye(4), intrinsic=np.eye(3),
            distortion_coefs=np.zeros(5), res=np.array([100, 100]),
            name=name,
        )
        for name in ("moved", "unchanged")
    })
    before = _snapshot_camera_parameters(cameras)
    cameras["moved"].intrinsic[0, 0] = 1.25

    persisted_path = tmp_path / "phase4-output.camset"
    cameras.save(persisted_path)
    persisted = load_CameraSet(persisted_path)
    drift = _camera_drift(before, persisted)["cameras"]

    assert persisted["moved"].intrinsic[0, 0] == 1.25
    assert drift["moved"]["intrinsic"] > 0.0
    assert drift["unchanged"] == {
        "intrinsic": 0.0, "distortion_coefs": 0.0, "extrinsic": 0.0,
    }


@pytest.mark.data
def test_gauge_selection_avoids_single_observation_points_and_replays_recorded_indices(
    charuco_problem,
):
    from pyCamSet.optimisation import standard_bundle_handler
    from pyCamSet.optimisation.standard_bundle_handler import (
        SelfBundleHandler, gauge_axis_for_indices,
    )

    original_module_bindings = dict(vars(standard_bundle_handler))
    original_class_bindings = dict(vars(SelfBundleHandler))
    target, detections, cams = charuco_problem
    handler = SelfBundleHandler(
        camset=cams, target=target, detection=detections,
        options={"outliers": "n"},
    )
    counts = np.bincount(
        handler._flat_detections()[:, 2].astype(int),
        minlength=len(handler.visible_feature_mask),
    )
    assert all(counts[index] >= 2 for index in handler.fixed_inds)

    recorded_indices = tuple(handler.fixed_inds)
    rebuilt = SelfBundleHandler(
        camset=cams, target=target, detection=detections,
        options={
            "outliers": "n",
            "recorded_gauge_indices": list(recorded_indices),
            "target_scale_rule": "legacy_mean_all_edges",
        },
    )
    assert rebuilt.fixed_inds == recorded_indices
    assert rebuilt.gauge_axis == gauge_axis_for_indices(
        rebuilt.flat_point_data.reshape(-1, 3), recorded_indices)
    assert rebuilt.problem_opts["target_scale_rule"] == "legacy_mean_all_edges"
    assert set(vars(standard_bundle_handler)) == set(original_module_bindings)
    assert all(
        vars(standard_bundle_handler)[name] is binding
        for name, binding in original_module_bindings.items()
    )
    assert set(vars(SelfBundleHandler)) == set(original_class_bindings)
    assert all(
        vars(SelfBundleHandler)[name] is binding
        for name, binding in original_class_bindings.items()
    )


@pytest.mark.parametrize(
    "family,coordinate_mm,expected_mm",
    [("PuzzleBoard", 2.0, 2.0), ("PuzzleBoard", 0.5, 0.5),
     ("ChArUco", 0.002, 2.0)],
)
def test_phase4_displacement_is_reported_in_mm_for_target_coordinate_units(
    family, coordinate_mm, expected_mm
):
    from pyCamSet.workflow.phase4 import _target_shape_change

    target_type = type(family, (), {
        "__module__": "pyCamSet.calibration_targets.synthetic",
    })
    target = target_type()
    target.point_data = np.array([[0.0, 0.0, 0.0]])
    handler = SimpleNamespace(
        target=target,
        get_updated_target=lambda _params: np.array([[coordinate_mm, 0.0, 0.0]]),
    )
    _, displacement = _target_shape_change(
        handler, SimpleNamespace(x=np.array([])), np.array([True]))

    assert np.isclose(displacement, expected_mm)


def test_phase4_scale_rule_excludes_once_seen_edges_but_keeps_legacy_selector():
    from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler

    handler = object.__new__(SelfBundleHandler)
    handler.target = type("PuzzleBoard", (), {
        "__module__": "pyCamSet.calibration_targets.synthetic",
        "valid_map": True,
        "square_size": 1.0,
    })()
    handler.visible_feature_mask = np.ones(4, dtype=bool)
    handler.problem_opts = {"target_scale_rule": "measured_edges_v1"}
    detections = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1],
                           [0, 1, 1], [0, 0, 2], [0, 1, 2], [0, 0, 3]])
    handler._flat_detections = lambda: detections
    nominal = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    solved = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                       [0.0, 2.0, 0.0], [10.0, 10.0, 0.0]])

    measured_scale = handler._scale_against_target(solved, nominal)
    handler.problem_opts["target_scale_rule"] = "legacy_mean_all_edges"
    legacy_scale = handler._scale_against_target(solved, nominal)

    assert measured_scale == 0.5
    assert not np.isclose(legacy_scale, measured_scale)


def _phase3_camera_collection(*, problem_opts=None):
    """A stand-in Phase 3 camset that passes every provenance check."""
    target = type("ChArUco", (), {
        "__module__": "pyCamSet.calibration_targets.synthetic",
    })()
    handler = SimpleNamespace(
        target=target,
        detection=SimpleNamespace(cam_names=["cam0"]),
        problem_opts=problem_opts or {"fixed_pose": 0},
    )
    camera = SimpleNamespace(
        intrinsic=np.eye(3), distortion_coefs=np.zeros(5), extrinsic=np.eye(4))

    class CameraCollection:
        _saved_point_data_units = "m"
        calibration_handler = handler

        def get_names(self):
            return ["cam0"]

        def __getitem__(self, _name):
            return camera

        def save(self, path):
            Path(path).write_text("camset", encoding="utf-8")

    return CameraCollection()


def test_phase4_quality_gate_records_measured_constraint_disposition(stub_jacobian_measurement):
    gate = phase4._quality_gate(
        _optimisation(), _Handler(), {"success": True}, np.ones((4, 2)),
        initial_euclid=10.0, final_euclid=2.0, observation_count=4,
        phase3_run={"status": "complete", "run_id": "p3"},
    )

    assert gate["constraint_disposition"]["status"] == "constrained"
    assert gate["constraint_disposition"]["global_nullity"] == 0
    assert gate["constraint_disposition"][
        "point_local_global_nullity_split"] == "measured"
    assert gate["constraint_disposition"]["fixed_pose_observed"] is True
    assert gate["jacobian_nullity"]["rank"] == 8
    # the keys the education view reads come through the Phase 3 base gate
    assert gate["observed_cameras"] == [0, 1]
    assert gate["expected_camera_count"] == 2
    assert gate["solver_success"] is True
    assert gate["error_reduced"] is True


def test_phase4_quality_gate_reports_excluded_images_as_a_notice(stub_jacobian_measurement):
    class _ExcludingDetection(_Detection):
        max_ims = 3

    class _ExcludingHandler(_Handler):
        detection = _ExcludingDetection()
        missing_poses = np.array([False, False, True])

    gate = phase4._quality_gate(
        _optimisation(), _ExcludingHandler(),
        {"success": True}, np.ones((4, 2)),
        initial_euclid=10.0, final_euclid=2.0, observation_count=4,
        phase3_run={"status": "complete", "run_id": "p3"},
    )

    assert gate["status"] == "complete"
    assert gate["excluded_images"] == [2]
    assert gate["notices"] == [
        "1 image(s) had no usable target pose and were excluded from the solve: [2]"]
    assert gate["constraint_disposition"][
        "fixed_pose_excluded_or_unobserved"] is False


def test_phase4_quality_gate_has_no_notice_without_excluded_images():
    gate = phase4._quality_gate(
        _optimisation(), _Handler(), {"success": True}, np.ones((4, 2)),
        initial_euclid=10.0, final_euclid=2.0, observation_count=4,
        phase3_run={"status": "complete", "run_id": "p3"},
    )

    assert gate["notices"] == []


def test_phase4_diagnostics_log_each_gate_notice(monkeypatch):
    monkeypatch.setattr(phase4, "_target_shape_change", lambda *_: (1.0, 0.0))
    monkeypatch.setattr(
        phase4, "per_camera_mean_reprojection",
        lambda *_: ({}, np.ones((4, 2))))
    monkeypatch.setattr(
        phase4, "_quality_gate",
        lambda *_args, **_kwargs: {"notices": ["first notice", "second notice"]})
    logged = []

    phase4._diagnostics(
        _optimisation(), _Handler(),
        {"success": True, "initial_euclid": 2.0, "final_euclid": 1.0},
        None, logged.append)

    assert "NOTE: first notice" in logged and "NOTE: second notice" in logged


def test_phase4_refuses_an_input_camset_that_is_not_the_recorded_one(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        phase4, "load_CameraSet", lambda _path: _phase3_camera_collection())
    monkeypatch.setattr(
        phase4, "solve",
        lambda *_args, **_kwargs: pytest.fail("provenance must refuse before solve"),
    )

    with pytest.raises(RuntimeError, match="does not verify against the selected"):
        phase4._solve(
            {"problem_options": {}}, tmp_path, tmp_path / "phase3.camset",
            {"run_id": "p3", "status": "complete",
             "artifacts": {"optimised_camset": str(tmp_path / "other.camset")}},
            lambda _line: None,
        )


def test_phase4_inherits_single_face_recovery_from_the_phase3_handler(
    tmp_path, monkeypatch
):
    camset_path = tmp_path / "phase3.camset"
    run = {"run_id": "p3", "status": "complete",
           "artifacts": {"optimised_camset": str(camset_path)}}
    captured = []

    def stop_after_option_capture(*args, options, **kwargs):
        captured.append(dict(options))
        raise RuntimeError("stop after checking options")

    monkeypatch.setattr(phase4, "solve", stop_after_option_capture)
    for opts, explicit, expected in (
        ({"fixed_pose": 0, "recover_single_face_telecentric": True}, {}, True),
        ({"fixed_pose": 0}, {}, False),
        ({"fixed_pose": 0, "recover_single_face_telecentric": True},
         {"recover_single_face_telecentric": False}, False),
    ):
        collection = _phase3_camera_collection(problem_opts=opts)
        monkeypatch.setattr(phase4, "load_CameraSet", lambda _path: collection)
        with pytest.raises(RuntimeError, match="stop after checking options"):
            phase4._solve(
                {"problem_options": explicit}, tmp_path, camset_path, run,
                lambda _line: None)
        assert captured[-1]["recover_single_face_telecentric"] is expected






def test_phase4_takes_the_run_as_provenance_only_for_its_own_camset(tmp_path):
    """An override camset that is not the selected run's output starts Phase 4 alone."""
    from pyCamSet.workflow.phase4 import phase3_run_for_camset

    own, other = tmp_path / "own.camset", tmp_path / "other.camset"
    own.write_bytes(b"")
    other.write_bytes(b"")
    run = {"run_id": "p3", "artifacts": {"optimised_camset": str(own)}}
    assert phase3_run_for_camset(run, own) is run
    assert phase3_run_for_camset(run, other) is None
    assert phase3_run_for_camset(None, own) is None
    used_only = {"run_id": "p3", "artifacts": {"phase3_camset_used": str(own)}}
    assert phase3_run_for_camset(used_only, own) is used_only
    assert phase3_run_for_camset({"run_id": "p3", "artifacts": {}}, own) is None
