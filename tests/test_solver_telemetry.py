'''Exercise bounded solver telemetry and Qt-free Phase 3 playback.
'''
import numpy as np
import pytest


from pyCamSet.workflow.education_playback import (
    AcceptedStepCollector,
    build_phase3_frames,
    read_accepted_step_trace,
)


class _Primitive:
    def return_bundle_primitives(self, params):
        return (np.array([[params[0]]]), np.array([[0.0, 0.0, params[0]]]),
                np.array([[0.0, 0.0, 0.0, 0.0, 0.0, params[0]]]))


class _Handler:
    bundlePrimitive = _Primitive()
    problem_opts = {"loss": "linear", "f_scale": 1.0}

    def make_loss_fun(self, _threads):
        def loss(x):
            costs = {0: 4.5, 1: 0.5, 2: 0.1}
            return np.array([np.sqrt(2.0 * costs[int(np.asarray(x)[0])])])
        return loss


def _trace(payload):
    return {"status": "available", "records": [
        {"kind": "accepted_step", "payload": payload}]}


def test_collector_copies_accepted_vectors_and_keeps_exact_endpoints():
    handler = _Handler()
    start = np.array([0.0])
    collector = AcceptedStepCollector(handler, start, max_frames=3, byte_budget=4096)
    accepted = np.array([1.0])
    collector.add_schur(1, 0.5, accepted)
    accepted[0] = 99.0
    collector.add_schur(2, 0.1, np.array([2.0]))
    collector.finish(np.array([2.0]), 0.1, nit=2)
    payload = collector.to_payload(solver="schur", problem_opts=handler.problem_opts)
    checked = read_accepted_step_trace(
        _trace(payload), expected_start=start, saved_params=np.array([2.0]),
        handler=handler)
    assert checked["status"] == "available"
    assert checked["arrays"][1][0] == 1.0
    assert payload["byte_budget_counts_endpoints"] is True
    assert payload["opencv_major_version"] >= 0


def test_reader_rejects_endpoint_nit_and_cost_contract_violations():
    handler = _Handler()
    collector = AcceptedStepCollector(handler, np.array([0.0]), byte_budget=4096)
    collector.add_schur(1, 0.5, np.array([1.0]))
    collector.finish(np.array([2.0]), 0.1, nit=2)
    payload = collector.to_payload(solver="schur", problem_opts=handler.problem_opts)
    assert read_accepted_step_trace(
        _trace(payload), expected_start=np.array([9.0]),
        saved_params=np.array([2.0]))["status"] == "unavailable"
    payload["steps"][1]["nit"] = payload["steps"][0]["nit"]
    assert "strictly increasing" in read_accepted_step_trace(
        _trace(payload), expected_start=np.array([0.0]),
        saved_params=np.array([2.0]))["reason"]
    payload["steps"][1]["nit"] = 1
    payload["steps"][-1]["cost"] = 3.0
    assert "non-increasing" in read_accepted_step_trace(
        _trace(payload), expected_start=np.array([0.0]),
        saved_params=np.array([2.0]))["reason"]


def test_no_accepted_steps_is_reported_as_such():
    collector = AcceptedStepCollector(_Handler(), np.array([0.0]), byte_budget=4096)
    collector.finish(np.array([0.0]), 0.0)
    payload = collector.to_payload(solver="schur", problem_opts=_Handler.problem_opts)
    result = read_accepted_step_trace(
        _trace(payload), expected_start=np.array([0.0]), saved_params=np.array([0.0]))
    assert result == {"status": "unavailable", "reason": "no accepted steps"}


def test_initial_residuals_avoid_a_second_loss_evaluation():
    handler = _Handler()
    handler.make_loss_fun = lambda _threads: (_ for _ in ()).throw(
        AssertionError("trace must reuse residuals already evaluated by the solver"))
    collector = AcceptedStepCollector(
        handler, np.array([0.0]), byte_budget=4096,
        initial_residuals=np.array([3.0]))
    collector.add_schur(1, 0.5, np.array([1.0]))
    collector.finish(np.array([1.0]), 0.5, nit=1)
    payload = collector.to_payload(solver="schur", problem_opts=handler.problem_opts)
    assert payload["steps"][0]["cost"] == 4.5


def test_phase3_builder_returns_copies_and_does_not_rewrite_inputs():
    handler = _Handler()
    start, final = np.array([0.0]), np.array([1.0])
    collector = AcceptedStepCollector(handler, start, byte_budget=4096)
    collector.add_schur(1, 0.5, final)
    collector.finish(final, 0.5)
    result_camset = np.array([42.0])
    before = result_camset.copy()
    frames = build_phase3_frames(
        lambda: _Handler(),
        _trace(collector.to_payload(solver="schur", problem_opts=handler.problem_opts)),
        expected_start=start, saved_params=final)
    assert frames["status"] == "available"
    assert np.array_equal(result_camset, before)
    assert len(frames["frames"]) == 2
    frames["frames"][0][1][0, 2] = 999.0
    assert frames["frames"][1][1][0, 2] == 1.0


@pytest.mark.parametrize("threads,force_parallel", [(1, False), (2, True)])
def test_recording_toggle_preserves_real_phase3_solver_result(
        charuco_problem, tmp_path, monkeypatch, threads, force_parallel):
    """Paired solves each reload the seed from disk to avoid mutable inputs."""
    from pyCamSet.optimisation.numba_schur import SchurSolver
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.phase3 import solve
    if force_parallel:
        monkeypatch.setattr(SchurSolver, "PARALLEL_MIN_DETECTIONS", 0)

    target, detections, seed = charuco_problem
    seed_path = tmp_path / "seed.camset"
    seed.save(seed_path)
    results = []
    for enabled in (False, True):
        fresh_seed = load_CameraSet(seed_path)
        handler, result, solved, _stats = solve(
            fresh_seed, detections, target,
            fixed_params={}, options={"outliers": "n", "max_nfev": 100},
            threads=threads, record_education=enabled)
        results.append((result, solved, handler))
    off, on = results
    np.testing.assert_array_equal(off[0].x, on[0].x)
    np.testing.assert_array_equal(off[0].fun, on[0].fun)
    assert off[0].cost == on[0].cost
    assert off[0].status == on[0].status
    assert off[0].nit == on[0].nit
    assert off[0].success == on[0].success
    assert off[0].message == on[0].message
    for camera_name in off[1].get_names():
        np.testing.assert_array_equal(
            off[1][camera_name].intrinsic, on[1][camera_name].intrinsic)
        np.testing.assert_array_equal(
            off[1][camera_name].distortion_coefs,
            on[1][camera_name].distortion_coefs)
        np.testing.assert_array_equal(
            off[1][camera_name].extrinsic, on[1][camera_name].extrinsic)
    assert not hasattr(off[2], "education_step_trace")
    payload = on[2].education_step_trace
    assert payload["steps"][-1]["x"] == on[0].x.tolist()


def test_scipy_callback_accepts_optimize_result_and_array_forms():
    from scipy.optimize import OptimizeResult

    handler = _Handler()
    collector = AcceptedStepCollector(handler, np.array([0.0]), byte_budget=4096)
    collector.scipy_callback(OptimizeResult(x=np.array([1.0]), cost=0.5, nit=1))
    collector.scipy_callback(np.array([2.0]))
    assert [frame[0] for frame in collector._frames] == [1, 2]
    np.testing.assert_allclose([frame[1] for frame in collector._frames], [0.5, 0.1])


def test_endpoints_only_fallback_is_not_playable():
    collector = AcceptedStepCollector(_Handler(), np.array([0.0]), byte_budget=4096)
    collector.endpoints_only = True
    collector.add_schur(1, 0.5, np.array([1.0]))
    collector.finish(np.array([1.0]), 0.5, nit=1)
    result = read_accepted_step_trace(
        _trace(collector.to_payload(solver="trf", problem_opts=_Handler.problem_opts)),
        expected_start=np.array([0.0]), saved_params=np.array([1.0]))
    assert result == {"status": "unavailable", "reason": "steps not recorded for this solver"}


def test_large_vector_budget_keeps_float64_endpoints_and_reports_limit():
    large = np.zeros(600_000, dtype=np.float64)
    collector = AcceptedStepCollector(_Handler(), large, byte_budget=8 * 1024 * 1024)
    collector.add_schur(1, 0.5, large + 1.0)
    collector.finish(large + 1.0, 0.5, nit=1)
    payload = collector.to_payload(solver="schur", problem_opts=_Handler.problem_opts)
    assert payload["steps"][0]["dtype"] == "float64"
    assert payload["steps"][-1]["dtype"] == "float64"
    assert payload["budget_exceeded_by_endpoints"] is True


def test_lowered_frame_cap_thins_by_motion_not_cost():
    class MotionHandler(_Handler):
        def make_loss_fun(self, _threads):
            return lambda x: np.array([10.0 - np.asarray(x)[0]])

    trajectory = np.array([0.1, 0.2, 0.3, 5.0, 5.1, 5.2, 8.0, 8.1, 10.0])
    handler = MotionHandler()
    collector = AcceptedStepCollector(
        handler, np.array([0.0]), max_frames=3, byte_budget=4096)
    for nit, position in enumerate(trajectory, start=1):
        cost = 0.5 * (10.0 - position) ** 2
        collector.add_schur(nit, cost, np.array([position]))
    collector.finish(np.array([trajectory[-1]]), 0.0, nit=len(trajectory))
    steps = collector.to_payload(
        solver="schur", problem_opts=handler.problem_opts)["steps"]
    retained = np.array([step["x"][0] for step in steps])
    assert len(steps) <= 3 + 2
    assert retained[0] == 0.0 and retained[-1] == trajectory[-1]
    uniform_indices = np.rint(np.linspace(0, len(trajectory), len(retained))).astype(int)
    uniform = np.concatenate(([0.0], trajectory))[uniform_indices]
    assert np.diff(retained).max() <= np.diff(uniform).max()


def test_scipy_solver_recording_parity_and_runtime_fallback(
        charuco_problem, tmp_path, monkeypatch):
    """Exercise SciPy callback capture and endpoints-only feature fallback."""
    import pyCamSet.optimisation.optimisation_handling as optimisation_handling
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow import phase3

    target, detections, seed = charuco_problem
    seed_path = tmp_path / "scipy-seed.camset"
    seed.save(seed_path)
    results = []
    for enabled in (False, True):
        fresh = load_CameraSet(seed_path)
        handler, result, _solved, _stats = phase3.solve(
            fresh, detections, target, fixed_params={},
            options={"solver": "trf", "loss": "linear", "max_nfev": 30},
            threads=1, record_education=enabled)
        results.append((handler, result))
    np.testing.assert_array_equal(results[0][1].x, results[1][1].x)
    np.testing.assert_array_equal(results[0][1].fun, results[1][1].fun)
    assert results[0][1].status == results[1][1].status
    assert not hasattr(results[0][0], "education_step_trace")
    assert results[1][0].education_step_trace["endpoints_only"] is False

    monkeypatch.setattr(
        optimisation_handling, "_least_squares_supports_callback", lambda: False)
    fallback = load_CameraSet(seed_path)
    handler, result, _solved, _stats = phase3.solve(
        fallback, detections, target, fixed_params={},
        options={"solver": "trf", "loss": "linear", "max_nfev": 30},
        threads=1, record_education=True)
    assert handler.education_step_trace["endpoints_only"] is True
    assert handler.education_step_trace["steps"][0]["nit"] == 0
    assert handler.education_step_trace["steps"][-1]["x"] == result.x.tolist()


def test_enabled_lockbox_recording_preserves_full_residual_and_endpoints(
        charuco_problem, tmp_path):
    from pyCamSet.optimisation.camera_lockbox import CameraLockboxConfig
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.phase3 import solve

    target, detections, seed = charuco_problem
    path = tmp_path / "lockbox-seed.camset"
    seed.save(path)
    _handler0, _result0, solved_rig, _stats0 = solve(
        load_CameraSet(path), detections, target, fixed_params={},
        options={"outliers": "n", "max_nfev": 12}, threads=1)
    rig_path = tmp_path / "completed-rig.camset"
    solved_rig.save(rig_path)
    config = CameraLockboxConfig(enabled=True, center_sigma=0.01)
    paired = []
    for enabled in (False, True):
        handler, result, _solved, _stats = solve(
            load_CameraSet(path), detections, target, fixed_params={},
            options={"outliers": "n", "max_nfev": 30}, threads=1,
            lockbox_config=config, lockbox_source_camset=load_CameraSet(rig_path),
            record_education=enabled)
        paired.append((handler, result))
    off, on = paired
    assert on[1].fun.size > on[0].get_base_residual_count()
    np.testing.assert_array_equal(off[1].x, on[1].x)
    np.testing.assert_array_equal(off[1].fun, on[1].fun)
    assert off[1].cost == on[1].cost
    assert off[1].nfev == on[1].nfev
    assert off[1].status == on[1].status
    assert on[0].education_step_trace["steps"][-1]["x"] == on[1].x.tolist()


def test_phase4_recording_toggle_preserves_result_and_saved_endpoint(
        charuco_problem, tmp_path):
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.phase3 import solve as solve_phase3
    from pyCamSet.workflow.phase4 import solve as solve_phase4

    target, detections, seed = charuco_problem
    _handler3, _result3, phase3_cams, _stats3 = solve_phase3(
        seed, detections, target, fixed_params={},
        options={"outliers": "n", "max_nfev": 30}, threads=1)
    phase3_path = tmp_path / "phase3.camset"
    phase3_cams.save(phase3_path)
    paired = []
    for enabled in (False, True):
        fresh_phase3 = load_CameraSet(phase3_path)
        handler, result, solved, _stats = solve_phase4(
            fresh_phase3, target, detections, fixed_params={},
            options={"loss": "soft_l1", "max_nfev": 30},
            threads=1, record_education=enabled)
        paired.append((handler, result, solved))
    off, on = paired
    np.testing.assert_array_equal(off[1].x, on[1].x)
    np.testing.assert_array_equal(off[1].fun, on[1].fun)
    assert off[1].cost == on[1].cost
    assert off[1].status == on[1].status
    assert off[1].nit == on[1].nit
    assert not hasattr(off[0], "education_step_trace")
    payload = on[0].education_step_trace
    assert payload["steps"][-1]["x"] == on[1].x.tolist()
    np.testing.assert_array_equal(on[2].calibration_params, on[1].x)
