'''Verify accepted-step trace reader and frame-builder edge cases.
'''
import numpy as np

from pyCamSet.workflow.education_playback import read_accepted_step_trace


def _trace(steps, *, endpoints_only=False):
    return {"status": "available", "records": [{
        "kind": "accepted_step",
        "payload": {"steps": steps, "endpoints_only": endpoints_only},
    }]}


def test_trace_with_only_start_state_is_no_accepted_steps():
    result = read_accepted_step_trace(
        _trace([{"nit": 0, "cost": 1.0, "x": [1.0], "dtype": "float64"}]),
        expected_start=np.array([1.0]), saved_params=np.array([1.0]))
    assert result == {"status": "unavailable", "reason": "no accepted steps"}


def test_float32_interior_is_allowed_but_endpoints_stay_exact():
    result = read_accepted_step_trace(_trace([
        {"nit": 0, "cost": 2.0, "x": [0.0, 0.0], "dtype": "float64"},
        {"nit": 1, "cost": 1.0, "x": [0.5, 0.5], "dtype": "float32"},
        {"nit": 2, "cost": 0.5, "x": [1.0, 1.0], "dtype": "float64"},
    ]), expected_start=np.array([0.0, 0.0]), saved_params=np.array([1.0, 1.0]))
    assert result["status"] == "available"
    assert result["arrays"][0].dtype == np.float64
    assert result["arrays"][-1].dtype == np.float64
    assert result["arrays"][1].dtype == np.float32


def test_unavailable_sidecar_fails_closed():
    assert read_accepted_step_trace(
        {"status": "unavailable", "reason": "checksum mismatch"},
        expected_start=np.array([0.0]), saved_params=np.array([1.0])) == {
            "status": "unavailable", "reason": "checksum mismatch"}
