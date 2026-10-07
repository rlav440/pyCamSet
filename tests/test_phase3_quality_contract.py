'''Purpose: Pin Phase 3's machine-readable quality disposition.
Status: Active regression coverage for the pcube Phase-3 contract.
Future: Extend with real artefact fixtures when a compact fixture is available.
'''
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pyCamSet.workflow import phase3
from pyCamSet.workflow.run_quality import blocking_reasons


class _Handler:
    cam_names = ["cam0"]
    missing_poses = np.array([False])

    def get_detection_data(self, flatten=True):
        return np.array([[0, 0, 0, 0.0, 0.0, 0.0, 0.0]])

    def get_base_residual_count(self):
        return 2


def test_phase3_quality_gate_does_not_call_an_unfinished_solver_complete():
    optimisation = SimpleNamespace(
        fun=np.array([1.0, 0.0]),
        x=np.array([2.0]),
        success=False,
        status=0,
        message="maximum evaluations",
        nfev=10,
    )
    diagnostics = phase3._diagnostics(
        optimisation,
        _Handler(),
        {
            "initial_euclid": 3.0,
            "final_euclid": 1.0,
            "param_count": 1,
            "observation_count": 1,
            "success": False,
            "status": 0,
            "message": "maximum evaluations",
            "nfev": 10,
        },
        3.0,
        1.0,
        lambda _line: None,
    )

    gate = diagnostics["quality_gate"]
    assert gate["status"] == "incomplete"
    assert any("solver" in flag for flag in gate["blocking_flags"])


def test_phase3_quality_gate_accepts_a_finite_reduced_successful_solve():
    optimisation = SimpleNamespace(
        fun=np.array([0.3, 0.4]),
        x=np.array([2.0]),
        success=True,
        status=1,
        message="gtol termination",
        nfev=3,
    )
    diagnostics = phase3._diagnostics(
        optimisation,
        _Handler(),
        {
            "initial_euclid": 3.0,
            "final_euclid": 0.5,
            "param_count": 1,
            "observation_count": 1,
            "success": True,
            "status": 1,
            "message": "gtol termination",
            "nfev": 3,
        },
        3.0,
        0.5,
        lambda _line: None,
    )

    gate = diagnostics["quality_gate"]
    assert gate["status"] == "complete"
    assert gate["blocking_flags"] == []
    assert gate["finite_parameters"] is True
    assert gate["finite_residuals"] is True
    assert gate["error_reduced"] is True


def test_gui_quality_gate_uses_the_backend_disposition():
    reasons = blocking_reasons({
        "status": "incomplete",
        "error": None,
        "report": {},
        "diagnostics": {
            "quality_gate": {
                "blocking_flags": ["solver did not report successful termination"]
            }
        },
    })

    assert reasons == ["solver did not report successful termination"]
