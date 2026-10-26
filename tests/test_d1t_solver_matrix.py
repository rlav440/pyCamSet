"""Solver-path cells for the education recording, closed without real-data fixtures.

Contract: (recording off/on bit-identical "with one numba thread and the default
thread count, Schur auto and forced parallel (``SchurSolver.PARALLEL_MIN_DETECTIONS = 0``)", "plus the
existing forced-parallel check against serial (<=1e-9 px)").

Why synthetic: the Schur "auto" mode only goes parallel at >= 15,000 detections
(``optimisation/numba_schur.py`` ``PARALLEL_MIN_DETECTIONS``) and the largest permitted repo fixture has
fewer than 5,000.  The tests force the parallel path by lowering the threshold; the natural
(unforced) branch is exercised here on the synthetic multi-camera generator
(``tests/equivalence/synthetic_rig.py``), which projects known geometry - no images, no private data.
"""
from __future__ import annotations

import numpy as np
import pytest

from equivalence.synthetic_rig import make_synthetic_rig

pytestmark = [pytest.mark.needs_jit]


def _noisy(detections, sigma_px: float, seed: int):
    """A copy of *detections* with deterministic pixel noise, so the solve does real work."""
    from pyCamSet.calibration_targets.core.target_detections import TargetDetection

    data = np.array(detections.get_data(), dtype=np.float64, copy=True)
    rng = np.random.default_rng(seed)
    data[:, -2:] += rng.normal(0.0, sigma_px, size=(len(data), 2))
    return TargetDetection(cam_names=list(detections.cam_names), data=data, max_ims=detections.max_ims)


@pytest.mark.slow
def test_natural_auto_parallel_branch_records_bit_identically(tmp_path, monkeypatch):
    """>= 15,000 synthetic detections: 'auto' picks the parallel kernel on its own; recording changes nothing."""
    from pyCamSet import ChArUco
    from pyCamSet.calibration.camera_calibrator import run_initial_calibration
    from pyCamSet.optimisation import numba_schur
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.phase3 import solve

    assert numba_schur.SchurSolver.PARALLEL_MIN_DETECTIONS == 15_000, "the threshold itself must stay unchanged"
    target = ChArUco(num_squares_x=20, num_squares_y=20, square_size=4.0, legacy=False)
    rig = make_synthetic_rig(target, camera_count=3, image_count=16, image_res=(1080, 1440))
    detections = _noisy(rig.detections, sigma_px=0.25, seed=20261004)
    n_det = len(detections.get_data())
    assert n_det >= 15_000, f"synthetic rig produced only {n_det} detections"

    seed_cams = run_initial_calibration(detections, target, [(1080, 1440)] * len(rig.camera_names), save=False)
    seed_path = tmp_path / "seed.camset"
    seed_cams.save(seed_path)

    calls = {"parallel": 0}
    original = numba_schur._accumulate_parallel

    def counting(*args, **kwargs):
        calls["parallel"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(numba_schur, "_accumulate_parallel", counting)
    results = []
    for enabled in (False, True):
        before = calls["parallel"]
        handler, result, _solved, _stats = solve(
            load_CameraSet(seed_path), detections, target,
            fixed_params={}, options={"outliers": "n", "max_nfev": 60},
            threads=2, record_education=enabled)
        assert calls["parallel"] > before, f"record_education={enabled}: the parallel kernel never ran"
        results.append((result, handler))
    (off, _), (on, on_handler) = results
    assert off.nit > 1, "the noisy solve should take accepted steps"
    np.testing.assert_array_equal(off.x, on.x)
    np.testing.assert_array_equal(off.fun, on.fun)
    assert (off.cost, off.status, off.nit) == (on.cost, on.status, on.nit)
    assert on_handler.education_step_trace["steps"][-1]["x"] == on.x.tolist()


@pytest.mark.slow
@pytest.mark.data
def test_forced_parallel_matches_serial_reprojection_within_1e9_px(charuco_problem, tmp_path, monkeypatch):
    """The plan's forced-parallel check against serial, on the repo ChArUco corpus, recording off."""
    from pyCamSet.optimisation.numba_schur import SchurSolver
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.phase3 import solve

    target, detections, seed = charuco_problem
    seed_path = tmp_path / "seed.camset"
    seed.save(seed_path)

    def mean_px(result) -> float:
        r = np.asarray(result.fun, dtype=np.float64).reshape(-1, 2)
        return float(np.mean(np.linalg.norm(r, axis=1)))

    serial = solve(load_CameraSet(seed_path), detections, target, fixed_params={},
                   options={"outliers": "n", "max_nfev": 100}, threads=1)[1]
    monkeypatch.setattr(SchurSolver, "PARALLEL_MIN_DETECTIONS", 0)
    parallel = solve(load_CameraSet(seed_path), detections, target, fixed_params={},
                     options={"outliers": "n", "max_nfev": 100}, threads=2)[1]
    assert abs(mean_px(serial) - mean_px(parallel)) <= 1e-9, (mean_px(serial), mean_px(parallel))


@pytest.mark.slow
@pytest.mark.data
def test_phase4_soft_l1_forced_parallel_records_bit_identically(charuco_problem, tmp_path, monkeypatch):
    """Phase 4 (soft_l1, its GUI default) on the forced-parallel Schur path: recording changes nothing."""
    from pyCamSet.optimisation.numba_schur import SchurSolver
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow.phase3 import solve as solve_phase3
    from pyCamSet.workflow.phase4 import solve as solve_phase4

    target, detections, seed = charuco_problem
    phase3_cams = solve_phase3(seed, detections, target, fixed_params={},
                               options={"outliers": "n", "max_nfev": 30}, threads=1)[2]
    phase3_path = tmp_path / "phase3.camset"
    phase3_cams.save(phase3_path)
    monkeypatch.setattr(SchurSolver, "PARALLEL_MIN_DETECTIONS", 0)
    paired = []
    for enabled in (False, True):
        handler, result, solved, _stats = solve_phase4(
            load_CameraSet(phase3_path), target, detections, fixed_params={},
            options={"loss": "soft_l1", "max_nfev": 30}, threads=2, record_education=enabled)
        paired.append((handler, result, solved))
    (_, off, _), (on_handler, on, on_solved) = paired
    np.testing.assert_array_equal(off.x, on.x)
    np.testing.assert_array_equal(off.fun, on.fun)
    assert (off.cost, off.status, off.nit) == (on.cost, on.status, on.nit)
    assert on_handler.education_step_trace["steps"][-1]["x"] == on.x.tolist()
    np.testing.assert_array_equal(on_solved.calibration_params, on.x)
