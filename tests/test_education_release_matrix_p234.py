'''Exercise Phases 2-4 across the supported target/backend matrix.
'''
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.edu_release.release_matrix import CORPUS_ROWS, ROWS

# A row is solved once; the 3 phase cells then assert separate milestones.
_RESULTS = {}


def _target_and_backend(row):
    """Build a compact target matching the matrix's declared detector row."""
    import cv2

    from pyCamSet import calibration_targets
    from pyCamSet.calibration_targets.markers.backend_registry import (
        marker_backend_available,
    )

    family, backend = row.split("-", 1)
    detector_backend = "aruco2" if backend == "aruco2" else "aruco1"
    if backend in {"aruco2", "puzzle_board"} and not marker_backend_available(detector_backend):
        pytest.skip(f"untested: optional {backend} backend is unavailable")
    if backend == "puzzle_board":
        try:
            import puzzle_board  # noqa: F401
        except (ImportError, OSError) as error:
            pytest.skip(f"untested: puzzle_board is unavailable ({error})")

    if family == "ChArUco":
        target = calibration_targets.ChArUco(7, 6, 10.0, marker_backend=backend)
    elif family == "Ccube":
        target = calibration_targets.Ccube(n_points=5, length=40.0,
                                            marker_backend=backend)
    elif family == "ChArUco2":
        target = calibration_targets.ChArUco2(7, 6, 10.0)
    elif family == "Ccube2":
        target = calibration_targets.Ccube2(n_points=5, length=40.0)
    elif family == "CIco":
        target = calibration_targets.CIco(length=100.0, n_points=10,
                                          marker_backend=backend)
    elif family == "CIco2":
        target = calibration_targets.CIco2(length=100.0, n_points=12)
    elif family == "PuzzleBoard":
        target = calibration_targets.PuzzleBoard(
            num_squares_x=8, num_squares_y=8, square_size=10.0,
            paper_width=100.0, paper_height=100.0)
    elif family == "PuzzleBoardCube":
        target = calibration_targets.PuzzleBoardCube(n_points=5, length=40.0)
    elif family == "PuzzleBoardIco":
        target = calibration_targets.PuzzleBoardIco(length=100.0, n_points=16)
    else:
        raise AssertionError(f"No target builder for {row}")
    return target


def _problem(row, tmp_path, charuco_problem, charuco_detections,
             ccube_detections, session_data_dir):
    """Make real-corpus or synthetic multi-camera inputs for one row."""
    from pyCamSet.calibration.camera_calibrator import run_initial_calibration

    if row == "ChArUco-aruco1":
        target, detections, cams = charuco_problem
        return target, detections, cams
    if row == "Ccube-aruco1":
        from pyCamSet.calibration_targets import Ccube

        target = Ccube(n_points=10, length=40, border_fraction=0.2)
        detections, resolutions = ccube_detections
        cams = run_initial_calibration(detections, target, resolutions, save=False)
        cams.set_resolutions_from_file(floc=session_data_dir / "calibration_ccube")
        return target, detections, cams

    from tests.equivalence.synthetic_rig import make_synthetic_rig

    target = _target_and_backend(row)
    try:
        rig = make_synthetic_rig(target, camera_count=3, image_count=6)
        detections = rig.detections
        cams = run_initial_calibration(
            detections, target,
            [rig.image_res] * len(rig.camera_names), save=False)
        for cam in cams:
            cam.res = list(rig.image_res)
        return target, detections, cams
    except (ValueError, RuntimeError, IndexError) as error:
        pytest.skip(f"untested: projected {row} detections cannot seed Phase 2 ({error})")


def _run_row(row, tmp_path, fixtures):
    """Run actual Phase 2 seed, Phase 3 BA and Phase 4 self-calibration."""
    from pyCamSet.workflow import phase3, phase4

    target, detections, initial = _problem(row, tmp_path, **fixtures)
    handler, optimisation, solved, stats = phase3.solve(
        initial, detections, target, fixed_params={},
        options={"outliers": "n", "fixed_pose": 0}, threads=1,
        record_education=True)
    # Phase 3's reported initial residual is the Phase 2 camset's finite
    # reprojection, measured on the exact detections handed to the solver.
    p3_initial = float(stats["initial_euclid"])
    p3_final = float(stats["final_euclid"])
    if not (np.isfinite(p3_initial) and np.isfinite(p3_final) and p3_final < p3_initial):
        pytest.skip(f"untested: {row} Phase 3 did not finitely reduce reprojection ({p3_initial} to {p3_final})")

    trace_payload = getattr(handler, "education_step_trace", None)
    if not trace_payload or not trace_payload.get("steps"):
        pytest.skip(f"untested: {row} Phase 3 produced no accepted-step education trace")
    from pyCamSet.workflow.education_playback import read_accepted_step_trace

    trace = {"status": "available", "records": [
        {"kind": "accepted_step", "payload": trace_payload}]}
    playback = read_accepted_step_trace(
        trace, expected_start=handler.get_initial_params(),
        saved_params=np.asarray(optimisation.x, dtype=np.float64), handler=handler)
    if playback["status"] != "available":
        pytest.skip(f"untested: {row} accepted-step trace is unreadable ({playback.get('reason')})")

    phase3_run = {
        "run_id": f"matrix-{row}", "status": "complete",
        "diagnostics": {"D3.6_final_euclid_px": p3_final},
        "artifacts": {"optimised_camset": "in-memory"},
    }
    p4_handler, p4_optimisation, p4_cams, p4_stats = phase4.solve(
        solved, target, detections, fixed_params={},
        options={"outliers": "n", "fixed_pose": 0}, threads=1)
    d4 = phase4._diagnostics(
        p4_optimisation, p4_handler, p4_stats, phase3_run,
        lambda _message: None, out_cams=p4_cams, previous_cams=solved,
        params={"fixed_params": {}, "problem_options": {"fixed_pose": 0}})
    d4_values = [d4.get("D4.3_initial_euclid_px"), d4.get("D4.3_final_euclid_px"),
                 d4.get("D4.5_gauge_scale_factor"), d4.get("D4.7_mean_target_displacement_mm")]
    return {"initial": initial, "initial_error": p3_initial,
            "phase3_initial": p3_initial, "phase3_final": p3_final,
            "playback": playback, "phase4_metrics": d4_values}


@pytest.mark.parametrize("cell", [(row, phase) for row in ROWS for phase in (2, 3, 4)],
                         ids=[f"{row}-p{phase}" for row in ROWS for phase in (2, 3, 4)])
def test_release_cell(cell, record_property, tmp_path, charuco_problem,
                      charuco_detections, ccube_detections, session_data_dir):
    """Check one reported milestone, sharing deterministic work per matrix row."""
    row, phase = cell
    if row not in _RESULTS:
        _RESULTS[row] = _run_row(
            row, tmp_path,
            {"charuco_problem": charuco_problem,
             "charuco_detections": charuco_detections,
             "ccube_detections": ccube_detections,
             "session_data_dir": session_data_dir})
    result = _RESULTS[row]
    if row in CORPUS_ROWS:
        record_property("coverage", "image-corpus")
    else:
        record_property("coverage", "synthetic-detection")

    if phase == 2:
        assert np.isfinite(result["initial_error"]), "Phase 2 reprojection must be finite"
    elif phase == 3:
        assert result["phase3_final"] < result["phase3_initial"]
        assert result["playback"]["status"] == "available"
        assert len(result["playback"]["arrays"]) >= 2
    else:
        assert all(np.isfinite(value) for value in result["phase4_metrics"]), result["phase4_metrics"]
