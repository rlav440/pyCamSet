"""
Phase 4: self-calibration, letting the target's own shape move.

Phase 3 solved the cameras against a target assumed to be exactly as drawn.
Here the target's points become free parameters too, gauged against a fixed
subset so the solve cannot simply scale everything away.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from pyCamSet.workflow.diagnostics import per_camera_mean_reprojection
from pyCamSet.workflow.logs import LogFn, captured_output, discard
from pyCamSet.workflow.workspace import (
    WorkspaceManager,
    as_io_path,
    make_run_id,
)

_LOG = logging.getLogger(__name__)

try:
    from pyCamSet.optimisation.optimisation_handling import (
        run_bundle_adjustment_with_stats,
    )
    from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler
    from pyCamSet.utils.saving import load_CameraSet

    BACKEND_OK = True
except ImportError as exc:
    run_bundle_adjustment_with_stats = None
    SelfBundleHandler = None
    load_CameraSet = None
    BACKEND_OK = False
    # See the note on the same guard in phase3.
    _LOG.warning("Phase 4 optimisation backend unavailable: %s", exc)


def run(params: dict,
        workspace: WorkspaceManager,
        log: LogFn = discard,
        phase3_run: Optional[dict] = None,
        phase3_camset: Optional[Path] = None) -> dict:
    """
    Refine the target's own shape alongside the cameras.

    :param params: the phase's settings, including its ``problem_options``
    :param workspace: the workspace to save the run to
    :param log: what to call with each line of output
    :param phase3_run: the run the starting camset came from
    :param phase3_camset: the camset to start from
    :return: the run's metadata record, saved to the workspace
    """
    if workspace.workspace_path is None:
        raise RuntimeError("Workspace path is not set.")
    if phase3_camset is None:
        raise RuntimeError("Phase 4 needs a Phase 3 camset to start from.")

    run_id = make_run_id()
    run_dir = workspace.run_dir("phase4", run_id)

    diagnostics: dict = {}
    error: Optional[str] = None
    camset_out: Optional[Path] = None

    with captured_output(log):
        try:
            camset_out, diagnostics = _solve(
                params, run_dir, Path(phase3_camset), phase3_run, log)
        except Exception as exc:
            error = str(exc)
            log(f"ERROR: {error}")

    artifacts: dict = {"phase3_camset_used": str(phase3_camset)}
    if camset_out is not None:
        artifacts["self_calibrated_camset"] = str(camset_out)
        # Under both names: this is the optimised camset of the run, and
        # whatever reads a run's result should not have to know which phase
        # produced it.
        artifacts["optimised_camset"] = str(camset_out)

    metadata = {
        "run_id": run_id,
        "phase": "phase4",
        "params": params,
        "diagnostics": diagnostics,
        "error": error,
        "inputs": {
            "phase3_run_id": phase3_run.get("run_id") if phase3_run else None,
        },
        "artifacts": artifacts,
    }
    workspace.save_run("phase4", run_id, metadata)
    if error is None:
        log(f"Run saved: {run_id}")
    return metadata


def solve(previous_cams, target, detections, *,
          fixed_params: Optional[dict] = None,
          options: Optional[dict] = None,
          threads: int = 1):
    """
    Self-calibrate from a solved camera set, from objects in hand.

    The solve, with nothing around it: no workspace, no artifact paths, no
    run record.  The phase runner below wraps it in those; a parameter
    search calls it directly, once per trial.

    :param previous_cams: the phase 3 cameras to start from
    :param target: the calibration target, whose points become free
    :param detections: the detections to solve against
    :param fixed_params: parameters to pin rather than solve for
    :param options: the solver options, as the handler reads them
    :param threads: Jacobian evaluation threads
    :return: the handler, the solver result, the solved cameras, the stats
    """
    if not BACKEND_OK:
        raise RuntimeError("pyCamSet optimisation modules are not importable.")

    handler = SelfBundleHandler(
        camset=previous_cams,
        target=target,
        detection=detections,
        fixed_params=fixed_params,
        options=options,
    )
    handler.set_from_templated_camset(previous_cams)
    optimisation, out_cams, stats = run_bundle_adjustment_with_stats(
        handler, threads=threads)
    return handler, optimisation, out_cams, stats


def _solve(params: dict, run_dir: Path, phase3_camset: Path,
           phase3_run: Optional[dict], log: LogFn) -> tuple[Path, dict]:
    """Run the self-calibration and compute its diagnostics."""
    if not BACKEND_OK:
        raise RuntimeError("pyCamSet optimisation modules are not importable.")

    previous_cams = load_CameraSet(as_io_path(phase3_camset))
    selected = list(params.get("selected_cameras") or [])
    if selected and set(previous_cams.get_names()) != set(selected):
        raise RuntimeError(
            "Phase 3 camset cameras do not match selected camera subset. "
            "Re-run Phase 3 with the same selected cameras.")

    previous_handler = getattr(previous_cams, "calibration_handler", None)
    if previous_handler is None:
        raise RuntimeError(
            "Selected Phase 3 camset has no calibration handler metadata.")

    handler, optimisation, out_cams, stats = solve(
        previous_cams, previous_handler.target, previous_handler.detection,
        fixed_params=params["fixed_params"],
        options=params["problem_options"],
        threads=params["threads"],
    )

    # A fixed short filename, so a long source folder does not push the run
    # past Windows' path limit.
    camset_out = run_dir / "self_calibrated_cameras.camset"
    out_cams.save(camset_out)

    diagnostics = _diagnostics(
        optimisation, handler, stats, phase3_run, log)
    return camset_out, diagnostics


def _diagnostics(optimisation, handler, stats: dict,
                 phase3_run: Optional[dict], log: LogFn) -> dict:
    """The D4 series: how far the target moved, and whether it paid off."""
    initial_euclid = float(stats.get("initial_euclid", float("nan")))
    final_euclid = float(stats.get("final_euclid", float("nan")))
    log(f"D4.3  Initial Euclidean reprojection error: {initial_euclid:.4f} px")
    log(f"D4.3  Final Euclidean reprojection error: {final_euclid:.4f} px")

    phase3_final = float(
        (phase3_run.get("diagnostics") or {}).get(
            "D3.6_final_euclid_px", float("nan"))) if phase3_run else float("nan")
    improvement = (float(phase3_final - final_euclid)
                   if np.isfinite(phase3_final) else float("nan"))
    log(f"D4.4  Improvement vs Phase 3 final error: {improvement:.4f} px")

    visible = np.array(getattr(handler, "visible_feature_mask", []), dtype=bool)
    fixed_indices = [int(i) for i in getattr(handler, "fixed_inds", [])]
    scale, displacement_mm = _target_shape_change(handler, optimisation, visible)

    raw_initial = getattr(handler, "initial_per_im_error", None)
    per_image_initial = (np.asarray(raw_initial, dtype=float)
                         if raw_initial is not None else np.array([]))

    per_camera, _ = per_camera_mean_reprojection(
        optimisation, handler, log, "D4.12")

    diagnostics = {
        "D4.1_n_free_target_points": int(np.sum(visible)),
        "D4.2_gauge_fixed_points": {
            "count": len(fixed_indices), "indices": fixed_indices},
        "D4.3_per_image_initial_reprojection": per_image_initial.tolist(),
        "D4.3_initial_euclid_px": initial_euclid,
        "D4.3_final_euclid_px": final_euclid,
        "D4.4_vs_phase3_delta_px": improvement,
        "D4.5_gauge_scale_factor": scale,
        "D4.7_mean_target_displacement_mm": displacement_mm,
        "D4.8_shape_change_arrows": "available in Assess Calibration",
        "D4.9_planarity_rms_mm": "available in backend special_plots",
        "D4.10_accuracy_precision": "available in Assess Calibration",
        "D4.12_per_camera_mean_reprojection": per_camera,
    }

    log(f"D4.1  Free target points: {diagnostics['D4.1_n_free_target_points']}")
    log(f"D4.2  Gauge-fixed points: count={len(fixed_indices)}, "
        f"indices={fixed_indices}")
    log(f"D4.5  Gauge scale factor: {scale:.6f}")
    log(f"D4.7  Mean target displacement: {displacement_mm:.5f} mm")
    _log_extremes(per_camera, log)
    return diagnostics


def _target_shape_change(handler, optimisation, visible) -> tuple[float, float]:
    """
    How much the solve rescaled and moved the target's points.

    Measured over the visible features only.  A flat PuzzleBoard's point data
    spans the whole 251,001-position virtual field while only the printed
    window is ever detected; the undetected points stay at their initial
    values, and including them would pull the median scale toward 1 and the
    mean displacement toward 0.  Every other target has no such padding, so
    the mask changes nothing for them.
    """
    updated = np.array(handler.get_updated_target(optimisation.x), dtype=float)
    reference = np.array(handler.target.point_data, dtype=float).reshape(-1, 3)

    if visible.size == reference.shape[0] and np.any(visible):
        updated, reference = updated[visible], reference[visible]

    with np.errstate(invalid="ignore", divide="ignore"):
        reference_norm = np.linalg.norm(reference, axis=1)
        updated_norm = np.linalg.norm(updated, axis=1)
        ratio = reference_norm / np.where(updated_norm == 0.0, np.nan, updated_norm)
    displacement = np.linalg.norm(updated - reference, axis=1)

    scale = float(np.nanmedian(ratio)) if ratio.size else float("nan")
    displacement_mm = (float(np.nanmean(displacement) * 1000.0)
                       if displacement.size else float("nan"))
    return scale, displacement_mm


def _log_extremes(per_camera: dict[str, float], log: LogFn) -> None:
    """Name the best and worst camera, when there is one to name."""
    if not per_camera:
        return
    valid = {name: value for name, value in per_camera.items()
             if np.isfinite(value)}
    if not valid:
        log("D4.12  Per-camera mean reprojection: no valid cameras")
        return
    best = min(valid, key=valid.get)
    worst = max(valid, key=valid.get)
    log(f"D4.12  Per-camera mean reprojection: best={best}={valid[best]:.2f}px, "
        f"worst={worst}={valid[worst]:.2f}px")
