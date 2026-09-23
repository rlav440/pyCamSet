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
    report: Optional[dict] = None
    error: Optional[str] = None
    camset_out: Optional[Path] = None

    with captured_output(log):
        try:
            camset_out, diagnostics, report = _solve(
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

    quality_gate = (diagnostics.get("quality_gate") or {}) if error is None else {}
    metadata = {
        "run_id": run_id,
        "phase": "phase4",
        "status": (
            "failed" if error else
            quality_gate.get("status", "incomplete")
        ),
        "params": params,
        "diagnostics": diagnostics,
        "report": report,
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
           phase3_run: Optional[dict],
           log: LogFn) -> tuple[Path, dict, Optional[dict]]:
    """Run the self-calibration and compute its diagnostics."""
    if not BACKEND_OK:
        raise RuntimeError("pyCamSet optimisation modules are not importable.")

    previous_cams = load_CameraSet(as_io_path(phase3_camset))
    if phase3_run is not None and phase3_run.get("status") == "failed":
        raise RuntimeError(
            "The selected Phase 3 run failed; refusing to start Phase 4. "
            "Re-run Phase 3 first.")
    selected = list(params.get("selected_cameras") or [])
    if selected and set(previous_cams.get_names()) != set(selected):
        raise RuntimeError(
            "Phase 3 camset cameras do not match selected camera subset. "
            "Re-run Phase 3 with the same selected cameras.")

    previous_handler = getattr(previous_cams, "calibration_handler", None)
    if previous_handler is None:
        raise RuntimeError(
            "Selected Phase 3 camset has no calibration handler metadata.")
    previous_detection = getattr(previous_handler, "detection", None)
    detection_names = set(getattr(previous_detection, "cam_names", []) or [])
    if detection_names != set(previous_cams.get_names()):
        raise RuntimeError(
            "Selected Phase 3 observations do not cover the saved camera set; "
            "refusing to run Phase 4 with silent camera loss.")

    options = dict(params.get("problem_options") or {})
    options.setdefault("fixed_pose", 0)
    options.setdefault("ref_cam", 0)
    options.setdefault("ref_pose", 0)
    options.setdefault("outliers", "n")
    options.setdefault("max_nfev", 100)
    handler, optimisation, out_cams, stats = solve(
        previous_cams, previous_handler.target, previous_handler.detection,
        fixed_params=params.get("fixed_params"),
        options=options,
        threads=int(params.get("threads", 1)),
    )

    # A fixed short filename, so a long source folder does not push the run
    # past Windows' path limit.
    camset_out = run_dir / "self_calibrated_cameras.camset"
    out_cams.save(camset_out)

    # A Phase 4 result is not accepted merely because the solver returned.
    # Reload the exact bytes written to disk and verify the camera identity
    # before recording a quality disposition.
    reloaded = load_CameraSet(as_io_path(camset_out))
    if set(reloaded.get_names()) != set(previous_cams.get_names()):
        raise RuntimeError(
            "Phase 4 save/reload changed the active camera set; refusing "
            "to publish a result with silent camera loss.")

    diagnostics = _diagnostics(
        optimisation, handler, stats, phase3_run, log,
        out_cams=out_cams, previous_cams=previous_cams,
        params=params, phase3_status=(phase3_run or {}).get("status"))
    report = getattr(out_cams, "calibration_report", None)
    return camset_out, diagnostics, (
        report.to_dict() if report is not None else None)


def _diagnostics(optimisation, handler, stats: dict,
                 phase3_run: Optional[dict], log: LogFn,
                 *, out_cams=None, previous_cams=None,
                 params: Optional[dict] = None,
                 phase3_status: Optional[str] = None) -> dict:
    """The D4 series: how far the target moved, and whether it paid off."""
    # The errors and the solver's own account of the run are the calibration
    # summary's, printed by the solve itself.  What follows is what only this
    # phase knows: how far the target's own shape moved, and whether letting
    # it move paid for itself.
    initial_euclid = float(stats.get("initial_euclid", float("nan")))
    final_euclid = float(stats.get("final_euclid", float("nan")))

    phase3_final = float(
        (phase3_run.get("diagnostics") or {}).get(
            "D3.6_final_euclid_px", float("nan"))) if phase3_run else float("nan")
    improvement = (float(phase3_final - final_euclid)
                   if np.isfinite(phase3_final) else float("nan"))
    log(f"D4.4  Improvement vs Phase 3 final error: {improvement:.4f} px")

    visible = np.array(getattr(handler, "visible_feature_mask", []), dtype=bool)
    fixed_indices = [int(i) for i in getattr(handler, "fixed_inds", [])]
    scale, displacement_mm = _target_shape_change(handler, optimisation, visible)

    per_image_initial = _initial_per_image_errors(handler, stats)

    per_camera, _ = per_camera_mean_reprojection(
        optimisation, handler, log, "D4.12")
    detection_data = np.asarray(handler.get_detection_data(flatten=True))
    residual_norm = np.linalg.norm(
        np.asarray(optimisation.fun[:2 * len(detection_data)], dtype=float).reshape(-1, 2),
        axis=1)
    per_image = {
        str(index): float(np.mean(residual_norm[detection_data[:, 1].astype(int) == index]))
        for index in sorted(set(detection_data[:, 1].astype(int).tolist()))
    }
    quality_gate = _quality_gate(
        optimisation, handler, stats, residual_norm.reshape(-1, 1),
        initial_euclid, final_euclid, int(len(detection_data)),
        out_cams=out_cams, previous_cams=previous_cams,
        params=params, phase3_status=phase3_status)

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
        "D4.13_per_image_mean_reprojection": per_image,
        "quality_gate": quality_gate,
    }

    log(f"D4.1  Free target points: {diagnostics['D4.1_n_free_target_points']}")
    log(f"D4.2  Gauge-fixed points: count={len(fixed_indices)}, "
        f"indices={fixed_indices}")
    log(f"D4.5  Gauge scale factor: {scale:.6f}")
    log(f"D4.7  Mean target displacement: {displacement_mm:.5f} mm")
    return diagnostics


def _initial_per_image_errors(handler, stats: dict) -> np.ndarray:
    """Return initial pose errors even when a handler exposes an empty cache."""
    raw_initial = getattr(handler, "initial_per_im_error", None)
    if raw_initial is not None:
        values = np.asarray(raw_initial, dtype=float)
        if values.size:
            return values
    return np.asarray([
        row.get("initial_error_px", float("nan"))
        for row in stats.get("per_pose_initial_error_px", [])
    ], dtype=float)


def _quality_gate(optimisation, handler, stats: dict,
                  residual_xy: np.ndarray, initial_euclid: float,
                  final_euclid: float, observation_count: int, *,
                  out_cams=None, previous_cams=None,
                  params: Optional[dict] = None,
                  phase3_status: Optional[str] = None) -> dict:
    """Fail closed when Phase 4 produced no scientifically usable result."""
    blocking: list[str] = []
    values = np.asarray(residual_xy, dtype=float)
    finite_parameters = bool(np.all(np.isfinite(np.asarray(
        getattr(optimisation, "x", []), dtype=float))))
    finite_residuals = bool(values.size and np.all(np.isfinite(values)))
    solver_success = bool(stats.get("success", getattr(optimisation, "success", False)))
    error_reduced = bool(
        np.isfinite(initial_euclid) and np.isfinite(final_euclid)
        and final_euclid < initial_euclid)

    if not finite_parameters:
        blocking.append("optimiser parameters are non-finite")
    if not finite_residuals:
        blocking.append("reprojection residuals are missing or non-finite")
    if not solver_success:
        blocking.append("solver did not report successful termination")
    if observation_count <= 0:
        blocking.append("no reprojection observations entered the solve")
    if not error_reduced:
        blocking.append("final reprojection error did not improve finitely")
    if phase3_status not in (None, "complete"):
        blocking.append(
            f"Phase 3 input disposition was {phase3_status}; re-run it before hand-off")

    detection_data = np.asarray(handler.get_detection_data(flatten=True))
    cam_indices = (detection_data[:, 0].astype(int)
                   if detection_data.ndim == 2 and detection_data.shape[1] > 0
                   else np.array([], dtype=int))
    image_indices = (detection_data[:, 1].astype(int)
                     if detection_data.ndim == 2 and detection_data.shape[1] > 1
                     else np.array([], dtype=int))
    expected_cameras = len(getattr(handler, "cam_names", []))
    expected_images = int(getattr(getattr(handler, "detection", None), "max_ims", 0))
    observed_cameras = sorted(set(cam_indices.tolist()))
    observed_images = sorted(set(image_indices.tolist()))
    # ``max_ims`` is the highest global image index plus one, not the number
    # of observed images.  Treat holes as explicitly missing observations;
    # silently pretending the list is contiguous would hide them from the GUI.
    missing_images = sorted(set(range(expected_images)) - set(observed_images))
    camera_coverage = observed_cameras == list(range(expected_cameras))
    image_coverage = bool(
        expected_images and set(observed_images) == set(range(expected_images)))
    if not camera_coverage:
        blocking.append("camera observation graph does not cover every active camera")
    if expected_images and not image_coverage:
        blocking.append("image observation graph has missing image indices")

    gauge = {
        "fixed_target_point_count": len(getattr(handler, "fixed_inds", [])),
        "fixed_target_point_indices": [int(i) for i in getattr(handler, "fixed_inds", [])],
        "fixed_pose": (getattr(handler, "problem_opts", {}) or {}).get("fixed_pose"),
        "ref_cam": (getattr(handler, "problem_opts", {}) or {}).get("ref_cam"),
        "ref_pose": (getattr(handler, "problem_opts", {}) or {}).get("ref_pose"),
    }
    if gauge["fixed_target_point_count"] < 3:
        blocking.append("target gauge does not fix three non-collinear points")
    if gauge["fixed_pose"] is None:
        blocking.append("pose gauge has no fixed reference pose")

    camera_quality = (_camera_quality(out_cams) if out_cams is not None else {
        "physically_plausible": True, "available": False, "cameras": {}})
    if not camera_quality["physically_plausible"]:
        blocking.append("camera intrinsics, distortion, or extrinsics are implausible")

    return {
        "status": "complete" if not blocking else "incomplete",
        "blocking_flags": blocking,
        "finite_parameters": finite_parameters,
        "finite_residuals": finite_residuals,
        "solver_success": solver_success,
        "error_reduced": error_reduced,
        "camera_coverage": camera_coverage,
        "image_coverage": image_coverage,
        "observed_cameras": observed_cameras,
        "expected_camera_count": expected_cameras,
        "observed_images": observed_images,
        "missing_images": missing_images,
        "expected_image_count": expected_images,
        "gauge": gauge,
        "camera_quality": camera_quality,
        "parameter_drift": _camera_drift(previous_cams, out_cams),
        "fixed_params": sorted((params or {}).get("fixed_params") or {}),
        "lockbox": {"enabled": False, "reason": "Phase 4 self-calibration does not apply Phase 3 lockboxes"},
    }


def _camera_quality(cams) -> dict:
    """Check saved camera parameters without assuming a particular lens model."""
    if cams is None:
        return {"physically_plausible": False, "cameras": {}}
    details: dict[str, dict] = {}
    plausible = True
    for cam in cams:
        intrinsic = np.asarray(getattr(cam, "intrinsic", []), dtype=float)
        distortion = np.asarray(getattr(cam, "distortion_coefs", []), dtype=float)
        extrinsic = np.asarray(getattr(cam, "extrinsic", []), dtype=float)
        finite = bool(np.all(np.isfinite(intrinsic)) and np.all(np.isfinite(distortion))
                      and np.all(np.isfinite(extrinsic)))
        shape_ok = intrinsic.shape == (3, 3) and extrinsic.shape == (4, 4)
        focal_ok = bool(shape_ok and intrinsic[0, 0] > 0 and intrinsic[1, 1] > 0)
        rotation = extrinsic[:3, :3] if shape_ok else np.zeros((3, 3))
        rotation_ok = bool(shape_ok and np.isclose(np.linalg.det(rotation), 1.0, atol=0.25)
                           and np.linalg.norm(rotation.T @ rotation - np.eye(3)) < 0.25)
        camera_ok = finite and shape_ok and focal_ok and rotation_ok
        plausible &= camera_ok
        details[str(getattr(cam, "name", len(details)))] = {
            "finite": finite,
            "intrinsic_shape": list(intrinsic.shape),
            "extrinsic_shape": list(extrinsic.shape),
            "positive_focal_lengths": focal_ok,
            "proper_rotation": rotation_ok,
        }
    return {"physically_plausible": plausible, "cameras": details}


def _camera_drift(previous_cams, out_cams) -> dict:
    """Report relative parameter movement between Phase 3 and Phase 4."""
    if previous_cams is None or out_cams is None:
        return {"available": False, "cameras": {}}
    old_names = set(previous_cams.get_names())
    new_names = set(out_cams.get_names())
    values: dict[str, dict[str, float]] = {}
    for name in sorted(old_names & new_names):
        old = previous_cams[name]
        new = out_cams[name]
        values[name] = {}
        for label in ("intrinsic", "distortion_coefs", "extrinsic"):
            before = np.asarray(getattr(old, label), dtype=float)
            after = np.asarray(getattr(new, label), dtype=float)
            denominator = max(float(np.linalg.norm(before)), 1e-12)
            values[name][label] = float(np.linalg.norm(after - before) / denominator)
    return {"available": True, "cameras": values}


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
