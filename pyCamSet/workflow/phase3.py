"""
Phase 3: bundle adjustment over the whole camera set at once.

Starts from phase 2's per-camera intrinsics and phase 1's detections, and
solves for the cameras and the target poses together -- optionally holding the
cameras near a known rig geometry through the lockbox priors.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from pyCamSet.workflow.detections import extract_detection
from pyCamSet.workflow.diagnostics import per_camera_mean_reprojection
from pyCamSet.workflow.logs import (
    LogFn,
    captured_output,
    discard,
    non_interactive_plotting,
)
from pyCamSet.workflow.phase1 import target_from_params
from pyCamSet.workflow.workspace import (
    WorkspaceManager,
    as_io_path,
    make_run_id,
    path_exists,
    resolve_phase1_pickle_artifact,
    resolve_phase2_camset_artifact,
)

_LOG = logging.getLogger(__name__)

try:
    from pyCamSet.optimisation.camera_lockbox import CameraLockboxConfig
    from pyCamSet.optimisation.optimisation_handling import (
        run_bundle_adjustment_with_stats,
    )
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler
    from pyCamSet.utils.saving import load_CameraSet, load_pickle

    BACKEND_OK = True
except ImportError as exc:
    CameraLockboxConfig = None
    run_bundle_adjustment_with_stats = None
    TemplateBundleHandler = None
    load_CameraSet = None
    load_pickle = None
    BACKEND_OK = False
    # Said out loud, because collapsing the cause into a boolean is how a
    # renamed backend symbol came to look like a broken install: the phase
    # reports the modules as unavailable, and nothing says which one or why.
    _LOG.warning("Phase 3 optimisation backend unavailable: %s", exc)


def run(params: dict,
        workspace: WorkspaceManager,
        log: LogFn = discard,
        phase2_run: Optional[dict] = None,
        phase1_run: Optional[dict] = None,
        camset_override: Optional[Path] = None) -> dict:
    """
    Solve the whole camera set against the target together.

    :param params: the phase's settings, including its ``lockbox`` block
    :param workspace: the workspace to read inputs from and save to
    :param log: what to call with each line of output
    :param phase2_run: the run supplying the initial camset
    :param phase1_run: the run supplying the detections
    :param camset_override: an initial camset to use instead of the run's
    :return: the run's metadata record, saved to the workspace
    """
    ws_path = workspace.workspace_path
    if ws_path is None:
        raise RuntimeError("Workspace path is not set.")

    run_id = make_run_id()
    run_dir = workspace.run_dir("phase3", run_id)

    diagnostics: dict = {}
    error: Optional[str] = None
    camset_in: Optional[Path] = None
    detections_path: Optional[Path] = None
    camset_out: Optional[Path] = None

    with captured_output(log), non_interactive_plotting():
        log("Phase 3 running in non-interactive plotting mode (thread-safe).")
        try:
            camset_in = _initial_camset(phase2_run, camset_override, ws_path)
            detections_path = _detections_path(phase1_run, ws_path)
            camset_out, diagnostics = _solve(
                params, run_dir, camset_in, detections_path, log)
        except Exception as exc:
            error = str(exc)
            log(f"ERROR: {error}")

    artifacts: dict = {
        "phase2_initial_camset_used": str(camset_in) if camset_in else None,
        "phase1_detection_pickle_used": (
            str(detections_path) if detections_path else None),
    }
    if camset_out is not None:
        artifacts["optimised_camset"] = str(camset_out)

    metadata = {
        "run_id": run_id,
        "phase": "phase3",
        "params": params,
        "diagnostics": diagnostics,
        "error": error,
        "inputs": {
            "phase2_run_id": phase2_run.get("run_id") if phase2_run else None,
            "phase1_run_id": phase1_run.get("run_id") if phase1_run else None,
        },
        "artifacts": artifacts,
    }
    workspace.save_run("phase3", run_id, metadata)
    if error is None:
        log(f"Run saved: {run_id}")
    return metadata


def _initial_camset(phase2_run: Optional[dict],
                    camset_override: Optional[Path],
                    ws_path: Path) -> Path:
    """The camset the solve starts from."""
    if camset_override is not None and path_exists(camset_override):
        return Path(camset_override)
    resolved = (resolve_phase2_camset_artifact(phase2_run, ws_path)
                if phase2_run else None)
    if resolved is None:
        raise RuntimeError("Phase 2 run is missing initial_camset artifact.")
    if not path_exists(resolved):
        raise RuntimeError(f"Phase 2 camset path does not exist: {resolved}")
    return resolved


def _detections_path(phase1_run: Optional[dict], ws_path: Path) -> Path:
    """The detections the solve reads."""
    resolved = (resolve_phase1_pickle_artifact(phase1_run, ws_path)
                if phase1_run else None)
    if resolved is None:
        raise RuntimeError(
            "Could not resolve Phase 1 detected_datapoints.pickle artifact.")
    return resolved


def _check_camera_subset(selected: list[str], cams, detections) -> None:
    """Refuse inputs built for a different set of cameras than this run."""
    if not selected:
        return
    wanted = set(selected)
    if set(cams.get_names()) != wanted:
        raise RuntimeError(
            "Phase 2 camset cameras do not match selected camera subset. "
            "Re-run Phase 2 with the same selected cameras.")
    if set(getattr(detections, "cam_names", []) or []) != wanted:
        raise RuntimeError(
            "Phase 1 detections do not match selected camera subset. "
            "Re-run Phase 1/2 with the same selected cameras.")


def _lockbox(params: dict, cams, log: LogFn):
    """The lockbox configuration and, when enabled, the rig it locks to."""
    settings = dict(params.get("lockbox") or {})
    config = CameraLockboxConfig(
        enabled=bool(settings.get("enabled", False)),
        rotation_half_width=float(settings.get("rotation_half_width", 0.1)),
        translation_half_width=float(settings.get("translation_half_width", 0.1)),
        rotation_sigma=float(settings.get("rotation_sigma", 0.05)),
        translation_sigma=float(settings.get("translation_sigma", 0.01)),
        center_sigma=float(settings.get("center_sigma", 0.0)),
    )
    if not config.enabled:
        return config, None, settings

    source = settings.get("source_camset")
    if not source:
        raise RuntimeError(
            "Lockbox is enabled but no effective lockbox source path was "
            "provided.")
    log(f"Loading effective lockbox source camset: {source}")
    if settings.get("original_source_camset") and settings.get("edited_source_camset"):
        log(f"Original source camset: {settings['original_source_camset']}")
        log(f"Edited lockbox copy: {settings['edited_source_camset']}")

    source_camset = load_CameraSet(as_io_path(source))
    if set(source_camset.get_names()) != set(cams.get_names()):
        raise RuntimeError(
            "Effective lockbox source camera names do not match the active "
            "Phase 2 camset. This would break TemplateBundleHandler.")
    return config, source_camset, settings


def _solve(params: dict, run_dir: Path, camset_in: Path,
           detections_path: Path, log: LogFn) -> tuple[Path, dict]:
    """Run the bundle adjustment and compute its diagnostics."""
    if not BACKEND_OK:
        raise RuntimeError("pyCamSet optimisation modules are not importable.")

    cams = load_CameraSet(as_io_path(camset_in))
    detections = extract_detection(load_pickle(as_io_path(detections_path)))
    if detections is None:
        raise RuntimeError(
            "Could not extract TargetDetection from Phase 1 pickle.")

    _check_camera_subset(
        list(params.get("selected_cameras") or []), cams, detections)

    target = target_from_params(params)
    lockbox_config, lockbox_source, lockbox_settings = _lockbox(params, cams, log)

    handler = TemplateBundleHandler(
        camset=cams,
        target=target,
        detection=detections,
        fixed_params=params["fixed_params"],
        options=params["problem_options"],
        lockbox_config=lockbox_config,
        lockbox_source_camset=lockbox_source,
        lockbox_warm_start=bool(lockbox_settings.get("warm_start", True)),
    )
    optimisation, out_cams, stats = run_bundle_adjustment_with_stats(
        handler, threads=params["threads"])

    initial_euclid = float(stats.get("initial_euclid", float("nan")))
    final_euclid = float(stats.get("final_euclid", float("nan")))
    log(f"D3.5  Initial Euclidean reprojection error: {initial_euclid:.4f} px")
    log(f"D3.6  Final Euclidean reprojection error: {final_euclid:.4f} px")
    if not bool(stats.get("success", optimisation.success)):
        log(f"Solver note: {stats.get('message', optimisation.message)}")
    log(f"Optimisation finished in "
        f"{float(stats.get('elapsed_sec', float('nan'))):.2f}s")

    camset_out = run_dir / "optimised_cameras.camset"
    out_cams.save(camset_out)

    diagnostics = _diagnostics(
        optimisation, handler, stats, initial_euclid, final_euclid, log)
    return camset_out, diagnostics


def _diagnostics(optimisation, handler, stats: dict,
                 initial_euclid: float, final_euclid: float,
                 log: LogFn) -> dict:
    """The D3 series: how the solve went, and where the error is left."""
    per_camera, residual_xy = per_camera_mean_reprojection(
        optimisation, handler, log, "D3.12")

    missing_before = np.array(
        getattr(handler, "missing_poses_before_outlier_rejection", []), dtype=bool)
    missing_after = np.array(
        getattr(handler, "missing_poses_after_outlier_rejection",
                handler.missing_poses if handler.missing_poses is not None else []),
        dtype=bool)
    per_image_initial = np.array(
        getattr(handler, "initial_per_im_error", []), dtype=float)

    param_count = int(stats.get("param_count", 0))
    observation_count = int(
        stats.get("observation_count", len(optimisation.fun) // 2))

    n_missing_before = int(np.sum(missing_before))
    n_missing_after = int(np.sum(missing_after))

    return {
        "D3.1_n_missing_poses": n_missing_before,
        "D3.2_n_outlier_removed": int(max(0, n_missing_after - n_missing_before)),
        "D3.3_per_image_initial_reprojection": per_image_initial.tolist(),
        "D3.4_initial_error_plot": "rendered in diagnostics tab",
        "D3.5_initial_euclid_px": initial_euclid,
        "D3.6_final_euclid_px": final_euclid,
        "D3.7_error_reduction_ratio": (
            float(initial_euclid / final_euclid)
            if final_euclid > 0 else float("inf")),
        "D3.8_solver_status": {
            "status": int(stats.get("status", optimisation.status)),
            "message": str(stats.get("message", optimisation.message)),
            "success": bool(stats.get("success", optimisation.success)),
        },
        "D3.9_nfev": int(stats.get("nfev", optimisation.nfev)),
        "D3.10_parameter_observation_ratio": {
            "param_count": param_count,
            "observation_count": observation_count,
            "ratio": float(param_count / max(observation_count, 1)),
        },
        "D3.11_residual_xy_scatter": residual_xy.tolist(),
        "D3.12_per_camera_mean_reprojection": per_camera,
        "D3.13_extrinsic_pose_view": "rendered in diagnostics tab",
    }
