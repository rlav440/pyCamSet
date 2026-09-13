"""
Phase 2: an initial intrinsic calibration of each camera on its own.

Reads phase 1's detections, calibrates every camera independently, and saves
the camset that phase 3's bundle adjustment starts from.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from pyCamSet.workflow.detections import (
    extract_detection_and_cam_res,
    selected_camera_folders,
    staged_camera_root,
)
from pyCamSet.workflow.diagnostics import per_view_reprojection
from pyCamSet.workflow.logs import LogFn, captured_output, discard
from pyCamSet.workflow.phase1 import target_from_params
from pyCamSet.workflow.workspace import (
    WorkspaceManager,
    as_io_path,
    make_run_id,
    path_exists,
    resolve_phase1_pickle_artifact,
)

_LOG = logging.getLogger(__name__)

try:
    from pyCamSet.calibration.camera_calibrator import (
        detect_datapoints_in_imfile,
        run_initial_calibration,
    )
    from pyCamSet.utils.saving import load_pickle

    BACKEND_OK = True
except ImportError as exc:
    detect_datapoints_in_imfile = None
    run_initial_calibration = None
    load_pickle = None
    BACKEND_OK = False
    _LOG.warning("Phase 2 calibration backend unavailable: %s", exc)


def run(params: dict,
        workspace: WorkspaceManager,
        log: LogFn = discard,
        phase1_run: Optional[dict] = None,
        override_pickle: Optional[Path] = None) -> dict:
    """
    Calibrate each camera's intrinsics from phase 1's detections.

    :param params: the phase's settings
    :param workspace: the workspace to read the detections from and save to
    :param log: what to call with each line of output
    :param phase1_run: the run supplying the detections, if there is one
    :param override_pickle: a detections file to read instead of the run's
    :return: the run's metadata record, saved to the workspace
    """
    ws_path = workspace.workspace_path
    if ws_path is None:
        raise RuntimeError("Workspace path is not set.")

    run_id = make_run_id()
    run_dir = workspace.run_dir("phase2", run_id)
    detections_path = _detections_path(
        phase1_run, override_pickle, ws_path, log)

    diagnostics: dict = {}
    error: Optional[str] = None
    camset_path: Optional[Path] = None

    with captured_output(log):
        try:
            camset_path, diagnostics = _calibrate(
                params, run_dir, detections_path, log)
        except Exception as exc:
            error = str(exc)
            log(f"ERROR: {error}")

    artifacts: dict = {
        "phase1_detection_pickle": (
            str(detections_path) if detections_path is not None else None),
        "detection_source_override": (
            str(override_pickle) if override_pickle is not None else None),
    }
    if camset_path is not None:
        artifacts["initial_camset"] = str(camset_path)

    metadata = {
        "run_id": run_id,
        "phase": "phase2",
        "params": params,
        "diagnostics": diagnostics,
        "error": error,
        "inputs": {
            "phase1_run_id": phase1_run.get("run_id") if phase1_run else None,
        },
        "artifacts": artifacts,
    }
    workspace.save_run("phase2", run_id, metadata)
    if error is None:
        log(f"Run saved: {run_id}")
    return metadata


def _detections_path(phase1_run: Optional[dict],
                     override_pickle: Optional[Path],
                     ws_path: Path,
                     log: LogFn) -> Optional[Path]:
    """Where to read the detections from, preferring an explicit override."""
    if override_pickle is not None:
        if path_exists(override_pickle):
            log(f"Using override detections: {override_pickle}")
            return override_pickle
        log(f"Override path missing: {override_pickle} "
            f"(falling back to Phase 1 source)")
    if phase1_run is not None:
        return resolve_phase1_pickle_artifact(phase1_run, ws_path)
    return None


def _calibrate(params: dict, run_dir: Path, detections_path: Optional[Path],
               log: LogFn) -> tuple[Path, dict]:
    """Run the initial calibration and compute its diagnostics."""
    if not BACKEND_OK:
        raise RuntimeError("pyCamSet calibration modules are not importable.")

    target = target_from_params(params)
    f_loc = Path(params["f_loc"])
    selected = list(params.get("selected_cameras") or [])

    cam_folders = selected_camera_folders(f_loc, selected)
    if selected and len(cam_folders) < 2:
        raise RuntimeError(
            "Need at least two selected camera folders for Phase 2.")

    with staged_camera_root(f_loc, cam_folders, log, "pycamset_phase2_") as root:
        detections, cam_res = _load_or_detect(
            params, target, root, detections_path, set(selected), log)

        cams, _, _ = run_initial_calibration(
            detection=detections,
            calibration_target=target,
            cam_res=cam_res,
            save=False,
            fixed_params=params["fixed_params"],
            return_poses_and_costs=True,
            min_detections_per_board=int(
                params.get("min_detections_per_board", 12)),
        )
        log("2b  Initial calibration completed.")

        if params["high_distortion"]:
            log("2c  High-distortion mode: re-running detection with initial "
                "intrinsics...")
            detections, _ = detect_datapoints_in_imfile(
                f_loc=root,
                calibration_target=target,
                caching=False,
                draw=False,
                n_lim=params["n_lim"],
                camset=cams,
            )
            cams, _, _ = run_initial_calibration(
                detection=detections,
                calibration_target=target,
                cam_res=cam_res,
                save=False,
                fixed_params=params["fixed_params"],
                return_poses_and_costs=True,
                min_detections_per_board=int(
                    params.get("min_detections_per_board", 12)),
            )
            log("2c  High-distortion refinement completed.")

    camset_path = run_dir / (
        "initial_cameras_high_distortion.camset"
        if params["high_distortion"]
        else "initial_cameras.camset"
    )
    cams.save(camset_path)

    diagnostics = diagnostics_of(detections, target, cams)
    log("Diagnostics computed (D2.1-D2.7).")
    return camset_path, diagnostics


def _load_or_detect(params: dict, target, root: Path,
                    detections_path: Optional[Path],
                    selected: set[str], log: LogFn):
    """Read phase 1's detections, or detect again if they cannot be used."""
    detections = cam_res = None

    if detections_path is not None and path_exists(detections_path):
        log(f"Using detections: {detections_path}")
        detections, cam_res = extract_detection_and_cam_res(
            load_pickle(as_io_path(detections_path)))
        if selected and set(getattr(detections, "cam_names", []) or []) != selected:
            log("Detection artifact camera set does not match selected "
                "cameras; running fresh detection on selected subset.")
            detections = cam_res = None

    if detections is None or cam_res is None:
        log("Detection artifact missing/incompatible, falling back to "
            "detection pass.")
        detections, cam_res = detect_datapoints_in_imfile(
            f_loc=root,
            calibration_target=target,
            caching=params["caching"],
            draw=False,
            n_lim=params["n_lim"],
        )
    return detections, cam_res


def diagnostics_of(detections, target, cams) -> dict:
    """The D2 series: what each camera's own calibration came out as.

    Public because the diagnostics tab builds a phase 2 run of its own, from
    detections it has pruned, and reports the same numbers about it.
    """
    per_view, per_camera_rms = per_view_reprojection(detections, target, cams)

    intrinsics: dict[str, dict] = {}
    distortion: dict[str, dict] = {}
    for cam_name, cam in zip(cams.get_names(), cams):
        matrix = np.array(cam.intrinsic)
        intrinsics[cam_name] = {
            "fx": float(matrix[0, 0]),
            "fy": float(matrix[1, 1]),
            "cx": float(matrix[0, 2]),
            "cy": float(matrix[1, 2]),
            "res": np.array(cam.res).astype(float).tolist(),
        }
        coefficients = np.array(cam.distortion_coefs).reshape(-1)
        distortion[cam_name] = {
            "coeffs": coefficients.astype(float).tolist(),
            "l2_norm": float(np.linalg.norm(coefficients)),
        }

    return {
        "D2.1_per_camera_rms_reprojection": per_camera_rms,
        "D2.2_intrinsics": intrinsics,
        "D2.3_distortion": distortion,
        "D2.5_intrinsic_stddev": "not available in current pyCamSet API",
        "D2.6_per_view_reprojection": per_view,
        "D2.7_per_view_error_plot": (
            "rendered in diagnostics tab (true per-image reprojection RMS)"),
    }
