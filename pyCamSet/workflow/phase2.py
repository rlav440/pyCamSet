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
    DetectionFilter,
    extract_detection_and_cam_res,
    save_detections,
    selected_camera_folders,
    staged_camera_root,
)
from pyCamSet.workflow.logs import LogFn, captured_output, discard
from pyCamSet.workflow.phase1 import target_of_params
from pyCamSet.workflow.workspace import (
    WorkspaceManager,
    as_io_path,
    make_run_id,
    path_exists,
    resolve_artifact,
)

_LOG = logging.getLogger(__name__)

try:
    from pyCamSet.calibration.camera_calibrator import (
        detect_datapoints_in_imfile,
        report_initial_calibration,
        run_initial_calibration,
    )
    from pyCamSet.utils.saving import load_pickle

    BACKEND_OK = True
except ImportError as exc:
    detect_datapoints_in_imfile = None
    report_initial_calibration = None
    run_initial_calibration = None
    load_pickle = None
    BACKEND_OK = False
    _LOG.warning("Phase 2 calibration backend unavailable: %s", exc)


def run(params: dict,
        workspace: WorkspaceManager,
        log: LogFn = discard,
        phase1_run: Optional[dict] = None,
        override_pickle: Optional[Path] = None,
        prune: Optional[DetectionFilter] = None,
        source_run: Optional[dict] = None) -> dict:
    """
    Calibrate each camera's intrinsics from phase 1's detections.

    :param params: the phase's settings
    :param workspace: the workspace to read the detections from and save to
    :param log: what to call with each line of output
    :param phase1_run: the run supplying the detections, if there is one
    :param override_pickle: a detections file to read instead of the run's
    :param prune: observations to leave out; see :func:`rerun`
    :param source_run: the phase 2 run this one is a re-run of
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
    report: Optional[dict] = None
    error: Optional[str] = None
    camset_path: Optional[Path] = None
    pruned_path: Optional[Path] = None

    with captured_output(log):
        try:
            camset_path, diagnostics, pruned_path, report = _calibrate(
                params, run_dir, detections_path, prune, log)
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
    if pruned_path is not None:
        artifacts["filtered_detection_pickle"] = str(pruned_path)
        artifacts["detection_source_override"] = str(pruned_path)

    metadata = {
        "run_id": run_id,
        "phase": "phase2",
        "params": params,
        "diagnostics": diagnostics,
        "report": report,
        "error": error,
        "inputs": {
            "phase1_run_id": phase1_run.get("run_id") if phase1_run else None,
            "phase2_run_id": source_run.get("run_id") if source_run else None,
        },
        "artifacts": artifacts,
    }
    if prune is not None and prune.record:
        metadata["threshold_pruning"] = prune.record

    workspace.save_run("phase2", run_id, metadata)
    if error is None:
        log(f"Run saved: {run_id}")
    return metadata


def rerun(source_run: dict,
          workspace: WorkspaceManager,
          log: LogFn = discard,
          prune: Optional[DetectionFilter] = None) -> dict:
    """
    Run phase 2 again on *source_run*'s settings, minus what *prune* drops.

    The settings and the detections both come from the run being repeated, so
    the new run differs from it only by what was left out.  The source is not
    touched.

    :param source_run: the phase 2 run to repeat
    :param workspace: the workspace holding it
    :param log: what to call with each line of output
    :param prune: observations to leave out
    :return: the new run's metadata record
    """
    return run(
        dict(source_run.get("params") or {}),
        workspace,
        log,
        phase1_run=workspace.linked_run("phase1", source_run),
        prune=prune,
        source_run=source_run,
    )


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
        return resolve_artifact(phase1_run, "phase1", ws_path)
    return None


def calibrate(detections, cam_res, target, *,
              fixed_params: Optional[dict] = None,
              min_detections_per_board: int = 12):
    """
    Calibrate each camera on its own, from detections already in hand.

    The solve, with nothing around it: no workspace, no image folder, no run
    record.  The phase runner below wraps it in those; a parameter search
    calls it directly, per trial.

    :param detections: the target detections to calibrate from
    :param cam_res: each camera's resolution
    :param target: the calibration target the detections were made against
    :param fixed_params: parameters to pin rather than solve for
    :param min_detections_per_board: how much of a board a view must show
    :return: the calibrated camera set
    """
    if not BACKEND_OK:
        raise RuntimeError("pyCamSet calibration modules are not importable.")

    cams, _, _ = run_initial_calibration(
        detection=detections,
        calibration_target=target,
        cam_res=cam_res,
        save=False,
        fixed_params=fixed_params,
        return_poses_and_costs=True,
        min_detections_per_board=int(min_detections_per_board),
    )
    return cams


def _calibrate(params: dict, run_dir: Path, detections_path: Optional[Path],
               prune: Optional[DetectionFilter],
               log: LogFn) -> tuple[Path, dict, Optional[Path], dict]:
    """Run the initial calibration and compute its diagnostics."""
    if not BACKEND_OK:
        raise RuntimeError("pyCamSet calibration modules are not importable.")

    target = target_of_params(params)

    if prune is not None:
        # Nothing re-detects here.  Detecting again would put back exactly the
        # observations the prune was asked to leave out, so a pruned run reads
        # the saved detections and nothing else -- which also means it needs
        # no images, and no staging folder to point at them.
        return _calibrate_pruned(
            params, run_dir, detections_path, prune, target, log)

    f_loc = Path(params["f_loc"])
    selected = list(params.get("selected_cameras") or [])

    cam_folders = selected_camera_folders(f_loc, selected)
    if selected and len(cam_folders) < 2:
        raise RuntimeError(
            "Need at least two selected camera folders for Phase 2.")

    with staged_camera_root(f_loc, cam_folders, log, "pycamset_phase2_") as root:
        detections, cam_res = _load_or_detect(
            params, target, root, detections_path, set(selected), log)

        cams = calibrate(
            detections, cam_res, target,
            fixed_params=params["fixed_params"],
            min_detections_per_board=params.get("min_detections_per_board", 12))

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
            cams = calibrate(
                detections, cam_res, target,
                fixed_params=params["fixed_params"],
                min_detections_per_board=params.get(
                    "min_detections_per_board", 12))

    camset_path = run_dir / (
        "initial_cameras_high_distortion.camset"
        if params["high_distortion"]
        else "initial_cameras.camset"
    )
    cams.save(camset_path)

    diagnostics, report = diagnostics_of(
        detections, target, cams,
        params.get("min_detections_per_board", 12))
    return camset_path, diagnostics, None, report


def _calibrate_pruned(params: dict, run_dir: Path,
                      detections_path: Optional[Path],
                      prune: DetectionFilter, target,
                      log: LogFn) -> tuple[Path, dict, Path, dict]:
    """Calibrate from saved detections with some observations left out."""
    if detections_path is None or not path_exists(detections_path):
        raise RuntimeError(
            "Could not resolve Phase 1 detected_datapoints.pickle for the "
            "source run.")

    log(f"Loading Phase 1 detections: {detections_path}")
    detections, cam_res = extract_detection_and_cam_res(
        load_pickle(as_io_path(detections_path)))

    filtered = prune.apply(detections, log)
    pruned_path = save_detections(
        run_dir / "filtered_detected_datapoints.pickle", filtered, cam_res)
    log(f"Saved filtered detections: {pruned_path}")

    log("Running Phase 2 initial calibration on filtered detections…")
    cams = calibrate(
        filtered, cam_res, target,
        fixed_params=params.get("fixed_params"),
        min_detections_per_board=params.get("min_detections_per_board", 12))

    camset_path = run_dir / "initial_cameras.camset"
    cams.save(camset_path)

    diagnostics, report = diagnostics_of(
        filtered, target, cams,
        params.get("min_detections_per_board", 12))
    return camset_path, diagnostics, pruned_path, report


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


def diagnostics_of(detections, target, cams,
                   min_detections_per_board: int = 12) -> tuple[dict, dict]:
    """The D2 series: what each camera's own calibration came out as.

    The numbers are the initial intrinsics report's own, so the block printed
    here and the diagnostics recorded beside it cannot disagree.

    Public because the diagnostics tab builds a phase 2 run of its own, from
    detections it has pruned, and reports the same numbers about it.

    :return: the diagnostics, and the report they were read off
    """
    report = report_initial_calibration(
        cams, detections, target, min_detections_per_board)

    intrinsics: dict[str, dict] = {}
    distortion: dict[str, dict] = {}
    for cam in report.per_camera:
        intrinsics[cam.name] = {
            "fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy,
            "res": np.array(cams[cam.name].res).astype(float).tolist(),
        }
        distortion[cam.name] = {
            "coeffs": np.array(
                cams[cam.name].distortion_coefs).reshape(-1).astype(float).tolist(),
            "l2_norm": cam.distortion_l2,
        }

    diagnostics = {
        "D2.1_per_camera_rms_reprojection": {
            cam.name: cam.rms_px for cam in report.per_camera},
        "D2.2_intrinsics": intrinsics,
        "D2.3_distortion": distortion,
        "D2.5_intrinsic_stddev": "not available in current pyCamSet API",
        "D2.6_per_view_reprojection": report.per_view,
        "D2.7_per_view_error_plot": (
            "rendered in diagnostics tab (true per-image reprojection RMS)"),
    }
    return diagnostics, report.to_dict()
