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

from pyCamSet.utils.paths import long_path
from pyCamSet.utils.saving import RES_ORDER
from pyCamSet.cameras.lens_models import DEFAULT_LENS_MODEL
from pyCamSet.calibration_targets.core.preparation import (
    detect_kwargs_of as preparation_detect_kwargs,
)
from pyCamSet.workflow.detections import (
    DetectionFilter,
    extract_detection_and_cam_res,
    save_detections,
    selected_camera_folders,
    staged_camera_root,
)
from pyCamSet.workflow.logs import LogFn, captured_output, discard
from pyCamSet.workflow.education_trace import (
    USED_DETECTIONS_KEY,
    make_used_detections_record,
    write_trace,
)
from pyCamSet.workflow.phase1 import target_of_params
from pyCamSet.workflow.workspace import (
    WorkspaceManager,
    make_run_id,
    path_exists,
    resolve_artifact,
)

_LOG = logging.getLogger(__name__)

try:
    from pyCamSet.calibration.camera_calibrator import (
        detect_datapoints_in_imfile,
        run_initial_calibration,
    )
    from pyCamSet.utils.intrinsics_report import report_initial_calibration
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

    params = dict(params)
    params["record_phase2_per_view"] = bool(params.get("record_phase2_per_view", False))
    run_id = make_run_id()
    run_dir = workspace.run_dir("phase2", run_id)
    detections_path = _detections_path(
        phase1_run, override_pickle, ws_path, log)

    diagnostics: dict = {}
    report: Optional[dict] = None
    error: Optional[str] = None
    camset_path: Optional[Path] = None
    pruned_path: Optional[Path] = None
    used_record_holder: dict = {}
    calibration_notes: list[str] = []

    def phase2_log(line: str) -> None:
        """Preserve actionable Zhang fallback warnings for downstream notes."""
        if "fitting the focal lengths alone" in line:
            note = ("Zhang closed-form principal point was outside the sensor; "
                    "focals_at_principal held it at image centre before refinement.")
            if note not in calibration_notes:
                calibration_notes.append(note)
        log(line)

    with captured_output(phase2_log):
        try:
            camset_path, diagnostics, pruned_path, report = _calibrate(
                params, run_dir, detections_path, prune, phase2_log,
                used_record_holder=used_record_holder)
        except Exception as exc:
            error = str(exc)
            phase2_log(f"ERROR: {error}")

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
    if used_record_holder.get("record") is not None:
        used_record = used_record_holder["record"]
        if (used_record.get("source") == "phase1_pickle"
                and override_pickle is not None and path_exists(override_pickle)
                and detections_path == override_pickle):
            used_record = dict(used_record, source="override_pickle")
    else:
        # If calibration failed before selecting inputs, do not infer which
        # detections it would have used from the requested settings.
        used_record = make_used_detections_record(
            source="legacy_unknown", path=None)
    artifacts[USED_DETECTIONS_KEY] = used_record

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
    if calibration_notes:
        metadata["calibration_notes"] = calibration_notes
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
              min_detections_per_board: int = 12,
              model: str = DEFAULT_LENS_MODEL,
              return_details: bool = False):
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
    :param model: the lens model to fit, a name from
        :data:`pyCamSet.cameras.lens_models.LENS_MODELS`
    :return: the calibrated camera set
    """
    if not BACKEND_OK:
        raise RuntimeError("pyCamSet calibration modules are not importable.")

    fixed = fixed_params or {}
    # Some target routes return a bare Camera when extrinsics or both
    # intrinsic blocks are fixed. Avoid requesting unavailable tuple details.
    details_supported = not any(
        "ext" in blocks or {"int", "dst"}.issubset(blocks)
        for blocks in fixed.values() if isinstance(blocks, dict))
    capture_details = bool(return_details and details_supported)
    result = run_initial_calibration(
        detection=detections,
        calibration_target=target,
        cam_res=cam_res,
        save=False,
        fixed_params=fixed_params,
        return_poses_and_costs=capture_details,
        min_detections_per_board=int(min_detections_per_board),
        model=model,
    )
    if return_details:
        return result if capture_details else (result, None, None)
    return result



def _write_per_view_trace(run_dir: Path, camera_records: list) -> None:
    """Write the opt-in per-view trace; one that cannot be written is logged, not raised."""
    try:
        write_trace(
            run_dir, run_dir.name, "phase2", "phase2_per_view",
            {"cost_semantics": "calibrate_zhang/calibrate_telecentric per-view RMS (px)",
             "view_order": "usable-view order returned by the selected seed calibration",
             "cameras": camera_records}, enabled=True)
    except (OSError, ValueError) as exc:
        _LOG.warning("Phase 2 per-view trace not written: %s", exc)


def _calibrate(params: dict, run_dir: Path, detections_path: Optional[Path],
               prune: Optional[DetectionFilter],
               log: LogFn, *, used_record_holder: Optional[dict] = None
               ) -> tuple[Path, dict, Optional[Path], dict]:
    """Run the initial calibration and compute its diagnostics."""
    if not BACKEND_OK:
        raise RuntimeError("pyCamSet calibration modules are not importable.")

    target = target_of_params(params)

    if prune is not None:
        if params.get("high_distortion"):
            raise RuntimeError(
                "Cannot prune a high-distortion Phase 2 rerun: the authoritative "
                "second-pass detections were not persisted. Re-run without pruning.")
        # Nothing re-detects here.  Detecting again would put back exactly the
        # observations the prune was asked to leave out, so a pruned run reads
        # the saved detections and nothing else -- which also means it needs
        # no images, and no staging folder to point at them.
        result = _calibrate_pruned(
            params, run_dir, detections_path, prune, target, log)
        if used_record_holder is not None:
            used_record_holder["record"] = make_used_detections_record(
                source="filtered_pickle", path=str(result[2]))
        return result

    f_loc = Path(params["f_loc"])
    selected = list(params.get("selected_cameras") or [])

    cam_folders = selected_camera_folders(f_loc, selected)
    if selected and len(cam_folders) < 2:
        raise RuntimeError(
            "Need at least two selected camera folders for Phase 2.")
    # This run's own selection, fixed here before any staging decision or
    # later race window -- passed through explicitly to every
    # detect_datapoints_in_imfile call below so it never re-derives the
    # camera list from a fresh scan of root/f_loc (round-2 review, P1, the
    # same gap phase1.py's own _detect was hardened against). Without this,
    # when selected_cameras is empty, staged_camera_root hands back root ==
    # f_loc (the real, shared image folder, not a staged copy), so a camera
    # folder that appears under f_loc after cam_folders was selected here
    # would otherwise be silently folded into this pass's own detections.
    cam_names = [folder.name for folder in cam_folders]

    pass_info: dict = {}
    with staged_camera_root(f_loc, cam_folders, log, "pycamset_phase2_") as root:
        detections, cam_res = _load_or_detect(
            params, target, root, detections_path, set(selected), cam_names, log,
            pass_info=pass_info)

        per_view_result = calibrate(
            detections, cam_res, target,
            fixed_params=params["fixed_params"],
            min_detections_per_board=params.get("min_detections_per_board", 12),
            model=params.get("lens_model", DEFAULT_LENS_MODEL),
            return_details=params.get("record_phase2_per_view", False))
        if params.get("record_phase2_per_view", False):
            cams, per_view_poses, per_view_costs = per_view_result
        else:
            cams = per_view_result
            per_view_poses = per_view_costs = None

        if params["high_distortion"]:
            log("2c  High-distortion mode: re-running detection using first-pass "
                "camera estimates...")
            detections, cam_res = detect_datapoints_in_imfile(
                f_loc=root,
                calibration_target=target,
                caching=False,
                draw=False,
                n_lim=params["n_lim"],
                camset=cams,
                cam_names=cam_names,
                # Phase 1's own rule: per-target defaults, explicit values win,
                # legacy params keep the PuzzleBoard-only gate.
                **preparation_detect_kwargs(params, target),
            )
            per_view_result = calibrate(
                detections, cam_res, target,
                fixed_params=params["fixed_params"],
                min_detections_per_board=params.get(
                    "min_detections_per_board", 12),
                model=params.get("lens_model", DEFAULT_LENS_MODEL),
                return_details=params.get("record_phase2_per_view", False))
            if params.get("record_phase2_per_view", False):
                cams, per_view_poses, per_view_costs = per_view_result
            else:
                cams = per_view_result
                per_view_poses = per_view_costs = None

    if used_record_holder is not None:
        if params.get("high_distortion"):
            record = make_used_detections_record(
                source="fresh_pass_not_saved", path=None,
                fresh_pass_reasons=pass_info.get("reasons", []) +
                ["high_distortion_second_pass"], cache_hit=False)
            record["passes"] = [
                {"pass": "initial", "cache_hit": (
                    pass_info.get("cache_hit") if pass_info.get("fresh_pass") else None)},
                {"pass": "high_distortion_second_pass", "cache_hit": False},
            ]
            used_record_holder["record"] = record
        elif pass_info.get("fresh_pass"):
            record = make_used_detections_record(
                source="fresh_pass_not_saved", path=None,
                fresh_pass_reasons=pass_info["reasons"],
                cache_hit=pass_info.get("cache_hit"))
            record["passes"] = [{"pass": "initial", "cache_hit":
                                 pass_info.get("cache_hit")}]
            used_record_holder["record"] = record
        else:
            used_record_holder["record"] = make_used_detections_record(
                source="phase1_pickle", path=str(detections_path))

    if params.get("record_phase2_per_view") and per_view_poses is not None:
        camera_records = []
        for camera_name, poses, costs in zip(
                detections.cam_names, per_view_poses, per_view_costs):
            camera_records.append({
                "camera": str(camera_name),
                "views": [
                    {"view_index": index, "pose_target_to_camera": np.asarray(pose).tolist(),
                     "per_view_rms_px": float(cost)}
                    for index, (pose, cost) in enumerate(zip(poses, costs))
                ],
            })
        _write_per_view_trace(run_dir, camera_records)

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
        load_pickle(long_path(detections_path)))

    filtered = prune.apply(detections, log)
    pruned_path = save_detections(
        run_dir / "filtered_detected_datapoints.pickle", filtered, cam_res)
    log(f"Saved filtered detections: {pruned_path}")

    log("Running Phase 2 initial calibration on filtered detections…")
    per_view_result = calibrate(
        filtered, cam_res, target,
        fixed_params=params.get("fixed_params"),
        min_detections_per_board=params.get("min_detections_per_board", 12),
        model=params.get("lens_model", DEFAULT_LENS_MODEL),
        return_details=params.get("record_phase2_per_view", False))
    if params.get("record_phase2_per_view", False):
        cams, poses, costs = per_view_result
    else:
        cams = per_view_result
        poses = costs = None

    camset_path = run_dir / "initial_cameras.camset"
    cams.save(camset_path)

    if poses is not None:
        camera_records = []
        for camera_name, camera_poses, camera_costs in zip(
                filtered.cam_names, poses, costs):
            camera_records.append({
                "camera": str(camera_name),
                "views": [
                    {"view_index": index,
                     "pose_target_to_camera": np.asarray(pose).tolist(),
                     "per_view_rms_px": float(cost)}
                    for index, (pose, cost) in enumerate(
                        zip(camera_poses, camera_costs))
                ],
            })
        _write_per_view_trace(run_dir, camera_records)

    diagnostics, report = diagnostics_of(
        filtered, target, cams,
        params.get("min_detections_per_board", 12))
    return camset_path, diagnostics, pruned_path, report


def _load_or_detect(params: dict, target, root: Path,
                    detections_path: Optional[Path],
                    selected: set[str], cam_names: list[str], log: LogFn,
                    *, pass_info: Optional[dict] = None):
    """Read phase 1's detections, or detect again if they cannot be used."""
    detections = cam_res = None
    reasons: list[str] = []
    if detections_path is None or not path_exists(detections_path):
        reasons.append("pickle_missing")

    if detections_path is not None and path_exists(detections_path):
        log(f"Using detections: {detections_path}")
        detections, cam_res = extract_detection_and_cam_res(
            load_pickle(long_path(detections_path)))
        if cam_res is None:
            reasons.append("no_cam_res")
        if selected and set(getattr(detections, "cam_names", []) or []) != selected:
            log("Detection artifact camera set does not match selected "
                "cameras; running fresh detection on selected subset.")
            detections = cam_res = None
            reasons.append("camera_subset")


    if detections is None or cam_res is None:
        log("Detection artifact missing/incompatible, falling back to "
            "detection pass.")
        side_channel: dict = {}
        detections, cam_res = detect_datapoints_in_imfile(
            f_loc=root,
            calibration_target=target,
            caching=params["caching"],
            draw=False,
            n_lim=params["n_lim"],
            # This pass's own selection, pinned by the caller -- see
            # _calibrate's own comment for why this must not be left to a
            # fresh scan of root/f_loc.
            cam_names=cam_names,
            **preparation_detect_kwargs(params, target),
            side_channel=side_channel,
        )
        if pass_info is not None:
            pass_info.update({"fresh_pass": True,
                              "reasons": sorted(set(reasons), key=(
                                  "camera_subset", "no_cam_res", "pickle_missing").index),
                              "cache_hit": side_channel.get("cache_hit")})
    elif pass_info is not None:
        pass_info.update({"fresh_pass": False, "reasons": [], "cache_hit": None})
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
            # records written before this key hold (height, width)
            "res_order": RES_ORDER,
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
