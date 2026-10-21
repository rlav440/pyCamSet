"""
Phase 3: bundle adjustment over the whole camera set at once.

Starts from phase 2's per-camera intrinsics and phase 1's detections, and
solves for the cameras and the target poses together -- optionally holding the
cameras near a known rig geometry through the lockbox priors.
"""
from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from pyCamSet.utils.paths import long_path
from pyCamSet.workflow.detections import (
    DetectionFilter,
    extract_detection,
    save_detections,
)
from pyCamSet.workflow.diagnostics import per_camera_mean_reprojection
from pyCamSet.workflow.logs import (
    LogFn,
    captured_output,
    discard,
    non_interactive_plotting,
)
from pyCamSet.workflow.phase1 import target_of_params
from pyCamSet.workflow.workspace import (
    WorkspaceManager,
    make_run_id,
    path_exists,
    resolve_artifact,
)
from pyCamSet.workflow.education_trace import atomic_write_bytes, write_trace

_LOG = logging.getLogger(__name__)
MAX_INITIAL_STATE_BYTES = 8 * 1024 * 1024


def _write_initial_state(run_dir: Path, calibration_params: np.ndarray,
                         missing_poses: np.ndarray,
                         initial_error_px: float) -> Path | None:
    """Atomically preserve a bounded snapshot of the Phase 3 solver start.

    Written only for a run that records for education mode.  A snapshot over
    :data:`MAX_INITIAL_STATE_BYTES` is not written (None) rather than failing
    the run: the explanation then says the starting vector was not saved.
    """
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        calibration_params=np.asarray(calibration_params, dtype=np.float64),
        missing_poses=np.asarray(missing_poses, dtype=np.bool_),
        initial_error_px=np.asarray(initial_error_px, dtype=np.float64),
    )
    payload = buffer.getvalue()
    if len(payload) > MAX_INITIAL_STATE_BYTES:
        _LOG.info("Phase 3 starting vector (%d bytes) is over the %d byte bound; not saved",
                  len(payload), MAX_INITIAL_STATE_BYTES)
        return None
    destination = run_dir / "phase3_initial_state.npz"
    atomic_write_bytes(destination, payload)
    return destination

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
        camset_override: Optional[Path] = None,
        prune: Optional[DetectionFilter] = None,
        source_run: Optional[dict] = None) -> dict:
    """
    Solve the whole camera set against the target together.

    :param params: the phase's settings, including its ``lockbox`` block
    :param workspace: the workspace to read inputs from and save to
    :param log: what to call with each line of output
    :param phase2_run: the run supplying the initial camset
    :param phase1_run: the run supplying the detections
    :param camset_override: an initial camset to use instead of the run's
    :param prune: observations to leave out; see :func:`rerun`
    :param source_run: the phase 3 run this one is a re-run of
    :return: the run's metadata record, saved to the workspace
    """
    ws_path = workspace.workspace_path
    if ws_path is None:
        raise RuntimeError("Workspace path is not set.")

    run_id = make_run_id()
    run_dir = workspace.run_dir("phase3", run_id)
    params = dict(params)
    params["record_education"] = bool(params.get("record_education", False))
    # On unless switched off, in scripts and the GUI alike (on by default).
    params["recover_single_face_telecentric"] = bool(params.get(
        "recover_single_face_telecentric",
        (params.get("problem_options") or {}).get("recover_single_face_telecentric", True)))

    diagnostics: dict = {}
    report: Optional[dict] = None
    error: Optional[str] = None
    camset_in: Optional[Path] = None
    detections_path: Optional[Path] = None
    camset_out: Optional[Path] = None
    pruned_path: Optional[Path] = None

    with captured_output(log), non_interactive_plotting():
        log("Phase 3 running in non-interactive plotting mode (thread-safe).")
        try:
            camset_in = _initial_camset(phase2_run, camset_override, ws_path)
            detections_path = _detections_path(phase1_run, ws_path)
            camset_out, diagnostics, pruned_path, report = _solve(
                params, run_dir, camset_in, detections_path, prune, log)
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
    if pruned_path is not None:
        artifacts["filtered_detection_pickle"] = str(pruned_path)
    if diagnostics.get("initial_state_artifact"):
        artifacts["initial_state"] = diagnostics["initial_state_artifact"]
    if diagnostics.get("initial_state_sha256"):
        artifacts["initial_state_sha256"] = diagnostics["initial_state_sha256"]
    if diagnostics.get("lockbox_source_sha256"):
        artifacts["lockbox_source_sha256"] = diagnostics["lockbox_source_sha256"]
    if diagnostics.get("education_trace"):
        artifacts["education_trace"] = diagnostics["education_trace"]

    metadata = {
        "run_id": run_id,
        "phase": "phase3",
        # A saved camset is useful for diagnosis even when the optimiser did
        # not finish.  Keep that distinct from a scientifically usable result
        # so the GUI cannot offer an unconverged run as a clean hand-off.
        "status": (
            "failed" if error else
            (diagnostics.get("quality_gate", {}).get("status", "incomplete")
             if diagnostics else "incomplete")
        ),
        "params": params,
        "diagnostics": diagnostics,
        "report": report,
        "error": error,
        "inputs": {
            "phase2_run_id": phase2_run.get("run_id") if phase2_run else None,
            "phase1_run_id": phase1_run.get("run_id") if phase1_run else None,
            "phase3_run_id": source_run.get("run_id") if source_run else None,
        },
        "artifacts": artifacts,
    }
    if prune is not None and prune.record:
        metadata["threshold_pruning"] = prune.record

    workspace.save_run("phase3", run_id, metadata)
    if error is None:
        log(f"Run saved: {run_id}")
    return metadata


def rerun(source_run: dict,
          workspace: WorkspaceManager,
          log: LogFn = discard,
          prune: Optional[DetectionFilter] = None) -> dict:
    """
    Solve *source_run* again, minus whatever *prune* leaves out.

    The settings, the initial camset and the detections all come from the run
    being repeated, so the new run differs from it only by what was dropped.
    The source is not touched.

    :param source_run: the phase 3 run to repeat
    :param workspace: the workspace holding it
    :param log: what to call with each line of output
    :param prune: observations to leave out
    :return: the new run's metadata record
    """
    return run(
        dict(source_run.get("params") or {}),
        workspace,
        log,
        phase2_run=workspace.linked_run("phase2", source_run),
        phase1_run=workspace.linked_run("phase1", source_run),
        prune=prune,
        source_run=source_run,
    )


def _initial_camset(phase2_run: Optional[dict],
                    camset_override: Optional[Path],
                    ws_path: Path) -> Path:
    """The camset the solve starts from."""
    if camset_override is not None and path_exists(camset_override):
        return Path(camset_override)
    resolved = (resolve_artifact(phase2_run, "phase2", ws_path)
                if phase2_run else None)
    if resolved is None:
        raise RuntimeError("Phase 2 run is missing initial_camset artifact.")
    if not path_exists(resolved):
        raise RuntimeError(f"Phase 2 camset path does not exist: {resolved}")
    return resolved


def _detections_path(phase1_run: Optional[dict], ws_path: Path) -> Path:
    """The detections the solve reads."""
    resolved = (resolve_artifact(phase1_run, "phase1", ws_path)
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

    source_camset = load_CameraSet(long_path(source))
    if set(source_camset.get_names()) != set(cams.get_names()):
        raise RuntimeError(
            "Effective lockbox source camera names do not match the active "
            "Phase 2 camset. This would break TemplateBundleHandler.")
    return config, source_camset, settings


def solve(cams, detections, target, *,
          fixed_params: Optional[dict] = None,
          options: Optional[dict] = None,
          threads: int = 1,
          lockbox_config=None,
          lockbox_source_camset=None,
          lockbox_warm_start: bool = True,
          record_education: bool = False):
    """
    Bundle-adjust a camera set against a target, from objects in hand.

    The solve, with nothing around it: no workspace, no artifact paths, no
    run record.  The phase runner below wraps it in those; a parameter
    search calls it directly, once per trial.

    :param cams: the camera set to start from
    :param detections: the detections to solve against
    :param target: the calibration target they were made against
    :param fixed_params: parameters to pin rather than solve for
    :param options: the solver options, as the handler reads them
    :param threads: Jacobian evaluation threads
    :param lockbox_config: the rig priors, or None for no priors
    :param lockbox_source_camset: the rig the priors hold the cameras near
    :param lockbox_warm_start: start from the lockbox rig rather than *cams*
    :return: the handler, the solver result, the solved cameras, the stats
    """
    if not BACKEND_OK:
        raise RuntimeError("pyCamSet optimisation modules are not importable.")

    handler = TemplateBundleHandler(
        camset=cams,
        target=target,
        detection=detections,
        fixed_params=fixed_params,
        options=options,
        lockbox_config=lockbox_config,
        lockbox_source_camset=lockbox_source_camset,
        lockbox_warm_start=bool(lockbox_warm_start),
        record_education=record_education,
    )
    optimisation, out_cams, stats = run_bundle_adjustment_with_stats(
        handler, threads=threads, record_education=record_education)
    return handler, optimisation, out_cams, stats


def _solve(params: dict, run_dir: Path, camset_in: Path,
           detections_path: Path, prune: Optional[DetectionFilter],
           log: LogFn) -> tuple[Path, dict, Optional[Path], Optional[dict]]:
    """Run the bundle adjustment and compute its diagnostics."""
    if not BACKEND_OK:
        raise RuntimeError("pyCamSet optimisation modules are not importable.")

    log(f"Loading Phase 2 camset: {camset_in}")
    cams = load_CameraSet(long_path(camset_in))
    log(f"Loading Phase 1 detections: {detections_path}")
    detections = extract_detection(load_pickle(long_path(detections_path)))
    if detections is None:
        raise RuntimeError(
            "Could not extract TargetDetection from Phase 1 pickle.")

    _check_camera_subset(
        list(params.get("selected_cameras") or []), cams, detections)

    pruned_path = None
    if prune is not None:
        detections = prune.apply(detections, log)
        pruned_path = save_detections(
            run_dir / "filtered_detected_datapoints.pickle", detections)
        log(f"Saved filtered detections: {pruned_path}")

    target = target_of_params(params)
    lockbox_config, lockbox_source, lockbox_settings = _lockbox(params, cams, log)

    # The setting rides in the handler's options, which the saved camset keeps, so a
    # rebuild of this run (Phase 4, the education replays) seeds the same way.
    options = dict(params["problem_options"] or {})
    options["recover_single_face_telecentric"] = bool(
        params.get("recover_single_face_telecentric", True))
    handler, optimisation, out_cams, stats = solve(
        cams, detections, target,
        fixed_params=params["fixed_params"],
        options=options,
        threads=params["threads"],
        lockbox_config=lockbox_config,
        lockbox_source_camset=lockbox_source,
        lockbox_warm_start=lockbox_settings.get("warm_start", True),
        record_education=bool(params.get("record_education", False)),
    )

    # The errors, the termination message and the timing are the calibration
    # summary's, printed by the solve itself.
    initial_euclid = float(stats.get("initial_euclid", float("nan")))
    final_euclid = float(stats.get("final_euclid", float("nan")))

    # When recording for education mode, preserve the actual solver input,
    # including the missing-pose mask at the same point, in a bounded
    # run-private binary artifact, and the lockbox source's digest.
    initial_state_path = None
    lockbox_source_sha256 = None
    lockbox_source_path = lockbox_settings.get("source_camset")
    if handler.record_education:
        initial_params = np.asarray(handler.get_initial_params(), dtype=np.float64).copy()
        missing_poses = np.asarray(handler.missing_poses, dtype=np.bool_).copy()
        initial_state_path = _write_initial_state(
            run_dir, initial_params, missing_poses, initial_euclid)
    if handler.record_education and lockbox_source_path:
        digest = hashlib.sha256()
        with open(long_path(lockbox_source_path), "rb") as source_stream:
            for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                digest.update(chunk)
        lockbox_source_sha256 = digest.hexdigest()

    camset_out = run_dir / "optimised_cameras.camset"
    out_cams.save(camset_out)

    diagnostics = _diagnostics(
        optimisation, handler, stats, initial_euclid, final_euclid, log)
    diagnostics["initial_error_px"] = initial_euclid
    if initial_state_path is not None:
        diagnostics["initial_state_artifact"] = str(initial_state_path)
        # The bounded run-private snapshot is independently bound to metadata;
        # accidental in-place edits cannot silently change a keyed seed chart.
        diagnostics["initial_state_sha256"] = hashlib.sha256(
            initial_state_path.read_bytes()).hexdigest()
    diagnostics["starting_state_disposition"] = getattr(
        handler, "starting_state_disposition", "starting state inconsistent")
    diagnostics["lockbox_source_sha256"] = lockbox_source_sha256
    if handler.record_education:
        diagnostics["seed_join_order"] = list(handler.seed_join_order)
        # The trace is opt-in teaching material: one that cannot be written (too large,
        # or an I/O error) is recorded as unavailable rather than failing a finished solve.
        try:
            trace_path = write_trace(
                run_dir, run_dir.name, "phase3", "seed_join_order",
                {"events": handler.seed_join_order}, enabled=True)
            diagnostics["education_trace"] = str(trace_path) if trace_path else None
            step_payload = getattr(handler, "education_step_trace", None)
            if step_payload is not None:
                step_payload["run_id"] = run_dir.name
                step_payload["phase"] = "phase3"
                trace_path = write_trace(
                    run_dir, run_dir.name, "phase3", "accepted_step",
                    step_payload, enabled=True)
                diagnostics["education_trace"] = str(trace_path) if trace_path else None
        except (OSError, ValueError) as exc:
            diagnostics["education_trace"] = f"unavailable: {exc}"
    report = getattr(out_cams, "calibration_report", None)
    return camset_out, diagnostics, pruned_path, (
        report.to_dict() if report is not None else None)


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

    quality_gate = solve_quality_gate(
        optimisation, handler, stats, residual_xy, initial_euclid,
        final_euclid, observation_count)

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
        # Telecentric images every camera saw one face of: placed, or why not. (On a real rig,
        # 17 such images; D3.11 was taken, hence D3.14.)
        "D3.14_single_face_recovery": dict(
            getattr(handler, "single_face_recovery", None) or {}),
        "quality_gate": quality_gate,
    }


def solve_quality_gate(optimisation, handler, stats: dict,
                  residual_xy: np.ndarray, initial_euclid: float,
                  final_euclid: float, observation_count: int) -> dict:
    """Return the fail-closed disposition for a Phase 3 result.

    Optimiser termination alone is not a calibration-quality claim.  These
    checks are intentionally small and deterministic: the parameters and
    residuals must be finite, the solver must report success, observations
    must exist for every camera in the active detection order, and the final
    reprojection error must improve strictly over the initial value.
    """
    blocking: list[str] = []
    finite_parameters = bool(np.all(np.isfinite(np.asarray(
        getattr(optimisation, "x", []), dtype=float))))
    finite_residuals = bool(np.all(np.isfinite(np.asarray(
        residual_xy, dtype=float)))) and residual_xy.size > 0
    solver_success = bool(stats.get("success", getattr(optimisation, "success", False)))
    error_reduced = bool(
        np.isfinite(initial_euclid)
        and np.isfinite(final_euclid)
        and final_euclid < initial_euclid
    )

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

    observed_cameras: list[int] = []
    try:
        data = np.asarray(handler.get_detection_data(flatten=True))
        if data.ndim == 2 and data.shape[1] > 0:
            observed_cameras = sorted(set(data[:, 0].astype(int).tolist()))
    except Exception:
        pass
    expected_cameras = len(getattr(handler, "cam_names", []))
    camera_coverage = (
        expected_cameras > 0
        and observed_cameras == list(range(expected_cameras))
    )
    if not camera_coverage:
        blocking.append("camera observation graph does not cover every active camera")

    return {
        "status": "complete" if not blocking else "incomplete",
        "blocking_flags": blocking,
        "finite_parameters": finite_parameters,
        "finite_residuals": finite_residuals,
        "solver_success": solver_success,
        "error_reduced": error_reduced,
        "camera_coverage": camera_coverage,
        "observed_cameras": observed_cameras,
        "expected_camera_count": expected_cameras,
        "observation_count": int(observation_count),
    }
