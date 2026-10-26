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
from scipy import sparse

from pyCamSet.utils.paths import long_path
from pyCamSet.workflow.diagnostics import per_camera_mean_reprojection
from pyCamSet.workflow.logs import LogFn, captured_output, discard
from pyCamSet.workflow.phase3 import solve_quality_gate
from pyCamSet.workflow.workspace import (
    WorkspaceManager,
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
            # A camset that is not the selected run's own output (an override path)
            # carries none of the run's recorded status: the checks and the quality gate
            # treat it as a camset alone, as for a headless caller. The run record still
            # names the selected run.
            camset_out, diagnostics, report = _solve(
                params, run_dir, Path(phase3_camset),
                phase3_run_for_camset(phase3_run, phase3_camset), log)
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

    options = dict(options or {})
    options.setdefault("target_scale_rule", "measured_edges_v1")
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

    previous_cams = load_CameraSet(long_path(phase3_camset))
    # Capture values, not Camera references: the self-calibration handler may
    # mutate the shared Camera objects while producing its output camset.
    previous_camera_snapshot = _snapshot_camera_parameters(previous_cams)
    # A headless caller may start from a camset alone, with no Phase 3 run
    # record; there is then no recorded path or status to check it against.
    if phase3_run is not None:
        if phase3_run.get("status") == "failed":
            raise RuntimeError(
                "The selected Phase 3 run failed; refusing to start Phase 4. "
                "Re-run Phase 3 first.")
        recorded_camset = ((phase3_run.get("artifacts") or {}).get(
            "optimised_camset") or (phase3_run.get("artifacts") or {}).get(
                "phase3_camset_used"))
        if not recorded_camset or not _same_resolved_path(
                recorded_camset, phase3_camset):
            raise RuntimeError(
                "Phase 4 input camset does not verify against the selected "
                "Phase 3 run; refusing to use its quality status as provenance.")
    selected = list(params.get("selected_cameras") or [])
    if selected and set(previous_cams.get_names()) != set(selected):
        raise RuntimeError(
            "Phase 3 camset cameras do not match selected camera subset. "
            "Re-run Phase 3 with the same selected cameras.")

    previous_handler = getattr(previous_cams, "calibration_handler", None)
    if previous_handler is None:
        raise RuntimeError(
            "Selected Phase 3 camset has no calibration handler metadata.")
    from pyCamSet.utils.target_units import (
        LEGACY_UNVERIFIED, MILLIMETRES, METRES,
        target_point_data_units,
    )
    rebuilt_units = target_point_data_units(previous_handler.target)
    saved_units = getattr(previous_cams, "_saved_point_data_units", LEGACY_UNVERIFIED)
    known_units = {METRES, MILLIMETRES}
    input_notices: list[str] = []
    if rebuilt_units in known_units and saved_units in known_units:
        # Both sides know their units: only a real disagreement is refused.
        if rebuilt_units != saved_units:
            raise RuntimeError(
                f"Phase 4 target units changed since Phase 3 ({saved_units} saved, "
                f"{rebuilt_units} rebuilt); refusing a mixed-unit solve.")
    elif rebuilt_units in known_units:
        # A camset saved before units were recorded cannot be checked, and
        # re-saving it cannot add the record; the rebuilt target's own units
        # are the best available, so carry on and say so.
        input_notices.append(
            "the saved Phase 3 camset predates target-unit records; assumed the "
            f"rebuilt target's own coordinate units ({rebuilt_units})")
    else:
        # The target is not in the unit table, so there is nothing to compare
        # against; refusing would block it for good.
        input_notices.append(
            "the target's coordinate units are not in the known-unit table, so "
            "they could not be checked against the saved Phase 3 camset "
            f"(saved record: {saved_units})")
    previous_detection = getattr(previous_handler, "detection", None)
    detection_names = set(getattr(previous_detection, "cam_names", []) or [])
    if detection_names != set(previous_cams.get_names()):
        raise RuntimeError(
            "Selected Phase 3 observations do not cover the saved camera set; "
            "refusing to run Phase 4 with silent camera loss.")

    options = dict(params.get("problem_options") or {})
    phase3_options = getattr(previous_handler, "problem_opts", {}) or {}
    phase3_fixed_pose = int(phase3_options.get("fixed_pose", 0))
    requested_fixed_pose = int(options.get("fixed_pose", phase3_fixed_pose))
    if requested_fixed_pose != phase3_fixed_pose:
        raise RuntimeError(
            "Phase 4 fixed_pose must match Phase 3's fixed_pose; refusing "
            "to start the rig in a different world frame.")
    options["fixed_pose"] = phase3_fixed_pose
    options.setdefault("ref_cam", 0)
    options.setdefault("ref_pose", 0)
    options.setdefault("outliers", "n")
    options.setdefault("max_nfev", 100)
    # Phase 4 re-seeds from the Phase 3 cameras: place the same single-face
    # images Phase 3 did, or they drop out of this solve.
    options.setdefault("recover_single_face_telecentric",
                       bool(phase3_options.get("recover_single_face_telecentric", False)))
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
    reloaded = load_CameraSet(long_path(camset_out))
    if set(reloaded.get_names()) != set(previous_cams.get_names()):
        raise RuntimeError(
            "Phase 4 save/reload changed the active camera set; refusing "
            "to publish a result with silent camera loss.")

    diagnostics = _diagnostics(
        optimisation, handler, stats, phase3_run, log,
        out_cams=out_cams, previous_cams=previous_cams,
        previous_camera_snapshot=previous_camera_snapshot,
        params=params, input_notices=input_notices)
    # Recorded after the gate, which knows nothing of these two facts.  The
    # camset path was checked above when a run was given (None: nothing to
    # check it against); its bytes are not hashed.
    diagnostics["quality_gate"]["phase3_input_camset_path_matches_recorded"] = (
        True if phase3_run is not None else None)
    diagnostics["quality_gate"]["phase3_camset_content_verified"] = False
    report = getattr(out_cams, "calibration_report", None)
    return camset_out, diagnostics, (
        report.to_dict() if report is not None else None)


def _diagnostics(optimisation, handler, stats: dict,
                 phase3_run: Optional[dict], log: LogFn,
                 *, out_cams=None, previous_cams=None,
                 previous_camera_snapshot=None,
                 params: Optional[dict] = None,
                 input_notices: Optional[list] = None) -> dict:
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

    per_camera, residual_xy = per_camera_mean_reprojection(
        optimisation, handler, log, "D4.12")
    detection_data = np.asarray(handler.get_detection_data(flatten=True))
    per_image: dict[str, float] = {}
    if detection_data.ndim == 2 and len(detection_data) == len(residual_xy):
        residual_norm = np.linalg.norm(residual_xy, axis=1)
        image_index = detection_data[:, 1].astype(int)
        per_image = {
            str(index): float(np.mean(residual_norm[image_index == index]))
            for index in np.unique(image_index).tolist()
        }
    else:
        log("Warning: D4.13 skipped (detections do not line up with residuals).")
    quality_gate = _quality_gate(
        optimisation, handler, stats, residual_xy,
        initial_euclid, final_euclid, int(len(detection_data)),
        out_cams=out_cams, previous_cams=previous_cams,
        previous_camera_snapshot=previous_camera_snapshot,
        params=params, phase3_run=phase3_run, input_notices=input_notices)
    for notice in quality_gate.get("notices", []):
        log(f"NOTE: {notice}")

    diagnostics = {
        "D4.1_n_free_target_points": int(np.sum(visible)),
        "D4.2_gauge_fixed_points": {
            "count": len(fixed_indices), "indices": fixed_indices},
        "D4.3_per_image_initial_reprojection": per_image_initial.tolist(),
        "D4.3_initial_euclid_px": initial_euclid,
        "D4.3_final_euclid_px": final_euclid,
        "D4.3_initial_objective_cost": stats.get("initial_objective_cost"),
        "D4.3_final_objective_cost": stats.get("final_objective_cost"),
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


_MAX_REDUCED_BYTES = 64 * 1024 * 1024


def _split_jacobian_nullity(jacobian, *,
                            point_column_start: int,
                            point_mask: np.ndarray,
                            detection_data: np.ndarray,
                            max_reduced_bytes: int = _MAX_REDUCED_BYTES) -> dict:
    """Measure total nullity and separate point-only from global modes.

    The point-only null space is the kernel of each feature's own residual
    block with camera/pose columns fixed. The remaining null directions are
    exactly those whose projection into global columns is non-zero.

    The reduced global Jacobian is reduced by a streamed QR (TSQR): the
    triangular factor R (globals x globals) is stacked with a chunk of newly
    reduced rows and re-factorised, so the observations never sit in memory
    together. ``max_reduced_bytes`` bounds the *stacked block* (R plus one
    chunk), so ``chunk_rows`` is sized from it. The true peak is a small
    multiple of that bound: the pending chunk, the stacked copy made from it,
    and LAPACK's own working copy of the stacked block can coexist (about
    three times the budget at worst), plus the sparse Jacobian itself.

    Ranks are counted against ``frobenius_norm * max(shape) * eps``. The
    Frobenius norm is never smaller than the largest singular value, so this
    tolerance is conservative: it can be larger than the conventional
    ``sigma_max * max(shape) * eps`` and so may report near-null modes as
    null (a rank deficit that a tighter tolerance would not show).
    """
    matrix = sparse.csr_matrix(jacobian, dtype=float)
    point_mask = np.asarray(point_mask, dtype=bool)
    detection_data = np.asarray(detection_data)
    if len(matrix.shape) != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("solution Jacobian must be a finite non-empty matrix")
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("solution Jacobian must be a finite non-empty matrix")
    if point_mask.ndim != 1 or point_mask.size % 3:
        raise ValueError("point-coordinate mask must contain 3D feature blocks")
    if detection_data.ndim != 2 or detection_data.shape[1] < 3:
        raise ValueError("detection rows must include a feature index")
    if matrix.shape[0] != 2 * detection_data.shape[0]:
        raise ValueError("Jacobian residual rows do not match detection rows")
    if point_column_start < 0 or point_column_start > matrix.shape[1]:
        raise ValueError("point parameter offset is outside the Jacobian")

    global_parameter_count = int(point_column_start)
    # The reduced global Jacobian is never held whole: its singular values are those of the
    # triangular factor R of a QR decomposition, built in row chunks (TSQR). Memory is set by
    # the global parameter count, not by the observations, and unlike J^T J the condition
    # number is not squared. (A dense (2 x observations) x globals copy exceeded
    # the old 16 MiB budget on every real 8-camera, 100-image rig, so every Phase 4 run was
    # gated 'weakly constrained' without being measured.)
    itemsize = np.dtype(float).itemsize
    reduced_bytes = int(global_parameter_count * global_parameter_count * itemsize)
    # Rows of the stacked block (R, then the chunk) that the budget allows. A chunk
    # under a quarter of R's height would make the repeated QR factorisations
    # dominate, so that is refused as over budget rather than run for hours.
    budget_rows = (int(max_reduced_bytes) // (global_parameter_count * itemsize)
                   if global_parameter_count else 0)
    chunk_rows = budget_rows - global_parameter_count
    if global_parameter_count and chunk_rows < max(global_parameter_count // 4, 1):
        raise ValueError(
            "the reduced global Jacobian's triangular factor exceeds the "
            f"{max_reduced_bytes // (1024 * 1024)} MiB measurement budget")
    # The Frobenius norm upper-bounds sigma_max without densifying the full matrix.
    norm_bound = float(np.linalg.norm(matrix.data))
    tolerance = (norm_bound * max(matrix.shape) * np.finfo(float).eps)
    local_nullity = 0
    local_rank_sum = 0
    point_column = point_column_start
    feature_indices = detection_data[:, 2].astype(int)
    feature_count = point_mask.size // 3
    if np.any(feature_indices < 0) or np.any(feature_indices >= feature_count):
        raise ValueError("detection feature index is outside the target point mask")
    triangular = np.empty((0, global_parameter_count), dtype=float)
    pending: list[np.ndarray] = []
    pending_rows = 0

    def _compress() -> None:
        nonlocal triangular, pending, pending_rows
        if pending:
            stacked = np.vstack([triangular, *pending])
            triangular = (np.linalg.qr(stacked, mode="r") if stacked.shape[0]
                          else stacked)
            pending, pending_rows = [], 0

    # Group the detections by feature once (a stable sort keeps each feature's
    # observations in detection order) instead of masking every detection per feature.
    order = np.argsort(feature_indices, kind="stable")
    group_ends = np.searchsorted(feature_indices[order], np.arange(feature_count), side="right")
    group_starts = np.concatenate(([0], group_ends[:-1]))

    reduced_row = 0
    for feature in range(feature_count):
        free_coordinates = int(np.count_nonzero(point_mask[3 * feature:3 * feature + 3]))
        columns = np.arange(point_column, point_column + free_coordinates)
        point_column += free_coordinates
        observations = order[group_starts[feature]:group_ends[feature]]
        if not len(observations):
            # No residual rows: every free coordinate of this feature is null.
            local_nullity += free_coordinates
            continue
        rows = np.column_stack((2 * observations, 2 * observations + 1)).reshape(-1)
        global_block = matrix[rows, :global_parameter_count].toarray()
        local_rank = 0
        if free_coordinates:
            local_block = matrix[np.ix_(rows, columns)].toarray()
            local_basis, local_singular_values, _ = np.linalg.svd(
                local_block, full_matrices=False)
            local_rank = int(np.count_nonzero(local_singular_values > tolerance))
            local_rank_sum += local_rank
            if local_rank:
                basis = local_basis[:, :local_rank]
                global_block -= basis @ (basis.T @ global_block)
        local_nullity += free_coordinates - local_rank
        if global_parameter_count:
            # Rows of different features need not stay together for a QR, so a
            # block longer than a chunk is fed in chunk-sized slices.
            for start in range(0, len(rows), chunk_rows):
                piece = global_block[start:start + chunk_rows]
                if pending_rows + len(piece) > chunk_rows:
                    _compress()
                pending.append(piece)
                pending_rows += len(piece)
        reduced_row += len(rows)

    if point_column != matrix.shape[1]:
        raise ValueError("point-coordinate mask does not span the Jacobian point columns")
    if reduced_row != matrix.shape[0]:
        raise ValueError("feature blocks do not span the Jacobian residual rows")
    _compress()
    reduced_singular_values = (np.linalg.svd(triangular, compute_uv=False)
                               if triangular.size else np.zeros(0, dtype=float))
    reduced_rank = int(np.count_nonzero(reduced_singular_values > tolerance))
    global_nullity = int(global_parameter_count - reduced_rank)
    rank = int(local_rank_sum + reduced_rank)
    total_nullity = int(matrix.shape[1] - rank)
    if total_nullity != local_nullity + global_nullity:
        raise ValueError("Schur nullity does not reconcile with the point/global split")
    return {
        "rank": rank,
        "free_parameters": int(matrix.shape[1]),
        "total_nullity": total_nullity,
        "point_local_nullity": int(local_nullity),
        "global_nullity": int(global_nullity),
        "point_local_global_nullity_split": "measured",
        "tolerance": float(tolerance),
        "rank_tolerance_method": "frobenius_norm_upper_bound * max(shape) * float64_epsilon",
        "smallest_singular_values": reduced_singular_values[
            -min(8, len(reduced_singular_values)):].tolist(),
        "jacobian_shape": [int(value) for value in matrix.shape],
        "reduced_global_shape": [int(matrix.shape[0]), int(global_parameter_count)],
        "reduced_global_method": "streamed QR (TSQR) triangular factor",
        "reduced_global_bytes": reduced_bytes,
        "max_reduced_global_bytes": int(max_reduced_bytes),
        "tsqr_chunk_rows": int(chunk_rows),
    }


def _jacobian_nullity_measurement(optimisation, handler) -> dict:
    """Measure the sparse final Jacobian already returned by the solver."""
    matrix = getattr(optimisation, "jac", None)
    if matrix is None:
        raise RuntimeError("solver result did not retain its final Jacobian")
    if not sparse.issparse(matrix):
        raise RuntimeError("bundle-adjustment Jacobian must remain sparse for bounded measurement")
    result = _split_jacobian_nullity(
        matrix,
        point_column_start=int(handler.bundlePrimitive.pose_end),
        point_mask=np.asarray(handler.feat_unfixed, dtype=bool),
        detection_data=np.asarray(handler.get_detection_data(flatten=True)),
    )
    result["jacobian_storage"] = "scipy_sparse"
    return result


def _quality_gate(optimisation, handler, stats: dict,
                  residual_xy: np.ndarray, initial_euclid: float,
                  final_euclid: float, observation_count: int, *,
                  out_cams=None, previous_cams=None,
                  previous_camera_snapshot=None,
                  params: Optional[dict] = None,
                  phase3_run: Optional[dict] = None,
                  input_notices: Optional[list] = None) -> dict:
    """Fail closed when Phase 4 produced no scientifically usable result.

    ``phase3_run`` is the run the starting camset came from, or None when the
    caller started from a camset alone; only a given run is checked for a
    complete status and a run id.  ``previous_camera_snapshot`` is the Phase 3
    cameras' values taken before the solve, which the drift is measured from;
    without it the drift reads ``previous_cams`` as they stand.
    ``input_notices`` are non-blocking facts about the inputs, found before the
    solve, that are listed with the gate's own notices.
    """
    base = solve_quality_gate(
        optimisation, handler, stats, residual_xy, initial_euclid,
        final_euclid, observation_count)
    blocking: list[str] = list(base["blocking_flags"])
    initial_cost = float(stats.get("initial_objective_cost", float("nan")))
    final_cost = float(stats.get("final_objective_cost", float("nan")))
    objective_cost_reduced = bool(
        np.isfinite(initial_cost) and np.isfinite(final_cost)
        and final_cost < initial_cost)
    phase3_provenance_complete = None
    if phase3_run is not None:
        phase3_status = phase3_run.get("status")
        phase3_provenance_complete = False
        if phase3_status != "complete":
            blocking.append(
                f"Phase 3 input disposition was {phase3_status}; re-run it before hand-off")
        elif not str(phase3_run.get("run_id") or "").strip():
            blocking.append("Phase 3 input run has no run id")
        else:
            phase3_provenance_complete = True

    detection_data = np.asarray(handler.get_detection_data(flatten=True))
    image_indices = (detection_data[:, 1].astype(int)
                     if detection_data.ndim == 2 and detection_data.shape[1] > 1
                     else np.array([], dtype=int))
    expected_images = int(getattr(getattr(handler, "detection", None), "max_ims", 0))
    observed_images = sorted(set(image_indices.tolist()))
    # ``max_ims`` is the highest global image index plus one, not the number
    # of observed images.  Treat holes as explicitly missing observations;
    # silently pretending the list is contiguous would hide them from the GUI.
    # An image the solve itself excluded -- no usable target pose, marked in
    # the handler's missing_poses and reported in its summary -- is not such
    # a hole: it is listed as excluded, and only unexplained holes block.
    marked = getattr(handler, "missing_poses", None)
    marked = (np.asarray(marked, dtype=bool) if marked is not None
              else np.zeros(0, dtype=bool))
    excluded_images = (sorted(int(i) for i in np.flatnonzero(marked))
                       if marked.size == expected_images else [])
    missing_images = sorted(
        set(range(expected_images)) - set(observed_images) - set(excluded_images))
    image_coverage = bool(expected_images and not missing_images)
    if expected_images and not image_coverage:
        blocking.append("image observation graph has missing image indices")
    notices: list[str] = list(input_notices or [])
    if excluded_images:
        notices.append(
            f"{len(excluded_images)} image(s) had no usable target pose and were "
            f"excluded from the solve: {excluded_images}")

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
    fixed_pose_observed = (
        gauge["fixed_pose"] is not None
        and int(gauge["fixed_pose"]) in set(observed_images))
    try:
        nullity_measurement = _jacobian_nullity_measurement(optimisation, handler)
    except Exception as exc:
        nullity_measurement = None
        blocking.append("Jacobian null-space measurement failed at solution")
        constraint_disposition = {
            "status": "weakly_constrained",
            "global_nullity": "not_verified",
            "point_local_global_nullity_split": "not_verified",
            "reason": f"Jacobian null-space measurement failed: {exc}",
        }
    else:
        global_nullity = nullity_measurement["global_nullity"]
        weakly_constrained = bool(global_nullity or not fixed_pose_observed)
        constraint_disposition = {
            "status": "weakly_constrained" if weakly_constrained else "constrained",
            "global_nullity": global_nullity,
            "point_local_nullity": nullity_measurement["point_local_nullity"],
            "total_nullity": nullity_measurement["total_nullity"],
            "rank": nullity_measurement["rank"],
            "jacobian_shape": nullity_measurement["jacobian_shape"],
            "rank_tolerance": nullity_measurement["tolerance"],
            "point_local_global_nullity_split": "measured",
            "reason": (
                f"measured {global_nullity} global null directions and "
                f"{nullity_measurement['point_local_nullity']} point-local directions"),
        }
    constraint_disposition.update({
        "fixed_pose_observed": fixed_pose_observed,
        "fixed_pose_excluded_or_unobserved": (
            gauge["fixed_pose"] is not None and not fixed_pose_observed),
    })

    camera_quality = (_camera_quality(out_cams) if out_cams is not None else {
        "physically_plausible": True, "available": False, "cameras": {}})
    if not camera_quality["physically_plausible"]:
        blocking.append("camera intrinsics, distortion, or extrinsics are implausible")

    return {
        **base,
        "status": "complete" if not blocking else "incomplete",
        "blocking_flags": blocking,
        "notices": notices,
        "phase3_provenance_complete": phase3_provenance_complete,
        "objective_cost_reduced": objective_cost_reduced,
        "image_coverage": image_coverage,
        "observed_images": observed_images,
        "missing_images": missing_images,
        "excluded_images": excluded_images,
        "expected_image_count": expected_images,
        "gauge": gauge,
        "constraint_disposition": constraint_disposition,
        "jacobian_nullity": nullity_measurement,
        "camera_quality": camera_quality,
        "parameter_drift": _camera_drift(
            previous_camera_snapshot if previous_camera_snapshot is not None
            else previous_cams, out_cams),
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


def phase3_run_for_camset(phase3_run: Optional[dict], phase3_camset) -> Optional[dict]:
    """
    The Phase 3 run whose status may stand as provenance for *phase3_camset*.

    That is *phase3_run* only when the camset is the run's own recorded output; a camset
    from anywhere else (an override path, another run's file) has no recorded status, so
    Phase 4 starts from the camset alone, as a headless caller does.

    :param phase3_run: the selected Phase 3 run record, or None
    :param phase3_camset: the camset Phase 4 will start from
    :return: *phase3_run*, or None
    """
    if phase3_run is None:
        return None
    artifacts = phase3_run.get("artifacts") or {}
    recorded = artifacts.get("optimised_camset") or artifacts.get("phase3_camset_used")
    if recorded and _same_resolved_path(recorded, phase3_camset):
        return phase3_run
    return None


def _same_resolved_path(first, second) -> bool:
    """Compare paths after platform-native resolution; uncertainty fails closed."""
    try:
        return Path(first).resolve() == Path(second).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _snapshot_camera_parameters(cams) -> dict[str, dict[str, np.ndarray]]:
    """Take independent array snapshots before a solve can mutate cameras."""
    if cams is None:
        return {}
    return {
        name: {
            label: np.array(getattr(cams[name], label), dtype=float, copy=True)
            for label in ("intrinsic", "distortion_coefs", "extrinsic")
        }
        for name in cams.get_names()
    }


def _camera_drift(previous_snapshot, out_cams) -> dict:
    """Report relative parameter movement between Phase 3 and Phase 4."""
    if previous_snapshot is None or out_cams is None:
        return {"available": False, "cameras": {}}
    if hasattr(previous_snapshot, "get_names"):
        previous_snapshot = _snapshot_camera_parameters(previous_snapshot)
    old_names = set(previous_snapshot)
    new_names = set(out_cams.get_names())
    values: dict[str, dict[str, float]] = {}
    for name in sorted(old_names & new_names):
        old = previous_snapshot[name]
        new = out_cams[name]
        values[name] = {}
        for label in ("intrinsic", "distortion_coefs", "extrinsic"):
            before = np.asarray(old[label], dtype=float)
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
    from pyCamSet.utils.target_units import millimetres_per_point_data_unit

    mm_per_unit = millimetres_per_point_data_unit(handler.target)
    displacement_mm = (
        float(np.nanmean(displacement) * mm_per_unit)
        if displacement.size and mm_per_unit is not None else float("nan"))
    return scale, displacement_mm
