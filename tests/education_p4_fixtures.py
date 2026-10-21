'''Build disposable corpus-backed Phase 4 camsets for education tests.
'''
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def generate_phase4_run(base_dir: Path, *, remove_e1_marker: bool = False,
                        legacy_scale_rule: bool = False,
                        displace_once_seen: bool = False) -> tuple[Path, dict[str, Any]]:
    """Generate a real ChArUco Phase 4 run and its saved diagnostics in temp storage."""
    from pyCamSet import ChArUco
    from pyCamSet.calibration.camera_calibrator import detect_datapoints_in_imfile, run_initial_calibration
    from pyCamSet.utils.saving import load_CameraSet
    from pyCamSet.workflow import phase3, phase4
    from pyCamSet.workflow.education_derive import (
        _phase4_counts, _phase4_rebuild, phase4_scale_consistency_ratio,
    )
    from pyCamSet.workflow.education_playback import _cost_at_start
    from pyCamSet.workflow.education_trace import read_trace, write_trace
    from pyCamSet.utils.general_utils import make_4x4h_tform
    from pyCamSet.optimisation.standard_bundle_handler import _gauge_square_size

    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    corpus = Path(__file__).parent / "test_data" / "calibration_charuco"
    target = ChArUco(num_squares_x=20, num_squares_y=20, square_size=4.0, legacy=False)
    detections, resolutions = detect_datapoints_in_imfile(
        f_loc=corpus, caching=False, calibration_target=target, threads=1)
    phase2 = run_initial_calibration(detections, target, resolutions, save=False)
    phase2.set_resolutions_from_file(floc=corpus)
    phase3_handler, phase3_result, phase3_camset, phase3_stats = phase3.solve(
        phase2, detections, target, fixed_params={},
        options={"outliers": "n", "fixed_pose": 0}, threads=1)
    del phase3_handler, phase3_result, phase3_stats
    options = {"outliers": "n", "fixed_pose": 0}
    if legacy_scale_rule:
        options["target_scale_rule"] = "legacy_mean_all_edges"
    handler, result, solved, stats = phase4.solve(
        phase3_camset, target, detections, fixed_params={}, options=options,
        threads=1, record_education=True)
    run_dir = root / "phase4_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    camset_path = run_dir / "self_calibrated_cameras.camset"
    solved.save(camset_path)
    diagnostics = phase4._diagnostics(
        result, handler, stats,
        {"run_id": "fixture-phase3", "status": "complete", "diagnostics": {}},
        lambda _message: None, out_cams=solved, previous_cams=phase3_camset,
        previous_camera_snapshot=phase4._snapshot_camera_parameters(phase3_camset),
        params={"fixed_pose": 0})
    payload = getattr(handler, "education_step_trace", None)
    if payload is not None:
        payload["run_id"] = "fixture-phase4"
        payload["phase"] = "phase4"
        write_trace(run_dir, "fixture-phase4", "phase4", "accepted_step",
                    payload, enabled=True)
    metadata = {
        "run_id": "fixture-phase4", "status": "complete", "phase": "phase4",
        "params": {"record_education": True},
        "artifacts": {"self_calibrated_camset": str(camset_path)},
        "diagnostics": diagnostics,
    }
    if remove_e1_marker or displace_once_seen:
        saved = load_CameraSet(camset_path)
        if remove_e1_marker:
            saved_handler = saved.calibration_handler
            saved_handler.problem_opts = dict(saved_handler.problem_opts)
            saved_handler.problem_opts.pop("target_scale_rule", None)
        handler = _phase4_rebuild(saved, diagnostics)
        params = np.asarray(saved.calibration_params, dtype=np.float64).copy()
        if remove_e1_marker:
            scale, displacement = phase4._target_shape_change(
                handler, type("OptimisationResult", (), {"x": params})(),
                handler.visible_feature_mask)
            diagnostics["D4.5_gauge_scale_factor"] = scale
            diagnostics["D4.7_mean_target_displacement_mm"] = displacement
        if displace_once_seen:
            rows = np.asarray(handler._flat_detections(), dtype=np.float64)
            counts = _phase4_counts(handler)
            free = np.asarray(handler.bundlePrimitive.bdpt_unfixed, dtype=bool).reshape(-1)
            free_indices = np.flatnonzero(free)
            point_start = int(handler.bundlePrimitive.pose_end)
            primitive = handler.bundlePrimitive.return_bundle_primitives(params.copy())
            points = np.asarray(primitive[3], dtype=np.float64).reshape(-1, 3).copy()
            extr, poses = np.asarray(primitive[1]), np.asarray(primitive[2])
            gauge = set(diagnostics["D4.2_gauge_fixed_points"]["indices"])
            moved = 0
            for point_index in np.flatnonzero(counts == 1):
                if int(point_index) in gauge:
                    continue
                scalar_ids = np.arange(3 * point_index, 3 * point_index + 3)
                packed = np.searchsorted(free_indices, scalar_ids)
                if np.any(packed >= len(free_indices)) or not np.array_equal(
                        free_indices[packed], scalar_ids):
                    continue
                row = rows[rows[:, 2].astype(int) == point_index][0]
                camera_index, image_index = int(row[0]), int(row[1])
                target_to_world = make_4x4h_tform(
                    poses[image_index, :3], poses[image_index, 3:])
                world_to_camera = make_4x4h_tform(
                    extr[camera_index, :3], extr[camera_index, 3:])
                camera_to_world = np.linalg.inv(world_to_camera)
                centre = (np.linalg.inv(target_to_world)
                          @ np.append(camera_to_world[:3, 3], 1.0))[:3]
                original = points[point_index].copy()
                displaced = centre + 1.05 * (original - centre)
                ray_before = world_to_camera @ target_to_world @ np.append(original, 1.0)
                ray_after = world_to_camera @ target_to_world @ np.append(displaced, 1.0)
                angle = np.degrees(np.arctan2(
                    np.linalg.norm(np.cross(ray_before[:3], ray_after[:3])),
                    float(ray_before[:3] @ ray_after[:3])))
                if angle > 1e-10:
                    raise AssertionError("once-seen displacement changed its observation ray")
                params[point_start + packed] = displaced
                points[point_index] = displaced
                moved += 1
            solved_points = np.asarray(
                handler.get_updated_target(params.copy()), dtype=np.float64).reshape(-1, 3)
            nominal = np.asarray(handler.target.point_data, dtype=np.float64).reshape(-1, 3)
            measured = counts >= 2
            pairs = handler.target.valid_map if isinstance(
                handler.target.valid_map, np.ndarray) else None
            ratio = phase4_scale_consistency_ratio(
                nominal, solved_points, measured,
                float(_gauge_square_size(handler.target)), pairs=pairs)
            if moved == 0 or abs(ratio - 1.0) <= 0.01:
                raise ValueError(
                    "fixture has too few displaceable once-seen points to drive the ratio past tolerance")
            scale, displacement = phase4._target_shape_change(
                handler, type("OptimisationResult", (), {"x": params})(),
                handler.visible_feature_mask)
            diagnostics["D4.5_gauge_scale_factor"] = scale
            diagnostics["D4.7_mean_target_displacement_mm"] = displacement
            saved.calibration_params = params.copy()
            saved.save(camset_path)
            trace = read_trace(run_dir, "fixture-phase4", "phase4")
            step = next(record["payload"] for record in trace["records"]
                        if record["kind"] == "accepted_step")
            step["steps"][-1]["x"] = params.tolist()
            step["steps"][-1]["dtype"] = "float64"
            step["steps"][-1]["cost"] = _cost_at_start(handler, params)
            write_trace(run_dir, "fixture-phase4", "phase4", "accepted_step",
                        step, enabled=True)
    return camset_path, metadata
