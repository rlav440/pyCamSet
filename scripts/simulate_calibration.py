"""
Simulate a calibration capture of a cube target and run the whole pipeline on
it, scoring every phase against the ground truth.

For each target asked for, this:

1. builds a ring of cameras (:func:`pyCamSet.simulation.ring_rig`) and a set
   of hand-held target poses (:func:`pyCamSet.simulation.held_target_poses`);
2. pastes the target's faces imperfectly
   (:meth:`pyCamSet.simulation.PastingErrors.by_hand`);
3. renders every camera's view of every pose, through a realistic sensor
   (:class:`pyCamSet.simulation.SensorModel`), into ``<out>/<name>/images``;
4. runs Phases 1-4 exactly as the GUI does -- the workflow modules, saving
   runs into the image folder's workspace, so the result can be opened in
   the GUI and its education windows afterwards;
5. compares the Phase 2, 3 and 4 camera sets (and the Phase 4 target shape)
   with the truth, and writes ``report.json`` and ``report.md``.

Example, a Ccube and a Ccube2::

    python scripts/simulate_calibration.py --out sim_runs \\
        --target '{"type": "Ccube", "length": 80, "n_points": 6}' \\
        --target '{"type": "Ccube2", "length": 80, "n_points": 6}'

See ``docs/how-to/simulate.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _run_phases(images: Path, spec: dict, threads: int, log) -> dict:
    """Phases 1-4 through the workflow modules; the run records by phase."""
    from pyCamSet.workflow import phase1, phase2, phase3, phase4
    from pyCamSet.workflow.workspace import WorkspaceManager, workspace_path_for

    workspace = WorkspaceManager(workspace_path_for(images))
    common = {"f_loc": str(images), "target": spec, "selected_cameras": [],
              "threads": threads, "n_lim": None, "caching": False,
              "upscale_factor": 1, "high_distortion": False,
              "fixed_params": None, "problem_options": None,
              "record_education": True, "record_phase1_trace": True,
              "manifest_digest": True}
    runs = {}
    started = time.perf_counter()
    runs["phase1"] = phase1.run(dict(common), workspace, log)
    _check(runs["phase1"], "Phase 1")
    runs["phase2"] = phase2.run(dict(common), workspace, log, phase1_run=runs["phase1"])
    _check(runs["phase2"], "Phase 2")
    runs["phase3"] = phase3.run(dict(common), workspace, log,
                                phase2_run=runs["phase2"], phase1_run=runs["phase1"])
    _check(runs["phase3"], "Phase 3")
    camset3 = Path(runs["phase3"]["artifacts"]["optimised_camset"])
    runs["phase4"] = phase4.run(dict(common), workspace, log,
                                phase3_run=runs["phase3"], phase3_camset=camset3)
    _check(runs["phase4"], "Phase 4")
    runs["seconds"] = time.perf_counter() - started
    return runs


def _check(run: dict, phase: str) -> None:
    if run.get("status") == "failed" or run.get("error"):
        raise RuntimeError(f"{phase} failed: {run.get('error')}")


def _score(runs: dict, capture) -> dict:
    """Each phase's camera set, and Phase 4's target, against the truth."""
    from pyCamSet.simulation import compare_to_truth
    from pyCamSet.utils.saving import load_CameraSet

    scores = {}
    for phase, key in (("phase2", "initial_camset"), ("phase3", "optimised_camset"),
                       ("phase4", "self_calibrated_camset")):
        path = (runs[phase].get("artifacts") or {}).get(key)
        if not path:
            continue
        cams = load_CameraSet(path)
        solved = None
        if phase == "phase4":
            handler = cams.calibration_handler
            solved = np.asarray(handler.get_updated_target(
                np.asarray(cams.calibration_params, float).copy()), float)
            solved = solved.reshape(capture.true_points.shape)
        scores[phase] = compare_to_truth(cams, capture, solved_points=solved,
                                         extrinsics=phase != "phase2")
        scores[phase]["status"] = runs[phase].get("status")
    return scores


def simulate_one(spec: dict, out: Path, args, log=print) -> dict:
    """One target: capture, calibrate, score."""
    from pyCamSet.calibration_targets.core.target_registry import build_target
    from pyCamSet.simulation import (PastingErrors, SensorModel, held_target_poses,
                                     ring_rig, simulate_capture)

    target = build_target(spec)
    name = (f"{spec['type'].lower()}_" + "_".join(
        f"{k}{v}" for k, v in spec.items() if k != "type")).replace(".", "p")
    images = out / name / "images"
    rig = ring_rig(args.cameras, radius=args.radius, res=tuple(args.res),
                   focal=args.focal, seed=args.seed)
    poses = held_target_poses(args.shots, translation=args.wander, seed=args.seed + 1)
    errors = (PastingErrors.none(len(target.faceData.ur_ftform)) if args.perfect
              else PastingErrors.by_hand(len(target.faceData.ur_ftform), seed=args.seed + 2))
    sensor = SensorModel(seed=args.seed + 3)

    # point_data is (faces, ..., 3): everything between is one face's grid of points.
    points_per_face = int(np.prod(target.point_data.shape[1:-1]))
    log(f"== {name}: {points_per_face} "
        f"points a face; rendering {args.cameras} x {args.shots} images")
    started = time.perf_counter()
    capture = simulate_capture(target, rig, poses, images, errors=errors, sensor=sensor,
                               log=log)
    render_s = time.perf_counter() - started
    runs = _run_phases(images, spec, args.threads, lambda line: None)
    detections = runs["phase1"].get("diagnostics", {})
    result = {
        "target": spec, "folder": str(images),
        "points_per_face": points_per_face,
        "images": args.cameras * args.shots,
        "visible_points": int(capture.visible_points.sum()),
        "detected_points": _detected_count(runs["phase1"]),
        "render_seconds": round(render_s, 1),
        "pipeline_seconds": round(runs["seconds"], 1),
        "phase1_diagnostics": {k: v for k, v in detections.items()
                               if isinstance(v, (int, float, str))},
        "scores": _score(runs, capture),
    }
    return result


def _detected_count(phase1_run: dict) -> int:
    from pyCamSet.utils.saving import load_pickle
    from pyCamSet.workflow.detections import extract_detection

    path = (phase1_run.get("artifacts") or {}).get("detected_datapoints_pickle")
    if not path:
        return 0
    detections = extract_detection(load_pickle(path))
    data = detections.get_data()
    return 0 if data is None else int(len(data))


def _markdown(results: list[dict]) -> str:
    lines = ["| Target | pts/face | detected / readable | phase | reproj RMS px | "
             "worst focal % | worst position mm | worst rotation deg | "
             "worst lens RMS px | worst lens shape px | target shape mm |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        label = ", ".join(f"{k}={v}" for k, v in r["target"].items() if k != "type")
        for phase, s in r["scores"].items():
            sm = s["summary"]
            lines.append(
                f"| {r['target']['type']} {label} | {r['points_per_face']} | "
                f"{r['detected_points']} / {r['visible_points']} | {phase} | "
                f"{s.get('reprojection_rms_px', float('nan')):.3f} | "
                f"{sm['worst_focal_error_pct']:.3f} | {sm['worst_position_error_mm']:.3f} | "
                f"{sm['worst_rotation_error_deg']:.4f} | {sm['worst_lens_rms_px']:.3f} | "
                f"{sm['worst_lens_shape_rms_px']:.3f} | "
                f"{s.get('target_shape_rms_mm', float('nan')):.3f} |")
    pasting = results[0]["scores"].get("phase2", {}).get("pasting_rms_mm") if results else None
    if pasting is not None:
        lines.append("")
        lines.append(f"Pasting error of the first target (nominal vs true points, after "
                     f"the best rigid fit): {pasting:.3f} mm RMS.")
    return "\n".join(lines) + "\n"


def rescore(out: Path) -> list[dict]:
    """Score the captures already in ``out`` again, from their latest runs."""
    from pyCamSet.simulation import load_ground_truth

    with open(out / "report.json", encoding="utf-8") as handle:
        results = json.load(handle)
    for result in results:
        images = Path(result["folder"])
        capture = load_ground_truth(images)
        runs = {}
        for phase in ("phase2", "phase3", "phase4"):
            run_dirs = sorted((images / ".pycamset_workspace" / f"{phase}_runs").iterdir())
            with open(run_dirs[-1] / "metadata.json", encoding="utf-8") as handle:
                runs[phase] = json.load(handle)
        result["scores"] = _score(runs, capture)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="where to write everything")
    parser.add_argument("--target", action="append",
                        help="a target spec as JSON; repeat for several")
    parser.add_argument("--rescore", action="store_true",
                        help="score the captures already in --out again, without "
                             "rendering or calibrating")
    parser.add_argument("--cameras", type=int, default=4)
    parser.add_argument("--shots", type=int, default=30)
    parser.add_argument("--radius", type=float, default=0.35, help="ring radius, metres")
    parser.add_argument("--wander", type=float, default=0.06,
                        help="how far the target is moved from the centre, metres, so "
                             "that it reaches the edges of the images")
    parser.add_argument("--focal", type=float, default=1800.0, help="pixels")
    parser.add_argument("--res", type=int, nargs=2, default=[1600, 1200],
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--perfect", action="store_true",
                        help="paste the faces perfectly instead of by hand")
    parser.add_argument("--threads", type=int, default=1,
                        help="detection processes; 1 runs in this process")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.rescore:
        results = rescore(args.out)
        with open(args.out / "report.json", "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=1)
        with open(args.out / "report.md", "w", encoding="utf-8") as handle:
            handle.write(_markdown(results))
        print(_markdown(results))
        return 0
    if not args.target:
        parser.error("give at least one --target, or --rescore")
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for text in args.target:
        result = simulate_one(json.loads(text), args.out, args)
        results.append(result)
        print(_markdown([result]))
        with open(args.out / "report.json", "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=1)
    with open(args.out / "report.md", "w", encoding="utf-8") as handle:
        handle.write(_markdown(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
