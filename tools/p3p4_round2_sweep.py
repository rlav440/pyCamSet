'''
Purpose: Round 2 sweep — looser P3 lockbox + no-lockbox baseline + aggressive P4.
Status:  One-shot experiment script.
Future:  Remove after analysis is complete.
'''
import sys
import os
import json
import contextlib
import time
from pathlib import Path

_conda_bin = r"D:\ProgramData\anaconda3\envs\calibration_07032026\Library\bin"
os.environ["PATH"] = _conda_bin + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import cv2

sys.path.insert(0, r"D:\Work\coding\reconstruction\pyCamSet")

from pyCamSet import load_CameraSet
from pyCamSet.utils.saving import load_pickle
from pyCamSet.gui.shared_functions import make_run_id, build_target, extract_detection
from pyCamSet.optimisation.camera_lockbox import CameraLockboxConfig
from pyCamSet.optimisation.template_handler import TemplateBundleHandler
from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler
from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment_with_stats

WS = Path(
    r"E:\M_Nebula\1 Data\2 Pilot Data\experiment_15052026"
    r"\calibration_ccube_images\filtered-per-image_reduced_org_by_cam_blue_calib"
    r"\.pycamset_workspace"
)
F_LOC = str(WS.parent)
PRIOR_DIR = WS / "lockbox_priors" / "20260701_204308"
LOCKBOX_SOURCE = str(PRIOR_DIR / "edited_lockbox_source.camset")
PHASE2_CAMSET = str(WS / "phase2_runs" / "20260628_171228_30481f" / "initial_cameras.camset")
PHASE1_PICKLE = str(WS / "phase1_runs" / "20260628_171157_f1445a" / "detected_datapoints.pickle")


class EmitStream:
    def __init__(self):
        self.log_list = []
        self.buffer = ""
    def write(self, text):
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                self.log_list.append(line)
    def flush(self):
        if self.buffer.strip():
            self.log_list.append(self.buffer)
            self.buffer = ""
    def fileno(self):
        raise OSError("no fileno")


def compute_rpe(handler, optimisation):
    per_cam = {}
    try:
        dd = np.asarray(handler.get_detection_data(flatten=True))
        res_xy = np.reshape(np.asarray(optimisation.fun, dtype=float), (-1, 2))
        res_norm = np.linalg.norm(res_xy, axis=1)
        if dd.ndim == 2 and dd.shape[1] >= 1:
            cam_idx = dd[:, 0].astype(int)
            if cam_idx.size != res_norm.size:
                n = min(cam_idx.size, res_norm.size)
                cam_idx, res_norm = cam_idx[:n], res_norm[:n]
            for idx, name in enumerate(handler.cam_names):
                mask = cam_idx == idx
                per_cam[name] = float(np.mean(res_norm[mask])) if np.any(mask) else float("nan")
    except Exception:
        pass
    return per_cam


def run_p3(lockbox_params, problem_options, use_lockbox=True):
    cams = load_CameraSet(PHASE2_CAMSET)
    payload = load_pickle(PHASE1_PICKLE)
    detections = extract_detection(payload)
    target = build_target("Ccube", 6, 10.0)

    run_id = make_run_id()
    run_dir = WS / "phase3_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    lockbox_config = CameraLockboxConfig(enabled=False)
    lockbox_src = None
    if use_lockbox:
        lockbox_config = CameraLockboxConfig(
            enabled=True,
            rotation_half_width=float(lockbox_params["rotation_half_width"]),
            translation_half_width=float(lockbox_params["translation_half_width"]),
            rotation_sigma=float(lockbox_params["rotation_sigma"]),
            translation_sigma=float(lockbox_params["translation_sigma"]),
            center_sigma=float(lockbox_params["center_sigma"]),
        )
        lockbox_src = load_CameraSet(LOCKBOX_SOURCE)

    handler = TemplateBundleHandler(
        camset=cams, target=target, detection=detections,
        fixed_params=None, options=problem_options,
        lockbox_config=lockbox_config,
        lockbox_source_camset=lockbox_src,
        lockbox_warm_start=True,
    )

    stream = EmitStream()
    t0 = time.time()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        optimisation, out_cams, stats = run_bundle_adjustment_with_stats(handler, threads=1)
    elapsed = time.time() - t0

    camset_path = run_dir / "optimised_cameras.camset"
    out_cams.save(str(camset_path))

    per_cam = compute_rpe(handler, optimisation)

    # Camera 4 position check
    ext4 = np.array(out_cams["camera_4"].extrinsic)
    C4 = -(ext4[:3, :3].T @ ext4[:3, 3])
    rv4, _ = cv2.Rodrigues(ext4[:3, :3])

    # Save metadata
    lb_meta = {"enabled": False} if not use_lockbox else {**lockbox_params, "enabled": True,
        "source_camset": LOCKBOX_SOURCE, "edited_source_camset": LOCKBOX_SOURCE,
        "warm_start": True, "implementation_mode": "extrinsic_parameter_mvp"}
    meta = {
        "run_id": run_id, "phase": "phase3",
        "params": {"f_loc": F_LOC, "threads": 1, "fixed_params": None,
                    "target_type": "Ccube", "n_points": 6, "length": 10.0,
                    "selected_cameras": [f"camera_{i}" for i in range(1, 9)],
                    "lockbox": lb_meta, "problem_options": problem_options},
        "diagnostics": {
            "D3.5_initial_euclid_px": float(stats.get("initial_euclid", float("nan"))),
            "D3.6_final_euclid_px": float(stats.get("final_euclid", float("nan"))),
            "D3.9_nfev": int(stats.get("nfev", 0)),
            "D3.12_per_camera_mean_reprojection": per_cam,
        },
        "error": None,
        "inputs": {"phase2_run_id": "20260628_171228_30481f", "phase1_run_id": "20260628_171157_f1445a"},
        "artifacts": {"optimised_camset": str(camset_path),
                       "phase2_initial_camset_used": PHASE2_CAMSET,
                       "phase1_detection_pickle_used": PHASE1_PICKLE},
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    return {
        "run_id": run_id, "camset_path": str(camset_path),
        "per_cam_rpe": per_cam,
        "final_euclid": float(stats.get("final_euclid", float("nan"))),
        "nfev": int(stats.get("nfev", 0)),
        "success": bool(stats.get("success", False)),
        "cam4_C": C4.tolist(), "cam4_C_norm": float(np.linalg.norm(C4)),
        "cam4_rvec_norm": float(np.linalg.norm(rv4)),
        "elapsed": elapsed,
    }


def run_p4(p3_camset_path, p3_run_id, problem_options):
    prev_cams = load_CameraSet(p3_camset_path)
    prev_handler = getattr(prev_cams, "calibration_handler", None)
    if prev_handler is None:
        raise RuntimeError("P3 camset has no calibration handler.")

    run_id = make_run_id()
    run_dir = WS / "phase4_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    handler = SelfBundleHandler(
        camset=prev_cams, target=prev_handler.target, detection=prev_handler.detection,
        fixed_params=None, options=problem_options,
    )
    handler.set_from_templated_camset(prev_cams)

    stream = EmitStream()
    t0 = time.time()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        optimisation, out_cams, stats = run_bundle_adjustment_with_stats(handler, threads=1)
    elapsed = time.time() - t0

    out_path = run_dir / "self_calibrated_cameras.camset"
    out_cams.save(str(out_path))

    per_cam = compute_rpe(handler, optimisation)

    updated_target = np.array(handler.get_updated_target(optimisation.x), dtype=float)
    ref_target = np.array(handler.target.point_data, dtype=float).reshape(-1, 3)
    displacement = np.linalg.norm(updated_target - ref_target, axis=1)
    mean_disp_mm = float(np.nanmean(displacement) * 1000.0)
    visible = np.array(getattr(handler, "visible_feature_mask", []), dtype=bool)
    n_free = int(np.sum(visible))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.linalg.norm(ref_target, axis=1) / np.where(np.linalg.norm(updated_target, axis=1) == 0, np.nan, 1)
    scale_est = float(np.nanmedian(ratio)) if ratio.size else float("nan")

    cam_positions = {}
    for name in sorted(out_cams.get_names()):
        ext = np.array(out_cams[name].extrinsic)
        C = -(ext[:3, :3].T @ ext[:3, 3])
        rv, _ = cv2.Rodrigues(ext[:3, :3])
        cam_positions[name] = {"|C|": float(np.linalg.norm(C)), "|rvec|": float(np.linalg.norm(rv))}

    meta = {
        "run_id": run_id, "phase": "phase4",
        "params": {"f_loc": F_LOC, "threads": 1, "fixed_params": None,
                    "selected_cameras": [f"camera_{i}" for i in range(1, 9)],
                    "problem_options": problem_options},
        "diagnostics": {
            "D4.1_n_free_target_points": n_free,
            "D4.3_initial_euclid_px": float(stats.get("initial_euclid", float("nan"))),
            "D4.3_final_euclid_px": float(stats.get("final_euclid", float("nan"))),
            "D4.5_gauge_scale_factor": scale_est,
            "D4.7_mean_target_displacement_mm": mean_disp_mm,
            "D4.12_per_camera_mean_reprojection": per_cam,
        },
        "error": None, "inputs": {"phase3_run_id": p3_run_id},
        "artifacts": {"self_calibrated_camset": str(out_path), "optimised_camset": str(out_path),
                       "phase3_camset_used": p3_camset_path},
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    return {
        "run_id": run_id, "per_cam_rpe": per_cam,
        "final_euclid": float(stats.get("final_euclid", float("nan"))),
        "nfev": int(stats.get("nfev", 0)), "success": bool(stats.get("success", False)),
        "mean_disp_mm": mean_disp_mm, "n_free_points": n_free,
        "gauge_scale": scale_est, "cam_positions": cam_positions, "elapsed": elapsed,
    }


P3_OPTS = {"verbosity": 0, "fixed_pose": 0, "ref_cam": 0, "ref_pose": 0, "outliers": "y", "max_nfev": 1000}

# Round 2 P3 sweep — much looser rotation, also no-lockbox baseline
P3_RUNS = [
    # No lockbox baseline — best P3 quality, but camera 4 may drift
    {"label": "NL", "use_lockbox": False, "params": {}},
    # Very loose lockbox — position hold only, rotation free
    {"label": "L1", "use_lockbox": True, "params": {"rotation_half_width": 1.0, "rotation_sigma": 1.0, "center_sigma": 0.1, "translation_half_width": 0.1, "translation_sigma": 0.01}},
    # Loose rotation, tight centre
    {"label": "L2", "use_lockbox": True, "params": {"rotation_half_width": 0.5, "rotation_sigma": 0.5, "center_sigma": 0.05, "translation_half_width": 0.1, "translation_sigma": 0.01}},
    # Best from round 1 for reference
    {"label": "L3", "use_lockbox": True, "params": {"rotation_half_width": 0.3, "rotation_sigma": 0.05, "center_sigma": 0.3, "translation_half_width": 0.1, "translation_sigma": 0.01}},
]

# P4 sweep from each P3 — soft_l1 with high nfev
P4_OPTS_BASE = {"verbosity": 0, "fixed_pose": 0, "ref_cam": 0, "ref_pose": 0, "outliers": "y"}
P4_RUNS = [
    {"label": "lin-200", "max_nfev": 200, "loss": "linear"},
    {"label": "sl1-200", "max_nfev": 200, "loss": "soft_l1"},
    {"label": "sl1-500", "max_nfev": 500, "loss": "soft_l1"},
]


def main():
    print("=" * 70)
    print("ROUND 2: Aggressive P3 + P4 Sweep")
    print("=" * 70)

    all_p3 = []
    all_p4 = []

    for p3_cfg in P3_RUNS:
        label = p3_cfg["label"]
        print(f"\n--- P3 {label} (lockbox={p3_cfg['use_lockbox']}) ---")
        try:
            r = run_p3(p3_cfg.get("params", {}), P3_OPTS.copy(), use_lockbox=p3_cfg["use_lockbox"])
            c4 = r["per_cam_rpe"].get("camera_4", float("nan"))
            mr = np.mean(list(r["per_cam_rpe"].values()))
            mx = max(r["per_cam_rpe"].values())
            print(f"  cam4={c4:.2f}  mean={mr:.2f}  max={mx:.2f}  euclid={r['final_euclid']:.2f}  nfev={r['nfev']}  |C4|={r['cam4_C_norm']:.4f}  |rv4|={r['cam4_rvec_norm']:.4f}  ({r['elapsed']:.1f}s)")
            r["label"] = label
            all_p3.append(r)

            # Run P4 from this P3
            for p4_cfg in P4_RUNS:
                opts = P4_OPTS_BASE.copy()
                opts["max_nfev"] = p4_cfg["max_nfev"]
                opts["loss"] = p4_cfg["loss"]
                opts["f_scale"] = 1.0
                plabel = f"{label}/{p4_cfg['label']}"
                print(f"  --- P4 {plabel} ---")
                try:
                    p4r = run_p4(r["camset_path"], r["run_id"], opts)
                    c4p = p4r["per_cam_rpe"].get("camera_4", float("nan"))
                    mrp = np.mean(list(p4r["per_cam_rpe"].values()))
                    mxp = max(p4r["per_cam_rpe"].values())
                    print(f"    cam4={c4p:.2f}  mean={mrp:.2f}  max={mxp:.2f}  disp={p4r['mean_disp_mm']:.4f}mm  scale={p4r['gauge_scale']:.4f}  nfev={p4r['nfev']}  ({p4r['elapsed']:.1f}s)")
                    p4r["label"] = plabel
                    p4r["p3_label"] = label
                    all_p4.append(p4r)
                except Exception as exc:
                    print(f"    ERROR: {exc}")
                    all_p4.append({"label": plabel, "p3_label": label, "error": str(exc)})
        except Exception as exc:
            print(f"  ERROR: {exc}")
            all_p3.append({"label": label, "error": str(exc)})

    # Summary
    print("\n" + "=" * 70)
    print("FULL SUMMARY")
    print("=" * 70)

    print("\n--- P3 Results ---")
    print(f"{'Label':>5} {'cam4':>7} {'mean':>7} {'max':>7} {'euclid':>8} {'nfev':>5} {'|C4|':>7} {'|rv4|':>7}")
    for r in all_p3:
        if "error" in r:
            print(f"{r['label']:>5}  ERROR")
        else:
            c4 = r["per_cam_rpe"].get("camera_4", float("nan"))
            mr = np.mean(list(r["per_cam_rpe"].values()))
            mx = max(r["per_cam_rpe"].values())
            print(f"{r['label']:>5} {c4:7.2f} {mr:7.2f} {mx:7.2f} {r['final_euclid']:8.2f} {r['nfev']:5d} {r['cam4_C_norm']:7.4f} {r['cam4_rvec_norm']:7.4f}")

    print("\n--- P4 Results ---")
    print(f"{'Label':>16} {'cam4':>7} {'mean':>7} {'max':>7} {'disp_mm':>8} {'scale':>7} {'nfev':>5}")
    for r in all_p4:
        if "error" in r:
            print(f"{r['label']:>16}  ERROR")
        else:
            c4 = r["per_cam_rpe"].get("camera_4", float("nan"))
            mr = np.mean(list(r["per_cam_rpe"].values()))
            mx = max(r["per_cam_rpe"].values())
            print(f"{r['label']:>16} {c4:7.2f} {mr:7.2f} {mx:7.2f} {r['mean_disp_mm']:8.4f} {r['gauge_scale']:7.4f} {r['nfev']:5d}")

    # Winner
    valid = [r for r in all_p4 if "error" not in r and r["mean_disp_mm"] < 1.0]
    if valid:
        winner = min(valid, key=lambda r: np.mean(list(r["per_cam_rpe"].values())))
        print(f"\n=== WINNER: {winner['label']} (P4 run {winner['run_id']}) ===")
        print(f"  mean RPE: {np.mean(list(winner['per_cam_rpe'].values())):.4f} px")
        print(f"  cam4 RPE: {winner['per_cam_rpe']['camera_4']:.4f} px")
        print(f"  max RPE: {max(winner['per_cam_rpe'].values()):.4f} px")
        print(f"  displacement: {winner['mean_disp_mm']:.4f} mm")
        print(f"  free pts: {winner['n_free_points']}")
        print(f"  Stretch goal (< 1.0 px mean): {'ACHIEVED' if np.mean(list(winner['per_cam_rpe'].values())) < 1.0 else 'NOT YET'}")

        summary = {
            "best_p4_label": winner["label"],
            "best_p4_run_id": winner["run_id"],
            "best_p4_mean_rpe": float(np.mean(list(winner["per_cam_rpe"].values()))),
            "best_p4_cam4_rpe": winner["per_cam_rpe"]["camera_4"],
            "best_p4_max_rpe": float(max(winner["per_cam_rpe"].values())),
            "best_p4_displacement_mm": winner["mean_disp_mm"],
            "best_p4_free_pts": winner["n_free_points"],
            "stretch_goal_achieved": float(np.mean(list(winner["per_cam_rpe"].values()))) < 1.0,
        }
        Path(r"D:\Work\coding\reconstruction\pyCamSet\tools\sweep_round2_results.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()