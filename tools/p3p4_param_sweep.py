'''
Purpose: Parameterised P3→P4 pipeline sweep for RPE minimisation.
         Runs Stage A (P3 lockbox params), Stage B (P4 max_nfev), and
         Stage C (P4 loss function) sequentially, recording all metrics.
Status:  One-shot experiment script.
Future:  Remove after analysis is complete.
'''
import sys
import os
import json
import contextlib
import time
from pathlib import Path
from datetime import datetime

# Cairo DLL path fix
_conda_bin = r"D:\ProgramData\anaconda3\envs\calibration_07032026\Library\bin"
os.environ["PATH"] = _conda_bin + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import cv2

sys.path.insert(0, r"D:\Work\coding\reconstruction\pyCamSet")

from pyCamSet import load_CameraSet
from pyCamSet.utils.saving import load_pickle
from pyCamSet.gui.shared_functions import (
    make_run_id,
    build_target,
    extract_detection,
)
from pyCamSet.optimisation.camera_lockbox import CameraLockboxConfig
from pyCamSet.optimisation.template_handler import TemplateBundleHandler
from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler
from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment_with_stats

# ── Paths ──
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

# Shared inputs loaded once
_cams = None
_detections = None
_target = None
_lockbox_source = None


def _load_shared():
    """Load fresh copies every call — handler mutates camset in place."""
    cams = load_CameraSet(PHASE2_CAMSET)
    payload = load_pickle(PHASE1_PICKLE)
    detections = extract_detection(payload)
    target = build_target("Ccube", 6, 10.0)
    lockbox_src = load_CameraSet(LOCKBOX_SOURCE)
    return cams, detections, target, lockbox_src


class EmitStream:
    def __init__(self, log_list):
        self.log_list = log_list
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


def run_p3(lockbox_params, problem_options, log_lines):
    """Run a single P3 bundle adjustment and return (metadata, camset_path)."""
    cams, detections, target, lockbox_src = _load_shared()

    run_id = make_run_id()
    run_dir = WS / "phase3_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    lockbox_config = CameraLockboxConfig(
        enabled=True,
        rotation_half_width=float(lockbox_params["rotation_half_width"]),
        translation_half_width=float(lockbox_params["translation_half_width"]),
        rotation_sigma=float(lockbox_params["rotation_sigma"]),
        translation_sigma=float(lockbox_params["translation_sigma"]),
        center_sigma=float(lockbox_params["center_sigma"]),
    )

    handler = TemplateBundleHandler(
        camset=cams,
        target=target,
        detection=detections,
        fixed_params=None,
        options=problem_options,
        lockbox_config=lockbox_config,
        lockbox_source_camset=lockbox_src,
        lockbox_warm_start=True,
    )

    stream = EmitStream(log_lines)
    t0 = time.time()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        optimisation, out_cams, stats = run_bundle_adjustment_with_stats(handler, threads=1)
    elapsed = time.time() - t0

    camset_path = run_dir / "optimised_cameras.camset"
    out_cams.save(str(camset_path))

    # Per-camera RPE
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

    params_record = {
        "f_loc": F_LOC, "threads": 1, "fixed_params": None,
        "target_type": "Ccube", "n_points": 6, "length": 10.0,
        "selected_cameras": [f"camera_{i}" for i in range(1, 9)],
        "lockbox": {**lockbox_params, "enabled": True,
                     "original_source_camset": None,
                     "edited_source_camset": LOCKBOX_SOURCE,
                     "source_camset": LOCKBOX_SOURCE,
                     "implementation_mode": "extrinsic_parameter_mvp",
                     "warm_start": True},
        "problem_options": problem_options,
    }

    metadata = {
        "run_id": run_id, "phase": "phase3", "params": params_record,
        "diagnostics": {
            "D3.5_initial_euclid_px": float(stats.get("initial_euclid", float("nan"))),
            "D3.6_final_euclid_px": float(stats.get("final_euclid", float("nan"))),
            "D3.9_nfev": int(stats.get("nfev", 0)),
            "D3.12_per_camera_mean_reprojection": per_cam,
        },
        "error": None,
        "inputs": {"phase2_run_id": "20260628_171228_30481f",
                    "phase1_run_id": "20260628_171157_f1445a"},
        "artifacts": {"optimised_camset": str(camset_path),
                       "phase2_initial_camset_used": PHASE2_CAMSET,
                       "phase1_detection_pickle_used": PHASE1_PICKLE},
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    return {
        "run_id": run_id, "camset_path": str(camset_path),
        "per_cam_rpe": per_cam,
        "final_euclid": float(stats.get("final_euclid", float("nan"))),
        "nfev": int(stats.get("nfev", 0)),
        "success": bool(stats.get("success", False)),
        "elapsed": elapsed,
        "metadata": metadata,
    }


def run_p4(p3_camset_path, p3_run_id, problem_options, log_lines):
    """Run a single P4 self-calibration and return metrics."""
    prev_cams = load_CameraSet(p3_camset_path)
    prev_handler = getattr(prev_cams, "calibration_handler", None)
    if prev_handler is None:
        raise RuntimeError("P3 camset has no calibration handler.")

    run_id = make_run_id()
    run_dir = WS / "phase4_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    handler = SelfBundleHandler(
        camset=prev_cams,
        target=prev_handler.target,
        detection=prev_handler.detection,
        fixed_params=None,
        options=problem_options,
    )
    handler.set_from_templated_camset(prev_cams)

    stream = EmitStream(log_lines)
    t0 = time.time()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        optimisation, out_cams, stats = run_bundle_adjustment_with_stats(handler, threads=1)
    elapsed = time.time() - t0

    out_path = run_dir / "self_calibrated_cameras.camset"
    out_cams.save(str(out_path))

    # Per-camera RPE
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

    # Target displacement and scale
    updated_target = np.array(handler.get_updated_target(optimisation.x), dtype=float)
    ref_target = np.array(handler.target.point_data, dtype=float).reshape(-1, 3)
    displacement = np.linalg.norm(updated_target - ref_target, axis=1)
    mean_disp_mm = float(np.nanmean(displacement) * 1000.0) if displacement.size else float("nan")
    visible = np.array(getattr(handler, "visible_feature_mask", []), dtype=bool)
    n_free = int(np.sum(visible))

    # Gauge scale
    with np.errstate(invalid="ignore", divide="ignore"):
        ref_norm = np.linalg.norm(ref_target, axis=1)
        upd_norm = np.linalg.norm(updated_target, axis=1)
        ratio = ref_norm / np.where(upd_norm == 0.0, np.nan, upd_norm)
    scale_est = float(np.nanmedian(ratio)) if ratio.size else float("nan")

    # Camera positions
    cam_positions = {}
    for name in sorted(out_cams.get_names()):
        ext = np.array(out_cams[name].extrinsic)
        C = -(ext[:3, :3].T @ ext[:3, 3])
        rv, _ = cv2.Rodrigues(ext[:3, :3])
        cam_positions[name] = {
            "|C|": float(np.linalg.norm(C)),
            "|rvec|": float(np.linalg.norm(rv)),
        }

    params_record = {
        "f_loc": F_LOC, "threads": 1, "fixed_params": None,
        "selected_cameras": [f"camera_{i}" for i in range(1, 9)],
        "problem_options": problem_options,
    }

    metadata = {
        "run_id": run_id, "phase": "phase4", "params": params_record,
        "diagnostics": {
            "D4.1_n_free_target_points": n_free,
            "D4.3_initial_euclid_px": float(stats.get("initial_euclid", float("nan"))),
            "D4.3_final_euclid_px": float(stats.get("final_euclid", float("nan"))),
            "D4.5_gauge_scale_factor": scale_est,
            "D4.7_mean_target_displacement_mm": mean_disp_mm,
            "D4.12_per_camera_mean_reprojection": per_cam,
        },
        "error": None,
        "inputs": {"phase3_run_id": p3_run_id},
        "artifacts": {
            "self_calibrated_camset": str(out_path),
            "optimised_camset": str(out_path),
            "phase3_camset_used": p3_camset_path,
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    return {
        "run_id": run_id,
        "per_cam_rpe": per_cam,
        "final_euclid": float(stats.get("final_euclid", float("nan"))),
        "nfev": int(stats.get("nfev", 0)),
        "success": bool(stats.get("success", False)),
        "mean_disp_mm": mean_disp_mm,
        "n_free_points": n_free,
        "gauge_scale": scale_est,
        "cam_positions": cam_positions,
        "elapsed": elapsed,
    }


# ── Sweep definitions ──

P3_BASE_OPTS = {
    "verbosity": 0, "fixed_pose": 0, "ref_cam": 0, "ref_pose": 0,
    "outliers": "y", "max_nfev": 1000,
}

P3_SWEEP = [
    {"label": "A1", "rotation_half_width": 0.3, "rotation_sigma": 0.05, "center_sigma": 0.3,
     "translation_half_width": 0.1, "translation_sigma": 0.01},
    {"label": "A2", "rotation_half_width": 0.3, "rotation_sigma": 0.05, "center_sigma": 0.1,
     "translation_half_width": 0.1, "translation_sigma": 0.01},
    {"label": "A3", "rotation_half_width": 0.5, "rotation_sigma": 0.05, "center_sigma": 0.3,
     "translation_half_width": 0.1, "translation_sigma": 0.01},
    {"label": "A4", "rotation_half_width": 0.3, "rotation_sigma": 0.02, "center_sigma": 0.3,
     "translation_half_width": 0.1, "translation_sigma": 0.01},
    {"label": "A5", "rotation_half_width": 0.3, "rotation_sigma": 0.10, "center_sigma": 0.3,
     "translation_half_width": 0.1, "translation_sigma": 0.01},
    {"label": "A6", "rotation_half_width": 0.5, "rotation_sigma": 0.10, "center_sigma": 0.3,
     "translation_half_width": 0.1, "translation_sigma": 0.01},
]

P4_BASE_OPTS = {
    "verbosity": 0, "fixed_pose": 0, "ref_cam": 0, "ref_pose": 0,
    "outliers": "y", "max_nfev": 30,
}

# Stage B: max_nfev sweep
P4_NFEV_SWEEP = [100, 200, 500, 1000]

# Stage C: loss function sweep
P4_LOSS_SWEEP = [
    {"loss": "soft_l1", "f_scale": 1.0, "max_nfev": 200},
    {"loss": "huber", "f_scale": 1.0, "max_nfev": 200},
]


def main():
    all_logs = []

    # ── Stage A: P3 sweep ──
    print("=" * 60)
    print("STAGE A: P3 Lockbox Parameter Sweep")
    print("=" * 60)

    p3_results = []
    for cfg in P3_SWEEP:
        label = cfg.pop("label")
        print(f"\n--- {label}: rhw={cfg['rotation_half_width']} rs={cfg['rotation_sigma']} cs={cfg['center_sigma']} ---")
        logs = []
        try:
            result = run_p3(cfg, P3_BASE_OPTS.copy(), logs)
            c4 = result["per_cam_rpe"].get("camera_4", float("nan"))
            mean_rpe = np.mean(list(result["per_cam_rpe"].values()))
            max_rpe = max(result["per_cam_rpe"].values())
            print(f"  cam4={c4:.2f}  mean={mean_rpe:.2f}  max={max_rpe:.2f}  euclid={result['final_euclid']:.2f}  nfev={result['nfev']}  ({result['elapsed']:.1f}s)")
            result["label"] = label
            result["config"] = cfg
            p3_results.append(result)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            p3_results.append({"label": label, "error": str(exc), "config": cfg})

    # Select best P3
    valid_p3 = [r for r in p3_results if "error" not in r]
    if not valid_p3:
        print("\nAll P3 runs failed!")
        return

    best_p3 = min(valid_p3, key=lambda r: r["per_cam_rpe"].get("camera_4", 999))
    print(f"\n=== Best P3: {best_p3['label']} (run {best_p3['run_id']}) ===")
    print(f"  cam4={best_p3['per_cam_rpe']['camera_4']:.4f}  mean={np.mean(list(best_p3['per_cam_rpe'].values())):.4f}")

    best_p3_path = best_p3["camset_path"]
    best_p3_id = best_p3["run_id"]

    # ── Stage B: P4 max_nfev sweep ──
    print("\n" + "=" * 60)
    print(f"STAGE B: P4 max_nfev Sweep (from P3 {best_p3['label']})")
    print("=" * 60)

    p4_results = []
    for nfev in P4_NFEV_SWEEP:
        opts = P4_BASE_OPTS.copy()
        opts["max_nfev"] = nfev
        print(f"\n--- max_nfev={nfev} ---")
        logs = []
        try:
            result = run_p4(best_p3_path, best_p3_id, opts, logs)
            c4 = result["per_cam_rpe"].get("camera_4", float("nan"))
            mean_rpe = np.mean(list(result["per_cam_rpe"].values()))
            max_rpe = max(result["per_cam_rpe"].values())
            print(f"  cam4={c4:.2f}  mean={mean_rpe:.2f}  max={max_rpe:.2f}  disp={result['mean_disp_mm']:.4f}mm  scale={result['gauge_scale']:.4f}  nfev={result['nfev']}  success={result['success']}  ({result['elapsed']:.1f}s)")
            result["label"] = f"B-nfev{nfev}"
            p4_results.append(result)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            p4_results.append({"label": f"B-nfev{nfev}", "error": str(exc)})

    # Select best P4 from Stage B
    valid_p4 = [r for r in p4_results if "error" not in r and r["mean_disp_mm"] < 1.0]
    if valid_p4:
        best_p4 = min(valid_p4, key=lambda r: np.mean(list(r["per_cam_rpe"].values())))
        print(f"\n=== Best P4 (Stage B): {best_p4['label']} (run {best_p4['run_id']}) ===")
        print(f"  mean={np.mean(list(best_p4['per_cam_rpe'].values())):.4f}  cam4={best_p4['per_cam_rpe']['camera_4']:.4f}  disp={best_p4['mean_disp_mm']:.4f}")
    else:
        best_p4 = None
        print("\nNo valid P4 from Stage B!")

    # ── Stage C: P4 loss function sweep ──
    print("\n" + "=" * 60)
    print(f"STAGE C: P4 Loss Function Sweep (from P3 {best_p3['label']})")
    print("=" * 60)

    for loss_cfg in P4_LOSS_SWEEP:
        opts = P4_BASE_OPTS.copy()
        opts.update(loss_cfg)
        label = f"C-{loss_cfg['loss']}"
        print(f"\n--- {label} (max_nfev={loss_cfg['max_nfev']}) ---")
        logs = []
        try:
            result = run_p4(best_p3_path, best_p3_id, opts, logs)
            c4 = result["per_cam_rpe"].get("camera_4", float("nan"))
            mean_rpe = np.mean(list(result["per_cam_rpe"].values()))
            max_rpe = max(result["per_cam_rpe"].values())
            print(f"  cam4={c4:.2f}  mean={mean_rpe:.2f}  max={max_rpe:.2f}  disp={result['mean_disp_mm']:.4f}mm  scale={result['gauge_scale']:.4f}  nfev={result['nfev']}  success={result['success']}  ({result['elapsed']:.1f}s)")
            result["label"] = label
            p4_results.append(result)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            p4_results.append({"label": label, "error": str(exc)})

    # ── Summary ──
    print("\n" + "=" * 60)
    print("FULL SUMMARY")
    print("=" * 60)

    print("\n--- P3 Results ---")
    print(f"{'Label':>5} {'cam4':>7} {'mean':>7} {'max':>7} {'euclid':>8} {'nfev':>5}")
    for r in p3_results:
        if "error" in r:
            print(f"{r['label']:>5}  ERROR: {r['error'][:40]}")
        else:
            c4 = r["per_cam_rpe"].get("camera_4", float("nan"))
            mr = np.mean(list(r["per_cam_rpe"].values()))
            mx = max(r["per_cam_rpe"].values())
            print(f"{r['label']:>5} {c4:7.2f} {mr:7.2f} {mx:7.2f} {r['final_euclid']:8.2f} {r['nfev']:5d}")

    print("\n--- P4 Results ---")
    print(f"{'Label':>12} {'cam4':>7} {'mean':>7} {'max':>7} {'disp_mm':>8} {'scale':>7} {'nfev':>5} {'success':>8}")
    for r in p4_results:
        if "error" in r:
            print(f"{r['label']:>12}  ERROR: {r['error'][:40]}")
        else:
            c4 = r["per_cam_rpe"].get("camera_4", float("nan"))
            mr = np.mean(list(r["per_cam_rpe"].values()))
            mx = max(r["per_cam_rpe"].values())
            print(f"{r['label']:>12} {c4:7.2f} {mr:7.2f} {mx:7.2f} {r['mean_disp_mm']:8.4f} {r['gauge_scale']:7.4f} {r['nfev']:5d} {str(r['success']):>8}")

    # Best overall
    all_valid_p4 = [r for r in p4_results if "error" not in r and r["mean_disp_mm"] < 1.0]
    if all_valid_p4:
        winner = min(all_valid_p4, key=lambda r: np.mean(list(r["per_cam_rpe"].values())))
        print(f"\n=== WINNER: {winner['label']} (P4 run {winner['run_id']}) ===")
        print(f"  P3: {best_p3['label']} (run {best_p3['run_id']})")
        print(f"  P4 mean RPE: {np.mean(list(winner['per_cam_rpe'].values())):.4f} px")
        print(f"  P4 cam4 RPE: {winner['per_cam_rpe']['camera_4']:.4f} px")
        print(f"  P4 max RPE: {max(winner['per_cam_rpe'].values()):.4f} px")
        print(f"  D4.7 displacement: {winner['mean_disp_mm']:.4f} mm")
        print(f"  D4.5 scale: {winner['gauge_scale']:.4f}")
        print(f"  D4.1 free pts: {winner['n_free_points']}")
        print(f"  Stretch goal (< 1.0 px mean): {'ACHIEVED' if np.mean(list(winner['per_cam_rpe'].values())) < 1.0 else 'NOT YET'}")

        # Save winner summary
        summary = {
            "best_p3_label": best_p3["label"],
            "best_p3_run_id": best_p3["run_id"],
            "best_p3_config": best_p3.get("config", {}),
            "best_p3_cam4_rpe": best_p3["per_cam_rpe"]["camera_4"],
            "best_p3_mean_rpe": float(np.mean(list(best_p3["per_cam_rpe"].values()))),
            "best_p4_label": winner["label"],
            "best_p4_run_id": winner["run_id"],
            "best_p4_mean_rpe": float(np.mean(list(winner["per_cam_rpe"].values()))),
            "best_p4_cam4_rpe": winner["per_cam_rpe"]["camera_4"],
            "best_p4_max_rpe": float(max(winner["per_cam_rpe"].values())),
            "best_p4_displacement_mm": winner["mean_disp_mm"],
            "best_p4_scale": winner["gauge_scale"],
            "best_p4_free_pts": winner["n_free_points"],
            "stretch_goal_achieved": float(np.mean(list(winner["per_cam_rpe"].values()))) < 1.0,
        }
        summary_path = Path(r"D:\Work\coding\reconstruction\pyCamSet\tools\sweep_results.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()