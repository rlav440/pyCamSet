'''
Purpose: Round 3 — Push P4 below 1 px with tighter convergence + xtol sweep.
         Uses the best P3 from round 1 (A1 = f077d8, cam4=9.89) as input.
         Tests: xtol tightening, max_nfev=2000, soft_l1 with small f_scale.
Status:  One-shot experiment script.
Future:  Remove after analysis is complete.
'''
import sys, os, json, contextlib, time
from pathlib import Path

_conda_bin = r"D:\ProgramData\anaconda3\envs\calibration_07032026\Library\bin"
os.environ["PATH"] = _conda_bin + os.pathsep + os.environ.get("PATH", "")

import numpy as np, cv2

sys.path.insert(0, r"D:\Work\coding\reconstruction\pyCamSet")

from pyCamSet import load_CameraSet
from pyCamSet.gui.shared_functions import make_run_id
from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler
from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment_with_stats

WS = Path(
    r"E:\M_Nebula\1 Data\2 Pilot Data\experiment_15052026"
    r"\calibration_ccube_images\filtered-per-image_reduced_org_by_cam_blue_calib"
    r"\.pycamset_workspace"
)

# Best P3 from round 1 sweep: A1 (f077d8) — cam4=9.89, mean=8.10
# Also try the original no-lockbox P3 (949b8a) — cam4=6.99, mean=7.48
P3_INPUTS = [
    ("A1", str(WS / "phase3_runs" / "20260701_220704_f077d8" / "optimised_cameras.camset"), "20260701_220704_f077d8"),
    ("NL", str(WS / "phase3_runs" / "20260628_171246_949b8a" / "optimised_cameras.camset"), "20260628_171246_949b8a"),
]

# P4 configs to try
P4_CONFIGS = [
    # Tighter xtol
    {"label": "sl1-xtol1e6", "max_nfev": 2000, "loss": "soft_l1", "f_scale": 1.0, "xtol": 1e-6},
    # Even tighter
    {"label": "sl1-xtol1e8", "max_nfev": 2000, "loss": "soft_l1", "f_scale": 1.0, "xtol": 1e-8},
    # Smaller f_scale — more aggressive outlier suppression
    {"label": "sl1-fs0.5", "max_nfev": 2000, "loss": "soft_l1", "f_scale": 0.5, "xtol": 1e-6},
    # Linear loss but very tight xtol and high nfev
    {"label": "lin-xtol1e8", "max_nfev": 2000, "loss": "linear", "f_scale": 1.0, "xtol": 1e-8},
]

P4_BASE = {"verbosity": 0, "fixed_pose": 0, "ref_cam": 0, "ref_pose": 0, "outliers": "y"}


class EmitStream:
    def __init__(self):
        self.buffer = ""
    def write(self, text):
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
    def flush(self):
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


def run_p4(p3_path, p3_id, opts):
    prev_cams = load_CameraSet(p3_path)
    prev_handler = getattr(prev_cams, "calibration_handler", None)
    if prev_handler is None:
        raise RuntimeError("No calibration handler")

    run_id = make_run_id()
    run_dir = WS / "phase4_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    handler = SelfBundleHandler(
        camset=prev_cams, target=prev_handler.target, detection=prev_handler.detection,
        fixed_params=None, options=opts,
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
    disp = np.linalg.norm(updated_target - ref_target, axis=1)
    mean_disp_mm = float(np.nanmean(disp) * 1000.0)
    visible = np.array(getattr(handler, "visible_feature_mask", []), dtype=bool)
    n_free = int(np.sum(visible))

    meta = {
        "run_id": run_id, "phase": "phase4",
        "params": {"problem_options": opts},
        "diagnostics": {
            "D4.1_n_free_target_points": n_free,
            "D4.3_initial_euclid_px": float(stats.get("initial_euclid", float("nan"))),
            "D4.3_final_euclid_px": float(stats.get("final_euclid", float("nan"))),
            "D4.7_mean_target_displacement_mm": mean_disp_mm,
            "D4.12_per_camera_mean_reprojection": per_cam,
            "solver_status": stats.get("status"),
            "solver_message": stats.get("message"),
            "solver_success": stats.get("success"),
            "solver_nfev": stats.get("nfev"),
        },
        "error": None, "inputs": {"phase3_run_id": p3_id},
        "artifacts": {"self_calibrated_camset": str(out_path), "phase3_camset_used": p3_path},
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    return {
        "run_id": run_id, "per_cam_rpe": per_cam,
        "final_euclid": float(stats.get("final_euclid", float("nan"))),
        "nfev": int(stats.get("nfev", 0)),
        "success": bool(stats.get("success", False)),
        "message": str(stats.get("message", "")),
        "mean_disp_mm": mean_disp_mm, "n_free_points": n_free,
        "elapsed": elapsed,
    }


def main():
    print("=" * 70)
    print("ROUND 3: P4 Convergence Push (target < 1.0 px mean)")
    print("=" * 70)

    all_results = []

    for p3_label, p3_path, p3_id in P3_INPUTS:
        print(f"\n=== P3 input: {p3_label} ({p3_id}) ===")
        for cfg in P4_CONFIGS:
            opts = {**P4_BASE, "max_nfev": cfg["max_nfev"], "loss": cfg["loss"],
                    "f_scale": cfg["f_scale"], "xtol": cfg["xtol"]}
            label = f"{p3_label}/{cfg['label']}"
            print(f"\n  --- {label} ---")
            try:
                r = run_p4(p3_path, p3_id, opts)
                c4 = r["per_cam_rpe"].get("camera_4", float("nan"))
                mr = np.mean(list(r["per_cam_rpe"].values()))
                mx = max(r["per_cam_rpe"].values())
                print(f"    cam4={c4:.3f}  mean={mr:.3f}  max={mx:.3f}  disp={r['mean_disp_mm']:.4f}mm  nfev={r['nfev']}  success={r['success']}  msg={r['message'][:60]}  ({r['elapsed']:.1f}s)")
                r["label"] = label
                all_results.append(r)
            except Exception as exc:
                print(f"    ERROR: {exc}")
                all_results.append({"label": label, "error": str(exc)})

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Label':>22} {'cam4':>7} {'mean':>7} {'max':>7} {'disp_mm':>8} {'nfev':>5} {'success':>8} {'message':>30}")
    for r in all_results:
        if "error" in r:
            print(f"{r['label']:>22}  ERROR")
        else:
            c4 = r["per_cam_rpe"].get("camera_4", float("nan"))
            mr = np.mean(list(r["per_cam_rpe"].values()))
            mx = max(r["per_cam_rpe"].values())
            print(f"{r['label']:>22} {c4:7.3f} {mr:7.3f} {mx:7.3f} {r['mean_disp_mm']:8.4f} {r['nfev']:5d} {str(r['success']):>8} {r['message'][:30]:>30}")

    valid = [r for r in all_results if "error" not in r and r["mean_disp_mm"] < 1.5]
    if valid:
        winner = min(valid, key=lambda r: np.mean(list(r["per_cam_rpe"].values())))
        print(f"\nWINNER: {winner['label']}  mean={np.mean(list(winner['per_cam_rpe'].values())):.4f}  cam4={winner['per_cam_rpe']['camera_4']:.4f}")
        print(f"  Stretch goal: {'ACHIEVED' if np.mean(list(winner['per_cam_rpe'].values())) < 1.0 else 'NOT YET'}")


if __name__ == "__main__":
    main()