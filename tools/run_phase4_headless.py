'''
Purpose: Run Phase 4 self-calibration headlessly from the new P3 result.
         Replicates the exact GUI worker function from phase_4_self_calibration.py.
Status:  One-shot script for the P3→P4 pipeline test.
Future:  Remove after the pipeline test is complete.
'''
import sys
import os
import json
import logging
import contextlib
import time
from pathlib import Path

# Add conda env Library/bin to PATH so cairocffi can find cairo.dll
_conda_bin = r"D:\ProgramData\anaconda3\envs\calibration_07032026\Library\bin"
os.environ["PATH"] = _conda_bin + os.pathsep + os.environ.get("PATH", "")

import numpy as np

sys.path.insert(0, r"D:\Work\coding\reconstruction\pyCamSet")

from pyCamSet import load_CameraSet
from pyCamSet.gui.shared_functions import make_run_id
from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler
from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment_with_stats

# ── Paths ──
WORKSPACE = Path(
    r"E:\M_Nebula\1 Data\2 Pilot Data\experiment_15052026\calibration_ccube_images"
    r"\filtered-per-image_reduced_org_by_cam_blue_calib\.pycamset_workspace"
)
F_LOC = str(WORKSPACE.parent)

# The new P3 run — MUST match the verified run
P3_RUN_ID = "20260701_214402_d76b6d"
P3_CAMSET = str(WORKSPACE / "phase3_runs" / P3_RUN_ID / "optimised_cameras.camset")

# Problem options (matching the source P4 run 1ca060)
PROBLEM_OPTIONS = {
    "verbosity": 0,
    "fixed_pose": 0,
    "ref_cam": 0,
    "ref_pose": 0,
    "outliers": "y",
    "max_nfev": 30,
}

SELECTED_CAMERAS = [
    "camera_1", "camera_2", "camera_3", "camera_4",
    "camera_5", "camera_6", "camera_7", "camera_8",
]


class EmitStream:
    """Redirect stdout/stderr to a list, avoiding recursion from print."""
    def __init__(self, log_list):
        self.log_list = log_list
        self.buffer = ""
        self._active = True
    def write(self, text):
        if not self._active:
            return
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


log_lines = []


def main():
    # Verify P3 camset exists
    if not os.path.exists(P3_CAMSET):
        raise FileNotFoundError(f"P3 camset not found: {P3_CAMSET}")

    # Create run directory
    run_id = make_run_id()
    run_dir = WORKSPACE / "phase4_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run ID: {run_id}")
    print(f"Run dir: {run_dir}")
    print(f"P3 camset: {P3_CAMSET}")
    print(f"P3 run ID: {P3_RUN_ID}")

    # Load P3 camset
    print("Loading P3 camset...")
    prev_cams = load_CameraSet(P3_CAMSET)
    print(f"  Cameras: {sorted(prev_cams.get_names())}")

    prev_handler = getattr(prev_cams, "calibration_handler", None)
    if prev_handler is None:
        raise RuntimeError("Selected Phase 3 camset has no calibration handler metadata.")
    print(f"  Handler: {type(prev_handler).__name__}")
    print(f"  Has target: {hasattr(prev_handler, 'target')}")
    print(f"  Has detection: {hasattr(prev_handler, 'detection')}")

    # Build self-calibration handler
    print("Building SelfBundleHandler...")
    handler = SelfBundleHandler(
        camset=prev_cams,
        target=prev_handler.target,
        detection=prev_handler.detection,
        fixed_params=None,
        options=PROBLEM_OPTIONS,
    )
    handler.set_from_templated_camset(prev_cams)

    # Run bundle adjustment
    print("Running self-calibration bundle adjustment...")
    t0 = time.time()
    stream = EmitStream(log_lines)

    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        optimisation, out_cams, stats = run_bundle_adjustment_with_stats(
            handler,
            threads=1,
        )

    elapsed = time.time() - t0
    print(f"Self-calibration completed in {elapsed:.2f}s")

    # Print captured log lines
    if log_lines:
        print("\n--- Captured stdout/stderr ---")
        for line in log_lines[-30:]:
            print(f"  {line}")
        print("--- end captured ---\n")

    print(f"  Success: {stats.get('success', optimisation.success)}")
    init_euclid = float(stats.get("initial_euclid", float("nan")))
    final_euclid = float(stats.get("final_euclid", float("nan")))
    print(f"  Initial Euclidean: {init_euclid:.4f} px")
    print(f"  Final Euclidean: {final_euclid:.4f} px")

    # Save camset
    out_path = run_dir / "self_calibrated_cameras.camset"
    out_cams.save(str(out_path))
    print(f"  Saved camset: {out_path}")

    # Compute per-camera RPE (D4.12)
    per_cam_err = {}
    try:
        dd = np.asarray(handler.get_detection_data(flatten=True))
        residual_xy = np.reshape(np.asarray(optimisation.fun, dtype=float), (-1, 2))
        residual_norm = np.linalg.norm(residual_xy, axis=1)
        if dd.ndim == 2 and dd.shape[1] >= 1:
            cam_idx = dd[:, 0].astype(int)
            if cam_idx.size != residual_norm.size:
                n = min(cam_idx.size, residual_norm.size)
                print(f"  Warning: D4.12 alignment mismatch; truncating to {n}.")
                cam_idx = cam_idx[:n]
                residual_norm = residual_norm[:n]
            cam_names = list(getattr(handler, "cam_names", []))
            for idx, name in enumerate(cam_names):
                mask = cam_idx == idx
                per_cam_err[name] = float(np.mean(residual_norm[mask])) if np.any(mask) else float("nan")
    except Exception as exc:
        print(f"  Warning: D4.12 skipped: {exc}")

    print("\n=== Per-camera RPE ===")
    for k, v in sorted(per_cam_err.items()):
        flag = " <<< HIGH" if v > 5 else ""
        print(f"  {k}: {v:.2f} px{flag}")

    # Compute target displacement (D4.7)
    visible = np.array(getattr(handler, "visible_feature_mask", []), dtype=bool)
    fixed_inds = list(getattr(handler, "fixed_inds", []))
    updated_target = np.array(handler.get_updated_target(optimisation.x), dtype=float)
    ref_target = np.array(handler.target.point_data, dtype=float).reshape(-1, 3)
    displacement = np.linalg.norm(updated_target - ref_target, axis=1)
    mean_disp_mm = float(np.nanmean(displacement) * 1000.0) if displacement.size else float("nan")
    print(f"\nmean_target_displacement_mm: {mean_disp_mm:.4f}")

    # Build metadata
    params = {
        "f_loc": F_LOC,
        "threads": 1,
        "fixed_params": None,
        "selected_cameras": SELECTED_CAMERAS,
        "problem_options": PROBLEM_OPTIONS,
    }

    diagnostics = {
        "D4.3_initial_euclid_px": init_euclid,
        "D4.3_final_euclid_px": final_euclid,
        "D4.7_mean_target_displacement_mm": mean_disp_mm,
        "D4.12_per_camera_mean_reprojection": per_cam_err,
    }

    metadata = {
        "run_id": run_id,
        "phase": "phase4",
        "params": params,
        "diagnostics": diagnostics,
        "error": None,
        "inputs": {"phase3_run_id": P3_RUN_ID},
        "artifacts": {
            "self_calibrated_camset": str(out_path),
            "optimised_camset": str(out_path),
            "phase3_camset_used": P3_CAMSET,
        },
    }

    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"\nMetadata saved: {metadata_path}")
    print(f"\n=== P4 RUN COMPLETE ===")
    print(f"P4_RUN_ID = {run_id}")
    print(f"P4_RUN_DIR = {run_dir}")


if __name__ == "__main__":
    main()