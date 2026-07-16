'''
Purpose: Run Phase 3 bundle adjustment headlessly with the new lockbox prior.
         Replicates the exact GUI worker function from phase_3_bundle_adjustment.py.
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
from datetime import datetime

# Add conda env Library/bin to PATH so cairocffi can find cairo.dll
_conda_bin = r"D:\ProgramData\anaconda3\envs\calibration_07032026\Library\bin"
os.environ["PATH"] = _conda_bin + os.pathsep + os.environ.get("PATH", "")

import numpy as np

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
from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment_with_stats

# ── Paths ──
WORKSPACE = Path(
    r"E:\M_Nebula\1 Data\2 Pilot Data\experiment_15052026\calibration_ccube_images"
    r"\filtered-per-image_reduced_org_by_cam_blue_calib\.pycamset_workspace"
)
F_LOC = str(WORKSPACE.parent)
PRIOR_DIR = WORKSPACE / "lockbox_priors" / "20260701_204308"
LOCKBOX_SOURCE = str(PRIOR_DIR / "edited_lockbox_source.camset")

# Phase 2 run
PHASE2_RUN_DIR = WORKSPACE / "phase2_runs" / "20260628_171228_30481f"
PHASE2_CAMSET = str(PHASE2_RUN_DIR / "initial_cameras.camset")

# Phase 1 run
PHASE1_RUN_DIR = WORKSPACE / "phase1_runs" / "20260628_171157_f1445a"
PHASE1_PICKLE = str(PHASE1_RUN_DIR / "detected_datapoints.pickle")

# Lockbox parameters (from task spec)
LOCKBOX_PARAMS = {
    "enabled": True,
    "rotation_half_width": 0.3,  # widened from 0.1 to let optimiser refine camera 4's rotation
    "translation_half_width": 0.1,
    "rotation_sigma": 0.05,
    "translation_sigma": 0.01,
    "center_sigma": 0.1,  # tighter centre constraint to keep |dC| < 0.10
    "warm_start": True,
}

# Problem options (matching the reference run fcc76a)
PROBLEM_OPTIONS = {
    "verbosity": 0,
    "fixed_pose": 0,
    "ref_cam": 0,
    "ref_pose": 0,
    "outliers": "y",
    "max_nfev": 1000,
}

SELECTED_CAMERAS = [
    "camera_1", "camera_2", "camera_3", "camera_4",
    "camera_5", "camera_6", "camera_7", "camera_8",
]

# Target parameters (from phase 2 metadata)
TARGET_TYPE = "Ccube"
N_POINTS = 6
LENGTH = 10.0


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


def emit(msg):
    log_lines.append(str(msg))


def main():
    # Create run directory
    run_id = make_run_id()
    run_dir = WORKSPACE / "phase3_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run ID: {run_id}")
    print(f"Run dir: {run_dir}")

    # Load inputs
    print("Loading Phase 2 camset...")
    cams = load_CameraSet(PHASE2_CAMSET)
    print(f"  Cameras: {sorted(cams.get_names())}")

    print("Loading Phase 1 detections...")
    payload = load_pickle(PHASE1_PICKLE)
    detections = extract_detection(payload)
    if detections is None:
        raise RuntimeError("Could not extract TargetDetection from Phase 1 pickle.")

    print("Loading lockbox source camset...")
    lockbox_source_camset = load_CameraSet(LOCKBOX_SOURCE)
    if set(lockbox_source_camset.get_names()) != set(cams.get_names()):
        raise RuntimeError(
            "Effective lockbox source camera names do not match the active Phase 2 camset."
        )

    # Build target
    target = build_target(TARGET_TYPE, N_POINTS, LENGTH)

    # Build lockbox config
    lockbox_config = CameraLockboxConfig(
        enabled=True,
        rotation_half_width=float(LOCKBOX_PARAMS["rotation_half_width"]),
        translation_half_width=float(LOCKBOX_PARAMS["translation_half_width"]),
        rotation_sigma=float(LOCKBOX_PARAMS["rotation_sigma"]),
        translation_sigma=float(LOCKBOX_PARAMS["translation_sigma"]),
        center_sigma=float(LOCKBOX_PARAMS["center_sigma"]),
    )

    # Build handler
    print("Building TemplateBundleHandler...")
    handler = TemplateBundleHandler(
        camset=cams,
        target=target,
        detection=detections,
        fixed_params=None,
        options=PROBLEM_OPTIONS,
        lockbox_config=lockbox_config,
        lockbox_source_camset=lockbox_source_camset,
        lockbox_warm_start=True,
    )

    # Run bundle adjustment
    print("Running bundle adjustment...")
    t0 = time.time()
    stream = EmitStream(log_lines)

    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        optimisation, out_cams, stats = run_bundle_adjustment_with_stats(
            handler,
            threads=1,
    )

    elapsed = time.time() - t0
    print(f"Bundle adjustment completed in {elapsed:.2f}s")

    # Print captured log lines
    if log_lines:
        print("\n--- Captured stdout/stderr ---")
        for line in log_lines[-50:]:  # last 50 lines
            print(f"  {line}")
        print("--- end captured ---\n")
    print(f"  Success: {stats.get('success', optimisation.success)}")
    print(f"  Initial Euclidean: {stats.get('initial_euclid', float('nan')):.4f} px")
    print(f"  Final Euclidean: {stats.get('final_euclid', float('nan')):.4f} px")

    # Save optimised camset
    camset_out_path = run_dir / "optimised_cameras.camset"
    out_cams.save(str(camset_out_path))
    print(f"  Saved camset: {camset_out_path}")

    # Compute per-camera RPE (D3.12)
    per_cam_err = {}
    try:
        dd = np.asarray(handler.get_detection_data(flatten=True))
        residual_xy = np.reshape(np.asarray(optimisation.fun, dtype=float), (-1, 2))
        residual_norm = np.linalg.norm(residual_xy, axis=1)

        if dd.ndim == 2 and dd.shape[1] >= 1:
            cam_idx = dd[:, 0].astype(int)
            if cam_idx.size != residual_norm.size:
                n = min(cam_idx.size, residual_norm.size)
                print(f"  Warning: D3.12 alignment mismatch; truncating to {n}.")
                cam_idx = cam_idx[:n]
                residual_norm = residual_norm[:n]

            for idx, name in enumerate(handler.cam_names):
                mask = cam_idx == idx
                per_cam_err[name] = (
                    float(np.mean(residual_norm[mask])) if np.any(mask) else float("nan")
                )
        else:
            print("  Warning: D3.12 skipped (unexpected detection-data shape).")
    except Exception as exc:
        print(f"  Warning: D3.12 skipped due to diagnostics error: {exc}")

    print("\n=== Per-camera RPE ===")
    for k, v in sorted(per_cam_err.items()):
        flag = " <<< HIGH" if v > 20 else ""
        print(f"  {k}: {v:.2f} px{flag}")

    # Build metadata
    params = {
        "f_loc": F_LOC,
        "threads": 1,
        "fixed_params": None,
        "target_type": TARGET_TYPE,
        "n_points": N_POINTS,
        "length": LENGTH,
        "selected_cameras": SELECTED_CAMERAS,
        "lockbox": {
            **LOCKBOX_PARAMS,
            "original_source_camset": r"E:/M_Nebula/1 Data/2 Pilot Data/experiment_15052026/calibration_ccube_images/filtered-per-image_reduced_org_by_cam_blue_calib/.pycamset_workspace/phase4_runs/20260628_171322_1ca060/self_calibrated_cameras.camset",
            "edited_source_camset": LOCKBOX_SOURCE,
            "edited_source_metadata": str(PRIOR_DIR / "edited_lockbox_source_metadata.json"),
            "source_camset": LOCKBOX_SOURCE,
            "implementation_mode": "extrinsic_parameter_mvp",
        },
        "problem_options": PROBLEM_OPTIONS,
    }

    init_euclid = float(stats.get("initial_euclid", float("nan")))
    final_euclid = float(stats.get("final_euclid", float("nan")))
    dt = float(stats.get("elapsed_sec", elapsed))

    diagnostics = {
        "D3.5_initial_euclid_px": init_euclid,
        "D3.6_final_euclid_px": final_euclid,
        "D3.7_error_reduction_ratio": float(init_euclid / final_euclid) if final_euclid > 0 else float("inf"),
        "D3.8_solver_status": {
            "status": int(stats.get("status", optimisation.status)),
            "message": str(stats.get("message", optimisation.message)),
            "success": bool(stats.get("success", optimisation.success)),
        },
        "D3.9_nfev": int(stats.get("nfev", optimisation.nfev)),
        "D3.12_per_camera_mean_reprojection": per_cam_err,
    }

    metadata = {
        "run_id": run_id,
        "phase": "phase3",
        "params": params,
        "diagnostics": diagnostics,
        "error": None,
        "inputs": {
            "phase2_run_id": "20260628_171228_30481f",
            "phase1_run_id": "20260628_171157_f1445a",
        },
        "artifacts": {
            "optimised_camset": str(camset_out_path),
            "phase2_initial_camset_used": PHASE2_CAMSET,
            "phase1_detection_pickle_used": PHASE1_PICKLE,
        },
    }

    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"\nMetadata saved: {metadata_path}")
    print(f"\n=== P3 RUN COMPLETE ===")
    print(f"P3_RUN_ID = {run_id}")
    print(f"P3_RUN_DIR = {run_dir}")


if __name__ == "__main__":
    main()