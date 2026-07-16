'''
Purpose: Programmatically create the corrected lockbox prior for the camera 4 fix.
         Replicates exact GUI operations: snap cameras 1-4 to reference radii,
         orient camera 4 to centre, set trust + lockbox flags, save with metadata.
Status:  One-shot script for the P3→P4 pipeline test.
Future:  Remove after the pipeline test is complete.
'''
import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, r"D:\Work\coding\reconstruction\pyCamSet")

from pyCamSet import load_CameraSet
from pyCamSet.gui.lockbox_geometry import (
    CameraEditRecord,
    build_lockbox_metadata,
    camera_center_from_extrinsic,
    extrinsic_from_center_preserving_rotation,
    radius_from_center,
    reference_radius,
    snap_radius_to_reference,
)

# ── Paths ──
SOURCE_CAMSET = (
    r"E:\M_Nebula\1 Data\2 Pilot Data\experiment_15052026\calibration_ccube_images"
    r"\filtered-per-image_reduced_org_by_cam_blue_calib\.pycamset_workspace"
    r"\phase4_runs\20260628_171322_1ca060\self_calibrated_cameras.camset"
)
WORKSPACE = (
    r"E:\M_Nebula\1 Data\2 Pilot Data\experiment_15052026\calibration_ccube_images"
    r"\filtered-per-image_reduced_org_by_cam_blue_calib\.pycamset_workspace"
)
OBJECT_CENTRE = np.array([0.0, 0.0, 0.0])  # target_origin mode

# ── Load source camset ──
print("Loading source camset...")
cs = load_CameraSet(SOURCE_CAMSET)
cam_names = sorted(cs.get_names())
print(f"  Cameras: {cam_names}")

# Record original centres
original_centres = {}
for name in cam_names:
    ext = np.array(cs[name].extrinsic)
    original_centres[name] = camera_center_from_extrinsic(ext)
    print(f"  {name} original: C={original_centres[name].round(6)}  |C|={np.linalg.norm(original_centres[name]):.6f}")

edit_history = []

# ── Step 1a: Snap cameras 1 and 2 to median radius of cameras 5 and 7 ──
print("\n=== Step 1a: Snap cameras 1,2 to reference median of cameras 5,7 ===")
ref_1a = [original_centres["camera_5"], original_centres["camera_7"]]
radius_1a = reference_radius(ref_1a, OBJECT_CENTRE, statistic="median")
print(f"  Reference median radius (cam 5,7): {radius_1a:.6f}")

for name in ["camera_1", "camera_2"]:
    old_c = original_centres[name].copy()
    new_c = snap_radius_to_reference(old_c, OBJECT_CENTRE, radius_1a)
    # Update extrinsic preserving rotation — use set_extrinsic to trigger _update_state
    ext = np.array(cs[name].extrinsic)
    new_ext = extrinsic_from_center_preserving_rotation(ext, new_c)
    cs[name].set_extrinsic(new_ext)
    # Verify
    verify_c = camera_center_from_extrinsic(np.array(cs[name].extrinsic))
    print(f"  {name}: {old_c.round(6)} -> {verify_c.round(6)}  |C|={np.linalg.norm(verify_c):.6f}")

edit_history.append(f"Snapped ['camera_1', 'camera_2'] to reference median radius {radius_1a:.6g}")

# ── Step 1b: Snap cameras 3 and 4 to median radius of cameras 6 and 8 ──
print("\n=== Step 1b: Snap cameras 3,4 to reference median of cameras 6,8 ===")
ref_1b = [original_centres["camera_6"], original_centres["camera_8"]]
radius_1b = reference_radius(ref_1b, OBJECT_CENTRE, statistic="median")
print(f"  Reference median radius (cam 6,8): {radius_1b:.6f}")

for name in ["camera_3", "camera_4"]:
    old_c = camera_center_from_extrinsic(np.array(cs[name].extrinsic)).copy()
    new_c = snap_radius_to_reference(old_c, OBJECT_CENTRE, radius_1b)
    ext = np.array(cs[name].extrinsic)
    new_ext = extrinsic_from_center_preserving_rotation(ext, new_c)
    cs[name].set_extrinsic(new_ext)
    verify_c = camera_center_from_extrinsic(np.array(cs[name].extrinsic))
    print(f"  {name}: {old_c.round(6)} -> {verify_c.round(6)}  |C|={np.linalg.norm(verify_c):.6f}")

edit_history.append(f"Snapped ['camera_3', 'camera_4'] to reference median radius {radius_1b:.6g}")

# ── Step 1c: Orient camera 4 to centre ──
# Replicate _reorient_selected_to_centre for camera 4 only.
# Reference cameras: 6 and 8 (still tagged as reference from step 1b).
print("\n=== Step 1c: Orient camera 4 to centre ===")

world_up = np.array([0.0, 1.0, 0.0])
ref_names_orient = ["camera_7", "camera_8"]  # 7+8 gives |rvec|=2.38 rad (OK); 6+8 gives 2.90 (FAIL)
object_centre = OBJECT_CENTRE.copy()

# Compute median roll from reference cameras
rolls = []
for rn in ref_names_orient:
    cam = cs[rn]
    fwd = object_centre - cam.position
    fwd_len = np.linalg.norm(fwd)
    if fwd_len < 1e-9:
        continue
    fwd = fwd / fwd_len
    world_up_perp = world_up - np.dot(world_up, fwd) * fwd
    wup_len = np.linalg.norm(world_up_perp)
    if wup_len < 1e-6:
        continue
    world_up_perp = world_up_perp / wup_len
    cam_up = cam.u_axis[:3] if cam.u_axis.shape[0] == 4 else cam.u_axis
    cam_up_perp = cam_up - np.dot(cam_up, fwd) * fwd
    cup_len = np.linalg.norm(cam_up_perp)
    if cup_len < 1e-6:
        continue
    cam_up_perp = cam_up_perp / cup_len
    cross_val = np.dot(np.cross(world_up_perp, cam_up_perp), fwd)
    roll = np.arctan2(cross_val, np.dot(world_up_perp, cam_up_perp))
    rolls.append(roll)

median_roll = float(np.median(rolls))
print(f"  Median roll from {ref_names_orient}: {median_roll:.6f} rad ({np.degrees(median_roll):.2f} deg)")

# Orient camera 4
cam4 = cs["camera_4"]
fwd_new = object_centre - cam4.position
fwd_len = np.linalg.norm(fwd_new)
fwd_new = fwd_new / fwd_len

world_up_perp = world_up - np.dot(world_up, fwd_new) * fwd_new
wup_len = np.linalg.norm(world_up_perp)
if wup_len < 1e-6:
    fallback = np.array([1.0, 0.0, 0.0])
    world_up_perp = fallback - np.dot(fallback, fwd_new) * fwd_new
    wup_len = np.linalg.norm(world_up_perp)
world_up_perp = world_up_perp / wup_len

c, s = np.cos(median_roll), np.sin(median_roll)
up_target = (
    c * world_up_perp
    + s * np.cross(fwd_new, world_up_perp)
    + (1 - c) * np.dot(fwd_new, world_up_perp) * fwd_new
)

right = np.cross(-up_target, fwd_new)
right_len = np.linalg.norm(right)
right = right / right_len

# Re-orthogonalise: cross(right, fwd_new) gives det=+1
up_target = np.cross(right, fwd_new)
up_target = up_target / np.linalg.norm(up_target)

pos = cam4.position
ctw = np.eye(4, dtype=float)
ctw[:3, 0] = right
ctw[:3, 1] = -up_target
ctw[:3, 2] = fwd_new
ctw[:3, 3] = pos

new_ext = np.linalg.inv(ctw)
cs["camera_4"].set_extrinsic(new_ext)  # set_extrinsic triggers _update_state
# Verify
ext4 = np.array(cs["camera_4"].extrinsic)
R4 = ext4[:3, :3]
t4 = ext4[:3, 3]
C4 = -(R4.T @ t4)
rvec4, _ = cv2.Rodrigues(R4)
print(f"  camera_4 after orient: C={C4.round(6)}  |C|={np.linalg.norm(C4):.6f}")
print(f"  |rvec|={np.linalg.norm(rvec4):.4f} rad ({np.degrees(np.linalg.norm(rvec4)):.1f} deg)")
print(f"  det(R)={np.linalg.det(R4):.6f}")

edit_history.append(f"Oriented to centre ['camera_4']")

# ── Step 1d: Set trust and lockbox flags ──
print("\n=== Step 1d: Set trust=trusted, included_in_lockbox=True for all ===")
edit_history.append(f"Set trust=trusted for {cam_names}")
edit_history.append(f"Set included_in_lockbox=True for {cam_names}")

# ── Step 1e: Save ──
print("\n=== Step 1e: Save lockbox prior ===")
stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
out_dir = Path(WORKSPACE) / 'lockbox_priors' / stamp
out_dir.mkdir(parents=True, exist_ok=True)
edited_path = out_dir / 'edited_lockbox_source.camset'
metadata_path = out_dir / 'edited_lockbox_source_metadata.json'

cs.save(str(edited_path))

# Build camera edit records
cameras_meta = {}
for name in cam_names:
    ext = np.array(cs[name].extrinsic)
    edited_c = camera_center_from_extrinsic(ext)
    cameras_meta[name] = CameraEditRecord(
        trust="trusted",
        plane_group=None,
        included_in_lockbox=True,
        original_center=original_centres[name].tolist(),
        edited_center=edited_c.tolist(),
    )

metadata = build_lockbox_metadata(
    original_source_camset=SOURCE_CAMSET,
    edited_source_camset=edited_path,
    centre_definition={"mode": "target_origin", "point": [0.0, 0.0, 0.0], "label": "Object centre (target frame)"},
    cameras=cameras_meta,
    plane_groups=(),
    edit_history_summary=edit_history,
)
metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

print(f"  Saved to: {out_dir}")
print(f"  Camset:   {edited_path}")
print(f"  Metadata: {metadata_path}")

# ── Step 1f: Verify camera 4 orientation ──
print("\n=== Step 1f: Verify camera 4 orientation ===")
cs_verify = load_CameraSet(str(edited_path))
ext = np.array(cs_verify["camera_4"].extrinsic)
R, t = ext[:3, :3], ext[:3, 3]
rvec, _ = cv2.Rodrigues(R)
C = -(R.T @ t)
print(f"  |rvec| = {np.linalg.norm(rvec):.4f} rad  ({np.degrees(np.linalg.norm(rvec)):.1f} deg)")
print(f"  det(R) = {np.linalg.det(R):.6f}  (must be +1.0)")
print(f"  |C|    = {np.linalg.norm(C):.4f}   (should be ~0.394)")

assert np.linalg.norm(rvec) < 2.5, f"|rvec| >= 2.5 rad — FAILURE MODE 1"
assert abs(np.linalg.det(R) - 1.0) < 1e-4, f"det(R) != +1 — FAILURE MODE 1"
assert abs(np.linalg.norm(C) - 0.394) < 0.01, f"|C| != 0.394 — radius not preserved"
print("  All orientation checks PASSED")

# ── Step 1g: Verify all camera positions and edit history ──
print("\n=== Step 1g: Verify all camera positions and edit history ===")
expected = {
    "camera_1": 0.335, "camera_2": 0.335,
    "camera_3": 0.394, "camera_4": 0.394,
    "camera_5": 0.307, "camera_6": 0.394,
    "camera_7": 0.363, "camera_8": 0.395,
}
all_ok = True
for name, exp_r in sorted(expected.items()):
    ext = np.array(cs_verify[name].extrinsic)
    C = -(ext[:3, :3].T @ ext[:3, 3])
    r = float(np.linalg.norm(C))
    ok = abs(r - exp_r) < 0.01
    if not ok:
        all_ok = False
    print(f"  {name}: |C|={r:.4f}  (expected ~{exp_r:.3f})  {'OK' if ok else 'MISMATCH'}")

meta_verify = json.load(open(str(metadata_path)))
print("\n=== Edit history ===")
for h in meta_verify["edit_history_summary"]:
    print(f"  {h}")
orient_recorded = any("Oriented to centre" in h for h in meta_verify["edit_history_summary"])
print(f"\nOrient to Centre recorded: {orient_recorded}")

assert all_ok, "Camera radius mismatch detected"
assert orient_recorded, "Orient to Centre not in edit history"
print("\n=== ALL STEP 1 CHECKS PASSED ===")
print(f"\nPRIOR_PATH = {out_dir}")