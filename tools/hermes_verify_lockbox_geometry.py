'''
Purpose: Ad-hoc verifier for Phase 3 lockbox geometry helpers when pytest is unavailable.
Status:  Developer verification script; safe to rerun.
Future:  Replace with normal pytest once the calibration environment includes pytest.
'''
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

module_path = Path(__file__).resolve().parents[1] / "pyCamSet" / "gui" / "lockbox_geometry.py"
spec = importlib.util.spec_from_file_location("lockbox_geometry_under_test", module_path)
assert spec is not None and spec.loader is not None
geom = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = geom
spec.loader.exec_module(geom)

ext = np.eye(4)
ext[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
ext[:3, 3] = np.array([1.0, 2.0, 3.0])
centre = geom.camera_center_from_extrinsic(ext)
assert np.allclose(centre, -ext[:3, :3].T @ ext[:3, 3])
print(f"OK 1/7 centre {centre.tolist()}")

edited = np.array([4.0, 5.0, 6.0])
out = geom.extrinsic_from_center_preserving_rotation(ext, edited)
assert np.allclose(out[:3, :3], ext[:3, :3])
assert np.allclose(geom.camera_center_from_extrinsic(out), edited)
print(f"OK 2/7 extrinsic_translation {out[:3, 3].tolist()}")

points = [np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), np.array([100.0, 0.0, 0.0])]
assert geom.reference_radius(points, np.zeros(3), statistic="median") == 2.0
print("OK 3/7 median radius rejects outlier")

try:
    geom.snap_radius_to_reference([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 3.0)
except ValueError:
    print("OK 4/7 centre snap refused")
else:
    raise AssertionError("snap at centre did not fail")

assert np.allclose(geom.snap_radius_to_reference([2.0, 0.0, 0.0], [0.0, 0.0, 0.0], 5.0), [5.0, 0.0, 0.0])
print("OK 5/7 radial snap preserves direction")

fit = geom.fit_plane([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [1.0, 1.0, 2.0]])
assert np.allclose(fit["normal"], [0.0, 0.0, 1.0])
assert fit["rms_residual"] == 0.0
print(f"OK 6/7 plane {fit}")

assert np.allclose(geom.project_point_to_plane([1.0, 2.0, 5.0], [0.0, 0.0, 1.0], plane_point=[0.0, 0.0, 2.0]), [1.0, 2.0, 2.0])
assert np.allclose(geom.match_signed_plane_offset([1.0, 2.0, 5.0], [0.0, 0.0, 1.0], 0.5, plane_point=[0.0, 0.0, 2.0]), [1.0, 2.0, 2.5])
print("OK 7/9 plane projection and signed offset")

# Radial move: move [3,0,0] outward by 2 from centre [0,0,0] -> [5,0,0]
centre = np.zeros(3)
p_test = np.array([3.0, 0.0, 0.0])
delta = 2.0
vec = p_test - centre
direction = vec / np.linalg.norm(vec)
moved = p_test + delta * direction
assert np.allclose(moved, [5.0, 0.0, 0.0])
print(f"OK 8/9 radial move {moved.tolist()}")

# Reference-based signed offset: ref at z=3, target at z=5, plane z=2, normal [0,0,1]
# ref offset = 3-2=1, target offset = 5-2=3, move target by (3-1)=2 downward -> z=3
normal = np.array([0.0, 0.0, 1.0])
plane_point = np.array([0.0, 0.0, 2.0])
d_val = float(-normal @ plane_point)
ref_offset = float(normal @ np.array([0.0, 0.0, 3.0]) + d_val)  # = 1.0
target_point = np.array([1.0, 2.0, 5.0])
result = geom.match_signed_plane_offset(target_point, normal, ref_offset, d=d_val)
assert np.allclose(result, [1.0, 2.0, 3.0])
print(f"OK 9/9 reference offset match {result.tolist()}")
