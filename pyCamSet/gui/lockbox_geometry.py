'''
Purpose: Geometry and metadata helpers for the Phase 3 lockbox prior editor.
Status:  Experimental v1 foundation for translation-only camera-centre edits.
Future:  Add radius-plus-plane constrained snapping and richer target-centre discovery.
'''
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

TrustState = Literal["trusted", "uncertain", "bad"]
RadiusStatistic = Literal["median", "mean"]


@dataclass(slots=True)
class PlaneGroup:
    """Persistent description of a user-defined coplanar camera group."""

    plane_id: str
    member_camera_names: list[str] = field(default_factory=list)
    fit_mode: str = "unfit"
    normal: list[float] | None = None
    point: list[float] | None = None
    rms_residual: float | None = None
    max_residual: float | None = None
    trusted_members: list[str] = field(default_factory=list)
    locked: bool = False
    active: bool = True

    def to_dict(self) -> dict:
        """Return a JSON-serialisable plane group payload."""
        return {
            "plane_id": self.plane_id,
            "member_camera_names": list(self.member_camera_names),
            "fit_mode": self.fit_mode,
            "normal": self.normal,
            "point": self.point,
            "rms_residual": self.rms_residual,
            "max_residual": self.max_residual,
            "trusted_members": list(self.trusted_members),
            "locked": bool(self.locked),
            "active": bool(self.active),
        }


@dataclass(slots=True)
class CameraEditRecord:
    """Per-camera provenance saved beside an edited lockbox camset."""

    trust: TrustState = "uncertain"
    plane_group: str | None = None
    included_in_lockbox: bool = True
    original_center: list[float] = field(default_factory=list)
    edited_center: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable camera edit payload with delta fields."""
        original = np.asarray(self.original_center, dtype=float)
        edited = np.asarray(self.edited_center, dtype=float)
        delta = edited - original
        return {
            "trust": self.trust,
            "plane_group": self.plane_group,
            "included_in_lockbox": bool(self.included_in_lockbox),
            "original_center": original.tolist(),
            "edited_center": edited.tolist(),
            "delta": delta.tolist(),
            "delta_norm": float(np.linalg.norm(delta)),
        }


def camera_center_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    """Return camera centre C=-R.T@t for x_cam=R*x_world+t."""
    ext = np.asarray(extrinsic, dtype=float)
    if ext.shape != (4, 4):
        raise ValueError(f"Expected 4x4 extrinsic, got {ext.shape}")
    rotation = ext[:3, :3]
    translation = ext[:3, 3]
    return -rotation.T @ translation


def extrinsic_from_center_preserving_rotation(extrinsic: np.ndarray, center: Iterable[float]) -> np.ndarray:
    """Return a new extrinsic with the same rotation and edited camera centre."""
    ext = np.array(extrinsic, dtype=float, copy=True)
    if ext.shape != (4, 4):
        raise ValueError(f"Expected 4x4 extrinsic, got {ext.shape}")
    new_center = np.asarray(center, dtype=float).reshape(3)
    rotation = ext[:3, :3]
    ext[:3, 3] = -rotation @ new_center
    ext[3, :] = np.array([0.0, 0.0, 0.0, 1.0])
    return ext


def radius_from_center(point: Iterable[float], object_center: Iterable[float]) -> float:
    """Return Euclidean radius of *point* around *object_center*."""
    return float(np.linalg.norm(np.asarray(point, dtype=float) - np.asarray(object_center, dtype=float)))


def reference_radius(points: Iterable[Iterable[float]], object_center: Iterable[float], statistic: RadiusStatistic = "median") -> float:
    """Return median or mean camera radius for trusted reference points."""
    radii = np.array([radius_from_center(point, object_center) for point in points], dtype=float)
    if radii.size == 0:
        raise ValueError("At least one reference camera is required")
    if statistic == "median":
        return float(np.median(radii))
    if statistic == "mean":
        return float(np.mean(radii))
    raise ValueError(f"Unsupported radius statistic: {statistic}")


def snap_radius_to_reference(point: Iterable[float], object_center: Iterable[float], target_radius: float, eps: float = 1e-12) -> np.ndarray:
    """Move *point* radially so it lies exactly at *target_radius* from *object_center*."""
    p = np.asarray(point, dtype=float)
    centre = np.asarray(object_center, dtype=float)
    vec = p - centre
    norm = float(np.linalg.norm(vec))
    if norm <= eps:
        raise ValueError("Cannot snap radius for a camera effectively at the object centre")
    return centre + (float(target_radius) * vec / norm)


def fit_plane(points: Iterable[Iterable[float]]) -> dict:
    """Fit n.x+d=0 using SVD and return canonicalised unit-normal diagnostics."""
    arr = np.asarray(list(points), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Plane fitting expects an Nx3 point array")
    if arr.shape[0] < 2:
        raise ValueError("At least two cameras are required before a plane group has geometry")

    centroid = np.mean(arr, axis=0)
    if arr.shape[0] == 2:
        direction = arr[1] - arr[0]
        axis = np.eye(3)[int(np.argmin(np.abs(direction)))]
        normal = np.cross(direction, axis)
        fit_mode = "two_point_low_confidence"
    else:
        _, _, vh = np.linalg.svd(arr - centroid, full_matrices=False)
        normal = vh[-1]
        fit_mode = "defined" if arr.shape[0] == 3 else "fitted"

    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise ValueError("Plane normal is undefined for the supplied camera positions")
    normal = normal / norm

    # Canonicalise sign by making the largest-magnitude component positive.
    dominant = int(np.argmax(np.abs(normal)))
    if normal[dominant] < 0:
        normal = -normal

    signed = (arr - centroid) @ normal
    return {
        "fit_mode": fit_mode,
        "normal": normal.tolist(),
        "point": centroid.tolist(),
        "d": float(-normal @ centroid),
        "rms_residual": float(np.sqrt(np.mean(signed**2))),
        "max_residual": float(np.max(np.abs(signed))),
    }


def project_point_to_plane(point: Iterable[float], normal: Iterable[float], plane_point: Iterable[float] | None = None, d: float | None = None) -> np.ndarray:
    """Project a point onto n.x+d=0, accepting either *d* or a point on the plane."""
    p = np.asarray(point, dtype=float)
    n = np.asarray(normal, dtype=float)
    n_norm = float(np.linalg.norm(n))
    if n_norm <= 1e-12:
        raise ValueError("Plane normal must be non-zero")
    n = n / n_norm
    if d is None:
        if plane_point is None:
            raise ValueError("Either plane_point or d is required")
        d = float(-n @ np.asarray(plane_point, dtype=float))
    signed = float(n @ p + d)
    return p - signed * n


def match_signed_plane_offset(point: Iterable[float], normal: Iterable[float], reference_offset: float, plane_point: Iterable[float] | None = None, d: float | None = None) -> np.ndarray:
    """Move a point along the normal so its signed plane offset matches *reference_offset*."""
    p = np.asarray(point, dtype=float)
    n = np.asarray(normal, dtype=float)
    n_norm = float(np.linalg.norm(n))
    if n_norm <= 1e-12:
        raise ValueError("Plane normal must be non-zero")
    n = n / n_norm
    if d is None:
        if plane_point is None:
            raise ValueError("Either plane_point or d is required")
        d = float(-n @ np.asarray(plane_point, dtype=float))
    signed = float(n @ p + d)
    return p - (signed - float(reference_offset)) * n


def resolve_object_centre(target: object | None) -> dict:
    """Resolve the default object-space centre for the editor metadata."""
    if target is None:
        point = np.zeros(3, dtype=float)
        mode = "target_origin"
    elif target.__class__.__name__ == "Ccube":
        point = np.zeros(3, dtype=float)
        mode = "target_origin"
    elif hasattr(target, "origin"):
        point = np.asarray(getattr(target, "origin"), dtype=float).reshape(3)
        mode = "target_defined_origin"
    elif hasattr(target, "point_data"):
        point_data = np.asarray(getattr(target, "point_data"), dtype=float).reshape(-1, 3)
        point = np.mean(point_data, axis=0)
        mode = "target_point_data_centroid"
    else:
        point = np.zeros(3, dtype=float)
        mode = "target_origin"
    return {"mode": mode, "point": point.tolist(), "label": "Object centre (target frame)"}


def build_lockbox_metadata(
    *,
    original_source_camset: Path | str,
    edited_source_camset: Path | str,
    centre_definition: dict,
    cameras: dict[str, CameraEditRecord],
    plane_groups: Iterable[PlaneGroup] = (),
    edit_history_summary: Iterable[str] = (),
) -> dict:
    """Build the sidecar JSON payload for an edited lockbox copy."""
    edited = str(edited_source_camset)
    return {
        "editor_version": "0.1",
        "implementation_mode": "extrinsic_parameter_mvp",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "original_source_camset": str(original_source_camset),
        "edited_source_camset": edited,
        "effective_lockbox_source": edited,
        "centre_definition": centre_definition,
        "cameras": {name: record.to_dict() for name, record in cameras.items()},
        "plane_groups": [group.to_dict() for group in plane_groups],
        "edit_history_summary": list(edit_history_summary),
    }
