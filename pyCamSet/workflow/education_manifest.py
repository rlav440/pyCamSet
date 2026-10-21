"""Record and verify Phase 1 image identity and cache provenance.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ManifestResult:
    """Manifest comparison with an explicit, fail-closed disposition."""

    status: str
    photo_names: Mapping[str, tuple[str, ...]]
    unavailable: Mapping[str, tuple[str, ...]]
    reason: str | None = None


def describe_enumeration(camera_paths: Mapping[str, tuple[Path, ...]],
                         camera_root: Path, subfolder_string: str | None = None,
                         include_digest: bool = False) -> dict[str, Any]:
    """Serialise the exact detector enumeration into relative per-camera records."""
    cameras: dict[str, list[dict[str, Any]]] = {}
    for camera_name, paths in camera_paths.items():
        folder = Path(camera_root) / camera_name
        if subfolder_string:
            folder /= subfolder_string
        base = folder.resolve()
        records = []
        for path in paths:
            resolved = Path(path).resolve()
            try:
                name = resolved.relative_to(base).as_posix()
                stat = resolved.stat()
                item: dict[str, Any] = {
                    "name": name,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
                if include_digest:
                    digest = hashlib.sha256()
                    with resolved.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    item["sha256"] = digest.hexdigest()
            except (OSError, ValueError):
                # A resolved symlink outside the camera folder is not safely
                # representable as a camera-relative name, so fail closed.
                records = []
                break
            records.append(item)
        cameras[str(camera_name)] = records
    return {"schema_version": 1,
            "recordable": bool(cameras) and all(cameras.values()),
            "cameras": cameras}


def _current_enumeration(root: Path, camera_names: tuple[str, ...],
                         subfolder_string: str | None, n_lim: int | None,
                         include_digest: bool) -> dict[str, Any]:
    from pyCamSet.calibration_targets.core.abstract_target import enumerate_detection_images

    paths: dict[str, tuple[Path, ...]] = {}
    for name in camera_names:
        folder = root / name
        if subfolder_string:
            folder /= subfolder_string
        paths[name] = tuple(enumerate_detection_images(folder, n_lim))
    return describe_enumeration(paths, root, subfolder_string, include_digest)


def verify(manifest: Mapping[str, Any] | None, f_loc: Path | str,
           cam_names: tuple[str, ...] | list[str] | None = None,
           subfolder_string: str | None = None, *, n_lim: int | None = None,
           include_digest: bool = False) -> ManifestResult:
    """Compare a saved detector enumeration with today's source images."""
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 1:
        return ManifestResult("source identity not recorded", {}, {},
                              "manifest missing or unsupported")
    stored_cameras = manifest.get("cameras")
    if not isinstance(stored_cameras, Mapping) or manifest.get("recordable") is not True:
        return ManifestResult("could not be checked", {}, {},
                              "manifest enumeration was not recordable")
    names = tuple(cam_names or stored_cameras.keys())
    if not names or any(name not in stored_cameras for name in names):
        return ManifestResult("could not be checked", {}, {},
                              "camera enumeration is incomplete")
    try:
        current = _current_enumeration(Path(f_loc), names, subfolder_string,
                                       n_lim, include_digest)
    except (OSError, ValueError, TypeError):
        return ManifestResult("could not be checked", {}, {},
                              "current image listing failed")
    if current.get("recordable") is not True:
        return ManifestResult("could not be checked", {}, {},
                              "current image listing was not recordable")

    photo_names: dict[str, tuple[str, ...]] = {}
    unavailable: dict[str, tuple[str, ...]] = {}
    mismatch = False
    for camera in names:
        recorded = stored_cameras[camera]
        current_records = current["cameras"][camera]
        if not isinstance(recorded, list) or any(not isinstance(item, Mapping) for item in recorded):
            return ManifestResult("could not be checked", {}, {},
                                  "stored camera enumeration is malformed")
        recorded_names = tuple(str(item["name"]) for item in recorded if isinstance(item.get("name"), str))
        current_by_name = {item["name"]: item for item in current_records}
        photo_names[camera] = recorded_names
        missing = tuple(name for name in recorded_names if name not in current_by_name)
        unavailable[camera] = missing
        if missing or tuple(item["name"] for item in current_records) != recorded_names:
            mismatch = True
        for old in recorded:
            fresh = current_by_name.get(old.get("name"))
            if fresh is None:
                continue
            changed = old.get("size") != fresh.get("size") or old.get("mtime_ns") != fresh.get("mtime_ns")
            if include_digest and old.get("sha256") != fresh.get("sha256"):
                changed = True
            mismatch = mismatch or changed
    return ManifestResult("mismatch" if mismatch else "verified by manifest",
                          photo_names, unavailable)
