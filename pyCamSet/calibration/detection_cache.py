"""
The detection cache: which slot a detection pass writes, and whether a slot
found on disk may be trusted.

One file per slot, an ``.npz`` holding three members::

    identity   json   the target, cameras and image cap it was made for
    meta       json   camera names, image count, resolutions, format version
    data       array  the detection itself

The file is moved into place rather than written in place, and names no
Python class. A cache that cannot be confirmed is a miss, and the pass
redetects.
"""
from __future__ import annotations

import json
import logging
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from pyCamSet.calibration_targets import TargetDetection
from pyCamSet.calibration_targets.core.target_registry import spec_of
from pyCamSet.utils.paths import long_path

logger = logging.getLogger(__name__)

#: Bumped when the members above change meaning. An unknown version is a miss.
FORMAT_VERSION = 1

#: What a damaged, foreign or half-written file raises. All mean: redetect.
_UNREADABLE = (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile)


def detector_backend_of(target) -> str | None:
    """
    The detector a target object is read with, when it says.

    A target readable only one way (ChArUco2, PuzzleBoard) has no
    ``marker_backend``, and is read with its only detector.

    :param target: any calibration target
    :return: a key of the target's ``DETECTOR_BACKENDS``, or None
    """
    backend = getattr(target, "marker_backend", None)
    if backend:
        return str(backend)
    backends = tuple(getattr(type(target), "DETECTOR_BACKENDS", None) or ())
    return backends[0] if len(backends) == 1 else None


def detection_cache_name(calibration_target, camset: Any | None = None,
                         upscale_factor: int = 1) -> str:
    """The cache slot a detection pass reads and writes.

    Each setting that changes what is detected gets its own slot.
    ``pyCamSet.workflow.detections.detection_cache_name`` states this same
    rule from the workflow side, which must not import this module.
    """
    name = "detected_datapoints"
    if camset is not None:
        name += "_with_calib"
    if upscale_factor != 1:
        name += f"_upscale{upscale_factor}x"
    if detector_backend_of(calibration_target) == "aruco2":
        name += "_aruco2"
    return name + ".npz"


def _target_identity(calibration_target, cam_names: list[str],
                      n_lim: int | None,
                      camset: Any | None = None) -> dict | None:
    """The identity a detection cache is checked against.

    None -- always a miss, and nothing written -- for a detection biased by
    a camset, which is not fingerprinted, or for an unregistered target.
    """
    if camset is not None:
        return None
    try:
        spec = spec_of(calibration_target)
    except ValueError:
        return None
    return {"target_spec": spec, "cam_names": sorted(cam_names), "n_lim": n_lim}


def _identity_text(payload: dict) -> str:
    # Canonical, so an identity written now compares equal to the same one
    # read back as JSON.
    return json.dumps(payload, sort_keys=True, default=str)


def _open_cache(cache_path: Path, calibration_target, cam_names: list[str],
                 n_lim: int | None, camset: Any | None):
    """The opened archive of a cache whose identity matches this call.

    Reads the ``identity`` member only, not the detection array beside it.

    :return: ``(archive, meta)``, which the caller must close, or None
    """
    identity = _target_identity(calibration_target, cam_names, n_lim, camset=camset)
    if identity is None:
        return None
    try:
        archive = np.load(long_path(cache_path), allow_pickle=False)
    except _UNREADABLE:
        # Missing, damaged, or not one of ours -- a cache written before
        # this format is a pickle, and reads as a miss here.
        return None
    try:
        if _identity_text(json.loads(str(archive["identity"]))) != _identity_text(identity):
            raise ValueError("identity mismatch")
        meta = json.loads(str(archive["meta"]))
        if meta.get("format_version") != FORMAT_VERSION:
            raise ValueError("unknown format version")
    except _UNREADABLE:
        archive.close()
        return None
    return archive, meta


def cache_matches(cache_path: Path, calibration_target, cam_names: list[str],
                   n_lim: int | None, camset: Any | None = None) -> bool:
    """Whether *cache_path* was produced for this exact target, camera
    selection and image cap. False whenever that cannot be confirmed.

    Reads the file's identity alone, for a caller deciding about a cache
    rather than reading one.
    """
    opened = _open_cache(cache_path, calibration_target, cam_names, n_lim, camset)
    if opened is None:
        return False
    opened[0].close()
    return True


def load_verified_cache(cache_path: Path, calibration_target,
                         cam_names: list[str], n_lim: int | None,
                         camset: Any | None = None):
    """A confirmed cache hit, as the detection pass returns it.

    The identity is read from the same open as the detection it guards, so
    a concurrent writer cannot swap one for the other in between.

    :return: ``(detected, cam_res)``, or ``None`` for anything
        :func:`cache_matches` calls a miss
    """
    opened = _open_cache(cache_path, calibration_target, cam_names, n_lim, camset)
    if opened is None:
        return None
    archive, meta = opened
    try:
        detected = TargetDetection.from_arrays(
            meta["cam_names"], archive["data"], meta["max_ims"])
        cam_res = [tuple(res) for res in meta["cam_res"]]
    except _UNREADABLE + (TypeError,) as exc:
        logger.warning("Ignoring an unreadable detection cache %s: %s",
                       cache_path, exc)
        return None
    finally:
        archive.close()
    return detected, cam_res


def save_to_cache(detected: TargetDetection, cam_res: list[tuple],
                  cache_path: Path, calibration_target, cam_names: list[str],
                  n_lim: int | None, camset: Any | None = None) -> None:
    """Write a finished detection pass to its cache slot.

    Written beside the slot and moved onto it, so a reader sees one whole
    version or the other. A failure is logged rather than raised, and an
    unconfirmable identity (see :func:`_target_identity`) writes nothing.
    """
    identity = _target_identity(calibration_target, cam_names, n_lim, camset=camset)
    if identity is None:
        return
    names, data, max_ims = detected.as_arrays()
    meta = {"format_version": FORMAT_VERSION, "cam_names": names,
            "max_ims": max_ims, "cam_res": [list(res) for res in cam_res]}
    cache_path = long_path(cache_path)
    staging = cache_path.with_name(cache_path.name + ".partial")
    try:
        with open(staging, "wb") as fh:
            np.savez(fh,
                     identity=np.array(_identity_text(identity)),
                     meta=np.array(_identity_text(meta)),
                     data=np.empty(0) if data is None else data)
        os.replace(staging, cache_path)
    except OSError as exc:
        logger.warning(
            "Could not write the detection cache %s: %s; this pass's "
            "detections are returned uncached.", cache_path, exc)
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            pass
