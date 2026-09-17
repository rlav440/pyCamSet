"""
The calibration targets this machine has detected with before.

A target is described by a handful of numbers that have to match the printed
object exactly -- the squares across it, the size of one of them, the marker
dictionary -- and getting one of them wrong is not an error anyone sees until
the detections come back empty.  They were retyped from memory every session.

Kept the same way the image folders are, and for the same reason: it is a
choice someone makes before any of the work exists, so nothing in the
workspace records it.  See :mod:`pyCamSet.workflow.recent_folders`.
"""
from __future__ import annotations

import json

from pyCamSet.workflow.targets import TARGET_KEY_TYPE
from pyCamSet.workflow.user_config import MAX_RECENT, read_list, write_list

_FILE_NAME = "recent_targets.json"
_KEY = "targets"

#: Spec keys that say how a target is read, not which target it is.
_READING_KEYS = frozenset({"detection_options", "marker_backend"})


def load_recent_targets() -> list[dict]:
    """
    The targets last detected with, most recent first.

    :return: the target specs, as :mod:`pyCamSet.workflow.targets` reads them
    """
    targets = [t for t in read_list(_FILE_NAME, _KEY)
               if isinstance(t, dict) and t.get(TARGET_KEY_TYPE)]
    return targets[:MAX_RECENT]


def remember_target(spec: dict) -> None:
    """
    Put a target at the top of the list.

    Called where the target is known to be real -- a phase that has accepted
    it and is about to detect with it -- rather than on every edit of a spin
    box on the way to it.

    :param spec: the target spec to remember
    """
    if not spec or not spec.get(TARGET_KEY_TYPE):
        return
    identity = _identity(spec)
    kept = [t for t in load_recent_targets() if _identity(t) != identity]
    write_list(_FILE_NAME, _KEY, [dict(spec), *kept])


def forget_target(spec: dict) -> None:
    """
    Drop a target from the list.

    :param spec: the target spec to forget
    """
    if not spec:
        return
    identity = _identity(spec)
    write_list(_FILE_NAME, _KEY,
               [t for t in load_recent_targets() if _identity(t) != identity])


def _identity(spec: dict) -> str:
    """
    What makes two remembered targets the same one.

    Everything the target is built from, so a board of a different size is a
    different entry -- but not the detector it was read with or that
    detector's tuning, which are how a marker is read rather than what the
    target is, and which would otherwise fill the list with the same board
    over and over.

    :param spec: the target spec
    """
    return json.dumps(
        {k: v for k, v in spec.items() if k not in _READING_KEYS},
        sort_keys=True, default=str)
