"""
Building a calibration target, and checking it against the run it will read.

A detection is stored as indices into a target's points, so a phase that
reuses another run's detections must build the same target the detections were
made against.  :func:`describe_target_mismatch` is where that is caught, while
the two can still be named -- otherwise it surfaces deep inside the target as
an out-of-bounds index that says nothing about targets.
"""
from __future__ import annotations

import math
from typing import Any

from pyCamSet.calibration_targets.target_registry import (
    TARGET_NAMES,
    TYPE_KEY as TARGET_KEY_TYPE,
    build_target,
)

#: Where a phase's parameters keep the target's spec.
TARGET_KEY = "target"

TARGET_CHOICES = list(TARGET_NAMES)

#: The targets whose phase 1 detection reads ChArUco corners, and which
#: therefore accept the detector options in
#: :mod:`pyCamSet.calibration_targets.charuco_parameters`.
CHARUCO_BASED_TARGETS = {"Ccube", "ChArUco"}


def target_of_params(params: dict):
    """
    Build the calibration target a phase's parameters describe.

    :param params: a phase's parameters, or a saved run's ``params``
    :raises ValueError: when the parameters carry no target spec
    """
    return build_target(target_spec_of(params))


def target_spec_of(params: dict) -> dict:
    """
    The target spec inside a phase's parameters.

    :param params: a phase's parameters, or a saved run's ``params``
    :raises ValueError: when there is none -- which is what a run recorded
        before targets were kept as a spec looks like
    """
    spec = (params or {}).get(TARGET_KEY)
    if not spec:
        raise ValueError(
            "These parameters carry no target spec, and need to re-run Phase 1 "
        )
    return dict(spec)


# ---------------------------------------------------------------------------
# Matching a phase's target to the run whose detections it reuses
# ---------------------------------------------------------------------------

# A detection stores its keys as indices into the target's points, so
# detections made against a different layout address points that do not
# exist: reusing them fails deep in the target as "index 80 is out of bounds
# for axis 1 with size 25" rather than as anything about targets.
#
# So two targets have to agree on everything except the fields below, which
# change how markers are read or how a target is drawn rather than what a
# found key means.  Stated as what to ignore rather than what to compare,
# because a spec is the target's own constructor arguments -- a new target,
# or a new argument on an existing one, is then compared by default instead
# of being silently left out of the check.
READING_ONLY_FIELDS = frozenset({
    "a_dict",              # ChArUco's marker dictionary
    "aruco_dict",          # and Ccube's name for it
    "marker_backend",      # which library reads the markers
    "detection_options",   # the detector's tuning
    "legacy",              # which of two marker layouts to expect
    "draw_res",            # only affects the printed image
    "line_fraction",       # likewise
})


def describe_target(params: dict) -> str:
    """
    A target in one line, for a log or a status label.

    Names the target and the arguments that decide its point layout, which
    are the ones someone reading a run wants to check.

    :param params: a phase's parameters, or a saved run's ``params``
    """
    spec = target_spec_of(params)
    settings = ", ".join(
        f"{key}={value}" for key, value in sorted(spec.items())
        if key != TARGET_KEY_TYPE and key not in READING_ONLY_FIELDS)
    return f"{spec.get(TARGET_KEY_TYPE, 'unknown')}({settings})"


def target_params_of_run(run: dict | None) -> dict:
    """
    The target settings a saved run was produced with.

    :param run: a run metadata dictionary, or None
    :return: the run's parameters, empty when there is no run
    """
    if not run:
        return {}
    return dict(run.get("params") or {})


def _same_value(left: Any, right: Any) -> bool:
    """Whether two target settings agree, comparing numbers as numbers."""
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    except (TypeError, ValueError):
        return str(left) == str(right)


def describe_target_mismatch(run_params: dict, current_params: dict) -> list[str]:
    """
    Where a saved run's target and the current settings disagree.

    Only the fields that decide the point layout are compared: everything in
    the two specs except :data:`READING_ONLY_FIELDS`.

    :param run_params: the parameters of the run supplying the detections
    :param current_params: the settings a phase is about to use
    :return: one sentence per disagreement, empty when they match
    :raises ValueError: when either side carries no target spec, because a
        target that cannot be read cannot be checked either
    """
    run_spec = target_spec_of(run_params)
    current_spec = target_spec_of(current_params)

    run_type, current_type = run_spec.get(TARGET_KEY_TYPE), current_spec.get(TARGET_KEY_TYPE)
    if run_type != current_type:
        return [f"target type: the run used {run_type}, "
                f"these settings say {current_type}"]

    differences = []
    for key in sorted(set(run_spec) | set(current_spec)):
        if key in READING_ONLY_FIELDS or key == TARGET_KEY_TYPE:
            continue
        if key not in run_spec or key not in current_spec:
            continue
        if not _same_value(run_spec[key], current_spec[key]):
            differences.append(
                f"{key}: the run used {run_spec[key]}, "
                f"these settings say {current_spec[key]}")
    return differences


def target_mismatch_message(run_id: str, differences: list[str]) -> str:
    """
    What to tell someone whose target does not match its detections.

    :param run_id: the run supplying the detections
    :param differences: the output of :func:`describe_target_mismatch`
    :return: the message to show
    """
    listed = "\n".join(f"  - {d}" for d in differences)
    return (
        f"The target settings do not match run {run_id}, which produced the "
        f"detections this phase would use:\n\n{listed}\n\n"
        f"Detections are stored as indices into the target's points, so a "
        f"target of a different size cannot read them. Either set the target "
        f"to match the run, or choose a run detected with this target."
    )
