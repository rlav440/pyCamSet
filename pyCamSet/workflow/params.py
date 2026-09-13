"""
Turning what someone typed into what a phase can run.

Every value a phase takes arrives as text -- from a form field, a config file,
a command line -- and every one of them has a rule.  The rules live here so
there is one place that decides what a valid ``n_lim`` is, and one message
saying so; the interface's job is only to say which field the text came from.
"""
from __future__ import annotations

import json
import math
from typing import Any, Optional

from pyCamSet.workflow.targets import (
    CHARUCO_BASED_TARGETS,
    target_spec_of,
    describe_target_mismatch,
    target_mismatch_message,
    target_params_of_run,
)


class ParamError(ValueError):
    """A parameter a phase cannot run with, described for the person who set it."""


def as_int(text: Any, field: str) -> int:
    """Parse *text* as an integer."""
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        raise ParamError(f"{field} must be a whole number.") from None


def as_positive_int(text: Any, field: str) -> int:
    """Parse *text* as an integer greater than zero."""
    value = as_int(text, field)
    if value <= 0:
        raise ParamError(f"{field} must be a positive integer.")
    return value


def as_optional_positive_int(text: Any, field: str) -> Optional[int]:
    """Parse *text* as a positive integer, or None when it is blank."""
    if not str(text or "").strip():
        return None
    return as_positive_int(text, field)


def as_float(text: Any, field: str) -> float:
    """Parse *text* as a number."""
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        raise ParamError(f"{field} must be a number.") from None


def as_positive_float(text: Any, field: str) -> float:
    """Parse *text* as a finite number greater than zero."""
    value = as_float(text, field)
    if not math.isfinite(value) or value <= 0.0:
        raise ParamError(f"{field} must be finite and greater than zero.")
    return value


def as_json_object(text: Any, field: str) -> Optional[dict]:
    """Parse *text* as a JSON object, or None when it is blank."""
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ParamError(f"{field}: {exc}") from None
    if not isinstance(value, dict):
        raise ParamError(f"{field} must be a JSON object.")
    return value


def as_outlier_mode(text: Any) -> str:
    """
    Normalise an outlier-rejection setting to the ``y``/``n`` the solver takes.

    ``ask`` means the solver prompts on stdin, which a phase run from a thread
    or a server has no way to answer, so it is read as enabled.  Anything
    unrecognised is read as disabled: rejecting observations is the surprising
    outcome, so it is never what an unreadable value turns into.
    """
    value = str(text or "").strip().lower()
    if value in {"y", "yes", "true", "1", "on", "enabled", "ask"}:
        return "y"
    return "n"


def require_image_folder(text: Any) -> str:
    """Return the image folder, refusing a blank one."""
    folder = str(text or "").strip()
    if not folder:
        raise ParamError("Image folder is required.")
    return folder


def require_marker_backend(params: dict) -> None:
    """
    Refuse a marker backend the target needs and the install does not have.

    PuzzleBoard and PuzzleBoardCube never read ArUco markers, so the backend
    is not their business and an absent one does not stop them.

    :param params: a phase's collected parameters
    :raises ParamError: when the selected target cannot read its markers
    """
    from pyCamSet.calibration_targets.backend_registry import marker_backend_available

    spec = target_spec_of(params)
    if spec.get("type") not in CHARUCO_BASED_TARGETS:
        return
    backend = spec.get("marker_backend", "aruco1")
    if marker_backend_available(backend):
        return
    raise ParamError(
        "ArUco 2 (aruco2) is selected but the 'aruco2' package is not "
        "installed. Install it with `pip install aruco2` or switch the "
        "marker backend to ArUco 1 (OpenCV)."
    )


def require_target_match(run: Optional[dict], params: dict) -> None:
    """
    Refuse a target that cannot read the detections it is about to be given.

    :param run: the run supplying the detections, or None to skip the check
    :param params: the target settings the phase is about to use
    :raises ParamError: when the two describe different point layouts
    """
    if run is None:
        return
    differences = describe_target_mismatch(target_params_of_run(run), params)
    if differences:
        raise ParamError(
            target_mismatch_message(str(run.get("run_id", "unknown")), differences))
