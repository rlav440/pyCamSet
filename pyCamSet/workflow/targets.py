"""
Building a calibration target, and checking it against the run it will read.

A detection is stored as indices into a target's points, so a phase that
reuses another run's detections must build the same target the detections were
made against.  :func:`describe_target_mismatch` is where that is caught, while
the two can still be named -- otherwise it surfaces deep inside the target as
an out-of-bounds index that says nothing about targets.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

TARGET_CHOICES = ["Ccube", "ChArUco", "PuzzleBoard", "PuzzleBoardCube"]

#: The targets whose phase 1 detection reads ChArUco corners, and which
#: therefore accept the detector options below.
CHARUCO_BASED_TARGETS = {"Ccube", "ChArUco"}



#: The ChArUco detector options a phase may set: what each one is called, how
#: to read a typed value for it, and the prose describing it.  Kept as data
#: rather than as a literal, because it is a table of settings, it is edited
#: as a table, and as a literal it was half of this module.
#:
#: The prose is for whoever is asking -- a form's help text, a docstring, a
#: command line's --help -- and not for one toolkit, which is why it lives
#: here with the parameter rather than with the widgets.  ``value_type`` says
#: what kind of value the option takes, not what control should collect it.
CHARUCO_DETECTION_OPTION_METADATA: list[dict[str, Any]] = json.loads(
    (Path(__file__).parent / "charuco_detection_options.json")
    .read_text(encoding="utf-8")
)

def _parse_charuco_value(meta: dict[str, Any], raw_value: Any):
    key = meta["key"]
    parser = meta["parser_type"]
    value = raw_value
    if isinstance(value, str):
        value = value.strip()
    if value in (None, ""):
        value = meta["default"]

    if parser == "enum":
        choices = list(meta.get("choices", []))
        if value not in choices:
            raise ValueError(f"{key} must be one of {choices}.")
        return value

    if parser in {"int", "positive_int", "int_range"}:
        try:
            out = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be an integer.") from exc
        if parser == "positive_int" and out <= 0:
            raise ValueError(f"{key} must be >= 1.")
        if parser == "int_range":
            min_value = int(meta.get("min_value", out))
            max_value = int(meta.get("max_value", out))
            if out < min_value or out > max_value:
                raise ValueError(f"{key} must be between {min_value} and {max_value}.")
        return out

    if parser in {"float", "positive_float", "non_negative_float"}:
        try:
            out = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a number.") from exc
        if parser == "positive_float" and out <= 0:
            raise ValueError(f"{key} must be > 0.")
        if parser == "non_negative_float" and out < 0:
            raise ValueError(f"{key} must be >= 0.")
        return out

    if parser == "optional_json_matrix_3x3":
        if value in ("", None):
            return None
        try:
            data = json.loads(value) if isinstance(value, str) else value
        except json.JSONDecodeError as exc:
            raise ValueError(f"{key} must be valid JSON.") from exc
        if not isinstance(data, list) or len(data) != 3:
            raise ValueError(f"{key} must be a 3x3 JSON array.")
        for row in data:
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError(f"{key} must be a 3x3 JSON array.")
            for elem in row:
                try:
                    float(elem)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{key} entries must be numeric.") from exc
        return data

    if parser == "optional_json_vector":
        if value in ("", None):
            return None
        try:
            data = json.loads(value) if isinstance(value, str) else value
        except json.JSONDecodeError as exc:
            raise ValueError(f"{key} must be valid JSON.") from exc
        if not isinstance(data, list):
            raise ValueError(f"{key} must be a JSON array.")
        for elem in data:
            if isinstance(elem, list):
                for sub_elem in elem:
                    try:
                        float(sub_elem)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"{key} entries must be numeric.") from exc
            else:
                try:
                    float(elem)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{key} entries must be numeric.") from exc
        return data

    raise ValueError(f"Unsupported parser type {parser!r} for {key}.")


def collect_charuco_detection_options(raw_options: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate and normalise ChArUco detector options from raw text values."""
    parsed_by_key: dict[str, Any] = {}
    for meta in CHARUCO_DETECTION_OPTION_METADATA:
        key = meta["key"]
        parsed_by_key[key] = _parse_charuco_value(meta, raw_options.get(key))

    min_key = "DetectorParameters.adaptiveThreshWinSizeMin"
    max_key = "DetectorParameters.adaptiveThreshWinSizeMax"
    if parsed_by_key[max_key] < parsed_by_key[min_key]:
        raise ValueError(f"{max_key} must be >= {min_key}.")

    options: dict[str, dict[str, Any]] = {
        "DetectorParameters": {},
        "CharucoParameters": {},
        "RefineParameters": {},
    }
    meta_by_key = {meta["key"]: meta for meta in CHARUCO_DETECTION_OPTION_METADATA}
    for key, value in parsed_by_key.items():
        group, name = key.split(".", 1)
        if value is None and meta_by_key.get(key, {}).get("drop_if_none", False):
            continue
        options[group][name] = value
    return options


def build_target(
    target_type: str,
    n_points: int,
    length: float,
    charuco_detection_options: dict[str, dict[str, Any]] | None = None,
    border_fraction: float = 0.1,
    marker_fraction: float = 0.8,
    marker_backend: str = "aruco1",
    aruco_dict: int | None = None,
    legacy: bool = False,
    # ChArUco, when it is not square.  A square board is the ordinary case
    # and says so once, as n_points; these are for a board that is not.
    charuco_squares_x: int | None = None,
    charuco_squares_y: int | None = None,
    # PuzzleBoard-only parameters:
    num_squares_x: int = 105,
    num_squares_y: int = 148,
    square_size: float = 2.0,
    start_x: int = 0,
    start_y: int = 0,
    paper_width: float = 210.0,
    paper_height: float = 297.0,
    min_width: int = 4,
    # PuzzleBoardCube-only parameters:
    pbc_n_points: int = 20,
    pbc_length: float = 200.0,
):
    """Construct a calibration target from a phase's target parameters."""
    from pyCamSet.calibration_targets.target_Ccube import Ccube
    from pyCamSet.calibration_targets.target_charuco import ChArUco

    if target_type == "Ccube":
        ccube_kwargs: dict[str, Any] = {
            "n_points": n_points,
            "length": length,
            "border_fraction": border_fraction,
            "marker_backend": marker_backend,
            "legacy": legacy,
            "detection_options": charuco_detection_options,  # Ccube detection also runs through ChArUco boards.
        }
        if aruco_dict is not None:
            ccube_kwargs["aruco_dict"] = aruco_dict
        return Ccube(**ccube_kwargs)
    if target_type == "ChArUco":
        charuco_kwargs: dict[str, Any] = {
            "num_squares_x": charuco_squares_x or n_points,
            "num_squares_y": charuco_squares_y or n_points,
            "square_size": length,
            "marker_fraction": marker_fraction,
            "marker_backend": marker_backend,
            "legacy": legacy,
            "detection_options": charuco_detection_options,
        }
        if aruco_dict is not None:
            charuco_kwargs["a_dict"] = aruco_dict
        return ChArUco(**charuco_kwargs)
    if target_type == "PuzzleBoard":
        from pyCamSet.calibration_targets.target_puzzleboard import PuzzleBoard

        return PuzzleBoard(
            num_squares_x=num_squares_x,
            num_squares_y=num_squares_y,
            square_size=square_size,
            start_x=start_x,
            start_y=start_y,
            paper_width=paper_width,
            paper_height=paper_height,
            min_width=min_width,
            detection_options=charuco_detection_options,
        )
    if target_type == "PuzzleBoardCube":
        from pyCamSet.calibration_targets.target_puzzleboard_cube import PuzzleBoardCube

        return PuzzleBoardCube(
            n_points=pbc_n_points,
            length=pbc_length,
            min_width=min_width,
            detection_options=charuco_detection_options,
        )
    raise ValueError(f"Unknown target type: {target_type!r}")


def target_from_params(params: dict):
    """
    Build the calibration target a phase's parameters describe.

    Reads the same keys a phase collects and a run records, so a saved run can
    rebuild the target it was made with.

    :param params: a phase's parameters, or a saved run's ``params``
    """
    return build_target(
        params.get("target_type", "Ccube"),
        params.get("n_points", 6),
        params.get("length", 30.0),
        charuco_detection_options=params.get("charuco_detection_options"),
        border_fraction=params.get("border_fraction", 0.1),
        marker_fraction=params.get("marker_fraction", 0.8),
        marker_backend=params.get("marker_backend", "aruco1"),
        aruco_dict=params.get("aruco_dict"),
        legacy=params.get("legacy", False),
        charuco_squares_x=params.get("charuco_squares_x"),
        charuco_squares_y=params.get("charuco_squares_y"),
        num_squares_x=params.get("num_squares_x", 105),
        num_squares_y=params.get("num_squares_y", 148),
        square_size=params.get("square_size", 2.0),
        start_x=params.get("start_x", 0),
        start_y=params.get("start_y", 0),
        paper_width=params.get("paper_width", 210.0),
        paper_height=params.get("paper_height", 297.0),
        min_width=params.get("min_width", 4),
        pbc_n_points=params.get("pbc_n_points", 20),
        pbc_length=params.get("pbc_length", 200.0),
    )

# ---------------------------------------------------------------------------
# Matching a phase's target to the run whose detections it reuses
# ---------------------------------------------------------------------------

# The target fields that decide the point layout, per target type.  A
# detection stores its keys as indices into that layout, so detections made
# against a different one address points that do not exist: reusing them
# fails deep in the target as "index 80 is out of bounds for axis 1 with
# size 25" rather than as anything about targets.
#
# Fields that only affect how markers are read -- the backend, the ArUco
# dictionary, the detector tuning -- are deliberately absent.  They change
# which points are found, not what a found key means.
TARGET_IDENTITY_KEYS: dict[str, tuple[str, ...]] = {
    "Ccube": ("n_points", "length", "border_fraction"),
    "ChArUco": ("n_points", "length", "marker_fraction",
                "charuco_squares_x", "charuco_squares_y"),
    "PuzzleBoard": (
        "num_squares_x", "num_squares_y", "square_size",
        "start_x", "start_y", "paper_width", "paper_height", "min_width",
    ),
    "PuzzleBoardCube": ("pbc_n_points", "pbc_length", "min_width"),
}

# What each identity key is called in the interface, for the message a
# person reads when the two disagree.
TARGET_KEY_LABELS: dict[str, str] = {
    "n_points": "n_points",
    "length": "length",
    "border_fraction": "border fraction",
    "marker_fraction": "marker fraction",
    "num_squares_x": "squares across",
    "num_squares_y": "squares down",
    "square_size": "square size",
    "start_x": "start x",
    "start_y": "start y",
    "paper_width": "paper width",
    "paper_height": "paper height",
    "min_width": "min width",
    "charuco_squares_x": "squares across",
    "charuco_squares_y": "squares down",
    "pbc_n_points": "n_points",
    "pbc_length": "length",
}


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


def describe_target_mismatch(
        run_params: dict, current_params: dict) -> list[str]:
    """
    Where a saved run's target and the current settings disagree.

    Only the fields that decide the point layout are compared, and only the
    ones that matter for the target type in question.

    :param run_params: the parameters of the run supplying the detections
    :param current_params: the target settings a phase is about to use
    :return: one sentence per disagreement, empty when they match
    """
    if not run_params or not current_params:
        return []

    run_type = str(run_params.get("target_type", "") or "")
    current_type = str(current_params.get("target_type", "") or "")
    if not run_type or not current_type:
        return []
    if run_type != current_type:
        return [f"target type: the run used {run_type}, "
                f"these settings say {current_type}"]

    differences = []
    for key in TARGET_IDENTITY_KEYS.get(run_type, ()):
        if key not in run_params or key not in current_params:
            continue
        if not _same_value(run_params[key], current_params[key]):
            label = TARGET_KEY_LABELS.get(key, key)
            differences.append(
                f"{label}: the run used {run_params[key]}, "
                f"these settings say {current_params[key]}")
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

