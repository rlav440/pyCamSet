"""
Metadata table describing ChArUco detector parameters exposed to the
Optimisation tab.

This module is the single source of truth used by both the GUI
(`pyCamSet.gui.optimisation_tab`) and the headless trial runner
(`pyCamSet.optimisation.optimisation_worker`).  It mirrors the calibria-style
metadata approach: each entry describes one parameter that a user can either
fix to a value or expose to Optuna sampling.

Only numeric parameters (``int`` / ``float``) are optimisable in v1, per the
Optimisation Tab Specification §4.5.  Boolean, categorical, enum and JSON-like
parameters are not exposed for sampling here.

Conventions
-----------
- ``key`` matches the attribute name on the OpenCV parameter object exactly.
- ``group`` is the OpenCV parameter-object group (one of
  ``"CharucoParameters"``, ``"DetectorParameters"``, ``"RefineParameters"``)
  used by :func:`pyCamSet.calibration_targets.charuco_detection._group_options`.
- ``min`` / ``max`` are absolute hard bounds.  Per-trial search bounds chosen
  by the user must lie inside this interval (§16.1).
- ``default`` is OpenCV's effective default and is used as the fixed value
  shown initially in the GUI and used for the §18 baseline detection run.
- ``step`` and ``decimals`` are GUI hints; the backend uses ``dtype`` for
  Optuna ``suggest_int`` vs ``suggest_float`` selection.

This module is intentionally GUI-free and safe to import in headless tests.
"""
from __future__ import annotations

from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Metadata records
# ---------------------------------------------------------------------------

CHARUCO_PARAMETER_METADATA: list[dict[str, Any]] = [
    # ---- aruco.DetectorParameters --------------------------------------
    {
        "key": "adaptiveThreshWinSizeMin",
        "label": "Adaptive Thresh Win Size Min",
        "group": "DetectorParameters",
        "dtype": "int",
        "default": 3,
        "min": 3,
        "max": 99,
        "step": 2,
        "concept": (
            "Minimum adaptive-threshold window size used during marker "
            "candidate extraction.  Must be odd."
        ),
    },
    {
        "key": "adaptiveThreshWinSizeMax",
        "label": "Adaptive Thresh Win Size Max",
        "group": "DetectorParameters",
        "dtype": "int",
        "default": 23,
        "min": 3,
        "max": 199,
        "step": 2,
        "concept": (
            "Maximum adaptive-threshold window size used during marker "
            "candidate extraction.  Must be odd and >= min."
        ),
    },
    {
        "key": "adaptiveThreshWinSizeStep",
        "label": "Adaptive Thresh Win Size Step",
        "group": "DetectorParameters",
        "dtype": "int",
        "default": 10,
        "min": 1,
        "max": 50,
        "step": 1,
        "concept": "Step between successive adaptive threshold window sizes.",
    },
    {
        "key": "adaptiveThreshConstant",
        "label": "Adaptive Thresh Constant",
        "group": "DetectorParameters",
        "dtype": "float",
        "default": 7.0,
        "min": 0.0,
        "max": 50.0,
        "step": 0.5,
        "decimals": 2,
        "concept": "Constant subtracted from the adaptive threshold mean.",
    },
    {
        "key": "minMarkerPerimeterRate",
        "label": "Min Marker Perimeter Rate",
        "group": "DetectorParameters",
        "dtype": "float",
        "default": 0.03,
        "min": 0.001,
        "max": 4.0,
        "step": 0.001,
        "decimals": 3,
        "concept": (
            "Reject marker candidates with perimeter smaller than this "
            "fraction of the image side."
        ),
    },
    {
        "key": "maxMarkerPerimeterRate",
        "label": "Max Marker Perimeter Rate",
        "group": "DetectorParameters",
        "dtype": "float",
        "default": 4.0,
        "min": 0.05,
        "max": 8.0,
        "step": 0.05,
        "decimals": 3,
        "concept": (
            "Reject marker candidates with perimeter larger than this "
            "fraction of the image side."
        ),
    },
    {
        "key": "polygonalApproxAccuracyRate",
        "label": "Polygonal Approx Accuracy Rate",
        "group": "DetectorParameters",
        "dtype": "float",
        "default": 0.03,
        "min": 0.001,
        "max": 0.5,
        "step": 0.001,
        "decimals": 3,
        "concept": (
            "Accuracy parameter for the polygonal approximation of marker "
            "contours."
        ),
    },
    {
        "key": "minCornerDistanceRate",
        "label": "Min Corner Distance Rate",
        "group": "DetectorParameters",
        "dtype": "float",
        "default": 0.05,
        "min": 0.0,
        "max": 0.5,
        "step": 0.005,
        "decimals": 4,
        "concept": "Minimum distance between marker corners, relative to the marker perimeter.",
    },
    {
        "key": "minDistanceToBorder",
        "label": "Min Distance To Border (px)",
        "group": "DetectorParameters",
        "dtype": "int",
        "default": 3,
        "min": 0,
        "max": 100,
        "step": 1,
        "concept": "Minimum pixel distance of a marker corner to the image border.",
    },
    {
        "key": "minMarkerDistanceRate",
        "label": "Min Marker Distance Rate",
        "group": "DetectorParameters",
        "dtype": "float",
        "default": 0.05,
        "min": 0.0,
        "max": 0.5,
        "step": 0.005,
        "decimals": 4,
        "concept": "Minimum distance between any two markers, relative to perimeter.",
    },
    {
        "key": "cornerRefinementWinSize",
        "label": "Corner Refinement Win Size",
        "group": "DetectorParameters",
        "dtype": "int",
        "default": 5,
        "min": 1,
        "max": 50,
        "step": 1,
        "concept": "Window size for subpixel corner refinement.",
    },
    {
        "key": "cornerRefinementMaxIterations",
        "label": "Corner Refinement Max Iterations",
        "group": "DetectorParameters",
        "dtype": "int",
        "default": 30,
        "min": 1,
        "max": 500,
        "step": 1,
        "concept": "Maximum iterations for subpixel corner refinement.",
    },
    {
        "key": "cornerRefinementMinAccuracy",
        "label": "Corner Refinement Min Accuracy",
        "group": "DetectorParameters",
        "dtype": "float",
        "default": 0.1,
        "min": 0.0001,
        "max": 1.0,
        "step": 0.005,
        "decimals": 4,
        "concept": "Convergence threshold for subpixel corner refinement.",
    },
    {
        "key": "errorCorrectionRate",
        "label": "Error Correction Rate",
        "group": "DetectorParameters",
        "dtype": "float",
        "default": 0.6,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "decimals": 3,
        "concept": (
            "Fraction of marker bit errors that may be corrected.  0 disables "
            "error correction; 1 allows maximal correction."
        ),
    },
    # ---- aruco.CharucoParameters ---------------------------------------
    {
        "key": "minMarkers",
        "label": "Min Markers (Charuco)",
        "group": "CharucoParameters",
        "dtype": "int",
        "default": 2,
        "min": 0,
        "max": 4,
        "step": 1,
        "concept": (
            "Minimum number of identified ArUco markers neighbouring a Charuco "
            "corner before it is interpolated."
        ),
    },
]
"""Source-of-truth list of optimisable numeric ChArUco detector parameters.

See module docstring for conventions.  Iteration order is the recommended row
order for the GUI.
"""


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def metadata_by_key() -> dict[str, dict[str, Any]]:
    """Return ``{key: record}`` mapping for fast lookup."""
    return {entry["key"]: entry for entry in CHARUCO_PARAMETER_METADATA}


def numeric_keys() -> list[str]:
    """Return the keys of every numeric optimisable parameter, preserving order."""
    return [entry["key"] for entry in CHARUCO_PARAMETER_METADATA]


def default_fixed_settings() -> dict[str, dict[str, Any]]:
    """Return the default detector-options dict using each parameter's ``default``.

    The result is grouped into the OpenCV sub-dicts that
    :func:`pyCamSet.calibration_targets.charuco_detection.build_charuco_detector_components`
    expects.
    """
    result: dict[str, dict[str, Any]] = {}
    for entry in CHARUCO_PARAMETER_METADATA:
        group = entry["group"]
        result.setdefault(group, {})[entry["key"]] = entry["default"]
    return result


def assemble_detection_options(values: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Group a flat ``{key: value}`` dict back into the OpenCV sub-dict shape.

    Unknown keys are dropped silently to keep the call boundary forgiving
    (forward/backward compatibility with OpenCV builds, mirroring the
    behaviour of ``build_charuco_parameters``).
    """
    by_key = metadata_by_key()
    grouped: dict[str, dict[str, Any]] = {}
    for key, value in values.items():
        entry = by_key.get(key)
        if entry is None:
            continue
        grouped.setdefault(entry["group"], {})[key] = value
    return grouped


def coerce_value(entry: dict[str, Any], value: Any) -> Any:
    """Coerce *value* to the dtype declared by *entry*.

    Raises :class:`ValueError` when coercion is impossible.
    """
    dtype = entry["dtype"]
    if dtype == "int":
        return int(round(float(value)))
    if dtype == "float":
        return float(value)
    raise ValueError(f"Unsupported dtype {dtype!r} for parameter {entry['key']!r}")


def clamp_to_bounds(entry: dict[str, Any], value: Any) -> Any:
    """Coerce and clamp *value* to ``[entry['min'], entry['max']]``."""
    v = coerce_value(entry, value)
    lo = coerce_value(entry, entry["min"])
    hi = coerce_value(entry, entry["max"])
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def validate_parameter_row(
    entry: dict[str, Any],
    *,
    fixed_value: Any,
    optimise: bool,
    lower: Any | None = None,
    upper: Any | None = None,
) -> list[str]:
    """Validate one parameter row (§16.1).

    Returns a list of human-readable error strings.  An empty list means the
    row is valid.  This function is pure — it does not raise.
    """
    errors: list[str] = []
    label = entry.get("label", entry["key"])
    try:
        fv = coerce_value(entry, fixed_value)
    except (TypeError, ValueError):
        errors.append(f"{label}: fixed value is not a valid {entry['dtype']}.")
        return errors
    abs_lo = coerce_value(entry, entry["min"])
    abs_hi = coerce_value(entry, entry["max"])
    if fv < abs_lo or fv > abs_hi:
        errors.append(
            f"{label}: fixed value {fv} is outside allowed bounds [{abs_lo}, {abs_hi}]."
        )

    if optimise:
        if lower is None or upper is None:
            errors.append(f"{label}: lower and upper bounds must be provided when optimising.")
            return errors
        try:
            lo = coerce_value(entry, lower)
            hi = coerce_value(entry, upper)
        except (TypeError, ValueError):
            errors.append(f"{label}: bounds are not valid {entry['dtype']}s.")
            return errors
        if lo < abs_lo or hi > abs_hi:
            errors.append(
                f"{label}: bounds [{lo}, {hi}] exceed allowed range [{abs_lo}, {abs_hi}]."
            )
        if lo > hi:
            errors.append(f"{label}: lower bound {lo} exceeds upper bound {hi}.")
    return errors


def validate_all_rows(rows: Iterable[dict[str, Any]]) -> list[str]:
    """Run :func:`validate_parameter_row` over an iterable of row state dicts.

    Each row dict must contain at minimum the keys ``key``, ``fixed`` and
    ``optimise``; if ``optimise`` is true, ``lower`` and ``upper`` are also
    required.  Returns a flat list of error messages.
    """
    by_key = metadata_by_key()
    errors: list[str] = []
    for row in rows:
        entry = by_key.get(row["key"])
        if entry is None:
            errors.append(f"Unknown parameter key {row['key']!r}.")
            continue
        errors.extend(
            validate_parameter_row(
                entry,
                fixed_value=row.get("fixed", entry["default"]),
                optimise=bool(row.get("optimise", False)),
                lower=row.get("lower"),
                upper=row.get("upper"),
            )
        )
    return errors
