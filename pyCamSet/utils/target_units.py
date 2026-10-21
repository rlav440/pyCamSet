"""Identify target-coordinate and persisted-solution units without Qt.

"""
from __future__ import annotations

from typing import Any


METRES = "m"
MILLIMETRES = "mm"
UNVERIFIED = "unverified"
LEGACY_UNVERIFIED = "legacy/unverified"

_TARGET_UNITS = {
    "ChArUco": METRES,
    "ChArUco2": METRES,
    "Ccube": METRES,
    "Ccube2": METRES,
    "CIco": METRES,
    "CIco2": METRES,
    "PuzzleBoard": MILLIMETRES,
    "PuzzleBoardCube": METRES,
    "PuzzleBoardIco": METRES,
}

_SQUARE_SIZE_UNITS = {
    "ChArUco": METRES,
    "ChArUco2": METRES,
    "Ccube": METRES,
    "Ccube2": METRES,
    "CIco": METRES,
    "CIco2": METRES,
    "PuzzleBoard": MILLIMETRES,
    "PuzzleBoardCube": MILLIMETRES,
    "PuzzleBoardIco": METRES,
}


def target_point_data_units(target: Any) -> str:
    """Return the known units of a target's rebuilt ``point_data``."""
    class_name = type(target).__name__
    module_name = type(target).__module__.casefold()
    if "pycamset.calibration_targets" not in module_name:
        return UNVERIFIED
    return _TARGET_UNITS.get(class_name, UNVERIFIED)


def saved_solution_units(target_config: dict[str, Any] | None) -> str:
    """Return units recorded for saved geometry, never infer legacy units."""
    if not isinstance(target_config, dict):
        return LEGACY_UNVERIFIED
    metres_per_unit = target_config.get("point_data_unit_m")
    if metres_per_unit == 1.0:
        return METRES
    if metres_per_unit == 0.001:
        return MILLIMETRES
    return LEGACY_UNVERIFIED


def point_data_unit_metres(target: Any) -> float | None:
    """Return metres per rebuilt point-data unit, or ``None`` if unknown."""
    units = target_point_data_units(target)
    if units == METRES:
        return 1.0
    if units == MILLIMETRES:
        return 0.001
    return None


def square_size_in_point_data_units(target: Any) -> float:
    """Convert declared square size to the units used by ``point_data``.

    Flat PuzzleBoard stores both values in millimetres. PuzzleBoardCube stores
    ``square_size`` in millimetres but its point coordinates in metres; other
    supported metric targets use metre coordinates and millimetre declarations.
    """
    import math

    try:
        declared = float(target.square_size)
    except (AttributeError, TypeError, ValueError):
        return math.nan
    square_units = _SQUARE_SIZE_UNITS.get(type(target).__name__)
    point_units = target_point_data_units(target)
    # Lightweight/custom targets without a registered unit contract express
    # their declared spacing in the same coordinates as their point data.
    if square_units is None and point_units == UNVERIFIED:
        return declared
    if square_units == point_units:
        return declared
    if square_units == MILLIMETRES and point_units == METRES:
        return declared / 1000.0
    if square_units == METRES and point_units == MILLIMETRES:
        return declared * 1000.0
    return math.nan


def millimetres_per_point_data_unit(target: Any) -> float | None:
    """Return the coordinate-to-mm scale, or ``None`` if it is unknown."""
    units = target_point_data_units(target)
    if units == METRES:
        return 1000.0
    if units == MILLIMETRES:
        return 1.0
    return None
