'''
Purpose: Headless registry of supported ArUco marker backends and dictionaries.
Status: Active. Shared by target construction and future GUI consumers without
        importing presentation-layer modules.
Future: Add a backend adapter record only when a third marker implementation is
        supported by the target APIs and covered by focused tests.
'''

from __future__ import annotations

import importlib


# Keep backend identifiers stable because they are persisted in target settings.
ARUCO1_BACKEND = "aruco1"
ARUCO2_BACKEND = "aruco2"
SUPPORTED_MARKER_BACKENDS = (ARUCO1_BACKEND, ARUCO2_BACKEND)

# These labels are presentation-neutral strings suitable for a caller-owned UI.
MARKER_BACKEND_LABELS = {
    "ArUco 1 (OpenCV)": ARUCO1_BACKEND,
    "ArUco 2 (aruco2)": ARUCO2_BACKEND,
}

# OpenCV exposes these 22 predefined dictionaries as integer ids 0 through 21.
ARUCO1_DICT_NAMES = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_5X5_1000",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_6X6_1000",
    "DICT_7X7_50",
    "DICT_7X7_100",
    "DICT_7X7_250",
    "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16h5",
    "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10",
    "DICT_APRILTAG_36h11",
    "DICT_ARUCO_MIP_36h12",
]

# ArUco2 adds the two dictionaries absent from OpenCV's predefined set.
ARUCO2_DICT_NAMES = ARUCO1_DICT_NAMES + [
    "DICT_ALVAR_5X5_256",
    "DICT_ALVAR_7X7_1000",
]


def validate_marker_backend(marker_backend: str) -> str:
    """Validate and return a persisted marker-backend identifier."""
    if marker_backend not in SUPPORTED_MARKER_BACKENDS:
        raise ValueError(
            "marker_backend must be 'aruco1' or 'aruco2', "
            f"got {marker_backend!r}"
        )
    return marker_backend


def dict_names_for_backend(marker_backend: str) -> list[str]:
    """Return a copy of the dictionary names supported by one backend."""
    validate_marker_backend(marker_backend)
    if marker_backend == ARUCO2_BACKEND:
        return list(ARUCO2_DICT_NAMES)
    return list(ARUCO1_DICT_NAMES)


def marker_backend_available(marker_backend: str) -> bool:
    """Return whether a backend can be imported in this Python environment."""
    validate_marker_backend(marker_backend)
    if marker_backend == ARUCO1_BACKEND:
        return True
    try:
        importlib.import_module("aruco2")
    except (ImportError, ModuleNotFoundError, OSError):
        # A broken or partially initialised optional import is unavailable.
        return False
    return True


def available_marker_backends() -> tuple[str, ...]:
    """Return supported backends whose optional dependencies are available."""
    return tuple(
        backend
        for backend in SUPPORTED_MARKER_BACKENDS
        if marker_backend_available(backend)
    )


def marker_backend_availability_text(marker_backend: str) -> str:
    """Return a concise, caller-neutral availability status string."""
    validate_marker_backend(marker_backend)
    if marker_backend == ARUCO1_BACKEND:
        return "backend: built-in OpenCV ArUco"
    if marker_backend_available(marker_backend):
        return "aruco2: available"
    return "aruco2: not installed - pip install aruco2"
