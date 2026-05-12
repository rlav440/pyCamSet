"""
Purpose: Share ChArUco detector-option construction across ChArUco-based targets.
Status: Active helper used by the ChArUco and Ccube targets.
Future: Extend this module if more ChArUco-based targets need the same detector setup.
"""
from __future__ import annotations  # Keep postponed annotations consistent with the package style.

import logging  # Emit compatibility warnings from one shared location.

import numpy as np  # Convert JSON-like matrix inputs to OpenCV-friendly arrays.
from cv2 import aruco  # Reuse OpenCV's ArUco/ChArUco APIs in one place.

_LOG = logging.getLogger(__name__)  # Reuse one logger for OpenCV constructor fallbacks.


def _group_options(detection_options: dict | None, group_name: str) -> dict:
    """Return one detector-option subgroup, normalised to a plain dict."""
    options = detection_options or {}  # Treat omitted detection options as empty input.
    group = options.get(group_name, {}) or {}  # Treat missing sub-groups as empty too.
    return dict(group)  # Copy to avoid mutating caller-owned option mappings.


def _coerce_corner_refinement_method(value):
    """Resolve enum-like string values to OpenCV constants when needed."""
    if isinstance(value, str):  # GUI values arrive as enum names.
        return getattr(aruco, value, value)  # Prefer the OpenCV constant when it exists.
    return value  # Pass through numeric/native values unchanged.


def build_charuco_parameters(detection_options: dict | None) -> aruco.CharucoParameters:
    """Build ``aruco.CharucoParameters`` from normalised GUI options."""
    params = aruco.CharucoParameters()  # Start from OpenCV's default Charuco parameters.
    params.tryRefineMarkers = True  # Preserve the existing default behaviour for both targets.
    for key, value in _group_options(detection_options, "CharucoParameters").items():  # Apply user overrides.
        if not hasattr(params, key):  # Ignore unknown keys for forward/backward compatibility.
            continue  # Skip unsupported parameters on this OpenCV build.
        if value is None:  # Optional GUI fields can intentionally stay unset.
            continue  # Leave the OpenCV default intact.
        if key in {"cameraMatrix", "distCoeffs"}:  # These fields cross the Python/C++ boundary.
            value = np.asarray(value, dtype=np.float64)  # Normalise them to float64 numpy arrays.
        setattr(params, key, value)  # Apply the validated option to OpenCV's parameter object.
    return params  # Return the fully configured Charuco parameters.


def build_detector_parameters(detection_options: dict | None) -> aruco.DetectorParameters:
    """Build ``aruco.DetectorParameters`` from normalised GUI options."""
    params = aruco.DetectorParameters()  # Start from OpenCV's default detector parameters.
    for key, value in _group_options(detection_options, "DetectorParameters").items():  # Apply user overrides.
        if not hasattr(params, key):  # Ignore keys missing on this OpenCV build.
            continue  # Preserve compatibility with older/newer bindings.
        if value is None:  # Skip intentionally empty values.
            continue  # Keep the OpenCV default for this field.
        if key == "cornerRefinementMethod":  # Enum values can arrive as strings from the GUI.
            value = _coerce_corner_refinement_method(value)  # Resolve names to OpenCV constants.
        setattr(params, key, value)  # Apply the validated option.
    return params  # Return the fully configured detector parameters.


def build_refine_parameters(detection_options: dict | None) -> aruco.RefineParameters:
    """Build ``aruco.RefineParameters`` from normalised GUI options."""
    params = aruco.RefineParameters()  # Start from OpenCV's default board-refinement parameters.
    for key, value in _group_options(detection_options, "RefineParameters").items():  # Apply user overrides.
        if not hasattr(params, key):  # Ignore unsupported keys for compatibility.
            continue  # Leave unsupported fields untouched.
        if value is None:  # Skip intentionally omitted values.
            continue  # Preserve the OpenCV default.
        setattr(params, key, value)  # Apply the validated option.
    return params  # Return the configured refinement parameters.


def build_charuco_detector_components(
    detection_options: dict | None,
) -> tuple[aruco.CharucoParameters, aruco.DetectorParameters, aruco.RefineParameters]:
    """Build the three OpenCV parameter objects needed for ChArUco detection."""
    charuco_parameters = build_charuco_parameters(detection_options)  # Build interpolation parameters first.
    detector_parameters = build_detector_parameters(detection_options)  # Build marker-detection parameters next.
    refine_parameters = build_refine_parameters(detection_options)  # Build board-guided refinement parameters last.
    return charuco_parameters, detector_parameters, refine_parameters  # Return the shared parameter bundle.


def construct_charuco_detector(
    board,
    charuco_parameters: aruco.CharucoParameters,
    detector_parameters: aruco.DetectorParameters,
    refine_parameters: aruco.RefineParameters,
):
    """Construct ``aruco.CharucoDetector`` with a backwards-compatible fallback."""
    try:  # Prefer the full modern constructor when the OpenCV build supports it.
        return aruco.CharucoDetector(  # Create the detector with all three parameter objects.
            board,  # Pass the target board geometry.
            charuco_parameters,  # Pass ChArUco interpolation parameters.
            detector_parameters,  # Pass marker detector parameters.
            refine_parameters,  # Pass board-refinement parameters.
        )
    except TypeError:  # Older OpenCV builds only support ``(board, charucoParams)``.
        _LOG.warning(  # Log the compatibility downgrade once per attempted construction.
            "OpenCV CharucoDetector constructor does not support DetectorParameters/RefineParameters; "
            "falling back to CharucoParameters-only detector construction."
        )
        return aruco.CharucoDetector(board, charuco_parameters)  # Fall back to the legacy constructor.
