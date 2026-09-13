"""Define named ChArUco detection-bound profiles for the Optimisation tab."""
from __future__ import annotations

from typing import Any


CHARUCO_DETECTION_PROFILE_NAMES: tuple[str, ...] = (
    "Balanced",
    "Low-Contrast / Dim",
    "Small / Distant Board",
    "Close / Large Board",
    "Aggressive Recovery",
    "Custom",
)
"""Display order for the ChArUco detection-profile selector."""


CHARUCO_DETECTION_PROFILES: dict[str, dict[str, Any]] = {
    "Balanced": {
        "name": "Balanced",
        "description": "General-purpose moderate search for first-pass optimisation.",
        "lower_bounds": {
            "adaptiveThreshWinSizeMin": 3,
            "adaptiveThreshWinSizeMax": 23,
            "adaptiveThreshWinSizeStep": 3,
            "adaptiveThreshConstant": 3.0,
            "minMarkerPerimeterRate": 0.01,
            "maxMarkerPerimeterRate": 2.0,
            "polygonalApproxAccuracyRate": 0.015,
            "minCornerDistanceRate": 0.02,
            "minDistanceToBorder": 1,
            "minMarkerDistanceRate": 0.01,
            "cornerRefinementWinSize": 3,
            "cornerRefinementMaxIterations": 20,
            "cornerRefinementMinAccuracy": 0.01,
            "errorCorrectionRate": 0.4,
            "minMarkers": 1,
        },
        "upper_bounds": {
            "adaptiveThreshWinSizeMin": 15,
            "adaptiveThreshWinSizeMax": 61,
            "adaptiveThreshWinSizeStep": 15,
            "adaptiveThreshConstant": 12.0,
            "minMarkerPerimeterRate": 0.08,
            "maxMarkerPerimeterRate": 5.0,
            "polygonalApproxAccuracyRate": 0.05,
            "minCornerDistanceRate": 0.08,
            "minDistanceToBorder": 8,
            "minMarkerDistanceRate": 0.08,
            "cornerRefinementWinSize": 9,
            "cornerRefinementMaxIterations": 60,
            "cornerRefinementMinAccuracy": 0.2,
            "errorCorrectionRate": 0.8,
            "minMarkers": 3,
        },
        "recommended_keys": [
            "adaptiveThreshWinSizeMax",
            "adaptiveThreshConstant",
            "minMarkerPerimeterRate",
            "polygonalApproxAccuracyRate",
            "cornerRefinementWinSize",
            "errorCorrectionRate",
            "minMarkers",
        ],
    },
    "Low-Contrast / Dim": {
        "name": "Low-Contrast / Dim",
        "description": "Bias the search towards weak illumination and low-contrast marker recovery.",
        "lower_bounds": {
            "adaptiveThreshWinSizeMin": 3,
            "adaptiveThreshWinSizeMax": 51,
            "adaptiveThreshWinSizeStep": 5,
            "adaptiveThreshConstant": 0.0,
            "minMarkerPerimeterRate": 0.005,
            "maxMarkerPerimeterRate": 2.0,
            "polygonalApproxAccuracyRate": 0.02,
            "minCornerDistanceRate": 0.0,
            "minDistanceToBorder": 0,
            "minMarkerDistanceRate": 0.0,
            "cornerRefinementWinSize": 5,
            "cornerRefinementMaxIterations": 30,
            "cornerRefinementMinAccuracy": 0.001,
            "errorCorrectionRate": 0.5,
            "minMarkers": 1,
        },
        "upper_bounds": {
            "adaptiveThreshWinSizeMin": 7,
            "adaptiveThreshWinSizeMax": 151,
            "adaptiveThreshWinSizeStep": 25,
            "adaptiveThreshConstant": 8.0,
            "minMarkerPerimeterRate": 0.05,
            "maxMarkerPerimeterRate": 6.0,
            "polygonalApproxAccuracyRate": 0.08,
            "minCornerDistanceRate": 0.06,
            "minDistanceToBorder": 5,
            "minMarkerDistanceRate": 0.06,
            "cornerRefinementWinSize": 15,
            "cornerRefinementMaxIterations": 120,
            "cornerRefinementMinAccuracy": 0.1,
            "errorCorrectionRate": 1.0,
            "minMarkers": 3,
        },
        "recommended_keys": [
            "adaptiveThreshWinSizeMax",
            "adaptiveThreshWinSizeStep",
            "adaptiveThreshConstant",
            "errorCorrectionRate",
            "cornerRefinementMaxIterations",
            "minMarkers",
        ],
    },
    "Small / Distant Board": {
        "name": "Small / Distant Board",
        "description": "Prioritise detection when markers occupy only a small image area.",
        "lower_bounds": {
            "adaptiveThreshWinSizeMin": 3,
            "adaptiveThreshWinSizeMax": 15,
            "adaptiveThreshWinSizeStep": 2,
            "adaptiveThreshConstant": 2.0,
            "minMarkerPerimeterRate": 0.001,
            "maxMarkerPerimeterRate": 0.5,
            "polygonalApproxAccuracyRate": 0.01,
            "minCornerDistanceRate": 0.0,
            "minDistanceToBorder": 0,
            "minMarkerDistanceRate": 0.0,
            "cornerRefinementWinSize": 2,
            "cornerRefinementMaxIterations": 20,
            "cornerRefinementMinAccuracy": 0.001,
            "errorCorrectionRate": 0.5,
            "minMarkers": 0,
        },
        "upper_bounds": {
            "adaptiveThreshWinSizeMin": 11,
            "adaptiveThreshWinSizeMax": 45,
            "adaptiveThreshWinSizeStep": 12,
            "adaptiveThreshConstant": 10.0,
            "minMarkerPerimeterRate": 0.03,
            "maxMarkerPerimeterRate": 3.0,
            "polygonalApproxAccuracyRate": 0.06,
            "minCornerDistanceRate": 0.04,
            "minDistanceToBorder": 4,
            "minMarkerDistanceRate": 0.05,
            "cornerRefinementWinSize": 8,
            "cornerRefinementMaxIterations": 100,
            "cornerRefinementMinAccuracy": 0.1,
            "errorCorrectionRate": 1.0,
            "minMarkers": 2,
        },
        "recommended_keys": [
            "minMarkerPerimeterRate",
            "maxMarkerPerimeterRate",
            "minDistanceToBorder",
            "adaptiveThreshWinSizeMin",
            "adaptiveThreshWinSizeMax",
            "errorCorrectionRate",
            "minMarkers",
        ],
    },
    "Close / Large Board": {
        "name": "Close / Large Board",
        "description": "Focus on larger in-frame boards with stronger marker support.",
        "lower_bounds": {
            "adaptiveThreshWinSizeMin": 3,
            "adaptiveThreshWinSizeMax": 23,
            "adaptiveThreshWinSizeStep": 3,
            "adaptiveThreshConstant": 5.0,
            "minMarkerPerimeterRate": 0.04,
            "maxMarkerPerimeterRate": 3.0,
            "polygonalApproxAccuracyRate": 0.01,
            "minCornerDistanceRate": 0.03,
            "minDistanceToBorder": 2,
            "minMarkerDistanceRate": 0.03,
            "cornerRefinementWinSize": 3,
            "cornerRefinementMaxIterations": 15,
            "cornerRefinementMinAccuracy": 0.01,
            "errorCorrectionRate": 0.3,
            "minMarkers": 2,
        },
        "upper_bounds": {
            "adaptiveThreshWinSizeMin": 13,
            "adaptiveThreshWinSizeMax": 71,
            "adaptiveThreshWinSizeStep": 15,
            "adaptiveThreshConstant": 15.0,
            "minMarkerPerimeterRate": 0.2,
            "maxMarkerPerimeterRate": 8.0,
            "polygonalApproxAccuracyRate": 0.04,
            "minCornerDistanceRate": 0.12,
            "minDistanceToBorder": 15,
            "minMarkerDistanceRate": 0.12,
            "cornerRefinementWinSize": 7,
            "cornerRefinementMaxIterations": 60,
            "cornerRefinementMinAccuracy": 0.15,
            "errorCorrectionRate": 0.8,
            "minMarkers": 4,
        },
        "recommended_keys": [
            "minMarkerPerimeterRate",
            "maxMarkerPerimeterRate",
            "minCornerDistanceRate",
            "minMarkerDistanceRate",
            "adaptiveThreshConstant",
            "minMarkers",
        ],
    },
    "Aggressive Recovery": {
        "name": "Aggressive Recovery",
        "description": (
            "Broad exploratory bounds for difficult datasets; higher risk of unstable detections."
        ),
        "lower_bounds": {
            "adaptiveThreshWinSizeMin": 3,
            "adaptiveThreshWinSizeMax": 23,
            "adaptiveThreshWinSizeStep": 1,
            "adaptiveThreshConstant": 0.0,
            "minMarkerPerimeterRate": 0.001,
            "maxMarkerPerimeterRate": 0.5,
            "polygonalApproxAccuracyRate": 0.005,
            "minCornerDistanceRate": 0.0,
            "minDistanceToBorder": 0,
            "minMarkerDistanceRate": 0.0,
            "cornerRefinementWinSize": 1,
            "cornerRefinementMaxIterations": 10,
            "cornerRefinementMinAccuracy": 0.0005,
            "errorCorrectionRate": 0.1,
            "minMarkers": 0,
        },
        "upper_bounds": {
            "adaptiveThreshWinSizeMin": 31,
            "adaptiveThreshWinSizeMax": 199,
            "adaptiveThreshWinSizeStep": 35,
            "adaptiveThreshConstant": 20.0,
            "minMarkerPerimeterRate": 0.15,
            "maxMarkerPerimeterRate": 8.0,
            "polygonalApproxAccuracyRate": 0.12,
            "minCornerDistanceRate": 0.2,
            "minDistanceToBorder": 20,
            "minMarkerDistanceRate": 0.2,
            "cornerRefinementWinSize": 20,
            "cornerRefinementMaxIterations": 200,
            "cornerRefinementMinAccuracy": 0.4,
            "errorCorrectionRate": 1.0,
            "minMarkers": 4,
        },
        "recommended_keys": [
            "adaptiveThreshWinSizeMin",
            "adaptiveThreshWinSizeMax",
            "adaptiveThreshConstant",
            "minMarkerPerimeterRate",
            "maxMarkerPerimeterRate",
            "polygonalApproxAccuracyRate",
            "errorCorrectionRate",
            "minMarkers",
        ],
    },
}
"""Profile payloads consumed by the Optimisation-tab ChArUco options section."""


def get_charuco_detection_profile(name: str) -> dict[str, Any]:
    """Return a profile payload by name."""
    if name not in CHARUCO_DETECTION_PROFILES:
        raise KeyError(f"Unknown ChArUco detection profile: {name!r}")
    return CHARUCO_DETECTION_PROFILES[name]


def make_profile_tooltip(name: str, key_to_label: dict[str, str]) -> str:
    """Render one profile's hover/help text.

    Parameters
    ----------
    name:
        Profile name selected in the GUI dropdown.
    key_to_label:
        Mapping from detector-parameter keys to user-facing row labels.

    Returns
    -------
    str
        Tooltip text containing a one-line profile purpose and the profile's
        recommended parameters to check for optimisation.
    """
    # Treat Custom as a special selector state with no auto recommendations.
    if name == "Custom":
        return (
            "Custom bounds edited manually.\n"
            "Recommended parameters to check for optimisation: keep this user-selected."
        )
    # Resolve the selected profile payload once for a consistent tooltip body.
    profile = get_charuco_detection_profile(name)
    # Convert profile keys into row labels so the UI text is easier to scan.
    recommended_labels = [
        key_to_label.get(key, key) for key in profile.get("recommended_keys", [])
    ]
    # Build the recommendation block in the same order as the profile data.
    recommended_block = "\n".join(f"- {label}" for label in recommended_labels) or "- (none)"
    # Return the final tooltip body shown in the combo and dropdown items.
    return (
        f"{profile['description']}\n\n"
        "Recommended parameters to check for optimisation:\n"
        f"{recommended_block}"
    )


__all__ = [
    "CHARUCO_DETECTION_PROFILE_NAMES",
    "CHARUCO_DETECTION_PROFILES",
    "get_charuco_detection_profile",
    "make_profile_tooltip",
]
