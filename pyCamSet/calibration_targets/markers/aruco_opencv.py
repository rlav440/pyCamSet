"""
OpenCV's ChArUco detector, and everything it can be told.

The parameters themselves are stored beside this file rather than written
here: twenty-one rows saying what each is called, what it defaults to, the
bounds a study may search between, and the prose a person reads while typing
one in, beside five named presets narrowing those bounds to one kind of
image.  This is what turns them into the three parameter objects OpenCV
actually wants, and into the detector that holds them.

Shared by ChArUco and Ccube, which read the same markers with the same
library and differ only in how many boards they point it at.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from cv2 import aruco

from pyCamSet.calibration_targets.core.parameters import (
    Parameter,
    DetectorParameterisation,
    Profile,
    parameters_from_json,
    profiles_from_json,
)

_LOG = logging.getLogger(__name__)

_PARAMETERS = parameters_from_json(
    Path(__file__).parent / "charuco_parameters.json")
_PROFILES = profiles_from_json(
    Path(__file__).parent / "charuco_profiles.json")


class ArucoOpenCVDetector(DetectorParameterisation):
    """OpenCV's ``aruco.CharucoDetector``, and the settings it takes."""

    name = "aruco1"

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        return _PARAMETERS

    def profiles(self) -> dict[str, Profile]:
        return dict(_PROFILES)

    def validate(self, values: dict[str, Any]) -> list[str]:
        """The one rule OpenCV holds between two of its parameters."""
        low = values.get("adaptiveThreshWinSizeMin")
        high = values.get("adaptiveThreshWinSizeMax")
        if low is None or high is None or high >= low:
            return []
        return ["adaptiveThreshWinSizeMax must be >= adaptiveThreshWinSizeMin."]

    # -- applying the settings -------------------------------------------

    def grouped(self, values: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """
        *values* arranged as the sub-objects OpenCV keeps them in.

        A key this build of OpenCV does not know is dropped rather than
        raised on, which is how :meth:`_apply` treats one too.
        """
        groups: dict[str, dict[str, Any]] = {}
        for parameter in self.parameters:
            if parameter.key in values:
                groups.setdefault(parameter.group, {})[
                    parameter.key] = values[parameter.key]
        return groups

    def _apply(self, params, group: dict[str, Any]):
        """Set what *group* says on one of OpenCV's parameter objects."""
        for key, value in group.items():
            if value is None or not hasattr(params, key):
                # An unset optional field keeps OpenCV's own default, and a
                # field this build has never heard of is not ours to set.
                continue
            if key in ("cameraMatrix", "distCoeffs"):
                value = np.asarray(value, dtype=np.float64)
            setattr(params, key, value)
        return params

    def build_parameters(
        self, values: dict[str, Any]
    ) -> tuple[aruco.CharucoParameters, aruco.DetectorParameters, aruco.RefineParameters]:
        """The three parameter objects ChArUco detection needs."""
        groups = self.grouped(values)
        charuco = aruco.CharucoParameters()
        charuco.tryRefineMarkers = True
        return (
            self._apply(charuco, groups.get("CharucoParameters", {})),
            self._apply(aruco.DetectorParameters(),
                        groups.get("DetectorParameters", {})),
            self._apply(aruco.RefineParameters(),
                        groups.get("RefineParameters", {})),
        )

    def build_detector(self, board, values: dict[str, Any]):
        """
        The detector *board* is read with, holding what *values* say.

        :param board: the ``aruco.CharucoBoard`` to detect
        :param values: the settings, as :meth:`resolve` returns them
        """
        charuco, detector, refine = self.build_parameters(values)
        try:
            return aruco.CharucoDetector(board, charuco, detector, refine)
        except TypeError:
            _LOG.warning(
                "OpenCV CharucoDetector constructor does not support "
                "DetectorParameters/RefineParameters; falling back to "
                "CharucoParameters-only detector construction.")
            return aruco.CharucoDetector(board, charuco)


#: Shared rather than built per target: the settings live on the target, and
#: this holds only the description of them.
ARUCO_OPENCV_DETECTOR = ArucoOpenCVDetector()


def marker_bit_grid(dictionary: cv2.aruco.Dictionary, marker_id: int) -> np.ndarray:
    """A marker as its printed cells, one-cell border included, 1 for black.

    Rendered and thresholded rather than unpacked from the dictionary's
    packed bytes, whose layout differs across OpenCV builds.
    """
    marker_size = int(dictionary.markerSize)
    n_cells = marker_size + 2
    cell_px = 24
    side = n_cells * cell_px

    marker_img = np.zeros((side, side), dtype=np.uint8)
    cv2.aruco.generateImageMarker(dictionary, int(marker_id), side, marker_img, 1)

    grid = np.zeros((n_cells, n_cells), dtype=np.uint8)
    for r in range(n_cells):
        for c in range(n_cells):
            block = marker_img[r * cell_px:(r + 1) * cell_px, c * cell_px:(c + 1) * cell_px]
            grid[r, c] = 1 if float(block.mean()) < 127.5 else 0
    return grid
