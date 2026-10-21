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
import re
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

#: The version of the OpenCV this process imported, and its major version.
OPENCV_VERSION = str(cv2.__version__)
OPENCV_MAJOR = int(OPENCV_VERSION.split(".")[0])

#: The registered targets whose corners come from ``CharucoDetector``, with
#: either marker backend.
CHARUCO_DETECTOR_TARGETS = frozenset({"ChArUco", "Ccube", "CIco"})

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


#: The one pixel frame every ChArUco-family corner and marker quad pyCamSet
#: returns is in, whichever OpenCV is installed: integer coordinates are
#: pixel centres (pixel ``(i, j)``'s centre is at ``(i, j)``). It is the frame
#: ``cv2.calibrateCamera`` and ``cv2.projectPoints`` assume.
PIXEL_CONVENTION = "pixel-centre"


def opencv_charuco_pixel_offset(version: str | None = None) -> float:
    """
    How far the installed OpenCV's ``CharucoDetector.detectBoard`` reports a
    ChArUco corner from the pixel-*centre* convention, on both axes, in
    pixels: 0.5 below OpenCV 4.14, 0.0 from it on.

    Measured on analytic renders whose truth is in the pixel-centre frame:
    OpenCV 4.11 reports corners +0.48 px off (+0.50 once its cornerSubPix is
    run to convergence; it starts half a pixel back and stops at its 0.1 px
    accuracy) and 4.13 +0.50 px; OpenCV 4.14 and 5.0 report them at the
    truth. Marker quads, from
    ``ArucoDetector`` or returned by ``detectBoard``, are pixel-centre under
    both. :func:`charuco_corners_to_pixel_centre` takes the offset off.

    :param version: an OpenCV version string to judge instead of the
        installed ``cv2.__version__``.
    """
    text = cv2.__version__ if version is None else version
    # "5.1.0-pre" and the like: the leading digits of the first two fields.
    major, minor = (int(re.match(r"\d+", part).group()) for part in str(text).split(".")[:2])
    return 0.0 if (major, minor) >= (4, 14) else 0.5


def charuco_corners_to_pixel_centre(corners, version: str | None = None):
    """
    ChArUco corners as ``CharucoDetector.detectBoard`` returned them, moved
    into :data:`PIXEL_CONVENTION`.

    Returned unchanged (the same object) where the installed OpenCV already
    reports pixel centres, and None stays None; otherwise a float32 array of
    the same shape, :func:`opencv_charuco_pixel_offset` less on both axes.
    """
    offset = opencv_charuco_pixel_offset(version)
    if corners is None or offset == 0.0:
        return corners
    return np.asarray(corners, dtype=np.float32) - np.float32(offset)


def detect_board(detector, image, **kwargs):
    """
    ``detector.detectBoard(image, **kwargs)``, with the ChArUco corners in
    :data:`PIXEL_CONVENTION` whatever OpenCV version is installed.

    Every ChArUco path goes through this -- ChArUco, Ccube and CIco with
    either marker backend -- so the one difference between OpenCV before 4.14
    and from it is taken out in one place (:func:`charuco_corners_to_pixel_centre`).  The
    marker corners it also returns are left as OpenCV gives them: they are
    pixel-centre under every version.

    :param detector: a ``cv2.aruco.CharucoDetector``.
    :param image: the image to detect in.
    :param kwargs: passed on, e.g. ``markerCorners`` and ``markerIds``.
    :return: ``(charuco_corners, charuco_ids, marker_corners, marker_ids)``,
        as ``detectBoard`` returns them.
    """
    corners, ids, marker_corners, marker_ids = detector.detectBoard(image, **kwargs)
    return charuco_corners_to_pixel_centre(corners), ids, marker_corners, marker_ids


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
