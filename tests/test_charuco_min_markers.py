"""
CharucoParameters.minMarkers stays inside what OpenCV accepts.

OpenCV's ``CharucoDetector::detectBoard`` asserts ``minMarkers`` is at most 2
(charuco_detector.cpp; seen on 4.11 and 5.0.0), so a value of 3 or 4 does not
filter harder -- it raises on every image. The parameter table allowed 0-4 and
four of the five tuning profiles searched up to 3 or 4.
"""
import cv2
import numpy as np
import pytest

from pyCamSet.calibration_targets.markers.aruco_opencv import ARUCO_OPENCV_DETECTOR

_OPENCV_MAX_MIN_MARKERS = 2


def _min_markers():
    return next(p for p in ARUCO_OPENCV_DETECTOR.parameters if p.key == "minMarkers")


def test_the_parameter_range_ends_where_opencv_does():
    parameter = _min_markers()
    assert parameter.minimum == 0
    assert parameter.maximum == _OPENCV_MAX_MIN_MARKERS
    assert parameter.minimum <= parameter.default <= parameter.maximum


@pytest.mark.parametrize("name", list(ARUCO_OPENCV_DETECTOR.profiles()))
def test_no_profile_searches_min_markers_outside_the_parameter_range(name):
    parameter = _min_markers()
    bounds = ARUCO_OPENCV_DETECTOR.profiles()[name].bounds_for("minMarkers")
    if bounds is None:
        return
    low, high = bounds
    assert parameter.minimum <= low <= high <= parameter.maximum, (name, bounds)


def _detect_with(min_markers: int):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((5, 5), 0.04, 0.03, dictionary)
    image = np.ascontiguousarray(board.generateImage((500, 500)), dtype=np.uint8)
    detector = ARUCO_OPENCV_DETECTOR.build_detector(board, {"minMarkers": min_markers})
    return detector.detectBoard(image)


def test_every_value_in_the_range_detects():
    parameter = _min_markers()
    for value in range(parameter.minimum, parameter.maximum + 1):
        corners, ids, _, _ = _detect_with(value)
        assert ids is not None and len(ids) > 0, value


def test_opencv_still_refuses_one_past_the_range():
    # If a future OpenCV lifts its assertion this fails, and the range can
    # then be widened deliberately rather than by accident.
    with pytest.raises(cv2.error):
        _detect_with(_OPENCV_MAX_MIN_MARKERS + 1)
