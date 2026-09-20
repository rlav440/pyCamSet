"""
Whether one frame's evidence says a ChArUco board was printed to the other
legacy pattern than the target is configured with.

Detection reads only the configured pattern: the corners a probe here finds
under the opposite one are evidence for a warning, never a result.
"""

from __future__ import annotations

import math

import cv2

from pyCamSet.calibration_targets.markers.aruco_opencv import ARUCO_OPENCV_DETECTOR

# A warning needs evidence at two stages: enough of the board's markers
# found with still no corners, then enough corners from the opposite-pattern
# probe. The second threshold scales with the board, since sizes differ by an
# order of magnitude -- two thirds of a 5x4's 12 corners is reachable, two
# thirds of a 20x20's 361 is not, so past LARGE_BOARD_TOTAL_CORNERS the
# markers found this frame cap it instead.
LEGACY_WARNING_MARKER_FRACTION = 0.25
LEGACY_WARNING_MIN_MARKERS = 4
LEGACY_WARNING_PROBE_FRACTION = 2 / 3
LEGACY_WARNING_PROBE_FLOOR = 6
LEGACY_WARNING_LARGE_BOARD_TOTAL_CORNERS = 108
LEGACY_WARNING_PROBE_MARKER_FRACTION = 1.0


def min_probe_corners_for(board, n_markers_found: int | None = None) -> int:
    """How many opposite-pattern probe corners count as a mismatch.

    Scaled to *board*'s own total corner count and capped at it: a probe
    cannot interpolate more corners than the board has.

    :param board: the ``cv2.aruco.CharucoBoard`` being read
    :param n_markers_found: markers this frame's configured pass found,
        which bounds the threshold on a large board; None scales by total
        alone
    :return: the minimum probe corner count that counts as a mismatch
    """
    n_x, n_y = board.getChessboardSize()
    total_corners = int((n_x - 1) * (n_y - 1))
    scaled = max(
        LEGACY_WARNING_PROBE_FLOOR,
        math.ceil(LEGACY_WARNING_PROBE_FRACTION * total_corners))
    threshold = min(total_corners, scaled)
    if (n_markers_found is not None
            and total_corners > LEGACY_WARNING_LARGE_BOARD_TOTAL_CORNERS):
        marker_scaled = max(
            LEGACY_WARNING_PROBE_FLOOR,
            math.ceil(LEGACY_WARNING_PROBE_MARKER_FRACTION * n_markers_found))
        threshold = min(threshold, marker_scaled)
    return threshold


def chessboard_row_count_is_even(board) -> bool:
    """Whether *board*'s legacy flag can change how it is read at all.

    The legacy origin only shifts the markers against the chessboard when
    the row count is even; for an odd one every geometry accessor is
    bit-identical either way, so such a board is never probed.

    :param board: the ``cv2.aruco.CharucoBoard`` being read
    :return: True when toggling legacy can matter for this board
    """
    n_y = int(board.getChessboardSize()[1])
    return n_y % 2 == 0


def marker_count_meets_warning_floor(board, n_markers_found: int) -> bool:
    """Whether this many markers, with zero corners, is worth probing."""
    total_markers = len(board.getIds())
    threshold = max(
        LEGACY_WARNING_MIN_MARKERS,
        int(total_markers * LEGACY_WARNING_MARKER_FRACTION))
    return n_markers_found >= threshold


def build_opposite_pattern_board(board) -> cv2.aruco.CharucoBoard:
    """*board*'s geometry under the opposite legacy flag, to probe with.

    A throwaway: a target's own board is never toggled during detection.
    """
    probe_board = cv2.aruco.CharucoBoard(
        board.getChessboardSize(), board.getSquareLength(),
        board.getMarkerLength(), board.getDictionary())
    probe_board.setLegacyPattern(not board.getLegacyPattern())
    return probe_board


def probe_opposite_pattern_corners(board, detection_options, image, marker_corners, marker_ids):
    """Interpolate already-detected markers under the opposite pattern.

    Evidence only. Neither *board* nor its cached detector is touched: the
    probe builds its own throwaway pair.

    :param board: the target's own configured board, read not mutated
    :param detection_options: resolved ArUco1 settings, so the probe
        detector matches however the real one was built
    :param image: the image the markers were found in
    :param marker_corners: as ``detectBoard`` takes and returns them
    :param marker_ids: as ``detectBoard`` takes and returns them
    :return: the (N,1,2) corners the probe interpolated, or None
    """
    probe_board = build_opposite_pattern_board(board)
    probe_detector = ARUCO_OPENCV_DETECTOR.build_detector(probe_board, detection_options)
    p_corners, _p_ids, _mloc, _mid = probe_detector.detectBoard(
        image, markerCorners=marker_corners, markerIds=marker_ids)
    return p_corners


def should_warn_legacy_mismatch(
    board, detection_options, image, n_markers_found, marker_corners, marker_ids,
) -> bool:
    """Whether a legacy-mismatch warning should fire for this frame.

    Call only once the configured pattern has been tried and found markers
    but zero corners. All three must hold: the flag can matter for this
    board at all, enough markers were found that no corners is suspicious,
    and the opposite-pattern probe interpolates enough corners for this
    board's size.

    :param board: the target's own configured board, read not mutated
    :param n_markers_found: markers the configured pass found while getting
        zero corners
    :return: whether the caller should fire its warning
    """
    if not chessboard_row_count_is_even(board):
        return False
    if not marker_count_meets_warning_floor(board, n_markers_found):
        return False
    probe_corners = probe_opposite_pattern_corners(
        board, detection_options, image, marker_corners, marker_ids)
    return (probe_corners is not None
            and len(probe_corners) >= min_probe_corners_for(board, n_markers_found))
