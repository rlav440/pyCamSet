"""
Shared legacy-pattern-mismatch detection for ChArUco/Ccube boards, used by
both the ArUco1 (OpenCV) and ArUco2 (aruco2 package) detection paths
(``charuco/target.py``, ``ccube/target.py``, ``markers/aruco2.py``).

Approved policy: detection reads ONLY the legacy pattern a target is
configured with -- no per-frame retry under the opposite one, and a target's
own board's legacy flag is never touched during detection. This module
decides only whether one frame's evidence is strong enough to WARN that the
physical board was probably printed the other way round; the corners a probe
finds under that other pattern are evidence only and must never be returned
as a detection.

Replaces the previous per-frame legacy retry (ArUco1 and ArUco2 paths) and
the ArUco2 adapter's ambiguity-between-both-patterns probe.
"""

from __future__ import annotations

import math

import cv2

from pyCamSet.calibration_targets.markers.aruco_opencv import ARUCO_OPENCV_DETECTOR

#: A frame can legitimately give no corners under the CORRECT pattern too --
#: ``CharucoParameters.minMarkers`` already requires two markers just to
#: try, and a small, genuinely marginal marker subset can fail cleanly with
#: no mismatch involved at all. So the mismatch probe below is only worth
#: running once a frame shows STRONG evidence: a decent fraction of the
#: board's own full marker set found, yet zero corners under the configured
#: pattern. Calibrated (tests/test_aruco2_backend.py) against odd x even
#: boards (7x6, 9x6): a quarter of the board's markers, floor 4, never
#: warns on a correctly configured board's random partial views, and a
#: misconfigured board still warns within the first few reasonable ones.
LEGACY_WARNING_MARKER_FRACTION = 0.25
LEGACY_WARNING_MIN_MARKERS = 4

#: How large a FRACTION of the board's own total chessboard-corner count the
#: OPPOSITE-pattern probe must itself interpolate from the SAME markers
#: before a mismatch is actually reported, and the absolute floor under that
#: (see :func:`min_probe_corners_for`).
#:
#: Round-1 review (P1) raised a flat ``LEGACY_WARNING_MIN_PROBE_CORNERS``
#: from 6 to 20: a correctly configured 7x6/9x6 board's partial view, when
#: embedded in a full-resolution background (a real camera frame's
#: board-in-a-scene, not a shrunk-to-crop array -- see
#: ``tests/test_aruco2_backend.py::test_legacy_correctly_configured_board_never_warns_with_background``),
#: can occasionally still clear a too-low floor: an empirical sweep of
#: 25200 such frames (7x6 and 9x6, 7 background shades, 30 seeds, 60 crops
#: each) found a hard ceiling of 16 probe corners (from 14/18-21 markers
#: found) for a frame that is NOT a mismatch, so 20 (with margin) was
#: chosen.
#:
#: Round-2 review (overseer addition (a)) found that flat 20 could not
#: scale down: a small odd-width x even-height board -- exactly the class a
#: legacy mismatch can mislabel -- can have FEWER than 20 chessboard
#: corners in total (5x4 has 12; 5x6 has 20; 7x4 has 18), so a misconfigured
#: one of those could never clear the flat floor even on an EXACT full view,
#: where the probe interpolates the board's WHOLE corner set. Replaced with
#: a threshold scaled to the board's own total corner count
#: (:func:`min_probe_corners_for`), re-run against the same background-
#: embedded sweep, widened to 9000 frames per size (5 background shades, 30
#: seeds, 60 crops) across 5x4, 7x4, 5x6, 7x6 and 9x6: only 7x6 produced any
#: non-zero noise at all, reproducing the same 16-corner ceiling the
#: round-1 sweep found (bg=255, seed=4) -- 5x4, 7x4, 5x6 and 9x6 all scored
#: zero probe corners across every qualifying frame. ``LEGACY_WARNING_PROBE_FRACTION
#: = 2/3`` reproduces the already-validated flat value exactly for 7x6
#: (ceil(2/3 * 30) == 20) while scaling proportionally for every other
#: size -- e.g. 12 for 5x4, 14 for 5x6, 12 for 7x4 -- always comfortably
#: below each board's own total corner count, so an exact full view of a
#: misconfigured board of any of these sizes (its probe always interpolates
#: exactly that total) still warns on the first frame.  Covered by
#: parametrised tests in ``tests/test_aruco2_backend.py`` across 5x4, 5x6,
#: 7x4, 7x6, 9x6, 11x8, 13x10 and 10x8.
LEGACY_WARNING_PROBE_FRACTION = 2 / 3

#: The absolute floor under the fraction above, for boards small enough
#: that a fraction alone would ask for very few corners. Capped by
#: :func:`min_probe_corners_for` at the board's own total corner count, so
#: it can never make the threshold unreachable on a tiny board.
LEGACY_WARNING_PROBE_FLOOR = 6

#: Round-3 review (P1): :data:`LEGACY_WARNING_PROBE_FRACTION` scales the
#: threshold to the board's own TOTAL corner count, which is right for a
#: board small enough that an ordinary photo can show most of it -- but a
#: large board (this project's own 20x20 reference board has 361) is never
#: seen anywhere near fully in one frame, so its total-scaled threshold
#: (241) sat far above what any realistic partial/oblique view could ever
#: clear: reproduced directly against the checked-in
#: ``tests/test_data/calibration_charuco`` corpus (20x20, legacy=False),
#: read with a MISCONFIGURED ``legacy=True`` target -- every one of its 90
#: images gave zero corners under the configured pattern with 46-78 markers
#: found and an opposite-pattern probe of only 81-130 corners, so the
#: mismatch warning this module exists to fire never did, on any frame.
#:
#: Below this many TOTAL corners (13x10's 108, the largest size the
#: existing :data:`LEGACY_WARNING_PROBE_FRACTION` sweep already validates
#: -- see ``tests/test_aruco2_backend.py``'s ``_LEGACY_POLICY_SIZES``),
#: :func:`min_probe_corners_for` is unchanged: that range is small enough
#: to plausibly appear whole in a frame, and its false-positive floor is
#: already calibrated close to the fraction-of-total threshold (the known
#: 7x6 false positive: 14 markers found, 16 probe corners -- a ratio of
#: ~1.14, too close to a genuine mismatch's own ratio, below, to also
#: anchor safely to markers found). ABOVE it, an ADDITIONAL ceiling
#: anchored to how many markers were actually found in THIS frame is also
#: applied (never a looser one -- see :func:`min_probe_corners_for`) --
#: markers actually seen, not the board's total size, bound how many
#: corners a real partial view's probe could ever interpolate. Calibrated
#: against a background-embedded partial-view sweep at 16x16 and 20x20 (44
#: and 104 qualifying frames respectively: markers met the warning floor
#: with zero corners under the CORRECTLY configured pattern) -- every one
#: of those 148 frames' opposite-pattern probe interpolated exactly ZERO
#: corners, while the genuinely misconfigured 20x20 corpus above always
#: interpolated at least 1.55x the markers found. 1.0x sits with wide
#: margin on both sides.
LEGACY_WARNING_LARGE_BOARD_TOTAL_CORNERS = 108
LEGACY_WARNING_PROBE_MARKER_FRACTION = 1.0


def min_probe_corners_for(board, n_markers_found: int | None = None) -> int:
    """How many opposite-pattern probe corners *board* needs before a
    mismatch is reported -- see :data:`LEGACY_WARNING_PROBE_FRACTION`'s
    docstring for how this was calibrated and why a flat threshold does not
    work across board sizes.

    Scaled to *board*'s own total interior chessboard-corner count
    (``(n_x - 1) * (n_y - 1)``), not an absolute constant, and capped at
    that same total: a probe can never interpolate more corners than the
    board has, so a threshold above the total would make a mismatch
    unreportable on ANY view, including the guaranteed exact full one.

    :param board: the ``cv2.aruco.CharucoBoard`` being read.
    :param n_markers_found: how many markers THIS frame's configured-pattern
        pass found, when known -- see
        :data:`LEGACY_WARNING_PROBE_MARKER_FRACTION`'s docstring. ``None``
        (the default) keeps the total-only scaling, e.g. for a caller that
        has no particular frame in mind.
    :return: the minimum probe corner count that counts as a mismatch.
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
    """Whether *board*'s legacy-pattern flag can change anything about how
    it is read at all.

    OpenCV's ``CharucoBoard`` legacy-pattern origin only shifts the marker
    layout relative to the chessboard when the board has an EVEN chessboard
    row count (``getChessboardSize()[1]``): ``getObjPoints()``,
    ``getChessboardCorners()``, ``getIds()`` and ``generateImage()`` are
    bit-identical between ``legacy=True`` and ``legacy=False`` for every ODD
    row count (verified for 5, 7, 9), and differ for every even one (4, 6,
    8). A board with an odd row count can therefore never actually show a
    legacy mismatch -- the probe below is skipped outright for one, rather
    than spending a detectBoard call to confirm a pattern that provably
    reads identically either way.

    :param board: the ``cv2.aruco.CharucoBoard`` being read.
    :return: True when toggling legacy can matter for this board's geometry.
    """
    n_y = int(board.getChessboardSize()[1])
    return n_y % 2 == 0


def marker_count_meets_warning_floor(board, n_markers_found: int) -> bool:
    """Whether *n_markers_found* (under the configured pattern, with zero
    corners) is enough evidence to even consider probing the opposite
    pattern -- see the module's threshold constants."""
    total_markers = len(board.getIds())
    threshold = max(
        LEGACY_WARNING_MIN_MARKERS,
        int(total_markers * LEGACY_WARNING_MARKER_FRACTION))
    return n_markers_found >= threshold


def build_opposite_pattern_board(board) -> cv2.aruco.CharucoBoard:
    """A THROWAWAY ``cv2.aruco.CharucoBoard``, same geometry as *board* but
    with the OPPOSITE legacy-pattern flag -- for probing only. Never the
    target's own board: a target's configured board must never have its
    legacy flag changed during detection.
    """
    probe_board = cv2.aruco.CharucoBoard(
        board.getChessboardSize(), board.getSquareLength(),
        board.getMarkerLength(), board.getDictionary())
    probe_board.setLegacyPattern(not board.getLegacyPattern())
    return probe_board


def probe_opposite_pattern_corners(board, detection_options, image, marker_corners, marker_ids):
    """Interpolate *marker_corners*/*marker_ids* (already-detected markers,
    in the shapes ``CharucoDetector.detectBoard`` both takes and echoes
    back) against a THROWAWAY board and detector built for the OPPOSITE
    legacy pattern to *board*'s own. Read-only evidence-gathering: never
    mutates *board*, never reuses *board*'s own (possibly cached) detector,
    and the corners returned here are evidence only -- never a caller's
    real detection result (see the module docstring).

    :param board: the target's own, configured board -- read but not
        mutated.
    :param detection_options: the resolved ArUco1 detector settings
        (:meth:`ArucoOpenCVDetector.resolve`) to build the probe detector
        with, for parity with however the real detector was built.
    :param image: the image the markers were found in.
    :param marker_corners: as ``detectBoard`` takes/returns them.
    :param marker_ids: as ``detectBoard`` takes/returns them.
    :return: the (N,1,2) corners array the probe interpolated, or None.
    """
    probe_board = build_opposite_pattern_board(board)
    probe_detector = ARUCO_OPENCV_DETECTOR.build_detector(probe_board, detection_options)
    p_corners, _p_ids, _mloc, _mid = probe_detector.detectBoard(
        image, markerCorners=marker_corners, markerIds=marker_ids)
    return p_corners


def should_warn_legacy_mismatch(
    board, detection_options, image, n_markers_found, marker_corners, marker_ids,
) -> bool:
    """Whether a legacy-pattern-mismatch warning should fire for this frame.

    Call this only once the CONFIGURED pattern has already been tried and
    found markers but zero corners. Strong evidence, by design, so a
    CORRECTLY configured board essentially never triggers it (see the
    module's threshold constants and tests/test_aruco2_backend.py's
    calibration tests):

    1. toggling legacy can change anything for this board's geometry at all
       (:func:`chessboard_row_count_is_even`);
    2. the CONFIGURED pattern found enough markers
       (:func:`marker_count_meets_warning_floor`) that "no corners at all"
       is suspicious rather than just a marginal/partial view;
    3. a throwaway board/detector built for the OPPOSITE pattern, given the
       SAME already-detected markers (never a fresh marker-detection pass,
       and never the target's own board), interpolates a substantial
       number of corners -- substantial relative to THIS board's own size
       (:func:`min_probe_corners_for`), not a flat constant: see that
       function's docstring for why one board size's calibration cannot be
       reused unscaled for another.

    :param board: the target's own, configured ``CharucoBoard`` -- read but
        never mutated here.
    :param n_markers_found: how many markers the CONFIGURED pattern's pass
        found (while still getting zero corners).
    :return: whether the caller should fire its legacy-mismatch warning.
    """
    if not chessboard_row_count_is_even(board):
        return False
    if not marker_count_meets_warning_floor(board, n_markers_found):
        return False
    probe_corners = probe_opposite_pattern_corners(
        board, detection_options, image, marker_corners, marker_ids)
    return (probe_corners is not None
            and len(probe_corners) >= min_probe_corners_for(board, n_markers_found))
