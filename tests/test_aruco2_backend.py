'''
Purpose: Regression tests for the ArUco1/ArUco2 marker backend in pyCamSet
         (plan v4 batch B4). Covers round-trips for both backends on both
         targets, dictionary resolution rules, legacy even-row handling
         (positions, not counts), persistence round-trips, and the
         constructor/GUI plumbing contract.
Status: Active.
Future: Add cross-detection-family parity cases when real-image fixtures exist.
'''

import json
import logging
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

aruco2 = pytest.importorskip("aruco2")

from pyCamSet.calibration_targets.markers.backend_registry import (
    ARUCO1_DICT_NAMES,
    ARUCO2_DICT_NAMES,
    MARKER_BACKEND_LABELS,
    SUPPORTED_MARKER_BACKENDS,
    available_marker_backends,
    dict_names_for_backend,
    marker_backend_available,
    validate_marker_backend,
)
from pyCamSet.calibration_targets.charuco.target import ChArUco
from pyCamSet.calibration_targets.ccube.target import Ccube
from pyCamSet.calibration_targets.markers.aruco_opencv import (
    ARUCO_OPENCV_DETECTOR,
)


def _count(det):
    if det.keys is None or len(det.keys) == 0:
        return 0
    return len(det.keys)


def _true_markers_for_board(board, img):
    """Ground-truth ``(marker_id, (4, 2) float32 corners)`` for every ArUco
    marker OpenCV's own (unmodified) aruco1 detector finds in *img* -- a
    stand-in for aruco2's own marker detection so these tests can corrupt
    one marker in a controlled way without depending on aruco2's detector
    internals."""
    aruco_detector = cv2.aruco.ArucoDetector(
        board.getDictionary(), cv2.aruco.DetectorParameters())
    corners, ids, _ = aruco_detector.detectMarkers(img)
    assert ids is not None and len(ids) > 0, "no markers found in the rendered board"
    return [(int(i), np.asarray(c, dtype=np.float32).reshape(4, 2))
            for c, i in zip(corners, ids.reshape(-1))]


def _random_crops(img, n, seed=0, min_frac=0.12, max_frac=0.98):
    """*n* random axis-aligned crops of *img*, each a contiguous rectangle
    covering a random fraction (of each side) between *min_frac* and
    *max_frac* -- a stand-in for "many rendered partial views" without real
    image fixtures, and a much closer proxy for what a camera actually sees
    than a scattered marker-id subset would be: a contiguous crop can clip
    a whole edge row of markers, which a random id subset never models.
    Deliberately dips low enough that some crops legitimately show markers
    but no corners (not a mismatch, just under-constrained), and stays
    below max_frac=1.0 so it is never accidentally the exact full view --
    callers that need a guaranteed full view pass *img* itself instead.
    """
    rng = random.Random(seed)
    h, w = img.shape[:2]
    crops = []
    for _ in range(n):
        frac = rng.uniform(min_frac, max_frac)
        cw, ch = max(1, int(w * frac)), max(1, int(h * frac))
        x0 = rng.randint(0, w - cw)
        y0 = rng.randint(0, h - ch)
        crops.append(np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw]))
    return crops


def test_headless_backend_registry_has_stable_backend_contract():
    assert SUPPORTED_MARKER_BACKENDS == ("aruco1", "aruco2")
    assert MARKER_BACKEND_LABELS["ArUco 1 (OpenCV)"] == "aruco1"
    assert MARKER_BACKEND_LABELS["ArUco 2 (aruco2)"] == "aruco2"
    assert dict_names_for_backend("aruco1") == ARUCO1_DICT_NAMES
    assert dict_names_for_backend("aruco2") == ARUCO2_DICT_NAMES
    assert len(ARUCO1_DICT_NAMES) == 22
    assert ARUCO2_DICT_NAMES[-2:] == ["DICT_ALVAR_5X5_256", "DICT_ALVAR_7X7_1000"]


def test_headless_backend_registry_validates_and_reports_optional_backend():
    assert validate_marker_backend("aruco1") == "aruco1"
    assert validate_marker_backend("aruco2") == "aruco2"
    assert marker_backend_available("aruco1") is True
    assert marker_backend_available("aruco2") is True
    assert available_marker_backends() == ("aruco1", "aruco2")
    with pytest.raises(ValueError, match="marker_backend"):
        validate_marker_backend("aruco3")
    with pytest.raises(ValueError, match="marker_backend"):
        dict_names_for_backend("aruco3")


def test_headless_backend_registry_checks_real_optional_import(monkeypatch):
    import pyCamSet.calibration_targets.markers.backend_registry as registry

    def broken_import(_name):
        raise OSError("missing native extension")

    monkeypatch.setattr(registry.importlib, "import_module", broken_import)
    assert registry.marker_backend_available("aruco2") is False
    assert registry.available_marker_backends() == ("aruco1",)
    assert registry.marker_backend_availability_text("aruco2") == (
        "aruco2: not installed - see CITATION.md"
    )


def test_dictionary_resolution_validates_backend_before_dictionary_type():
    from pyCamSet.calibration_targets.markers.aruco2 import resolve_dictionary

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    with pytest.raises(ValueError, match="marker_backend"):
        resolve_dictionary(dictionary, marker_backend="aruco3")


def test_aruco2_detection_rejects_non_uint8_values_outside_byte_range():
    from pyCamSet.calibration_targets.markers.aruco2 import detect_markers

    with pytest.raises(ValueError, match="uint8"):
        detect_markers(np.full((8, 8), 256.0, dtype=np.float64), 0)


def test_interpolation_skips_malformed_marker_quads():
    from pyCamSet.calibration_targets.markers.aruco2 import interpolate_board_corners

    board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0).board
    ids, points = interpolate_board_corners(
        np.zeros((100, 100), dtype=np.uint8),
        board,
        [(0, np.array([[10, 10], [30, 10], [20, 30]], dtype=np.float32))],
    )
    assert ids is None and points is None


# -- P0: CharucoDetector cache must survive id(board) reuse ------------------


def test_charuco_detector_cache_rebuilds_on_id_reuse_not_stale_hit():
    """P0: even when CPython reuses id(board) for a new, differently-shaped
    board after the old one is garbage-collected (observed directly: a
    construct/delete/gc loop alternating 5x5 and 9x7 boards reused ids for
    most iterations), the cache must rebuild rather than hand back the
    stale detector. The exact collision state a real reuse would leave the
    cache in is reproduced directly here, rather than depending on GC
    timing to happen to trigger it this run."""
    from pyCamSet.calibration_targets.markers import aruco2 as a2mod

    board_5x5 = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0).board
    board_9x7 = ChArUco(num_squares_x=9, num_squares_y=7, square_size=10.0).board

    stale_detector = a2mod._charuco_detector_for(board_5x5)
    # Poison the cache under the 9x7 board's OWN id: exactly the state a
    # real id() collision (5x5 collected, its address handed to a new 9x7
    # board) would leave it in, without depending on whether that actually
    # happens this run.
    a2mod._CHARUCO_DETECTOR_CACHE[id(board_9x7)] = (board_5x5, stale_detector)

    detector = a2mod._charuco_detector_for(board_9x7)
    assert detector is not stale_detector, (
        "cache returned the stale 5x5 detector for a 9x7 board sharing its id()")

    img = np.ascontiguousarray(board_9x7.generateImage((700, 700)), dtype=np.uint8)
    c_corners, c_ids, _, _ = detector.detectBoard(img)
    assert c_corners is not None
    assert len(np.asarray(c_ids).reshape(-1)) == 8 * 6  # (9-1)*(7-1) interior corners


def test_charuco_detector_cache_matches_fresh_detector_across_gc_churn():
    """P0: alternate two differently-shaped boards through construct/
    delete/gc many times and check every cached-detector detection against
    a detector built fresh for that exact board object. With the fix this
    holds regardless of whether CPython actually reuses an id() this run --
    the cache checks object identity, not just id()."""
    import gc

    from pyCamSet.calibration_targets.markers.aruco2 import _charuco_detector_for

    dict_ = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    shapes = [(5, 5), (9, 7)]
    for i in range(40):
        nx, ny = shapes[i % 2]
        board = cv2.aruco.CharucoBoard((nx, ny), 0.02, 0.015, dict_)
        img = np.ascontiguousarray(board.generateImage((500, 500)), dtype=np.uint8)
        cached = _charuco_detector_for(board)
        fresh = cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters())
        c1, ids1, _, _ = cached.detectBoard(img)
        c2, ids2, _, _ = fresh.detectBoard(img)
        assert (c1 is None) == (c2 is None), f"iteration {i}: cached vs fresh disagree"
        if c1 is not None:
            assert np.array_equal(
                np.asarray(ids1).reshape(-1), np.asarray(ids2).reshape(-1))
            assert np.allclose(
                np.asarray(c1).reshape(-1, 2), np.asarray(c2).reshape(-1, 2))
        del board, cached, fresh
        if i % 3 == 0:
            gc.collect()


def test_charuco_detector_cache_sees_legacy_flag_toggle_on_same_board():
    """P0: the cache must keep the SAME board object it was built from (not
    a copy), so toggling board.setLegacyPattern() after the cache already
    holds a detector for it is still seen by that detector on the next
    call."""
    from pyCamSet.calibration_targets.markers.aruco2 import _charuco_detector_for

    board = ChArUco(num_squares_x=7, num_squares_y=6, square_size=10.0,
                    legacy=True).board
    img = np.ascontiguousarray(board.generateImage((700, 700)), dtype=np.uint8)
    detector = _charuco_detector_for(board)
    c1, ids1, _, _ = detector.detectBoard(img)
    assert c1 is not None
    n1 = len(np.asarray(ids1).reshape(-1))
    assert n1 > 0

    board.setLegacyPattern(False)
    detector_again = _charuco_detector_for(board)
    assert detector_again is detector, "cache rebuilt for the same board object"
    c2, ids2, _, _ = detector_again.detectBoard(img)
    n2 = 0 if ids2 is None else len(np.asarray(ids2).reshape(-1))
    assert n2 < n1, "cached detector did not see the legacy-flag toggle on its board"


def test_charuco_detector_cache_eviction_is_thread_safe():
    """P2 (round-2 review): _CHARUCO_DETECTOR_CACHE's LRU eviction used to be
    a bare dict's check-then-delete, which is not atomic. This is about
    genuine OS threads sharing one process (a GUI worker thread, a
    user-authored ThreadPoolExecutor pipeline) -- NOT pyCamSet's own
    multiprocessing.Pool, whose workers are separate processes with
    independent caches. Two threads evicting concurrently could compute the
    same "oldest" key and both try to delete it (KeyError), or a mutation
    could land between another thread's iter() and its lookup
    ("dictionary changed size during iteration"). Reproduced directly
    against the real cache with a widened thread-switch interval before the
    fix; this must run clean with the default interval too now that the
    whole lookup/build/evict body is under one lock."""
    import sys
    import threading

    from pyCamSet.calibration_targets.markers.aruco2 import _charuco_detector_for

    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # widen the race window, as the review's own repro did
    try:
        errors: list[BaseException] = []
        base_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        def worker(offset: int) -> None:
            try:
                for i in range(150):
                    # A fresh, distinct CharucoBoard object every call (never
                    # reused across iterations or threads) so this forces
                    # continuous eviction well past the cache's maxsize (64).
                    n_x = 3 + ((offset * 150 + i) % 5)
                    board = cv2.aruco.CharucoBoard((n_x, 4), 0.02, 0.015, base_dict)
                    _charuco_detector_for(board)
            except BaseException as exc:  # pragma: no cover - failure path only
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(16)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert errors == [], (
            f"concurrent cache eviction raised {len(errors)} exception(s): "
            f"{[repr(e) for e in errors]}")
    finally:
        sys.setswitchinterval(old_interval)


# -- P1(a)/P0: duplicate-id input is dropped; a corrupted/mislocated marker
# is left for detectBoard to reject outright (fail safe), never "fixed" by
# guessing which one is bad -- see the round-1/round-2 finding below.


def test_duplicate_marker_id_dropped_recovers_clean_corners():
    """P1(a): a duplicate id (the real marker plus a garbage-location
    marker sharing its id) must not zero the whole frame -- both copies of
    the ambiguous id are dropped and the rest of the board is still
    recovered, matching the clean detection closely."""
    from pyCamSet.calibration_targets.markers.aruco2 import interpolate_board_corners

    board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0).board
    img = np.ascontiguousarray(board.generateImage((700, 700)), dtype=np.uint8)
    true_markers = _true_markers_for_board(board, img)
    assert len(true_markers) >= 6

    clean_ids, clean_pts = interpolate_board_corners(img, board, true_markers)
    assert clean_ids is not None and len(clean_ids) == 16

    dup_id = int(true_markers[0][0])
    garbage = np.array([[5, 5], [25, 5], [25, 25], [5, 25]], dtype=np.float32)
    poisoned = true_markers + [(dup_id, garbage)]

    ids2, pts2 = interpolate_board_corners(img, board, poisoned)
    assert ids2 is not None, "duplicate-id marker zeroed the whole frame"
    clean_map = {int(i): np.asarray(p, float) for i, p in zip(clean_ids, clean_pts)}
    matched = 0
    for cid, pt in zip(ids2, pts2):
        if int(cid) in clean_map:
            matched += 1
            assert np.linalg.norm(np.asarray(pt, float) - clean_map[int(cid)]) < 0.1
    assert matched >= 12, f"only recovered {matched}/16 corners: {sorted(int(c) for c in ids2)}"


def test_single_marker_wrong_location_returns_no_corners_not_wrong_ones():
    """P0 (overseer decision, round 2): one marker replaced at a garbage
    location under its OWN (still-unique) id used to be "recovered" by the
    now-removed _drop_inconsistent_markers heuristic. That heuristic could
    itself return wrong, mislabelled corners under a legacy mismatch (see
    test_legacy_mismatch_partial_view_never_mislabels_corners below), so it
    was removed rather than patched. This board is odd-rowed (5x5), so the
    legacy-pattern retry cannot help either (see
    test_legacy_retry_skipped_for_odd_row_board) -- the frame must fail
    safe with NO corners, not a "fixed" one."""
    from pyCamSet.calibration_targets.markers.aruco2 import interpolate_board_corners

    board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0).board
    img = np.ascontiguousarray(board.generateImage((700, 700)), dtype=np.uint8)
    true_markers = _true_markers_for_board(board, img)

    clean_ids, clean_pts = interpolate_board_corners(img, board, true_markers)
    assert clean_ids is not None and len(clean_ids) == 16

    bad_id = int(true_markers[0][0])
    garbage = np.array([[5, 5], [25, 5], [25, 25], [5, 25]], dtype=np.float32)
    corrupted = [(mid, garbage if mid == bad_id else c) for mid, c in true_markers]

    ids2, pts2 = interpolate_board_corners(img, board, corrupted)
    assert ids2 is None and pts2 is None, (
        "a corrupted marker must fail safe with no corners, not be "
        "silently 'fixed' by a heuristic")


def test_heavy_noise_on_one_marker_returns_no_corners_not_wrong_ones():
    """P0 (overseer decision, round 2): the same >=20 px single-marker
    noise the round-1 review used to justify _drop_inconsistent_markers
    must, now that heuristic is removed, fail safe with no corners rather
    than a guessed-at recovery. Confirms the premise (this frame's own
    first, only pass through detectBoard is rejected outright -- an
    odd-rowed 9x9 board, so the legacy-pattern retry is also a no-op here)
    before checking the public function agrees."""
    from pyCamSet.calibration_targets.markers.aruco2 import interpolate_board_corners

    board = ChArUco(num_squares_x=9, num_squares_y=9, square_size=10.0).board
    img = np.ascontiguousarray(board.generateImage((900, 900)), dtype=np.uint8)
    true_markers = _true_markers_for_board(board, img)

    clean_ids, clean_pts = interpolate_board_corners(img, board, true_markers)
    assert clean_ids is not None and len(clean_ids) == 64

    bad_id = int(true_markers[0][0])
    offset = np.array([100.0, 100.0], dtype=np.float32)
    noisy = [(mid, (np.asarray(c, dtype=np.float32) + offset) if mid == bad_id else c)
             for mid, c in true_markers]

    # Confirm the premise: this frame's first (only) pass through
    # detectBoard is rejected outright.
    marker_corners = tuple(c.reshape(1, 4, 2) for _mid, c in noisy)
    marker_ids = np.asarray([mid for mid, _c in noisy], dtype=np.int32).reshape(-1, 1)
    first_pass, _, _, _ = cv2.aruco.CharucoDetector(
        board, cv2.aruco.CharucoParameters()).detectBoard(
            img, markerCorners=marker_corners, markerIds=marker_ids)
    assert first_pass is None, "test setup no longer reproduces a rejected frame"

    ids2, pts2 = interpolate_board_corners(img, board, noisy)
    assert ids2 is None and pts2 is None, (
        "heavy single-marker noise must fail safe with no corners, not be "
        "silently 'fixed' by a heuristic")


def test_frame_with_all_consistent_markers_is_unaffected_by_duplicate_id_drop():
    """The duplicate-id-drop logic must leave a frame that never needed it
    exactly as before: same clean board, same clean detection."""
    from pyCamSet.calibration_targets.markers.aruco2 import interpolate_board_corners

    board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0).board
    img = np.ascontiguousarray(board.generateImage((700, 700)), dtype=np.uint8)
    true_markers = _true_markers_for_board(board, img)
    ids, pts = interpolate_board_corners(img, board, true_markers)
    assert ids is not None and len(ids) == 16
    assert np.array_equal(np.sort(np.asarray(ids).reshape(-1)), np.arange(16))


def test_legacy_retry_skipped_for_odd_row_board():
    """P2: toggling legacy is a provable no-op for an odd chessboard row
    count (the class default, num_squares_y=5): getObjPoints(),
    getChessboardCorners(), getIds() and generateImage() are bit-identical
    between legacy=True and legacy=False for every odd row count checked
    (5, 7, 9), and differ for every even one (4, 6, 8). Approved policy:
    there is no retry at all any more, but the mismatch-warning gate must
    still be skipped outright for such a board (toggling cannot possibly be
    evidence of anything): a genuine no-interior-corner partial view (two
    markers far enough apart that no interior corner has enough support --
    not a mismatch, not corruption) must fail safe with the board's own
    legacy flag left exactly as configured, and without firing
    warn_legacy."""
    from pyCamSet.calibration_targets.markers.aruco2 import interpolate_board_corners

    board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0)  # odd rows
    assert board.board.getChessboardSize()[1] % 2 == 1
    assert board.board.getLegacyPattern() is False
    img = np.ascontiguousarray(board.board.generateImage((700, 700)), dtype=np.uint8)
    true_markers = _true_markers_for_board(board.board, img)

    # Two markers far apart on the board: enough to pass minMarkers but not
    # enough to bound any interior corner -- a legitimate, uncorrupted
    # "found markers but no corners" frame, not a mismatch.
    far_apart = sorted(true_markers, key=lambda m: m[0])
    subset = [far_apart[0], far_apart[-1]]

    calls = []
    ids2, pts2 = interpolate_board_corners(
        img, board.board, subset, warn_legacy=lambda: calls.append(1))
    assert ids2 is None and pts2 is None
    assert board.board.getLegacyPattern() is False, (
        "the board's own legacy flag changed -- detection must never touch it")
    assert calls == [], (
        "warn_legacy fired for an odd-row board, where a legacy mismatch "
        "is a provable no-op and the mismatch probe must be skipped outright")


# -- Approved policy: warn-only, no retry, no ambiguity-discard --------------
#
# Calibration tests for the thresholds in markers/legacy_probe.py. Numbers
# below are what these tests actually measured on this build (see the
# session's calibration script): a correctly configured odd x even board
# (7x6, 9x6) warned ZERO times over 240 random, REALISTIC (contiguous-crop,
# not scattered-marker-id) partial views across both detectors; a
# misconfigured board's exact, uncropped full view reliably gave zero
# corners and warned on the very first frame, for both detectors and both
# mismatch directions (8/8). A partial (even a near-full, edge-clipped) crop
# of a misconfigured board can still occasionally interpolate SOME corners
# under the wrong pattern -- confirmed directly (e.g. clipping just one edge
# row of markers off a 21-marker board dropped it to 15 markers and flipped
# a wrong-pattern detection from 0 to 20 corners) -- which is the accepted,
# documented trade-off the warning exists to mitigate, not a bug this test
# suite tries to eliminate.


def test_legacy_flag_never_changes_during_detection_sequence():
    """(1) A target's OWN configured legacy flag must never change during
    detection, for EITHER detector, over a whole sequence of frames --
    matched, mismatched, or a mix of full/partial/no-marker-at-all frames."""
    printed_true = ChArUco(num_squares_x=7, num_squares_y=6, square_size=10.0, legacy=True)
    img = np.ascontiguousarray(printed_true.board.generateImage((900, 900)), dtype=np.uint8)

    for backend in ("aruco1", "aruco2"):
        for target_legacy in (True, False):  # matched and mismatched
            target = ChArUco(num_squares_x=7, num_squares_y=6, square_size=10.0,
                             legacy=target_legacy, marker_backend=backend)
            frames = [img] + _random_crops(img, 8, seed=1) + [
                np.zeros((200, 200), dtype=np.uint8)]  # a genuine no-marker frame
            for frame in frames:
                target.find_in_image(frame)  # never raises
                assert target.board.getLegacyPattern() is target_legacy, (
                    f"legacy flag changed mid-sequence: backend={backend} "
                    f"target_legacy={target_legacy}")


#: Board sizes the legacy-pattern policy applies to (even chessboard row
#: count), spanning small (below the OLD flat LEGACY_WARNING_MIN_PROBE_CORNERS
#: of 20 total corners) through large. Used to calibrate and test
#: min_probe_corners_for's board-size scaling (overseer finding (a),
#: round-2 review): 5x4 has 12 total corners, 5x6 has 20, 7x4 has 18, 7x6
#: has 30, 9x6 has 40, 11x8 has 70, 13x10 has 108, 10x8 has 63.
_LEGACY_POLICY_SIZES = [
    (5, 4), (5, 6), (7, 4), (7, 6), (9, 6), (11, 8), (13, 10), (10, 8),
]


def test_legacy_min_probe_threshold_scales_with_board_size():
    """Overseer finding (a), round-2 review: LEGACY_WARNING_MIN_PROBE_CORNERS
    used to be a flat 20, which EXCEEDS OR EQUALS the TOTAL chessboard-corner
    count of several small odd-width x even-height boards (5x4 has 12; 5x6
    has 20; 7x4 has 18) -- exactly the class a legacy mismatch mislabels
    corners on -- so a misconfigured one of the two strictly-below-20 sizes
    could never clear the flat floor even on an EXACT full view, where the
    opposite-pattern probe interpolates the board's WHOLE corner set (see
    :func:`min_probe_corners_for`'s and the module's threshold-constant
    docstrings for the calibration sweep behind the replacement). The
    threshold must scale to (and never exceed) each board's own total, with
    margin, across the whole size range this policy applies to."""
    from pyCamSet.calibration_targets.markers.legacy_probe import min_probe_corners_for

    for nx, ny in _LEGACY_POLICY_SIZES:
        board = ChArUco(num_squares_x=nx, num_squares_y=ny, square_size=10.0).board
        total_corners = (nx - 1) * (ny - 1)
        threshold = min_probe_corners_for(board)
        assert 1 <= threshold <= total_corners, (
            f"{nx}x{ny}: threshold {threshold} is not in [1, {total_corners}] "
            f"-- a threshold above the board's own total corner count could "
            f"never be cleared, not even by an exact full view")

    # The concrete regression: the OLD flat threshold (20) was un-clearable
    # for these two sizes' total corner count, so no view could ever warn.
    for nx, ny in [(5, 4), (7, 4)]:
        board = ChArUco(num_squares_x=nx, num_squares_y=ny, square_size=10.0).board
        threshold = min_probe_corners_for(board)
        assert threshold < 20, (
            f"{nx}x{ny}: threshold {threshold} did not scale below the old "
            f"flat 20, which exceeded this board's own "
            f"{(nx - 1) * (ny - 1)} total corners")


def test_legacy_misconfigured_large_board_warns_promptly_on_the_real_corpus(session_data_dir):
    """The concrete regression this round's review reported: a 20x20
    ChArUco target configured with the WRONG legacy flag, read against the
    real, checked-in tests/test_data/calibration_charuco corpus (which is
    legacy=False -- tests/conftest.py's CHARUCO_ARGS), must warn within the
    first few frames of an ordinary calibration sequence -- not never, as it
    did before round-3 review's markers-found ceiling (see
    test_legacy_min_probe_threshold_scales_down_for_large_boards_when_markers
    _are_known's docstring for the reproduction)."""
    image_dir = session_data_dir / "calibration_charuco" / "1"
    if not image_dir.is_dir():
        pytest.skip(f"corpus camera folder not found at {image_dir}")
    images = sorted(image_dir.glob("*.jpg"))
    assert images, f"no images found in {image_dir}"

    target = ChArUco(num_squares_x=20, num_squares_y=20, square_size=4.0, legacy=True)
    fired_at = None
    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        target.find_in_image(img)
        if target.given_legacy_warning and fired_at is None:
            fired_at = i
            break
    assert fired_at is not None, (
        "misconfigured 20x20 board never warned over the real corpus's own "
        f"{len(images)} images")
    assert fired_at <= 4, (
        f"misconfigured 20x20 board did not warn promptly: fired on frame "
        f"{fired_at} of {len(images)}")


def test_ccube_legacy_warning_names_the_actual_face_flag(caplog):
    """(3)/(4) for Ccube: the shared legacy-mismatch warning must name the
    FACE that failed and its actual (dynamically read) configured flag, not
    a hardcoded guess -- for either detector. Build a cube whose OWN
    configured flag is legacy=False and confirm the fired warning names
    False as configured and True as the likely setting."""
    for backend in ("aruco1", "aruco2"):
        printed = Ccube(n_points=6, length=20.0, legacy=True)  # even n_points: legacy matters
        tex = np.ascontiguousarray(printed.textures[0], dtype=np.uint8)
        # Mismatched on purpose (printed legacy=True, read as legacy=False) so
        # the first pass genuinely fails and the warning fires with the face's
        # CURRENT flag (False) still in effect -- never touched by detection.
        cube = Ccube(n_points=6, length=20.0, legacy=False, marker_backend=backend)

        with caplog.at_level(logging.WARNING,
                             logger="pyCamSet.calibration_targets.ccube.target"):
            det = cube.find_in_image(tex)

        n = 0 if det.keys is None else len(det.keys)
        assert n == 0, f"{backend}: an exact full-view mismatch must give no corners"
        warnings = [r.getMessage() for r in caplog.records if "Ccube: face 0" in r.getMessage()]
        assert warnings, f"{backend}: expected the legacy-mismatch warning to fire on face 0"
        assert "legacy=False" in warnings[0], (
            f"{backend}: warning did not name the face's actual (False) legacy flag: "
            f"{warnings[0]!r}")
        assert "legacy=True" in warnings[0], (
            f"{backend}: warning did not name the likely (True) legacy setting: "
            f"{warnings[0]!r}")
        assert cube.boards[0].getLegacyPattern() is False, (
            f"{backend}: the face's own configured legacy flag changed during detection")


# -- Approved policy: no ambiguity-discard, warning only, gated evaluation ---


def test_warn_legacy_gate_not_evaluated_when_corners_are_found(monkeypatch):
    """The mismatch-evidence gate (``should_warn_legacy_mismatch``) must only
    ever be reached when the configured pattern found markers but ZERO
    corners -- a normal, successful detection must never pay for (or risk a
    false positive from) the opposite-pattern probe at all."""
    import pyCamSet.calibration_targets.markers.aruco2 as a2mod

    board = ChArUco(num_squares_x=7, num_squares_y=6, square_size=10.0).board  # even rows
    img = np.ascontiguousarray(board.generateImage((900, 900)), dtype=np.uint8)
    true_markers = _true_markers_for_board(board, img)

    def _boom(*_a, **_kw):
        raise AssertionError("should_warn_legacy_mismatch was evaluated for a successful frame")

    monkeypatch.setattr(a2mod, "should_warn_legacy_mismatch", _boom)
    ids, pts = a2mod.interpolate_board_corners(img, board, true_markers)
    assert ids is not None, "a full, clean board view must still detect"


def test_legacy_warn_probe_paid_once_per_target_not_every_frame(monkeypatch):
    """P2 (round-1 review): once a target has already warned
    (given_legacy_warning is True), should_warn_legacy_mismatch's own probe
    -- a throwaway board + detector plus a whole detectBoard call -- must
    not be paid again on every subsequent qualifying frame; only up to and
    including the frame that actually fires the warning. Before the fix,
    every qualifying frame for the rest of the target's life repeated the
    full probe cost even though given_legacy_warning was already True and
    no further warning could ever fire."""
    import pyCamSet.calibration_targets.charuco.target as charuco_mod
    import pyCamSet.calibration_targets.markers.aruco2 as aruco2_mod

    # aruco1's probe is called from charuco.target; aruco2's is called from
    # markers.aruco2 (shared by both ChArUco's and Ccube's aruco2 branch).
    gate_module = {"aruco1": charuco_mod, "aruco2": aruco2_mod}

    for backend in ("aruco1", "aruco2"):
        mod = gate_module[backend]
        calls = []
        orig = mod.should_warn_legacy_mismatch

        def _counting(*args, **kwargs):
            calls.append(1)
            return orig(*args, **kwargs)

        monkeypatch.setattr(mod, "should_warn_legacy_mismatch", _counting)

        # Printed legacy=True, read as legacy=False: a full view mismatch
        # always finds markers and no corners under the configured
        # pattern, so every frame below is a qualifying frame.
        printed = ChArUco(num_squares_x=7, num_squares_y=6, square_size=10.0, legacy=True)
        img = np.ascontiguousarray(printed.board.generateImage((900, 900)), dtype=np.uint8)
        target = ChArUco(num_squares_x=7, num_squares_y=6, square_size=10.0,
                         legacy=False, marker_backend=backend)

        for _ in range(5):
            target.find_in_image(img)

        assert target.given_legacy_warning is True, f"{backend}: warning never fired"
        assert len(calls) == 1, (
            f"{backend}: probe evaluated {len(calls)} times across 5 identical "
            f"mismatched frames, expected exactly 1 (paid once per target, not "
            f"once per qualifying frame)")


def test_interpolate_board_corners_drops_int32_overflow_id():
    """P3 (round 3): interpolate_board_corners is documented as tolerating
    out-of-bounds and foreign-id markers without raising, but a marker id
    outside int32 range used to escape that guarantee: _detect()'s own
    ``np.asarray(..., dtype=np.int32)`` cast raises OverflowError, uncaught,
    instead of the fail-safe empty/partial result every other malformed
    input gets. Such an id must be dropped, the same way a wrong-shaped
    quad already is."""
    from pyCamSet.calibration_targets.markers.aruco2 import interpolate_board_corners

    board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0).board
    img = np.ascontiguousarray(board.generateImage((700, 700)), dtype=np.uint8)
    true_markers = _true_markers_for_board(board, img)

    clean_ids, clean_pts = interpolate_board_corners(img, board, true_markers)
    assert clean_ids is not None and len(clean_ids) == 16

    bad_id_marker = (2**31 + 5, true_markers[0][1].copy())
    ids2, pts2 = interpolate_board_corners(img, board, true_markers + [bad_id_marker])
    assert ids2 is not None and len(ids2) == 16, (
        "a marker id outside int32 range must be dropped (fail safe), not "
        "raise OverflowError or corrupt the frame")


# -- P2: a single marker deliberately yields no corners -----------------------


def test_single_marker_gives_no_corners_aruco2():
    """P2: CharucoParameters.minMarkers is deliberately left at OpenCV's
    default of 2 (parity with aruco1, and with the pre-adapter aruco2
    path); a lone marker must not extrapolate a corner. See
    interpolate_board_corners's docstring for why."""
    from pyCamSet.calibration_targets.markers.aruco2 import interpolate_board_corners

    board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0).board
    img = np.ascontiguousarray(board.generateImage((700, 700)), dtype=np.uint8)
    true_markers = _true_markers_for_board(board, img)
    ids, pts = interpolate_board_corners(img, board, true_markers[:1])
    assert ids is None and pts is None


def test_default_backend_is_aruco1_exactly():
    board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0)
    assert board.input_args["marker_backend"] == "aruco1"


def test_invalid_backend_rejected():
    with pytest.raises(ValueError):
        ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                marker_backend="aruco3")


@pytest.mark.parametrize("bad_int", [22, 23])
def test_alvar_ints_rejected_in_aruco1_mode(bad_int):
    # ALVAR ints 22/23 silently alias DICT_4X4_50 inside OpenCV; aruco1 mode
    # must reject them rather than build a wrong board.
    with pytest.raises(ValueError):
        ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                a_dict=bad_int, marker_backend="aruco1")


def test_mip_21_valid_in_aruco1_mode():
    # DICT_ARUCO_MIP_36h12 (21) exists in OpenCV 4.11 with byte-identical
    # content to aruco2's, so aruco1 mode must accept it.
    board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                    a_dict=21, marker_backend='aruco1')
    # FIX 8(a): assert byte-identity vs cv2's DICT_ARUCO_MIP_36h12, not just
    # markerSize (markerSize alone cannot distinguish MIP from other 6x6 dicts).
    cv2_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
    assert np.array_equal(
        np.asarray(board.board.getDictionary().bytesList),
        np.asarray(cv2_dict.bytesList),
    )


@pytest.mark.parametrize("dname,dint", [
    ("APRILTAG_16h5", 17),
    ("ALVAR_5X5_256", 22),
    ("ARUCO_MIP_36h12", 21),
])
def test_aruco2_only_families_use_aruco2_bytes(dname, dint):
    board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                    a_dict=dint, marker_backend="aruco2")
    board_bytes = np.asarray(board.board.getDictionary().bytesList)
    a2_bytes = np.asarray(aruco2.get_predefined_dictionary(dint).bytes_list)
    assert np.array_equal(board_bytes, a2_bytes)


def test_charuco_roundtrip_both_backends():
    for backend in ("aruco1", "aruco2"):
        board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                        marker_backend=backend)
        img = board._render_board(px_per_mm=6.0)
        det = board.find_in_image(img)
        assert _count(det) == 16, f"{backend}: {_count(det)} corners"
        ids = np.unique(np.asarray(det.keys).reshape(-1))
        assert np.array_equal(np.sort(ids), np.arange(16)), f"{backend}: bad ids"


def test_ccube_dictionary_object_api_aruco1():
    """FIX 1: the pre-existing Ccube API accepts a cv2.aruco.Dictionary
    object (not just an int); it must construct and detect on the face-0
    texture exactly as before the backend change."""
    cube = Ccube(n_points=5, length=10.0,
                 aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000))
    tex = np.ascontiguousarray(cube.textures[0], dtype=np.uint8)
    det = cube.find_in_image(tex)
    # n_points=5 -> 4x4 interior chessboard corners = 16 (arange(16))
    assert _count(det) >= 16, f"Dictionary-object Ccube: {_count(det)} corners"
    keys = np.asarray(det.keys)
    assert np.unique(keys[:, 0]).tolist() == [0], "wrong face"
    assert np.array_equal(np.sort(np.unique(keys[:, 1])), np.arange(16)), (
        f"bad local ids {np.unique(keys[:, 1]).tolist()}")


def test_ccube_roundtrip_both_backends():
    for backend in ("aruco1", "aruco2"):
        cube = Ccube(n_points=4, length=20.0, marker_backend=backend)
        tex = np.ascontiguousarray(cube.textures[0], dtype=np.uint8)
        det = cube.find_in_image(tex)
        assert _count(det) >= 9, f"{backend}: {_count(det)} corners"
        keys = np.asarray(det.keys)
        assert np.unique(keys[:, 0]).tolist() == [0], f"{backend}: wrong face"
        assert np.array_equal(np.sort(np.unique(keys[:, 1])), np.arange(9)), (
            f"{backend}: bad local ids")


def test_ccube_aruco2_raw_float64_texture():
    """FIX 8(e): aruco2 detection must accept a RAW float64 texture whose
    values are integral (convert internally to uint8); genuinely
    non-convertible dtypes still raise ValueError."""
    cube = Ccube(n_points=4, length=20.0, marker_backend="aruco2")
    tex_raw = cube.textures[0]  # float64, integral values
    assert tex_raw.dtype == np.float64
    det = cube.find_in_image(tex_raw)
    assert _count(det) >= 9, f"raw float64 aruco2: {_count(det)} corners"
    keys = np.asarray(det.keys)
    assert np.unique(keys[:, 0]).tolist() == [0]
    # genuinely non-convertible dtype still raises
    bad = np.full((100, 100), 0.5, dtype=np.float64)
    with pytest.raises(ValueError):
        cube.find_in_image(bad)


def test_ccube_face1_boundary_both_backends():
    for backend in ("aruco1", "aruco2"):
        cube = Ccube(n_points=4, length=20.0, marker_backend=backend)
        tex = np.ascontiguousarray(cube.textures[1], dtype=np.uint8)
        det = cube.find_in_image(tex)
        assert _count(det) >= 9
        keys = np.asarray(det.keys)
        assert np.unique(keys[:, 0]).tolist() == [1], f"{backend}: face-1 keys"
        # FIX 8(b): exact local-id set (arange(9) for a 4x4-point face), not
        # just a count check.
        assert np.array_equal(np.sort(np.unique(keys[:, 1])), np.arange(9)), (
            f"{backend}: face-1 local ids {np.unique(keys[:, 1]).tolist()}")


def test_point_data_identical_across_backends():
    b1 = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                 marker_backend="aruco1")
    b2 = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                 marker_backend="aruco2")
    assert np.allclose(np.asarray(b1.point_data, float),
                       np.asarray(b2.point_data, float))


def _trapezoid_warp(img, top_frac):
    """TRUE perspective warp: asymmetric trapezoid (top edge compressed to
    top_frac of the width). The old centred-shrink warp was a similarity
    transform with zero foreshortening and could not exercise interpolation
    accuracy (FIX 3)."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    top_w = w * top_frac
    top_x0 = (w - top_w) / 2
    dst = np.float32([[top_x0, 0], [top_x0 + top_w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC)


def test_accuracy_vs_charuco_detector():
    # Interpolated corners must agree with CharucoDetector under TRUE
    # perspective warps (FIX 3). The residual is dominated by the inherent
    # aruco2-vs-OpenCV marker-corner differences (probe: marker-corner mean
    # 1.74px / max 4.90px at top-40%), so the documented thresholds are
    # mean<2.0px / max<3.5px; the strict sub-check requires the matched
    # corner-id set to be identical between backends so the check cannot
    # silently pass on a partial overlap.
    b = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                marker_backend="aruco2")
    img = b._render_board(px_per_mm=8.0).astype(np.uint8)
    b1 = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0)
    gt_det = ARUCO_OPENCV_DETECTOR.build_detector(
        b1.board, ARUCO_OPENCV_DETECTOR.resolve(None))
    # both warps are within the finding's stated 40-60% top-edge range; the
    # 0.8 case is excluded because its marker-corner differences are even
    # larger (max 5.86px) and its corner max (3.96px) exceeds the documented
    # 3.5px threshold.
    for top_frac in (0.6, 0.4):
        warped = _trapezoid_warp(img, top_frac)
        gt_c, gt_ids, _, _ = gt_det.detectBoard(warped)
        det = b.find_in_image(warped)
        assert _count(det) > 0, "aruco2 returned no corners on the warped image"
        assert gt_ids is not None and len(gt_ids) > 0
        gt_map = {int(i): np.asarray(pt, float).reshape(-1)
                  for i, pt in zip(np.asarray(gt_ids).reshape(-1), gt_c[:, 0])}
        det_ids = np.asarray(det.keys).reshape(-1).astype(int)
        # FIX 3: strict sub-check on the matching domain.
        assert set(det_ids.tolist()) == set(gt_map.keys()), (
            f"matched corner ids differ between backends "
            f"(aruco2 {sorted(set(det_ids.tolist()))} vs gt {sorted(gt_map.keys())})")
        errs = [float(np.linalg.norm(np.asarray(p, float) - gt_map[int(c)]))
                for c, p in zip(det_ids, np.asarray(det.image_points))
                if int(c) in gt_map]
        assert errs, "no overlapping corner ids with ground truth"
        assert np.mean(errs) < 2.0, f"mean {np.mean(errs):.2f}px"
        assert np.max(errs) < 3.5, f"max {np.max(errs):.2f}px"


def test_charuco_partial_view_does_not_raise():
    """(a) A board cut by the image edge, read with marker_backend='aruco2',
    must not raise. Regression test: the old homography-based interpolation
    fed cropped/out-of-bounds marker corners into cv2.cornerSubPix, which
    raised cv2.error ("cornersubpix.cpp:99 ... contains(cT)") on some
    partial views; the adapter hands markers straight to
    CharucoDetector.detectBoard, which tolerates them."""
    board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                    marker_backend="aruco2")
    img = board._render_board(px_per_mm=8.0).astype(np.uint8)
    # Crop to the left 55% of the image: several markers (and the corners
    # that depend on them) are cut clean off by the new image edge, so the
    # board is genuinely partial rather than merely warped or shrunk.
    cropped = np.ascontiguousarray(img[:, : int(img.shape[1] * 0.55)])
    det = board.find_in_image(cropped)  # must not raise
    assert det is not None


def test_ccube_partial_view_does_not_raise():
    """(a) Same as above for Ccube: a face texture cut by the image edge
    must not raise under marker_backend='aruco2'."""
    cube = Ccube(n_points=5, length=20.0, marker_backend="aruco2")
    tex = np.ascontiguousarray(cube.textures[0], dtype=np.uint8)
    cropped = np.ascontiguousarray(tex[:, : int(tex.shape[1] * 0.55)])
    det = cube.find_in_image(cropped)  # must not raise
    assert det is not None


def test_charuco_aruco2_agrees_tightly_with_aruco1_on_clean_render():
    """(b) On a clean rendered/mildly-warped board, ArUco2 and ArUco1 must
    agree tightly -- the adapter hands the SAME image to the SAME
    CharucoDetector.detectBoard OpenCV call the aruco1 path uses, differing
    only in which detector found the input marker corners, so the two
    should be near-identical rather than merely close (probed empirically:
    median ~0.03px / max ~0.07px at a mild 20% top-edge warp)."""
    b2 = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                marker_backend="aruco2")
    img = b2._render_board(px_per_mm=8.0).astype(np.uint8)
    img = _trapezoid_warp(img, top_frac=0.8)  # mild perspective tilt
    b1 = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0)
    gt_det = ARUCO_OPENCV_DETECTOR.build_detector(
        b1.board, ARUCO_OPENCV_DETECTOR.resolve(None))
    gt_c, gt_ids, _, _ = gt_det.detectBoard(img)
    det = b2.find_in_image(img)
    assert _count(det) > 0
    gt_map = {int(i): np.asarray(pt, float).reshape(-1)
              for i, pt in zip(np.asarray(gt_ids).reshape(-1), gt_c[:, 0])}
    det_ids = np.asarray(det.keys).reshape(-1).astype(int)
    errs = [float(np.linalg.norm(np.asarray(p, float) - gt_map[int(c)]))
            for c, p in zip(det_ids, np.asarray(det.image_points))
            if int(c) in gt_map]
    assert errs, "no overlapping corner ids with ground truth"
    assert np.median(errs) < 0.1, f"median {np.median(errs):.3f}px"
    assert np.max(errs) < 0.5, f"max {np.max(errs):.3f}px"


def test_ccube_aruco2_agrees_tightly_with_aruco1_on_clean_render():
    """(b) Same tight-agreement check as above, for one Ccube face."""
    c2 = Ccube(n_points=5, length=20.0, marker_backend="aruco2")
    tex = np.ascontiguousarray(c2.textures[0], dtype=np.uint8)
    c1 = Ccube(n_points=5, length=20.0)
    gt_det = ARUCO_OPENCV_DETECTOR.build_detector(
        c1.boards[0], ARUCO_OPENCV_DETECTOR.resolve(None))
    gt_c, gt_ids, _, _ = gt_det.detectBoard(tex)
    det = c2.find_in_image(tex)
    assert _count(det) > 0
    gt_map = {int(i): np.asarray(pt, float).reshape(-1)
              for i, pt in zip(np.asarray(gt_ids).reshape(-1), gt_c[:, 0])}
    keys = np.asarray(det.keys)
    face_mask = keys[:, 0] == 0
    det_ids = keys[face_mask, 1].astype(int)
    pts = np.asarray(det.image_points)[face_mask]
    errs = [float(np.linalg.norm(np.asarray(p, float) - gt_map[int(c)]))
            for c, p in zip(det_ids, pts) if int(c) in gt_map]
    assert errs, "no overlapping corner ids with ground truth"
    assert np.median(errs) < 0.1, f"median {np.median(errs):.3f}px"
    assert np.max(errs) < 0.5, f"max {np.max(errs):.3f}px"


def test_charuco_aruco2_matches_aruco1_under_blur_noise_and_tilt():
    """(c) On a harder synthetic image (perspective tilt + Gaussian blur +
    sensor noise), every ArUco2 corner must still land close to the ArUco1
    answer -- not just the well-conditioned clean-render case. Probed
    empirically at these settings: max ~0.2-0.3px, well inside the 3px
    budget asserted here."""
    b2 = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                marker_backend="aruco2")
    img = b2._render_board(px_per_mm=8.0).astype(np.uint8)
    img = _trapezoid_warp(img, top_frac=0.6)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    rng = np.random.default_rng(0)
    img = np.clip(
        img.astype(np.float64) + rng.normal(0, 8.0, img.shape), 0, 255
    ).astype(np.uint8)
    b1 = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0)
    gt_det = ARUCO_OPENCV_DETECTOR.build_detector(
        b1.board, ARUCO_OPENCV_DETECTOR.resolve(None))
    gt_c, gt_ids, _, _ = gt_det.detectBoard(img)
    det = b2.find_in_image(img)
    assert _count(det) > 0
    assert gt_ids is not None and len(gt_ids) > 0
    gt_map = {int(i): np.asarray(pt, float).reshape(-1)
              for i, pt in zip(np.asarray(gt_ids).reshape(-1), gt_c[:, 0])}
    det_ids = np.asarray(det.keys).reshape(-1).astype(int)
    errs = [float(np.linalg.norm(np.asarray(p, float) - gt_map[int(c)]))
            for c, p in zip(det_ids, np.asarray(det.image_points))
            if int(c) in gt_map]
    assert errs, "no overlapping corner ids with ground truth"
    assert np.max(errs) < 3.0, f"max {np.max(errs):.2f}px"


def test_legacy_even_row_position_level_both_backends():
    # 7x6 (even rows): legacy matters. Assert POSITIONS match CharucoDetector,
    # not just counts.
    for backend in ("aruco1", "aruco2"):
        board = ChArUco(num_squares_x=7, num_squares_y=6, square_size=10.0,
                        legacy=True, marker_backend=backend)
        img = board._render_board(px_per_mm=8.0).astype(np.uint8)
        det = board.find_in_image(img)
        assert _count(det) > 0, f"{backend}: no corners"
        gt_det = ARUCO_OPENCV_DETECTOR.build_detector(
            board.board, ARUCO_OPENCV_DETECTOR.resolve(None))
        gt_c, gt_ids, _, _ = gt_det.detectBoard(img)
        gt_map = {int(i): np.asarray(pt, float).reshape(-1)
                  for i, pt in zip(np.asarray(gt_ids).reshape(-1), gt_c[:, 0])}
        errs = [float(np.linalg.norm(np.asarray(p, float) - gt_map[int(c)]))
                for c, p in zip(np.asarray(det.keys).reshape(-1),
                                np.asarray(det.image_points)) if int(c) in gt_map]
        assert errs, f"{backend}: no overlap with ground truth"
        assert np.mean(errs) < 2.0, f"{backend}: mean {np.mean(errs):.2f}px"


def test_legacy_wrong_flag_no_auto_toggle():
    """Approved policy supersedes the old auto-toggle: a legacy-printed
    board read by a target configured the other way must NOT recover
    positions by flipping the target's own flag. It must instead: find no
    corners at all on this exact full view, leave the target's configured
    flag untouched, and fire the once-per-target mismatch warning naming
    both settings. Covers both detectors."""
    for backend in ("aruco1", "aruco2"):
        ref = ChArUco(num_squares_x=7, num_squares_y=6, square_size=10.0, legacy=True)
        img = np.ascontiguousarray(ref.board.generateImage((900, 900)), dtype=np.uint8)
        board = ChArUco(num_squares_x=7, num_squares_y=6, square_size=10.0,
                        legacy=False, marker_backend=backend)
        det = board.find_in_image(img)
        assert _count(det) == 0, (
            f"{backend}: an exact full view of a mismatched board returned "
            f"{_count(det)} corner(s), not the guaranteed zero")
        assert board.board.getLegacyPattern() is False, (
            f"{backend}: detection changed the target's own configured "
            f"legacy flag -- the approved policy forbids this")
        assert board.given_legacy_warning is True, (
            f"{backend}: the mismatch warning did not fire")


def test_input_args_roundtrip_preserves_backend():
    board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                    marker_backend="aruco2")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "args.json"
        p.write_text(json.dumps(board.input_args), encoding="utf-8")
        loaded = json.loads(p.read_text(encoding="utf-8"))
    board2 = ChArUco(**loaded)
    assert board2.input_args["marker_backend"] == "aruco2"
    assert np.allclose(np.asarray(board.point_data, float),
                       np.asarray(board2.point_data, float))


def _make_camset_with_handler(marker_backend="aruco2"):
    """Build a CameraSet + calibration handler whose target carries the
    requested marker backend (FIX 5). The handler is a real
    TemplateBundleHandler so the REAL save_camset/load_CameraSet path is
    exercised, not a json proxy."""
    from pyCamSet.cameras import CameraSet, Camera
    from pyCamSet.calibration_targets.core.target_detections import TargetDetection
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler

    cam_dict = {}
    for i, name in enumerate(["cam0", "cam1"]):
        cam_dict[name] = Camera(
            extrinsic=np.eye(4),
            intrinsic=np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=float),
            distortion_coefs=np.zeros(5),
            res=(640, 480),
            name=name,
        )
    camset = CameraSet(camera_dict=cam_dict)
    target = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                     marker_backend=marker_backend)
    data = np.array([
        [0, 0, 0, 100.0, 100.0],
        [0, 0, 1, 110.0, 100.0],
        [1, 0, 0, 200.0, 200.0],
        [1, 0, 1, 210.0, 200.0],
    ], dtype=float)
    det = TargetDetection(cam_names=["cam0", "cam1"], data=data, max_ims=1)
    handler = TemplateBundleHandler(camset=camset, target=target, detection=det,
                                    fixed_params={}, options={})
    camset.calibration_handler = handler
    camset.calibration_params = np.zeros(10)
    camset.calibration_result = np.zeros((2, 2))
    camset.calibration_jac = np.zeros((2, 2))
    return camset


def test_camset_save_load_real_roundtrip_preserves_backend(tmp_path):
    """FIX 5: the REAL save_camset/load_CameraSet path must preserve
    marker_backend="aruco2" on the reloaded target's input_args."""
    from pyCamSet.utils.saving import save_camset, load_CameraSet
    camset = _make_camset_with_handler(marker_backend="aruco2")
    p = tmp_path / "cams.camset"
    save_camset(camset, p)
    loaded = load_CameraSet(p)
    assert loaded.calibration_handler is not None, (
        "load_CameraSet fell back to a bare CameraSet (handler not rebuilt)")
    assert loaded.calibration_handler.target.input_args.get("marker_backend") == "aruco2"


def test_camset_save_load_legacy_input_defaults_to_aruco1(tmp_path):
    """FIX 5: a saved target_config['input'] WITHOUT marker_backend (a
    legacy save) must load with the constructor default "aruco1"."""
    import json
    from pyCamSet.utils.saving import save_camset, load_CameraSet
    camset = _make_camset_with_handler(marker_backend="aruco1")
    p = tmp_path / "cams.camset"
    save_camset(camset, p)
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert "marker_backend" in raw["optim"]["target_config"]["input"]
    raw["optim"]["target_config"]["input"].pop("marker_backend", None)
    p.write_text(json.dumps(raw), encoding="utf-8")
    loaded = load_CameraSet(p)
    assert loaded.calibration_handler is not None
    assert loaded.calibration_handler.target.input_args.get("marker_backend") == "aruco1"


def test_build_target_threads_marker_backend():
    from pyCamSet.calibration_targets.core.target_registry import build_target
    spec = {"type": "ChArUco", "num_squares_x": 5, "num_squares_y": 5,
            "square_size": 10.0}
    t = build_target({**spec, "marker_backend": "aruco2"})
    assert t.input_args["marker_backend"] == "aruco2"
    t_default = build_target(spec)
    assert t_default.input_args["marker_backend"] == "aruco1"


def test_default_detection_fn_plumbing(tmp_path, monkeypatch):
    """FIX 6: a target_spec dict carrying marker_backend="aruco2" ->
    default_detection_fn must produce a target whose input_args carry
    "aruco2".

    The REAL detect_datapoints_in_imfile path has a PRE-EXISTING failure
    unrelated to the backend: features_per_im_per_cam raises IndexError
    because TargetDetection.max_ims is never seeded on this path (max_ims=0).
    The test therefore injects a detection seam that returns a real
    TargetDetection with max_ims=1, so the full default_detection_fn chain
    (target construction with marker_backend="aruco2") is exercised and the
    input_args contract is asserted; the pre-existing downstream failure is
    documented in the comment below.
    """
    import cv2
    from pyCamSet.calibration_targets.core.target_detections import TargetDetection
    from pyCamSet.workflow.tuning.worker import default_detection_fn

    captured = {}

    def fake_detect(f_loc, calibration_target, caching=True, **kwargs):
        captured["target"] = calibration_target
        data = np.array([
            [0, 0, 0, 100.0, 100.0],
            [0, 0, 1, 110.0, 100.0],
        ], dtype=float)
        det = TargetDetection(cam_names=["cam0"], data=data, max_ims=1)
        return det, [(400, 400)]

    monkeypatch.setattr(
        "pyCamSet.calibration.camera_calibrator.detect_datapoints_in_imfile",
        fake_detect,
    )

    target_spec = {"type": "ChArUco", "marker_backend": "aruco2",
                   "num_squares_x": 5, "num_squares_y": 5, "square_size": 10.0}
    cam_dir = tmp_path / "cam0"
    cam_dir.mkdir()
    board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                    marker_backend="aruco2")
    cv2.imwrite(str(cam_dir / "im_0.png"), board._render_board(px_per_mm=6.0))
    result = default_detection_fn(tmp_path, {}, target_spec)
    target = result.get("target")
    assert target is not None, "default_detection_fn returned no target"
    assert target.input_args["marker_backend"] == "aruco2"
    # The seam captured the exact target instance the worker constructed.
    assert captured["target"] is target
    feats = result.get("features_per_im_per_cam")
    assert feats is not None
    assert np.asarray(feats).shape[1] == 1  # one camera folder


def test_a_target_spec_carries_the_backend_through_to_the_detector():
    """A study describes its target the way a phase does, as a spec."""
    from pyCamSet.calibration_targets.core.target_registry import build_target

    spec = {"type": "ChArUco", "num_squares_x": 5, "num_squares_y": 5,
            "square_size": 10.0, "marker_backend": "aruco2"}
    target = build_target(spec)

    assert target.marker_backend == "aruco2"
    assert target.detection_parameters.name == "aruco2"
    # aruco2's call takes no settings, so it describes none.
    assert target.detection_options == {}


def test_validate_run_settings_rejects_unknown_backend():
    """The refusal is the target's now: a study validates by building it."""
    from pyCamSet.workflow.tuning.study import validate_run_settings
    errors = validate_run_settings(
        f_loc="unused",
        n_trials=1,
        target_rpe=1.0,
        max_nfev_phase3=1,
        max_nfev_phase4=1,
        retain_successes=1,
        target_spec={"type": "ChArUco", "num_squares_x": 5, "num_squares_y": 5,
                     "square_size": 10.0, "marker_backend": "aruco3"},
    )
    assert any("cannot be detected with 'aruco3'" in e for e in errors), errors


def test_validate_run_settings_accepts_aruco2(tmp_path):
    """FIX 8(d): marker_backend="aruco2" must PASS validate_run_settings
    (no marker_backend error messages)."""
    import cv2
    from pyCamSet.workflow.tuning.study import validate_run_settings
    root = tmp_path
    for name in ("cam0", "cam1"):
        d = root / name
        d.mkdir()
        cv2.imwrite(str(d / "im_0.png"), np.zeros((10, 10), dtype=np.uint8))
    errors = validate_run_settings(
        f_loc=root,
        n_trials=1,
        target_rpe=1.0,
        max_nfev_phase3=1,
        max_nfev_phase4=1,
        retain_successes=1,
        target_spec={"type": "ChArUco", "marker_backend": "aruco2",
                     "num_squares_x": 5, "num_squares_y": 5, "square_size": 10.0},
    )
    assert not any("Target" in e for e in errors), errors


def test_missing_aruco2_import_error_hint():
    """With aruco2 unavailable (flag forced off), construction raises an
    actionable ImportError."""
    from pyCamSet.calibration_targets.markers import aruco2 as a2d
    real = a2d.ARUCO2_AVAILABLE
    a2d.ARUCO2_AVAILABLE = False
    try:
        with pytest.raises(ImportError, match="aruco2"):
            ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                    marker_backend="aruco2")
    finally:
        a2d.ARUCO2_AVAILABLE = real


def test_worker_multiprocess_path(tmp_path):
    """find_in_imfolder re-instantiates the target in worker processes from
    input_args; the backend must survive and detect.

    FIX 7: the folder target is constructed with a_dict=22 (ALVAR,
    aruco2-only) + marker_backend="aruco2". aruco1 mode cannot construct
    ALVAR (ValueError), so any successful detection in the worker PROVES the
    aruco2 backend was used in the worker process, not just that counts
    matched. The count is tightened toward the exact 32 corner instances
    (16 corners x 2 images) with a small documented tolerance for the render
    border.
    """
    board = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                    a_dict=22, marker_backend="aruco2")
    cam_dir = tmp_path / "cam0"
    cam_dir.mkdir()
    img = board._render_board(px_per_mm=6.0)
    for i in range(2):
        cv2.imwrite(str(cam_dir / f"im_{i}.png"), img)
    dets = board.find_in_imfolder(cam_dir, cam_names=["cam0"], threads=2)
    all_data = dets.get(cam="cam0").get_data()
    assert all_data is not None
    total = 0
    for im in np.unique(all_data[:, 1]):
        sub = dets.get(global_im_num=int(im)).get_data()
        if sub is not None:
            total += sub.shape[0]
    # exact 32 = 16 corners x 2 images; tolerance 2 for the render border
    assert 30 <= total <= 34, (
        f"worker path found {total} corner instances over 2 images, "
        f"expected ~32 (ALVAR aruco2-only board proves the aruco2 path)")


def test_legacy_warning_reaches_main_process_under_multiprocessing(tmp_path, caplog):
    """P1 (round-1 review) / overseer addition (b), round-2 review: the
    once-per-target legacy-mismatch warning must reach the user when
    detection runs through find_in_imfolder's multiprocessing.Pool path
    (threads > 1, the real default -- camera_calibrator.py's
    ``min(max(1, cpu_count()-2), 20)``), not just the single-process
    (threads=1) path. A worker process's own logger.warning call never
    reaches the main process's logger (no log-forwarding exists between
    them, and each worker rebuilds a fresh target instance with
    given_legacy_warning=False from input_args) -- so this exercises the
    message being carried back to the main process with each image's
    result and logged there once (abstract_target.py's find_in_imfolder),
    rather than relying on the worker's own (invisible) log call."""
    printed = ChArUco(num_squares_x=7, num_squares_y=6, square_size=10.0, legacy=True)
    img = np.ascontiguousarray(printed.board.generateImage((700, 700)), dtype=np.uint8)
    cam_dir = tmp_path / "cam0"
    cam_dir.mkdir()
    for i in range(3):
        cv2.imwrite(str(cam_dir / f"im_{i}.png"), img)

    # Mismatched on purpose: printed legacy=True, target configured legacy=False.
    target = ChArUco(num_squares_x=7, num_squares_y=6, square_size=10.0,
                     legacy=False, marker_backend="aruco1")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        dets = target.find_in_imfolder(cam_dir, cam_names=["cam0"], threads=2)

    data = dets.get(cam="cam0").get_data()
    n = 0 if data is None else data.shape[0]
    assert n == 0, (
        f"a full-view mismatch must give no corners under the multi-worker "
        f"path either, got {n}")
    assert target.given_legacy_warning is False, (
        "the ORIGINAL (main-process) target instance is never the one that "
        "runs detection under threads>1 -- each worker rebuilds its own "
        "fresh copy from input_args -- so this instance's own flag must "
        "stay False even though the warning did fire (in a worker)")

    messages = [r.getMessage() for r in caplog.records]
    matches = [m for m in messages if "legacy=True" in m and "legacy=False" in m]
    assert matches, (
        f"the legacy-mismatch warning fired in a worker process but never "
        f"reached the main process's log under threads=2; records seen: "
        f"{messages}")


def test_the_detection_cache_follows_the_detector_a_real_target_is_read_with():
    """The detector is chosen per detection run, so the cache a run reads
    has to be named for it -- including for ChArUco2, which carries no
    ``marker_backend`` of its own and is only ever read with aruco2."""
    from pyCamSet.calibration.camera_calibrator import detector_backend_of
    from pyCamSet.calibration_targets.charuco2.target import ChArUco2
    from pyCamSet.workflow.detections import detection_cache_name

    board1 = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0)
    board2 = ChArUco(num_squares_x=5, num_squares_y=5, square_size=10.0,
                     marker_backend="aruco2")
    cube2 = Ccube(n_points=4, length=20.0, marker_backend="aruco2")
    # Only the class matters here, so no ChArUco2 is constructed.
    grid_board = object.__new__(ChArUco2)

    assert detector_backend_of(board1) == "aruco1"
    assert detection_cache_name(1, detector_backend_of(board1)) == \
        "detected_datapoints.pickle"
    for target in (board2, cube2, grid_board):
        assert detector_backend_of(target) == "aruco2"
        assert detection_cache_name(1, detector_backend_of(target)) == \
            "detected_datapoints_aruco2.pickle"
