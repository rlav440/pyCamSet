"""
A PuzzleBoard key decoded at two places in one image is dropped, not fed to the solve.

The upstream detector can report one code position at more than one image
location in a single image (measured: 18 repeated ids in one synthetic scene of
7415 points; 8 keys placed 100-250 px apart in one live pcube image).  At most
one of the copies is right and nothing says which, so every copy is dropped.
These tests hand each PuzzleBoard target a constructed detector answer with a
repeated position and check what comes out.
"""
import logging

import numpy as np
import pytest

from pyCamSet.calibration_targets import puzzleboard, puzzleboard_cube, puzzleboard_ico
from pyCamSet.calibration_targets.markers.puzzleboard import drop_duplicate_keys
from pyCamSet.calibration_targets.puzzleboard import _CODE_SIZE, PuzzleBoard
from pyCamSet.calibration_targets.puzzleboard_cube import PuzzleBoardCube, _drop_duplicate_keys
from pyCamSet.calibration_targets.puzzleboard_ico import PuzzleBoardIco


def _fake_detector(monkeypatch, module, positions, coordinates):
    """Make *module*'s detector return these [row, column] positions and coordinates."""
    def detect(image, min_width=4, **_):
        return [np.array(p) for p in positions], [np.array(c, dtype=float) for c in coordinates]
    monkeypatch.setattr(module, "detect_puzzleboard_image", detect)


_IMAGE = np.zeros((64, 64), dtype=np.uint8)


# -- the shared rule -----------------------------------------------------------

def test_every_copy_of_a_duplicated_pair_key_is_dropped_and_the_rest_kept():
    keys = [[4, 3], [4, 5], [4, 3], [1, 0]]
    points = [[161.9, 360.0], [10.0, 20.0], [73.7, 283.7], [5.0, 6.0]]
    out_keys, out_points = drop_duplicate_keys(keys, points)
    assert out_keys == [[4, 5], [1, 0]]
    assert out_points == [[10.0, 20.0], [5.0, 6.0]]


def test_a_key_found_three_times_loses_all_three():
    keys = np.array([7, 7, 2, 7], dtype=np.int64)
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    out_keys, out_points = drop_duplicate_keys(keys, points)
    assert isinstance(out_keys, np.ndarray)
    assert out_keys.tolist() == [2]
    assert out_points.tolist() == [[2.0, 2.0]]


def test_unique_keys_pass_through_unchanged():
    keys, points = [[0, 1], [0, 2]], [[1.0, 1.0], [2.0, 2.0]]
    assert drop_duplicate_keys(keys, points) == (keys, points)
    assert drop_duplicate_keys([], []) == ([], [])
    assert _drop_duplicate_keys(keys, points) == (keys, points)


def test_a_drop_is_logged_with_its_counts(caplog):
    with caplog.at_level(logging.INFO):
        drop_duplicate_keys([5, 5, 6, 8, 8], [[0, 0]] * 5, "PuzzleBoard")
    message = " ".join(record.getMessage() for record in caplog.records)
    assert "dropped 4 point(s) carrying 2 id(s)" in message
    assert "[5, 8]" in message


# -- each target ---------------------------------------------------------------

def test_planar_puzzleboard_drops_both_copies_of_a_repeated_position(monkeypatch):
    target = PuzzleBoard(num_squares_x=10, num_squares_y=10)
    positions = [[2, 3], [2, 4], [2, 3], [5, 5]]
    coordinates = [[10.0, 20.0], [11.0, 21.0], [40.0, 50.0], [30.0, 30.0]]
    _fake_detector(monkeypatch, puzzleboard, positions, coordinates)
    found = target.find_in_image(_IMAGE)
    assert found.keys.tolist() == [2 * _CODE_SIZE + 4, 5 * _CODE_SIZE + 5]
    assert found.image_points.tolist() == [[21.0, 11.0], [30.0, 30.0]]


def test_planar_puzzleboard_with_only_duplicates_finds_nothing(monkeypatch):
    target = PuzzleBoard(num_squares_x=10, num_squares_y=10)
    _fake_detector(monkeypatch, puzzleboard, [[1, 1], [1, 1]], [[1.0, 1.0], [9.0, 9.0]])
    assert not target.find_in_image(_IMAGE).has_data


def test_puzzleboard_cube_drops_both_copies_of_a_repeated_position(monkeypatch):
    target = PuzzleBoardCube(n_points=10)
    origin_x, origin_y = target.face_origins[2]
    positions = [[origin_y + 1, origin_x + 1], [origin_y + 1, origin_x + 2],
                 [origin_y + 1, origin_x + 1]]
    coordinates = [[10.0, 20.0], [11.0, 21.0], [40.0, 50.0]]
    _fake_detector(monkeypatch, puzzleboard_cube, positions, coordinates)
    found = target.find_in_image(_IMAGE)
    assert found.keys.tolist() == [[2, 1 * 10 + 2]]
    assert found.image_points.tolist() == [[21.0, 11.0]]


def test_puzzleboard_ico_drops_both_copies_of_a_repeated_position(monkeypatch):
    target = PuzzleBoardIco()
    lookup = target._live_corner_lookup()
    size = target.n_points
    # Two printed corners of face 3; the detector's position is one square
    # down and left of the lattice corner (see find_in_image).
    usable = sorted(corner for corner in lookup
                    if 0 <= corner[0] - 1 < size and 0 <= corner[1] - 1 < size)
    (col_a, row_a), (col_b, row_b) = usable[0], usable[1]
    origin_x, origin_y = target.face_origins[3]
    pos_a = [origin_y + row_a - 1, origin_x + col_a - 1]
    pos_b = [origin_y + row_b - 1, origin_x + col_b - 1]
    _fake_detector(monkeypatch, puzzleboard_ico, [pos_a, pos_b, pos_a],
                   [[10.0, 20.0], [11.0, 21.0], [40.0, 50.0]])
    found = target.find_in_image(_IMAGE)
    assert found.keys.tolist() == [[3, lookup[(col_b, row_b)]]]
    assert found.image_points.tolist() == [[21.0, 11.0]]


@pytest.mark.parametrize("make, module", [
    (lambda: PuzzleBoard(num_squares_x=10, num_squares_y=10), puzzleboard),
    (lambda: PuzzleBoardCube(n_points=10), puzzleboard_cube),
])
def test_a_clean_answer_is_untouched(monkeypatch, make, module):
    target = make()
    origin_x, origin_y = getattr(target, "face_origins", [(0, 0)])[0]
    positions = [[origin_y + 1, origin_x + 1], [origin_y + 1, origin_x + 2]]
    _fake_detector(monkeypatch, module, positions, [[1.0, 2.0], [3.0, 4.0]])
    found = target.find_in_image(_IMAGE)
    assert len(found.keys) == 2
    assert found.image_points.tolist() == [[2.0, 1.0], [4.0, 3.0]]
