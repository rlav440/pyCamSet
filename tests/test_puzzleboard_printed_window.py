"""
A planar PuzzleBoard keeps only keys inside its printed window.

Every one of the 501 x 501 code positions has an object point, so a key the
detector misdecodes (a rotated or shifted reading of the code: measured as a
whole board shifted by 167 rows, and patches relabelled by 90/180/270 degrees)
would otherwise reach the solve as a wrong correspondence.  The cube and the
icosahedron already drop positions outside their face windows; these tests
hand the planar board a constructed detector answer and check what comes out,
and that a detection cache written before the key revision is a miss.
"""
import numpy as np
import pytest

from pyCamSet.calibration_targets import puzzleboard
from pyCamSet.calibration_targets.puzzleboard import _CODE_SIZE, PuzzleBoard

_IMAGE = np.zeros((64, 64), dtype=np.uint8)


def _fake_detector(monkeypatch, positions, coordinates):
    def detect(image, min_width=4, **_):
        return [np.array(p) for p in positions], [np.array(c, dtype=float) for c in coordinates]
    monkeypatch.setattr(puzzleboard, "detect_puzzleboard_image", detect)


def test_keys_outside_the_printed_window_are_dropped(monkeypatch):
    target = PuzzleBoard(num_squares_x=22, num_squares_y=15, start_x=40, start_y=100)
    positions = [
        [100, 40],    # first printed corner
        [114, 61],    # last printed corner
        [99, 45],     # one row above the window
        [115, 45],    # one row below it
        [105, 39],    # one column left of it
        [105, 62],    # one column right of it
        [267, 45],    # the measured 167-row shift
        [0, 0],       # the code origin, a valid key of some other board
    ]
    coordinates = [[float(i), float(i + 1)] for i in range(len(positions))]
    _fake_detector(monkeypatch, positions, coordinates)
    found = target.find_in_image(_IMAGE)
    assert found.keys.tolist() == [100 * _CODE_SIZE + 40, 114 * _CODE_SIZE + 61]
    assert found.image_points.tolist() == [[1.0, 0.0], [2.0, 1.0]]  # [row, column] -> [x, y]


def test_a_detection_entirely_outside_the_window_finds_nothing(monkeypatch):
    target = PuzzleBoard(num_squares_x=10, num_squares_y=10)
    _fake_detector(monkeypatch, [[167, 2], [180, 3]], [[1.0, 1.0], [2.0, 2.0]])
    found = target.find_in_image(_IMAGE)
    assert len(found.keys) == 0


def test_the_window_ends_at_the_code_edge(monkeypatch):
    target = PuzzleBoard(num_squares_x=22, num_squares_y=15, start_x=479, start_y=486)
    _fake_detector(monkeypatch, [[500, 500], [486, 479], [485, 479]],
                   [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    found = target.find_in_image(_IMAGE)
    assert found.keys.tolist() == [500 * _CODE_SIZE + 500, 486 * _CODE_SIZE + 479]


def _puzzleboard_cache(tmp_path, spec, revision=None):
    """A cache written for *spec*, as if by a build whose PuzzleBoard key
    revision was *revision* (None: this build's)."""
    import pyCamSet.calibration.detection_cache as cache_module
    from pyCamSet.calibration_targets import TargetDetection
    from pyCamSet.calibration_targets.core.target_registry import build_target
    target = build_target(spec)
    path = tmp_path / "detected_datapoints.npz"
    with pytest.MonkeyPatch.context() as patch:
        if revision is not None:
            patch.setattr(cache_module, "PUZZLEBOARD_KEY_REVISION", revision)
        cache_module.save_to_cache(
            TargetDetection(cam_names=["cam0"], data=np.array([[0, 0, 0, 1.0, 2.0]])),
            [(8, 12)], path, target, ["cam0"], None)
    return target, path


@pytest.mark.parametrize("type_", ["PuzzleBoard", "PuzzleBoardCube", "PuzzleBoardIco"])
def test_a_puzzleboard_cache_from_an_earlier_key_revision_is_a_miss(tmp_path, type_):
    from pyCamSet.calibration.detection_cache import cache_matches
    from pyCamSet.calibration_targets.markers.puzzleboard import PUZZLEBOARD_KEY_REVISION
    target, path = _puzzleboard_cache(tmp_path, {"type": type_}, revision=PUZZLEBOARD_KEY_REVISION - 1)
    assert cache_matches(path, target, ["cam0"], None) is False
    target, path = _puzzleboard_cache(tmp_path, {"type": type_})
    assert cache_matches(path, target, ["cam0"], None) is True


@pytest.mark.parametrize("spec", [
    {"type": "ChArUco", "marker_backend": "aruco1"},
    {"type": "ChArUco2"},
], ids=["ChArUco-aruco1", "ChArUco2"])
def test_other_targets_caches_do_not_depend_on_the_key_revision(tmp_path, spec):
    from pyCamSet.calibration.detection_cache import cache_matches
    from pyCamSet.calibration_targets.markers.puzzleboard import PUZZLEBOARD_KEY_REVISION
    target, path = _puzzleboard_cache(tmp_path, spec, revision=PUZZLEBOARD_KEY_REVISION + 7)
    assert cache_matches(path, target, ["cam0"], None) is True
