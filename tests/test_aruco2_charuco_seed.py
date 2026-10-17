"""aruco2 on classic ChArUco boards: the marker quads detectBoard is seeded with.

aruco2 refines every marker corner with one image-wide cornerSubPix window of
at least 3 px. On small squares that window reaches the adjacent chessboard
corner and drags the marker corner towards it, and detectBoard (a 1 px
half-window at those sizes) returns the ChArUco corner where the biased seed
put it. ``markers.aruco2._seed_quads`` corrects the quads before detectBoard.

The board here is rendered sharp: a 10x oversampled ``generateImage``,
rotated, then area-averaged down, so every chessboard corner's true position
is known exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("aruco2")

import cv2  # noqa: E402

from pyCamSet.calibration.detection_cache import (  # noqa: E402
    cache_matches,
    save_to_cache,
)
from pyCamSet.calibration_targets import TargetDetection  # noqa: E402
from pyCamSet.calibration_targets.ccube import Ccube  # noqa: E402
from pyCamSet.calibration_targets.charuco import ChArUco  # noqa: E402
from pyCamSet.calibration_targets.cico import CIco  # noqa: E402
from pyCamSet.calibration_targets.core.target_registry import build_target  # noqa: E402
from pyCamSet.calibration_targets.markers import aruco2 as a2  # noqa: E402
from pyCamSet.calibration_targets.markers.aruco_opencv import charuco_corners_to_pixel_centre  # noqa: E402

#: Oversampling of the rendered board before it is area-averaged down.
_OVERSAMPLE = 10

#: (square size in px, rotation in degrees): small squares, where aruco2's
#: 3 px window reaches the chessboard corner.
_VIEWS = [(16, 7.0), (16, 23.0), (18, 7.0), (18, 23.0), (20, 7.0), (20, 23.0)]

#: A corner further than this from the truth is a false corner.
_FALSE_PX = 2.0


def _target():
    return ChArUco(num_squares_x=9, num_squares_y=6, square_size=10.0,
                   marker_backend="aruco2", legacy=False)


def _render(board, square_px, angle):
    """A sharp, rotated view of *board* and its chessboard corners' truth.

    Truth is in OpenCV's pixel-centre convention (pixel i's centre at i).
    """
    nx, ny = board.getChessboardSize()
    big_square = square_px * _OVERSAMPLE
    pad = 2 * big_square
    width, height = nx * big_square + 2 * pad, ny * big_square + 2 * pad
    canvas = np.full((height, width), 255, np.uint8)
    canvas[pad:pad + ny * big_square, pad:pad + nx * big_square] = board.generateImage(
        (nx * big_square, ny * big_square), marginSize=0, borderBits=1)
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated = cv2.warpAffine(canvas, rotation, (width, height),
                             flags=cv2.INTER_LINEAR, borderValue=255)
    image = cv2.resize(rotated, (width // _OVERSAMPLE, height // _OVERSAMPLE),
                       interpolation=cv2.INTER_AREA)

    # Corner positions: board units -> oversampled pixel edges -> pixel
    # centres, through the rotation, then down to the final image.
    unit = np.asarray(board.getChessboardCorners())[:, :2] / board.getSquareLength()
    big = pad + unit * big_square - 0.5
    big = (rotation[:, :2] @ big.T).T + rotation[:, 2]
    truth = (big + 0.5) / _OVERSAMPLE - 0.5
    return np.ascontiguousarray(image), truth


def _unseeded(image, board, markers):
    """The corners the adapter returned before seeding: aruco2's quads
    straight into the same cached detectBoard."""
    detector = a2._charuco_detector_for(board)
    corners, ids, _, _ = detector.detectBoard(
        image,
        markerCorners=tuple(q.reshape(1, 4, 2) for _mid, q in markers),
        markerIds=np.asarray([mid for mid, _q in markers], dtype=np.int32).reshape(-1, 1))
    if ids is None:
        return np.zeros(0, dtype=int), np.zeros((0, 2))
    # In the pixel-centre frame the adapter returns (OpenCV 4.x is half a pixel off it).
    return np.asarray(ids).reshape(-1), np.asarray(charuco_corners_to_pixel_centre(corners)).reshape(-1, 2)


def _views():
    target = _target()
    for square_px, angle in _VIEWS:
        image, truth = _render(target.board, square_px, angle)
        markers = a2.detect_markers(image, target._aruco_dict_int)
        assert len(markers) >= 20, "rendered board not read by aruco2"
        yield target, image, truth, markers


def test_seeding_removes_most_false_corners_on_small_squares():
    old_false = new_false = 0
    for target, image, truth, markers in _views():
        old_ids, old_pts = _unseeded(image, target.board, markers)
        new_ids, new_pts = a2.interpolate_board_corners(image, target.board, markers)
        old_false += int((np.linalg.norm(old_pts - truth[old_ids], axis=1) > _FALSE_PX).sum())
        new_false += int((np.linalg.norm(new_pts - truth[new_ids], axis=1) > _FALSE_PX).sum())
    # The unseeded reading must show the bias, or this test proves nothing.
    assert old_false >= 30, f"setup no longer reproduces the bias ({old_false} false corners)"
    assert new_false <= old_false // 10, f"{new_false} false corners vs {old_false} unseeded"


def test_seeding_keeps_every_corner_the_unseeded_reading_found():
    for target, image, truth, markers in _views():
        old_ids, _ = _unseeded(image, target.board, markers)
        new_ids, _ = a2.interpolate_board_corners(image, target.board, markers)
        assert set(old_ids.tolist()) <= set(np.asarray(new_ids).tolist())


def test_seeding_keeps_marker_ids_and_order():
    for target, image, _truth, markers in _views():
        seeded = a2._seed_quads(image, target.board, markers)
        assert [mid for mid, _q in seeded] == [mid for mid, _q in markers]
        for _mid, quad in seeded:
            assert quad.shape == (4, 2) and quad.dtype == np.float32


def test_seeding_is_deterministic():
    for target, image, _truth, markers in _views():
        first = a2.interpolate_board_corners(image, target.board, markers)
        again = a2.interpolate_board_corners(image, target.board, markers)
        assert np.array_equal(first[0], again[0])
        assert np.array_equal(first[1], again[1])


def test_seeding_reads_a_colour_image_as_its_grey():
    target, image, _truth, markers = next(_views())
    colour = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    grey_read = a2._seed_quads(image, target.board, markers)
    colour_read = a2._seed_quads(colour, target.board, markers)
    for (_m1, q1), (_m2, q2) in zip(grey_read, colour_read):
        assert np.array_equal(q1, q2)


@pytest.mark.parametrize("method", [cv2.aruco.CORNER_REFINE_SUBPIX, cv2.aruco.CORNER_REFINE_APRILTAG],
                         ids=["subpix", "apriltag"])
def test_opencv_seed_quads_are_never_corner_refined(monkeypatch, method):
    """The OpenCV quads are contour corners whatever refinement a caller
    configured: SUBPIX would bring back a window bias, and APRILTAG returns
    corners in AprilTag's pixel convention (about +0.45 px in x and y).
    Simulated here by making every new DetectorParameters ask for *method*."""
    target, image, _truth, markers = next(_views())
    expected = a2._seed_quads(image, target.board, markers)

    make_parameters = cv2.aruco.DetectorParameters
    used = []

    def preset():
        params = make_parameters()
        params.cornerRefinementMethod = method
        return params

    make_detector = cv2.aruco.ArucoDetector

    def spy(dictionary, params, *args):
        used.append(params.cornerRefinementMethod)
        return make_detector(dictionary, params, *args)

    monkeypatch.setattr(cv2.aruco, "DetectorParameters", preset)
    monkeypatch.setattr(cv2.aruco, "ArucoDetector", spy)
    seeded = a2._seed_quads(image, target.board, markers)
    assert used and set(used) == {cv2.aruco.CORNER_REFINE_NONE}
    for (_m1, q1), (_m2, q2) in zip(expected, seeded):
        assert np.array_equal(q1, q2)


@pytest.mark.parametrize("method", [cv2.aruco.CORNER_REFINE_SUBPIX, cv2.aruco.CORNER_REFINE_APRILTAG],
                         ids=["subpix", "apriltag"])
def test_a_targets_corner_refinement_setting_does_not_change_its_aruco2_reading(method):
    plain = _target()
    configured = ChArUco(num_squares_x=9, num_squares_y=6, square_size=10.0, marker_backend="aruco2",
                         legacy=False, detection_options={"cornerRefinementMethod": method})
    image, _truth = _render(plain.board, 18, 7.0)
    first, second = plain.find_in_image(image), configured.find_in_image(image)
    assert np.array_equal(np.asarray(first.keys), np.asarray(second.keys))
    assert np.array_equal(np.asarray(first.image_points), np.asarray(second.image_points))


def test_a_far_away_opencv_reading_of_the_same_id_is_not_used():
    """OpenCV's quad replaces aruco2's only when both read the same marker:
    a quad displaced by a whole side, though within the window's reach,
    never takes OpenCV's quad of the real marker."""
    target, image, _truth, markers = next(_views())
    mid, quad = markers[0]
    side = a2._mean_side(quad)
    moved = [(mid, quad + np.float32(side))] + markers[1:]
    opencv = a2._opencv_quads(image, target.board.getDictionary(), [quad])
    assert mid in opencv, "setup: OpenCV no longer reads the marker"
    seeded = a2._seed_quads(image, target.board, moved)
    assert np.max(np.linalg.norm(seeded[0][1] - opencv[mid], axis=1)) > 0.5 * side


def test_markers_out_of_the_windows_reach_are_left_exactly_as_aruco2_read_them(monkeypatch):
    """Large squares: no marker corner is within reach of aruco2's window, so
    nothing is swapped or refitted, and OpenCV's marker detection never runs."""
    target = _target()
    image = np.ascontiguousarray(target.board.generateImage((9 * 80, 6 * 80)), dtype=np.uint8)
    markers = a2.detect_markers(image, target._aruco_dict_int)
    assert len(markers) >= 20
    calls = []
    make_detector = cv2.aruco.ArucoDetector

    def spy(*args):
        calls.append(args)
        return make_detector(*args)

    monkeypatch.setattr(cv2.aruco, "ArucoDetector", spy)
    seeded = a2._seed_quads(image, target.board, markers)
    assert calls == []
    for (m1, q1), (m2, q2) in zip(markers, seeded):
        assert m1 == m2 and np.array_equal(q1, q2)


@pytest.mark.parametrize("make", [
    lambda: Ccube(n_points=4, length=20.0, marker_backend="aruco2"),
    lambda: CIco(length=100.0, n_points=10, marker_backend="aruco2"),
], ids=["Ccube", "CIco"])
def test_cube_targets_seed_each_face_through_the_adapter(monkeypatch, make):
    target = make()
    face = 1
    image = np.ascontiguousarray(
        target.boards[face].generateImage((900, 900)), dtype=np.uint8)
    seen = []
    real = a2._seed_quads

    def spy(image, board, markers):
        seen.append(board)
        return real(image, board, markers)

    monkeypatch.setattr(a2, "_seed_quads", spy)
    detection = target.find_in_image(image)
    assert any(board is target.boards[face] for board in seen)
    assert detection.keys is not None and len(detection.keys) > 0


def _cache(tmp_path, spec, revision=None):
    """A cache written for *spec*, as if by a build whose seeding revision
    was *revision* (None: this build's)."""
    import pyCamSet.calibration.detection_cache as cache_module
    target = build_target(spec)
    path = tmp_path / "detected_datapoints.npz"
    with pytest.MonkeyPatch.context() as patch:
        if revision is not None:
            patch.setattr(cache_module, "CHARUCO_SEED_REVISION", revision)
        save_to_cache(TargetDetection(cam_names=["cam0"], data=np.array([[0, 0, 0, 1.0, 2.0]])),
                      [(8, 12)], path, target, ["cam0"], None)
    return target, path


@pytest.mark.parametrize("type_", ["ChArUco", "Ccube", "CIco"])
def test_an_aruco2_cache_from_an_earlier_seeding_is_a_miss(tmp_path, type_):
    spec = {"type": type_, "marker_backend": "aruco2"}
    target, path = _cache(tmp_path, spec, revision=a2.CHARUCO_SEED_REVISION - 1)
    assert cache_matches(path, target, ["cam0"], None) is False
    target, path = _cache(tmp_path, spec)
    assert cache_matches(path, target, ["cam0"], None) is True


@pytest.mark.parametrize("spec", [
    {"type": "ChArUco", "marker_backend": "aruco1"},
    {"type": "Ccube", "marker_backend": "aruco1"},
    {"type": "ChArUco2"},
], ids=["ChArUco-aruco1", "Ccube-aruco1", "ChArUco2"])
def test_other_detectors_caches_do_not_depend_on_the_seeding(tmp_path, spec):
    target, path = _cache(tmp_path, spec, revision=a2.CHARUCO_SEED_REVISION + 7)
    assert cache_matches(path, target, ["cam0"], None) is True
