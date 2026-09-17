"""ChArUco2/Ccube2 detection at the image edge, and the boards aruco2 cannot read.

``aruco2.detect_grid_board`` pushes a black marker's corners a few pixels
outwards before refining the board's corners with ``cv2.cornerSubPix``, which
asserts every starting point is inside the image -- so a board cut by the
image edge made it raise, and one such frame aborted a whole folder.
:func:`~pyCamSet.calibration_targets.markers.aruco2_gridboard
.detect_grid_board_corners` now pads the image first. The crops below are
ones the unpadded call raised on with aruco2 0.1.1.

Gates itself on the optional ``aruco2`` package, as
``test_charuco2_target.py`` does.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

aruco2 = pytest.importorskip("aruco2")

from pyCamSet.calibration_targets.charuco2 import layout
from pyCamSet.calibration_targets.charuco2.target import ChArUco2
from pyCamSet.calibration_targets.ccube2.target import Ccube2
from pyCamSet.calibration_targets.markers import aruco2_gridboard
from pyCamSet.calibration_targets.markers.aruco2_gridboard import (
    detect_grid_board_corners,
    rotation_ambiguous_marker_ids,
)

_GRIDBOARD_LOGGER = aruco2_gridboard.__name__


def _crop(image: np.ndarray, side: str, cut: int) -> tuple[np.ndarray, np.ndarray]:
    """``image`` with ``cut`` pixels taken off one side, and the (x, y) shift
    that takes a pixel coordinate in ``image`` to one in the crop."""
    h, w = image.shape[:2]
    cropped, shift = {
        "left": (image[:, cut:], (-cut, 0)),
        "top": (image[cut:, :], (0, -cut)),
        "right": (image[:, :w - cut], (0, 0)),
        "bottom": (image[:h - cut, :], (0, 0)),
    }[side]
    return np.ascontiguousarray(cropped), np.asarray(shift, dtype=np.float64)


def _in_from_edge(points: np.ndarray, shape) -> np.ndarray:
    """How far each (x, y) point lies inside an image of ``shape``."""
    h, w = shape[:2]
    return np.minimum.reduce(
        [points[:, 0], points[:, 1], w - 1 - points[:, 0], h - 1 - points[:, 1]])


def _board_pixels(target: ChArUco2, px_per_mm: float, border_mm: float) -> np.ndarray:
    """Where each corner of ``target._render(dpi, border_mm)`` is, in the
    pixel-*corner* convention ``detect_grid_board_corners`` returns (see its
    "Pixel convention" docstring note): a corner on the boundary between
    pixels n-1 and n is at exactly n, matching pyCamSet's OpenCV-backed
    ChArUco/Ccube. (0.5 px from aruco2's own raw pixel-*centre* convention,
    where that same corner is at n - 0.5.)"""
    band_mm = target.square_size * 1000.0 / 4
    corners_mm = target.point_data.reshape(-1, 3)[:, :2] * 1000.0
    return (corners_mm + band_mm + border_mm) * px_per_mm


# -- a board cut by the image edge ----------------------------------------------


@pytest.mark.parametrize("dpi, side, cut", [
    (150, "left", 45), (150, "top", 45), (100, "right", 30), (100, "top", 30),
])
def test_a_board_cut_by_the_image_edge_is_read_where_it_is(dpi, side, cut) -> None:
    """The corners still in the picture are returned, on their own
    positions, and none so close to the edge that its refinement window left
    the image."""
    target = ChArUco2(num_squares_x=6, num_squares_y=6, square_size=10.0)
    image, px_per_mm = target._render(dpi, border_width=5.0)
    cropped, shift = _crop(image, side, cut)
    expected = _board_pixels(target, px_per_mm, 5.0) + shift

    detection = target.find_in_image(cropped)

    assert detection.has_data
    keys = np.asarray(detection.keys).reshape(-1)
    points = np.asarray(detection.image_points).reshape(-1, 2)
    assert np.linalg.norm(points - expected[keys], axis=1).max() < 1.0
    assert _in_from_edge(points, cropped.shape).min() >= aruco2_gridboard._edge_margin()
    # Every corner a full refinement window inside the picture is found.
    well_inside = _in_from_edge(expected, cropped.shape) >= aruco2_gridboard._edge_margin() + 2
    assert set(np.flatnonzero(well_inside).tolist()) <= set(keys.tolist())


@pytest.mark.parametrize("side, cut", [("left", 48), ("top", 48)])
def test_a_ccube2_face_cut_by_the_image_edge_is_read_where_it_is(side, cut) -> None:
    cube = Ccube2(length=40.0, n_points=5)
    cropped, shift = _crop(cube.textures[0], side, cut)
    lattice = layout.grid_board_corners(
        cube.grid_size, cube.square_size, origin=(cube.margin, cube.margin))
    expected = lattice * (cube.draw_res[0] / cube.length) - 0.5 + shift

    detection = cube.find_in_image(cropped)

    assert detection.has_data
    assert set(detection.keys[:, 0].tolist()) == {0}
    error = np.linalg.norm(detection.image_points - expected[detection.keys[:, 1]], axis=1)
    assert error.max() < 1.0
    well_inside = _in_from_edge(expected, cropped.shape) >= aruco2_gridboard._edge_margin() + 2
    assert set(np.flatnonzero(well_inside).tolist()) <= set(detection.keys[:, 1].tolist())


def test_padding_leaves_a_full_frame_detection_where_it_was(monkeypatch) -> None:
    """aruco2 is handed a padded image, and the corners come back in the
    caller's pixels: the same, to float precision, as aruco2 finds unpadded
    (once the +0.5 pixel-convention shift -- applied regardless of padding --
    is accounted for). Called with refine/validate off, so this isolates
    padding's own effect from refinement's."""
    target = ChArUco2(num_squares_x=5, num_squares_y=5, square_size=10.0)
    image, _ = target._render(300, border_width=10.0)

    found, board = aruco2.detect_grid_board(image, target.grid_size, target._aruco_dict_int)
    assert found
    obj_points, unpadded = aruco2.get_solve_pnp_points(board, marker_size=1.0)
    obj_points = np.asarray(obj_points).reshape(-1, 3)
    unpadded_by_id = dict(zip(
        (np.round(obj_points[:, 1]) * 6 + np.round(obj_points[:, 0])).astype(int).tolist(),
        np.asarray(unpadded, dtype=np.float64).reshape(-1, 2)))

    shapes = []
    real_detect = aruco2.detect_grid_board

    def spy(img, *args):
        shapes.append(np.asarray(img).shape)
        return real_detect(img, *args)

    monkeypatch.setattr(aruco2_gridboard.aruco2, "detect_grid_board", spy)
    ids, points = detect_grid_board_corners(
        image, target.grid_size, target._aruco_dict_int, target.square_size,
        refine=False, validate=False)

    pad = aruco2_gridboard._grid_board_padding(image.shape[1])
    assert pad > 0
    assert shapes == [(image.shape[0] + 2 * pad, image.shape[1] + 2 * pad)]
    assert sorted(ids.tolist()) == sorted(unpadded_by_id)
    for gid, point in zip(ids.tolist(), points):
        # float32 corners carried 'pad' pixels further out: a few 1e-4 px,
        # plus the deliberate +0.5 pixel-convention shift.
        assert np.abs(point - (unpadded_by_id[gid] + 0.5)).max() < 1e-3


@pytest.mark.parametrize("width", [640, 1499, 2448, 4000, 8192])
def test_the_padding_covers_aruco2s_corner_push_and_refinement_window(width) -> None:
    """At least the half window plus aruco2's largest push-out for the
    padded width, plus a pixel."""
    pad = aruco2_gridboard._grid_board_padding(width)
    passes = max(2, int(2.0 * (width + 2 * pad) / 2000.0 + 0.5))
    assert pad >= 9 + (passes + 1) + 1


def test_an_opencv_failure_inside_aruco2_loses_the_board_not_the_image(monkeypatch, caplog) -> None:
    target = ChArUco2(num_squares_x=5, num_squares_y=5, square_size=10.0)
    image, _ = target._render(150)

    def raising(*args):
        raise ValueError(
            "OpenCV(4.11.0) cornersubpix.cpp:99: error: (-215:Assertion failed) "
            "Rect(0, 0, src.cols, src.rows).contains(cT) in function 'cv::cornerSubPix'\n")

    monkeypatch.setattr(aruco2_gridboard.aruco2, "detect_grid_board", raising)
    with caplog.at_level(logging.WARNING, logger=_GRIDBOARD_LOGGER):
        detection = target.find_in_image(image)
    assert not detection.has_data
    assert any("cornerSubPix" in r.getMessage() and "5x5" in r.getMessage()
               for r in caplog.records)


def test_is_cornersubpix_edge_failure_matches_the_exact_assertion() -> None:
    """The one assertion this exists to recognise: both 'cornerSubPix' and
    'contains(cT)' present, either case."""
    assert aruco2_gridboard._is_cornersubpix_edge_failure(ValueError(
        "OpenCV(4.11.0) .../cornersubpix.cpp:99: error: (-215:Assertion "
        "failed) Rect(0, 0, src.cols, src.rows).contains(cT) in function "
        "'cv::cornerSubPix'"))
    assert aruco2_gridboard._is_cornersubpix_edge_failure(ValueError(
        "opencv(4.11.0) cornersubpix.cpp:99: error CONTAINS(CT) CV::CORNERSUBPIX"))


def test_is_cornersubpix_edge_failure_rejects_a_different_cornersubpix_assertion() -> None:
    """It used to match on 'cornersubpix' OR 'contains(ct)' alone, so ANY
    cornerSubPix-related OpenCV assertion -- not just the specific
    starting-point-outside-the-image one -- was misread as the recoverable
    edge case. A different cornerSubPix assertion (a bad window size, say)
    must not match: both substrings are required, not either one."""
    assert not aruco2_gridboard._is_cornersubpix_edge_failure(ValueError(
        "OpenCV(4.11.0) .../cornersubpix.cpp:65: error: (-215:Assertion "
        "failed) win.width > 0 && win.height > 0 in function "
        "'cv::cornerSubPix'"))
    # And the other substring alone -- some other function's assertion that
    # merely happens to mention "contains(cT)" -- must not match either.
    assert not aruco2_gridboard._is_cornersubpix_edge_failure(ValueError(
        "OpenCV(4.11.0) .../rect.cpp:12: error: (-215:Assertion failed) "
        "contains(cT) in function 'cv::someUnrelatedCheck'"))


def test_a_different_cornersubpix_assertion_is_not_swallowed(monkeypatch) -> None:
    """End-to-end: a cornerSubPix assertion that is not the specific
    starting-point-outside-the-image one must propagate, not be read as
    'this board is lost'."""
    target = ChArUco2(num_squares_x=5, num_squares_y=5, square_size=10.0)
    image, _ = target._render(150)

    def raising(*args):
        raise ValueError(
            "OpenCV(4.11.0) .../cornersubpix.cpp:65: error: (-215:Assertion "
            "failed) win.width > 0 && win.height > 0 in function "
            "'cv::cornerSubPix'")

    monkeypatch.setattr(aruco2_gridboard.aruco2, "detect_grid_board", raising)
    with pytest.raises(ValueError, match="win.width"):
        target.find_in_image(image)


def test_a_valueerror_that_is_not_the_cornersubpix_assertion_is_not_swallowed(monkeypatch) -> None:
    """The except clause used to catch every ValueError from
    aruco2.detect_grid_board, not just the cornerSubPix edge-of-image
    assertion it exists for -- so a different ValueError (a real bug, a
    malformed board) was silently read as 'no detection'. It must now
    propagate."""
    target = ChArUco2(num_squares_x=5, num_squares_y=5, square_size=10.0)
    image, _ = target._render(150)

    def raising(*args):
        raise ValueError("board_ids and grid_size disagree on square count")

    monkeypatch.setattr(aruco2_gridboard.aruco2, "detect_grid_board", raising)
    with pytest.raises(ValueError, match="disagree on square count"):
        target.find_in_image(image)


def test_an_error_that_is_not_aruco2s_is_not_swallowed(monkeypatch) -> None:
    target = ChArUco2(num_squares_x=5, num_squares_y=5, square_size=10.0)
    image, _ = target._render(150)

    def raising(*args):
        raise RuntimeError("not an OpenCV assertion")

    monkeypatch.setattr(aruco2_gridboard.aruco2, "detect_grid_board", raising)
    with pytest.raises(RuntimeError):
        target.find_in_image(image)


# -- boards aruco2 cannot read --------------------------------------------------


def test_the_rotation_ambiguous_markers_are_computed_from_the_bits() -> None:
    dict_int = int(aruco2.DICT_ARUCO_ORIGINAL)
    assert rotation_ambiguous_marker_ids(dict_int, range(1024)) == [1023]
    bits = np.asarray(aruco2.get_predefined_dictionary(dict_int).get_marker_bits(1023))
    assert np.array_equal(bits, np.rot90(bits, 2))
    assert rotation_ambiguous_marker_ids(int(aruco2.DICT_4X4_1000), range(1000)) == []


def test_a_charuco2_carrying_a_rotation_ambiguous_marker_is_refused() -> None:
    with pytest.raises(ValueError, match=r"\[1023\].*half turn"):
        ChArUco2(num_squares_x=32, num_squares_y=32, a_dict="DICT_ARUCO_ORIGINAL")
    # One square fewer stops short of marker 1023.
    ChArUco2(num_squares_x=31, num_squares_y=33, a_dict="DICT_ARUCO_ORIGINAL")


def test_a_ccube2_face_carrying_a_rotation_ambiguous_marker_is_refused(monkeypatch) -> None:
    """No real dictionary gives a cube one -- six faces cannot reach marker
    1023 -- so which markers are ambiguous is stood in for, and the cube
    must ask rather than assume."""
    monkeypatch.setattr(
        aruco2_gridboard, "rotation_ambiguous_marker_ids",
        lambda dict_int, ids: [i for i in ids if i == 30])
    with pytest.raises(ValueError, match=r"Face 1 .*\[30\].*half turn"):
        Ccube2(n_points=5, draw_res=(100, 100))


def test_a_large_4x4_charuco2_is_warned_about_aruco2s_false_marker(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=_GRIDBOARD_LOGGER):
        ChArUco2(num_squares_x=27, num_squares_y=27)
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 1
    assert "688" in warned[0] and "17" in warned[0] and "27x27" in warned[0]


@pytest.mark.parametrize("values", [
    dict(num_squares_x=26, num_squares_y=26),
    dict(num_squares_x=27, num_squares_y=27, a_dict="DICT_5X5_1000"),
])
def test_a_charuco2_without_the_false_marker_pair_is_not_warned_about(values, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=_GRIDBOARD_LOGGER):
        ChArUco2(**values)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
