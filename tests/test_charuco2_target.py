"""ChArUco2 (aruco2's GridBoard) target regressions.

Gates itself on the optional ``aruco2`` package, matching
``test_aruco2_backend.py``'s own pattern (see that module and
``tests/conftest.py``'s "Optional-environment probes" note): a module whose
only prerequisite is an optional package skips itself with
``pytest.importorskip`` before its other imports, rather than being dropped
from collection, so a missing dependency shows up as one reported skip
instead of thirty tests silently never running.

The board is printed from a vector layout (``charuco2/layout.py``), not from
aruco2's raster, so the layout is checked here against aruco2's own
``get_grid_board_image`` pixel for pixel. That comparison is exact only at
scales where every cell and band edge lands on a whole pixel: aruco2 draws a
square ``marker_bits * bit_size`` pixels wide and its band ``// 4`` of that,
so the square's pixel width must divide by both ``marker_bits + 2`` and 4
(bit sizes 3, 28, 4 and 36 for the 4x4, 5x5, 6x6 and 7x7 families).
"""

from __future__ import annotations

import io
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

aruco2 = pytest.importorskip("aruco2")

from pyCamSet.calibration_targets.markers import gridboard_layout as layout
from pyCamSet.calibration_targets.charuco2 import ChArUco2
from pyCamSet.calibration_targets.core.abstract_target import EXPORT_KINDS
from pyCamSet.calibration_targets.markers.aruco2_gridboard import (
    detect_grid_board_corners,
    dictionary_marker_bits,
    grid_board_marker_bits,
    render_grid_board_image,
)

#: (dictionary, the smallest bit size at which aruco2's raster has every
#: edge on a whole pixel) -- see the module docstring.
_COMPATIBLE_BIT_SIZES = [
    ("DICT_4X4_1000", 3),
    ("DICT_5X5_1000", 28),
    ("DICT_6X6_1000", 4),
    ("DICT_7X7_1000", 36),
    ("DICT_ARUCO_MIP_36h12", 4),
]

#: Odd and even widths and heights, square and not.
_BOARD_SHAPES = [(5, 5), (4, 3), (3, 6), (2, 2), (6, 4)]


def _skip_without_cairo():
    """Skip when the native cairo library is missing, as it is on test CI.

    cairosvg raises OSError, not ImportError, when the library it binds to is
    absent, so importorskip does not catch it.
    """
    try:
        import pyCamSet.utils.cairo_dll_helper  # noqa: F401
        import cairosvg  # noqa: F401
    except (ImportError, OSError) as err:
        pytest.skip(f"native cairo is unavailable: {err}")


def _corner_positions_mm(target: ChArUco2) -> np.ndarray:
    """``point_data`` as (n, 2) millimetres, flattened out of its face axis."""
    return target.point_data.reshape(-1, 3)[:, :2] * 1000.0


def _expected_pixels(target: ChArUco2, px_per_mm: float,
                     border_mm: float = 0.0) -> np.ndarray:
    """Where each corner is in a raster whose pixel (0, 0) starts at the
    band's outer corner, less ``border_mm``.

    ``detect_grid_board_corners`` reports image points in the pixel-*corner*
    convention (see its docstring): a corner on the boundary between pixels
    n-1 and n is at exactly n -- 0.5 px from aruco2's own raw pixel-*centre*
    convention, where that same corner is at n - 0.5.
    """
    band_mm = target.square_size * 1000.0 / 4
    return (_corner_positions_mm(target) + band_mm + border_mm) * px_per_mm


def _svg_size_mm(text: str) -> tuple[float, float]:
    w_mm = float(re.search(r'width="([\d.]+)mm"', text).group(1))
    h_mm = float(re.search(r'height="([\d.]+)mm"', text).group(1))
    return w_mm, h_mm


def _rasterise_svg(svg_path: Path, px_per_mm: float) -> np.ndarray:
    import cairosvg
    from PIL import Image

    w_mm, h_mm = _svg_size_mm(svg_path.read_text(encoding="utf-8"))
    png_bytes = cairosvg.svg2png(
        url=str(svg_path),
        output_width=int(round(w_mm * px_per_mm)),
        output_height=int(round(h_mm * px_per_mm)),
        background_color="white")
    return np.array(Image.open(io.BytesIO(png_bytes)).convert("L"))


def test_charuco2_offers_only_the_aruco2_backend() -> None:
    assert list(ChArUco2.DETECTOR_BACKENDS) == ["aruco2"]


def test_charuco2_excludes_apriltag_dictionaries() -> None:
    choices = ChArUco2.construction_parameters().parameter("a_dict").choices
    values = [c.value for c in choices]
    assert values, "ChArUco2 must offer at least one dictionary"
    assert not any(v.startswith("DICT_APRILTAG_") for v in values)


def test_charuco2_construction_builds_a_deterministic_corner_grid() -> None:
    target = ChArUco2(num_squares_x=5, num_squares_y=7, square_size=10.0)
    assert target.point_data.ndim == 3
    assert target.point_data.shape == (1, 6 * 8, 3)

    positions = _corner_positions_mm(target)
    # Corner gid = row * (W+1) + col sits at (col, row) * square_size -- the
    # array index and the corner id are the same thing by construction.
    expected = np.array(
        [[col * 10.0, row * 10.0] for row in range(8) for col in range(6)])
    assert np.allclose(positions, expected)


def test_charuco2_refuses_a_board_its_dictionary_cannot_fill() -> None:
    """One marker per square: a 50-marker dictionary fills 7x7, not 8x7."""
    ChArUco2(num_squares_x=7, num_squares_y=7, a_dict="DICT_4X4_50")
    with pytest.raises(ValueError, match="needs 56 markers.*holds only 50"):
        ChArUco2(num_squares_x=8, num_squares_y=7, a_dict="DICT_4X4_50")


@pytest.mark.parametrize("values,refused", [
    ({"num_squares_x": 5.7, "num_squares_y": 5.0}, "whole number of squares"),
    ({"num_squares_x": 5.0, "num_squares_y": 5.5}, "whole number of squares"),
    ({"num_squares_x": True, "num_squares_y": 5}, "whole number of squares"),
])
def test_a_board_that_is_not_a_charuco2_is_refused(values, refused) -> None:
    """A non-integer num_squares_x/num_squares_y must be refused outright,
    not silently truncated to a smaller board (int(5.7) == 5) -- see
    Ccube2's identical guard on n_points."""
    with pytest.raises(ValueError, match=refused):
        ChArUco2(square_size=10.0, **values)


def test_a_whole_number_of_squares_written_as_a_float_builds() -> None:
    """A spec read back from JSON may carry 5.0 for 5."""
    board = ChArUco2(num_squares_x=5.0, num_squares_y=6.0, square_size=10.0)
    assert board.num_squares_x == 5 and isinstance(board.num_squares_x, int)
    assert board.num_squares_y == 6 and isinstance(board.num_squares_y, int)


# -- the layout against aruco2's own renderer ---------------------------------


@pytest.mark.parametrize("shape", _BOARD_SHAPES)
@pytest.mark.parametrize("dict_name, bit_size", _COMPATIBLE_BIT_SIZES)
@pytest.mark.parametrize("id_set", ["default", "offset", "shuffled"])
def test_layout_raster_is_aruco2s_board_image_pixel_for_pixel(
        shape, dict_name, bit_size, id_set) -> None:
    """The vector layout, rasterised, is exactly what aruco2 draws: cell
    polarity and orientation, inverted squares, band tabs and corner squares
    -- for default ids and for a board with its own id set, as a cube face
    has."""
    dict_int = int(getattr(aruco2, dict_name))
    n = shape[0] * shape[1]
    ids = {
        "default": None,
        "offset": list(range(100, 100 + n)),
        "shuffled": [int(i) for i in
                     np.random.default_rng(n).permutation(250)[:n]],
    }[id_set]

    reference = render_grid_board_image(shape, dict_int, bit_size, ids)

    # A square size that is not the pixel size, so the scale is exercised.
    square_size = 7.0
    marker_bits = dictionary_marker_bits(dict_int)
    px_per_unit = marker_bits * bit_size / square_size
    rects = layout.grid_board_rectangles(
        shape, square_size, grid_board_marker_bits(shape, dict_int, ids))
    x_min, y_min, _, _ = layout.grid_board_bounds(shape, square_size)
    drawn = layout.rasterise_rectangles(
        rects, px_per_unit, top_left=(x_min, y_min), shape=reference.shape)

    assert np.array_equal(drawn, reference)


def test_charuco2_render_is_aruco2s_board_image_at_a_compatible_dpi() -> None:
    """The target's own raster output is the same image: at 1 px/mm a 12 mm
    square is 12 px, bit size 3 for a 4x4 dictionary."""
    target = ChArUco2(num_squares_x=5, num_squares_y=4, square_size=12.0)
    image, px_per_mm = target._render(25.4)

    direct = aruco2.get_grid_board_image((5, 4), target._aruco_dict_int, 3)
    assert px_per_mm == pytest.approx(1.0)
    assert np.array_equal(image, direct)


def test_layout_imports_without_aruco2() -> None:
    """The shared geometry is numpy only, so a cube face (or a docs build)
    can use it with aruco2 blocked."""
    probe = (
        "import sys\n"
        "sys.modules['aruco2'] = None\n"
        "from pyCamSet.calibration_targets.markers import gridboard_layout as layout\n"
        "import numpy as np\n"
        "bits = np.zeros((4, 4, 4), bool)\n"
        "rects = layout.grid_board_rectangles((2, 2), 1.0, bits, origin=(3, 4))\n"
        "assert rects.shape[1] == 4\n"
        "print('OK')\n"
    )
    finished = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[1], timeout=300)
    assert finished.returncode == 0, finished.stderr
    assert "OK" in finished.stdout


def test_layout_origin_moves_every_shape_and_corner() -> None:
    bits = grid_board_marker_bits((3, 2), int(aruco2.DICT_4X4_1000))
    at_zero = layout.grid_board_rectangles((3, 2), 5.0, bits)
    moved = layout.grid_board_rectangles((3, 2), 5.0, bits, origin=(2.0, -1.5))
    assert np.allclose(moved, at_zero + [2.0, -1.5, 2.0, -1.5])

    corners = layout.grid_board_corners((3, 2), 5.0, origin=(2.0, -1.5))
    assert corners.shape == (12, 2)
    assert np.allclose(corners[4], [2.0, 3.5])  # gid 4 = row 1, col 0
    assert layout.grid_board_bounds((3, 2), 5.0, origin=(2.0, -1.5)) == \
        pytest.approx((0.75, -2.75, 18.25, 9.75))


# -- printing -----------------------------------------------------------------


@pytest.mark.parametrize("kind", EXPORT_KINDS)
def test_charuco2_save_printable_all_kinds(tmp_path: Path, kind: str) -> None:
    if kind == "pdf_vector":
        _skip_without_cairo()
    target = ChArUco2(num_squares_x=5, num_squares_y=5, square_size=10.0)
    out = target.save_printable(tmp_path / f"board.{kind}", kind=kind)
    assert out.exists()
    assert out.stat().st_size > 1024
    if kind.startswith("pdf_"):
        # The vector PDF is drawn, never a raster embedded in a page; the
        # raster one is exactly that.
        embeds_raster = re.search(rb"/Subtype\s*/Image", Path(out).read_bytes())
        assert bool(embeds_raster) == (kind == "pdf_raster")


def test_charuco2_svg_is_true_vector(tmp_path: Path) -> None:
    """No embedded raster anywhere: every black shape is one path."""
    target = ChArUco2(num_squares_x=4, num_squares_y=3, square_size=15.0)
    text = target.save_to_svg(tmp_path / "board.svg").read_text(encoding="utf-8")
    assert "<image" not in text
    assert "base64" not in text
    assert text.count("<path") == 1
    assert 'fill-rule="nonzero"' in text


def test_charuco2_svg_is_true_millimetre_scale(tmp_path: Path) -> None:
    """The page is the board, its quarter-square band on each side, and the
    white border -- in mm, whatever dpi is passed."""
    target = ChArUco2(num_squares_x=4, num_squares_y=3, square_size=15.0)
    svg_path = target.save_to_svg(tmp_path / "board.svg", border_width=10.0, dpi=300)
    w_mm, h_mm = _svg_size_mm(svg_path.read_text(encoding="utf-8"))

    assert w_mm == pytest.approx(4 * 15.0 + 2 * 15.0 / 4 + 2 * 10.0, abs=1e-6)
    assert h_mm == pytest.approx(3 * 15.0 + 2 * 15.0 / 4 + 2 * 10.0, abs=1e-6)


def test_charuco2_detects_its_own_rendered_board() -> None:
    """Detect straight from the raster output, no SVG round trip. Every
    corner must be found, at its own recorded position."""
    target = ChArUco2(num_squares_x=5, num_squares_y=5, square_size=10.0)
    image, px_per_mm = target._render(300)

    detection = target.find_in_image(image)
    assert detection.has_data
    n_corners = target.point_data.shape[-2]
    assert len(detection.keys) == n_corners
    assert set(detection.keys.tolist()) == set(range(n_corners))

    expected_px = _expected_pixels(target, px_per_mm)
    by_id = dict(zip(detection.keys.tolist(), detection.image_points))
    for gid, expected in enumerate(expected_px):
        assert np.linalg.norm(by_id[gid] - expected) < 1.0


def test_charuco2_detection_survives_occlusion() -> None:
    """Corners that remain visible after part of the board is blanked out
    must still map to their correct, unchanged 3-D position."""
    target = ChArUco2(num_squares_x=5, num_squares_y=5, square_size=10.0)
    image, _ = target._render(300)
    full = target.find_in_image(image)
    full_by_id = dict(zip(full.keys.tolist(), full.image_points))

    occluded = image.copy()
    occluded[image.shape[0] // 2:, image.shape[1] // 2:] = 255
    detection = target.find_in_image(occluded)

    assert 0 < len(detection.keys) < len(full.keys)
    for gid, pt in zip(detection.keys.tolist(), detection.image_points):
        assert np.linalg.norm(np.asarray(pt) - full_by_id[gid]) < 1.0


def test_charuco2_detection_survives_rotation() -> None:
    """A rotated capture must still recover every corner's true id -- the
    id comes from the detected object-point geometry, not from image-space
    position or detection order."""
    target = ChArUco2(num_squares_x=5, num_squares_y=5, square_size=10.0)
    image, _ = target._render(300)

    rotated = np.ascontiguousarray(np.rot90(image, k=1))
    detection = target.find_in_image(rotated)
    n_corners = target.point_data.shape[-2]
    assert len(detection.keys) == n_corners
    assert set(detection.keys.tolist()) == set(range(n_corners))


def test_charuco2_detection_survives_perspective_warp() -> None:
    import cv2

    target = ChArUco2(num_squares_x=5, num_squares_y=5, square_size=10.0)
    image, _ = target._render(300)
    full = target.find_in_image(image)
    full_by_id = dict(zip(full.keys.tolist(), full.image_points))

    h, w = image.shape
    pad = 150
    canvas = np.full((h + 2 * pad, w + 2 * pad), 255, np.uint8)
    canvas[pad:pad + h, pad:pad + w] = image
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]]) + pad
    dst = np.float32([[60, 30], [w + pad - 15, 8],
                       [w + pad - 45, h + pad - 20], [30, h + pad - 45]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(canvas, matrix, (w + 2 * pad, h + 2 * pad),
                                  borderValue=255)

    detection = target.find_in_image(warped)
    assert len(detection.keys) > 0
    for gid, pt in zip(detection.keys.tolist(), detection.image_points):
        expected = full_by_id[gid] + np.array([pad, pad])
        projected = matrix @ np.array([expected[0], expected[1], 1.0])
        projected = projected[:2] / projected[2]
        assert np.linalg.norm(np.asarray(pt) - projected) < 3.0


def test_charuco2_svg_rasterises_exactly_to_the_layout(tmp_path: Path) -> None:
    """cairo draws the SVG, at a scale where every edge is on a whole pixel,
    as a pure black-and-white image identical to the target's own raster --
    and so, through the layout test above, identical to aruco2's board
    image. Every edge on a whole pixel leaves nothing to anti-alias, so this
    says nothing about seams; the test below does."""
    _skip_without_cairo()
    target = ChArUco2(num_squares_x=5, num_squares_y=4, square_size=10.0)
    svg_path = target.save_to_svg(tmp_path / "board.svg", border_width=10.0)

    # 12 px/mm: a 10 mm 4x4 square is 120 px, 20 px a cell, 30 px its band.
    raster = _rasterise_svg(svg_path, 12.0)
    image, _ = target._render(12.0 * 25.4, border_width=10.0)
    assert set(np.unique(raster).tolist()) == {0, 255}
    assert np.array_equal(raster, image)


@pytest.mark.parametrize("px_per_mm", [11.37, 300 / 25.4])
def test_charuco2_svg_leaves_no_seams_between_abutting_cells(tmp_path: Path, px_per_mm) -> None:
    """At a scale where cell edges fall between pixels, a pixel wholly inside
    black -- black across itself and its eight neighbours, in an 8x
    supersampled raster of the layout -- comes out black. Separate paths
    anti-alias each shared edge on its own, leaving a grey seam there (over
    a thousand such pixels for this board); one path fills them as one."""
    _skip_without_cairo()
    target = ChArUco2(num_squares_x=5, num_squares_y=4, square_size=10.0)
    raster = _rasterise_svg(
        target.save_to_svg(tmp_path / "board.svg", border_width=10.0), px_per_mm)

    k = 8
    reference, _ = target._render(k * px_per_mm * 25.4, border_width=10.0)
    rows = min(raster.shape[0], reference.shape[0] // k)
    cols = min(raster.shape[1], reference.shape[1] // k)
    blocks = reference[:rows * k, :cols * k].reshape(rows, k, cols, k)
    black = np.pad((blocks == 0).all(axis=(1, 3)), 1)
    inside = np.ones((rows, cols), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            inside &= black[dy:dy + rows, dx:dx + cols]
    assert inside.sum() > 0.2 * inside.size, "the comparison has black to look at"
    assert int(np.count_nonzero(raster[:rows, :cols][inside] > 40)) == 0


@pytest.mark.parametrize("px_per_mm", [300 / 25.4, 12.0])
def test_charuco2_svg_round_trip_is_detectable(tmp_path: Path, px_per_mm) -> None:
    """Rasterise the printable SVG at a known px/mm and confirm aruco2
    detects every corner, at the position the SVG's own real-world mm scale
    implies -- at a print-like dpi as well as at an exact one."""
    _skip_without_cairo()
    target = ChArUco2(num_squares_x=5, num_squares_y=5, square_size=10.0)
    svg_path = target.save_to_svg(
        tmp_path / target.printable_name(
            {"num_squares_x": 5, "num_squares_y": 5, "square_size": 10.0}),
        border_width=10.0)

    raster = _rasterise_svg(svg_path, px_per_mm)
    detection = target.find_in_image(raster)
    n_corners = target.point_data.shape[-2]
    assert len(detection.keys) == n_corners

    expected_px = _expected_pixels(target, px_per_mm, border_mm=10.0)
    by_id = dict(zip(detection.keys.tolist(), detection.image_points))
    for gid, expected in enumerate(expected_px):
        assert np.linalg.norm(by_id[gid] - expected) < 1.0


# -- marker ids ---------------------------------------------------------------


def test_grid_board_ids_pass_through_to_aruco2() -> None:
    """A board printed with its own ids -- as each cube face is -- is found
    only when detected with those ids, with every corner where it should be."""
    dict_int = int(aruco2.DICT_4X4_1000)
    ids = list(range(100, 125))
    image = render_grid_board_image((5, 5), dict_int, 30, ids)
    square_px = 4 * 30

    corner_ids, points = detect_grid_board_corners(
        image, (5, 5), dict_int, 1.0, ids=ids)
    assert corner_ids is not None
    assert sorted(corner_ids.tolist()) == list(range(36))
    expected = layout.grid_board_corners((5, 5), square_px) + square_px / 4
    assert np.abs(points - expected[corner_ids]).max() < 1.0

    default_ids, _ = detect_grid_board_corners(image, (5, 5), dict_int, 1.0)
    assert default_ids is None


def test_grid_board_ids_must_cover_every_square() -> None:
    dict_int = int(aruco2.DICT_4X4_1000)
    image = render_grid_board_image((3, 3), dict_int, 30)
    with pytest.raises(ValueError, match="needs that many ids; got 8"):
        render_grid_board_image((3, 3), dict_int, 30, list(range(8)))
    with pytest.raises(ValueError, match="needs that many ids; got 10"):
        detect_grid_board_corners(image, (3, 3), dict_int, 1.0, ids=list(range(10)))
    with pytest.raises(ValueError, match="needs that many ids"):
        grid_board_marker_bits((3, 3), dict_int, list(range(4)))


@pytest.mark.parametrize("ids, refused", [
    ([0.9] * 9, "whole numbers"),
    ([0.0, 1.5, 2, 3, 4, 5, 6, 7, 8], "whole numbers"),
    ([0, 1, 2, 3, 4, 5, 6, 7, 7], r"ids \[7\] are repeated"),
    ([0, 1, 2, 3, 4, 5, 6, 7, 60], r"\[60\] are outside this dictionary's 50"),
    ([-1, 1, 2, 3, 4, 5, 6, 7, 8], r"\[-1\] are outside"),
])
def test_grid_board_ids_are_distinct_markers_of_the_dictionary(ids, refused) -> None:
    """A truncated, repeated or missing id would print, or look for, a
    marker nobody asked for -- and detection would say nothing about it."""
    dict_int = int(aruco2.DICT_4X4_50)
    image = render_grid_board_image((3, 3), dict_int, 30)
    with pytest.raises(ValueError, match=refused):
        detect_grid_board_corners(image, (3, 3), dict_int, 1.0, ids=ids)
    with pytest.raises(ValueError, match=refused):
        render_grid_board_image((3, 3), dict_int, 30, ids)
    with pytest.raises(ValueError, match=refused):
        grid_board_marker_bits((3, 3), dict_int, ids)
    # Whole numbers held as floats are still ids.
    grid_board_marker_bits((3, 3), dict_int, [float(i) for i in range(9)])


def test_grid_board_marker_bits_refuses_an_id_the_dictionary_lacks() -> None:
    with pytest.raises(ValueError, match="outside this dictionary's 50 markers"):
        grid_board_marker_bits((2, 2), int(aruco2.DICT_4X4_50), [0, 1, 2, 50])


def test_a_charuco2_board_names_itself_and_writes_itself(tmp_path: Path) -> None:
    spec = {"num_squares_x": 5, "num_squares_y": 7, "square_size": 4}
    target = ChArUco2(**spec)
    assert target.point_data.shape == (1, 6 * 8, 3)
    assert ChArUco2.printable_name(spec, "pdf_vector") == "charuco2_5x7_4mm.pdf"

    saved = target.save_printable(tmp_path / "nested/charuco2.txt", "svg")
    assert saved == (tmp_path / "nested/charuco2.svg").resolve()
    assert saved.exists()
    assert saved.stat().st_size > 0
