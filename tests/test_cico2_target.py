"""
What a CIco2 is.

A ChArUco2 board clipped to a triangle, twenty times.  Two things about
ChArUco2 decide whether that is worth doing, and both are checked here: that a
marker on every square keeps the corners on a clipped board's *boundary*, which
is most of what a clipped board has; and that the ring of tabs which gives a
rectangular board's outer corners their contrast still does so along a
staircase.
"""
import re

import numpy as np
import pytest

aruco2 = pytest.importorskip("aruco2")

from pyCamSet import calibration_targets  # noqa: E402
from pyCamSet.calibration_targets import TARGET_NAMES  # noqa: E402
from pyCamSet.calibration_targets.cico2 import (  # noqa: E402
    CIco2, corners_touching_cells,
)
from pyCamSet.calibration_targets.core.abstract_target import (  # noqa: E402
    EXPORT_KINDS,
)
from pyCamSet.calibration_targets.core.parameters import (  # noqa: E402
    DocumentedParameters,
)
from pyCamSet.calibration_targets.core.target_registry import (  # noqa: E402
    TARGET_LABELS,
)
from pyCamSet.calibration_targets.markers.gridboard_layout import (  # noqa: E402
    grid_board_corners,
)
from pyCamSet.calibration_targets.polyhedra import (  # noqa: E402
    TRIANGLE_CORNERS, clip_lattice_to_face, corners_within_cells,
)


def _skip_without_cairo():
    try:
        import cairosvg  # noqa: F401
    except (ImportError, OSError) as error:
        pytest.skip(f"native cairo is unavailable: {error}")


@pytest.fixture(scope="module")
def target():
    return CIco2(length=100.0, n_points=12)


def _render(target, face_index, width=1600):
    import cv2
    return cv2.cvtColor(
        target._face_texture(face_index, width), cv2.COLOR_RGB2GRAY)


# -- what it is ---------------------------------------------------------------

def test_cico2_is_registered_labelled_and_exported():
    assert "CIco2" in list(TARGET_NAMES)
    assert TARGET_LABELS["CIco2"] == "ChArUco2 icosahedron"
    assert "CIco2" in calibration_targets.__all__
    assert calibration_targets.CIco2 is CIco2


def test_it_is_read_with_aruco2_only():
    assert list(CIco2.DETECTOR_BACKENDS) == ["aruco2"]


def test_it_declares_the_arguments_that_decide_its_geometry():
    offered = CIco2.construction_parameters()
    assert isinstance(offered, DocumentedParameters)
    assert [p.key for p in offered.parameters] == [
        "n_points", "length", "border_fraction", "aruco_dict"]


def test_a_printable_is_named_for_what_it_is():
    assert CIco2.printable_name(
        {"n_points": 8, "length": 100.0}, "svg") == "cico2_8points_100mm.svg"


# -- why it is a ChArUco2 board and not a ChArUco one -------------------------

def test_a_marker_on_every_square_is_what_makes_clipping_worth_it():
    """
    The reason for preferring ChArUco2 here, asserted rather than assumed.

    A ChArUco1 corner is the meeting of four printed squares; a ChArUco2
    corner is a corner of any printed square, because that square carries a
    whole marker of its own.  Clipping throws away a board's outside, which is
    exactly where the difference is, so it is much larger here than it is on a
    full rectangle.
    """
    ratios = []
    for n_points in (7, 8, 10):
        cells = clip_lattice_to_face(TRIANGLE_CORNERS, n_points)
        within = len(corners_within_cells(cells))
        touching = len(corners_touching_cells(cells))
        assert touching > 2 * within, (
            f"at {n_points} squares: {touching} against {within}")
        ratios.append(touching / within)

    # The advantage is in the boundary, so it is largest where the boundary is
    # most of the board: 6x at seven squares to an edge, falling to 2.6x at
    # ten.  A face this target can afford is at the coarse end of that.
    assert ratios == sorted(ratios, reverse=True)


def test_every_face_carries_the_same_lattice(target):
    assert target.point_data.shape == (20, target.points_per_face, 3)
    assert target.point_local.shape == target.point_data.shape
    assert target.points_per_face == 65


def test_each_face_is_flat(target):
    for face in target.point_data:
        centred = face - face.mean(axis=0)
        assert np.linalg.svd(centred, compute_uv=False)[-1] < 1e-9


def test_the_points_sit_inside_the_face_they_belong_to(target):
    corners = target.basis.face_corners(target.length)
    for face_index, face in enumerate(target.point_data):
        triangle = corners[face_index]
        origin = triangle[0]
        basis = np.stack([triangle[1] - origin, triangle[2] - origin], axis=1)
        coefficients, *_ = np.linalg.lstsq(basis, (face - origin).T, rcond=None)
        a, b = coefficients
        assert np.all(a > -1e-9) and np.all(b > -1e-9)
        assert np.all(a + b < 1 + 1e-9)


# -- the marker alphabet ------------------------------------------------------

def test_only_the_squares_a_face_prints_take_an_id_of_their_own(target):
    """
    The squares clipping threw away share one block of filler between all
    twenty faces, and only the printed ones get a block each.  That is what
    lets a face be twelve squares to an edge inside a thousand-marker
    dictionary; a block each would need 1920.
    """
    assert len(target.face_ids) == 20
    squares = target.board_columns * target.board_rows
    assert all(len(ids) == squares for ids in target.face_ids)

    # aruco2 requires a distinct id per square of a board.
    for face_index, ids in enumerate(target.face_ids):
        assert len(set(ids)) == len(ids), f"face {face_index} repeats an id"

    printed = [set(ids) - set(target.filler_ids) for ids in target.face_ids]
    assert all(len(p) == target.markers_per_face for p in printed)
    for i in range(20):
        for j in range(i + 1, 20):
            assert not printed[i] & printed[j], (
                f"faces {i} and {j} share a printed marker")
            assert set(target.face_ids[i]) & set(target.face_ids[j]) == set(
                target.filler_ids), "faces should share the filler and nothing else"

    total = 20 * target.markers_per_face + len(target.filler_ids)
    assert total <= 1000, "the whole point is that this fits"


def test_a_shared_filler_marker_is_never_printed(target):
    """
    Sharing is only safe because no face prints one.  A filler marker that
    reached paper would be on twenty boards at once, and the face it was found
    on would be a coin toss.
    """
    printed_cells = target._printed_cells()
    columns = target.board_columns
    for ids in target.face_ids:
        for column, row in printed_cells:
            assert ids[row * columns + column] not in target.filler_ids


def test_a_face_too_fine_for_the_dictionary_is_refused():
    """Sharing the filler raises the ceiling; it does not remove it."""
    with pytest.raises(ValueError, match="markers"):
        CIco2(length=100.0, n_points=16)


@pytest.mark.parametrize("values, refused", [
    ({"length": 0.0}, "edge length"),
    ({"n_points": 4}, "at least"),
    ({"n_points": 12.5}, "whole number"),
    ({"border_fraction": 0.0}, "neither all of it nor none"),
    ({"border_fraction": 0.01}, "band of tabs"),
])
def test_an_impossible_icosahedron_is_refused(values, refused):
    with pytest.raises(ValueError, match=refused):
        CIco2(**{"length": 100.0, "n_points": 12, **values})


# -- reading one back ---------------------------------------------------------

def test_a_printed_face_decodes_as_that_face_and_no_other(target):
    _skip_without_cairo()
    for face_index in (0, 7, 13, 19):
        detection = target.find_in_image(_render(target, face_index))
        assert detection.has_data, f"face {face_index} read as nothing"
        keys = np.asarray(detection.keys)
        assert set(keys[:, 0].tolist()) == {face_index}
        assert len(keys) == target.points_per_face


def test_the_band_keeps_the_corners_on_a_clipped_board_s_edge(target):
    """
    A rectangular board gets a ring of tabs so its outer corners have a black
    side and a white one; clipped, its outer boundary is a staircase instead.
    If the band did not follow it, the boundary corners -- most of a clipped
    board's corners -- would not be found.
    """
    _skip_without_cairo()
    printed = target._printed_cells()
    boundary = [
        index for index, (column, row) in enumerate(target.live_corners)
        if not all((column + dx, row + dy) in printed
                   for dx in (-1, 0) for dy in (-1, 0))
    ]
    assert len(boundary) > len(target.live_corners) // 2, (
        "most of a clipped board's corners should be on its boundary")

    keys = np.asarray(target.find_in_image(_render(target, 0)).keys)
    found = set(keys[:, 1].tolist())
    assert set(boundary) <= found, "boundary corners were not found"


def test_the_band_is_the_chessboard_carried_on_outside_the_board(target):
    """
    Every tab lies in a square the chessboard would have printed black.

    The rule has to be taken from the chessboard, not from the boundary.
    Across an edge the parity flips, so "the square outside is black" and
    "the square inside is white" agree, and either reading works on a
    rectangle.  Across a corner it does not flip, and a staircase is mostly
    corners -- so a rule about the inside squares puts the corner tabs on
    white and breaks the pattern, which is what this catches.
    """
    from pyCamSet.calibration_targets.markers.gridboard_layout import band_depth

    square = target.square_size
    depth = band_depth(square)
    ox, oy = target.board_offset
    printed = target._printed_cells()

    tabs = target._staircase_band()
    assert tabs, "a clipped board has a boundary, so it has a band"
    for x0, y0, x1, y1 in tabs:
        # The square this tab lies in, found from its middle so that a strip
        # along an edge is attributed to the square outside, not the one in.
        column = int(round(((x0 + x1) / 2 - ox) // square))
        row = int(round(((y0 + y1) / 2 - oy) // square))
        assert (column, row) not in printed, "a tab is on the board itself"
        assert (column + row) % 2 == 0, (
            f"tab at square ({column}, {row}) sits on a white square")
        assert x1 - x0 <= square + 1e-9 and y1 - y0 <= square + 1e-9
        assert min(x1 - x0, y1 - y0) <= depth + 1e-9, "a tab is deeper than the band"


def test_the_band_matches_the_real_one_on_an_unclipped_board(target):
    """
    Pinned against the band aruco2's own layout draws.

    Given a board with nothing clipped away, the staircase rule has to
    reproduce the rectangular band -- less the two squares aruco2 puts at the
    far corners of the board, which sit on white and are that design's own
    anchors rather than part of the chessboard.
    """
    import numpy as np
    from pyCamSet.calibration_targets.markers.aruco2_gridboard import (
        grid_board_marker_bits)
    from pyCamSet.calibration_targets.markers.gridboard_layout import (
        band_depth, grid_board_rectangles)

    columns, rows = target.grid_size
    square = target.square_size
    whole = {(c, r) for c in range(columns) for r in range(rows)}

    unclipped = object.__new__(type(target))
    unclipped.__dict__.update(target.__dict__)
    unclipped.cells = np.array(sorted(whole))
    unclipped.board_offset = np.zeros(2)

    bits = grid_board_marker_bits(
        target.grid_size, target._aruco_dict_int, ids=target.face_ids[0])
    everything = grid_board_rectangles(
        target.grid_size, square, bits, origin=(0.0, 0.0))
    depth = band_depth(square)
    real = [tuple(np.round(r, 9)) for r in everything
            if r[0] < -1e-9 or r[1] < -1e-9
            or r[2] > columns * square + 1e-9 or r[3] > rows * square + 1e-9]

    def paint(rects, per=16):
        scale = per / square
        grid = np.zeros((int((rows + 2) * per), int((columns + 2) * per)), bool)
        for x0, y0, x1, y1 in rects:
            grid[int(round((y0 + square) * scale)):int(round((y1 + square) * scale)),
                 int(round((x0 + square) * scale)):int(round((x1 + square) * scale))] = True
        return grid

    drawn = paint(unclipped._staircase_band())
    expected = paint(real)
    assert not (drawn & ~expected).any(), "the band draws outside the real one"

    missing = expected & ~drawn
    # Only the two far-corner anchors, each one band square.
    assert missing.sum() == 2 * round(depth / square * 16) ** 2


def test_a_printed_face_puts_its_corners_where_it_says_they_are(target):
    _skip_without_cairo()
    width = 1600
    detection = target.find_in_image(_render(target, 0, width))
    scale = width / target.length
    all_corners = np.asarray(grid_board_corners(
        target.grid_size, target.square_size,
        origin=tuple(target.board_offset)))
    keep = [target._corner_index(c, r) for c, r in target.live_corners]
    expected = all_corners[keep] * scale

    keys = np.asarray(detection.keys)
    found = np.asarray(detection.image_points)
    error = np.linalg.norm(found - expected[keys[:, 1]], axis=1)
    assert error.max() < 0.02 * target.square_size * scale


def test_the_printed_net_gives_back_every_face(target):
    _skip_without_cairo()
    from io import BytesIO

    import cairosvg
    from PIL import Image

    drawing, _, _ = target._svg_document(
        border_width=5.0, draw_cut_outline=False, draw_face_ids=False)
    png = cairosvg.svg2png(
        bytestring=drawing.tostring().encode("utf-8"), output_width=6000)
    with Image.open(BytesIO(png)) as image:
        net = np.asarray(image.convert("L"))

    keys = np.asarray(target.find_in_image(net).keys)
    assert sorted(set(keys[:, 0].tolist())) == list(range(20))
    for face_index in range(20):
        assert int(np.sum(keys[:, 0] == face_index)) == target.points_per_face


# -- printing -----------------------------------------------------------------

@pytest.mark.parametrize("kind", EXPORT_KINDS)
def test_it_writes_itself_as_every_format_it_offers(target, kind, tmp_path):
    _skip_without_cairo()
    written = target.save_printable(tmp_path / f"net_{kind}", kind=kind)
    assert written.exists() and written.stat().st_size > 1024
    if kind.startswith("pdf"):
        embedded = bool(re.search(rb"/Subtype\s*/Image", written.read_bytes()))
        assert embedded == (kind == "pdf_raster")


def test_a_format_it_is_not_is_refused(target, tmp_path):
    with pytest.raises(ValueError, match="cannot be written as"):
        target.save_printable(tmp_path / "net", kind="dxf")


def test_the_solid_writes_itself_for_printing(target, tmp_path):
    pv = pytest.importorskip("pyvista")
    mesh = pv.read(str(target.to_stl(tmp_path / "core")))
    assert mesh.n_points == 12
    assert mesh.n_faces == 20


# -- the spec round trip ------------------------------------------------------

def test_it_can_be_rebuilt_from_what_it_was_made_with(target):
    from pyCamSet.calibration_targets.core.target_registry import (
        build_target, spec_of)

    spec = spec_of(target)
    assert spec["type"] == "CIco2"
    assert np.allclose(build_target(spec).point_data, target.point_data)
