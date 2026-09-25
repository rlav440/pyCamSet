"""
The printed layout of a ChArUco2 grid board, as pure geometry.

This is the single description of what a grid board looks like on paper --
the SVG, the vector and raster PDFs, :meth:`ChArUco2.plot` and the tests all
draw from it, so there is no second rendering path to fall out of step with
what aruco2 detects. It needs numpy and nothing else: the marker bits come
from the caller (see :func:`pyCamSet.calibration_targets.markers
.aruco2_gridboard.grid_board_marker_bits`), so this module imports cleanly
without aruco2.

The layout reproduces aruco2's own ``getGridBoardImage``. For a board of
``W x H`` squares of side ``s`` and a dictionary of ``m x m`` bit markers:

- every square is one whole marker, ``(m+2) x (m+2)`` equal cells of side
  ``s / (m+2)``: a one-cell border around the ``m x m`` payload;
- square ``(x, y)`` holds marker ``ids[y * W + x]``, standard (black border)
  when ``x + y`` is even and inverted (every cell flipped, so a white border)
  when it is odd;
- a band of depth ``s / 4`` surrounds the board, white except for black
  ``s x s/4`` tabs opposite every inverted edge square and a black
  ``s/4 x s/4`` square at each of its four outer corners.

Coordinates are board-local: ``origin`` is the top-left corner of square
``(0, 0)`` -- the first intersection corner -- with x to the right and y
down, in whatever unit ``square_size`` is given in. The band lies outside
``[0, W*s] x [0, H*s]``, so a cube face can place a board anywhere on the
face just by moving its origin.

Every shape is a black axis-aligned rectangle ``(x0, y0, x1, y1)``; white is
the absence of one.
"""

from __future__ import annotations

import numpy as np

#: How deep the band around a board is, as a fraction of its square size.
#: aruco2 draws it ``markerSizePix // 4`` pixels deep.
BAND_FRACTION = 0.25


def band_depth(square_size: float) -> float:
    """The depth of the tabbed band drawn around a board of this square size."""
    return float(square_size) * BAND_FRACTION


def grid_board_bounds(
    grid_size: tuple[int, int], square_size: float,
    origin: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float, float, float]:
    """
    Everything a board prints on, band included.

    :param grid_size: ``(W, H)``, in squares.
    :param square_size: the side of one square.
    :param origin: where the top-left corner of square ``(0, 0)`` sits.
    :return: ``(x_min, y_min, x_max, y_max)``.
    """
    grid_w, grid_h = int(grid_size[0]), int(grid_size[1])
    s = float(square_size)
    b = band_depth(s)
    ox, oy = float(origin[0]), float(origin[1])
    return (ox - b, oy - b, ox + grid_w * s + b, oy + grid_h * s + b)


def grid_board_corners(
    grid_size: tuple[int, int], square_size: float,
    origin: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    The board's intersection corners, row-major.

    Row ``gid`` of the result is aruco2's global corner id ``gid = row *
    (W+1) + col``, at ``origin + (col, row) * square_size``.

    :return: an ``((W+1) * (H+1), 2)`` float array.
    """
    grid_w, grid_h = int(grid_size[0]), int(grid_size[1])
    cols, rows = np.meshgrid(
        np.arange(grid_w + 1, dtype=np.float64),
        np.arange(grid_h + 1, dtype=np.float64),
    )
    return np.stack(
        [float(origin[0]) + cols.ravel() * float(square_size),
         float(origin[1]) + rows.ravel() * float(square_size)],
        axis=-1,
    )


def grid_board_cells(
    grid_size: tuple[int, int], marker_bits: np.ndarray,
) -> np.ndarray:
    """
    Which marker cells of a board are black, as one boolean cell grid.

    :param grid_size: ``(W, H)``, in squares.
    :param marker_bits: ``(W*H, m, m)`` payload bits, one marker per square
        in row-major order, truthy where the *standard* (black-bordered)
        marker is black.
    :return: an ``(H*(m+2), W*(m+2))`` bool array, True for black -- border
        cells included, inverted squares already flipped.
    """
    grid_w, grid_h = int(grid_size[0]), int(grid_size[1])
    bits = np.asarray(marker_bits)
    if bits.ndim != 3 or bits.shape[1] != bits.shape[2]:
        raise ValueError(
            f"marker_bits must be (W*H, m, m) square bit grids; got shape "
            f"{bits.shape}.")
    if bits.shape[0] != grid_w * grid_h:
        raise ValueError(
            f"A {grid_w}x{grid_h} board needs {grid_w * grid_h} markers' "
            f"bits; got {bits.shape[0]}.")
    n = bits.shape[1] + 2

    # Pad every payload with its black one-cell border.
    cells = np.pad(bits.astype(bool), ((0, 0), (1, 1), (1, 1)),
                   constant_values=True)
    cells = cells.reshape(grid_h, grid_w, n, n)
    # Squares with an odd x + y carry the inverted marker.
    parity = (np.add.outer(np.arange(grid_h), np.arange(grid_w)) % 2).astype(bool)
    cells = cells ^ parity[:, :, None, None]
    # (H, W, n, n) -> (H, n, W, n) -> one grid of cells, row-major.
    return cells.transpose(0, 2, 1, 3).reshape(grid_h * n, grid_w * n)


def _cell_edges(n_squares: int, cells_per_square: int, square_size: float) -> np.ndarray:
    """
    The coordinate of every cell edge along one axis.

    Built from whole squares plus a fraction of one, so an edge that falls on
    a square boundary is exactly ``k * square_size`` -- the same value a band
    tab beside it is built from.
    """
    idx = np.arange(n_squares * cells_per_square + 1)
    whole, part = np.divmod(idx, cells_per_square)
    return whole * float(square_size) + part * (float(square_size) / cells_per_square)


def grid_board_rectangles(
    grid_size: tuple[int, int], square_size: float, marker_bits: np.ndarray,
    origin: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Every black shape a grid board prints, as rectangles.

    Horizontal runs of black cells within one cell row are merged into one
    rectangle, which keeps an SVG small; the band's tabs and corner squares
    follow.

    :param grid_size: ``(W, H)``, in squares.
    :param square_size: the side of one square.
    :param marker_bits: ``(W*H, m, m)`` payload bits, row-major, truthy where
        the standard marker is black (see :func:`grid_board_cells`).
    :param origin: where the top-left corner of square ``(0, 0)`` sits.
    :return: an ``(N, 4)`` float array of ``(x0, y0, x1, y1)``, with
        ``x0 < x1`` and ``y0 < y1``.
    """
    grid_w, grid_h = int(grid_size[0]), int(grid_size[1])
    s = float(square_size)
    ox, oy = float(origin[0]), float(origin[1])
    cells = grid_board_cells((grid_w, grid_h), marker_bits)
    n = cells.shape[1] // grid_w
    x_edges = ox + _cell_edges(grid_w, n, s)
    y_edges = oy + _cell_edges(grid_h, n, s)

    rects: list[tuple[float, float, float, float]] = []
    for r, row in enumerate(cells):
        # Runs of True: a run starts where the padded row steps 0 -> 1 and
        # ends (exclusive) where it steps 1 -> 0.
        steps = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        starts = np.flatnonzero(steps == 1)
        ends = np.flatnonzero(steps == -1)
        for c0, c1 in zip(starts, ends):
            rects.append((x_edges[c0], y_edges[r], x_edges[c1], y_edges[r + 1]))

    b = band_depth(s)
    right, bottom = ox + grid_w * s, oy + grid_h * s
    # Band tabs, in the same order and on the same squares aruco2 draws them.
    for x in range(1, grid_w, 2):
        rects.append((ox + x * s, oy - b, ox + (x + 1) * s, oy))
    for x in range(grid_h % 2, grid_w, 2):
        rects.append((ox + x * s, bottom, ox + (x + 1) * s, bottom + b))
    for y in range(1, grid_h, 2):
        rects.append((ox - b, oy + y * s, ox, oy + (y + 1) * s))
    for y in range(grid_w % 2, grid_h, 2):
        rects.append((right, oy + y * s, right + b, oy + (y + 1) * s))
    # The band's four outer corner squares.
    rects.append((ox - b, oy - b, ox, oy))
    rects.append((right, oy - b, right + b, oy))
    rects.append((ox - b, bottom, ox, bottom + b))
    rects.append((right, bottom, right + b, bottom + b))

    return np.asarray(rects, dtype=np.float64).reshape(-1, 4)


def rasterise_rectangles(
    rects: np.ndarray, px_per_unit: float,
    top_left: tuple[float, float] = (0.0, 0.0),
    shape: tuple[int, int] | None = None,
    image: np.ndarray | None = None,
) -> np.ndarray:
    """
    Draw black rectangles into a white uint8 image.

    A pixel is black when its centre lies inside a rectangle (half-open, so
    two abutting rectangles never both claim, or both miss, the pixels on
    their shared edge). At a scale where every edge lands on a whole pixel
    this is exact; elsewhere a cell is at most one pixel wider or narrower
    than its neighbour.

    :param rects: ``(N, 4)`` ``(x0, y0, x1, y1)`` rectangles.
    :param px_per_unit: pixels per unit of the rectangles' coordinates.
    :param top_left: the coordinate pixel ``(0, 0)``'s top-left corner sits at.
    :param shape: ``(rows, cols)`` of a new white image to draw into.
    :param image: an existing uint8 image to draw into instead, in place.
    :return: the image drawn into.
    """
    if image is None:
        if shape is None:
            raise ValueError("Give either the shape of a new image or an image.")
        image = np.full((int(shape[0]), int(shape[1])), 255, dtype=np.uint8)
    rows, cols = image.shape[:2]
    r = np.asarray(rects, dtype=np.float64).reshape(-1, 4)
    ppu = float(px_per_unit)
    # Pixel j's centre is at (j + 0.5) / ppu, so the first centre at or past
    # an edge e is j = ceil(e * ppu - 0.5). When e sits exactly on a pixel
    # centre this pre-ceil value is mathematically an exact integer, but
    # float64 noise of a few ULP can land it a hair either side -- so two
    # edges at the same true fractional pixel offset (e.g. every lattice line
    # of a board whose margin is a half-integer number of pixels) can ceil to
    # different pixels and make otherwise-identical cells one pixel wider or
    # narrower than their neighbours. Snapping to 1e-6 px before the ceiling
    # removes that noise -- far finer than a pixel, so it never masks a
    # genuine sub-pixel edge position -- while keeping the half-open,
    # int64/shape-preserving behaviour unchanged.
    x = np.ceil(np.round((r[:, [0, 2]] - float(top_left[0])) * ppu - 0.5, 6)).astype(np.int64)
    y = np.ceil(np.round((r[:, [1, 3]] - float(top_left[1])) * ppu - 0.5, 6)).astype(np.int64)
    x = np.clip(x, 0, cols)
    y = np.clip(y, 0, rows)
    for (c0, c1), (r0, r1) in zip(x, y):
        if c1 > c0 and r1 > r0:
            image[r0:r1, c0:c1] = 0
    return image


def rectangles_svg_path(
    rects: np.ndarray,
    affine: np.ndarray | None = None,
    offset: tuple[float, float] = (0.0, 0.0),
    decimals: int = 8,
) -> str:
    """
    SVG path data drawing every rectangle as a closed sub-path.

    Every sub-path winds the same way, so written as one ``<path>`` with
    ``fill-rule="nonzero"`` the rectangles fill as a single shape: abutting
    cells meet on shared coordinates and rasterise without anti-aliased
    seams between them.

    :param rects: ``(N, 4)`` ``(x0, y0, x1, y1)`` rectangles.
    :param affine: optional 2x3 or 3x3 matrix applied to every corner, e.g.
        to place a cube face in its net. A rotation keeps every sub-path's
        winding the same; a reflection flips all of them together, which
        ``nonzero`` fills just the same.
    :param offset: added to every corner after ``affine``.
    :param decimals: fixed-point digits written per coordinate.
    :return: the path's ``d`` attribute.
    """
    r = np.asarray(rects, dtype=np.float64).reshape(-1, 4)
    # Corners in drawing order: top-left, top-right, bottom-right, bottom-left.
    corners = np.stack(
        [r[:, [0, 1]], r[:, [2, 1]], r[:, [2, 3]], r[:, [0, 3]]], axis=1)
    if affine is not None:
        a = np.asarray(affine, dtype=np.float64)
        corners = corners @ a[:2, :2].T + a[:2, 2]
    corners = corners + np.asarray(offset, dtype=np.float64)

    fmt = f"{{:.{int(decimals)}f}}"
    parts = []
    for quad in corners:
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = (
            (fmt.format(px), fmt.format(py)) for px, py in quad)
        parts.append(f"M{x0} {y0}L{x1} {y1}L{x2} {y2}L{x3} {y3}Z")
    return "".join(parts)
