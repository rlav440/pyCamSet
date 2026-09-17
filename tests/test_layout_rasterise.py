"""Regression tests for ``charuco2/layout.py``'s ``rasterise_rectangles``.

Pure-numpy unit tests for the rasteriser itself, independent of aruco2 (the
module docstring promises it "imports cleanly without aruco2" -- see
``test_layout_imports_without_aruco2`` in ``test_charuco2_target.py``), so
these run even where the optional aruco2 package is absent.

The bug under test: ``rasterise_rectangles`` finds a rectangle edge's first
covered pixel as ``ceil((edge - top_left) * ppu - 0.5)``. When an edge sits
exactly on a pixel *centre*, that pre-ceil value is mathematically an exact
integer -- but float64 rounding of ``(edge - top_left) * ppu`` can land it a
few ULP either side of the true half-integer, so two edges at the same true
fractional pixel offset ceil to different pixels. A board whose margin is a
half-integer number of pixels and whose square size is a whole number of
pixels puts *every* lattice line at exactly this same fractional offset, so
every square should come out the same pixel width and height; before the
fix, some don't.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from pyCamSet.calibration_targets.charuco2 import layout


def _exact_first_pixel(edge: Fraction, top_left: Fraction, ppu: Fraction) -> int:
    """The first pixel index whose centre is at or past ``edge``, computed
    with exact rational arithmetic -- the same rule ``rasterise_rectangles``
    documents (``j = ceil(e * ppu - 0.5)``), but with no float64 involved
    anywhere, so this is the mathematically exact answer to compare against.
    """
    return math.ceil((edge - top_left) * ppu - Fraction(1, 2))


def test_rasterise_rectangles_matches_exact_rational_coverage_on_half_pixel_lattice() -> None:
    """A lattice whose every edge sits at the same true fractional pixel
    offset (a half-integer-pixel margin, whole-pixel squares) must rasterise
    to exactly the half-open pixel-centre coverage a rational-arithmetic
    oracle gives -- for every edge, not just the ones that happen to hit an
    exact float64 half-integer.

    ``ppu = 1607 / 12`` is not special beyond being awkward enough in binary
    floating point that ``i * (12 / ppu) * ppu`` does not always come back to
    exactly ``12 * i`` -- e.g. edge index 11 below rasterises (before the fix)
    from a scaled value of ``134.50000000000003`` while edge index 10 is the
    exact ``122.5``: the same nominal half-pixel offset, computed to
    different sides of it by float noise.
    """
    ppu_exact = Fraction(1607, 12)
    square_px = 12          # whole number of pixels per square
    border_px = Fraction(5, 2)  # a half-integer-pixel margin
    n = 60

    ppu = float(ppu_exact)
    square_size = square_px / ppu     # as the real pipeline computes it
    border = float(border_px) / ppu
    top_left_x = -border
    x0_exact = Fraction(top_left_x).limit_denominator(10**12)

    # Sanity check this reproduces the float noise the bug report describes,
    # so the test is not vacuous once the fix is in place.
    scaled = np.array([i * square_size for i in range(n + 1)]) * ppu
    nominal = np.array([i * square_px for i in range(n + 1)]) + float(border_px)
    assert np.any(scaled != nominal), (
        "expected setup to hit float64 noise on some edges; scaled == nominal "
        "everywhere, so this scale no longer exercises the bug")

    canvas_cols = n * square_px + 2 * int(math.ceil(border_px)) + 4
    for i in range(n):
        left = _exact_first_pixel(Fraction(i) * Fraction(square_px, 1) / ppu_exact,
                                   x0_exact, ppu_exact)
        right = _exact_first_pixel(Fraction(i + 1) * Fraction(square_px, 1) / ppu_exact,
                                    x0_exact, ppu_exact)
        # The oracle itself must land on a whole-square width here: that is
        # the point of a half-integer margin with whole-pixel squares.
        assert right - left == square_px

        # Each square rasterised on its own (not as part of an abutting row
        # of same-colour squares, which would merge into one run and hide a
        # misplaced boundary between two black neighbours -- see the module
        # docstring) so its own pixel extent can be read back unambiguously.
        rect = np.array([[i * square_size, 0.0, (i + 1) * square_size, square_size]])
        image = layout.rasterise_rectangles(
            rect, ppu, top_left=(top_left_x, top_left_x),
            shape=(square_px + 4, canvas_cols))
        row = image[2, :]
        black = np.flatnonzero(row == 0)
        actual_left, actual_right = int(black[0]), int(black[-1]) + 1
        assert (actual_left, actual_right) == (left, right), (
            f"square {i}: rational oracle says pixel columns [{left}, {right}), "
            f"rasteriser produced [{actual_left}, {actual_right})")


def test_rasterise_rectangles_checkerboard_squares_all_same_pixel_size() -> None:
    """A whole checkerboard at the half-pixel-margin, whole-pixel-square
    scale: every black square must rasterise to the identical width and
    height in pixels -- none a pixel wider or narrower than its neighbours.
    """
    ppu = 1607.0 / 12
    square_px = 12
    border_px = 2.5
    n = 24  # squares per side

    square_size = square_px / ppu
    border = border_px / ppu
    top_left = (-border, -border)

    rects = [
        (i * square_size, j * square_size, (i + 1) * square_size, (j + 1) * square_size)
        for j in range(n) for i in range(n) if (i + j) % 2 == 0
    ]
    board_px = int(round(n * square_px + 2 * border_px))
    image = layout.rasterise_rectangles(
        np.asarray(rects), ppu, top_left=top_left, shape=(board_px, board_px))

    def run_lengths(line: np.ndarray) -> np.ndarray:
        black = line == 0
        steps = np.diff(np.concatenate(([0], black.astype(np.int8), [0])))
        starts = np.flatnonzero(steps == 1)
        ends = np.flatnonzero(steps == -1)
        return ends - starts

    mid = board_px // 2
    widths = run_lengths(image[mid, :])
    heights = run_lengths(image[:, mid])

    assert widths.tolist(), "expected at least one black run to measure"
    assert set(widths.tolist()) == {square_px}, (
        f"square widths are not all {square_px}px: {sorted(set(widths.tolist()))}")
    assert set(heights.tolist()) == {square_px}, (
        f"square heights are not all {square_px}px: {sorted(set(heights.tolist()))}")


def test_rasterise_rectangles_snaps_reproduced_float_noise_example() -> None:
    """The exact scenario the bug report names: one edge's scaled value comes
    back as ``134.50000000000003`` (float noise) and an earlier one as the
    exact ``122.5`` -- the same nominal half-pixel offset (``i * 12 + 2.5``
    for i=10 and i=11). Both must ceil-after-snap the same way relative to
    their own square, so both squares are exactly ``square_px`` wide.
    """
    ppu = 1607.0 / 12
    square_px = 12
    border_px = 2.5
    square_size = square_px / ppu
    border = border_px / ppu
    top_left_x = -border

    # Confirm the float noise this test is named for is actually present on
    # disk right now (documents the bug; guards against the repro drifting).
    scaled_10 = (10 * square_size - top_left_x) * ppu
    scaled_11 = (11 * square_size - top_left_x) * ppu
    assert scaled_10 == 122.5
    assert scaled_11 == 134.50000000000003

    # Rasterised separately (rather than as two abutting black rectangles in
    # one image, which would just merge into a single run and hide the very
    # off-by-one-pixel this test is checking for), so each square's own
    # pixel width is measured in isolation.
    widths = []
    for i in (10, 11):
        rect = np.array([[i * square_size, 0.0, (i + 1) * square_size, square_size]])
        image = layout.rasterise_rectangles(
            rect, ppu, top_left=(top_left_x, top_left_x), shape=(square_px + 4, 160))
        row = image[2, :]
        black = np.flatnonzero(row == 0)
        widths.append(int(black[-1] - black[0] + 1) if black.size else 0)

    assert widths == [square_px, square_px], (
        f"expected both squares to rasterise to {square_px}px, got {widths}")
