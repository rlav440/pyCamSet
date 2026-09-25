"""Generates the lattice placement each CIco2 face size is drawn with.

A design-time tool, not library code.  Its output is the ``FACE_LATTICE``
constant checked in at ``pyCamSet/calibration_targets/cico2.py``, so the
package already carries the result and nothing imports this at runtime.  Run
it directly to regenerate it, alongside calculate_ico_transforms.py.

A square lattice does not fit a triangle, so a CIco2 face prints the whole
cells that fall inside one and throws the rest away.  Where the lattice sits
inside the triangle is free -- nothing downstream cares -- and it is worth
choosing rather than leaving at the corner.  Two things are free:

``phase``
    how far the lattice is shifted, in cells, before it is clipped.  Only the
    x shift is ever useful: the face's bottom edge lies along y = 0, so a y
    shift can only spoil the row that is flush with it, and a search over
    both confirms every optimum has y = 0.

``scale``
    how many cells actually span the face's bottom edge, which need not be
    the whole number ``n_points`` names.  Letting it fall slightly short
    makes the squares slightly larger for the same corner count.

What is being maximised is the corner count, because that is what a face
contributes to a calibration, and a ChArUco2 corner survives clipping if any
one of its four squares was printed.  The marker alphabet is the constraint:
every printed square of every face takes an id of its own, so twenty faces of
a finer lattice run a dictionary out.  The search is held to ``BUDGET``, which
is what the dictionaries in ordinary use hold; without it the twelve-square
face -- the default, and the largest that fits a thousand markers -- would be
handed the 68-corner placement, which needs 1032, and stop building.

Among the placements that tie on corners the largest squares win, subject to
``MIN_WINDOW``, and the phase taken is the middle of the band that wins.

Checked against the constant it generates by
``tests/test_cico2_target.py``, by
``test_the_checked_in_lattice_is_what_the_generator_makes``.
"""
from __future__ import annotations

import numpy as np

#: The face, edge length one, in its own z = 0 plane.
TRIANGLE = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])

#: How many markers the search may spend, over all twenty faces.  What the
#: dictionaries in ordinary use hold; the two larger ones the backend offers
#: are a 1024 and a 2320, and neither buys a whole extra square per edge.
BUDGET = 1000

#: How many faces an icosahedron has.
FACES = 20

#: The smallest face worth clipping a board to, as ``cico2._MIN_POINTS``.
MIN_POINTS = 6

#: How wide the band of phases that reach the best corner count must be, in
#: cells, for a scale to be worth taking.  The band narrows to nothing as the
#: scale is backed off, and its last sliver is a placement that fits only
#: because a lattice corner grazes the face's edge.  Insisting on a little
#: width discards those, and costs about a tenth of a percent of square size.
MIN_WINDOW = 0.02

#: How finely scale and phase are searched, in cells.
STEP = 0.002


def _edges(polygon: np.ndarray) -> np.ndarray:
    return np.roll(polygon, -1, axis=0) - polygon


def clip(scale: float, phase: float) -> np.ndarray:
    """Return the cells wholly inside the face, as ``(n, 2)`` indices."""
    pitch = 1.0 / scale
    span = int(np.ceil(scale)) + 2
    grid = np.stack(np.meshgrid(np.arange(-1, span), np.arange(-1, span),
                                indexing="xy"), axis=-1).reshape(-1, 2)
    edges = _edges(TRIANGLE)
    whole = np.ones(len(grid), dtype=bool)
    for corner in ((0, 0), (1, 0), (1, 1), (0, 1)):
        points = (grid + np.array([phase + corner[0], corner[1]])) * pitch
        offsets = points[:, None, :] - TRIANGLE[None, :, :]
        whole &= np.all(edges[None, :, 0] * offsets[..., 1]
                        - edges[None, :, 1] * offsets[..., 0] > -1e-12, axis=1)
    return grid[whole]


def _sweep(scale: float, phases: np.ndarray):
    """Every phase at one scale: cells kept, corners, and board rows."""
    pitch = 1.0 / scale
    span = int(np.ceil(scale)) + 2
    side = span + 1
    grid = np.stack(np.meshgrid(np.arange(-1, span), np.arange(-1, span),
                                indexing="xy"), axis=-1).reshape(-1, 2)
    edges = _edges(TRIANGLE)
    whole = np.ones((len(phases), len(grid)), dtype=bool)
    for corner in ((0, 0), (1, 0), (1, 1), (0, 1)):
        shift = np.stack([phases + corner[0],
                          np.full_like(phases, float(corner[1]))], axis=-1)
        points = (grid[None] + shift[:, None]) * pitch
        offsets = points[:, :, None, :] - TRIANGLE[None, None]
        cross = (edges[None, None, :, 0] * offsets[..., 1]
                 - edges[None, None, :, 1] * offsets[..., 0])
        whole &= np.all(cross > -1e-12, axis=-1)
    occupied = whole.reshape(len(phases), side, side)
    touched = np.zeros((len(phases), side + 1, side + 1), dtype=bool)
    for dy in (0, 1):
        for dx in (0, 1):
            touched[:, dy:dy + side, dx:dx + side] |= occupied
    cells = occupied.sum(axis=(1, 2))
    # The grid starts one row below the face, so a row's index into
    # ``occupied`` is one more than its lattice row, and the highest index is
    # the row count ``cico2`` gives the board.
    rows = np.array([np.nonzero(one.any(axis=1))[0].max() if one.any() else 0
                     for one in occupied])
    return cells, touched.sum(axis=(1, 2)), rows


def markers_needed(n_points: int, cells: int, rows: int) -> int:
    """What ``cico2`` spends on a face of this shape, over all twenty."""
    return FACES * cells + (n_points * rows - cells)


def _widest_run(winning: np.ndarray) -> tuple[int, int]:
    """The longest run of winning phases, as ``(start, length)``, wrapping."""
    count = len(winning)
    if winning.all():
        return 0, count
    best_start = best_length = 0
    start = length = 0
    for index in range(2 * count):
        if winning[index % count]:
            if length == 0:
                start = index % count
            length += 1
            if length > best_length:
                best_start, best_length = start, length
        else:
            length = 0
    return best_start, min(best_length, count)


def solve(n_points: int) -> dict | None:
    """Return the placement a face of ``n_points`` squares is drawn with."""
    scales = np.round(np.arange(n_points - 1.0 + STEP,
                                n_points + STEP / 2, STEP), 6)
    phases = np.round(np.arange(0.0, 1.0, STEP), 6)
    swept = [_sweep(scale, phases) for scale in scales]
    cells = np.stack([one[0] for one in swept])
    corners = np.stack([one[1] for one in swept])
    board_rows = np.stack([one[2] for one in swept])

    affordable = markers_needed(n_points, cells, board_rows) <= BUDGET
    if not affordable.any():
        return None
    best = int(corners[affordable].max())
    winning = affordable & (corners == best)

    # The largest squares that reach it, which is the smallest scale whose
    # band of winning phases is wide enough to be a fit rather than a graze.
    for index, row in enumerate(winning):
        start, length = _widest_run(row)
        if length * STEP < MIN_WINDOW:
            continue
        phase = float(phases[(start + (length - 1) // 2) % len(phases)])
        chosen = np.argmin(np.abs(phases - phase))
        return dict(
            n_points=n_points, scale=float(scales[index]), phase=phase,
            corners=best, cells=int(cells[index, chosen]),
            rows=int(board_rows[index, chosen]),
            markers=int(markers_needed(n_points, cells[index, chosen],
                                       board_rows[index, chosen])),
            window=length * STEP,
        )
    return None


def main() -> None:
    # Up to the face the budget stops paying for.  Past it a finer lattice
    # cannot be afforded, and the best placement within the budget falls back
    # to the one the face below already has -- the same corners under another
    # name, which is worse than being told the dictionary is too small.
    solved = []
    n_points = MIN_POINTS
    while True:
        found = solve(n_points)
        if found is None:
            break
        if solved and found["corners"] <= solved[-1]["corners"]:
            break
        solved.append(found)
        n_points += 1

    print("#: scale and phase, by squares per edge.  See")
    print("#: setup_scripts/calculate_cico2_lattice.py.")
    print("FACE_LATTICE: dict[int, tuple[float, float]] = {")
    for row in solved:
        flush = len(clip(float(row["n_points"]), 0.0))
        was = int(_sweep(float(row["n_points"]), np.array([0.0]))[1][0])
        print(f"    {row['n_points']}: "
              f"({row['scale']:.3f}, {row['phase']:.3f}),"
              f"  # {row['corners']:>3} corners, was {was:>3};"
              f" {row['cells']} squares of {flush}; {row['markers']} markers")
    print("}")
    print()
    print(f"{'n':>3} {'scale':>7} {'phase':>6} {'corners':>8} {'was':>4} "
          f"{'squares':>8} {'markers':>8} {'window':>7} {'bigger':>7}")
    for row in solved:
        was = int(_sweep(float(row["n_points"]), np.array([0.0]))[1][0])
        bigger = 100 * (row["n_points"] / row["scale"] - 1)
        print(f"{row['n_points']:>3} {row['scale']:>7.3f} "
              f"{row['phase']:>6.3f} "
              f"{row['corners']:>8} {was:>4} {row['cells']:>8} "
              f"{row['markers']:>8} {row['window']:>7.3f} {bigger:>6.2f}%")


if __name__ == "__main__":
    main()
