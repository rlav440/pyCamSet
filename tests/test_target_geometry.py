"""A target's object points must sit on the features its detector reports.

Everything downstream trusts ``point_data`` as ground truth, so an error in it
is not an error the solver can see: the bundle adjustment fits the model it is
given and reports a small residual against the wrong geometry.  For the puzzle
targets the trap is a specific one.  Their checkerboard squares are centred on
integer code positions and span half a square each way, so the corner the
detector decodes for code (row, column) lies half a square down and right of
that integer position.  Taking the integer position instead puts every point at
a square centre.

On a flat board that is a translation of the whole lattice, which the target
pose absorbs.  On the cube it is applied in six different face frames, so it is
not a rigid transform of anything -- on a 200 mm cube it moves the modelled
geometry by 9 mm rms against the printed one.  These tests read the positions
off the rasterised pattern rather than restating the arithmetic, which is the
only way for them to disagree with the code they check.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest


def _rasterise(svg: str, width_mm: float, height_mm: float, scale: float) -> np.ndarray:
    """Render an SVG at `scale` pixels per millimetre, skipping without cairo.

    The scale has to be the same in both axes or the squares stop being square
    and nothing below means anything.  cairosvg raises OSError, not ImportError,
    when the native library it binds to is absent, so importorskip misses it.
    """
    try:
        import cairosvg
    except (ImportError, OSError) as err:
        pytest.skip(f"native cairo is unavailable: {err}")
    from PIL import Image

    png = cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        output_width=int(width_mm * scale),
        output_height=int(height_mm * scale),
    )
    with Image.open(BytesIO(png)) as image:
        return np.asarray(image.convert("L")).astype(float)


def _is_a_four_square_corner(pattern: np.ndarray, x: float, y: float, probe: int) -> bool:
    """True when the four quadrants around (x, y) alternate black and white."""
    column, row = int(round(x)), int(round(y))
    quadrants = [pattern[row + dy * probe, column + dx * probe] > 127
                 for dy in (-1, 1) for dx in (-1, 1)]
    return sum(quadrants) == 2 and quadrants[0] != quadrants[1] and quadrants[0] == quadrants[3]


def test_a_cube_face_carries_its_points_on_the_printed_corners():
    """Read the pattern the face is printed with, and look at each point."""
    from pyCamSet import PuzzleBoardCube

    target = PuzzleBoardCube(n_points=10, length=200.0)
    side_mm = target.face_length * 1000.0
    scale = 2000 / side_mm  # pixels per millimetre
    pattern = _rasterise(target._face_svg(0), side_mm, side_mm, scale)
    resolution = pattern.shape[0]
    probe = int(target.square_size * scale * 0.25)  # a quarter square out, inside one quadrant

    checked = 0
    for x_m, y_m, _ in target.faceData.face_local_coords[0]:
        x, y = x_m * 1000.0 * scale, y_m * 1000.0 * scale
        if not (probe < x < resolution - probe and probe < y < resolution - probe):
            continue  # the face's outer ring of points has no four-square neighbourhood
        assert _is_a_four_square_corner(pattern, x, y, probe), (
            f"point at ({x_m * 1000:.1f}, {y_m * 1000:.1f}) mm is not on a corner")
        checked += 1
    assert checked > 50, "too few interior points were checked for this to mean anything"


def test_the_cube_lattice_is_centred_on_its_face():
    """The same statement without cairo: a centred lattice cannot be off by half.

    An off-by-half lattice touches one edge of the face and stops a full square
    short of the other, which this catches wherever the rasteriser does not run.
    """
    from pyCamSet import PuzzleBoardCube

    target = PuzzleBoardCube(n_points=10, length=200.0)
    face = target.faceData.face_local_coords[0][:, :2]
    pitch = target.square_size / 1000.0

    assert np.allclose(face.min(axis=0), pitch / 2)
    assert np.allclose(face.max(axis=0), target.face_length - pitch / 2)


def test_a_flat_board_carries_its_points_on_the_printed_corners():
    """The same check for the planar board, against the page it prints."""
    from pyCamSet import PuzzleBoard

    target = PuzzleBoard(num_squares_x=10, num_squares_y=10, square_size=2.0)
    scale = 10.0  # pixels per millimetre, so one 2 mm square is 20 px across
    pattern = _rasterise(
        target._svg_document().tostring(), target.paper_width, target.paper_height, scale)
    off_x, off_y = target._board_offsets()
    probe = int(target.square_size * scale * 0.25)

    points = target.point_data[0]
    checked = 0
    for row in range(1, target.num_squares_y - 1):  # interior only, as above
        for column in range(1, target.num_squares_x - 1):
            key = (target.start_y + row) * 501 + (target.start_x + column)
            x = (off_x + points[key, 0]) * scale
            y = (off_y + points[key, 1]) * scale
            assert _is_a_four_square_corner(pattern, x, y, probe), (
                f"code ({row}, {column}) is not on a corner")
            checked += 1
    assert checked == 64
