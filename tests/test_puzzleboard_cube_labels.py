"""A PuzzleBoardCube face number is readable on every face, at every size."""

from __future__ import annotations

import io
import re

import numpy as np
import pytest

from pyCamSet.calibration_targets.puzzleboard_cube import PuzzleBoardCube

# Just inside the face's corner cell, left of and below where the number is
# drawn (at 2% and 98.5% of the face), in the face's own coordinates.
CORNER_SAMPLE = (0.005, 0.995)


def _render(svg: str, width_px: int) -> np.ndarray:
    """The SVG as a grey image, or a skip where Cairo cannot render here."""
    try:
        import cairosvg
    except (ImportError, OSError) as exc:  # cairocffi raises OSError without native Cairo
        pytest.skip(f"cairosvg cannot render here: {exc}")
    from PIL import Image

    png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=width_px)
    return np.asarray(Image.open(io.BytesIO(png)).convert("L"), dtype=float)


def _colour(image: np.ndarray, x_frac: float, y_frac: float) -> str:
    height, width = image.shape
    value = image[min(int(height * y_frac), height - 1), min(int(width * x_frac), width - 1)]
    return "black" if value < 128 else "white"


def _label_fills(svg: str) -> list[str]:
    """The fill of each face-number text element, in face order."""
    fills = re.findall(r'<text[^>]*fill="(\w+)"', svg)
    assert fills, "no numbered label was drawn"
    return fills


@pytest.mark.parametrize("n_points", [5, 6])
def test_every_face_texture_label_contrasts_with_its_corner(n_points):
    """An odd square count flips some corner cells to white; the label follows."""
    cube = PuzzleBoardCube(n_points=n_points, length=60.0)
    for face in range(6):
        svg = cube._face_svg(face)
        corner = _colour(_render(svg, 400), *CORNER_SAMPLE)
        assert _label_fills(svg)[0] != corner, (
            f"face {face + 1} of a {n_points}-square cube draws its number in {corner} "
            f"on a {corner} corner")


@pytest.mark.parametrize("n_points", [5, 6])
def test_every_printed_net_label_contrasts_with_its_corner(n_points):
    """The same holds on the printable net, where the faces are rotated into place."""
    cube = PuzzleBoardCube(n_points=n_points, length=60.0)
    drawing, width_mm, height_mm = cube._svg_document(draw_face_ids=True)
    svg = drawing.tostring()
    image = _render(svg, 1600)
    _, _, offset_m = cube._net_bounds(10.0)
    fills = _label_fills(svg)
    for face in range(6):
        local = np.array([[CORNER_SAMPLE[0] * cube.face_length,
                           CORNER_SAMPLE[1] * cube.face_length]])
        point_mm = cube._apply_affine(local, cube._net_affine_for_face(face))[0] * 1000.0 \
            + offset_m * 1000.0
        corner = _colour(image, point_mm[0] / width_mm, point_mm[1] / height_mm)
        assert fills[face] != corner, (
            f"printed face {face + 1} of a {n_points}-square cube draws its number in "
            f"{corner} on a {corner} corner")
