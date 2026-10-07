"""A PuzzleBoardCube face number is readable on every face, at every size."""

from __future__ import annotations

import re

import numpy as np
import pytest
from scipy import ndimage

from conftest import rasterise_svg
from pyCamSet.calibration_targets.puzzleboard_cube import PuzzleBoardCube

_LABEL = re.compile(r'<text class="face-label"[^>]*>[^<]*</text>')


def _glyph_only(svg: str) -> str:
    """The same document with nothing drawn but the face numbers, in black on white."""
    root = re.match(r"(?:<\?xml[^>]*\?>\s*)?<svg[^>]*>", svg).group(0)
    labels = "".join(label.replace('fill="white"', 'fill="black"') for label in _LABEL.findall(svg))
    assert labels, "no numbered label was drawn"
    return f'{root}<rect width="100%" height="100%" fill="white"/>{labels}</svg>'


def _assert_label_contrasts(svg: str, width_mm: float, height_mm: float, scale: float, ring_px: int):
    """The numbers are white inside and every pixel just outside their outline is black."""
    page = rasterise_svg(svg, width_mm, height_mm, scale)
    glyph = rasterise_svg(_glyph_only(svg), width_mm, height_mm, scale) < 128
    rows, columns = np.nonzero(glyph)
    window = (slice(max(rows.min() - 2 * ring_px, 0), rows.max() + 2 * ring_px),
              slice(max(columns.min() - 2 * ring_px, 0), columns.max() + 2 * ring_px))
    page, glyph = page[window], glyph[window]
    inside = ndimage.binary_erosion(glyph, iterations=1)
    ring = ndimage.binary_dilation(glyph, iterations=ring_px) & ~ndimage.binary_dilation(glyph, iterations=1)
    assert page[inside].mean() > 200, "the number is not drawn white"
    assert page[ring].mean() < 55, (
        f"{np.mean(page[ring] > 127):.0%} of the outline around the number is white")


@pytest.mark.parametrize("n_points", [5, 20, 40])
@pytest.mark.parametrize("face", range(6))
def test_every_face_texture_label_contrasts_with_its_surroundings(n_points, face):
    cube = PuzzleBoardCube(n_points=n_points, length=60.0)
    _assert_label_contrasts(cube._face_svg(face), 60.0, 60.0, scale=40.0, ring_px=4)


def test_every_printed_net_label_contrasts_with_its_surroundings():
    cube = PuzzleBoardCube(n_points=20, length=60.0)
    drawing, width_mm, height_mm = cube._svg_document(draw_face_ids=True)
    _assert_label_contrasts(drawing.tostring(), width_mm, height_mm, scale=40.0, ring_px=4)
