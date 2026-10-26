"""
PuzzleBoard detection gives the same answer from concurrent threads as serially.

The upstream ``puzzle_board`` package keeps detection state at module and
class level (the Hessian/corner-checker objects with reused, size-dependent
buffers, and the union-find root ring ``Grid.first_root``).  Calibria's grid
search calls the detector from four threads, and without serialisation those
calls corrupted each other: wrong or empty results on same-size images, a
broadcasting ``ValueError`` across sizes, and one hang.  These tests run real
renders through ``detect_puzzleboard_image`` from a thread pool and compare
every result with a serial reference.
"""
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

pytest.importorskip("puzzle_board")

from pyCamSet.calibration_targets.markers.puzzleboard import detect_puzzleboard_image
from pyCamSet.calibration_targets.puzzleboard import PuzzleBoard

_SUPERSAMPLE = 4


def _render(edge_px: float) -> np.ndarray:
    """A 22 x 15 corner PuzzleBoard at *edge_px* pixels per square, rasterised from its own SVG."""
    target = PuzzleBoard(num_squares_x=22, num_squares_y=15, square_size=10.0,
                         paper_width=260.0, paper_height=190.0)
    scale = edge_px / target.square_size * _SUPERSAMPLE  # supersampled pixels per millimetre
    width = int(round(target.paper_width * edge_px / target.square_size))
    height = int(round(target.paper_height * edge_px / target.square_size))
    big = np.full((height * _SUPERSAMPLE, width * _SUPERSAMPLE), 255, dtype=np.uint8)
    shift = 4  # cv2 fixed-point fraction bits, for sub-pixel shapes
    fixed = float(1 << shift)
    for element in target._svg_document().elements:
        attribs = element.attribs
        colour = 0 if attribs.get("fill") == "black" else 255
        if element.elementname == "rect":
            x0, y0 = float(attribs["x"]) * scale, float(attribs["y"]) * scale
            x1 = x0 + float(attribs["width"]) * scale
            y1 = y0 + float(attribs["height"]) * scale
            quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) * fixed
            cv2.fillPoly(big, [np.round(quad).astype(np.int32)], colour, cv2.LINE_8, shift)
        elif element.elementname == "circle":
            centre = (int(round(float(attribs["cx"]) * scale * fixed)),
                      int(round(float(attribs["cy"]) * scale * fixed)))
            cv2.circle(big, centre, int(round(float(attribs["r"]) * scale * fixed)), colour, -1,
                       cv2.LINE_8, shift)
    return cv2.resize(big, (width, height), interpolation=cv2.INTER_AREA)


def _signature(image):
    """The detection as comparable bytes: positions and coordinates, in detector order."""
    positions, coordinates = detect_puzzleboard_image(image)
    return (np.asarray(positions, dtype=np.int64).tobytes(),
            np.asarray(coordinates, dtype=np.float64).tobytes(),
            len(positions))


@pytest.fixture(scope="module")
def images():
    # Two sizes: same-size images corrupted each other's results, mixed sizes raised.
    return [_render(10.0), _render(12.0), _render(10.0), _render(12.0)]


def test_the_renders_are_detected(images):
    for image in images:
        positions, _ = detect_puzzleboard_image(image)
        assert len(positions) >= 300  # 330 corners printed; a few at the border may be lost


def test_concurrent_calls_match_the_serial_answer(images):
    reference = [_signature(image) for image in images]
    jobs = [i % len(images) for i in range(32)]
    pool = ThreadPoolExecutor(max_workers=4)  # the width of Calibria's grid search
    try:
        # A bounded wait: unserialised, one measured run of this kind never finished.
        results = list(pool.map(lambda i: (i, _signature(images[i])), jobs, timeout=300))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    mismatched = [i for i, signature in results if signature != reference[i]]
    assert mismatched == []
