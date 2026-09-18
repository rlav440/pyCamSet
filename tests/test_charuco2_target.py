"""ChArUco2 (aruco2's GridBoard) target regressions.

Gates itself on the optional ``aruco2`` package, matching
``test_aruco2_backend.py``'s own pattern (see that module and
``tests/conftest.py``'s "Optional-environment probes" note): a module whose
only prerequisite is an optional package skips itself with
``pytest.importorskip`` before its other imports, rather than being dropped
from collection, so a missing dependency shows up as one reported skip
instead of thirty tests silently never running.
"""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path

import numpy as np
import pytest

aruco2 = pytest.importorskip("aruco2")

from pyCamSet.calibration_targets.charuco2.target import ChArUco2
from pyCamSet.calibration_targets.core.abstract_target import EXPORT_KINDS
from pyCamSet.calibration_targets.core.target_registry import (
    TARGET_NAMES, build_target, spec_of,
)
from pyCamSet.calibration_targets.markers.aruco2_gridboard import (
    render_grid_board_image,
)


def _corner_positions_mm(target: ChArUco2) -> np.ndarray:
    """``point_data`` as (n, 2) millimetres, flattened out of its face axis."""
    return target.point_data.reshape(-1, 3)[:, :2] * 1000.0


def test_charuco2_is_registered() -> None:
    assert "ChArUco2" in TARGET_NAMES


def test_charuco2_offers_only_the_aruco2_backend() -> None:
    assert list(ChArUco2.DETECTOR_BACKENDS) == ["aruco2"]


def test_charuco2_excludes_apriltag_dictionaries() -> None:
    choices = ChArUco2.construction_parameters().parameter("a_dict").choices
    values = [c.value for c in choices]
    assert values, "ChArUco2 must offer at least one dictionary"
    assert not any(v.startswith("DICT_APRILTAG_") for v in values)


def test_charuco2_detector_takes_no_parameters() -> None:
    """aruco2.detect_grid_board takes no DetectionParameters -- there must
    be nothing here for a form to show or a study to sweep."""
    detector = ChArUco2.DETECTOR_BACKENDS["aruco2"]
    assert detector.parameters == ()


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


def test_charuco2_rebuilds_from_what_it_recorded() -> None:
    spec = {"type": "ChArUco2", "num_squares_x": 5, "num_squares_y": 6,
            "square_size": 12.5}
    built = build_target(spec)
    again = build_target(spec_of(built))
    assert type(again) is type(built)
    assert again.input_args == built.input_args
    assert (again.point_data == built.point_data).all()


@pytest.mark.parametrize("kind", EXPORT_KINDS)
def test_charuco2_save_printable_all_kinds(tmp_path: Path, kind: str) -> None:
    target = ChArUco2(num_squares_x=5, num_squares_y=5, square_size=10.0)
    out = target.save_printable(tmp_path / f"board.{kind}", kind=kind)
    assert out.exists()
    assert out.stat().st_size > 1024


def test_charuco2_svg_is_true_millimetre_scale(tmp_path: Path) -> None:
    target = ChArUco2(num_squares_x=4, num_squares_y=3, square_size=15.0)
    svg_path = target.save_to_svg(tmp_path / "board.svg", border_width=10.0, dpi=300)
    text = svg_path.read_text(encoding="utf-8")
    w_mm = float(re.search(r'width="([\d.]+)mm"', text).group(1))
    h_mm = float(re.search(r'height="([\d.]+)mm"', text).group(1))

    image, px_per_mm = target._render(300)
    expected_w = image.shape[1] / px_per_mm + 2 * 10.0
    expected_h = image.shape[0] / px_per_mm + 2 * 10.0
    assert w_mm == pytest.approx(expected_w, abs=1e-3)
    assert h_mm == pytest.approx(expected_h, abs=1e-3)


def test_charuco2_render_matches_aruco2_get_grid_board_image() -> None:
    """The printable/detectable raster is exactly aruco2's own board image,
    not a re-implementation of its layout."""
    target = ChArUco2(num_squares_x=5, num_squares_y=5, square_size=10.0)
    image, px_per_mm = target._render(300)

    marker_bits = target._marker_bits()
    bit_size = max(1, round((300 / 25.4) * 10.0 / marker_bits))
    direct = aruco2.get_grid_board_image((5, 5), target._aruco_dict_int, bit_size)

    assert image.shape == direct.shape
    assert np.array_equal(image, direct)
    assert px_per_mm == pytest.approx((marker_bits * bit_size) / 10.0)


def test_charuco2_detects_its_own_rendered_board() -> None:
    """The fast path: detect straight from the raster aruco2 renders, no
    SVG round trip. Every corner must be found, at its own recorded
    position, exactly."""
    target = ChArUco2(num_squares_x=5, num_squares_y=5, square_size=10.0)
    image, px_per_mm = target._render(300)

    detection = target.find_in_image(image)
    assert detection.has_data
    n_corners = target.point_data.shape[-2]
    assert len(detection.keys) == n_corners
    assert set(detection.keys.tolist()) == set(range(n_corners))

    marker_bits = target._marker_bits()
    bit_size = max(1, round((300 / 25.4) * 10.0 / marker_bits))
    border_native_mm = ((marker_bits * bit_size) // 4) / px_per_mm
    expected_px = (_corner_positions_mm(target) + border_native_mm) * px_per_mm

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


def test_charuco2_svg_round_trip_is_detectable(tmp_path: Path) -> None:
    """Priority 1's own check: rasterise the printable SVG at a known
    px/mm and confirm aruco2 detects it back to the same corners, at the
    positions the SVG's own real-world mm scale implies -- not just that
    the in-memory raster (already covered above) is correct."""
    pyCamSet_utils_cairo = pytest.importorskip(
        "pyCamSet.utils.cairo_dll_helper")
    cairosvg = pytest.importorskip("cairosvg")

    target = ChArUco2(num_squares_x=5, num_squares_y=5, square_size=10.0)
    svg_path = target.save_to_svg(
        tmp_path / target.printable_name(
            {"num_squares_x": 5, "num_squares_y": 5, "square_size": 10.0}),
        border_width=10.0, dpi=300)
    try:
        text = svg_path.read_text(encoding="utf-8")
        w_mm = float(re.search(r'width="([\d.]+)mm"', text).group(1))
        h_mm = float(re.search(r'height="([\d.]+)mm"', text).group(1))

        px_per_mm = 300 / 25.4
        out_w = int(round(w_mm * px_per_mm))
        out_h = int(round(h_mm * px_per_mm))
        png_bytes = cairosvg.svg2png(
            url=str(svg_path), output_width=out_w, output_height=out_h,
            background_color="white")
        from PIL import Image
        raster = np.array(Image.open(io.BytesIO(png_bytes)).convert("L"))

        detection = target.find_in_image(raster)
        n_corners = target.point_data.shape[-2]
        assert len(detection.keys) == n_corners

        marker_bits = target._marker_bits()
        bit_size = max(1, round(px_per_mm * 10.0 / marker_bits))
        border_native_mm = ((marker_bits * bit_size) // 4) / (
            (marker_bits * bit_size) / 10.0)
        offset_mm = 10.0 + border_native_mm
        expected_px = (_corner_positions_mm(target) + offset_mm) * px_per_mm

        by_id = dict(zip(detection.keys.tolist(), detection.image_points))
        for gid, expected in enumerate(expected_px):
            assert np.linalg.norm(by_id[gid] - expected) < 2.0
    finally:
        svg_path.unlink(missing_ok=True)
