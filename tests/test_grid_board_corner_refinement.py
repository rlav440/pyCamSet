"""How a ChArUco2, Ccube2 or CIco2 target's corners are refined, as a detection option."""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from conftest import skip_without_aruco2
from pyCamSet.calibration_targets.core.target_registry import build_target, target_class
from pyCamSet.calibration_targets.markers.aruco2_gridboard import (
    CORNER_REFINEMENTS,
    DEFAULT_CORNER_REFINEMENT,
    refines_with_newton,
)

#: Each grid-board target, a small spec of it, and the module whose
#: ``detect_grid_board_corners`` its ``find_in_image`` calls.
TARGETS = {
    "ChArUco2": ({"type": "ChArUco2", "num_squares_x": 3, "num_squares_y": 3,
                  "square_size": 12.0}, "pyCamSet.calibration_targets.charuco2"),
    "Ccube2": ({"type": "Ccube2", "n_points": 3, "length": 20.0, "border_fraction": 0.2},
               "pyCamSet.calibration_targets.ccube2"),
    "CIco2": ({"type": "CIco2", "length": 100.0, "n_points": 12},
              "pyCamSet.calibration_targets.cico2"),
}


@pytest.mark.parametrize("name", TARGETS)
def test_every_grid_board_offers_both_refinements_with_newton_by_default(name):
    parameter = target_class(name).detector_parameterisation().parameter("corner_refinement")
    assert tuple(choice.value for choice in parameter.choices) == CORNER_REFINEMENTS
    assert parameter.default == DEFAULT_CORNER_REFINEMENT == "newton_saddle"
    assert parameter.settable, "the Phase 1 form shows only settable parameters"


@pytest.mark.parametrize("name", TARGETS)
def test_an_unknown_refinement_is_refused_when_the_target_is_built(name):
    skip_without_aruco2(name)
    spec = dict(TARGETS[name][0], detection_options={"corner_refinement": "monkey_saddle"})
    with pytest.raises(ValueError, match="corner_refinement"):
        build_target(spec)


@pytest.mark.parametrize("name", TARGETS)
@pytest.mark.parametrize("method, newton", [(None, True), ("newton_saddle", True),
                                            ("cornersubpix", False)])
def test_find_in_image_refines_as_the_option_says(name, method, newton, monkeypatch):
    skip_without_aruco2(name)
    spec, module = TARGETS[name]
    if method is not None:
        spec = dict(spec, detection_options={"corner_refinement": method})
    target = build_target(spec)
    assert target.detection_options["corner_refinement"] == (method or "newton_saddle")

    seen = []

    def spy(*args, refine=True, **kwargs):
        seen.append(refine)
        return None, None

    monkeypatch.setattr(importlib.import_module(module), "detect_grid_board_corners", spy)
    target.find_in_image(np.zeros((64, 64), dtype=np.uint8))
    assert seen and set(seen) == {newton}


def test_refines_with_newton_reads_a_missing_option_as_the_default():
    assert refines_with_newton(None) is True
    assert refines_with_newton({}) is True
    assert refines_with_newton({"corner_refinement": "cornersubpix"}) is False
    with pytest.raises(ValueError, match="corner_refinement"):
        refines_with_newton({"corner_refinement": "sector"})


def test_cornersubpix_returns_aruco2s_own_corners():
    """The option's two values are the detector's refine switch, on a real board."""
    pytest.importorskip("aruco2")
    import cv2
    from pyCamSet.calibration_targets.markers.aruco2_gridboard import detect_grid_board_corners

    target = build_target(TARGETS["ChArUco2"][0])
    image, _ = target._render(dpi=600)
    # Blur, then a tone curve, so the two place the corners differently (by
    # about 0.02 px here: the corners stay symmetric, so the gap is small).
    image = cv2.GaussianBlur(image, (0, 0), 2.5).astype(np.float64)
    image = (255.0 * (image / 255.0) ** 0.4).round().astype(np.uint8)
    found = {}
    for method in CORNER_REFINEMENTS:
        board = build_target(dict(TARGETS["ChArUco2"][0],
                                  detection_options={"corner_refinement": method}))
        detection = board.find_in_image(image)
        direct_ids, direct_points = detect_grid_board_corners(
            image, board.grid_size, board._aruco_dict_int, board.square_size,
            refine=method == "newton_saddle")
        assert direct_ids is not None
        found[method] = np.asarray(direct_points)
        assert detection.has_data
    shift = np.abs(found["newton_saddle"] - found["cornersubpix"]).max()
    assert shift > 1e-3, f"the two refinements differ by only {shift:.4f} px"
