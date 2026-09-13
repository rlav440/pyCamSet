"""One table of ChArUco detector parameters, for both the ways they are set.

There were two, describing the same OpenCV fields for different reasons: one
gave each parameter bounds so a study could search over it, the other gave it
rules for reading a value someone typed.  Twelve parameters were in both, by
hand, with nothing keeping them in step.

These tests are about the properties that made merging them safe, and that
would have to stay true of any row added later.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from cv2 import aruco

from pyCamSet.calibration_targets import charuco_parameters as params
from pyCamSet.calibration_targets.charuco_detection import (
    build_charuco_detector_components,
)

GROUPS = {"DetectorParameters", "CharucoParameters", "RefineParameters"}


def test_the_table_ships_with_the_package():
    """It is JSON beside the module, so it has to be in the wheel."""
    path = Path(params.__file__).with_name("charuco_parameters.json")
    assert path.exists()
    assert json.loads(path.read_text()) == params.CHARUCO_PARAMETERS
    assert len(params.CHARUCO_PARAMETERS) == 21


def test_every_row_says_what_it_is_and_where_it_goes():
    for row in params.CHARUCO_PARAMETERS:
        assert row["group"] in GROUPS, row["key"]
        assert "." not in row["key"], "keys are bare; the group is its own field"
        assert row["label"] and row["concept"]
        assert "default" in row


def test_a_row_is_searchable_or_typeable_or_both_but_never_neither():
    """The two reasons a parameter is in the table at all."""
    for row in params.CHARUCO_PARAMETERS:
        searchable = "min" in row and "max" in row
        typeable = "parser_type" in row
        assert searchable or typeable, row["key"]

    assert len(params.searchable()) == 16
    assert len(params.typeable()) == 17
    overlap = {r["key"] for r in params.searchable()} & {
        r["key"] for r in params.typeable()}
    assert len(overlap) == 12


def test_a_searchable_row_carries_bounds_its_default_sits_inside():
    """A study samples between these, so a default outside them is a trap."""
    for row in params.searchable():
        assert row["dtype"] in {"int", "float"}
        low, high = params.coerce_value(row, row["min"]), params.coerce_value(row, row["max"])
        assert low <= high, row["key"]
        assert low <= params.coerce_value(row, row["default"]) <= high, row["key"]


def test_every_default_survives_being_typed_back_in():
    """The form starts at these, so they must be values the form accepts.

    A blank default means the parameter is left unset, and reads back as
    None rather than as itself -- which is what ``drop_if_none`` says.
    """
    for row in params.typeable():
        shown = params.label_for(row, row["default"]) if "choices" in row else row["default"]
        parsed = params.parse_value(row, shown)
        if row.get("drop_if_none") and row["default"] == "":
            assert parsed is None, row["key"]
        else:
            assert parsed == row["default"], row["key"]


def test_the_choice_names_are_opencvs_own():
    """Both old tables transcribed these by hand, and disagreed: one said
    ``REFINE_SUBPIX`` where the constant is ``CORNER_REFINE_SUBPIX``."""
    row = params.by_key()["cornerRefinementMethod"]
    for choice in row["choices"]:
        assert getattr(aruco, choice["label"]) == choice["value"]
    assert params.choice_labels(row)[0] == "CORNER_REFINE_NONE"
    assert row["default"] == aruco.CORNER_REFINE_NONE


def test_a_choice_reads_back_from_either_its_name_or_its_value():
    """The form offers names; a study samples numbers; OpenCV takes numbers."""
    row = params.by_key()["cornerRefinementMethod"]
    assert params.parse_value(row, "CORNER_REFINE_SUBPIX") == 1
    assert params.parse_value(row, 1) == 1
    with pytest.raises(ValueError, match="CORNER_REFINE_NONE"):
        params.parse_value(row, "REFINE_SUBPIX")


def test_reading_a_form_produces_options_opencv_accepts():
    """The whole point of the parse rules: what comes out is applied."""
    typed = {row["key"]: (params.label_for(row, row["default"]) if "choices" in row
                          else row["default"])
             for row in params.typeable()}
    typed["cornerRefinementMethod"] = "CORNER_REFINE_SUBPIX"
    typed["adaptiveThreshWinSizeMin"] = 5

    options = params.collect_detection_options(typed)

    assert set(options) <= GROUPS
    _charuco, detector, _refine = build_charuco_detector_components(options)
    assert detector.cornerRefinementMethod == aruco.CORNER_REFINE_SUBPIX
    assert detector.adaptiveThreshWinSizeMin == 5


def test_an_adaptive_window_that_ends_before_it_starts_is_refused():
    typed = {row["key"]: (params.label_for(row, row["default"]) if "choices" in row
                          else row["default"])
             for row in params.typeable()}
    typed["adaptiveThreshWinSizeMin"] = 31
    typed["adaptiveThreshWinSizeMax"] = 7

    with pytest.raises(ValueError, match="adaptiveThreshWinSizeMax"):
        params.collect_detection_options(typed)


def test_the_defaults_a_study_starts_from_are_grouped_for_opencv():
    settings = params.default_fixed_settings()

    assert set(settings) <= GROUPS
    flat = {k: v for group in settings.values() for k, v in group.items()}
    assert flat == {row["key"]: row["default"] for row in params.searchable()}
    build_charuco_detector_components(settings)  # must not raise


def test_a_parameter_this_opencv_has_never_heard_of_is_dropped():
    """``build_charuco_parameters`` ignores unknown fields, and so does this,
    so a table entry from a newer OpenCV does not break an older one."""
    grouped = params.assemble_detection_options(
        {"minMarkerPerimeterRate": 0.05, "somethingFromTheFuture": 1})

    assert grouped == {"DetectorParameters": {"minMarkerPerimeterRate": 0.05}}


def test_a_study_sees_its_rows_in_its_own_order():
    """The table is stored in form order, so the form's priority groups stay
    contiguous; a study's rows are ordered by ``search_order`` instead."""
    order = [row["search_order"] for row in params.searchable()]
    assert order == sorted(order)
    assert params.numeric_keys()[0] == "adaptiveThreshWinSizeMin"
    assert params.typeable()[0]["key"] == "minMarkerPerimeterRate"


def test_a_row_a_study_cannot_use_is_named_as_such():
    known = params.by_key()
    assert known["cameraMatrix"]["value_type"] == "json_matrix"
    assert "min" not in known["cameraMatrix"]

    errors = params.validate_all_rows([
        {"key": "cameraMatrix", "fixed": 0, "optimise": False},
        {"key": "notAParameter", "fixed": 0, "optimise": False},
    ])
    # A real parameter with no bounds, and a key that is not a parameter at
    # all, are different mistakes and say so differently.
    assert any("cameraMatrix" in e and "no bounds" in e for e in errors)
    assert any("notAParameter" in e and "Unknown" in e for e in errors)


def test_bounds_a_study_could_not_sample_between_are_reported():
    errors = params.validate_all_rows([
        {"key": "adaptiveThreshWinSizeMin", "fixed": 3,
         "optimise": True, "lower": 30, "upper": 5},
    ])
    assert errors
