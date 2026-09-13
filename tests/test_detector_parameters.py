"""
``DetectorParameterisation``: what a detector says it can be told.

Two kinds of test live here.  The first are the contract every detector
keeps, written against the abstract class and a small made-up detector, so
that a new backend can be checked against them.  The second are ChArUco's
own: twenty-one parameters that a form types in and a study sweeps, whose
defaults and choice names OpenCV has to accept.
"""

from __future__ import annotations

import pytest
from cv2 import aruco

from pyCamSet.calibration_targets.aruco2_detection import ARUCO2_DETECTOR
from pyCamSet.calibration_targets.charuco_detection import ARUCO_OPENCV_DETECTOR
from pyCamSet.calibration_targets.detector_parameters import (
    NO_DETECTOR_PARAMETERS,
    Choice,
    DetectorParameter,
    DetectorParameterisation,
    combine,
)

GROUPS = {"CharucoParameters", "DetectorParameters", "RefineParameters"}


class Pair(DetectorParameterisation):
    """A made-up detector: one number to sweep, one name to pick."""

    name = "pair"

    @property
    def parameters(self):
        return (
            DetectorParameter(
                key="width", label="Width", default=4, dtype="int",
                tunable=True, minimum=2, maximum=10, step=1),
            DetectorParameter(
                key="mode", label="Mode", default=0, dtype="int",
                choices=(Choice("fast", 0), Choice("careful", 1))),
        )

    def validate(self, values):
        if values.get("mode") == 1 and values.get("width", 0) < 4:
            return ["careful mode needs a width of at least 4."]
        return []


# ---------------------------------------------------------------------------
# The contract every detector keeps
# ---------------------------------------------------------------------------


def test_a_detector_that_takes_nothing_still_works():
    """The case a target that is never detected lands in."""
    assert len(NO_DETECTOR_PARAMETERS) == 0
    assert NO_DETECTOR_PARAMETERS.resolve({"anything": 1}) == {}
    assert NO_DETECTOR_PARAMETERS.parse({}) == {}
    assert NO_DETECTOR_PARAMETERS.unavailable_reason() is None


def test_resolving_fills_in_defaults_and_coerces_what_it_is_given():
    detector = Pair()
    assert detector.resolve(None) == {"width": 4, "mode": 0}
    assert detector.resolve({"width": 99})["width"] == 10, "clamped to its maximum"
    assert detector.resolve({"width": 3.7})["width"] == 4, "cast to its dtype"


def test_resolving_drops_a_key_the_detector_does_not_take():
    """A target rebuilt from a saved spec should not fail on a renamed key."""
    assert "gone" not in Pair().resolve({"gone": 1, "width": 5})


def test_a_tunable_parameter_must_say_what_to_search_between():
    with pytest.raises(ValueError, match="needs both a minimum and a maximum"):
        DetectorParameter(key="k", label="K", default=1, dtype="int", tunable=True)


def test_a_study_is_refused_a_parameter_with_no_bounds():
    """Real, settable, and still not something a study can sample."""
    errors = Pair().validate_rows([{"key": "mode", "fixed": 0, "optimise": True}])
    assert errors and "no bounds" in errors[0]

    errors = Pair().validate_rows([{"key": "nope", "fixed": 0, "optimise": False}])
    assert errors and "Unknown parameter" in errors[0]


def test_a_study_is_refused_bounds_outside_the_parameters_own():
    errors = Pair().validate_rows(
        [{"key": "width", "fixed": 4, "optimise": True, "lower": 1, "upper": 50}])
    assert errors and "exceed allowed range" in errors[0]


def test_rules_spanning_two_parameters_are_the_detectors_own():
    assert Pair().validate({"width": 4, "mode": 1}) == []
    assert Pair().validate({"width": 2, "mode": 1}) == ["careful mode needs a width of at least 4."]
    with pytest.raises(ValueError, match="careful mode"):
        Pair().parse({"width": 2, "mode": "careful"})


def test_composing_a_target_with_its_backend_reads_as_one_detector():
    """A target's own settings beside the backend's, with one lookup."""
    both = combine(Pair(), ARUCO_OPENCV_DETECTOR)
    assert len(both) == len(Pair()) + len(ARUCO_OPENCV_DETECTOR)
    assert both.parameter("width").key == "width"
    assert both.parameter("minMarkers").key == "minMarkers"
    assert both.validate({"width": 2, "mode": 1}), "each part still checks its own"


def test_composing_with_nothing_gives_back_the_other_part():
    """Which is why a target that adds nothing of its own costs nothing."""
    assert combine(NO_DETECTOR_PARAMETERS, ARUCO_OPENCV_DETECTOR) is ARUCO_OPENCV_DETECTOR
    assert combine(NO_DETECTOR_PARAMETERS, ARUCO2_DETECTOR) is NO_DETECTOR_PARAMETERS


def test_two_parts_cannot_describe_the_same_parameter():
    with pytest.raises(ValueError, match="more than one"):
        combine(Pair(), Pair())


# ---------------------------------------------------------------------------
# The aruco2 backend
# ---------------------------------------------------------------------------


def test_aruco2_takes_no_settings_at_all():
    """``detect_fiducial_markers`` is given an image and a dictionary.

    Previously a target read this way accepted OpenCV's settings, warned
    they were ignored, and ignored them.
    """
    assert len(ARUCO2_DETECTOR) == 0
    assert ARUCO2_DETECTOR.resolve({"adaptiveThreshWinSizeMin": 9}) == {}


# ---------------------------------------------------------------------------
# ChArUco's own parameters
# ---------------------------------------------------------------------------


def test_every_parameter_is_sweepable_or_typeable_or_both_but_never_neither():
    """The two reasons a parameter is described at all."""
    for parameter in ARUCO_OPENCV_DETECTOR.parameters:
        assert parameter.tunable or parameter.settable, parameter.key

    assert len(ARUCO_OPENCV_DETECTOR) == 21
    assert len(ARUCO_OPENCV_DETECTOR.tunable()) == 16
    assert len(ARUCO_OPENCV_DETECTOR.settable()) == 17
    overlap = ({p.key for p in ARUCO_OPENCV_DETECTOR.tunable()}
               & {p.key for p in ARUCO_OPENCV_DETECTOR.settable()})
    assert len(overlap) == 12


def test_a_tunable_parameter_carries_bounds_its_default_sits_inside():
    """A study samples between these, so a default outside them is a trap."""
    for parameter in ARUCO_OPENCV_DETECTOR.tunable():
        assert parameter.dtype in {"int", "float"}
        low, high = parameter.cast(parameter.minimum), parameter.cast(parameter.maximum)
        assert low <= high, parameter.key
        assert low <= parameter.cast(parameter.default) <= high, parameter.key


def test_every_default_survives_being_typed_back_in():
    """The form starts at these, so they must be values the form accepts.

    A blank default means the parameter is left unset, and reads back as
    None rather than as itself -- which is what ``drop_if_none`` says.
    """
    for parameter in ARUCO_OPENCV_DETECTOR.settable():
        shown = (parameter.label_for(parameter.default) if parameter.choices
                 else parameter.default)
        parsed = parameter.parse(shown)
        if parameter.drop_if_none and parameter.default == "":
            assert parsed is None, parameter.key
        else:
            assert parsed == parameter.default, parameter.key


def test_the_choice_names_are_opencvs_own():
    """Two tables transcribed these by hand and disagreed: one said
    ``REFINE_SUBPIX`` where the constant is ``CORNER_REFINE_SUBPIX``."""
    parameter = ARUCO_OPENCV_DETECTOR.parameter("cornerRefinementMethod")
    for choice in parameter.choices:
        assert getattr(aruco, choice.label) == choice.value
    assert parameter.choice_labels()[0] == "CORNER_REFINE_NONE"
    assert parameter.default == aruco.CORNER_REFINE_NONE


def test_a_choice_reads_back_from_either_its_name_or_its_value():
    """The form offers names; a study samples numbers; OpenCV takes numbers."""
    parameter = ARUCO_OPENCV_DETECTOR.parameter("cornerRefinementMethod")
    assert parameter.parse("CORNER_REFINE_SUBPIX") == 1
    assert parameter.parse(1) == 1
    with pytest.raises(ValueError, match="CORNER_REFINE_NONE"):
        parameter.parse("REFINE_SUBPIX")


def test_a_window_size_opencv_needs_odd_is_rounded_up_to_odd():
    parameter = ARUCO_OPENCV_DETECTOR.parameter("adaptiveThreshWinSizeMin")
    assert parameter.odd and parameter.coerce(4) == 5


def test_a_typed_value_outside_its_bounds_is_refused_not_clamped():
    parameter = ARUCO_OPENCV_DETECTOR.parameter("adaptiveThreshWinSizeMin")
    with pytest.raises(ValueError, match="between 3 and 99"):
        parameter.parse(1)


def test_reading_a_form_produces_options_opencv_accepts():
    """The whole point of the parse rules: what comes out is applied."""
    typed = {p.key: (p.label_for(p.default) if p.choices else p.default)
             for p in ARUCO_OPENCV_DETECTOR.settable()}
    typed["cornerRefinementMethod"] = "CORNER_REFINE_SUBPIX"
    typed["adaptiveThreshWinSizeMin"] = 5

    values = ARUCO_OPENCV_DETECTOR.parse(typed)

    assert set(ARUCO_OPENCV_DETECTOR.grouped(values)) <= GROUPS
    assert "cameraMatrix" not in values, "an unset optional field is dropped"

    charuco, detector, _refine = ARUCO_OPENCV_DETECTOR.build_parameters(
        ARUCO_OPENCV_DETECTOR.resolve(values))
    assert detector.cornerRefinementMethod == aruco.CORNER_REFINE_SUBPIX
    assert detector.adaptiveThreshWinSizeMin == 5
    assert charuco.tryRefineMarkers is True


def test_the_one_rule_opencv_holds_between_two_of_its_parameters():
    assert ARUCO_OPENCV_DETECTOR.validate(
        {"adaptiveThreshWinSizeMin": 3, "adaptiveThreshWinSizeMax": 23}) == []
    with pytest.raises(ValueError, match="adaptiveThreshWinSizeMax"):
        typed = {p.key: p.default for p in ARUCO_OPENCV_DETECTOR.settable()}
        typed["adaptiveThreshWinSizeMin"] = 23
        typed["adaptiveThreshWinSizeMax"] = 5
        ARUCO_OPENCV_DETECTOR.parse(typed)


def test_a_detector_built_from_defaults_is_the_one_detection_uses():
    board = aruco.CharucoBoard((5, 5), 0.004, 0.0032,
                               aruco.getPredefinedDictionary(aruco.DICT_4X4_1000))
    assert ARUCO_OPENCV_DETECTOR.build_detector(
        board, ARUCO_OPENCV_DETECTOR.resolve(None)) is not None
