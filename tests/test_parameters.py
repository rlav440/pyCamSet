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

from pyCamSet.calibration_targets.markers.aruco2 import ARUCO2_DETECTOR
from pyCamSet.calibration_targets.markers.aruco_opencv import ARUCO_OPENCV_DETECTOR
from pyCamSet.calibration_targets.core.parameters import (
    NO_PARAMETERS,
    Choice,
    Parameter,
    DetectorParameterisation,
    Profile,
    combine,
    parameters_from_docstring,
)

GROUPS = {"CharucoParameters", "DetectorParameters", "RefineParameters"}


class Pair(DetectorParameterisation):
    """A made-up detector: one number to sweep, one name to pick."""

    name = "pair"

    @property
    def parameters(self):
        return (
            Parameter(
                key="width", label="Width", default=4, dtype="int",
                tunable=True, minimum=2, maximum=10, step=1),
            Parameter(
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
    assert len(NO_PARAMETERS) == 0
    assert NO_PARAMETERS.resolve({"anything": 1}) == {}
    assert NO_PARAMETERS.parse({}) == {}
    assert NO_PARAMETERS.unavailable_reason() is None


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
        Parameter(key="k", label="K", default=1, dtype="int", tunable=True)


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
    assert combine(NO_PARAMETERS, ARUCO_OPENCV_DETECTOR) is ARUCO_OPENCV_DETECTOR
    assert combine(NO_PARAMETERS, NO_PARAMETERS) is NO_PARAMETERS


def test_a_detector_with_no_parameters_is_not_nothing():
    """aruco2 takes no settings and still says whether it is installed.

    Dropping it for having no parameters lost that, and a target read with
    a backend that is not installed was accepted right up until detection.
    """
    composed = combine(NO_PARAMETERS, ARUCO2_DETECTOR)
    assert composed is ARUCO2_DETECTOR
    assert composed.unavailable_reason() == ARUCO2_DETECTOR.unavailable_reason()


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


# ---------------------------------------------------------------------------
# The presets a detector offers for its own bounds
# ---------------------------------------------------------------------------


def test_a_detector_offers_only_presets_for_parameters_it_has():
    """A preset narrows a study's search, so a key the detector does not
    take is a bound that can never be applied."""
    for name, profile in ARUCO_OPENCV_DETECTOR.profiles().items():
        keys = set(profile.lower_bounds) | set(profile.upper_bounds)
        unknown = keys - {p.key for p in ARUCO_OPENCV_DETECTOR.parameters}
        assert not unknown, f"{name}: {sorted(unknown)}"
        assert set(profile.recommended_keys) <= keys, name


def test_a_preset_says_nothing_about_a_parameter_it_does_not_cover():
    """Which is how a row keeps the bounds it has."""
    balanced = ARUCO_OPENCV_DETECTOR.profiles()["Balanced"]
    assert balanced.bounds_for("adaptiveThreshWinSizeMax") == (23, 61)
    assert balanced.bounds_for("cameraMatrix") is None


def test_every_preset_sits_inside_the_bounds_its_parameters_allow():
    """A study clamps to the parameter, so a preset outside it is a lie
    about where the search will go."""
    for name, profile in ARUCO_OPENCV_DETECTOR.profiles().items():
        for key in profile.lower_bounds:
            parameter = ARUCO_OPENCV_DETECTOR.parameter(key)
            low, high = profile.bounds_for(key)
            assert parameter.cast(parameter.minimum) <= parameter.cast(low), (name, key)
            assert parameter.cast(high) <= parameter.cast(parameter.maximum), (name, key)
            assert parameter.cast(low) <= parameter.cast(high), (name, key)


def test_a_detector_with_no_presets_says_so():
    """Rather than the selector offering ChArUco's over another detector."""
    assert Pair().profiles() == {}
    assert NO_PARAMETERS.profiles() == {}


def test_a_preset_renders_as_hover_text_over_the_form_labels():
    profile = Profile(
        name="P", description="What it is for.",
        lower_bounds={"width": 2}, upper_bounds={"width": 8},
        recommended_keys=("width",))
    assert profile.tooltip({"width": "Width"}) == (
        "What it is for.\n\n"
        "Recommended parameters to check for optimisation:\n"
        "- Width")
    assert "(none)" in Profile("P", "d").tooltip({})


# ---------------------------------------------------------------------------
# A board OpenCV cannot build
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(0, 0), (1, 1), (1, 5), (5, 1), (2, 2)])
def test_a_board_too_small_for_opencv_is_refused_before_opencv_sees_it(size):
    """OpenCV does not refuse a zero- or one-dimension ChArUco board.

    ``aruco.CharucoBoard((0, 0), ...)`` raises SystemError and leaves the
    aruco module in a state where the *next* board built, or image
    detected, aborts the process -- no exception, no traceback, nothing to
    catch.  A study validates its target by building it, so a board size
    someone typed reaches this constructor directly.
    """
    from pyCamSet.calibration_targets.charuco.target import ChArUco

    with pytest.raises(ValueError, match="at least 2"):
        ChArUco(num_squares_x=size[0], num_squares_y=size[1], square_size=30.0)


@pytest.mark.parametrize("n_points", [0, 1, 2])
def test_a_cube_face_too_small_for_opencv_is_refused_too(n_points):
    """Every Ccube face is a ChArUco board, with the same landmine under it."""
    from pyCamSet.calibration_targets.ccube.target import Ccube

    with pytest.raises(ValueError, match="at least 3x3"):
        Ccube(n_points=n_points, length=20.0)


def test_the_smallest_usable_boards_are_still_built():
    """The guards refuse what breaks, and nothing more.

    A 2x2 ChArUco has one chessboard corner, which squeezes down to a bare
    point and takes ``make_local`` out with an IndexError; 2x3 has two, and
    is the smallest board this class can describe.
    """
    from pyCamSet.calibration_targets.ccube.target import Ccube
    from pyCamSet.calibration_targets.charuco.target import ChArUco

    assert ChArUco(num_squares_x=2, num_squares_y=3,
                   square_size=30.0).point_data.shape == (1, 2, 3)
    assert Ccube(n_points=3, length=20.0).point_data.shape == (6, 4, 3)


# ---------------------------------------------------------------------------
# A parameter is its argument, and what the argument's own docstring says
# ---------------------------------------------------------------------------
#
# Every one of these was written twice: once as the argument a constructor
# takes and documents, and again as a table beside it saying the same name,
# the same default and the same prose in different words.  The two drifted,
# and the form showed whichever half it read.


def documented_example(
    needed,
    width: int = 4,
    ratio: float = 0.5,
    quiet: bool = False,
    mode="fast",
    unnamed: int = 1,
    unmentioned: int = 2,
):
    """
    A made-up thing, with arguments a form could offer.

    :param needed: Needed -- an argument with nothing to start from.
    :param width: Width -- how wide the thing is, in whole
        somethings. Suggested: 4-8.
    :param ratio: Ratio -- how much of the thing is the other thing.
    :param quiet: Quiet -- whether it says anything at all.
    :param mode: Mode -- which way round it goes.
    :param unnamed: an argument that never names itself.
    """


def test_an_argument_says_its_own_name_prose_default_and_dtype():
    """The four things that were written twice, read once from the one
    place a person maintaining the argument is already looking."""
    width, = parameters_from_docstring(documented_example, "width")

    assert width.key == "width"
    assert width.label == "Width"
    assert width.concept == "how wide the thing is, in whole somethings."
    assert width.suggested == "4-8"
    assert width.default == 4
    assert width.dtype == "int"


def test_the_prose_reads_the_same_however_the_docstring_was_wrapped():
    """A description is written to the margin, and read back as a sentence."""
    width, = parameters_from_docstring(documented_example, "width")

    assert "\n" not in width.concept and "  " not in width.concept


def test_what_an_argument_holds_is_its_annotation_or_failing_that_its_default():
    """A target module postpones its annotations, so they arrive as names
    rather than as types, and an unannotated argument still has a value."""
    ratio, quiet, mode = parameters_from_docstring(
        documented_example, "ratio", "quiet", "mode")

    assert (ratio.dtype, quiet.dtype, mode.dtype) == ("float", "bool", "str")


def test_only_the_arguments_a_form_asks_for_are_offered():
    """A constructor takes drawing resolutions and detector settings that
    are nobody's business here; naming an argument is what offers it."""
    offered = parameters_from_docstring(documented_example, "mode", "width")

    assert [p.key for p in offered] == ["mode", "width"], "in the order given"


def test_the_values_an_argument_may_take_are_the_one_thing_named_for_it():
    """Which library reads the markers decides what the alphabets are
    called, and no docstring can say that for both of them."""
    mode, = parameters_from_docstring(
        documented_example, "mode", choices={"mode": ("fast", "careful")})

    assert mode.choice_labels() == ["fast", "careful"]
    assert mode.parse("careful") == "careful"


def test_values_named_for_an_argument_nobody_is_asked_about_are_refused():
    with pytest.raises(ValueError, match="does not offer"):
        parameters_from_docstring(
            documented_example, "width", choices={"mode": ("fast",)})


def test_an_argument_the_callable_does_not_take_is_refused():
    """Which is a control that builds something nobody asked for."""
    with pytest.raises(ValueError, match="takes no argument 'height'"):
        parameters_from_docstring(documented_example, "height")


def test_an_argument_with_nothing_to_start_from_is_refused():
    """A form opens at the default, so there has to be one."""
    with pytest.raises(ValueError, match="must have one"):
        parameters_from_docstring(documented_example, "needed")


def test_an_argument_the_docstring_says_nothing_about_is_refused():
    """The docstring is the only place the prose lives now, so a silent
    argument is a form control with no label and no hover text."""
    with pytest.raises(ValueError, match="must say something"):
        parameters_from_docstring(documented_example, "unmentioned")


def test_an_argument_that_does_not_name_itself_is_refused():
    """A form needs something short to put beside the box, which is the
    half of the description before the dash."""
    with pytest.raises(ValueError, match="Squares across"):
        parameters_from_docstring(documented_example, "unnamed")


def test_a_heading_the_tooltip_adds_is_not_written_into_the_prose():
    """Seventeen of OpenCV's twenty-one concepts began 'Concept:' and four
    did not, because the heading belongs to the tooltip, not the text."""
    from pyCamSet.calibration_targets.markers.puzzleboard import (
        PUZZLEBOARD_DETECTOR)

    for detector in (ARUCO_OPENCV_DETECTOR, PUZZLEBOARD_DETECTOR):
        for parameter in detector.parameters:
            assert not parameter.concept.startswith("Concept:"), parameter.key


# ---------------------------------------------------------------------------
# What a target says it is
# ---------------------------------------------------------------------------
#
# A target's constructor arguments were written into each interface, as a map
# of which widget each target reads.  There were five such maps, subtly
# different, and a target none of them named had no form at all.


def _targets():
    from pyCamSet.calibration_targets.core.target_registry import TARGET_NAMES, target_class

    return [(name, target_class(name)) for name in TARGET_NAMES]


@pytest.mark.parametrize("name,cls", _targets(), ids=[n for n, _ in _targets()])
def test_every_declared_argument_is_one_the_constructor_takes(name, cls):
    """The declaration is what a form builds itself from, so an argument
    that has drifted from the constructor is a control that builds a target
    nobody asked for -- or a TypeError."""
    import inspect

    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    declared = {p.key for p in cls.construction_parameters().parameters}
    assert declared <= accepted, sorted(declared - accepted)


@pytest.mark.parametrize("name,cls", _targets(), ids=[n for n, _ in _targets()])
def test_every_declared_default_is_the_constructors_own(name, cls):
    """A form starts at these, so a default that disagrees with the
    constructor quietly builds a different target than the one left alone."""
    import inspect

    signature = inspect.signature(cls.__init__).parameters
    for parameter in cls.construction_parameters().parameters:
        expected = signature[parameter.key].default
        assert expected is not inspect.Parameter.empty, parameter.key
        assert parameter.default == expected, parameter.key
        assert type(parameter.default) is type(expected), parameter.key


@pytest.mark.parametrize("name,cls", _targets(), ids=[n for n, _ in _targets()])
def test_a_target_is_described_by_its_own_docstrings(name, cls):
    """A new target gets a form by documenting its constructor, and cannot
    get one by writing the same things out a second time beside it."""
    from pyCamSet.calibration_targets.core.parameters import DocumentedParameters

    assert isinstance(cls.construction_parameters(), DocumentedParameters)
    assert isinstance(cls.export_parameters(), DocumentedParameters)


@pytest.mark.parametrize("name,cls", _targets(), ids=[n for n, _ in _targets()])
def test_what_a_form_says_about_an_argument_is_what_the_target_says(name, cls):
    """Verbatim, so that the sentence a person reads while typing a value
    in is the sentence maintained beside the code that uses it."""
    import inspect

    for source, parameterisation in (
            (cls.__init__, cls.construction_parameters()),
            (cls.save_printable, cls.export_parameters())):
        documented = " ".join((inspect.getdoc(source) or "").split())
        for parameter in parameterisation.parameters:
            assert parameter.label, parameter.key
            assert parameter.concept, parameter.key
            assert f"{parameter.label} -- {parameter.concept}" in documented, \
                parameter.key
            if parameter.suggested:
                assert f"Suggested: {parameter.suggested}" in documented, \
                    parameter.key


@pytest.mark.parametrize("name,cls", _targets(), ids=[n for n, _ in _targets()])
def test_a_target_builds_from_the_arguments_it_declares(name, cls):
    """The whole point: a form that knows nothing about this target can
    still collect what it needs to build one."""
    from pyCamSet.calibration_targets.core.target_registry import build_target

    spec = {"type": name, **cls.construction_parameters().defaults()}
    target = build_target(spec)

    assert type(target) is cls
    assert target.point_data is not None and target.point_data.size


@pytest.mark.parametrize("name,cls", _targets(), ids=[n for n, _ in _targets()])
def test_how_big_a_target_may_be_is_the_targets_own_business(name, cls):
    """A target is an object someone made, and the interface has no say in
    how large.  Each of these once carried a range invented for the spin
    box that showed it: a board of at most 100 squares, a cube of at most
    200mm.  A form built from them clamped a real target to a made-up
    ceiling, silently, and the ceilings did not even hold -- a Ccube was
    capped at 40 squares a side, which no dictionary has the markers
    for."""
    for parameterisation in (cls.construction_parameters(),
                             cls.export_parameters()):
        for parameter in parameterisation.parameters:
            assert parameter.minimum is None, parameter.key
            assert parameter.maximum is None, parameter.key
            assert parameter.step is None, parameter.key
            assert parameter.decimals is None, parameter.key


def test_a_target_larger_than_the_form_used_to_allow_is_built():
    """The ceiling was 100 squares a side, and the board above it is fine.

    How large a target is is the printer's business: what an interface
    offered was never a limit, only the range invented for the spin box
    that showed it."""
    from pyCamSet.calibration_targets.charuco.target import ChArUco
    from pyCamSet.calibration_targets.puzzleboard_cube.target import PuzzleBoardCube

    assert ChArUco(num_squares_x=150, num_squares_y=150,
                   square_size=4.0).point_data.shape == (1, 149 * 149, 3)
    assert PuzzleBoardCube(length=5000.0).point_data.shape[0] == 6


@pytest.mark.parametrize("values,refused", [
    ({"square_size": 0.0}, "has a size"),
    ({"square_size": -4.0}, "has a size"),
    ({"marker_fraction": 0.0}, "fills some of its square"),
    ({"marker_fraction": 1.5}, "no more than all of it"),
])
def test_a_board_that_is_not_a_board_is_refused_by_the_board(values, refused):
    """Not a range a spin box was given: each of these is a board OpenCV
    raises out of with an exception still set, which aborts the next call
    to build one.  The target is what knows that, so the target says so."""
    from pyCamSet.calibration_targets.charuco.target import ChArUco

    with pytest.raises(ValueError, match=refused):
        ChArUco(**values)


@pytest.mark.parametrize("values,refused", [
    ({"length": 0.0}, "has an edge length"),
    ({"border_fraction": 0.0}, "neither all of it nor none"),
    ({"border_fraction": 1.0}, "neither all of it nor none"),
])
def test_a_cube_that_is_not_a_cube_is_refused_by_the_cube(values, refused):
    from pyCamSet.calibration_targets.ccube.target import Ccube

    with pytest.raises(ValueError, match=refused):
        Ccube(**values)


def test_the_only_ceiling_a_cube_has_is_the_one_its_alphabet_gives_it():
    """Six faces are cut from one marker dictionary, and it ends.  The
    invented cap said 40 squares a side, which no dictionary has the
    markers for; the real limit under the default alphabet is 18, and a
    larger alphabet raises it."""
    from pyCamSet.calibration_targets.ccube.target import Ccube

    assert Ccube(n_points=18).point_data.shape == (6, 17 * 17, 3)
    with pytest.raises(ValueError, match="needs 1200 markers, six faces of 200"):
        Ccube(n_points=20)


def test_what_a_target_may_be_is_the_targets_own_to_refuse():
    """A form collects arguments and a target decides whether they make
    one.  The alternative was a second statement of each rule, beside the
    controls that offered it, checked before the target ever saw them."""
    from pyCamSet.calibration_targets.charuco.target import ChArUco
    from pyCamSet.calibration_targets.puzzleboard.target import PuzzleBoard

    assert ChArUco.construction_parameters().validate(
        {"num_squares_x": 2, "num_squares_y": 2}) == [], "nothing to restate"

    with pytest.raises(ValueError, match="chessboard corners"):
        ChArUco(num_squares_x=2, num_squares_y=2, square_size=30.0)
    with pytest.raises(ValueError, match="must not exceed 501"):
        PuzzleBoard(num_squares_x=500, num_squares_y=10, start_x=100)


def test_the_dictionary_a_marker_target_offers_depends_on_its_backend():
    """aruco2 has two dictionaries OpenCV does not, so the names a form
    offers are the selected backend's."""
    from pyCamSet.calibration_targets.charuco.target import ChArUco

    aruco1 = ChArUco.construction_parameters("aruco1").parameter("a_dict")
    aruco2 = ChArUco.construction_parameters("aruco2").parameter("a_dict")

    assert set(aruco1.choice_labels()) < set(aruco2.choice_labels())
    assert "DICT_ALVAR_7X7_1000" in aruco2.choice_labels()


def test_a_dictionary_is_named_rather_than_numbered():
    """A spec carries the name someone picked; the target resolves it in
    whichever backend's id space it is read with."""
    import cv2

    from pyCamSet.calibration_targets.charuco.target import ChArUco

    target = ChArUco(num_squares_x=5, num_squares_y=5, square_size=4.0,
                     a_dict="DICT_5X5_250")

    assert target._aruco_dict_int == cv2.aruco.DICT_5X5_250
    assert target.input_args["a_dict"] == "DICT_5X5_250", "the spec keeps the name"


# ---------------------------------------------------------------------------
# How a target is printed, which is not what it is
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,cls", _targets(), ids=[n for n, _ in _targets()])
def test_every_declared_export_option_is_one_saving_takes(name, cls):
    """The Create Target dialog builds its options from these and hands
    them to save_printable, so an option that has drifted is a TypeError."""
    import inspect

    accepted = set(inspect.signature(cls.save_printable).parameters)
    declared = {p.key for p in cls.export_parameters().parameters}
    assert declared <= accepted, sorted(declared - accepted)


@pytest.mark.parametrize("name,cls", _targets(), ids=[n for n, _ in _targets()])
def test_every_declared_export_default_is_savings_own(name, cls):
    import inspect

    signature = inspect.signature(cls.save_printable).parameters
    for parameter in cls.export_parameters().parameters:
        assert parameter.default == signature[parameter.key].default, parameter.key


@pytest.mark.parametrize("name,cls", _targets(), ids=[n for n, _ in _targets()])
def test_a_target_names_its_own_file_from_its_own_arguments(name, cls):
    """Named from the values rather than a built target, because a form
    shows the name as it is typed into."""
    from pyCamSet.calibration_targets.core.abstract_target import EXPORT_KINDS

    values = cls.construction_parameters().defaults()
    for kind in EXPORT_KINDS:
        filename = cls.printable_name(values, kind)
        assert filename.endswith(".svg" if kind == "svg" else ".pdf"), kind
        assert "/" not in filename and filename.strip() == filename


def test_every_target_writes_itself_as_every_format(tmp_path):
    """One export path, rather than the four copies of the same dispatch
    the target generators each carried."""
    from pyCamSet.calibration_targets.core.target_registry import build_target

    # Small enough to draw quickly; the point is the path, not the page.
    small = {
        "ChArUco": {"num_squares_x": 4, "num_squares_y": 4, "square_size": 10.0},
        "Ccube": {"n_points": 4, "length": 20.0},
        "PuzzleBoard": {"num_squares_x": 8, "num_squares_y": 8, "square_size": 2.0},
        "PuzzleBoardCube": {"n_points": 5, "length": 100.0},
    }
    for name, cls in _targets():
        target = build_target({"type": name, **small[name]})
        options = cls.export_parameters().defaults()
        written = target.save_printable(
            tmp_path / cls.printable_name(small[name], "svg"), "svg", **options)

        assert written.exists() and written.stat().st_size > 0, name

        with pytest.raises(ValueError, match="cannot be written as"):
            target.save_printable(tmp_path / "x", "postcard", **options)


@pytest.mark.parametrize("name,cls", _targets(), ids=[n for n, _ in _targets()])
def test_everything_a_target_detects_with_is_a_detector_parameterisation(name, cls):
    """A composite asks each of its parts whether its dependency is
    installed and what presets it offers, so a part that is only a
    Parameterisation takes the whole thing down at the first lookup."""
    from pyCamSet.calibration_targets.core.parameters import DetectorParameterisation

    assert isinstance(cls.own_detector_parameters(), DetectorParameterisation)
    for backend, parameterisation in cls.DETECTOR_BACKENDS.items():
        assert isinstance(parameterisation, DetectorParameterisation), backend

    # Which is what makes these answerable for every target.
    for backend in cls.DETECTOR_BACKENDS or [None]:
        composed = cls.detector_parameterisation(backend)
        assert composed.profiles() is not None
        composed.unavailable_reason()


# ---------------------------------------------------------------------------
# A target's own detection, beside its detector's
# ---------------------------------------------------------------------------
#
# PuzzleBoardCube takes twelve settings that alter what find_in_image does
# with the points the detector hands back. They were constructor arguments,
# so nothing could reach them: not the phase 1 form, not a study.


def test_the_cube_declares_the_stages_it_runs_over_its_detections():
    from pyCamSet.calibration_targets.puzzleboard_cube.target import PuzzleBoardCube

    own = PuzzleBoardCube.own_detector_parameters()
    assert {"plane_consistency_gate", "face_reassignment"} <= {
        p.key for p in own.parameters}
    # The PuzzleBoard detector's own setting is not among them: what the
    # cube does with the points is not how the points were found.
    assert "min_width" not in own


def test_the_cube_sweeps_its_own_thresholds_and_its_detectors():
    """The composition, through the tuner: one parameterisation, both halves."""
    from pyCamSet.workflow.tuning.worker import (
        ParameterRowConfig, RunConfig, build_effective_settings)

    config = RunConfig(
        f_loc=".",
        target_spec={"type": "PuzzleBoardCube", "n_points": 6, "length": 100.0},
        parameter_rows=[
            ParameterRowConfig(key="plane_gate_inlier_squares", fixed=0.5,
                               optimise=True, lower=0.2, upper=1.0),
            ParameterRowConfig(key="min_width", fixed=4, optimise=True,
                               lower=3, upper=8),
        ])
    detector = config.detector()

    assert detector.validate_rows(config.parameter_rows) == []
    swept = build_effective_settings(
        config.parameter_rows, detector,
        sampled={"plane_gate_inlier_squares": 0.31, "min_width": 6})
    assert swept["plane_gate_inlier_squares"] == 0.31
    assert swept["min_width"] == 6

    # And a trial's settings build the target it will detect with.
    from pyCamSet.calibration_targets.core.target_registry import build_target
    target = build_target({**config.target_spec, "detection_options": swept})
    assert target.plane_gate_inlier_squares == 0.31
    assert target.min_width == 6


def test_a_row_from_another_detector_is_still_refused():
    """Composition widens what a target takes; it does not open it up."""
    from pyCamSet.calibration_targets.puzzleboard_cube.target import PuzzleBoardCube

    errors = PuzzleBoardCube.detector_parameterisation().validate_rows(
        [{"key": "adaptiveThreshWinSizeMin", "fixed": 3, "optimise": False}])
    assert errors and "Unknown parameter" in errors[0]


def test_reassignment_needs_the_gate_that_finds_what_it_reassigns():
    from pyCamSet.calibration_targets.puzzleboard_cube.target import PuzzleBoardCube

    detector = PuzzleBoardCube.detector_parameterisation()
    assert detector.validate({"face_reassignment": True,
                              "plane_consistency_gate": True}) == []
    assert detector.validate({"face_reassignment": True,
                              "plane_consistency_gate": False})

    with pytest.raises(ValueError, match="needs plane_consistency_gate"):
        PuzzleBoardCube(n_points=6, length=100.0,
                        detection_options={"face_reassignment": True})


def test_each_part_of_a_composite_orders_its_own_sweep():
    """``search_order`` is what a parameterisation says about its own
    parameters; two of them saying "first" is not a disagreement."""
    from pyCamSet.calibration_targets.puzzleboard_cube.target import PuzzleBoardCube

    swept = [p.key for p in PuzzleBoardCube.detector_parameterisation().tunable()]
    own = [p.key for p in PuzzleBoardCube.own_detector_parameters().tunable()]

    assert swept[:len(own)] == own, "the target's own come first, in its order"
    assert swept[len(own):] == ["min_width"]
