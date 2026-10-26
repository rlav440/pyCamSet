"""
A parameter may offer a choice as a fixed value that a study never samples.

Bounds cannot exclude a categorical value: Optuna's ``suggest_categorical`` is
given the choice list and ignores a row's lower and upper.  OpenCV's
``CORNER_REFINE_APRILTAG`` (3) returns marker quads shifted by about +0.4 px
in x and y (AprilTag's pixel-corner convention inside OpenCV's pixel-centre
API) and loses markers on small squares, so a study that sampled it would
compare corners in two conventions.  It stays a fixed choice; the search
leaves it out through ``searchable_choices``.
"""
import pytest

from pyCamSet.calibration_targets.core.parameters import Choice, Parameter
from pyCamSet.calibration_targets.markers.aruco_opencv import ARUCO_OPENCV_DETECTOR
from pyCamSet.workflow.tuning.optuna_adapter import suggest_for_row
from pyCamSet.workflow.tuning.worker import ParameterRowConfig


class _Trial:
    """Records what a study was offered, and picks the last offer."""

    def __init__(self):
        self.offered = {}

    def suggest_categorical(self, key, values):
        self.offered[key] = list(values)
        return values[-1]


def _method():
    return ARUCO_OPENCV_DETECTOR.parameter("cornerRefinementMethod")


def test_apriltag_is_still_a_fixed_choice():
    parameter = _method()
    assert [c.value for c in parameter.choices] == [0, 1, 2, 3]
    assert parameter.parse("CORNER_REFINE_APRILTAG") == 3
    assert parameter.validate_row(fixed=3, optimise=False) == []


def test_a_study_never_samples_apriltag():
    trial = _Trial()
    row = ParameterRowConfig(key="cornerRefinementMethod", fixed=3, optimise=True, lower=0, upper=3)
    value = suggest_for_row(trial, row, _method())
    assert trial.offered["cornerRefinementMethod"] == [0, 1, 2]
    assert value == 2


def test_without_searchable_choices_every_choice_is_searched():
    parameter = Parameter(key="k", label="K", default=0, dtype="int",
                          choices=(Choice("a", 0), Choice("b", 1)))
    assert parameter.search_choices == [0, 1]
    trial = _Trial()
    suggest_for_row(trial, ParameterRowConfig(key="k", fixed=0, optimise=True), parameter)
    assert trial.offered["k"] == [0, 1]


def test_searchable_choices_come_from_a_stored_row_in_choice_order():
    parameter = Parameter.from_row({
        "key": "k", "label": "K", "default": 0, "dtype": "int",
        "choices": [{"label": "a", "value": 0}, {"label": "b", "value": 1}, {"label": "c", "value": 2}],
        "searchable_choices": [2, 0],
    })
    assert parameter.searchable_choices == (2, 0)
    assert parameter.search_choices == [0, 2]


def test_a_searchable_value_that_is_not_a_choice_is_refused():
    with pytest.raises(ValueError, match="searchable_choices"):
        Parameter(key="k", label="K", default=0, dtype="int",
                  choices=(Choice("a", 0),), searchable_choices=(5,))
