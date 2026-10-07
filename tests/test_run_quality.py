"""Whether a finished run is worth carrying into the next phase.

A phase can finish and still have produced nothing to go on -- a camera that
saw the target in no image, a solve that ended at NaN -- and the next phase
was one green button away from being handed it.  These cover the rule that
decides, and the button that shows it.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyCamSet.workflow.run_quality import blocking_reasons


def _run(**metadata) -> dict:
    """A run record with the keys the runners always save."""
    return {"run_id": "r1", "error": None, "report": None, **metadata}


def test_a_clean_run_blocks_nothing():
    assert blocking_reasons(_run(report={"flags": [], "blocking_flags": []})) == []


def test_a_run_that_never_happened_blocks_nothing():
    """The phase tabs answer "nothing has been run yet" themselves, and in
    their own words."""
    assert blocking_reasons(None) == []
    assert blocking_reasons({}) == []


def test_an_errored_run_reports_its_error():
    reasons = blocking_reasons(_run(error="No selected camera sub-folders found."))

    assert reasons == ["No selected camera sub-folders found."]


def test_a_blind_camera_blocks_the_next_phase():
    reasons = blocking_reasons(_run(report={
        "flags": ['camera "cam2" detected the target in none of the 20 images'],
        "blocking_flags": [
            'camera "cam2" detected the target in none of the 20 images'],
    }))

    assert len(reasons) == 1
    assert "cam2" in reasons[0]


def test_a_merely_poor_run_is_not_blocked():
    """A high reprojection error is a warning on a run that still produced a
    camera set: whether to trust it is the reader's call, not this one's."""
    reasons = blocking_reasons(_run(report={
        "flags": ["final mean error 7.20 px is above the 5 px this check expects"],
        "blocking_flags": [],
    }))

    assert reasons == []


# --------------------------------------------------------------------------
# What each report calls blocking
# --------------------------------------------------------------------------

def test_a_detection_report_blocks_on_a_camera_that_saw_nothing():
    from pyCamSet.calibration_targets.core.target_detections import (
        ImageDetection, TargetDetection)
    from pyCamSet.utils.setup_reports import DetectionReport

    class _Corners:
        point_data = np.zeros((1, 4, 3))

    detection = TargetDetection(cam_names=["seeing", "blind"])
    detection.max_ims = 2
    for im_num in range(2):
        detection.add_detection(
            "seeing", im_num,
            ImageDetection(keys=np.arange(4),
                           image_points=np.tile([1.0, 2.0], (4, 1))))

    report = DetectionReport.from_detection(detection, _Corners())

    assert len(report.blocking_flags) == 1
    assert 'camera "blind"' in report.blocking_flags[0]


def test_a_calibration_report_blocks_on_a_nan_result():
    from scipy.optimize import OptimizeResult

    from pyCamSet.calibration_targets.core.target_detections import (
        ImageDetection, TargetDetection)
    from pyCamSet.utils.calibration_report import CalibrationReport

    detection = TargetDetection(cam_names=["cam"])
    detection.max_ims = 1
    detection.add_detection(
        "cam", 0,
        ImageDetection(keys=np.arange(2),
                       image_points=np.tile([1.0, 2.0], (2, 1))))

    class _Handler:
        def __init__(self):
            self.detection = detection
            self.missing_poses = None

    report = CalibrationReport.from_optimisation(
        OptimizeResult(fun=np.full(4, np.nan), x=np.zeros(3)),
        _Handler(), initial_error_px=1.0, duration_s=0.1, solver="trf")

    assert report.blocking_flags == [f for f in report.flags if "NaN" in f]
    assert report.blocking_flags


# --------------------------------------------------------------------------
# The button
# --------------------------------------------------------------------------

@pytest.fixture
def qt_app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.mark.gui
def test_a_blocked_run_turns_the_button_red_and_says_why(qt_app, monkeypatch):
    from pyCamSet.gui import shared_functions as sf

    clicks = []
    btn = sf.make_continue_button(lambda: clicks.append(1))
    lines = []
    terminal = type("T", (), {"append_line": lambda _self, line: lines.append(line)})()

    sf.gate_continue_button(btn, terminal, {
        "error": None,
        "report": {"blocking_flags": ["cam2 saw nothing"]},
    })

    # The theme styles the role; the role, not an inline colour, is the contract.
    assert btn.property("designRole") == "warning"
    assert "cam2 saw nothing" in btn.toolTip()
    assert lines == ["Cannot continue: cam2 saw nothing"]

    # Enabled, but it asks first -- and takes no for an answer.
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: QMessageBox.StandardButton.No)
    assert btn.isEnabled()
    btn.click()
    assert clicks == []

    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)
    btn.click()
    assert clicks == [1]


@pytest.mark.gui
def test_a_clean_run_leaves_the_button_green(qt_app):
    from pyCamSet.gui import shared_functions as sf

    clicks = []
    btn = sf.make_continue_button(lambda: clicks.append(1))
    lines = []
    terminal = type("T", (), {"append_line": lambda _self, line: lines.append(line)})()

    sf.gate_continue_button(btn, terminal, {"error": None, "report": {}})

    assert btn.property("designRole") == "success"
    assert lines == []
    btn.click()          # no dialog to answer
    assert clicks == [1]
