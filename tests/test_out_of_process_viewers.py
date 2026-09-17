"""Every native window the GUI opens, it opens in another process.

``visualise_calibration`` opens native matplotlib and pyvista windows.
Called from a Qt slot, pyvista's Cocoa render window runs
``[NSRunLoop runUntilDate:]`` -- a nested event loop that re-enters Qt's
event delivery and repaints the widget tree from inside an event Qt has not
finished dispatching.  The crash report put the fault at
``QMacCGContext::QMacCGContext`` with ``vtkCocoaRenderWindow::Render()``
sixteen frames below it and ``QTabBar::paintEvent`` in between.

Assess Calibration was the first such window.  Create Target was the
second: ``Ccube.plot()`` and ``PuzzleBoardCube.plot()`` reach
``pv.Plotter().show()`` the same way.  The crash arrives either under
``vtkCocoaRenderWindow::Render()`` or on whichever repaint follows, so a
window that appeared to work is not evidence that it was safe.

Both run out of process now, and ``refuse_window_inside_qt`` is the net
under whatever is found next: the same mistake three times is enough to
stop diagnosing it from crash reports.

Most of this holds without a GUI toolkit installed -- the point is that the
GUI process is *not* the one drawing -- so only the handful that stand a
real QApplication up carry the ``gui`` marker.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

from pyCamSet.calibration_targets.core.target_registry import TARGET_NAMES
from pyCamSet.gui import assess_calibration, viewer_process
from pyCamSet.optimisation import optimisation_handling as backend
from pyCamSet.utils import gui_safety, visualise_camset, visualise_target

# --------------------------------------------------------------------------
# Drawing a calibration
# --------------------------------------------------------------------------


def test_the_viewer_runs_as_a_module():
    """The GUI starts it with -m, so it needs a __main__ and a main()."""
    assert callable(visualise_camset.main)

    result = subprocess.run(
        [sys.executable, "-m", "pyCamSet.utils.visualise_camset", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "camset" in result.stdout


def test_a_missing_camset_is_reported_not_raised(capsys):
    assert visualise_camset.main(["/no/such/file.camset"]) == 2
    assert "No such camset" in capsys.readouterr().err


def test_the_launcher_starts_a_process_instead_of_drawing(monkeypatch, tmp_path):
    """The regression: nothing may draw in the Qt process.

    A viewer that draws here is the segfault, so this fails if the drawing
    call ever comes back into the launcher.
    """
    started = []

    class _FakeProcess:
        def poll(self):
            return None

    def _fake_popen(command, *args, **kwargs):
        started.append(command)
        return _FakeProcess()

    def _must_not_run(*args, **kwargs):
        raise AssertionError(
            "visualise_calibration was called in the GUI process; this is "
            "the nested Cocoa run loop that crashes Qt."
        )

    monkeypatch.setattr(viewer_process.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(
        assess_calibration, "visualise_calibration", _must_not_run)

    camset = tmp_path / "run.camset"
    camset.write_bytes(b"not read here")

    ok, detail = assess_calibration.spawn_calibration_viewer(camset)

    assert ok, detail
    assert len(started) == 1
    command = started[0]
    assert command[0] == sys.executable
    assert command[1:3] == ["-m", "pyCamSet.utils.visualise_camset"]
    assert command[3] == str(camset)


def test_a_viewer_that_will_not_start_is_reported(monkeypatch, tmp_path):
    def _refuse(command, *args, **kwargs):
        raise OSError("no exec for you")

    monkeypatch.setattr(viewer_process.subprocess, "Popen", _refuse)

    ok, detail = assess_calibration.spawn_calibration_viewer(
        tmp_path / "run.camset")

    assert ok is False
    assert "no exec for you" in detail


def test_finished_viewers_are_not_left_behind(monkeypatch, tmp_path):
    """Nothing waits on them, so they have to be reaped somewhere."""
    class _Exited:
        def poll(self):
            return 0

    monkeypatch.setattr(
        viewer_process.subprocess, "Popen", lambda *a, **k: _Exited())
    viewer_process._VIEWER_PROCESSES.clear()

    assess_calibration.spawn_calibration_viewer(tmp_path / "a.camset")
    assess_calibration.spawn_calibration_viewer(tmp_path / "b.camset")

    assert viewer_process._VIEWER_PROCESSES == []


@pytest.mark.data
# The one test in this file that draws rather than checking that something
# else does not: it runs the viewer's main in process, on purpose, and the
# figures have to come out.  So it needs a real OpenGL context.
@pytest.mark.needs_opengl
def test_the_viewer_draws_a_real_calibration(charuco_problem, tmp_path):
    """End to end, with no window: the figures have to actually come out."""
    from copy import deepcopy

    from pyCamSet.optimisation.template_handler import TemplateBundleHandler

    target, detections, cams = charuco_problem
    _optimisation, solved = backend.run_bundle_adjustment(
        TemplateBundleHandler(
            camset=deepcopy(cams), target=target, detection=detections,
            options={"outliers": "n", "max_nfev": 2, "verbosity": 0},
        ),
        threads=1,
    )
    camset_path = tmp_path / "solved.camset"
    solved.save(str(camset_path))

    figures = tmp_path / "figures"
    assert visualise_camset.main(
        [str(camset_path), "--no-show", "--save-dir", str(figures)]) == 0

    written = sorted(p.name for p in figures.glob("*.png"))
    assert written, "the viewer drew nothing"


@pytest.mark.data
def test_a_camset_with_no_calibration_says_so(charuco_problem, tmp_path, capsys):
    """A Phase 2 camset has cameras but no solve to draw."""
    _target, _detections, cams = charuco_problem
    camset_path = tmp_path / "uncalibrated.camset"
    cams.save(str(camset_path))

    assert visualise_camset.main([str(camset_path), "--no-show"]) == 1
    assert "nothing to draw" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Drawing a target
# --------------------------------------------------------------------------


def test_create_target_no_longer_draws_in_process():
    """The regression: the Visualise button must not call .plot() here."""
    # find_spec rather than an import: the module is read as text here, and
    # importing it would drag Qt in on an install that has none.
    source = pathlib.Path(
        importlib.util.find_spec("pyCamSet.gui.create_target").origin
    ).read_text(encoding="utf-8")

    assert ".plot()" not in source, (
        "Create Target draws in the GUI process again; Ccube.plot() opens a "
        "pyvista window, which is the crash."
    )
    assert "spawn_viewer" in source


def test_the_target_viewer_runs_as_a_module():
    result = subprocess.run(
        [sys.executable, "-m", "pyCamSet.utils.visualise_target", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0


@pytest.mark.parametrize("target_type", sorted(TARGET_NAMES))
def test_every_target_type_can_be_built_from_what_the_dialog_sends(target_type):
    """The dialog sends a spec, which is what every other phase sends too.

    It used to send its own payload, read through a second table of which
    arguments each target takes -- a table that had drifted from the
    constructors it named.
    """
    from pyCamSet.calibration_targets.core.target_registry import target_class
    from pyCamSet.calibration_targets.markers.aruco2 import ARUCO2_AVAILABLE

    if target_type in ("ChArUco2", "Ccube2") and not ARUCO2_AVAILABLE:
        # ChArUco2 and Ccube2 have no aruco1 equivalent -- they cannot be
        # built at all without aruco2, unlike every other registered target.
        pytest.skip("aruco2 is not installed")

    spec = {"type": target_type,
            **target_class(target_type).construction_parameters().defaults()}

    assert visualise_target.build_target(spec) is not None


def test_a_spec_naming_no_target_says_so():
    with pytest.raises(ValueError, match="which target it is"):
        visualise_target.build_target({"length": 1.0})


def test_an_unknown_target_type_is_refused():
    with pytest.raises(ValueError, match="Unknown target type"):
        visualise_target.build_target({"type": "Trapezoid"})


def test_bad_json_is_reported_not_raised(capsys):
    assert visualise_target.main(["{not json"]) == 2
    assert "Could not read the target settings" in capsys.readouterr().err


# --- the net --------------------------------------------------------------


def test_the_guard_is_quiet_when_no_qt_is_running():
    """The viewer processes and plain scripts must be unaffected.

    In a process of its own, because that is the claim: asserting it here
    would hold only until some earlier test in the session stood a
    QApplication up, and the failure would then be about collection order
    rather than about the guard.
    """
    result = subprocess.run(
        [sys.executable, "-c",
         "from pyCamSet.utils import gui_safety\n"
         "assert gui_safety.qt_application_is_running() is False\n"
         "gui_safety.refuse_window_inside_qt('anything')  # must not raise\n"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.gui
def test_the_guard_names_the_call_under_a_qt_application(monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(gui_safety.sys, "platform", "darwin")
    monkeypatch.delenv(gui_safety._OVERRIDE, raising=False)

    assert gui_safety.qt_application_is_running() is True
    with pytest.raises(RuntimeError, match="CameraSet.plot"):
        gui_safety.refuse_window_inside_qt("CameraSet.plot")


@pytest.mark.gui
def test_the_guard_can_be_overridden(monkeypatch):
    """Someone embedding pyCamSet may have arranged things differently."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(gui_safety.sys, "platform", "darwin")
    monkeypatch.setenv(gui_safety._OVERRIDE, "1")

    gui_safety.refuse_window_inside_qt("CameraSet.plot")   # must not raise


@pytest.mark.gui
def test_the_guard_only_enforces_where_the_crash_happens(monkeypatch):
    """X11 and Windows survive a second native loop; refusing there would
    break setups that work today."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    monkeypatch.delenv(gui_safety._OVERRIDE, raising=False)
    monkeypatch.setattr(gui_safety.sys, "platform", "linux")

    gui_safety.refuse_window_inside_qt("CameraSet.plot")   # must not raise


@pytest.mark.gui
@pytest.mark.data
def test_the_guard_stops_a_real_plot_call(monkeypatch, charuco_problem):
    """End to end: the crash becomes an exception naming the call."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(gui_safety.sys, "platform", "darwin")
    monkeypatch.delenv(gui_safety._OVERRIDE, raising=False)

    _target, _detections, cams = charuco_problem

    with pytest.raises(RuntimeError, match="native window"):
        cams.plot()
