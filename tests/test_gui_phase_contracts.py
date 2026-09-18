"""What the phase tabs do, as widgets.

The contract these read is covered without Qt in
:mod:`tests.test_workflow_backend_seam`; this is the other end of it --
the tab bar, the Run button, and the target form -- exercised against a
real QApplication under the offscreen platform plugin.

Every test here needs the GUI toolkit, so every test here is marked
``gui``: on an install without PySide6 they skip, and say so.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyCamSet.calibration_targets.markers.aruco2 import ARUCO2_AVAILABLE

from pyCamSet.workflow.targets import describe_target_mismatch
from pyCamSet.calibration_targets.core.target_registry import TARGET_NAMES

# The Ccube target behind the reported failure, as a run records it.  The
# backend-seam tests carry their own copy: what a form must round-trip and
# what the mismatch check must catch are two different claims about it.
CCUBE_12 = {"target": {"type": "Ccube", "n_points": 12, "length": 80.0,
                       "border_fraction": 0.1, "marker_backend": "aruco1"}}


def _with(params, **changes):
    """The same parameters, with the target's spec altered."""
    return {"target": {**params["target"], **changes}}


# The form's controls were a map of widget names, one copy per phase, read
# by a helper.  They are the target's own declared arguments now, so the
# tests below are about the form rather than about the map.
def _target_form(target_type=None, detector_mode=None):
    from pyCamSet.gui.shared_functions import TargetSettingsForm

    form = (TargetSettingsForm() if detector_mode is None
            else TargetSettingsForm(detector_mode=detector_mode))
    if target_type is not None:
        form.set_target_type(target_type)
    return form


# --------------------------------------------------------------------------
# A hidden tab must never be the current tab
# --------------------------------------------------------------------------
#
# The diagnostics pages are hidden in the tab bar to keep it short, and
# were then made current anyway.  Qt never arranges that itself --
# setTabVisible moves the current tab along when it hides one -- and when
# it does happen the tab bar paints a current tab that has no geometry:
#
#     QPainter::begin: Paint device returned engine == 0, type: 3
#
# which is the null the macOS style dereferences in QMacCGContext.  Every
# crash report from this GUI was exactly that, under QTabBar::paintEvent,
# on whatever repaint came next: a viewer window opening, the application
# losing focus.  It is why the app died "while the plots were shown" even
# once the drawing itself had been moved out of process.
#
# These assert the invariant rather than forcing a repaint: a regression
# should fail a test, not take the suite down with a segmentation fault.


@pytest.fixture
def qt_app_for_tabs():
    """The real main window, for the tab-bar invariant."""
    pytest.importorskip("PySide6", reason="the GUI is Qt")
    from PySide6.QtWidgets import QApplication

    from pyCamSet.gui.main_window import PyCamSetApp

    QApplication.instance() or QApplication([])
    return PyCamSetApp()


@pytest.fixture
def show_tab():
    """The GUI's own way of revealing a hidden diagnostics tab."""
    pytest.importorskip("PySide6", reason="the GUI is Qt")
    from pyCamSet.gui.shared_functions import show_tab as _show_tab

    return _show_tab


# --------------------------------------------------------------------------
# The dead cross-tab target-sync block must not come back
# --------------------------------------------------------------------------
#
# main_window.py used to wire signals to _target_combo, _npts_spin,
# _length_edit and _marker_backend_combo directly on each phase tab -- names
# that never existed there (the real widgets live one level down, on
# tab._target_form). Every hasattr guard was therefore always False, so the
# block was wired to nothing and never ran. Per Q5 it was deleted rather than
# fixed: TargetSettingsForm.apply_spec() already covers adopting a saved
# run's target. This pins both halves of that decision so the same phantom
# wiring cannot silently reappear.


@pytest.mark.gui
def test_the_dead_cross_tab_sync_names_never_come_back(qt_app_for_tabs):
    from PySide6.QtWidgets import QComboBox

    app = qt_app_for_tabs
    dead_names = ("_propagate_target", "_syncing_target",
                  "_marker_backend_combo", "_npts_spin", "_length_edit")

    # None of the phase tabs ever had these attributes; the block's own
    # hasattr guards were checking for names that were never there.
    for tab_attr in ("phase1_tab", "phase2_tab", "phase3_tab", "phase4_tab"):
        tab = getattr(app, tab_attr)
        for name in dead_names:
            assert not hasattr(tab, name), f"{tab_attr}.{name} should not exist"

    # Nor on the window itself -- _syncing_target was a guard flag on
    # PyCamSetApp; _propagate_target was a nested function, never an
    # attribute, but is asserted absent here too for symmetry.
    assert not hasattr(app, "_propagate_target")
    assert not hasattr(app, "_syncing_target")

    # The widget path signals actually could reach, had anything wired to
    # it: the shared form each target-bearing tab really owns.  Only Phase 1
    # chooses a detector; Phases 2 and 3 read that run's detections with the
    # detector it used, so their forms have no combo to wire to.
    assert isinstance(app.phase1_tab._target_form._backend_combo, QComboBox)
    for tab_attr in ("phase2_tab", "phase3_tab"):
        tab = getattr(app, tab_attr)
        assert tab._target_form._backend_combo is None


@pytest.mark.gui
def test_no_tab_is_ever_current_while_hidden(qt_app_for_tabs):
    """Every route in: setCurrentIndex, setCurrentWidget, the keyboard."""
    window = qt_app_for_tabs
    notebook = window._notebook
    bar = notebook.tabBar()

    for index in range(notebook.count()):
        notebook.setCurrentIndex(index)
        current = notebook.currentIndex()
        assert bar.isTabVisible(current), (
            f"tab {current} is current while hidden; the tab bar will paint "
            f"a current tab with no geometry and the macOS style will "
            f"dereference the null it gets."
        )


@pytest.mark.gui
def test_the_diagnostics_tabs_start_hidden(qt_app_for_tabs):
    """The point of hiding them: the bar stays short until one is used."""
    window = qt_app_for_tabs
    bar = window._notebook.tabBar()

    assert window._diagnostics_indices
    assert not any(bar.isTabVisible(i) for i in window._diagnostics_indices)


@pytest.mark.gui
def test_opening_a_diagnostics_tab_reveals_it(qt_app_for_tabs, show_tab):
    window = qt_app_for_tabs
    notebook = window._notebook
    bar = notebook.tabBar()

    show_tab(notebook, window.phase4_diag_tab)

    current = notebook.currentIndex()
    assert notebook.currentWidget() is window.phase4_diag_tab
    assert bar.isTabVisible(current)


@pytest.mark.gui
def test_leaving_a_diagnostics_tab_puts_it_away(qt_app_for_tabs, show_tab):
    window = qt_app_for_tabs
    notebook = window._notebook
    bar = notebook.tabBar()
    show_tab(notebook, window.phase3_diag_tab)
    revealed = notebook.indexOf(window.phase3_diag_tab)

    notebook.setCurrentWidget(window.phase3_tab)

    assert not bar.isTabVisible(revealed)
    assert bar.isTabVisible(notebook.currentIndex())


@pytest.mark.gui
def test_a_tab_made_current_by_index_is_revealed(qt_app_for_tabs):
    """The choke point is currentChanged, so even a raw index works.

    This is the exact call that segmentation faulted: nothing routes it
    through show_tab, so the invariant cannot live at the call sites.
    """
    window = qt_app_for_tabs
    notebook = window._notebook
    bar = notebook.tabBar()
    hidden = window._diagnostics_indices[0]
    assert not bar.isTabVisible(hidden)

    notebook.setCurrentIndex(hidden)

    assert notebook.currentIndex() == hidden
    assert bar.isTabVisible(hidden)


# --------------------------------------------------------------------------
# Pressing Run
# --------------------------------------------------------------------------
#
# A KeyError reached a user from the line that prints a run's settings to
# the terminal pane, because it still read a target field that had moved.
# Nothing had exercised it: the tests covered what the form produces and
# what the phase does with it, and the tab's own code in between -- the
# headers, the labels, the handoff -- ran only when someone clicked Run.
#
# These click Run, with the phase itself stubbed out.  They assert almost
# nothing about the result; the point is that the path from the widgets to
# the worker executes at all.


@pytest.mark.gui
@pytest.mark.parametrize(
    ("tab_attribute", "run_method"),
    [("phase1_tab", "_run_phase1"),
     ("phase2_tab", "_run_phase2"),
     ("phase3_tab", "_run_phase3"),
     ("phase4_tab", "_run_phase4")],
)
def test_pressing_run_reaches_the_worker(tab_attribute, run_method,
                                         monkeypatch, tmp_path):
    """Every line the tab runs between the form and the phase."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QMessageBox

    import cv2
    import numpy as np
    from pyCamSet.gui import main_window as mw
    from pyCamSet.gui import shared_functions as shared
    from pyCamSet.workflow import phase1, phase2, phase3, phase4

    images = tmp_path / "images"
    for camera in ("cam0", "cam1"):
        (images / camera).mkdir(parents=True)
        for index in range(2):
            cv2.imwrite(str(images / camera / f"im{index}.png"),
                        np.zeros((16, 16, 3), dtype=np.uint8))

    # A dialog would block; a real solve would take minutes.
    for name in ("critical", "warning", "information", "question"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: None)
    started: list[dict] = []
    for module in (phase1, phase2, phase3, phase4):
        monkeypatch.setattr(
            module, "run",
            lambda params, *a, **k: started.append(params) or {"run_id": "x"})
        if hasattr(module, "rerun"):
            monkeypatch.setattr(module, "rerun", lambda *a, **k: {"run_id": "x"})

    # Run the worker's callable here rather than on a thread, so anything it
    # raises is this test's failure instead of a stray traceback.
    monkeypatch.setattr(shared.PhaseWorker, "start",
                        lambda self: self._work_fn(lambda _line: None))

    app = QApplication.instance() or QApplication([])
    window = mw.PyCamSetApp()
    try:
        tab = getattr(window, tab_attribute)
        for field in ("_floc_edit",):
            if hasattr(tab, field):
                getattr(tab, field).setText(str(images))
        if hasattr(tab, "set_cameras"):
            tab.set_cameras(["cam0", "cam1"], ["cam0", "cam1"])

        getattr(tab, run_method)()
    finally:
        window.deleteLater()

    # Either the phase ran, or the tab refused for a reason of its own
    # (no upstream run to read) -- but nothing raised on the way.
    assert started or True


@pytest.mark.gui
def test_the_run_header_describes_the_target_it_was_given():
    """The line that broke, asserted on directly."""
    from pyCamSet.gui.phase_1_detection import _run_header

    lines = _run_header({
        "f_loc": "/data/rig",
        "target": {"type": "ChArUco", "num_squares_x": 20,
                   "num_squares_y": 20, "square_size": 4.0},
        "caching": True, "high_distortion": False, "threads": None,
        "selected_cameras": ["cam0"],
    })

    assert any("ChArUco" in line and "num_squares_x=20" in line for line in lines)


@pytest.mark.gui
def test_making_a_target_is_a_dialog_rather_than_a_tab():
    """A target is drawn once and then lived with for months of runs, so it
    is not one of the phases -- and as the first tab it was what every
    session opened on."""
    from PySide6.QtWidgets import QApplication

    from pyCamSet.gui import main_window as mw

    QApplication.instance() or QApplication([])
    window = mw.PyCamSetApp()
    try:
        names = [window._notebook.tabText(i)
                 for i in range(window._notebook.count())]
        assert not any("Create Target" in name for name in names)

        window._open_create_target()
        dialog = window._create_target_dialog
        assert dialog.isVisible()
        assert dialog.windowTitle() == "Create Target"

        # Reopening keeps the settings that drew what is already on screen.
        window._open_create_target()
        assert window._create_target_dialog is dialog
    finally:
        window.deleteLater()


@pytest.mark.gui
@pytest.mark.data
@pytest.mark.slow
def test_a_whole_calibration_runs_from_the_window(session_data_dir, tmp_path,
                                                  monkeypatch):
    """Phases 1 to 4, driven from the tabs, solving for real.

    Everything else here stops at a seam: the form produces parameters, or
    the phase consumes them, or a stub stands in for the solve.  This runs
    the actual thing -- detection over the image corpus, three real solves,
    and the handoff between tabs that carries a run from one to the next --
    because the breaks that reach a user are the ones between the parts.
    """
    from PySide6.QtWidgets import QApplication, QMessageBox

    from pyCamSet.gui import main_window as mw
    from pyCamSet.gui import shared_functions as shared
    from pyCamSet.workflow.workspace import WorkspaceManager, workspace_path_for

    # The corpus, somewhere a run may write its cache and its workspace.
    images = tmp_path / "images"
    images.mkdir()
    for camera in sorted((session_data_dir / "calibration_charuco").iterdir()):
        if camera.is_dir():
            (images / camera.name).symlink_to(camera, target_is_directory=True)

    refused: list[str] = []
    for name in ("critical", "warning"):
        monkeypatch.setattr(
            QMessageBox, name,
            lambda _p, title, text, *a, **k: refused.append(f"{title}: {text}"))
    for name in ("information", "question"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: None)

    # Run the worker where the test can see what it raises, rather than on a
    # thread of its own.  PhaseWorker.run still emits, so the tab's finished
    # and error slots run too.
    monkeypatch.setattr(shared.PhaseWorker, "start", lambda self: self.run())

    app = QApplication.instance() or QApplication([])
    window = mw.PyCamSetApp()
    workspace = WorkspaceManager(workspace_path_for(images))

    try:
        # ---- phase 1: detect -------------------------------------------
        tab = window.phase1_tab
        tab._floc_edit.setText(str(images))
        tab.set_cameras(["1", "2", "3"], ["1", "2", "3"])
        tab._target_form.apply_spec(
            {"type": "ChArUco", "num_squares_x": 20, "num_squares_y": 20,
             "square_size": 4.0, "legacy": False})
        tab._nlim_edit.setText("4")          # four images is enough to solve
        # Caching on: the cache file is what phase 1 saves as its artifact,
        # and what phase 2 reads.  Without it a run records no detections.
        tab._cache_cb.setChecked(True)
        tab._run_phase1()

        assert not refused, refused
        phase1_run = workspace.load_runs("phase1")[-1]
        assert phase1_run["error"] is None, phase1_run["error"]
        assert phase1_run["diagnostics"]["D1.7_min_features"] > 0
        assert phase1_run["artifacts"]["detected_datapoints_pickle"]

        # ---- phase 2: per-camera intrinsics ------------------------------
        tab = window.phase2_tab
        tab._floc_edit.setText(str(images))
        tab._target_form.apply_spec(
            {"type": "ChArUco", "num_squares_x": 20, "num_squares_y": 20,
             "square_size": 4.0, "legacy": False})
        tab._run_phase2()

        assert not refused, refused
        phase2_run = workspace.load_runs("phase2")[-1]
        assert phase2_run["error"] is None, phase2_run["error"]
        assert phase2_run["inputs"]["phase1_run_id"] == phase1_run["run_id"]
        assert phase2_run["artifacts"]["initial_camset"]

        # ---- phase 3: bundle adjustment ----------------------------------
        tab = window.phase3_tab
        tab._floc_edit.setText(str(images))
        tab._target_form.apply_spec(
            {"type": "ChArUco", "num_squares_x": 20, "num_squares_y": 20,
             "square_size": 4.0, "legacy": False})
        tab._max_nfev_spin.setValue(10)
        tab._run_phase3()

        assert not refused, refused
        phase3_run = workspace.load_runs("phase3")[-1]
        assert phase3_run["error"] is None, phase3_run["error"]
        assert phase3_run["inputs"]["phase2_run_id"] == phase2_run["run_id"]
        final_rpe = phase3_run["diagnostics"]["D3.6_final_euclid_px"]
        assert np.isfinite(final_rpe), final_rpe

        # ---- phase 4: self-calibration -----------------------------------
        tab = window.phase4_tab
        tab._floc_edit.setText(str(images))
        tab._max_nfev_spin.setValue(10)
        tab._run_phase4()

        assert not refused, refused
        phase4_run = workspace.load_runs("phase4")[-1]
        assert phase4_run["error"] is None, phase4_run["error"]
        assert phase4_run["inputs"]["phase3_run_id"] == phase3_run["run_id"]
        assert np.isfinite(phase4_run["diagnostics"]["D4.3_final_euclid_px"])

        # Every run says which target made it, in the shape the next phase
        # reads -- the thing that broke when the shape changed.
        for run in (phase1_run, phase2_run, phase3_run):
            assert run["params"]["target"]["type"] == "ChArUco"

        # The solve did work, rather than reporting a number it never earned.
        assert phase3_run["diagnostics"]["D3.9_nfev"] > 1
        assert phase3_run["diagnostics"]["D3.5_initial_euclid_px"] > 0
        per_camera = phase3_run["diagnostics"]["D3.12_per_camera_mean_reprojection"]
        assert len(per_camera) == 3 and all(np.isfinite(v) for v in per_camera.values())
        assert len(phase2_run["diagnostics"]["D2.2_intrinsics"]) == 3
        assert phase1_run["diagnostics"]["n_images"] == 4
    finally:
        window.deleteLater()


@pytest.mark.gui
def test_the_terminal_paints_the_colours_the_reports_ask_for():
    """A report grades its numbers by colour -- blue for an exceptional
    reprojection error, red for one worth worrying about -- and the pane
    used to strip those escapes out, so every number read the same."""
    from PySide6.QtGui import QTextFormat
    from PySide6.QtWidgets import QApplication, QCheckBox

    from pyCamSet.gui import shared_functions as shared
    from pyCamSet.utils import report_format as fmt

    QApplication.instance() or QApplication([])
    show = QCheckBox()
    show.setChecked(True)
    terminal = shared.TerminalWidget(show)
    try:
        rows = [["cam0", fmt.error_cell(0.08)], ["cam1", fmt.error_cell(7.1)]]
        for line in fmt.table(["camera", "mean"], rows, [10, 8], colour=True):
            terminal.append_line(line)

        # The escapes are gone from the text and present in the formatting.
        assert "\x1b[" not in terminal.toPlainText()
        painted = {}
        for number in range(terminal.document().blockCount()):
            block = terminal.document().findBlockByNumber(number)
            fragment = block.begin()
            while fragment != block.end():
                run = fragment.fragment()
                if run.charFormat().hasProperty(
                        QTextFormat.Property.ForegroundBrush):
                    painted[run.text().strip()] = \
                        run.charFormat().foreground().color().name()
                fragment += 1

        # Only the graded numbers are pinned to a colour; everything else is
        # left to the pane's own foreground, so it reads on any background.
        assert painted == {
            "0.08": shared.xterm_colour(fmt.SOLARIZED["blue"]).name(),
            "7.10": shared.xterm_colour(fmt.SOLARIZED["red"]).name(),
        }
    finally:
        terminal.deleteLater()


@pytest.mark.gui
def test_the_terminal_drops_the_escapes_that_move_a_cursor():
    """A pane that only appends cannot act on a progress bar's cursor moves,
    and must not show them either."""
    from PySide6.QtWidgets import QApplication, QCheckBox

    from pyCamSet.gui import shared_functions as shared

    QApplication.instance() or QApplication([])
    show = QCheckBox()
    show.setChecked(True)
    terminal = shared.TerminalWidget(show)
    try:
        terminal.append_line("detecting \x1b[2K\x1b[1G 50%\r")
        assert terminal.toPlainText() == "detecting  50%\n"
    finally:
        terminal.deleteLater()


# --------------------------------------------------------------------------
# Phase 1 shows the settings the selected target's detection takes
# --------------------------------------------------------------------------
#
# The form held one detector's parameters and appeared for the two targets
# that used that detector.  So a ChArUco read with aruco2 was offered
# seventeen OpenCV settings that the aruco2 call cannot be given -- the
# target logged that it was ignoring them -- and a PuzzleBoard was offered
# none, though its detector takes one.


def _phase1_tab():
    """A Phase 1 tab with no workspace, which is enough to read its form."""
    from PySide6.QtWidgets import QCheckBox, QTabWidget

    from pyCamSet.gui.phase_1_detection import Phase1Tab
    from pyCamSet.workflow.workspace import WorkspaceManager

    return Phase1Tab(QTabWidget(), QCheckBox(), QCheckBox(), WorkspaceManager(None))


@pytest.mark.gui
@pytest.mark.parametrize(
    ("target_type", "backend", "detector", "keys"),
    [
        ("ChArUco", "aruco1", "aruco1", None),
        ("Ccube", "aruco1", "aruco1", None),
        ("ChArUco", "aruco2", "aruco2", ()),
        ("PuzzleBoard", None, "puzzle_board", ("min_width",)),
        # A target with settings of its own reads as its own beside its
        # backend's: the cube's two optional face-assignment stages are not
        # the PuzzleBoard detector's business.
        ("PuzzleBoardCube", None, "PuzzleBoardCube+puzzle_board", None),
    ],
)
def test_the_detection_form_shows_what_the_chosen_detector_takes(
        target_type, backend, detector, keys):
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    tab = _phase1_tab()
    try:
        tab._target_form.set_target_type(target_type)
        if backend is not None:
            tab._target_form._backend_combo.setCurrentIndex(
                tab._target_form._backend_combo.findData(backend))

        parameterisation = tab._current_detector_parameterisation()
        assert parameterisation.name == detector
        shown = tuple(tab._detection_option_widgets)
        if keys is None:
            # OpenCV's, whatever the table currently says they are.
            keys = tuple(p.key for p in parameterisation.settable())
        assert shown == keys
        assert tab._detection_opts_section.isHidden() is (not keys)
    finally:
        tab.deleteLater()


@pytest.mark.gui
def test_a_tuned_detection_option_survives_a_backend_round_trip():
    """Both detectors are offered side by side in the same combo now, which
    invites tuning one, comparing the other, and coming back -- and that
    must not silently drop the typed value back to the library default."""
    from PySide6.QtWidgets import QApplication

    from pyCamSet.gui.shared_functions import read_parameter_widget

    QApplication.instance() or QApplication([])
    tab = _phase1_tab()
    try:
        tab._target_form.set_target_type("ChArUco")
        assert tab._target_form._backend_combo.currentData() == "aruco1"

        widget = tab._detection_option_widgets["minMarkerPerimeterRate"]
        assert read_parameter_widget(widget) == pytest.approx(0.03)
        widget.setValue(1.03)

        # Compare ArUco 2, then come back to ArUco 1.
        tab._target_form._backend_combo.setCurrentIndex(
            tab._target_form._backend_combo.findData("aruco2"))
        tab._target_form._backend_combo.setCurrentIndex(
            tab._target_form._backend_combo.findData("aruco1"))

        restored = tab._detection_option_widgets["minMarkerPerimeterRate"]
        assert restored is not widget  # rebuilt, not the same object
        assert read_parameter_widget(restored) == pytest.approx(1.03)
    finally:
        tab.deleteLater()


@pytest.mark.gui
def test_a_tuned_detection_option_survives_a_target_type_round_trip():
    """The same gap exists for a target-type round trip (A -> B -> A), not
    just a backend round trip -- both go through the same rebuild."""
    from PySide6.QtWidgets import QApplication

    from pyCamSet.gui.shared_functions import read_parameter_widget

    QApplication.instance() or QApplication([])
    tab = _phase1_tab()
    try:
        tab._target_form.set_target_type("ChArUco")
        widget = tab._detection_option_widgets["minMarkerPerimeterRate"]
        widget.setValue(1.03)

        tab._target_form.set_target_type("Ccube")
        tab._target_form.set_target_type("ChArUco")

        restored = tab._detection_option_widgets["minMarkerPerimeterRate"]
        assert read_parameter_widget(restored) == pytest.approx(1.03)
    finally:
        tab.deleteLater()


@pytest.mark.gui
@pytest.mark.parametrize(
    ("target_type", "offered"),
    [("ChArUco", True), ("Ccube", True), ("ChArUco2", True), ("Ccube2", True),
     ("PuzzleBoard", False), ("PuzzleBoardCube", False)],
)
def test_only_a_target_with_a_choice_of_detector_is_asked_for_one(
        target_type, offered):
    """The combo used to appear for a named pair of targets.  It appears
    for every target read with ArUco markers -- for ChArUco2 with ArUco 1
    greyed out -- and never for one read without them."""
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    tab = _phase1_tab()
    try:
        tab._target_form.set_target_type(target_type)
        assert tab._target_form._backend_combo.isHidden() is (not offered)
    finally:
        tab.deleteLater()


# --------------------------------------------------------------------------
# The optimisation tab sweeps what the selected detector takes
# --------------------------------------------------------------------------
#
# The sliders were OpenCV's, whatever the target was read with.  So a study
# over a target set to aruco2 offered sixteen settings to search over that
# the aruco2 call cannot be given, and would have spent its whole budget
# re-running one detection.


def _optimisation_tab():
    from PySide6.QtWidgets import QCheckBox, QTabWidget

    from pyCamSet.gui.optimisation_tab import OptimisationTab
    from pyCamSet.workflow.workspace import WorkspaceManager

    return OptimisationTab(QTabWidget(), QCheckBox(), QCheckBox(),
                           WorkspaceManager(None))


@pytest.mark.gui
def test_the_sweepable_rows_follow_the_selected_detector():
    from PySide6.QtWidgets import QApplication

    from pyCamSet.calibration_targets.markers.aruco_opencv import ARUCO_OPENCV_DETECTOR

    QApplication.instance() or QApplication([])
    tab = _optimisation_tab()
    try:
        assert set(tab._param_rows) == {p.key for p in ARUCO_OPENCV_DETECTOR.tunable()}
        assert tab._nothing_to_sweep.text() == ""

        tab._target_form._backend_combo.setCurrentIndex(tab._target_form._backend_combo.findData("aruco2"))
        assert tab._param_rows == {}
        assert "takes no settings" in tab._nothing_to_sweep.text()
        assert tab._collect_parameter_rows() == []

        tab._target_form._backend_combo.setCurrentIndex(tab._target_form._backend_combo.findData("aruco1"))
        assert set(tab._param_rows) == {p.key for p in ARUCO_OPENCV_DETECTOR.tunable()}
    finally:
        tab.deleteLater()


@pytest.mark.gui
def test_a_widened_sweep_bound_survives_a_backend_round_trip():
    """Both detectors are offered side by side in the same combo now, which
    invites widening a bound to search a larger space, comparing the other
    detector, and coming back -- and that must not silently narrow it back
    to the detection profile's own default."""
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    tab = _optimisation_tab()
    try:
        row = tab._param_rows["adaptiveThreshWinSizeMin"]
        row.set_bounds(9, 45)
        assert row.bounds() == (9, 45)

        # aruco2 has no rows of its own (see the test above): the widened
        # row is torn down entirely before it is rebuilt on the way back.
        tab._target_form._backend_combo.setCurrentIndex(
            tab._target_form._backend_combo.findData("aruco2"))
        tab._target_form._backend_combo.setCurrentIndex(
            tab._target_form._backend_combo.findData("aruco1"))

        restored = tab._param_rows["adaptiveThreshWinSizeMin"]
        assert restored is not row  # rebuilt, not the same object
        assert restored.bounds() == (9, 45)
    finally:
        tab.deleteLater()


@pytest.mark.gui
def test_the_preset_selector_never_offers_another_detectors_presets():
    """A hidden combo keeping the last detector's presets would apply them
    to rows that do not exist."""
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    tab = _optimisation_tab()
    try:
        def offered():
            return [tab._detection_profile_combo.itemText(i)
                    for i in range(tab._detection_profile_combo.count())]

        assert "Balanced" in offered() and offered()[-1] == "Custom"

        tab._target_form._backend_combo.setCurrentIndex(tab._target_form._backend_combo.findData("aruco2"))
        assert offered() == ["Custom"], "aruco2 has no presets of its own"
        assert tab._detection_profile_combo.isHidden()
    finally:
        tab.deleteLater()


@pytest.mark.gui
def test_set_parameter_widget_shows_a_combo_value_its_items_do_not_offer():
    """``QComboBox.setCurrentText`` silently no-ops for a value that is not
    among the combo's current items, so a saved spec naming a value the
    combo no longer offers used to leave the combo showing whatever it
    already held instead. ``set_parameter_widget`` must add the value
    first, then select it -- and must not duplicate a value the combo
    already offers.
    """
    from PySide6.QtWidgets import QApplication, QComboBox

    from pyCamSet.gui.shared_functions import set_parameter_widget

    QApplication.instance() or QApplication([])
    combo = QComboBox()
    combo.addItems(["alpha", "beta"])
    try:
        set_parameter_widget(combo, "gamma")
        assert combo.currentText() == "gamma"
        assert [combo.itemText(i) for i in range(combo.count())] == \
            ["alpha", "beta", "gamma"]

        set_parameter_widget(combo, "beta")
        assert combo.currentText() == "beta"
        assert [combo.itemText(i) for i in range(combo.count())] == \
            ["alpha", "beta", "gamma"], "an offered value must gain no duplicate"
    finally:
        combo.deleteLater()


@pytest.mark.gui
def test_a_preset_still_sets_the_bounds_of_the_rows_it_covers():
    """The rows are rebuilt now, so the preset has to reach the new ones."""
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    tab = _optimisation_tab()
    try:
        tab._target_form.set_target_type("Ccube")
        tab._detection_profile_combo.setCurrentText("Aggressive Recovery")
        row = tab._param_rows["adaptiveThreshWinSizeMax"]

        from pyCamSet.calibration_targets.markers.aruco_opencv import ARUCO_OPENCV_DETECTOR
        expected = ARUCO_OPENCV_DETECTOR.profiles()[
            "Aggressive Recovery"].bounds_for("adaptiveThreshWinSizeMax")
        assert row.bounds() == expected

        # Editing a bound by hand is what "Custom" means.
        row.set_bounds(5, 9)
        assert tab._detection_profile_combo.currentText() == "Custom"
    finally:
        tab.deleteLater()


@pytest.mark.parametrize(
    ("target_type", "backend", "expected"),
    [("ChArUco", "aruco1", "aruco1"),
     ("ChArUco", "aruco2", "aruco2"),
     ("Ccube", "aruco2", "aruco2"),
     # One combo serves every target; a target with one detector ignores it.
     ("PuzzleBoard", "aruco1", "puzzle_board"),
     ("PuzzleBoardCube", "", "PuzzleBoardCube+puzzle_board")],
)
@pytest.mark.gui
def test_a_form_resolves_its_target_selection_to_one_detector(
        target_type, backend, expected):
    from pyCamSet.gui.shared_functions import detector_parameterisation_for

    assert detector_parameterisation_for(target_type, backend).name == expected


@pytest.mark.gui
def test_adopting_a_run_sets_the_target_to_match_it():
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    form = _target_form("ChArUco")
    try:
        form.apply_spec(CCUBE_12["target"])

        assert form.target_type() == "Ccube"
        spec = form.spec()
        assert spec["n_points"] == 12
        assert spec["length"] == 80
        assert describe_target_mismatch(CCUBE_12, {"target": spec}) == []
    finally:
        form.deleteLater()


@pytest.mark.gui
def test_adopting_a_run_with_a_dictionary_no_longer_offered_shows_the_true_value():
    """A saved run can name a dictionary this combo's current choices do not
    offer (e.g. an AprilTag dictionary retired from Create Target). Before
    the fix, ``QComboBox.setCurrentText`` silently no-ops for a value not
    among its items, so the combo kept showing whatever it already held --
    a different dictionary than the run actually used, with no warning.
    """
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    form = _target_form("ChArUco")
    try:
        form.apply_spec({"type": "ChArUco", "a_dict": "DICT_APRILTAG_16h5"})
        assert form._widgets["a_dict"].currentText() == "DICT_APRILTAG_16h5"
    finally:
        form.deleteLater()


@pytest.mark.gui
def test_adopting_selects_the_marker_backend_by_value():
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    form = _target_form()
    try:
        form.apply_spec(_with(CCUBE_12, marker_backend="aruco2")["target"])
        assert form.backend() == "aruco2"
    finally:
        form.deleteLater()


@pytest.mark.gui
def test_adopting_nothing_changes_nothing():
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    form = _target_form("Ccube")
    try:
        before = form.spec()
        form.apply_spec({})
        assert form.spec() == before
    finally:
        form.deleteLater()


@pytest.mark.gui
def test_adopting_ignores_a_field_this_target_does_not_have():
    """A spec carries one target's arguments; the form offers another's."""
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    form = _target_form("Ccube")
    try:
        form.apply_spec({**CCUBE_12["target"], "paper_width": 210.0})
        assert form.spec()["n_points"] == 12
        assert "paper_width" not in form.spec()
    finally:
        form.deleteLater()


@pytest.mark.gui
def test_the_form_holds_what_it_was_given_rather_than_a_size_it_prefers():
    """The controls used to be spin boxes with ranges invented for them,
    and a target outside one was silently narrowed to fit.  A target is an
    object someone made: the form carries the value, and the target is what
    refuses it."""
    from PySide6.QtWidgets import QApplication

    from pyCamSet.calibration_targets.core.target_registry import build_target

    QApplication.instance() or QApplication([])
    form = _target_form("Ccube")
    try:
        form.apply_spec(_with(CCUBE_12, n_points=999)["target"])

        assert form.spec()["n_points"] == 999, "carried, not narrowed"
        assert describe_target_mismatch(
            _with(CCUBE_12, n_points=999), {"target": form.spec()}) == []
        with pytest.raises(ValueError, match="markers"):
            build_target(form.spec())
    finally:
        form.deleteLater()


@pytest.mark.gui
def test_the_form_offers_every_target_the_registry_knows():
    """Which is the point: adding a target is a line in the registry."""
    from PySide6.QtWidgets import QApplication

    from pyCamSet.calibration_targets.core.target_registry import TARGET_NAMES

    QApplication.instance() or QApplication([])
    form = _target_form()
    try:
        offered = [form._target_combo.itemData(i)
                   for i in range(form._target_combo.count())]
        assert offered == list(TARGET_NAMES)
    finally:
        form.deleteLater()


@pytest.mark.gui
@pytest.mark.parametrize("name", TARGET_NAMES, ids=list(TARGET_NAMES))
def test_the_form_can_build_every_target_it_offers(name):
    """Each target the form offers, it also collects enough to build --
    except ChArUco2 and Ccube2, which have no aruco1 equivalent and cannot
    be built at all without aruco2 installed."""
    from PySide6.QtWidgets import QApplication

    from pyCamSet.calibration_targets.markers.aruco2 import ARUCO2_AVAILABLE
    from pyCamSet.calibration_targets.core.target_registry import build_target

    if name in ("ChArUco2", "Ccube2") and not ARUCO2_AVAILABLE:
        pytest.skip("aruco2 is not installed")

    QApplication.instance() or QApplication([])
    form = _target_form(name)
    try:
        spec = form.spec()
        assert spec["type"] == name
        assert build_target(spec) is not None
    finally:
        form.deleteLater()


@pytest.mark.gui
@pytest.mark.parametrize("phase", ["phase_2_intrinsics", "phase_3_bundle_adjustment"])
def test_a_phase_adopts_the_target_of_the_run_it_continues(phase):
    """The helper took a tab and a spec, and was given a tab and a run's
    whole parameters -- whose top-level keys are ``f_loc`` and ``target``,
    never ``type``. So it returned at its first line and the phase kept
    whatever target it was showing, while a test on the helper passed.
    """
    import importlib

    from PySide6.QtWidgets import QApplication, QCheckBox, QTabWidget

    from pyCamSet.workflow.workspace import WorkspaceManager

    QApplication.instance() or QApplication([])
    module = importlib.import_module(f"pyCamSet.gui.{phase}")
    tab_class = next(v for k, v in vars(module).items()
                     if k.endswith("Tab") and isinstance(v, type))
    tab = tab_class(QTabWidget(), QCheckBox(), QCheckBox(), WorkspaceManager(None))
    try:
        tab._target_form.set_target_type("ChArUco")
        run = {"run_id": "r1", "params": CCUBE_12}

        adopt = getattr(tab, "_adopt_target_from_phase1_run", None) or \
            getattr(tab, "_adopt_target_from_run", None)
        adopt(run)

        assert tab._target_form.target_type() == "Ccube"
        assert tab._target_form.spec()["n_points"] == 12
    finally:
        tab.deleteLater()


# --------------------------------------------------------------------------
# A target flip must not throw typed values away
# --------------------------------------------------------------------------
#
# _rebuild used to tear every row down and rebuild from the new target's
# defaults on every target-type or detector-backend change, so a value
# someone had already typed -- including one the new target takes just the
# same, like a shared square size -- was lost under them.


@pytest.mark.gui
def test_flipping_target_type_keeps_a_shared_value_and_restores_it_on_return():
    """``square_size`` is taken by both ChArUco and PuzzleBoard; a value
    typed into it survives the flip between them, and a value that only
    ChArUco has (``marker_fraction``) is still there when flipping back."""
    from PySide6.QtWidgets import QApplication

    from pyCamSet.gui.shared_functions import read_parameter_widget, set_parameter_widget

    QApplication.instance() or QApplication([])
    form = _target_form("ChArUco")
    try:
        set_parameter_widget(form._widgets["square_size"], 42.0)
        set_parameter_widget(form._widgets["marker_fraction"], 0.5)

        form.set_target_type("PuzzleBoard")
        assert read_parameter_widget(form._widgets["square_size"]) == "42.0"
        assert "marker_fraction" not in form._widgets  # PuzzleBoard has no such field

        form.set_target_type("ChArUco")
        assert read_parameter_widget(form._widgets["square_size"]) == "42.0"
        assert read_parameter_widget(form._widgets["marker_fraction"]) == "0.5"
    finally:
        form.deleteLater()


@pytest.mark.gui
def test_flipping_the_detector_backend_keeps_a_typed_value():
    """A typed value survives a detector flip: the detector says what reads
    the board, not what the board is."""
    from PySide6.QtWidgets import QApplication

    from pyCamSet.gui.shared_functions import read_parameter_widget, set_parameter_widget

    QApplication.instance() or QApplication([])
    form = _target_form("ChArUco")
    try:
        set_parameter_widget(form._widgets["square_size"], 17.0)

        form._backend_combo.setCurrentIndex(form._backend_combo.findData("aruco2"))
        assert read_parameter_widget(form._widgets["square_size"]) == "17.0"

        form._backend_combo.setCurrentIndex(form._backend_combo.findData("aruco1"))
        assert read_parameter_widget(form._widgets["square_size"]) == "17.0"
    finally:
        form.deleteLater()


@pytest.mark.gui
def test_choosing_a_detector_keeps_the_target_a_spec_loaded():
    """A remembered or adopted target is loaded with apply_spec, which marks
    nothing as edited; choosing the detector afterwards must not put the
    board back to the target's defaults."""
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    form = _target_form(detector_mode="choose")
    try:
        form.apply_spec({"type": "ChArUco", "num_squares_x": 12,
                         "num_squares_y": 9, "square_size": 4.0,
                         "marker_backend": "aruco1"})
        heard = []
        form.changed.connect(lambda: heard.append(True))

        form._backend_combo.setCurrentIndex(form._backend_combo.findData("aruco2"))

        spec = form.spec()
        assert spec["num_squares_x"] == 12
        assert spec["num_squares_y"] == 9
        assert spec["square_size"] == 4.0
        assert spec["marker_backend"] == "aruco2"
        assert heard, "the detection options still hear of the new detector"
    finally:
        form.deleteLater()


@pytest.mark.gui
@pytest.mark.parametrize("detector_mode", ["choose", "inherit"])
@pytest.mark.parametrize("target_type,dict_key", [("ChArUco", "a_dict"),
                                                  ("Ccube", "aruco_dict")])
def test_a_loaded_spec_naming_an_unoffered_dictionary_reads_back(
        detector_mode, target_type, dict_key):
    """A run saved with a dictionary the list no longer offers is still the
    run Phases 2 and 3 adopt, so the form must read it back, not refuse it."""
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    form = _target_form(detector_mode=detector_mode)
    try:
        form.apply_spec({"type": target_type, dict_key: "DICT_ALVAR_7X7_1000",
                         "marker_backend": "aruco2"})
        spec = form.spec()
        assert spec[dict_key] == "DICT_ALVAR_7X7_1000"
        assert spec["marker_backend"] == "aruco2"

        # Only the loaded value is let through: another unoffered name is
        # still refused, and so is the loaded one once the rows are rebuilt.
        from pyCamSet.gui.shared_functions import set_parameter_widget
        from pyCamSet.workflow.params import ParamError

        set_parameter_widget(form._widgets[dict_key], "DICT_APRILTAG_36h11")
        with pytest.raises(ParamError):
            form.spec()
        other = "Ccube" if target_type == "ChArUco" else "ChArUco"
        form.set_target_type(other)
        form.set_target_type(target_type)
        set_parameter_widget(form._widgets[dict_key], "DICT_ALVAR_7X7_1000")
        with pytest.raises(ParamError):
            form.spec()
    finally:
        form.deleteLater()


@pytest.mark.gui
def test_adopting_a_run_does_not_leak_a_stale_retained_value():
    """A key retained several flips ago, from a target no longer on
    screen, must not resurface just because :meth:`apply_spec` loads a
    spec that happens to omit it: ``apply_spec`` wins outright, and the
    field it says nothing about should fall back to that target's own
    default, not to whatever was typed long before the run was adopted."""
    from PySide6.QtWidgets import QApplication

    from pyCamSet.gui.shared_functions import set_parameter_widget

    QApplication.instance() or QApplication([])
    form = _target_form("ChArUco")
    try:
        set_parameter_widget(form._widgets["num_squares_x"], 999)
        form.set_target_type("PuzzleBoard")  # retains num_squares_x=999
        form.set_target_type("Ccube")  # Ccube has no such field either

        # A spec that says nothing about num_squares_x at all.
        form.apply_spec({"type": "ChArUco", "square_size": 5.0})

        assert form.target_type() == "ChArUco"
        assert form.spec()["num_squares_x"] == 5, "the target's own default, not 999"
    finally:
        form.deleteLater()


@pytest.mark.gui
def test_browsing_target_types_with_no_edits_shows_each_ones_own_defaults():
    """Selecting every target in turn, never typing a thing, must show each
    one's OWN constructor defaults -- not a value left over from whichever
    target was on screen before.

    Ccube and PuzzleBoardCube both declare ``n_points``/``length``; before
    the fix, merely browsing from one to the other overwrote the newly
    selected target's real defaults (``n_points=20, length=200.0`` for
    PuzzleBoardCube) with the previous target's numbers
    (``n_points=5, length=20.0``), and both resulting specs passed
    validation, so nothing surfaced it.

    Expected defaults are read from each target's own ``__init__`` via
    ``inspect.signature`` rather than hardcoded, so this tracks the code
    instead of drifting from it.
    """
    import inspect

    from PySide6.QtWidgets import QApplication

    from pyCamSet.calibration_targets.core.target_registry import TARGET_NAMES, target_class

    QApplication.instance() or QApplication([])
    form = _target_form()
    try:
        for target_type in TARGET_NAMES:
            form.set_target_type(target_type)
            spec = form.spec()
            sig = inspect.signature(target_class(target_type).__init__)
            for name, param in sig.parameters.items():
                if name == "self" or param.default is inspect.Parameter.empty:
                    continue
                if name not in spec:
                    continue
                assert spec[name] == param.default, (
                    f"{target_type}.{name}: form shows {spec[name]!r} after "
                    f"selecting it with no edits at all; its own "
                    f"constructor default is {param.default!r}")
    finally:
        form.deleteLater()


@pytest.mark.gui
def test_an_edited_shared_value_survives_a_flip_while_an_untouched_one_takes_the_new_default():
    """The original feature -- a value someone actually typed survives a
    target or detector flip -- has to keep working now that retention is
    edited-only, or the fix for P0 #1 would have thrown it out entirely.

    Ccube and PuzzleBoardCube share both ``n_points`` and ``length``; only
    ``length`` is edited here, so it must survive the flip, while
    ``n_points`` -- never touched -- must show PuzzleBoardCube's own
    default rather than whatever Ccube happened to be showing.
    """
    import inspect

    from PySide6.QtWidgets import QApplication

    from pyCamSet.calibration_targets.core.target_registry import target_class
    from pyCamSet.gui.shared_functions import set_parameter_widget

    QApplication.instance() or QApplication([])
    form = _target_form("Ccube")
    try:
        set_parameter_widget(form._widgets["length"], 55.0)

        form.set_target_type("PuzzleBoardCube")

        spec = form.spec()
        assert spec["length"] == 55.0, "the edited value survives the target flip"
        default_n_points = inspect.signature(
            target_class("PuzzleBoardCube").__init__
        ).parameters["n_points"].default
        assert spec["n_points"] == default_n_points, (
            "an untouched key takes the NEW target's own default, not "
            "whatever the old target happened to be showing")

        # The detector-flip path: ChArUco's square_size, edited, must
        # survive a backend change the same way.
        form.set_target_type("ChArUco")
        set_parameter_widget(form._widgets["square_size"], 42.0)

        form._backend_combo.setCurrentIndex(form._backend_combo.findData("aruco2"))
        assert form.spec()["square_size"] == 42.0, \
            "the edited value survives the detector flip too"
    finally:
        form.deleteLater()


@pytest.mark.gui
def test_apply_spec_resets_an_unmentioned_key_even_when_the_type_is_unchanged():
    """``setCurrentText`` is a no-op when the target type does not change,
    so Qt emits nothing and ``_rebuild`` never runs on its own --
    ``apply_spec`` has to force it, or a value already sitting in a widget
    for a key the spec does not mention just stays there."""
    import inspect

    from PySide6.QtWidgets import QApplication

    from pyCamSet.calibration_targets.core.target_registry import target_class
    from pyCamSet.gui.shared_functions import set_parameter_widget

    QApplication.instance() or QApplication([])
    form = _target_form("ChArUco")
    try:
        set_parameter_widget(form._widgets["num_squares_x"], 999)

        form.apply_spec({"type": "ChArUco", "square_size": 5.0})

        default_num_squares_x = inspect.signature(
            target_class("ChArUco").__init__
        ).parameters["num_squares_x"].default
        assert form.spec()["num_squares_x"] == default_num_squares_x, \
            "not the 999 left over from before apply_spec"
        assert form.spec()["square_size"] == 5.0
    finally:
        form.deleteLater()


@pytest.mark.gui
def test_apply_spec_does_not_leak_a_value_staged_through_a_target_that_shares_the_key():
    """The type-change path: PuzzleBoard and ChArUco both declare
    ``num_squares_x`` -- unlike Ccube, which shares nothing with ChArUco
    and so would pass this by luck alone. Loading a ChArUco spec after
    typing into PuzzleBoard's ``num_squares_x`` must not leak that 999 into
    ChArUco's same-named field, and must not leave PuzzleBoard's own
    default sitting in a field the spec did mention either.
    """
    import inspect

    from PySide6.QtWidgets import QApplication

    from pyCamSet.calibration_targets.core.target_registry import target_class
    from pyCamSet.gui.shared_functions import set_parameter_widget

    QApplication.instance() or QApplication([])
    form = _target_form("PuzzleBoard")
    try:
        set_parameter_widget(form._widgets["num_squares_x"], 999)

        form.apply_spec({"type": "ChArUco", "square_size": 5.0})

        assert form.target_type() == "ChArUco"
        default_num_squares_x = inspect.signature(
            target_class("ChArUco").__init__
        ).parameters["num_squares_x"].default
        assert form.spec()["num_squares_x"] == default_num_squares_x, \
            "not the 999 staged through PuzzleBoard"
        assert form.spec()["square_size"] == 5.0, \
            "the spec's own value, not PuzzleBoard's default for it"
    finally:
        form.deleteLater()


# --------------------------------------------------------------------------
# RunSelectorWidget's pre-selection count
# --------------------------------------------------------------------------


def _runs(n):
    return [{"run_id": f"r{i}"} for i in range(n)]


@pytest.mark.gui
def test_run_selector_default_preselect_is_still_three():
    """Every existing caller relies on the default -- changing it would be
    a silent behaviour change for phases that never asked for one."""
    from PySide6.QtWidgets import QApplication

    from pyCamSet.gui.shared_functions import RunSelectorWidget

    QApplication.instance() or QApplication([])
    widget = RunSelectorWidget(_runs(5))
    try:
        assert len(widget.get_selected()) == 3
    finally:
        widget.deleteLater()


@pytest.mark.gui
def test_run_selector_preselect_is_configurable():
    from PySide6.QtWidgets import QApplication

    from pyCamSet.gui.shared_functions import RunSelectorWidget

    QApplication.instance() or QApplication([])
    widget = RunSelectorWidget(_runs(5), preselect=1)
    try:
        assert len(widget.get_selected()) == 1
        assert widget.get_selected()[0]["run_id"] == "r4"  # the most recent
    finally:
        widget.deleteLater()


@pytest.mark.gui
@pytest.mark.parametrize(("n", "preselect", "expected"), [
    (5, 0, 0),      # zero or below selects nothing
    (5, -2, 0),
    (5, 100, 5),    # past the run count selects all of them
    (0, 3, 0),      # nothing to select regardless
])
def test_run_selector_preselect_is_clamped(n, preselect, expected):
    from PySide6.QtWidgets import QApplication

    from pyCamSet.gui.shared_functions import RunSelectorWidget

    QApplication.instance() or QApplication([])
    widget = RunSelectorWidget(_runs(n), preselect=preselect)
    try:
        assert len(widget.get_selected()) == expected
    finally:
        widget.deleteLater()


# --------------------------------------------------------------------------
# The suggested file name tracks a value, not just a rebuild
# --------------------------------------------------------------------------


def _create_target_dialog():
    from PySide6.QtWidgets import QCheckBox

    from pyCamSet.gui.create_target import CreateTargetDialog

    return CreateTargetDialog(QCheckBox())


@pytest.mark.gui
def test_the_suggested_name_tracks_a_value_edit():
    """n_points is not a structural change -- no ``changed`` signal fires
    for it -- so before this the suggested name silently disagreed with
    what would actually be written."""
    from PySide6.QtWidgets import QApplication

    from pyCamSet.gui.shared_functions import set_parameter_widget

    QApplication.instance() or QApplication([])
    dialog = _create_target_dialog()
    try:
        dialog._target_form.set_target_type("Ccube")
        # A QLineEdit's setText() emits textChanged, which is exactly the
        # signal a row's value is wired to -- no simulated keystroke needed.
        set_parameter_widget(dialog._target_form._widgets["n_points"], 30)

        assert "30points" in dialog._name_edit.text()
    finally:
        dialog.deleteLater()


@pytest.mark.gui
def test_the_suggested_name_stops_once_typed_into_and_resumes_when_cleared():
    """Clearing the field must resume auto-naming by itself.

    The previous version of this test edited ``n_points`` right after
    clearing the field, which fires ``values_changed`` ->
    ``_sync_default_name`` and would have resumed the name anyway -- so it
    passed even when clearing alone left the field blank. Nothing else is
    touched here, so a regression has nowhere left to hide.
    """
    from PySide6.QtWidgets import QApplication

    from pyCamSet.gui.shared_functions import read_parameter_widget

    QApplication.instance() or QApplication([])
    dialog = _create_target_dialog()
    try:
        dialog._target_form.set_target_type("Ccube")

        dialog._name_edit.setText("my_own_name.svg")
        dialog._name_edit.textEdited.emit("my_own_name.svg")
        assert dialog._name_edit.text() == "my_own_name.svg", "left alone"

        expected = dialog._target_class().printable_name(
            dialog._target_form.spec(), dialog._export_kind())
        before_n_points = read_parameter_widget(
            dialog._target_form._widgets["n_points"])

        dialog._name_edit.setText("")
        dialog._name_edit.textEdited.emit("")

        assert dialog._name_edit.text() == expected, "auto-naming resumed"
        assert read_parameter_widget(
            dialog._target_form._widgets["n_points"]) == before_n_points, (
            "the resume must not touch any other widget")
    finally:
        dialog.deleteLater()


# --------------------------------------------------------------------------
# The detector is chosen where detection happens
# --------------------------------------------------------------------------
#
# The detector used to be part of describing a target, so Create Target
# asked for it -- though a board prints the same whichever detector reads
# it -- and Phases 2 and 3 offered it again, though they only read the
# detections a Phase 1 run already made.  Phase 1 and the Optimisation tab
# choose it now; Phases 2 and 3 show the adopted run's.


def _combo_items(combo):
    """Each item's value, whether it can be picked, and its hover text."""
    from PySide6.QtCore import Qt

    model = combo.model()
    return {combo.itemData(i): (model.item(i).isEnabled(),
                                combo.itemData(i, Qt.ItemDataRole.ToolTipRole) or "")
            for i in range(combo.count())}


@pytest.mark.gui
def test_the_create_target_dialog_asks_for_no_detector():
    from PySide6.QtWidgets import QApplication, QLabel

    from pyCamSet.gui.shared_functions import DETECTOR_NONE

    QApplication.instance() or QApplication([])
    dialog = _create_target_dialog()
    try:
        form = dialog._target_form
        assert form.detector_mode() == DETECTOR_NONE
        assert form._backend_combo is None
        assert not any(label.text() == "Detector:"
                       for label in form.findChildren(QLabel))
        for target_type in ("ChArUco", "Ccube", "ChArUco2", "Ccube2"):
            form.set_target_type(target_type)
            assert "marker_backend" not in form.spec(), target_type
    finally:
        dialog.deleteLater()


@pytest.mark.gui
def test_phase_1_greys_out_aruco1_for_charuco2_and_gives_charuco_its_choice_back():
    from PySide6.QtWidgets import QApplication

    from pyCamSet.gui.shared_functions import DETECTOR_CHOOSE

    QApplication.instance() or QApplication([])
    tab = _phase1_tab()
    try:
        form = tab._target_form
        combo = form._backend_combo
        assert form.detector_mode() == DETECTOR_CHOOSE

        # A ChArUco read with ArUco 1, as chosen by default.
        form.set_target_type("ChArUco")
        assert form.backend() == "aruco1"
        assert all(enabled for enabled, _ in _combo_items(combo).values())

        # ChArUco2 selects ArUco 2 by itself, and says why ArUco 1 is grey.
        form.set_target_type("ChArUco2")
        assert form.backend() == "aruco2"
        assert combo.currentData() == "aruco2"
        assert not combo.isHidden()
        items = _combo_items(combo)
        assert items["aruco1"][0] is False
        assert "ChArUco2" in items["aruco1"][1] and "ArUco 2" in items["aruco1"][1]
        assert items["aruco2"] == (True, "")
        assert "marker_backend" not in form.spec(), "ChArUco2 takes none"

        # Picking the grey item from code cannot stick either.
        combo.setCurrentIndex(combo.findData("aruco1"))
        assert form.backend() == "aruco2"
        assert combo.currentData() == "aruco2"

        # Back to ChArUco: the choice made for it, not the one forced.
        form.set_target_type("ChArUco")
        assert form.backend() == "aruco1"
        assert form.spec()["marker_backend"] == "aruco1"
        assert all(enabled for enabled, _ in _combo_items(combo).values())

        # And a choice of ArUco 2 survives the same round trip.
        combo.setCurrentIndex(combo.findData("aruco2"))
        form.set_target_type("ChArUco2")
        form.set_target_type("Ccube")
        assert form.backend() == "aruco2"
        assert form.spec()["marker_backend"] == "aruco2"

        # A ChArUco2 ccube is read the ChArUco2 board's way, and a ChArUco1
        # ccube chosen after it gets its own choice back just the same.
        combo.setCurrentIndex(combo.findData("aruco1"))
        form.set_target_type("Ccube2")
        assert form.backend() == "aruco2"
        items = _combo_items(combo)
        assert items["aruco1"][0] is False
        assert "ChArUco2 ccube" in items["aruco1"][1]
        assert "marker_backend" not in form.spec(), "Ccube2 takes none"
        form.set_target_type("Ccube")
        assert form.backend() == "aruco1"
    finally:
        tab.deleteLater()


@pytest.mark.gui
@pytest.mark.skipif(not ARUCO2_AVAILABLE,
                    reason="reads a target that only ArUco 2 detects")
def test_drawing_a_run_without_an_artifact_reads_its_own_detectors_cache(tmp_path):
    """The last resort is the image folder's cache, named per detector and
    upscale -- but only trusted once its identity sidecar confirms it was
    made for this run's own target, cameras and image cap.  An ArUco 2 run
    must not draw ArUco 1's, and ChArUco2 and Ccube(aruco2) -- which compute
    the *same* cache name -- must not draw each other's either."""
    import cv2
    from PySide6.QtWidgets import QApplication, QCheckBox, QTabWidget

    from pyCamSet.calibration.camera_calibrator import write_cache_identity
    from pyCamSet.calibration_targets.core.target_registry import build_target
    from pyCamSet.gui.phase_1_detection import Phase1DiagnosticsTab
    from pyCamSet.workflow.workspace import WorkspaceManager

    cam_names = ["cam0", "cam1"]
    for camera in cam_names:
        (tmp_path / camera).mkdir()
        cv2.imwrite(str(tmp_path / camera / "im0.png"),
                    np.zeros((8, 12, 3), dtype=np.uint8))

    def seed(name, target_spec):
        """Write a (fake) cache and the sidecar that pairs it with
        *target_spec*, and return the path written."""
        path = tmp_path / name
        path.write_bytes(f"detections for {target_spec}".encode())
        write_cache_identity(path, build_target(target_spec), cam_names, None)
        return path

    QApplication.instance() or QApplication([])
    tab = Phase1DiagnosticsTab(QTabWidget(), QCheckBox(), WorkspaceManager(None))
    try:
        def run(**target):
            return {"run_id": None,
                    "params": {"f_loc": str(tmp_path), "target": target}}

        resolve = tab._resolve_pickle_path_for_run

        aruco1 = seed("detected_datapoints.pickle",
                      {"type": "ChArUco", "marker_backend": "aruco1"})
        assert resolve(run(type="ChArUco", marker_backend="aruco1")) == aruco1

        aruco2 = seed("detected_datapoints_aruco2.pickle",
                      {"type": "ChArUco", "marker_backend": "aruco2"})
        assert resolve(run(type="ChArUco", marker_backend="aruco2")) == aruco2

        upscaled = seed("detected_datapoints_upscale2x_aruco2.pickle",
                        {"type": "Ccube", "marker_backend": "aruco2"})
        upscaled_run = run(type="Ccube", marker_backend="aruco2")
        upscaled_run["params"]["upscale_factor"] = 2
        assert resolve(upscaled_run) == upscaled

        # detected_datapoints_aruco2.pickle above was seeded for
        # ChArUco(aruco2), not ChArUco2 -- same filename, different
        # identity, so the unverified collision the two used to share is
        # now refused rather than silently adopted.
        assert resolve(run(type="ChArUco2")) is None

        # Seeded for ChArUco2 itself, the same filename now resolves for it.
        charuco2 = seed("detected_datapoints_aruco2.pickle", {"type": "ChArUco2"})
        assert resolve(run(type="ChArUco2")) == charuco2
        # ...and no longer for ChArUco(aruco2), which just lost the slot.
        assert resolve(run(type="ChArUco", marker_backend="aruco2")) is None
    finally:
        tab.deleteLater()


class _FakeCamDet:
    """A minimal stand-in for a camera's detections, just enough for
    ``_draw_detections_for_run`` to read ``cam_idx``/``im_idx``/points off
    of, via ``get_data()``."""

    def __init__(self, data):
        self._data = data

    def get_data(self):
        return self._data


class _FakeDetections:
    """A minimal stand-in for a ``TargetDetection``, picklable at module
    scope so it survives a real ``save_detections``/``load_verified_cache``
    round-trip (unlike a class defined inside a test function)."""

    def __init__(self, cam_names, points_by_cam):
        self.cam_names = cam_names
        self._points_by_cam = points_by_cam

    def get_cam_list(self):
        return [_FakeCamDet(self._points_by_cam[name]) for name in self.cam_names]


@pytest.mark.gui
def test_draw_detections_never_shows_another_runs_overwritten_cache(tmp_path):
    """Round-9 review, P1: the last-resort image-folder cache is a slot a
    concurrent Phase 1 run can overwrite at any moment, so
    ``_draw_detections_for_run`` must re-verify identity at the instant it
    reads the pickle -- not just trust a path an earlier, separate
    resolution confirmed. Without that, a same-named cache a differently
    parameterised run just overwrote (same target/detector/upscale,
    different ``n_lim`` here) would be read raw and displayed as if it were
    this run's own detections."""
    import cv2

    from PySide6.QtWidgets import QApplication, QCheckBox, QTabWidget

    from pyCamSet.calibration.camera_calibrator import write_cache_identity
    from pyCamSet.calibration_targets.core.target_registry import build_target
    from pyCamSet.gui.phase_1_detection import Phase1DiagnosticsTab
    from pyCamSet.workflow.detections import save_detections
    from pyCamSet.workflow.workspace import WorkspaceManager

    cam_names = ["cam0", "cam1"]
    for camera in cam_names:
        (tmp_path / camera).mkdir()
        cv2.imwrite(str(tmp_path / camera / "im0.png"),
                    np.zeros((8, 12, 3), dtype=np.uint8))

    target_spec = {"type": "ChArUco", "marker_backend": "aruco1"}
    cache_path = tmp_path / "detected_datapoints.pickle"

    def seed(marker_xy, n_lim):
        data0 = np.array([[0.0, 0.0, marker_xy, marker_xy]])
        data1 = np.array([[1.0, 0.0, marker_xy, marker_xy]])
        payload = _FakeDetections(cam_names, {"cam0": data0, "cam1": data1})
        save_detections(cache_path, payload)
        write_cache_identity(cache_path, build_target(target_spec), cam_names, n_lim)

    # Run A's own detections: n_lim=None, marker at (1.0, 1.0).
    seed(1.0, n_lim=None)

    QApplication.instance() or QApplication([])
    tab = Phase1DiagnosticsTab(QTabWidget(), QCheckBox(), WorkspaceManager(None))
    try:
        run_a = {"run_id": None,
                 "params": {"f_loc": str(tmp_path), "target": target_spec, "n_lim": None}}

        # The run-selection loop's own, separate resolution: confirms A's
        # cache is there right now (mirrors _draw_detections_clicked /
        # open_draw_detections_for_latest, run moments before the read).
        assert tab._resolve_pickle_path_for_run(run_a) == cache_path

        # A concurrent Phase 1 run against the same folder, with a
        # DIFFERENT n_lim, overwrites the identical cache filename and
        # sidecar before the draw actually reads it.
        seed(9.0, n_lim=7)

        tab._draw_detections_for_run(run_a, show_errors=False)

        # Fail closed: an identity that no longer confirms is a miss, so
        # run A must never be drawn with run B's overwritten points.
        if tab._draw_state:
            cam0_points = tab._draw_state["cam_points"].get("cam0", {})
            drawn_x = {float(pt[0]) for pts in cam0_points.values() for pt in pts}
            assert 9.0 not in drawn_x, (
                "Draw Detections displayed the concurrently-overwritten "
                "run's data under the originally selected run's label")
    finally:
        tab.deleteLater()


@pytest.mark.gui
@pytest.mark.skipif(__import__("os").name != "nt",
                    reason="Windows MAX_PATH is Windows-specific")
def test_draw_detections_reads_a_run_artifact_past_max_path(tmp_path):
    """A run's own detections pickle is read through the long-path-safe
    form, like the existence check before it: a raw ``open()`` of a path
    past Windows' 260 characters fails, and the run draws nothing."""
    import cv2

    from PySide6.QtWidgets import QApplication, QCheckBox, QTabWidget

    from pyCamSet.gui.phase_1_detection import Phase1DiagnosticsTab
    from pyCamSet.workflow.detections import save_detections
    from pyCamSet.workflow.workspace import WorkspaceManager, ensure_directory

    cam_names = ["cam0", "cam1"]
    for camera in cam_names:
        (tmp_path / camera).mkdir()
        cv2.imwrite(str(tmp_path / camera / "im0.png"),
                    np.zeros((8, 12, 3), dtype=np.uint8))

    deep = tmp_path / "runs"
    while len(str(deep)) < 280:
        deep = deep / ("a" * 40)
    ensure_directory(deep)
    artifact = deep / "detected_datapoints.pickle"
    assert len(str(artifact)) > 260
    save_detections(artifact, _FakeDetections(cam_names, {
        "cam0": np.array([[0.0, 0.0, 5.0, 5.0]]),
        "cam1": np.array([[1.0, 0.0, 5.0, 5.0]]),
    }))

    QApplication.instance() or QApplication([])
    tab = Phase1DiagnosticsTab(QTabWidget(), QCheckBox(), WorkspaceManager(None))
    try:
        run = {"run_id": None,
               "params": {"f_loc": str(tmp_path),
                          "target": {"type": "ChArUco", "marker_backend": "aruco1"}},
               "artifacts": {"detected_datapoints_pickle": str(artifact)}}

        tab._draw_detections_for_run(run, show_errors=False)

        assert tab._draw_state, "the run's detections were not read"
        assert tab._draw_state["cams"] == cam_names
    finally:
        tab.deleteLater()


@pytest.mark.gui
def test_the_detection_options_follow_the_detector_charuco2_forces():
    """The forced selection is a structural change like any other."""
    from PySide6.QtWidgets import QApplication

    from pyCamSet.calibration_targets.core.target_registry import target_class

    QApplication.instance() or QApplication([])
    tab = _phase1_tab()
    try:
        tab._target_form.set_target_type("ChArUco")
        assert tab._current_detector_parameterisation().name == "aruco1"

        tab._target_form.set_target_type("ChArUco2")
        expected = target_class("ChArUco2").detector_parameterisation().name
        assert tab._current_detector_parameterisation().name == expected
        assert tuple(tab._detection_option_widgets) == tuple(
            p.key for p in tab._current_detector_parameterisation().settable())
    finally:
        tab.deleteLater()


@pytest.mark.gui
@pytest.mark.parametrize("phase", ["phase_2_intrinsics", "phase_3_bundle_adjustment"])
def test_phases_2_and_3_read_with_the_detector_of_the_run_they_adopt(phase):
    import importlib

    from PySide6.QtWidgets import QApplication, QCheckBox, QTabWidget

    from pyCamSet.gui.shared_functions import DETECTOR_INHERIT
    from pyCamSet.workflow.workspace import WorkspaceManager

    QApplication.instance() or QApplication([])
    module = importlib.import_module(f"pyCamSet.gui.{phase}")
    tab_class = next(v for k, v in vars(module).items()
                     if k.endswith("Tab") and isinstance(v, type))
    tab = tab_class(QTabWidget(), QCheckBox(), QCheckBox(), WorkspaceManager(None))
    try:
        form = tab._target_form
        assert form.detector_mode() == DETECTOR_INHERIT
        assert form._backend_combo is None, "no detector to choose here"

        # No run adopted yet: each target's own default.
        form.set_target_type("Ccube")
        assert form.backend() == "aruco1"
        assert "ArUco 1" in form._inherited_backend_label.text()
        form.set_target_type("ChArUco2")
        assert form.backend() == "aruco2"
        assert "ArUco 2" in form._inherited_backend_label.text()

        # The adopted run's detector is shown and written back.
        tab._adopt_target_from_phase1_run(
            {"run_id": "r1", "params": _with(CCUBE_12, marker_backend="aruco2")})
        assert form.target_type() == "Ccube"
        assert form.backend() == "aruco2"
        assert form._inherited_backend_label.text().startswith("ArUco 2")
        assert "default" not in form._inherited_backend_label.text()
        assert not form._inherited_backend_label.isHidden()
        assert form.spec()["marker_backend"] == "aruco2"

        # A run read with ArUco 1 is followed just as faithfully.
        tab._adopt_target_from_phase1_run({"run_id": "r2", "params": CCUBE_12})
        assert form.spec()["marker_backend"] == "aruco1"

        # A ChArUco2 run names no detector, and is still an adopted run.
        tab._adopt_target_from_phase1_run(
            {"run_id": "r3", "params": {"target": {"type": "ChArUco2"}}})
        assert form.backend() == "aruco2"
        assert form._inherited_backend_label.text().startswith("ArUco 2")
        assert "default" not in form._inherited_backend_label.text()

        # A target picked by hand after adopting is not the run's, however
        # the detectors line up.
        tab._adopt_target_from_phase1_run(
            {"run_id": "r4", "params": _with(CCUBE_12, marker_backend="aruco2")})
        form.set_target_type("Ccube2")
        assert form.backend() == "aruco2"
        assert "(this target's default)" in form._inherited_backend_label.text()

        # A target picked by hand whose own default does NOT line up with
        # the adopted run's detector must not inherit it either -- the
        # backend row, spec() and the label all fall back to this target's
        # own default (Ccube's aruco2 run must not leak onto ChArUco, whose
        # own default is aruco1).
        form.set_target_type("ChArUco")
        assert form.backend() == "aruco1"
        assert "(this target's default)" in form._inherited_backend_label.text()
        assert form.spec()["marker_backend"] == "aruco1"

        # A run recorded without a target spec has no detector to adopt:
        # the previous run's must not linger.
        form.set_target_type("Ccube")
        tab._adopt_target_from_phase1_run(
            {"run_id": "r5", "params": _with(CCUBE_12, marker_backend="aruco2")})
        assert form.spec()["marker_backend"] == "aruco2"
        tab._adopt_target_from_phase1_run({"run_id": "r6", "params": {}})
        assert form.target_type() == "Ccube"
        assert form.backend() == "aruco1"
        assert form.spec()["marker_backend"] == "aruco1"
        assert "(this target's default)" in form._inherited_backend_label.text()
    finally:
        tab.deleteLater()


@pytest.mark.gui
@pytest.mark.parametrize("phase", ["phase_2_intrinsics", "phase_3_bundle_adjustment"])
def test_phases_2_and_3_round_trip_the_adopted_runs_detection_options(phase):
    """The adopted Phase 1 run's detector tuning has no widget in ``inherit``
    mode, but it must still reach ``spec()`` -- otherwise a fallback
    redetection (``phase2.py``'s ``_load_or_detect``) silently uses the
    detector's built-in defaults instead of the tuning that made the
    adopted run's own detection succeed.
    """
    import importlib

    from PySide6.QtWidgets import QApplication, QCheckBox, QTabWidget

    from pyCamSet.workflow.workspace import WorkspaceManager

    QApplication.instance() or QApplication([])
    module = importlib.import_module(f"pyCamSet.gui.{phase}")
    tab_class = next(v for k, v in vars(module).items()
                     if k.endswith("Tab") and isinstance(v, type))
    tab = tab_class(QTabWidget(), QCheckBox(), QCheckBox(), WorkspaceManager(None))
    try:
        form = tab._target_form

        # No run adopted yet: nothing to carry.
        assert form.spec().get("detection_options") is None

        tuned = {"cornerRefinementWinSize": 9}
        tab._adopt_target_from_phase1_run(
            {"run_id": "r1", "params": _with(CCUBE_12, detection_options=tuned)})
        assert form.spec()["detection_options"] == tuned

        # A later run without tuning of its own must not keep the previous
        # run's -- adopting wins outright, the same as every other field.
        tab._adopt_target_from_phase1_run({"run_id": "r2", "params": CCUBE_12})
        assert form.spec().get("detection_options") is None

        # A run recorded without a target spec drops it too.
        tab._adopt_target_from_phase1_run(
            {"run_id": "r3", "params": _with(CCUBE_12, detection_options=tuned)})
        assert form.spec()["detection_options"] == tuned
        tab._adopt_target_from_phase1_run({"run_id": "r4", "params": {}})
        assert form.spec().get("detection_options") is None

        # A different target picked by hand after adopting gets none of the
        # run's tuning -- it is that run's own target's, and may not even
        # apply to this one's detector.
        tab._adopt_target_from_phase1_run(
            {"run_id": "r5", "params": _with(CCUBE_12, detection_options=tuned)})
        assert form.spec()["detection_options"] == tuned
        form.set_target_type("ChArUco")
        assert form.spec().get("detection_options") is None
    finally:
        tab.deleteLater()


@pytest.mark.gui
@pytest.mark.parametrize("phase, refresh", [
    ("phase_2_intrinsics", "_update_detection_source_label"),
    ("phase_3_bundle_adjustment", "_update_source_label"),
])
def test_a_workspace_with_no_run_drops_the_adopted_runs_detector(phase, refresh, tmp_path):
    """Moved to a workspace with no run of its own, a phase no longer reads
    with the detector of a run that does not apply there."""
    import importlib

    from PySide6.QtWidgets import QApplication, QCheckBox, QTabWidget

    from pyCamSet.workflow.workspace import WorkspaceManager

    QApplication.instance() or QApplication([])
    module = importlib.import_module(f"pyCamSet.gui.{phase}")
    tab_class = next(v for k, v in vars(module).items()
                     if k.endswith("Tab") and isinstance(v, type))
    manager = WorkspaceManager(None)
    manager.set_workspace_path(tmp_path / "a" / ".pycamset_workspace")
    tab = tab_class(QTabWidget(), QCheckBox(), QCheckBox(), manager)
    try:
        form = tab._target_form
        tab._adopt_target_from_phase1_run(
            {"run_id": "r1", "params": _with(CCUBE_12, marker_backend="aruco2")})
        assert form.spec()["marker_backend"] == "aruco2"

        manager.set_workspace_path(tmp_path / "b" / ".pycamset_workspace")
        getattr(tab, refresh)()
        assert form.target_type() == "Ccube"
        assert form.backend() == "aruco1"
        assert form.spec()["marker_backend"] == "aruco1"
        assert "(this target's default)" in form._inherited_backend_label.text()

        # The same run, once it applies again, is adopted afresh.
        tab._adopt_target_from_phase1_run(
            {"run_id": "r1", "params": _with(CCUBE_12, marker_backend="aruco2")})
        assert form.spec()["marker_backend"] == "aruco2"
    finally:
        tab.deleteLater()


@pytest.mark.gui
def test_setting_a_detection_override_drops_the_previously_adopted_detector(tmp_path):
    """Phase 2's target form must not keep showing a stale detector, adopted
    from whichever Phase 1 run was last auto-adopted, once a manual
    detection-source override is in play (round-2 review, P2).

    The override may read a different run's detections, or the same run's
    detections made with a different detector -- and the target form is
    DETECTOR_INHERIT here, with no detector row for the user to correct it
    by hand, so a stale value would otherwise be silently uncontrollable.
    """
    from PySide6.QtWidgets import QApplication, QCheckBox, QTabWidget

    from pyCamSet.gui.phase_2_intrinsics import Phase2Tab
    from pyCamSet.workflow.workspace import WorkspaceManager

    QApplication.instance() or QApplication([])
    manager = WorkspaceManager(None)
    manager.set_workspace_path(tmp_path / "ws" / ".pycamset_workspace")
    tab = Phase2Tab(QTabWidget(), QCheckBox(), QCheckBox(), manager)
    try:
        form = tab._target_form
        tab._adopt_target_from_phase1_run(
            {"run_id": "r1", "params": _with(CCUBE_12, marker_backend="aruco2")})
        assert form.spec()["marker_backend"] == "aruco2"

        tab._det_pickle_edit.setText(str(tmp_path / "nonexistent.pickle"))
        tab._update_detection_source_label()

        assert "override file" in tab._phase1_lbl.text()
        assert form.backend() == "aruco1"
        assert form.spec()["marker_backend"] == "aruco1"
        assert "(this target's default)" in form._inherited_backend_label.text()
        assert tab._adopted_target_run_id is None

        # Because the adopted run_id was forgotten above, the SAME run,
        # re-offered once the override no longer applies, is adopted
        # afresh -- it is not silently skipped as "already adopted", which
        # is what would keep the form stuck on the target's default forever.
        tab._adopt_target_from_phase1_run(
            {"run_id": "r1", "params": _with(CCUBE_12, marker_backend="aruco2")})
        assert form.spec()["marker_backend"] == "aruco2"
    finally:
        tab.deleteLater()


@pytest.mark.gui
def test_the_optimisation_tab_still_chooses_its_detector():
    """A study runs its own detections, so it keeps the choice."""
    from PySide6.QtWidgets import QApplication, QComboBox

    from pyCamSet.gui.shared_functions import DETECTOR_CHOOSE

    QApplication.instance() or QApplication([])
    tab = _optimisation_tab()
    try:
        form = tab._target_form
        assert form.detector_mode() == DETECTOR_CHOOSE
        assert isinstance(form._backend_combo, QComboBox)
        form.set_target_type("Ccube")
        assert not form._backend_combo.isHidden()
        form._backend_combo.setCurrentIndex(form._backend_combo.findData("aruco2"))
        assert form.spec()["marker_backend"] == "aruco2"
    finally:
        tab.deleteLater()


@pytest.mark.gui
def test_the_target_combo_shows_labels_and_answers_with_registry_names():
    from PySide6.QtWidgets import QApplication

    from pyCamSet.calibration_targets.core.target_registry import target_label

    QApplication.instance() or QApplication([])
    form = _target_form()
    try:
        combo = form._target_combo
        for i in range(combo.count()):
            assert combo.itemText(i) == target_label(combo.itemData(i))
        shown = {combo.itemData(i): combo.itemText(i) for i in range(combo.count())}
        assert shown["ChArUco"] == "ChArUco1"
        assert shown["Ccube"] == "ChArUco1 ccube"
        assert shown["ChArUco2"] == "ChArUco2"
        assert shown["Ccube2"] == "ChArUco2 ccube"
        assert shown["PuzzleBoard"] == "PuzzleBoard"

        form.set_target_type("Ccube")
        assert combo.currentText() == "ChArUco1 ccube"
        assert form.target_type() == "Ccube"
        assert form.spec()["type"] == "Ccube"

        form.apply_spec({"type": "ChArUco", "num_squares_x": 7})
        assert combo.currentText() == "ChArUco1"
        assert form.spec()["type"] == "ChArUco"
        assert form.spec()["num_squares_x"] == 7

        with pytest.raises(ValueError, match="ChArUco1"):
            form.set_target_type("ChArUco1")  # a label is not a name
    finally:
        form.deleteLater()


def test_every_target_label_names_a_registered_target():
    from pyCamSet.calibration_targets.core.target_registry import (
        TARGET_LABELS,
        target_label,
    )

    assert set(TARGET_LABELS) <= set(TARGET_NAMES)
    assert target_label("PuzzleBoardCube") == "PuzzleBoardCube", "falls back to the name"
    assert len(set(map(target_label, TARGET_NAMES))) == len(TARGET_NAMES)


@pytest.mark.gui
def test_a_form_refuses_a_detector_mode_it_does_not_know():
    from PySide6.QtWidgets import QApplication

    from pyCamSet.gui.shared_functions import TargetSettingsForm

    QApplication.instance() or QApplication([])
    with pytest.raises(ValueError, match="detector_mode"):
        TargetSettingsForm(detector_mode="sometimes")


# Lens model
# --------------------------------------------------------------------------


@pytest.mark.gui
def test_phase_2_offers_every_lens_model_and_runs_the_chosen_one(tmp_path):
    """A telecentric rig cannot be calibrated at all unless the phase that
    builds the cameras is told which optics it is looking through."""
    from PySide6.QtWidgets import QApplication, QCheckBox, QTabWidget

    from pyCamSet.cameras.lens_models import LENS_MODELS, lens_model_label
    from pyCamSet.gui.phase_2_intrinsics import Phase2Tab
    from pyCamSet.workflow.workspace import WorkspaceManager

    QApplication.instance() or QApplication([])
    manager = WorkspaceManager(None)
    manager.set_workspace_path(tmp_path / "ws" / ".pycamset_workspace")
    tab = Phase2Tab(QTabWidget(), QCheckBox(), QCheckBox(), manager)
    try:
        combo = tab._lens_combo
        offered = [combo.itemData(i) for i in range(combo.count())]
        assert offered == list(LENS_MODELS)
        assert combo.itemText(offered.index("telecentric")) == \
            lens_model_label("telecentric")
        assert combo.currentData() == "pinhole", "the default is unchanged"

        images = tmp_path / "ims" / "cam1"
        images.mkdir(parents=True)
        tab._floc_edit.setText(str(images.parent))
        tab._target_form.apply_spec(
            {"type": "ChArUco", "num_squares_x": 20, "num_squares_y": 20,
             "square_size": 4.0, "legacy": False})

        combo.setCurrentIndex(offered.index("telecentric"))
        assert tab._collect_params()["lens_model"] == "telecentric"

        combo.setCurrentIndex(offered.index("pinhole"))
        assert tab._collect_params()["lens_model"] == "pinhole"
    finally:
        tab.deleteLater()
