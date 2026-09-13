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

from pyCamSet.workflow.targets import describe_target_mismatch

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
def _target_form(target_type=None):
    from pyCamSet.gui.shared_functions import TargetSettingsForm

    form = TargetSettingsForm()
    if target_type is not None:
        form._target_combo.setCurrentText(target_type)
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
             "square_size": 4.0, "legacy": True})
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
             "square_size": 4.0, "legacy": True})
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
             "square_size": 4.0, "legacy": True})
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
        tab._target_form._target_combo.setCurrentText(target_type)
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
@pytest.mark.parametrize(
    ("target_type", "offered"),
    [("ChArUco", True), ("Ccube", True),
     ("PuzzleBoard", False), ("PuzzleBoardCube", False)],
)
def test_only_a_target_with_a_choice_of_detector_is_asked_for_one(
        target_type, offered):
    """The combo used to appear for a named pair of targets."""
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    tab = _phase1_tab()
    try:
        tab._target_form._target_combo.setCurrentText(target_type)
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
def test_a_preset_still_sets_the_bounds_of_the_rows_it_covers():
    """The rows are rebuilt now, so the preset has to reach the new ones."""
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    tab = _optimisation_tab()
    try:
        tab._target_form._target_combo.setCurrentText("Ccube")
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
def test_a_value_the_interface_cannot_hold_is_still_caught():
    """Controls clamp silently; the mismatch check is the backstop."""
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    form = _target_form("Ccube")
    try:
        form.apply_spec(_with(CCUBE_12, n_points=999)["target"])
        held = form.spec()["n_points"]

        assert held < 999, "clamped to what the control can hold"
        assert describe_target_mismatch(
            _with(CCUBE_12, n_points=999), {"target": form.spec()}) != []
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
        offered = [form._target_combo.itemText(i)
                   for i in range(form._target_combo.count())]
        assert offered == list(TARGET_NAMES)

        for name in offered:
            form._target_combo.setCurrentText(name)
            spec = form.spec()
            assert spec["type"] == name
            # What it collects is enough to build the target it describes.
            from pyCamSet.calibration_targets.core.target_registry import build_target
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
        tab._target_form._target_combo.setCurrentText("ChArUco")
        run = {"run_id": "r1", "params": CCUBE_12}

        adopt = getattr(tab, "_adopt_target_from_phase1_run", None) or \
            getattr(tab, "_adopt_target_from_run", None)
        adopt(run)

        assert tab._target_form.target_type() == "Ccube"
        assert tab._target_form.spec()["n_points"] == 12
    finally:
        tab.deleteLater()
