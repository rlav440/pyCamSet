"""Test the shared run-bound education dialog and playback seams.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


def _application():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.mark.gui
def test_run_directory_view_is_read_only_and_metadata_is_recursively_immutable(tmp_path):
    _application()
    from pyCamSet.gui.education_dialog import EducationRunView

    run_dir = tmp_path / ".pycamset_workspace" / "phase1_runs" / "run-1"
    run_dir.mkdir(parents=True)
    metadata = '{"run_id":"run-1","status":"complete","params":{"nested":[1,2]}}\n'
    (run_dir / "metadata.json").write_text(metadata, encoding="utf-8")
    view = EducationRunView.from_run_dir(run_dir)

    assert view.phase == "phase1" and view.status == "complete"
    with pytest.raises(TypeError):
        view.metadata["status"] = "failed"
    with pytest.raises(TypeError):
        view.metadata["params"]["new"] = True
    assert view.metadata["params"]["nested"] == (1, 2)
    assert (run_dir / "metadata.json").read_text(encoding="utf-8") == metadata
    assert sorted(path.name for path in run_dir.parent.parent.iterdir()) == ["phase1_runs"]


@pytest.mark.gui
def test_zero_one_and_multiple_selection_routes_to_chooser_then_phase_factory(monkeypatch):
    _application()
    from PySide6.QtWidgets import QLabel
    from pyCamSet.gui.education_dialog import EducationDialog, EducationRunView

    runs = [EducationRunView("phase2", f"run-{i}", Path("."), "complete", {}, object())
            for i in (1, 2)]
    factory_calls = []
    fake_module = types.ModuleType("pyCamSet.gui.education_phase2")

    def make_view(context, parent=None):
        factory_calls.append((context.run_id, parent))
        return QLabel(f"view for {context.run_id}", parent)

    fake_module.create_education_phase_view = make_view
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    empty = EducationDialog([])
    assert empty._run_list.count() == 1
    assert not empty._open_button.isEnabled()
    chooser = EducationDialog(runs, [])
    assert chooser._stack.currentIndex() == 0
    chooser._open_selection()
    assert chooser._active.run_id == "run-1"
    assert chooser._stack.currentWidget() is chooser._detail
    assert factory_calls[-1][0] == "run-1"
    single = EducationDialog(runs, ["run-2"])
    assert single._active.run_id == "run-2"
    assert factory_calls[-1][0] == "run-2"
    # The header and chooser name the phase as a learner reads it, not by its slug.
    assert single._header_label.text() == (
        "Run run-2 · Phase 2 — Calibrate individual cameras · complete")
    assert chooser._run_list.item(0).text() == (
        "run-1 — Phase 2 — Calibrate individual cameras — complete")
    for dialog in (empty, chooser, single):
        dialog.close()
        dialog.deleteLater()


@pytest.mark.gui
def test_phase_factory_import_is_lazy_and_uses_fixed_contract(monkeypatch):
    _application()
    from PySide6.QtWidgets import QLabel
    from pyCamSet.gui.education_dialog import EducationRunView, discover_phase_view

    run = EducationRunView("phase4", "r", Path("."), "complete", {}, object())
    calls = []
    module = types.ModuleType("pyCamSet.gui.education_phase4")
    module.create_education_phase_view = lambda context, parent=None: (
        calls.append((context, parent)) or QLabel("mock phase view", parent))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert calls == []
    parent = QLabel()
    assert discover_phase_view(run, parent).text() == "mock phase view"
    assert calls == [(run, parent)]


@pytest.mark.gui
def test_recorded_step_controls_are_manual_accessible_and_stop_at_last(tmp_path, monkeypatch):
    _application()
    from pyCamSet.gui import education_preferences
    from pyCamSet.gui.education_dialog import RecordedStep, RecordedStepControls

    path = tmp_path / "education_preferences.json"
    monkeypatch.setattr(education_preferences, "preferences_path", lambda: path)
    steps = [RecordedStep(1, "first"), RecordedStep(3, "middle"), RecordedStep(5, "end")]
    controls = RecordedStepControls(steps, total_steps=5, reduce_motion=False)
    announcements = []
    frames = []
    controls.announcement.connect(announcements.append)
    controls.frameChanged.connect(frames.append)
    assert not controls._timer.isActive()
    assert controls.current_step.frame == "first"
    assert not controls._play.isHidden()
    assert controls._scrubber.accessibleName() == "Recorded solver steps"
    # Without a starting point every position is an accepted step.
    assert "starting point" not in controls._scrubber.accessibleDescription()
    controls.step_forward()
    assert controls.current_step.solver_step == 3
    assert announcements == ["Step 3 of 5 (3 recorded). Steps 2–2 not kept."]
    controls.toggle_playback()
    assert controls._timer.isActive()
    controls._advance_playback()
    assert controls.current_step.solver_step == 5
    assert not controls._timer.isActive()
    assert announcements == ["Step 3 of 5 (3 recorded). Steps 2–2 not kept."]
    assert frames == ["middle", "end"]
    controls.deleteLater()


@pytest.mark.gui
def test_recorded_step_controls_offer_the_starting_point_without_counting_it(
        tmp_path, monkeypatch):
    """A1 section 8: the start frame is shown first but is never step 0."""
    _application()
    from pyCamSet.gui import education_preferences
    from pyCamSet.gui.education_dialog import RecordedStep, RecordedStepControls

    monkeypatch.setattr(education_preferences, "preferences_path",
                        lambda: tmp_path / "education_preferences.json")
    steps = [RecordedStep(1, "first"), RecordedStep(3, "end")]
    emitted = []
    controls = RecordedStepControls(steps, total_steps=3, reduce_motion=False,
                                    start_frame="start")
    controls.frameChanged.connect(emitted.append)
    announcements = []
    controls.announcement.connect(announcements.append)
    assert controls.at_start and controls.current_step is None
    assert controls.current_frame == "start"
    assert controls._status.text() == (
        "Starting point — where the solver began (not an accepted step). "
        "2 of 3 steps recorded.")
    assert not controls._previous.isEnabled() and controls._next.isEnabled()
    assert controls._scrubber.maximum() == 2
    controls.step_forward()
    assert controls.current_step.solver_step == 1
    # The first kept step reports no gap against the starting point.
    assert announcements == ["Step 1 of 3 (2 recorded)."]
    controls.step_forward()
    assert announcements[-1] == "Step 3 of 3 (2 recorded). Steps 2–2 not kept."
    assert not controls._next.isEnabled()
    controls.step_back()
    controls.step_back()
    assert controls.at_start and controls.current_frame == "start"
    assert announcements[-1].startswith("Starting point — where the solver began")
    # Playing from the start runs through every kept step and stops at the last.
    controls.toggle_playback()
    while controls._timer.isActive():
        controls._advance_playback()
    assert controls.current_step.solver_step == 3
    assert emitted == ["first", "end", "first", "start", "first", "end"]
    # Without a start frame the controls behave as before: they open on step 1.
    plain = RecordedStepControls(steps, total_steps=3, reduce_motion=False)
    assert not plain.at_start and plain.current_step.solver_step == 1
    assert plain._scrubber.maximum() == 1
    controls.deleteLater()
    plain.deleteLater()


@pytest.mark.gui
@pytest.mark.parametrize("width", [560, 720, 1100])
def test_recorded_step_figures_stay_put_while_stepping(tmp_path, monkeypatch, width):
    """ux-r5: the status reserves its tallest text, so content below never jumps."""
    app = _application()
    from PySide6.QtWidgets import QVBoxLayout, QWidget
    from pyCamSet.gui import education_preferences
    from pyCamSet.gui.education_dialog import RecordedStep, RecordedStepControls

    monkeypatch.setattr(education_preferences, "preferences_path",
                        lambda: tmp_path / "education_preferences.json")
    view = QWidget()
    layout = QVBoxLayout(view)
    # Step 3 follows a thinned gap, so its status carries the gap sentence.
    controls = RecordedStepControls([RecordedStep(1, "a"), RecordedStep(3, "b")],
                                    total_steps=13, parent=view, reduce_motion=False,
                                    start_frame="start")
    layout.addWidget(controls)
    canvas = QWidget(view)
    canvas.setMinimumHeight(200)
    layout.addWidget(canvas)
    layout.addStretch(1)
    view.resize(width, 900)
    view.show()
    try:
        tops = []
        for move in (None, controls.step_forward, controls.step_forward,
                     controls.step_back, controls.step_back):
            if move is not None:
                move()
            app.processEvents()
            tops.append(canvas.mapTo(view, canvas.rect().topLeft()).y())
        assert controls.at_start
        assert len(set(tops)) == 1, (width, tops)
        # UX-3 (round 1): one compact row. The counter sits in the row beside the scrubber
        # at a width reserved for its longest text; preferences are not in the bar.
        row_y = controls._scrubber.geometry().center().y()
        for part in (controls._first, controls._previous, controls._play, controls._next,
                     controls._status):
            assert part.geometry().center().y() == pytest.approx(row_y, abs=8), part
        assert controls.height() <= 44
        assert controls._status.displayText() in ("Start", "Step 1 of 13", "Step 3 of 13")
        assert not hasattr(controls, "_reduce_motion_control")
        assert not hasattr(controls, "_shortcuts_control")
        # The slider says what position 0 is and has a tooltip like the buttons.
        assert controls._scrubber.accessibleName() == "Recorded solver steps"
        assert "Position 0 is the starting point, not an accepted step" in (
            controls._scrubber.accessibleDescription())
        assert "starting point" in controls._scrubber.toolTip()
    finally:
        view.close()
        view.deleteLater()


@pytest.mark.gui
def test_reduce_motion_stops_playback_hides_play_and_moves_focus(tmp_path, monkeypatch):
    app = _application()
    from pyCamSet.gui import education_preferences
    from pyCamSet.gui.education_dialog import RecordedStep, RecordedStepControls

    monkeypatch.setattr(education_preferences, "preferences_path",
                        lambda: tmp_path / "education_preferences.json")
    controls = RecordedStepControls([RecordedStep(1, 0), RecordedStep(2, 1)], reduce_motion=False)
    controls.show()
    controls._play.setFocus()
    controls.toggle_playback()
    assert controls._timer.isActive()
    controls.set_reduce_motion(True)
    app.processEvents()
    assert not controls._timer.isActive()
    assert not controls._play.isVisible()
    assert not controls._play.hasFocus()
    assert education_preferences.load_preferences()["reduce_motion"] is True
    controls.deleteLater()


@pytest.mark.gui
def test_shortcut_table_has_modifier_chords_and_reserved_conflict_gate():
    _application()
    from PySide6.QtGui import QKeySequence
    from pyCamSet.gui.education_dialog import PLAYBACK_SHORTCUTS, shortcut_conflicts

    assert all(QKeySequence(chord).count() == 1 and
               QKeySequence(chord).toString().startswith(("Ctrl+", "Alt+", "Meta+"))
               for chord in PLAYBACK_SHORTCUTS.values())
    assert shortcut_conflicts() == ()
    assert shortcut_conflicts({"bad": "Alt+F4"}) == ("Alt+F4",)
    assert shortcut_conflicts({"a": "Ctrl+1", "b": "Ctrl+1"}) == ("Ctrl+1",)


@pytest.mark.gui
def test_modifier_shortcuts_are_inactive_in_combo_editor_and_3d_view():
    app = _application()
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QComboBox, QLineEdit, QPushButton
    from pyCamSet.gui.education_dialog import RecordedStep, RecordedStepControls

    controls = RecordedStepControls([RecordedStep(1, "one"), RecordedStep(2, "two")])
    combo = QComboBox(controls)
    combo.addItems(("first", "second"))
    editor = QLineEdit(controls)
    render = QPushButton("3D placeholder", controls)
    render.setProperty("education3DView", True)
    controls.layout().addWidget(combo)
    controls.layout().addWidget(editor)
    controls.layout().addWidget(render)
    controls.show()

    controls._set_index(1, announce=False)
    combo.setFocus()
    app.processEvents()
    assert not controls._shortcuts["next"].isEnabled()
    QTest.keyClick(combo, Qt.Key.Key_Down, Qt.KeyboardModifier.AltModifier)
    app.processEvents()
    assert combo.view().isVisible(), "Alt+Down should remain native combo-box input"
    assert controls.current_step.solver_step == 2
    combo.hidePopup()

    for blocked_widget in (editor, render):
        blocked_widget.setFocus()
        app.processEvents()
        assert not controls._shortcuts["next"].isEnabled()
        QTest.keyClick(blocked_widget, Qt.Key.Key_Right,
                       Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier)
        app.processEvents()
        assert controls.current_step.solver_step == 2
    controls.close()
    controls.deleteLater()


@pytest.mark.gui
def test_shortcuts_pause_while_scrubber_has_focus_and_can_be_disabled(tmp_path, monkeypatch):
    _application()
    from pyCamSet.gui import education_preferences
    from pyCamSet.gui.education_dialog import RecordedStep, RecordedStepControls

    monkeypatch.setattr(education_preferences, "preferences_path",
                        lambda: tmp_path / "education_preferences.json")
    controls = RecordedStepControls([RecordedStep(1, 0), RecordedStep(2, 1)])
    controls._update_shortcut_focus(None, controls._scrubber)
    assert controls._shortcuts["toggle"].isEnabled()
    assert not controls._shortcuts["previous"].isEnabled()
    assert not controls._shortcuts["next"].isEnabled()
    controls.set_shortcuts_enabled(False)
    assert education_preferences.load_preferences()["shortcuts_enabled"] is False
    controls.deleteLater()


@pytest.mark.gui
def test_render_widget_filter_blocks_vtk_character_keys_by_key_not_text():
    _application()
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QLabel
    from pyCamSet.gui.education_dialog import RenderWindowKeyFilter

    target = QLabel()
    key_filter = RenderWindowKeyFilter(target)
    blocked = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_W, Qt.KeyboardModifier.NoModifier,
                        "w")
    allowed = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Tab, Qt.KeyboardModifier.NoModifier,
                        "\t")
    assert key_filter.eventFilter(target, blocked) is True
    assert key_filter.eventFilter(target, allowed) is False
    target.deleteLater()


@pytest.mark.gui
def test_render_window_filter_is_installed_and_escape_returns_focus_before_close():
    app = _application()
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QLabel
    from pyCamSet.gui.education_dialog import (
        EducationDialog, EducationRunView, install_render_window_key_filter,
    )

    render = QLabel("render surface")
    key_filter = install_render_window_key_filter(render)
    assert render.property("education3DView") is True
    assert render._education_key_filter is key_filter

    run = EducationRunView("phase2", "escape-test", Path("."), "complete", {}, object())
    dialog = EducationDialog([run], [run.run_id])
    dialog.show()
    QTest.qWaitForWindowExposed(dialog)
    dialog._run_list.setFocus()
    QTest.keyClick(dialog._run_list, Qt.Key.Key_Escape)
    app.processEvents()
    assert dialog.isVisible()
    assert dialog._change_button.hasFocus()
    QTest.keyClick(dialog._change_button, Qt.Key.Key_Escape)
    app.processEvents()
    assert not dialog.isVisible()
    dialog.deleteLater()
    render.deleteLater()


@pytest.mark.gui
def test_render_filter_blocks_real_qt_key_delivery_and_unfiltered_control_receives_it():
    """Exercise QApplication delivery, not only a direct call to eventFilter."""
    _application()
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QWidget
    from pyCamSet.gui.education_dialog import install_render_window_key_filter

    class KeyRecorder(QWidget):
        def __init__(self):
            super().__init__()
            self.keys = []

        def keyPressEvent(self, event):
            self.keys.append(event.key())
            super().keyPressEvent(event)

    control = KeyRecorder()
    control.show()
    control.setFocus()
    QTest.keyClick(control, Qt.Key.Key_W)
    assert control.keys == [Qt.Key.Key_W], "negative control did not receive a delivered key"

    guarded = KeyRecorder()
    install_render_window_key_filter(guarded)
    guarded.show()
    guarded.setFocus()
    QTest.keyClick(guarded, Qt.Key.Key_W)
    assert guarded.keys == [], "installed Qt event filter failed to consume delivered VTK key"
    guarded.deleteLater()
    control.deleteLater()


@pytest.mark.gui
@pytest.mark.needs_opengl
def test_qtinteractor_suppresses_representation_and_stereo_keys_across_styles():
    """Run native Qt/VTK input assertions only where the suite's GL probe passes."""
    app = _application()
    pytest.importorskip("pyvista")
    pytest.importorskip("pyvistaqt")
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtCore import QObject, QEvent
    from PySide6.QtWidgets import QVBoxLayout, QWidget, QPushButton
    from pyvistaqt import QtInteractor
    from pyCamSet.gui.education_dialog import install_render_window_key_filter

    class AllowedKeyRecorder(QObject):
        def __init__(self):
            super().__init__()
            self.keys = []

        def eventFilter(self, watched, event):
            if event.type() == QEvent.Type.KeyPress:
                self.keys.append(event.key())
            return False

    host = QWidget()
    layout = QVBoxLayout(host)
    interactor = QtInteractor(host, off_screen=False)
    layout.addWidget(interactor)
    after_view = QPushButton("After 3D view", host)
    layout.addWidget(after_view)
    import pyvista as pv

    interactor.add_mesh(pv.Sphere())
    allowed_keys = AllowedKeyRecorder()
    interactor.installEventFilter(allowed_keys)
    install_render_window_key_filter(interactor)
    host.resize(600, 500)
    host.show()
    interactor.render()
    app.processEvents()

    actor = next(iter(interactor.renderer.actors.values()))
    prop = actor.GetProperty()
    for change_style in (interactor.enable_trackball_style, interactor.enable_2d_style):
        change_style()
        app.processEvents()
        assert prop.GetRepresentation() == 2  # VTK_SURFACE
        assert not interactor.render_window.GetStereoRender()
        QTest.keyClick(interactor, Qt.Key.Key_W)
        QTest.keyClick(interactor, Qt.Key.Key_3)
        app.processEvents()
        assert prop.GetRepresentation() == 2  # w must not switch to wireframe
        assert not interactor.render_window.GetStereoRender()  # 3 must not enable stereo
        assert Qt.Key.Key_W not in allowed_keys.keys
        assert Qt.Key.Key_3 not in allowed_keys.keys

    # The allowlist passes through the Qt filter, including focus traversal,
    # the view reset key, Escape, and both arrow-key zoom controls.
    QTest.keyClick(interactor, Qt.Key.Key_Tab)
    app.processEvents()
    assert after_view.hasFocus()
    QTest.keyClick(interactor, Qt.Key.Key_Backtab, Qt.KeyboardModifier.ShiftModifier)
    app.processEvents()
    assert Qt.Key.Key_Backtab in allowed_keys.keys
    for key in (Qt.Key.Key_Escape, Qt.Key.Key_R, Qt.Key.Key_Up, Qt.Key.Key_Down):
        QTest.keyClick(interactor, key)
        app.processEvents()
        assert key in allowed_keys.keys

    interactor.enable_trackball_style()
    renderer = interactor.renderer
    renderer.ResetCamera()
    camera = renderer.GetActiveCamera()
    reset_position = camera.GetPosition()
    camera.SetPosition(reset_position[0] * 4 + 10, reset_position[1] * 4 + 10,
                       reset_position[2] * 4 + 10)
    interactor.render()
    QTest.keyClick(interactor, Qt.Key.Key_R)
    interactor.render()
    assert camera.GetDistance() < 100, "R did not reset the deliberately displaced camera"

    interactor.close()
    host.close()
    interactor.deleteLater()
    host.deleteLater()


@pytest.mark.gui
def test_close_cancels_slow_child_work_before_a_reopened_dialog_can_receive_frames():
    """A delayed renderer callback from a closed shell must not leak into its replacement."""
    app = _application()
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QLabel
    from pyCamSet.gui.education_dialog import EducationDialog, EducationRunView

    frames = []

    class SlowDerivation(QLabel):
        _next_generation = 0

        def __init__(self, parent=None):
            super().__init__("deriving", parent)
            type(self)._next_generation += 1
            self.generation = type(self)._next_generation
            self.timer = QTimer(self)
            self.timer.setSingleShot(True)
            self.timer.timeout.connect(lambda: frames.append(self.generation))
            self.timer.start(100)

        def stop_for_context_change(self):
            self.timer.stop()

    run = EducationRunView("phase2", "slow-close", Path("."), "complete", {}, object())
    module = types.ModuleType("pyCamSet.gui.education_phase2")
    module.create_education_phase_view = lambda context, parent=None: SlowDerivation(parent)
    old_modules = sys.modules.get(module.__name__)
    sys.modules[module.__name__] = module
    try:
        first = EducationDialog([run], [run.run_id])
        first.show()
        app.processEvents()
        assert first._view.timer.isActive()
        first.close()
        second = EducationDialog([run], [run.run_id])
        second.show()
        QTest = __import__("PySide6.QtTest", fromlist=["QTest"]).QTest
        QTest.qWait(140)
        assert frames == [second._view.generation], "closed dialog leaked a stale frame into the reopened view"
        second.close()
        first.deleteLater()
        second.deleteLater()
    finally:
        if old_modules is None:
            sys.modules.pop(module.__name__, None)
        else:
            sys.modules[module.__name__] = old_modules


@pytest.mark.gui
def test_supported_dialog_sizes_keep_shell_controls_inside_visible_geometry():
    app = _application()
    from pyCamSet.gui.education_dialog import EducationDialog

    dialog = EducationDialog([])
    for width, height in ((560, 420), (800, 600), (1280, 800)):
        dialog.resize(width, height)
        dialog.show()
        app.processEvents()
        bounds = dialog.rect()
        for widget in (dialog._run_list, dialog._open_button):
            assert bounds.contains(widget.geometry()), (width, height, widget.objectName())
        dialog.hide()
    dialog.deleteLater()


@pytest.mark.gui
def test_developer_launcher_loads_requested_run_without_entering_event_loop(tmp_path, monkeypatch):
    _application()
    from pyCamSet.gui import education_dialog as module

    run_dir = tmp_path / ".pycamset_workspace" / "phase1_runs" / "pilot-1"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text('{"run_id":"pilot-1"}', encoding="utf-8")
    created = []
    monkeypatch.setattr(module.QApplication, "exec", lambda self: 0)
    original_dialog = module.EducationDialog

    class RecordingDialog(original_dialog):
        def __init__(self, runs, selected_run_ids=(), parent=None):
            created.append((runs, selected_run_ids))
            super().__init__(runs, selected_run_ids, parent)

        def show(self):
            pass

    monkeypatch.setattr(module, "EducationDialog", RecordingDialog)
    assert module.main(["--run", str(run_dir)]) == 0
    assert created[0][0][0].run_id == "pilot-1"
    assert created[0][1] == ["pilot-1"]


@pytest.mark.gui
def test_locked_dialog_palette_tracks_live_theme_changes():
    app = _application()
    from PySide6.QtCore import QCoreApplication, QEvent
    from pyCamSet.gui.education_dialog import EducationDialog
    from pyCamSet.gui.theme import THEME_TOKENS

    app.setProperty("pycamsetTheme", "Light")
    dialog = EducationDialog([])
    assert THEME_TOKENS["Light"]["surface"] in dialog.styleSheet()
    app.setProperty("pycamsetTheme", "Dark")
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DynamicPropertyChange)
    app.processEvents()
    assert THEME_TOKENS["Dark"]["surface"] in dialog.styleSheet()
    dialog.close()
    dialog.deleteLater()
    app.setProperty("pycamsetTheme", "Light")


@pytest.mark.gui
def test_context_change_stops_recorded_step_playback(tmp_path, monkeypatch):
    _application()
    from pyCamSet.gui import education_preferences
    from pyCamSet.gui.education_dialog import RecordedStep, RecordedStepControls

    monkeypatch.setattr(education_preferences, "preferences_path",
                        lambda: tmp_path / "education_preferences.json")
    controls = RecordedStepControls([RecordedStep(1, 0), RecordedStep(2, 1)], reduce_motion=False)
    controls.toggle_playback()
    assert controls._timer.isActive()
    controls.stop_for_context_change()
    assert not controls._timer.isActive()
    controls.deleteLater()


@pytest.mark.gui
def test_locked_education_palette_meets_contrast_on_actual_theme_surface():
    pytest.importorskip("PySide6")
    from pyCamSet.gui.theme import THEME_TOKENS, contrast_ratio
    from pyCamSet.gui.visual_style import education_palette

    for theme in ("Light", "Sepia", "Dark"):
        palette = education_palette(theme)
        surface = THEME_TOKENS[theme]["surface"]
        assert palette["surface"] == surface
        assert palette["text"] == THEME_TOKENS[theme]["text"]
        assert contrast_ratio(palette["text"], surface) >= 4.5
        for role in ("accent", "secondary", "tertiary", "grid"):
            assert contrast_ratio(palette[role], surface) >= 3.0, (theme, role)


@pytest.mark.gui
def test_figure_card_provenance_is_part_of_accessible_description():
    _application()
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from pyCamSet.gui.shared_functions import MatplotlibFigureCard

    card = MatplotlibFigureCard("Residual figure", Figure(), FigureCanvasQTAgg,
                                provenance="saved run r-9; recalculated",
                                alt_text="Three cameras; errors fall from 4 to 1 px.")
    try:
        assert card._canvas.accessibleName() == "Residual figure"
        assert card._canvas.accessibleDescription() == (
            "Three cameras; errors fall from 4 to 1 px. Provenance: saved run r-9; recalculated")
    finally:
        card.deleteLater()


@pytest.mark.gui
def test_playback_shortcuts_are_shown_in_tooltips_and_documented(tmp_path, monkeypatch):
    """F3 item 27: every playback chord is visible on its control and in the how-to."""
    _application()
    from pyCamSet.gui import education_preferences
    from pyCamSet.gui.education_dialog import (
        PLAYBACK_SHORTCUTS, RecordedStep, RecordedStepControls, shortcut_text,
    )

    monkeypatch.setattr(education_preferences, "preferences_path",
                        lambda: tmp_path / "education_preferences.json")
    controls = RecordedStepControls([RecordedStep(1, "a"), RecordedStep(2, "b")],
                                    reduce_motion=False)
    try:
        for button, action in ((controls._previous, "previous"), (controls._next, "next"),
                               (controls._play, "toggle")):
            assert shortcut_text(action) in button.toolTip()
        # The shortcut switch lives in the dialog header's Settings menu (UX-3).
        from pyCamSet.gui.education_dialog import EducationDialog
        dialog = EducationDialog([])
        assert all(shortcut_text(action) in dialog._shortcuts_action.toolTip()
                   for action in PLAYBACK_SHORTCUTS)
        dialog.deleteLater()
    finally:
        controls.deleteLater()
    guide = (Path(__file__).resolve().parents[1] / "docs" / "how-to" / "gui.md").read_text(
        encoding="utf-8")
    for chord in PLAYBACK_SHORTCUTS.values():
        assert f"| {chord} |" in guide


@pytest.mark.gui
def test_dialog_scrolls_tall_explanations_and_playback_controls_fit_minimum_width(
        tmp_path, monkeypatch):
    """At 853x480 a tall view scrolls to its caption; controls never overlap at 560 px."""
    app = _application()
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget
    from pyCamSet.gui import education_preferences
    from pyCamSet.gui.education_dialog import (
        EducationDialog, EducationRunView, RecordedStep, RecordedStepControls,
    )

    monkeypatch.setattr(education_preferences, "preferences_path",
                        lambda: tmp_path / "education_preferences.json")
    built = {}

    def make_view(context, parent=None):
        view = QWidget(parent)
        layout = QVBoxLayout(view)
        figure = QWidget(view)
        figure.setMinimumHeight(900)  # taller than the window, like Phase 4
        layout.addWidget(figure)
        built["caption"] = QLabel("Caption that explains the figure.", view)
        built["caption"].setWordWrap(True)
        layout.addWidget(built["caption"])
        built["controls"] = RecordedStepControls(
            [RecordedStep(1, "a"), RecordedStep(2, "b")], parent=view, reduce_motion=False)
        layout.addWidget(built["controls"])
        return view

    module = types.ModuleType("pyCamSet.gui.education_phase2")
    module.create_education_phase_view = make_view
    monkeypatch.setitem(sys.modules, module.__name__, module)
    run = EducationRunView("phase2", "tall", Path("."), "complete", {}, object())
    dialog = EducationDialog([run], ["tall"])
    try:
        available = dialog.screen().availableGeometry()
        assert dialog.height() <= max(dialog.minimumHeight(), int(available.height() * 0.9))
        for width, height in ((853, 480), (560, 420)):
            dialog.resize(width, height)
            dialog.show()
            app.processEvents()
            scroll = dialog._phase_scroll
            assert scroll.verticalScrollBar().maximum() > 0
            # Offscreen test fonts are about twice as wide as Segoe UI, so the
            # reachability check runs at the reviewer's 853 px size; the
            # no-overlap check below runs at the 560 px minimum as well.
            for widget in ((built["caption"], built["controls"]._play) if width == 853 else ()):
                scroll.ensureWidgetVisible(widget)
                app.processEvents()
                top_left = widget.mapTo(scroll.viewport(), widget.rect().topLeft())
                bottom_right = widget.mapTo(scroll.viewport(), widget.rect().bottomRight())
                assert scroll.viewport().rect().contains(top_left), (width, widget)
                assert scroll.viewport().rect().contains(bottom_right), (width, widget)
            controls = built["controls"]
            parts = [controls._first, controls._previous, controls._next, controls._play,
                     controls._scrubber, controls._status]
            rects = [part.geometry().translated(part.parentWidget().mapTo(controls, part.pos())
                                                - part.pos()) for part in parts]
            assert all(not left.intersects(right)
                       for index, left in enumerate(rects) for right in rects[index + 1:]), width
            assert all(controls.rect().contains(rect) for rect in rects), width
            dialog.hide()
    finally:
        dialog.close()
        dialog.deleteLater()


@pytest.mark.gui
def test_explain_selection_hint_names_the_problem(monkeypatch):
    """A Diagnostics click with 0 or 2+ runs selected says why nothing opened."""
    app = _application()
    from PySide6.QtWidgets import QPushButton, QToolTip
    from pyCamSet.gui import run_results
    from pyCamSet.gui.run_results import explain_selection_hint

    announced = []
    real_announce = run_results._announce
    monkeypatch.setattr(run_results, "_announce", lambda widget, text: (
        announced.append((widget, text)), real_announce(widget, text)))
    button = QPushButton("Explain selected run")
    button.show()
    app.processEvents()
    try:
        explain_selection_hint(button, 0)
        assert "No run is selected" in QToolTip.text()
        explain_selection_hint(button, 2)
        assert "2 runs are selected" in QToolTip.text()
        # The transient tooltip is also announced for screen-reader users.
        assert [text for widget, text in announced if widget is button] == [
            "No run is selected. Select one run in the list, then press ‘Explain selected run’.",
            "2 runs are selected. Select only one run to explain."]
    finally:
        QToolTip.hideText()
        button.deleteLater()


@pytest.mark.gui
def test_open_education_run_lists_every_saved_run_so_change_run_can_switch(tmp_path, monkeypatch):
    """Production entry points open one run, but 'Change run' must offer the workspace's others."""
    _application()
    from pyCamSet.gui import education_dialog, run_results

    workspace = tmp_path / ".pycamset_workspace"
    for phase, run_id in (("phase1", "20260101_000000_aaaaaa"), ("phase2", "20260101_000000_bbbbbb"),
                          ("phase2", "20260102_000000_cccccc"), ("phase2", "broken")):
        run_dir = workspace / f"{phase}_runs" / run_id
        run_dir.mkdir(parents=True)
        if run_id != "broken":
            (run_dir / "metadata.json").write_text(
                f'{{"run_id":"{run_id}","status":"complete"}}', encoding="utf-8")
    opened = {}

    class FakeDialog:
        def __init__(self, runs, selected_run_ids=(), parent=None):
            opened["runs"] = [(run.phase, run.run_id) for run in runs]
            opened["selected"] = list(selected_run_ids)
            opened["attributes"] = []
            opened["deleted"] = False

        def setAttribute(self, attribute, on=True):  # noqa: N802 - Qt API name
            opened["attributes"].append((attribute, on))

        def exec(self):
            return 0

        def deleteLater(self):  # noqa: N802 - Qt API name
            opened["deleted"] = True

    monkeypatch.setattr(run_results, "education_mode_enabled", lambda: True)
    monkeypatch.setattr(education_dialog, "EducationDialog", FakeDialog)
    clicked = {"run_id": "20260101_000000_bbbbbb",
               "_run_dir": str(workspace / "phase2_runs" / "20260101_000000_bbbbbb")}
    assert run_results.open_education_run(clicked, "phase2", workspace)
    assert opened["selected"] == ["20260101_000000_bbbbbb"]
    # Closing the dialog must release it: delete-on-close is set and deleteLater runs after exec.
    from PySide6.QtCore import Qt
    assert opened["attributes"] == [(Qt.WidgetAttribute.WA_DeleteOnClose, True)]
    assert opened["deleted"] is True
    # Every readable run, grouped by phase and newest first; the unreadable one is left out.
    assert opened["runs"] == [("phase1", "20260101_000000_aaaaaa"),
                              ("phase2", "20260102_000000_cccccc"),
                              ("phase2", "20260101_000000_bbbbbb")]


@pytest.mark.gui
def test_change_run_selects_the_open_run_in_the_chooser(monkeypatch):
    _application()
    from PySide6.QtWidgets import QLabel
    from pyCamSet.gui.education_dialog import EducationDialog, EducationRunView

    fake_module = types.ModuleType("pyCamSet.gui.education_phase2")
    fake_module.create_education_phase_view = lambda context, parent=None: QLabel(context.run_id, parent)
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    runs = [EducationRunView("phase2", f"run-{i}", Path("."), "complete", {}, object())
            for i in (1, 2, 3)]
    dialog = EducationDialog(runs, ["run-3"])
    dialog._change_run()
    assert dialog._stack.currentIndex() == 0
    assert dialog._run_list.currentRow() == 2
    dialog._run_list.setCurrentRow(0)
    dialog._open_selection()
    assert dialog._active.run_id == "run-1"
    dialog.deleteLater()


@pytest.mark.gui
def test_dialog_opens_at_reading_size_not_the_views_minimal_hint(monkeypatch):
    """A small size hint must not make the dialog open cramped; the 90% cap still holds."""
    _application()
    from PySide6.QtWidgets import QLabel
    from pyCamSet.gui.education_dialog import EducationDialog, EducationRunView

    fake_module = types.ModuleType("pyCamSet.gui.education_phase2")
    fake_module.create_education_phase_view = lambda context, parent=None: QLabel("small", parent)
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    run = EducationRunView("phase2", "small", Path("."), "complete", {}, object())
    dialog = EducationDialog([run], ["small"])
    try:
        available = dialog.screen().availableGeometry()
        target_width, target_height = EducationDialog._TARGET_SIZE
        assert dialog.width() == max(dialog.minimumWidth(),
                                     min(target_width, int(available.width() * 0.9)))
        assert dialog.height() == max(dialog.minimumHeight(),
                                      min(target_height, int(available.height() * 0.9)))
    finally:
        dialog.close()
        dialog.deleteLater()


@pytest.mark.gui
def test_settings_menu_drives_reduce_motion_and_shortcuts_in_every_open_bar(tmp_path, monkeypatch):
    """UX-3: preferences live in the dialog header's Settings menu; every bar follows live."""
    _application()
    from pyCamSet.gui import education_preferences
    from pyCamSet.gui.education_dialog import (
        EducationDialog, RecordedStep, RecordedStepControls, TransportBar,
    )

    monkeypatch.setattr(education_preferences, "preferences_path",
                        lambda: tmp_path / "education_preferences.json")
    dialog = EducationDialog([])
    bars = [RecordedStepControls([RecordedStep(1, "a"), RecordedStep(2, "b")]),
            TransportBar(4, names=["One", "Two", "Three", "Four"])]
    try:
        assert all(not bar.reduce_motion for bar in bars)
        assert dialog._settings_button.accessibleName() == "Playback settings"
        bars[0].toggle_playback()
        assert bars[0].is_playing()
        dialog._reduce_motion_action.setChecked(True)
        assert all(bar.reduce_motion for bar in bars)
        assert not bars[0].is_playing() and bars[0]._play.isHidden()
        assert education_preferences.load_preferences()["reduce_motion"] is True
        dialog._shortcuts_action.setChecked(False)
        assert all(not shortcut.isEnabled() for bar in bars
                   for shortcut in bar._shortcuts.values())
        assert education_preferences.load_preferences()["shortcuts_enabled"] is False
        # A bar's own setter persists and reaches the others too.
        bars[1].set_reduce_motion(False)
        assert not bars[0].reduce_motion
    finally:
        for bar in bars:
            bar.deleteLater()
        dialog.deleteLater()


@pytest.mark.gui
def test_transport_bar_named_steps_use_pills_and_keep_one_row(tmp_path, monkeypatch):
    app = _application()
    from pyCamSet.gui import education_preferences
    from pyCamSet.gui.education_dialog import TransportBar

    monkeypatch.setattr(education_preferences, "preferences_path",
                        lambda: tmp_path / "education_preferences.json")
    names = ["Outlines", "Squares", "Read IDs", "Markers", "Corners", "Sub-pixel"]
    bar = TransportBar(len(names), names=names, unit_label="Detector steps",
                       interval_ms=1600, min_interval_ms=400, reduce_motion=False)
    seen, picks = [], []
    bar.positionChanged.connect(seen.append)
    bar.add_menu_choice("Exaggerate", ["×1", "×5", "×20"], 1, picks.append)
    bar.resize(1100, 60)
    bar.show()
    app.processEvents()
    try:
        assert bar._scrubber.isHidden() and bar._pills is not None
        assert bar._pills.count() == len(names)
        bar.step_forward()
        bar._pills.setCurrentIndex(4)
        assert seen == [1, 4] and bar.position() == 4
        assert bar._status.displayText() == "5 / 6"
        assert bar._status.text() == "Detector steps 5 of 6: Corners."
        assert not bar._options.isHidden()
        menu = bar._options_menu.actions()[0].menu()
        menu.actions()[2].trigger()
        assert picks == [2]
        # Play restarts from the first step when it is pressed at the end.
        bar.go_to(5)
        bar.toggle_playback()
        assert bar.is_playing() and bar.position() == 0
        bar.stop_for_context_change()
        assert not bar.is_playing()
        # Everything sits on one row.
        y = bar._play.geometry().center().y()
        for part in (bar._first, bar._previous, bar._next, bar._pills, bar._status, bar.speed):
            assert abs(part.geometry().center().y() - y) <= 8, part
    finally:
        bar.close()
        bar.deleteLater()
