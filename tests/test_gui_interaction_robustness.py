"""Purpose: Lock down safe keyboard-driven interaction in the Qt widget tree.
Status:  GUI contract tests; no calibration backend execution.
Future:  Extend with native-window and screen-reader checks when those runs are available.
"""
from __future__ import annotations

import pytest


@pytest.fixture

def qt_app():
    """Provide one real QApplication for offscreen interaction tests."""
    pytest.importorskip("PySide6", reason="the GUI is Qt")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wheel_event():
    """Construct one upward wheel gesture for a real Qt widget."""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    return QWheelEvent(
        QPointF(2, 2), QPointF(2, 2), QPoint(), QPoint(0, 120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )


@pytest.mark.gui
@pytest.mark.parametrize("control_kind", ["spin", "combo"])
def test_unfocused_wheel_cannot_change_parameter_controls(qt_app, control_kind):
    """A stray wheel event must not mutate an unfocused numeric or choice input."""
    from PySide6.QtWidgets import QComboBox, QSpinBox

    from pyCamSet.gui.shared_functions import WheelMutationGuard

    control = QSpinBox() if control_kind == "spin" else QComboBox()
    if isinstance(control, QSpinBox):
        control.setRange(0, 10)
        control.setValue(4)
    else:
        control.addItems(["first", "second", "third"])
        control.setCurrentIndex(1)
    control.show()
    control.clearFocus()
    before = control.value() if isinstance(control, QSpinBox) else control.currentIndex()

    guard = WheelMutationGuard(qt_app)
    qt_app.installEventFilter(guard)
    try:
        qt_app.sendEvent(control, _wheel_event())
    finally:
        qt_app.removeEventFilter(guard)
        control.close()

    after = control.value() if isinstance(control, QSpinBox) else control.currentIndex()
    assert after == before


@pytest.mark.gui
def test_focused_numeric_control_keeps_intentional_wheel_input(qt_app):
    """Explicitly focused editing remains available to the user."""
    from PySide6.QtWidgets import QSpinBox

    from pyCamSet.gui.shared_functions import WheelMutationGuard

    control = QSpinBox()
    control.setRange(0, 10)
    control.setValue(4)
    control.show()
    control.setFocus()
    qt_app.processEvents()
    assert control.hasFocus()

    guard = WheelMutationGuard(qt_app)
    qt_app.installEventFilter(guard)
    try:
        qt_app.sendEvent(control, _wheel_event())
    finally:
        qt_app.removeEventFilter(guard)
        control.close()

    assert control.value() == 5


@pytest.mark.gui
def test_phase_zero_key_controls_have_accessible_names(qt_app):
    """Verify names on real user-facing controls, not a mocked widget tree."""
    from pyCamSet.gui.main_window import PyCamSetApp

    window = PyCamSetApp()
    try:
        assert window._info_cb.accessibleName() == "Enable informational tooltips"
        assert window._terminal_cb.accessibleName() == "Show terminal output"
        assert window.phase0_tab._floc_edit.accessibleName() == "Image folder (f_loc)"
        assert window.phase0_tab._confirm_btn.accessibleName() == "Confirm image folder validity"
        assert window.phase0_tab._continue_btn.accessibleName() == "Continue to Phase 1"
        assert window.phase0_tab._status_lbl.accessibleDescription()
    finally:
        window.close()


@pytest.mark.gui
def test_continue_confirmation_defaults_to_no(qt_app, monkeypatch):
    """A run flagged as weak cannot be continued by an accidental Enter."""
    from PySide6.QtWidgets import QMessageBox

    from pyCamSet.gui.shared_functions import make_continue_button, set_continue_blocked

    called = []
    observed = {}

    def warning(_parent, _title, _text, _buttons, default_button):
        observed["default"] = default_button
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "warning", warning)
    button = make_continue_button(lambda: called.append(True))
    set_continue_blocked(button, ["The selected run is incomplete."])
    button.click()

    assert observed["default"] == QMessageBox.StandardButton.No
    assert called == []


@pytest.mark.gui
def test_phase_tabs_remain_keyboard_navigable(qt_app):
    """Keyboard tab changes use the same visibility invariant as mouse paths."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from pyCamSet.gui.main_window import PyCamSetApp

    window = PyCamSetApp()
    try:
        window.show()
        bar = window._notebook.tabBar()
        window.phase0_tab.setFocus()
        QApplication.processEvents()
        QTest.keyClick(bar, Qt.Key.Key_Right)
        QApplication.processEvents()
        current = window._notebook.currentIndex()
        assert window._notebook.tabBar().isTabVisible(current)
        assert current != 0
    finally:
        window.close()


@pytest.mark.gui
def test_target_dialog_stays_within_available_screen_and_names_actions(qt_app):
    """Secondary target actions remain reachable on constrained displays."""
    from PySide6.QtWidgets import QCheckBox

    from pyCamSet.gui.create_target import CreateTargetDialog

    dialog = CreateTargetDialog(QCheckBox())
    try:
        available = dialog.screen().availableGeometry().size()
        assert dialog.width() <= available.width()
        assert dialog.height() <= available.height()
        assert dialog._format_combo.accessibleName() == "Target export format"
        assert dialog._name_edit.accessibleName() == "Target output file name"
        assert dialog._status.accessibleDescription()
        assert dialog.minimumWidth() <= available.width()
        assert dialog.minimumHeight() <= available.height()
    finally:
        dialog.close()
