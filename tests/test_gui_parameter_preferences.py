"""Purpose: Verify per-user GUI input persistence and per-tab default reset.
Status: Offscreen regression tests for the local parameter settings contract.
Future: Extend with real native click-through on each supported desktop.
"""
from __future__ import annotations

import json

import pytest


@pytest.mark.gui
def test_parameter_round_trip_and_phase_reset(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QPushButton
    from pyCamSet.gui.main_window import PyCamSetApp

    monkeypatch.setenv("PYCAMSET_CONFIG_DIR", str(tmp_path))
    app = QApplication.instance() or QApplication([])
    first = PyCamSetApp()
    assert first.phase3_tab._max_nfev_spin.value() == 1000
    assert not first.phase1_tab.findChildren(QPushButton) == []
    first.phase3_tab._max_nfev_spin.setValue(100)
    first.phase1_tab._preprocessing_gamma_edit.setText("0.7")
    first.close()
    path = tmp_path / "parameters.json"
    assert path.is_file()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["pages"]["phase3"]["_max_nfev_spin"] == 100

    second = PyCamSetApp()
    try:
        assert second.phase3_tab._max_nfev_spin.value() == 100
        assert second.phase1_tab._preprocessing_gamma_edit.text() == "0.7"
        assert all(second._parameter_preferences.pages[name].findChildren(QPushButton)
                   for name in second._parameter_preferences.pages)
        second._parameter_preferences.reset("phase3")
        assert second.phase3_tab._max_nfev_spin.value() == 1000
        assert second.phase1_tab._preprocessing_gamma_edit.text() == "0.7"
    finally:
        second.close()
    third = PyCamSetApp()
    try:
        assert third.phase3_tab._max_nfev_spin.value() == 1000
        assert third.phase1_tab._preprocessing_gamma_edit.text() == "0.7"
    finally:
        third.close()


@pytest.mark.gui
def test_other_parameter_tabs_and_target_round_trip(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from pyCamSet.gui.main_window import PyCamSetApp

    monkeypatch.setenv("PYCAMSET_CONFIG_DIR", str(tmp_path))
    QApplication.instance() or QApplication([])
    first = PyCamSetApp()
    first.phase1_tab._target_form.set_target_type("ChArUco")
    first.phase1_tab._nlim_edit.setText("32")
    first.phase2_tab._min_dtct_spin.setValue(7)
    first.phase4_tab._max_nfev_spin.setValue(123)
    first.export_calibration_tab._depth_num_edit.setText("42")
    first.optimisation_tab._trials_spin.setValue(17)
    first.detection_cost_tab._max_frames_spin.setValue(11)
    first.close()
    second = PyCamSetApp()
    try:
        assert second.phase1_tab._target_form.target_type() == "ChArUco"
        assert second.phase1_tab._nlim_edit.text() == "32"
        assert second.phase2_tab._min_dtct_spin.value() == 7
        assert second.phase4_tab._max_nfev_spin.value() == 123
        assert second.export_calibration_tab._depth_num_edit.text() == "42"
        assert second.optimisation_tab._trials_spin.value() == 17
        assert second.detection_cost_tab._max_frames_spin.value() == 11
        for page in ("phase1", "phase2", "phase4", "export", "optimisation", "detection_cost"):
            second._parameter_preferences.reset(page)
        assert second.phase1_tab._nlim_edit.text() == ""
        assert second.phase4_tab._max_nfev_spin.value() == 1000
        assert second.optimisation_tab._trials_spin.value() != 17
    finally:
        second.close()


@pytest.mark.gui
def test_invalid_saved_parameters_preserved_without_applying(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from pyCamSet.gui.main_window import PyCamSetApp

    monkeypatch.setenv("PYCAMSET_CONFIG_DIR", str(tmp_path))
    source = tmp_path / "parameters.json"
    invalid = '{"schema": "pycamset.gui-parameters", "version": 999, "pages": {}}'
    source.write_text(invalid, encoding="utf-8")
    QApplication.instance() or QApplication([])
    window = PyCamSetApp()
    assert window._parameter_preferences.load_error is not None
    assert window.phase3_tab._max_nfev_spin.value() == 1000
    window.close()
    assert (tmp_path / "parameters.json.corrupt").read_text(encoding="utf-8") == invalid
    assert json.loads(source.read_text(encoding="utf-8"))["version"] == 1


@pytest.mark.gui
def test_detector_option_restores_and_reset_uses_backend_default(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QSpinBox
    from pyCamSet.gui.main_window import PyCamSetApp

    monkeypatch.setenv("PYCAMSET_CONFIG_DIR", str(tmp_path))
    QApplication.instance() or QApplication([])
    first = PyCamSetApp()
    candidates = [(key, widget) for key, widget in first.phase1_tab._detection_option_widgets.items()
                  if isinstance(widget, QSpinBox) and widget.value() < widget.maximum()]
    assert candidates
    key, widget = candidates[0]
    original = widget.value()
    widget.setValue(original + 1)
    first.close()
    second = PyCamSetApp()
    try:
        assert second._parameter_preferences.load_error is None, second._parameter_preferences.load_error
        assert second.phase1_tab._detection_option_widgets[key].value() == original + 1
        second._parameter_preferences.reset("phase1")
        assert second.phase1_tab._detection_option_widgets[key].value() == original
    finally:
        second.close()


@pytest.mark.gui
def test_style_dialog_fits_screen_and_noop_cancel_keeps_artists(tmp_path, monkeypatch):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialogButtonBox, QScrollArea
    from matplotlib.figure import Figure
    from pyCamSet.gui.visual_style import VisualStyle, VisualStyleDialog

    QApplication.instance() or QApplication([])
    fig = Figure()
    axes = fig.add_subplot(111)
    image = axes.imshow([[1, 2], [3, 4]])
    axes.set_position((0.2, 0.25, 0.55, 0.6))
    before = tuple(axes.get_position().bounds)
    dialog = VisualStyleDialog(fig, "test:diagnostic", VisualStyle(), "Light")
    assert dialog.findChild(QScrollArea) is not None
    buttons = dialog.findChild(QDialogButtonBox)
    assert buttons is not None and buttons.parent() is dialog
    assert dialog.maximumHeight() <= dialog.screen().availableGeometry().height()
    preview_calls = []
    dialog.on_preview = lambda *_: preview_calls.append(True)
    QTimer.singleShot(0, dialog.reject)
    assert not dialog.exec()
    assert not preview_calls
    assert tuple(axes.get_position().bounds) == before
    assert image.get_array().tolist() == [[1, 2], [3, 4]]
    accepted = VisualStyleDialog(fig, "test:diagnostic", VisualStyle(), "Light")
    accepted.on_preview = lambda *_: preview_calls.append(True)
    QTimer.singleShot(0, accepted.accept)
    assert accepted.exec()
    assert not preview_calls
    assert tuple(axes.get_position().bounds) == before
