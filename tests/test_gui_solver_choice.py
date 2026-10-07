"""The bundle adjustment solver is a visible, changeable choice, Schur by default."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def window(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    monkeypatch.setenv("PYCAMSET_CONFIG_DIR", str(tmp_path / "config"))
    application = QApplication.instance() or QApplication([])
    from pyCamSet.gui import phase_3_bundle_adjustment, phase_4_self_calibration
    from pyCamSet.gui.main_window import PyCamSetApp

    # The image folder is not what is under test.
    for module in (phase_3_bundle_adjustment, phase_4_self_calibration):
        monkeypatch.setattr(module, "require_image_folder", lambda text: text)
    app_window = PyCamSetApp()
    yield app_window
    app_window.close()
    app_window.deleteLater()
    application.processEvents()


@pytest.mark.parametrize("tab_name", ["phase3_tab", "phase4_tab"])
def test_each_solve_offers_the_solver_and_starts_on_schur(window, tab_name):
    tab = getattr(window, tab_name)
    combo = tab._solver_combo
    assert [combo.itemText(i) for i in range(combo.count())] == ["Schur", "Trust region"]
    assert combo.currentText() == "Schur"
    assert tab._read_params()["problem_options"]["solver"] == "schur"
    combo.setCurrentIndex(combo.findText("Trust region"))
    assert tab._read_params()["problem_options"]["solver"] == "trf"


def test_phase4_starts_on_plain_least_squares(window):
    """Linear by default, as the library's own; the robust losses stay a choice."""
    tab = window.phase4_tab
    assert tab._loss_combo.currentText() == "linear"
    assert not tab._f_scale_spin.isEnabled()  # f_scale has no effect on a linear loss
    assert tab._read_params()["problem_options"]["loss"] == "linear"
    tab._loss_combo.setCurrentText("soft_l1")
    assert tab._f_scale_spin.isEnabled()
    assert tab._read_params()["problem_options"]["loss"] == "soft_l1"


def test_the_solver_choice_is_remembered():
    from pyCamSet.gui.parameter_preferences import _FIELDS

    assert "_solver_combo" in _FIELDS["phase3"]
    assert "_solver_combo" in _FIELDS["phase4"]
