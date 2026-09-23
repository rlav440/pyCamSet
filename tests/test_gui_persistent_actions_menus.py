"""Offscreen contracts for fixed phase actions and global menus."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QMenu,
    QPushButton,
    QScrollArea,
    QWidgetAction,
)


@pytest.fixture(scope="module")
def application():
    """Create the Qt application needed by the offscreen widget tests."""
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("size", [(1140, 820), (860, 640)])
def test_phase_actions_remain_in_viewport_at_supported_sizes(application, size):
    """Keep run/navigation actions outside scrollable parameter forms."""
    from pyCamSet.gui.main_window import PyCamSetApp

    window = PyCamSetApp()
    window.resize(*size)
    window.show()
    application.processEvents()
    try:
        phases = (
            (window.phase0_tab, ("Confirm Image Folder Validity", "Continue to Next Phase ▶")),
            (window.phase1_tab, ("▶  Run Phase 1", "Diagnostics ▼", "Continue to Next Phase ▶")),
            (window.phase2_tab, ("▶  Run Phase 2", "Diagnostics ▼", "Continue to Next Phase ▶")),
            (window.phase3_tab, ("▶  Run Phase 3", "Diagnostics ▼", "Phase 4 - Self-Calibration", "Assess Calibration")),
            (window.phase4_tab, ("▶  Run Phase 4", "Diagnostics ▼", "Assess Calibration")),
        )
        for tab, expected_labels in phases:
            window._notebook.setCurrentWidget(tab)
            application.processEvents()
            tab_rect = tab.rect()
            for label in expected_labels:
                button = next(
                    (item for item in tab.findChildren(QPushButton) if item.text() == label),
                    None,
                )
                assert button is not None, f"missing {label!r} at {size}"
                top_left = button.mapTo(tab, button.rect().topLeft())
                bottom_right = button.mapTo(tab, button.rect().bottomRight())
                assert tab_rect.contains(top_left) and tab_rect.contains(bottom_right), (
                    f"{label!r} is clipped at {size}: {button.geometry()}"
                )
                assert button.isVisible(), f"{label!r} is hidden at {size}"
                parent = button.parentWidget()
                while parent is not None and parent is not tab:
                    assert not isinstance(parent, QScrollArea), (
                        f"{label!r} remains coupled to parameter scrolling"
                    )
                    parent = parent.parentWidget()
                if label == "Diagnostics ▼":
                    assert button.property("designRole") == "warning"
    finally:
        window.close()
        window.deleteLater()
        application.processEvents()


def test_global_menu_controls_reuse_live_widget_state(application):
    """Menu-hosted controls must be the same objects and signals as before."""
    from pyCamSet.gui.main_window import PyCamSetApp

    window = PyCamSetApp()
    try:
        menus = {menu.title(): menu for menu in window.menuBar().findChildren(QMenu)}
        assert {"File", "Edit", "Settings"}.issubset(menus)
        edit_action = next(action for action in menus["Edit"].actions() if isinstance(action, QWidgetAction))
        settings_widgets = [
            action for action in menus["Settings"].actions()
            if isinstance(action, QWidgetAction)
        ]
        assert edit_action.defaultWidget() is window._info_cb
        assert settings_widgets[0].defaultWidget() is window._terminal_cb
        assert settings_widgets[1].defaultWidget() is window._theme_combo
        assert isinstance(window._info_cb, QCheckBox)
        assert isinstance(window._theme_combo, QComboBox)
        window._info_cb.setChecked(False)
        assert QApplication.instance().property("tooltipsEnabled") is False
        window._theme_combo.setCurrentText("Dark")
        assert QApplication.instance().property("pycamsetTheme") == "Dark"
    finally:
        window.close()
        window.deleteLater()
        application.processEvents()


def test_diagnostics_settings_arrows_use_semantic_warning_role(application):
    """Keep each diagnostics-to-settings affordance theme-driven."""
    from pyCamSet.gui.main_window import PyCamSetApp

    window = PyCamSetApp()
    try:
        diagnostics_tabs = (
            window.phase1_diag_tab,
            window.phase2_diag_tab,
            window.phase3_diag_tab,
            window.phase4_diag_tab,
        )
        for tab in diagnostics_tabs:
            arrow = next(
                button for button in tab.findChildren(QPushButton)
                if button.text().startswith("▲ ")
            )
            assert arrow.property("designRole") == "warning"
        assert window.phase4_tab._cancel_btn.property("designRole") == "secondary"
        assert window.optimisation_tab._cancel_btn.property("designRole") == "secondary"
    finally:
        window.close()
        window.deleteLater()
        application.processEvents()
