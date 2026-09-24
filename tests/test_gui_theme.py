"""Contracts for semantic application themes and contrast safeguards."""
from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QCheckBox, QLabel, QLineEdit
from matplotlib.figure import Figure
from matplotlib.colors import to_rgba

from pyCamSet.gui.theme import (
    THEME_TOKENS,
    apply_theme,
    contrast_ratio,
    validate_theme_tokens,
)


@pytest.fixture(scope="module")
def application():
    """Use one offscreen QApplication, restoring its pre-test presentation."""
    app = QApplication.instance() or QApplication([])
    previous_style = app.styleSheet()
    previous_palette = app.palette()
    previous_theme = app.property("pycamsetTheme")
    yield app
    app.setStyleSheet(previous_style)
    app.setPalette(previous_palette)
    app.setProperty("pycamsetTheme", previous_theme)


def test_all_themes_have_identical_complete_semantic_tokens():
    """Reject theme drift at the contract boundary."""
    validate_theme_tokens(THEME_TOKENS)
    assert set(THEME_TOKENS["Light"]) == set(THEME_TOKENS["Dark"]) == set(THEME_TOKENS["Sepia"])
    assert len(THEME_TOKENS["Light"]) >= 19


def test_text_and_operated_border_contrast_meet_declared_floors():
    """Check actual semantic foreground/background pairs in both themes."""
    for tokens in THEME_TOKENS.values():
        assert contrast_ratio(tokens["text"], tokens["surface"]) >= 4.5
        # 3:1 is a deliberately conservative legibility floor for disabled
        # text, despite disabled controls being exempt from WCAG text contrast.
        for background in ("background", "surface", "surface_alt"):
            assert contrast_ratio(tokens["text_disabled"], tokens[background]) >= 3.0
        assert contrast_ratio(tokens["on_accent"], tokens["accent"]) >= 4.5
        assert contrast_ratio(tokens["on_warning"], tokens["warning"]) >= 4.5
        assert contrast_ratio(tokens["border_strong"], tokens["surface"]) >= 3.0


def test_contrast_gate_fails_known_bad_negative_control():
    """Prove the theme gate detects a deliberately unreadable text token."""
    invalid = deepcopy(THEME_TOKENS)
    invalid["Dark"]["text"] = invalid["Dark"]["surface"]
    with pytest.raises(ValueError, match="text contrast"):
        validate_theme_tokens(invalid)


def test_disabled_text_contrast_gate_fails_known_bad_negative_control():
    """Prove disabled text cannot silently collapse into its control surface."""
    invalid = deepcopy(THEME_TOKENS)
    invalid["Dark"]["text_disabled"] = invalid["Dark"]["surface"]
    with pytest.raises(ValueError, match="text_disabled contrast"):
        validate_theme_tokens(invalid)


def test_disabled_non_button_widgets_receive_semantic_text_colour(application):
    """Exercise disabled controls against the actual application QSS in both themes."""
    controls = (QLineEdit("disabled input"), QCheckBox("disabled check"),
                QLabel("disabled label"))
    for theme_name, tokens in THEME_TOKENS.items():
        apply_theme(application, theme_name)
        for control in controls:
            control.setEnabled(False)
            control.ensurePolished()
            resolved = control.palette().color(control.palette().ColorGroup.Disabled,
                                               control.palette().ColorRole.Text)
            # Check the actual disabled palette role and selector, not merely token presence.
            assert resolved == QColor(tokens["text_disabled"])
        assert "QLineEdit:disabled" in application.styleSheet()
        assert "QCheckBox:disabled" in application.styleSheet()
        assert "QLabel:disabled" in application.styleSheet()
    for control in controls:
        control.deleteLater()


def test_live_theme_switch_updates_palette_and_accessible_state(application):
    """Apply both themes to the live QApplication and verify reversible state."""
    apply_theme(application, "Light")
    light_style = application.styleSheet()
    light_window = application.palette().color(application.palette().ColorRole.Window)
    apply_theme(application, "Dark")
    dark_window = application.palette().color(application.palette().ColorRole.Window)
    assert application.property("pycamsetTheme") == "Dark"
    assert dark_window != light_window
    assert application.styleSheet() != light_style
    assert QColor(THEME_TOKENS["Dark"]["background"]) == dark_window
    apply_theme(application, "Light")
    assert application.property("pycamsetTheme") == "Light"


def test_matplotlib_theme_changes_chrome_without_changing_scientific_data(application, tmp_path):
    """Theme figure chrome across all three themes without changing series data/colour."""
    from pyCamSet.gui.theme import apply_matplotlib_theme, refresh_matplotlib_theme

    figure = Figure()
    axes = figure.add_subplot(111)
    line, = axes.plot([0, 1], [2, 3], color="#d62728", label="measurement")
    axes.grid(True)
    axes.set_title("Diagnostic")
    axes.set_xlabel("View")
    axes.set_ylabel("Error (px)")
    scientific_annotation = axes.text(0.5, 0.5, "scientific label", color="#7a2e8e")
    themed_placeholder = axes.text(0.5, 0.6, "placeholder", color="#000000")
    themed_placeholder.set_gid("phase1:unreadable-placeholder")
    axes.legend()
    original_x = line.get_xdata().copy()
    original_y = line.get_ydata().copy()
    gridlines = axes.xaxis.get_gridlines() + axes.yaxis.get_gridlines()
    assert gridlines and all(gridline.get_visible() for gridline in gridlines)
    for theme_name in ("Light", "Dark", "Sepia"):
        apply_matplotlib_theme(figure, theme_name)
        assert figure.get_facecolor() == to_rgba(THEME_TOKENS[theme_name]["background"])
        assert axes.get_facecolor() == to_rgba(THEME_TOKENS[theme_name]["surface"])
        assert themed_placeholder.get_color() == THEME_TOKENS[theme_name]["text"]
        assert contrast_ratio(themed_placeholder.get_color(), THEME_TOKENS[theme_name]["surface"]) >= 4.5
        assert scientific_annotation.get_color() == "#7a2e8e"
        assert line.get_color() == "#d62728"
        assert line.get_xdata().tolist() == original_x.tolist()
        assert line.get_ydata().tolist() == original_y.tolist()
        assert all(gridline.get_color() == THEME_TOKENS[theme_name]["border"]
                   for gridline in gridlines)

    # A disabled grid stays disabled; theming must not create visible gridlines.
    no_grid_figure = Figure()
    no_grid_axes = no_grid_figure.add_subplot(111)
    no_grid_figure.canvas.draw()
    assert not any(gridline.get_visible() for gridline in
                   no_grid_axes.xaxis.get_gridlines() + no_grid_axes.yaxis.get_gridlines())
    apply_matplotlib_theme(no_grid_figure, "Dark")
    assert not any(gridline.get_visible() for gridline in
                   no_grid_axes.xaxis.get_gridlines() + no_grid_axes.yaxis.get_gridlines())

    # Live refresh updates the registered figure and does not reconstruct data.
    apply_theme(application, "Sepia")
    refresh_matplotlib_theme("Sepia")
    assert axes.title.get_color() == THEME_TOKENS["Sepia"]["text"]
    assert themed_placeholder.get_color() == THEME_TOKENS["Sepia"]["text"]
    assert scientific_annotation.get_color() == "#7a2e8e"
    assert line.get_ydata().tolist() == original_y.tolist()
    export_path = tmp_path / "sepia-figure.png"
    figure.savefig(export_path, dpi=40)
    from PIL import Image
    with Image.open(export_path) as exported:
        assert exported.getpixel((0, 0))[:3] == (243, 236, 223)


def test_token_validator_rejects_missing_or_extra_keys():
    """Prove incomplete and drifting token dictionaries cannot pass silently."""
    missing = deepcopy(THEME_TOKENS)
    del missing["Dark"]["focus"]
    with pytest.raises(ValueError, match="same non-empty token key set"):
        validate_theme_tokens(missing)
    extra = deepcopy(THEME_TOKENS)
    extra["Light"]["accidental"] = "#000000"
    with pytest.raises(ValueError, match="same non-empty token key set"):
        validate_theme_tokens(extra)


def test_main_window_theme_selector_switches_and_persists(tmp_path, monkeypatch):
    """Exercise the actual global selector and QSettings persistence path."""
    import pyCamSet.gui.main_window as main_window_module

    settings_path = tmp_path / "theme.ini"

    def settings_factory(*_args):
        return QSettings(str(settings_path), QSettings.Format.IniFormat)

    monkeypatch.setattr(main_window_module, "QSettings", settings_factory)
    settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
    settings.clear()
    window = main_window_module.PyCamSetApp()
    from pyCamSet.gui.theme import apply_matplotlib_theme
    figure = Figure()
    axes = figure.add_subplot(111)
    axes.plot([0, 1], [1, 2], color="#d62728")
    apply_matplotlib_theme(figure, "Light")
    try:
        assert window._theme_combo.accessibleName() == "Colour theme"
        assert [window._theme_combo.itemText(i) for i in range(window._theme_combo.count())] == [
            "Light", "Dark", "Sepia"]
        window._theme_combo.setCurrentText("Sepia")
        assert QApplication.instance().property("pycamsetTheme") == "Sepia"
        assert settings.value("appearance/theme") == "Sepia"
        assert figure.get_facecolor() == to_rgba(THEME_TOKENS["Sepia"]["background"])
        assert axes.get_lines()[0].get_color() == "#d62728"
        window.close()
        restored_window = main_window_module.PyCamSetApp()
        assert restored_window._theme_combo.currentText() == "Sepia"
        restored_window.close()
    finally:
        window.close()
        settings.clear()
