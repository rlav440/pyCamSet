'''Purpose: Verify palette colour picking, per-element figure styles and themed title bars.
Status: Active; offscreen tests, with the Windows frame call replaced by a recorder.
Future: Add a native Linux/macOS frame check if a CI host exposes one.
'''
from __future__ import annotations

import json

import pytest

pytest.importorskip("PySide6")

from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from pyCamSet.gui.visual_style import (  # noqa: E402
    VisualStyle, apply_visual_style, style_from_json, style_to_json, styleable_elements,
)


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _mixed_figure():
    """A data line, a labelled threshold, bars and scatter points on one axes."""
    figure = Figure()
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    line, = axes.plot([0, 1, 2], [1, 2, 1], color="#123456", label="median")
    threshold = axes.axhline(1.5, color="#d62728", linestyle="--", label="MAD threshold")
    bars = axes.bar([0, 1, 2], [3, 2, 1], color="#1f77b4")
    points = axes.scatter([0, 1], [1, 2], s=16, c="#2ca02c")
    return figure, axes, line, threshold, bars, points


def test_every_element_kind_is_listed_with_a_stable_id():
    figure, *_ = _mixed_figure()
    rows = {row["id"]: row["kind"] for row in styleable_elements(figure)}
    assert rows == {"line:0:median": "line", "reference:0:MAD threshold": "reference",
                    "bars:0:#1": "bars", "points:0:#1": "points"}


def test_threshold_bars_and_points_restyle_and_then_restore(qapp):
    figure, _axes, _line, threshold, bars, points = _mixed_figure()
    style = VisualStyle(series_styles={
        "reference:0:MAD threshold": {"colour": "#0072b2", "line_width": 3.0, "line_style": ":"},
        "bars:0:#1": {"colour": "#009e73", "edge_colour": "#000000", "line_width": 1.5},
        "points:0:#1": {"colour": "#e69f00", "marker": "s", "size": 9.0, "filled": False},
    })
    apply_visual_style(figure, style, "Light")
    assert to_hex(threshold.get_color()) == "#0072b2"
    assert threshold.get_linewidth() == 3.0 and threshold.get_linestyle() == ":"
    assert all(to_hex(patch.get_facecolor()) == "#009e73" for patch in bars.patches)
    assert all(to_hex(patch.get_edgecolor()) == "#000000" for patch in bars.patches)
    assert len(points.get_facecolors()) == 0  # hollow
    assert to_hex(points.get_edgecolors()[0]) == "#e69f00"
    assert points.get_sizes()[0] == pytest.approx(81.0)
    # Clearing every setting returns each element to how it was drawn.
    apply_visual_style(figure, VisualStyle(), "Light")
    assert to_hex(threshold.get_color()) == "#d62728" and threshold.get_linestyle() == "--"
    assert all(to_hex(patch.get_facecolor()) == "#1f77b4" for patch in bars.patches)
    assert to_hex(points.get_facecolors()[0]) == "#2ca02c"
    assert points.get_sizes()[0] == pytest.approx(16.0)


def test_detection_markers_can_be_hollow_or_crosses(qapp):
    figure = Figure()
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    overlay = axes.scatter([1, 2], [1, 2], s=10, c="lime", linewidths=0.4)
    overlay.set_gid("detection-overlay:phase1:cam0")
    apply_visual_style(figure, VisualStyle(overlay_marker="o", overlay_filled=False,
                                           overlay_colour="#e69f00"), "Light")
    assert len(overlay.get_facecolors()) == 0
    assert to_hex(overlay.get_edgecolors()[0]) == "#e69f00"
    assert overlay.get_linewidths()[0] >= 1.2  # an outline must stay visible
    apply_visual_style(figure, VisualStyle(overlay_marker="x", overlay_colour="#d62728"), "Light")
    assert len(overlay.get_facecolors()) == 0  # strokes have no fill
    assert to_hex(overlay.get_edgecolors()[0]) == "#d62728"
    apply_visual_style(figure, VisualStyle(overlay_marker="s", overlay_filled=True,
                                           overlay_colour="#0072b2"), "Light")
    assert to_hex(overlay.get_facecolors()[0]) == "#0072b2"


def test_invalid_element_settings_are_rejected():
    with pytest.raises(ValueError):
        VisualStyle(series_styles={"bars:0:#1": {"marker": "o"}}).validate()
    with pytest.raises(ValueError):
        VisualStyle(series_styles={"points:0:#1": {"filled": "yes"}}).validate()
    with pytest.raises(ValueError):
        VisualStyle(overlay_filled="no").validate()


def test_version_5_documents_load_with_theme_fill():
    document = json.loads(style_to_json(VisualStyle(overlay_size=7.0), "v"))
    document["version"] = 5
    del document["style"]["overlay_filled"]
    style = style_from_json(json.dumps(document), "v")
    assert style.overlay_filled is None and style.overlay_size == 7.0


def test_colour_picker_offers_named_colours_and_exact_rgb(qapp, monkeypatch):
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QColorDialog, QToolButton

    from pyCamSet.gui.colour_picker import PALETTE, ColourPicker, colour_name

    picker = ColourPicker("")
    seen = []
    picker.textChanged.connect(seen.append)
    swatches = picker._menu.findChildren(QToolButton, "colourSwatch")
    assert len(swatches) == len(PALETTE) >= 20
    assert {swatch.accessibleName() for swatch in swatches} >= {"Black", "Vermilion", "Sky blue"}
    swatches[0].click()
    assert picker.text() == "#000000" and colour_name(picker.text()) == "Black"
    monkeypatch.setattr(QColorDialog, "getColor", lambda *args: QColor("#12ab34"))
    picker._custom()
    assert picker.text() == "#12ab34"
    picker.clear()
    assert picker.text() == "" and seen[-1] == ""
    picker.setText("not a colour")
    assert picker.text() == ""


def test_dialog_lists_every_element_with_only_its_controls(qapp):
    from pyCamSet.gui.visual_style import VisualStyleDialog

    figure, *_ = _mixed_figure()
    dialog = VisualStyleDialog(figure, "figure:mixed", VisualStyle(), "Light")
    try:
        kinds = [row["kind"] for row in dialog.elements]
        assert sorted(kinds) == ["bars", "line", "points", "reference"]
        by_kind = dict(zip(kinds, dialog._element_controls))
        assert not by_kind["bars"]["marker"].isEnabled()
        assert not by_kind["reference"]["filled"].isEnabled()
        assert by_kind["points"]["filled"].isEnabled()
        assert not dialog.advanced.isVisible() and not dialog.advanced_toggle.isChecked()
        by_kind["reference"]["colour"].setText("#0072b2")
        by_kind["reference"]["line_style"].setCurrentIndex(by_kind["reference"]["line_style"].findData(":"))
        chosen = dialog._read()
        assert chosen.series_styles["reference:0:MAD threshold"] == {"colour": "#0072b2", "line_style": ":"}
        dialog._reset()
        assert dialog._read().series_styles == {}
    finally:
        dialog.deleteLater()


def test_title_bar_takes_exact_theme_colours_on_windows(qapp, monkeypatch):
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QWidget

    from pyCamSet.gui import window_chrome
    from pyCamSet.gui.theme import THEME_TOKENS

    assert window_chrome._colorref("#102030") == 0x302010
    calls = []
    monkeypatch.setattr(window_chrome.sys, "platform", "win32")
    monkeypatch.setattr(QGuiApplication, "platformName", staticmethod(lambda: "windows"))
    monkeypatch.setattr(window_chrome, "_dwm_set_attribute",
                        lambda hwnd, attribute, value: calls.append((attribute, value)) or True)
    window = QWidget()
    try:
        tokens = THEME_TOKENS["Sepia"]
        assert window_chrome.apply_window_chrome(window, tokens, dark=False)
        attributes = dict(calls)
        assert attributes[20] == 0
        assert attributes[35] == window_chrome._colorref(tokens["surface"])
        assert attributes[36] == window_chrome._colorref(tokens["text"])
    finally:
        window.deleteLater()


def test_title_bar_is_left_alone_off_windows(qapp, monkeypatch):
    from PySide6.QtWidgets import QWidget

    from pyCamSet.gui import window_chrome
    from pyCamSet.gui.theme import THEME_TOKENS

    monkeypatch.setattr(window_chrome.sys, "platform", "linux")
    window = QWidget()
    try:
        assert window_chrome.apply_window_chrome(window, THEME_TOKENS["Dark"], dark=True) is False
    finally:
        window.deleteLater()


def test_theme_switch_requests_the_matching_native_colour_scheme():
    """The offscreen test platform ignores the hint, so record the request instead."""
    from PySide6.QtCore import Qt

    from pyCamSet.gui import window_chrome

    class Hints:
        def __init__(self):
            self.scheme = Qt.ColorScheme.Unknown
            self.requests = []

        def colorScheme(self):  # noqa: N802
            return self.scheme

        def setColorScheme(self, scheme):  # noqa: N802
            self.requests.append(scheme)
            self.scheme = scheme

    hints = Hints()
    application = type("App", (), {"styleHints": lambda self: hints})()
    assert window_chrome.set_colour_scheme(application, dark=True)
    assert window_chrome.set_colour_scheme(application, dark=True)  # no second request
    window_chrome.set_colour_scheme(application, dark=False)
    assert hints.requests == [Qt.ColorScheme.Dark, Qt.ColorScheme.Light]
