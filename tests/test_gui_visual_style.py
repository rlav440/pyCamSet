'''Purpose: Test visual style validation, GUI preview and scientific-data invariants.
Status: Active; bounded tests for the per-visual presentation-style contract.
Future: Add platform-native rendering checks when cross-OS GUI runners are available.
'''
from __future__ import annotations

import json

import pytest

pytest.importorskip("PySide6")

from matplotlib.figure import Figure

from pyCamSet.gui.visual_style import (
    VisualStyle,
    apply_visual_style,
    scale_bar_unavailable,
    style_from_json,
    style_path_for_visual,
    style_to_json,
)


def test_style_json_round_trip_is_versioned_and_visual_specific():
    style = VisualStyle(font_family="DejaVu Sans", line_width=2.5,
                        series_colours={"line:0:error": "#123456"})
    text = style_to_json(style, "phase2:view-errors")
    assert style_from_json(text, "phase2:view-errors") == style
    with pytest.raises(ValueError, match="different visual"):
        style_from_json(text, "phase3:residuals")


def test_style_json_rejects_malformed_unknown_and_invalid_values():
    for content in ("{", json.dumps({"schema": "x"})):
        with pytest.raises(ValueError):
            style_from_json(content)
    document = json.loads(style_to_json(VisualStyle(), "view"))
    document["style"]["surprise"] = True
    with pytest.raises(ValueError, match="unknown keys"):
        style_from_json(json.dumps(document))
    with pytest.raises(ValueError, match="line_width"):
        style_from_json(style_to_json(VisualStyle(line_width=0.1), "view"))


def test_style_paths_are_separate_and_visual_ids_do_not_collide(tmp_path):
    first = style_path_for_visual(tmp_path, "phase1:summary")
    second = style_path_for_visual(tmp_path, "phase2:summary")
    assert first.parent == second.parent == tmp_path / "visual-styles"
    assert first != second
    assert first.suffix == ".json"


def test_apply_style_preserves_data_limits_colormap_and_axes_state():
    figure = Figure()
    axes = figure.add_subplot(111)
    line, = axes.plot([0, 1], [2, 4], label="error")
    axes.set_xlim(-1, 2)
    axes.set_ylim(0, 5)
    axes.grid(True)
    image = axes.imshow([[0.0, 1.0]], cmap="viridis", extent=(0, 2, 0, 1))
    original = (line.get_xdata().copy(), line.get_ydata().copy(),
                axes.get_xlim(), axes.get_ylim(), image.get_cmap().name,
                image.get_array().copy(), line.get_color())
    style = VisualStyle(line_width=3.0, marker_size=8.0, grid_visible=False,
                        axes_background="#fafafa", text_colour="#101010")
    apply_visual_style(figure, style, "Sepia")
    assert line.get_xdata().tolist() == original[0].tolist()
    assert line.get_ydata().tolist() == original[1].tolist()
    assert axes.get_xlim() == original[2] and axes.get_ylim() == original[3]
    assert image.get_cmap().name == original[4]
    assert image.get_array().tolist() == original[5].tolist()
    assert line.get_color() == original[6]
    assert line.get_linewidth() == 3.0
    assert axes.get_facecolor() == (0.9803921568627451, 0.9803921568627451,
                                    0.9803921568627451, 1.0)
    assert not any(grid.get_visible() for grid in axes.xaxis.get_gridlines() +
                   axes.yaxis.get_gridlines())


def test_theme_defaults_and_explicit_overrides_coexist_on_theme_switch():
    from pyCamSet.gui.theme import apply_matplotlib_theme, refresh_matplotlib_theme

    figure = Figure()
    axes = figure.add_subplot(111)
    line, = axes.plot([0, 1], [2, 3], label="residual")
    line.set_gid("phase3:residual:camera-01")
    explicit = VisualStyle(axes_background="#fafafa", line_width=2.0,
                           series_colours={"phase3:residual:camera-01": "#123456"})
    apply_visual_style(figure, explicit, "Light")
    apply_matplotlib_theme(figure, "Dark")
    refresh_matplotlib_theme("Dark")
    assert figure.get_facecolor() != (1.0, 1.0, 1.0, 1.0)
    assert axes.get_facecolor() == (250 / 255, 250 / 255, 250 / 255, 1.0)
    assert line.get_color() == "#123456"
    assert line.get_linewidth() == 2.0


def test_scale_bar_fails_closed_without_calibration_transform_and_units():
    reason = scale_bar_unavailable()
    assert reason and "pixel-to-world" in reason
    assert scale_bar_unavailable([[0.1, 0], [0, 0.1]], "mm") is None


def test_style_dialog_preview_and_cancel_restore_the_figure():
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from pyCamSet.gui.visual_style import VisualStyleDialog

    app = QApplication.instance() or QApplication([])
    figure = Figure()
    axes = figure.add_subplot(111)
    line, = axes.plot([0, 1], [1, 3], label="series")
    original = (axes.get_facecolor(), line.get_ydata().copy())
    dialog = VisualStyleDialog(figure, "test:visual", VisualStyle(), "Light")
    dialog.axes_background.setText("#fafafa")
    dialog._preview()
    assert axes.get_facecolor() != original[0]
    QTimer.singleShot(0, dialog.reject)
    assert dialog.exec() != 1
    assert axes.get_facecolor() == original[0]
    assert line.get_ydata().tolist() == original[1].tolist()
