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
    _validate_user_style_filename,
)


def test_style_json_round_trip_is_versioned_and_visual_specific():
    style = VisualStyle(font_family="DejaVu Sans", line_width=2.5,
                        font_weight="bold", title_colour="#111111",
                        series_colours={"line:0:error": "#123456"},
                        series_styles={"line:0:error": {"line_width": 3, "line_style": "--",
                                                          "marker": "o"}})
    text = style_to_json(style, "phase2:view-errors")
    assert json.loads(text)["version"] == 2
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


def test_v1_style_document_migrates_without_losing_saved_values():
    style = VisualStyle(font_family="DejaVu Sans", line_width=2.5,
                        series_colours={"stable:residual": "#123456"})
    old_document = json.loads(style_to_json(style, "view"))
    old_document["version"] = 1
    for key in ("font_weight", "title_colour", "tick_colour", "axes_colour", "legend_colour",
                "series_styles", "colormap", "suggested_preset"):
        old_document["style"].pop(key)
    migrated = style_from_json(json.dumps(old_document), "view")
    assert migrated.font_family == style.font_family
    assert migrated.line_width == style.line_width
    assert migrated.series_colours == style.series_colours
    assert migrated.font_weight is None and migrated.series_styles == {}


def test_style_paths_are_separate_and_visual_ids_do_not_collide(tmp_path):
    first = style_path_for_visual(tmp_path, "phase1:summary")
    second = style_path_for_visual(tmp_path, "phase2:summary")
    assert first.parent == second.parent == tmp_path / "visual-styles"
    assert first != second
    assert first.suffix == ".json"


@pytest.mark.parametrize("filename", [
    "CON", "prn.txt", "Aux.backup.json", "NUL.txt", "COM1.log", "LPT9.data",
    "COM¹.txt", "LPT².json", "folder\\NUL.txt",
])
def test_user_style_filename_rejects_windows_reserved_names(filename):
    with pytest.raises(ValueError, match="reserved filename"):
        _validate_user_style_filename(filename)


def test_user_style_filename_rejects_nul_and_accepts_ordinary_names():
    with pytest.raises(ValueError, match="NUL character"):
        _validate_user_style_filename("style\0.json")
    _validate_user_style_filename("ordinary-style.json")


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
                        axes_background="#fafafa", text_colour="#101010",
                        title_colour="#123456", tick_colour="#654321",
                        axes_colour="#abcdef")
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
    assert axes.title.get_color() == "#123456"
    assert axes.xaxis.label.get_color() == "#abcdef"
    assert axes.get_xticklabels()[0].get_color() == "#654321"
    assert not any(grid.get_visible() for grid in axes.xaxis.get_gridlines() +
                   axes.yaxis.get_gridlines())


def test_series_styles_apply_by_stable_gid_and_preserve_numeric_arrays():
    figure = Figure()
    axes = figure.add_subplot(111)
    line, = axes.plot([0, 1], [2, 4], label="residual")
    line.set_gid("phase3:residual:cam-01")
    x, y = line.get_xdata().copy(), line.get_ydata().copy()
    style = VisualStyle(series_styles={"phase3:residual:cam-01": {
        "line_width": 3.5, "line_style": "--", "marker": "s", "colour": "#123456"}})
    apply_visual_style(figure, style)
    assert (line.get_linewidth(), line.get_linestyle(), line.get_marker(), line.get_color()) == (
        3.5, "--", "s", "#123456")
    assert line.get_xdata().tolist() == x.tolist()
    assert line.get_ydata().tolist() == y.tolist()


def test_colormap_override_requires_explicit_opt_in_and_preserves_norm_and_values():
    figure = Figure()
    axes = figure.add_subplot(111)
    image = axes.imshow([[0.0, 0.5, 1.0]], cmap="viridis", vmin=0, vmax=1)
    image.set_gid("semantic:reprojection-error")
    original = (image.get_array().copy(), image.norm.vmin, image.norm.vmax, image.get_cmap().name)
    with pytest.raises(ValueError, match="explicitly supports"):
        apply_visual_style(figure, VisualStyle(colormap="plasma"))
    assert image.get_cmap().name == original[3]
    image.set_gid("style:colormap:generic-scalar")
    apply_visual_style(figure, VisualStyle(colormap="plasma"))
    assert image.get_cmap().name == "plasma"
    assert image.norm.vmin == original[1] and image.norm.vmax == original[2]
    assert image.get_array().tolist() == original[0].tolist()


def test_unsupported_colour_map_and_series_controls_fail_closed():
    with pytest.raises(ValueError, match="unsupported colormap"):
        VisualStyle(colormap="rainbow").validate()
    with pytest.raises(ValueError, match="unsupported per-series"):
        VisualStyle(series_styles={"id": {"visible": False}}).validate()


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
    axes.set_title("Title", fontweight="bold")
    axes.set_xlabel("X axis", fontweight="normal")
    axes.set_ylabel("Y axis", fontweight="bold")
    line, = axes.plot([0, 1], [1, 3], label="series")
    axes.set_xticks([0, 1], labels=["left", "right"])
    axes.set_yticks([1, 3], labels=["low", "high"])
    for tick in axes.get_xticklabels() + axes.get_yticklabels():
        tick.set_fontweight("bold")
    axes.grid(True)
    legend = axes.legend()
    for text in legend.get_texts():
        text.set_fontweight("bold")
    text_artists = figure.findobj(
        match=lambda artist: hasattr(artist, "set_fontsize") and hasattr(artist, "get_color"))
    original_weights = [artist.get_fontweight() for artist in text_artists]
    overlay = axes.scatter([0.5], [2], s=9, c="#123456")
    overlay.set_gid("detection-overlay:observed")
    original = (figure.get_facecolor(), axes.get_facecolor(), line.get_ydata().copy(),
                line.get_linewidth(), line.get_markersize(), line.get_color(),
                [grid.get_visible() for grid in axes.xaxis.get_gridlines()],
                legend.get_visible(), overlay.get_sizes().copy(),
                overlay.get_facecolors().copy(), overlay.get_edgecolors().copy())
    dialog = VisualStyleDialog(figure, "test:visual", VisualStyle(), "Light")
    dialog.axes_background.setText("#fafafa")
    dialog.line_width.setValue(4)
    dialog.marker_size.setValue(12)
    dialog.text_colour.setText("#abcdef")
    dialog.grid.setChecked(True)
    dialog.grid_value.setChecked(False)
    dialog.legend.setChecked(True)
    dialog.legend_value.setChecked(False)
    dialog.overlay_size.setValue(12)
    dialog.overlay_colour.setText("#abcdef")
    dialog.font_weight.setCurrentIndex(2)
    assert axes.get_facecolor() != original[1]
    assert all(artist.get_fontweight() == "bold" for artist in text_artists)
    QTimer.singleShot(0, dialog.reject)
    assert dialog.exec() != 1
    assert figure.get_facecolor() == original[0]
    assert axes.get_facecolor() == original[1]
    assert line.get_ydata().tolist() == original[2].tolist()
    assert line.get_linewidth() == original[3]
    assert line.get_markersize() == original[4]
    assert line.get_color() == original[5]
    assert [grid.get_visible() for grid in axes.xaxis.get_gridlines()] == original[6]
    assert legend.get_visible() == original[7]
    assert overlay.get_sizes().tolist() == original[8].tolist()
    assert overlay.get_facecolors().tolist() == original[9].tolist()
    assert overlay.get_edgecolors().tolist() == original[10].tolist()
    assert [artist.get_fontweight() for artist in text_artists] == original_weights


def test_style_dialog_accept_keeps_font_weight_preview():
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from pyCamSet.gui.visual_style import VisualStyleDialog

    app = QApplication.instance() or QApplication([])
    figure = Figure()
    axes = figure.add_subplot(111)
    axes.set_title("Title")
    axes.set_xlabel("X axis")
    axes.set_ylabel("Y axis")
    axes.plot([0, 1], [1, 3], label="series")
    axes.legend()
    dialog = VisualStyleDialog(figure, "test:visual", VisualStyle(), "Light")
    dialog.font_weight.setCurrentIndex(2)
    text_artists = figure.findobj(
        match=lambda artist: hasattr(artist, "set_fontsize") and hasattr(artist, "get_color"))
    assert text_artists
    assert all(artist.get_fontweight() == "bold" for artist in text_artists)
    QTimer.singleShot(0, dialog.accept)
    assert dialog.exec() == 1
    assert dialog.current.font_weight == "bold"
    assert all(artist.get_fontweight() == "bold" for artist in text_artists)


def test_style_dialog_reset_clears_overlay_and_series_overrides():
    from PySide6.QtWidgets import QApplication
    from pyCamSet.gui.visual_style import VisualStyleDialog

    app = QApplication.instance() or QApplication([])
    figure = Figure()
    axes = figure.add_subplot(111)
    line, = axes.plot([0, 1], [1, 3], label="series")
    line.set_gid("stable:series")
    style = VisualStyle(overlay_size=9, overlay_colour="#123456",
                        overlay_edge_colour="#654321",
                        series_colours={"stable:series": "#abcdef"})
    dialog = VisualStyleDialog(figure, "test:visual", style, "Light")
    dialog._reset()
    reset = dialog._read()
    assert reset.overlay_size is None
    assert reset.overlay_colour is None
    assert reset.overlay_edge_colour is None
    assert reset.series_colours == {}


def test_style_dialog_json_load_populates_every_serialised_control(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QFileDialog
    from pyCamSet.gui.visual_style import VisualStyleDialog

    app = QApplication.instance() or QApplication([])
    figure = Figure()
    axes = figure.add_subplot(111)
    line, = axes.plot([0, 1], [1, 3], label="series")
    line.set_gid("stable:series")
    candidate = VisualStyle(font_family="DejaVu Sans", font_size=14, text_colour="#111111",
                            figure_background="#222222", axes_background="#333333",
                            line_width=2.5, marker_size=8, grid_visible=True,
                            legend_visible=False, overlay_size=11, overlay_colour="#444444",
                            overlay_edge_colour="#555555",
                            series_colours={"stable:series": "#666666"})
    path = tmp_path / "style.json"
    path.write_text(style_to_json(candidate, "test:visual"), encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(path), "JSON"))
    dialog = VisualStyleDialog(figure, "test:visual", VisualStyle(), "Light")
    dialog._load()
    assert dialog._read() == candidate
