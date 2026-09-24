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
    SUGGESTED_PRESET_CITATIONS,
    apply_visual_style,
    scale_bar_unavailable,
    SUGGESTED_PRESET_REGISTRY_VERSION,
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
                                                          "marker": "o"}},
                        suggested_preset="Science",
                        suggested_preset_registry=SUGGESTED_PRESET_REGISTRY_VERSION)
    text = style_to_json(style, "phase2:view-errors")
    assert json.loads(text)["version"] == 4
    assert style_from_json(text, "phase2:view-errors") == style
    assert json.loads(text)["style"]["suggested_preset_registry"] == SUGGESTED_PRESET_REGISTRY_VERSION
    with pytest.raises(ValueError, match="different visual"):
        style_from_json(text, "phase3:residuals")


def test_science_suggestion_cites_reviewed_2025_guide_not_old_pdf():
    assert SUGGESTED_PRESET_REGISTRY_VERSION == "figure-suggestions-v2"
    assert "Science/AAAS: Guide to Preparing Figures (2025)" in SUGGESTED_PRESET_CITATIONS
    assert "67f37ac8-4d02-4625-8a05-230568cb8323/author_prep_guide_2025.pdf" in SUGGESTED_PRESET_CITATIONS
    assert "archived PDF inspected" in SUGGESTED_PRESET_CITATIONS
    assert "not journal compliance" in SUGGESTED_PRESET_CITATIONS
    assert "author_figure_prep_guide_2022" not in SUGGESTED_PRESET_CITATIONS
    assert "sciadv_guide_to_preparing_figures_2026" not in SUGGESTED_PRESET_CITATIONS


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
                "series_styles", "colormap", "suggested_preset", "suggested_preset_registry",
                "overlay_marker", "overlay_line_width", "overlay_line_style", "overlay_opacity"):
        old_document["style"].pop(key)
    migrated = style_from_json(json.dumps(old_document), "view")
    assert migrated.font_family == style.font_family
    assert migrated.line_width == style.line_width
    assert migrated.series_colours == style.series_colours
    assert migrated.font_weight is None and migrated.series_styles == {}


def test_v2_preset_name_survives_with_unknown_provenance_and_malformed_revision_fails():
    document = json.loads(style_to_json(VisualStyle(
        suggested_preset="Science",
        suggested_preset_registry=SUGGESTED_PRESET_REGISTRY_VERSION), "view"))
    document["version"] = 2
    for key in ("suggested_preset_registry", "overlay_marker", "overlay_line_width",
                "overlay_line_style", "overlay_opacity"):
        document["style"].pop(key)
    migrated = style_from_json(json.dumps(document), "view")
    assert migrated.suggested_preset == "Science"
    assert migrated.suggested_preset_registry is None

    document["version"] = 4
    for key in ("overlay_marker", "overlay_line_width", "overlay_line_style", "overlay_opacity"):
        document["style"][key] = None
    document["style"]["suggested_preset_registry"] = ""
    with pytest.raises(ValueError, match="registry revision"):
        style_from_json(json.dumps(document), "view")


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


def test_v3_style_migrates_overlay_defaults_without_changing_existing_preset_provenance():
    style = VisualStyle(suggested_preset="Science",
                        suggested_preset_registry=SUGGESTED_PRESET_REGISTRY_VERSION)
    document = json.loads(style_to_json(style, "view"))
    document["version"] = 3
    for key in ("overlay_marker", "overlay_line_width", "overlay_line_style", "overlay_opacity"):
        document["style"].pop(key)
    migrated = style_from_json(json.dumps(document), "view")
    assert migrated.suggested_preset_registry == SUGGESTED_PRESET_REGISTRY_VERSION
    assert migrated.overlay_marker is None and migrated.overlay_opacity is None


def test_detection_overlay_customisation_preserves_detection_and_source_image_and_exports(tmp_path):
    import numpy as np

    figure = Figure()
    axes = figure.add_subplot(111)
    source_pixels = np.arange(100, dtype=np.uint8).reshape(10, 10)
    image = axes.imshow(source_pixels, cmap="gray")
    marker_xy = np.array([[2.0, 3.0], [7.0, 8.0]])
    # Representative producer adapter: raw get_data points feed the displayed scatter offsets.
    class DetectionAdapter:
        def get_data(self):
            return np.column_stack((marker_xy, [11, 12]))

    observed = DetectionAdapter().get_data()
    overlay = axes.scatter(observed[:, 0], observed[:, 1], s=100, c="lime", linewidths=0.4)
    overlay.set_gid("detection-overlay:phase1:camera-0")
    before = (observed.copy(), overlay.get_offsets().copy(), image.get_array().copy(),
              image.get_cmap().name, axes.get_xlim(), axes.get_ylim())
    style = VisualStyle(overlay_marker="s", overlay_size=12, overlay_colour="#123456",
                        overlay_edge_colour="#fedcba", overlay_line_width=2,
                        overlay_line_style="--", overlay_opacity=0.35)
    apply_visual_style(figure, style, "Dark")
    assert len(overlay.get_paths()[0].vertices) == 5
    assert overlay.get_sizes().tolist() == [144]
    assert overlay.get_alpha() == 0.35
    assert overlay.get_linestyles()[0][1] != [1.0]
    assert np.array_equal(DetectionAdapter().get_data(), before[0])
    assert np.array_equal(overlay.get_offsets(), before[1])
    assert np.array_equal(image.get_array(), before[2])
    assert image.get_cmap().name == before[3] == "gray"
    assert axes.get_xlim() == before[4] and axes.get_ylim() == before[5]
    figure.savefig(tmp_path / "styled-montage.png", format="png")
    assert (tmp_path / "styled-montage.png").stat().st_size > 0


@pytest.mark.parametrize("style", [
    VisualStyle(overlay_marker="invalid"),
    VisualStyle(overlay_opacity=1.1),
    VisualStyle(overlay_line_style="dotted"),
    VisualStyle(overlay_line_width=0.1),
])
def test_detection_overlay_invalid_controls_fail_closed(style):
    with pytest.raises(ValueError):
        style.validate()


def test_detection_overlay_clear_restores_captured_producer_defaults():
    figure = Figure()
    axes = figure.add_subplot(111)
    # Phase 1 producer uses s=10, a circle, and 0.4 edge width.
    overlay = axes.scatter([1, 2], [3, 4], s=10, marker="o", linewidths=0.4,
                           linestyles="-", alpha=None)
    overlay.set_gid("detection-overlay:phase1:camera-0")
    baseline = (overlay.get_sizes().copy(), [path.vertices.copy() for path in overlay.get_paths()],
                overlay.get_linewidths().copy(), overlay.get_linestyles(), overlay.get_alpha())

    apply_visual_style(figure, VisualStyle(overlay_marker="s", overlay_size=12,
                                            overlay_line_width=2, overlay_line_style="--",
                                            overlay_opacity=0.35), "Dark")
    apply_visual_style(figure, VisualStyle(), "Light")

    assert overlay.get_sizes().tolist() == baseline[0].tolist() == [10]
    assert [path.vertices.tolist() for path in overlay.get_paths()] == [path.tolist() for path in baseline[1]]
    assert overlay.get_linewidths().tolist() == baseline[2].tolist() == pytest.approx([0.4])
    assert overlay.get_linestyles() == baseline[3]
    assert overlay.get_alpha() is baseline[4] is None


def test_detection_overlay_theme_defaults_refresh_but_explicit_colours_win():
    figure = Figure()
    axes = figure.add_subplot(111)
    overlay = axes.scatter([1], [2])
    overlay.set_gid("detection-overlay:phase1:camera-0")
    apply_visual_style(figure, VisualStyle(), "Light")
    light_face = overlay.get_facecolors().copy()
    apply_visual_style(figure, VisualStyle(), "Dark")
    assert overlay.get_facecolors().tolist() != light_face.tolist()
    from matplotlib.colors import to_rgba
    from pyCamSet.gui.theme import THEME_TOKENS
    # The theme default for detection overlays is the accent colour.
    assert overlay.get_facecolors()[0].tolist() == pytest.approx(list(to_rgba(THEME_TOKENS["Dark"]["accent"])))
    explicit = VisualStyle(overlay_colour="#123456", overlay_edge_colour="#654321")
    apply_visual_style(figure, explicit, "Sepia")
    assert overlay.get_facecolors()[0].tolist() == pytest.approx([18 / 255, 52 / 255, 86 / 255, 1.0])
    assert overlay.get_edgecolors()[0].tolist() == pytest.approx([101 / 255, 67 / 255, 33 / 255, 1.0])


def test_detection_overlay_theme_refresh_keeps_overrides_then_reset_restores_baseline():
    from pyCamSet.gui.theme import apply_matplotlib_theme, refresh_matplotlib_theme

    figure = Figure()
    axes = figure.add_subplot(111)
    overlay = axes.scatter([1], [2], s=10, marker="o", linewidths=0.4, alpha=None)
    overlay.set_gid("detection-overlay:phase1:camera-0")
    apply_visual_style(figure, VisualStyle(overlay_marker="s", overlay_size=12,
                                            overlay_line_width=2, overlay_opacity=0.35), "Light")
    apply_matplotlib_theme(figure, "Dark")
    refresh_matplotlib_theme("Dark")
    assert overlay.get_sizes().tolist() == [144]
    assert overlay.get_alpha() == 0.35
    apply_visual_style(figure, VisualStyle(), "Dark")
    assert overlay.get_sizes().tolist() == [10]
    assert len(overlay.get_paths()[0].vertices) == 26
    assert overlay.get_linewidths().tolist() == pytest.approx([0.4])
    assert overlay.get_alpha() is None


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


@pytest.mark.parametrize("transform, unit", [
    (None, "mm"),
    ([[0.1, 0], [0, 0.1]], "mm"),
    (object(), "mm"),
    ([[float("nan"), 0], [0, 1]], "mm"),
    ([[1, 0, 0], [0, 1, 0]], "mm"),
    ([[1, 0], [0, 1]], ""),
])
def test_scale_bar_fails_closed_without_a_supported_calibration_contract(transform, unit):
    reason = scale_bar_unavailable(transform, unit)
    assert reason and "pixel-to-world" in reason


def test_phase1_failed_style_write_restores_saved_style_and_reports_error(tmp_path, monkeypatch):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PySide6.QtWidgets import QApplication, QCheckBox, QTabWidget, QDialog
    from pyCamSet.gui import visual_style
    from pyCamSet.gui.phase_1_detection import Phase1DiagnosticsTab
    from pyCamSet.workflow.workspace import WorkspaceManager

    app = QApplication.instance() or QApplication([])
    tab = Phase1DiagnosticsTab(QTabWidget(), QCheckBox(), WorkspaceManager(None))
    figure = Figure()
    axes = figure.add_subplot(111)
    overlay = axes.scatter([2.0], [3.0], s=100, c="#123456")
    overlay.set_gid("detection-overlay:phase1:camera-0")
    canvas = FigureCanvasAgg(figure)
    committed = VisualStyle(overlay_size=8, overlay_colour="#123456")
    visual_id = "phase1:detection-overlay"
    style_path = visual_style.style_path_for_visual(tmp_path, visual_id)
    style_path.parent.mkdir(parents=True)
    original_sidecar = visual_style.style_to_json(committed, visual_id)
    style_path.write_text(original_sidecar, encoding="utf-8")
    visual_style.apply_visual_style(figure, committed, "Light")
    tab._draw_state = {"fig": figure, "canvas": canvas, "style": committed}
    errors = []
    monkeypatch.setattr("pyCamSet.gui.preferences.config_directory", lambda: tmp_path)
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.warning",
                        lambda *args: errors.append(args[2]))

    make_dialog = visual_style.VisualStyleDialog

    def accepted_preview(*args, **kwargs):
        dialog = make_dialog(*args, **kwargs)

        def accept_with_preview():
            dialog.overlay_size.setValue(16)
            dialog.overlay_colour.setText("#abcdef")
            dialog._preview()
            return QDialog.DialogCode.Accepted

        dialog.exec = accept_with_preview
        return dialog

    monkeypatch.setattr(visual_style, "VisualStyleDialog", accepted_preview)
    real_replace = type(style_path).replace

    def fail_sidecar_replace(path, *args, **kwargs):
        if path.name.startswith(f".{style_path.name}."):
            raise OSError("injected persistence failure")
        return real_replace(path, *args, **kwargs)

    monkeypatch.setattr(type(style_path), "replace", fail_sidecar_replace)
    try:
        tab._edit_detection_style()
        assert overlay.get_sizes().tolist() == [64]
        assert overlay.get_facecolors()[0].tolist() == [18 / 255, 52 / 255, 86 / 255, 1]
        assert tab._draw_state["style"] == committed
        assert visual_style._VISUAL_OVERRIDES[figure] == committed
        assert style_path.read_text(encoding="utf-8") == original_sidecar
        assert list(style_path.parent.iterdir()) == [style_path]
        assert errors and "injected persistence failure" in errors[0]
    finally:
        tab.deleteLater()


def test_phase1_montage_navigation_buttons_have_accessible_names():
    from PySide6.QtWidgets import QApplication, QCheckBox, QTabWidget
    from pyCamSet.gui.phase_1_detection import Phase1DiagnosticsTab
    from pyCamSet.workflow.workspace import WorkspaceManager

    app = QApplication.instance() or QApplication([])
    tab = Phase1DiagnosticsTab(QTabWidget(), QCheckBox(), WorkspaceManager(None))
    try:
        assert tab._draw_prev_btn.accessibleName() == "Previous detection image"
        assert tab._draw_prev_btn.toolTip() == "Show the previous detection image"
        assert tab._draw_next_btn.accessibleName() == "Next detection image"
        assert tab._draw_next_btn.toolTip() == "Show the next detection image"
    finally:
        tab.deleteLater()


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
    original_overlay_paths = [path.vertices.copy() for path in overlay.get_paths()]
    original_overlay_widths = overlay.get_linewidths().copy()
    original_overlay_styles = overlay.get_linestyles()
    original_overlay_alpha = overlay.get_alpha()
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
    dialog.overlay_marker.setCurrentIndex(dialog.overlay_marker.findData("s"))
    dialog.overlay_line_width.setValue(2.0)
    dialog.overlay_line_style.setCurrentIndex(2)
    dialog.overlay_opacity.setValue(0.25)
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
    assert [path.vertices.tolist() for path in overlay.get_paths()] == [path.tolist() for path in original_overlay_paths]
    assert overlay.get_linewidths().tolist() == original_overlay_widths.tolist()
    assert overlay.get_linestyles() == original_overlay_styles
    assert overlay.get_alpha() == original_overlay_alpha
    assert dialog.overlay_marker.accessibleName() == "Detection marker shape"
    assert dialog.overlay_opacity.accessibleName() == "Detection marker opacity"
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
    overlay = axes.scatter([1], [2], s=100, marker="o", linewidths=0.4, alpha=None)
    overlay.set_gid("detection-overlay:phase1:camera-0")
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
    assert reset.overlay_marker is None
    assert reset.overlay_line_width is None
    assert reset.overlay_line_style is None
    assert reset.overlay_opacity is None
    assert reset.series_colours == {}
    assert overlay.get_sizes().tolist() == [100]
    assert len(overlay.get_paths()[0].vertices) == 26  # Matplotlib's producer circle path.
    assert overlay.get_linewidths().tolist() == pytest.approx([0.4])
    assert overlay.get_alpha() is None


def test_style_dialog_size_six_is_an_explicit_override_and_cancel_restores_default():
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from pyCamSet.gui.visual_style import VisualStyleDialog

    QApplication.instance() or QApplication([])
    figure = Figure()
    axes = figure.add_subplot(111)
    overlay = axes.scatter([1], [2], s=100, marker="o", linewidths=0.4, alpha=None)
    overlay.set_gid("detection-overlay:phase1:camera-0")
    dialog = VisualStyleDialog(figure, "phase1:overlay", VisualStyle(), "Light")
    assert dialog.overlay_size.value() == 10
    dialog.overlay_size.setValue(6)
    assert dialog.current.overlay_size == 6
    assert overlay.get_sizes().tolist() == [36]
    dialog._reset()
    assert dialog.current.overlay_size is None
    assert overlay.get_sizes().tolist() == [100]
    dialog.overlay_size.setValue(6)
    QTimer.singleShot(0, dialog.reject)
    assert dialog.exec() != 1
    assert overlay.get_sizes().tolist() == [100]


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
                            overlay_marker="D", overlay_edge_colour="#555555",
                            overlay_line_width=1.2, overlay_line_style=":", overlay_opacity=0.6,
                            series_colours={"stable:series": "#666666"})
    path = tmp_path / "style.json"
    path.write_text(style_to_json(candidate, "test:visual"), encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(path), "JSON"))
    dialog = VisualStyleDialog(figure, "test:visual", VisualStyle(), "Light")
    dialog._load()
    assert dialog._read() == candidate


def test_raw_detection_image_keeps_colormap_and_scale_bar_controls_disabled():
    from PySide6.QtWidgets import QApplication
    from pyCamSet.gui.visual_style import VisualStyleDialog

    QApplication.instance() or QApplication([])
    figure = Figure()
    axes = figure.add_subplot(111)
    axes.imshow([[0, 1], [2, 3]], cmap="gray")
    dialog = VisualStyleDialog(figure, "phase1:detection-overlay", VisualStyle(), "Light")
    assert not dialog.colormap.isEnabled()
    from PySide6.QtWidgets import QScrollArea
    assert dialog.findChild(QScrollArea) is not None
    scale_controls = [widget for widget in dialog.findChildren(type(dialog.grid))
                      if widget.text() == "Enable scale bar"]
    assert len(scale_controls) == 1 and not scale_controls[0].isEnabled()
    assert "pixel-to-world" in scale_controls[0].toolTip()
