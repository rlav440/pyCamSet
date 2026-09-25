'''Purpose: Verify shared figure presets, the global default style and the font policy.
Status: Active; offscreen tests using the isolated per-test config directory.
Future: Add a native-font check when a CI host installs non-bundled families.
'''
from __future__ import annotations

import json

import pytest

pytest.importorskip("PySide6")

from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from pyCamSet.gui import figure_fonts  # noqa: E402
from pyCamSet.gui.visual_style import (  # noqa: E402
    PRESETS, SCHEMA, VisualStyle, apply_visual_style, as_default_style,
    clear_default_style, default_style_path, load_default_style,
    load_style_for_visual, save_default_style, style_from_json,
    style_path_for_visual, style_to_json,
)


def _figure():
    figure = Figure()
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    first, = axes.plot([0, 1], [1, 2], color="#123456", label="first")
    second, = axes.plot([0, 1], [2, 3], color="#654321", label="second")
    threshold = axes.axhline(1.5, color="#d62728", label="threshold")
    return figure, axes, first, second, threshold


def test_presets_match_the_optical_mapping_set():
    assert list(PRESETS) == ["Nature", "Science", "Cell", "IEEE", "Presentation", "Grayscale"]
    assert PRESETS["Presentation"]["font_size"] == 18.0
    assert PRESETS["IEEE"]["font_family"] == "DejaVu Serif"
    greys = PRESETS["Grayscale"]["series_palette"]
    assert all(colour[1:3] == colour[3:5] == colour[5:7] for colour in greys)
    for preset in PRESETS.values():
        style = VisualStyle(**{key: value for key, value in preset.items() if key != "label"},
                            suggested_preset=None)
        style.validate()


def test_palette_recolours_data_series_but_never_reference_lines():
    figure, _axes, first, second, threshold = _figure()
    palette = PRESETS["Grayscale"]["series_palette"]
    apply_visual_style(figure, VisualStyle(series_palette=list(palette)), "Light")
    assert to_hex(first.get_color()) == palette[0]
    assert to_hex(second.get_color()) == palette[1]
    assert to_hex(threshold.get_color()) == "#d62728"
    # Dropping the palette restores each series' own colour.
    apply_visual_style(figure, VisualStyle(), "Light")
    assert to_hex(first.get_color()) == "#123456"
    assert to_hex(second.get_color()) == "#654321"


def test_explicit_series_colour_outranks_the_palette():
    figure, _axes, first, _second, _threshold = _figure()
    style = VisualStyle(series_palette=["#000000"], series_colours={"line:0:first": "#abcdef"})
    apply_visual_style(figure, style, "Light")
    assert to_hex(first.get_color()) == "#abcdef"


def test_version_4_documents_load_without_a_palette():
    document = json.loads(style_to_json(VisualStyle(font_size=9.0), "v"))
    document["version"] = 4
    del document["style"]["series_palette"]
    style = style_from_json(json.dumps(document), "v")
    assert style.series_palette is None and style.font_size == 9.0
    assert json.loads(style_to_json(style, "v"))["schema"] == SCHEMA


def test_invalid_palette_is_rejected():
    with pytest.raises(ValueError, match="series_palette"):
        VisualStyle(series_palette=[]).validate()
    with pytest.raises(ValueError, match="series_palette"):
        VisualStyle(series_palette=["red"]).validate()


def test_fonts_resolve_to_approved_open_source_families():
    assert figure_fonts.resolve_figure_font(None) is None
    assert figure_fonts.resolve_figure_font("DejaVu Serif") == "DejaVu Serif"
    assert figure_fonts.resolve_figure_font("dejavu sans") == "DejaVu Sans"
    assert figure_fonts.resolve_figure_font("Arial") == "DejaVu Sans"
    assert figure_fonts.resolve_figure_font("Times New Roman") == "DejaVu Serif"
    assert figure_fonts.resolve_figure_font("Courier New") == "DejaVu Sans Mono"
    offered = figure_fonts.available_fonts()
    assert offered[:3] == ("DejaVu Sans", "DejaVu Serif", "DejaVu Sans Mono")
    assert all(figure_fonts.approved_font(name) is not None for name in offered)
    assert all(font.spdx and font.source for font in figure_fonts.APPROVED_FONTS)


def test_an_unapproved_saved_family_renders_with_its_approved_substitute():
    figure, axes, *_ = _figure()
    axes.set_title("Title")
    apply_visual_style(figure, VisualStyle(font_family="Georgia"), "Light")
    assert axes.title.get_fontfamily() == ["DejaVu Serif"]


def test_default_style_round_trip_strips_per_figure_settings(tmp_path):
    style = VisualStyle(font_size=12.0, series_colours={"line:0:a": "#111111"},
                        series_styles={"line:0:a": {"line_width": 2.0}})
    save_default_style(tmp_path, style)
    loaded = load_default_style(tmp_path)
    assert loaded == as_default_style(style)
    assert loaded.series_colours == {} and loaded.series_styles == {}
    clear_default_style(tmp_path)
    assert load_default_style(tmp_path) is None
    clear_default_style(tmp_path)  # clearing twice is harmless


def test_a_damaged_default_fails_closed(tmp_path):
    path = default_style_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert load_default_style(tmp_path) is None


def test_style_resolution_order_is_own_then_default_then_theme(tmp_path):
    assert load_style_for_visual(tmp_path, "figure:a") == (VisualStyle(), "theme")
    save_default_style(tmp_path, VisualStyle(font_size=11.0))
    style, source = load_style_for_visual(tmp_path, "figure:a")
    assert (style.font_size, source) == (11.0, "default")
    own = style_path_for_visual(tmp_path, "figure:a")
    own.write_text(style_to_json(VisualStyle(font_size=14.0), "figure:a"), encoding="utf-8")
    style, source = load_style_for_visual(tmp_path, "figure:a")
    assert (style.font_size, source) == (14.0, "visual")


def test_new_figure_cards_use_the_saved_default(isolated_user_config):
    from PySide6.QtWidgets import QApplication
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from pyCamSet.gui.shared_functions import MatplotlibFigureCard

    QApplication.instance() or QApplication([])
    save_default_style(isolated_user_config, VisualStyle(font_size=13.0))
    figure = Figure()
    axes = figure.add_subplot(111)
    axes.set_title("Per-camera error")
    card = MatplotlibFigureCard("Per-camera error", figure, FigureCanvasQTAgg)
    assert card._style.font_size == 13.0
    assert axes.title.get_fontsize() == 13.0


def test_dialog_offers_every_preset_and_saves_the_default(isolated_user_config):
    from PySide6.QtWidgets import QApplication
    from pyCamSet.gui.visual_style import VisualStyleDialog

    QApplication.instance() or QApplication([])
    figure, *_ = _figure()
    dialog = VisualStyleDialog(figure, "figure:dialog", VisualStyle(), "Light")
    try:
        labels = [dialog.preset.itemText(i) for i in range(dialog.preset.count())]
        assert labels[0] == "Custom"
        assert "Presentation (large)" in labels and "Grayscale/minimal" in labels
        assert all(figure_fonts.approved_font(dialog.font.itemText(i))
                   for i in range(1, dialog.font.count()))
        dialog.preset.setCurrentIndex(dialog.preset.findData("Grayscale"))
        chosen = dialog._read()
        assert chosen.suggested_preset == "Grayscale"
        assert chosen.series_palette == PRESETS["Grayscale"]["series_palette"]
        assert chosen.text_colour == "#000000"
        assert not dialog.clear_default.isEnabled()
        dialog._save_as_default()
        assert load_default_style(isolated_user_config).suggested_preset == "Grayscale"
        assert dialog.clear_default.isEnabled()
        dialog._clear_default()
        assert load_default_style(isolated_user_config) is None
    finally:
        dialog.deleteLater()


def test_legend_swatches_follow_restyled_lines():
    figure, axes, first, _second, threshold = _figure()
    legend = axes.legend()
    handles = dict(zip((text.get_text() for text in legend.get_texts()),
                       getattr(legend, "legend_handles", None) or legend.legendHandles))
    apply_visual_style(figure, VisualStyle(series_palette=["#000000", "#888888"]), "Light")
    assert to_hex(handles["first"].get_color()) == "#000000"
    assert to_hex(handles["second"].get_color()) == "#888888"
    assert to_hex(handles["threshold"].get_color()) == "#d62728"
