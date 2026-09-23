"""Purpose: verify diagnostic run-selector and figure accessibility states.
Status: Focused offscreen presentation contracts for the diagnostics UI.
Future: Extend with image-based geometry checks when stable screenshot fixtures exist.
"""

import pytest


@pytest.mark.gui
def test_run_selector_explains_empty_and_populated_states():
    from PySide6.QtWidgets import QApplication

    from pyCamSet.gui.shared_functions import RunSelectorWidget

    QApplication.instance() or QApplication([])
    widget = RunSelectorWidget([])
    try:
        assert widget._list.isHidden()
        assert not widget._empty_lbl.isHidden()
        assert "No saved runs" in widget._empty_lbl.text()
        assert "0 of 0 runs selected" in widget._selection_summary.text()

        widget.refresh([{"run_id": "run-a"}, {"run_id": "run-b"}])
        assert widget._list.count() == 2
        assert not widget._list.isHidden()
        assert widget._empty_lbl.isHidden()
        assert "2 of 2 runs selected" in widget._selection_summary.text()

        widget._list.item(0).setSelected(False)
        assert "1 of 2 runs selected" in widget._selection_summary.text()
        assert widget._list.accessibleName() == "Saved runs"
        assert "diagnostics" in widget._list.accessibleDescription()
    finally:
        widget.deleteLater()


@pytest.mark.gui
def test_matplotlib_figure_card_exposes_visible_title_to_accessibility():
    from PySide6.QtWidgets import QApplication
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    from pyCamSet.gui.shared_functions import MatplotlibFigureCard

    QApplication.instance() or QApplication([])
    card = MatplotlibFigureCard(
        "D2.6 Per-view reprojection error (px)",
        Figure(),
        FigureCanvasQTAgg,
    )
    try:
        assert card._canvas.accessibleName() == "D2.6 Per-view reprojection error (px)"
        assert "Scientific figure" in card._canvas.accessibleDescription()
    finally:
        card.deleteLater()


@pytest.mark.gui
def test_diagnostic_cards_reuse_stable_style_and_isolate_diagnostics(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    from pyCamSet.gui import preferences
    from pyCamSet.gui.shared_functions import MatplotlibFigureCard
    from pyCamSet.gui.visual_style import VisualStyle, style_path_for_visual, style_to_json

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(preferences, "config_directory", lambda: tmp_path)
    diagnostic_id = "diagnostic:phase2:d2.6-d2.7-per-image-reprojection"
    other_id = "diagnostic:phase3:d3.4-per-image-initial-reprojection"
    saved_path = style_path_for_visual(tmp_path, diagnostic_id)
    other_path = style_path_for_visual(tmp_path, other_id)
    saved_path.parent.mkdir(parents=True)
    saved_path.write_text(style_to_json(VisualStyle(line_width=4.25), diagnostic_id), encoding="utf-8")
    other_path.write_text(style_to_json(VisualStyle(line_width=2.75), other_id), encoding="utf-8")

    first = MatplotlibFigureCard("D2.6/D2.7 reprojection error (run-A)", Figure(),
                                 FigureCanvasQTAgg, visual_id=diagnostic_id)
    second = MatplotlibFigureCard("D2.6/D2.7 reprojection error (run-B)", Figure(),
                                  FigureCanvasQTAgg, visual_id=diagnostic_id)
    third = MatplotlibFigureCard("D3.4 initial reprojection error (run-A)", Figure(),
                                 FigureCanvasQTAgg, visual_id=other_id)
    fourth = MatplotlibFigureCard("D3.4 initial reprojection error (run-B)", Figure(),
                                  FigureCanvasQTAgg, visual_id=other_id)
    try:
        assert first._visual_id == second._visual_id == diagnostic_id
        assert first._style_path == second._style_path == saved_path
        assert first._style.line_width == second._style.line_width == 4.25
        assert third._style_path == fourth._style_path == other_path
        assert third._visual_id == fourth._visual_id == other_id
        assert third._style.line_width == fourth._style.line_width == 2.75
        assert other_path != saved_path
    finally:
        first.deleteLater()
        second.deleteLater()
        third.deleteLater()
        fourth.deleteLater()


@pytest.mark.gui
def test_stable_diagnostic_style_reads_legacy_title_path_without_rewriting(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    from pyCamSet.gui import preferences
    from pyCamSet.gui.shared_functions import MatplotlibFigureCard
    from pyCamSet.gui.visual_style import VisualStyle, style_path_for_visual, style_to_json

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(preferences, "config_directory", lambda: tmp_path)
    old_title = "D2.6/D2.7 per-image reprojection error (run-A)"
    old_id = "figure:d2-6-d2-7-per-image-reprojection-error-run-a"
    old_path = style_path_for_visual(tmp_path, old_id)
    old_path.parent.mkdir(parents=True)
    old_text = style_to_json(VisualStyle(font_size=17), old_id)
    old_path.write_text(old_text, encoding="utf-8")
    stable_id = "diagnostic:phase2:d2.6-d2.7-per-image-reprojection"

    card = MatplotlibFigureCard(old_title, Figure(), FigureCanvasQTAgg, visual_id=stable_id)
    try:
        assert card._style.font_size == 17
        assert card._style_path != old_path
        assert old_path.read_text(encoding="utf-8") == old_text
        assert not card._style_path.exists()
    finally:
        card.deleteLater()
