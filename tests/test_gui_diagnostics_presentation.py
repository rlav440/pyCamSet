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
