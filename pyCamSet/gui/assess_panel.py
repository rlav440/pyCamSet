"""The Assess Calibration page shared by the Phase 3 and Phase 4 diagnostics.

The three 2D assessment figures draw in the page as soon as it is opened for
a run: they are cheap once the calibration's diagnostics are computed, and
that computation runs off the GUI thread.  The two PyVista scenes are heavier
and wait for **Visualise Calibration**.  They are embedded with ``pyvistaqt``,
whose render window lives inside Qt's own event loop; what cannot share the
GUI process is a native pyvista window, which runs a nested Cocoa loop on
macOS (see :mod:`pyCamSet.utils.visualise_camset`).  So the separate-window
choices still go to a viewer process, as does Open3D.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import shiboken6
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QMenu, QMessageBox, QScrollArea, QSizePolicy,
    QToolButton, QVBoxLayout, QWidget,
)

from pyCamSet.gui import assess_calibration as assess
from pyCamSet.gui.theme import set_text_role

_LOGGER = logging.getLogger(__name__)

#: The 2D figures, in page order: key, card title.
FIGURES = (
    ("error_distribution", "Error distribution"),
    ("per_camera_coverage", "Camera coverage"),
    ("accuracy_precision", "Accuracy vs precision"),
)

#: Where the 3D scenes go, as shown in the backend chooser.
BACKEND_EMBEDDED = "embedded"
BACKEND_PYVISTA_WINDOW = "pyvista-window"
BACKEND_OPEN3D_WINDOW = "open3d-window"

EXPORT_SIZES = (
    ("Screen · 160 mm · 150 dpi", (160.0, 150)),
    ("Single column · 85 mm · 300 dpi", (85.0, 300)),
    ("Double column · 180 mm · 300 dpi", (180.0, 300)),
)


def embedding_available() -> bool:
    """Whether PyVista scenes can be embedded in the page (``pyvistaqt``)."""
    try:
        import pyvistaqt  # noqa: F401  (availability probe)
    except Exception:  # ImportError, or a Qt binding clash inside it
        return False
    return True


def _make_interactor(parent: QWidget):
    """An embedded PyVista view.  Separate so tests can replace it."""
    from pyvistaqt import QtInteractor

    return QtInteractor(parent)


def _active_theme() -> str:
    application = QApplication.instance()
    return (application.property("pycamsetTheme") if application else None) or "Light"


def _run_key(run: Optional[dict]) -> Optional[tuple]:
    """What identifies a run's drawing: its camset file and when it was written."""
    if run is None:
        return None
    path = assess.resolve_run_camset_artifact(run)
    if path is None:
        return None
    try:
        return str(path), path.stat().st_mtime_ns
    except OSError:
        return None


class _DiagnosticsWorker(QThread):
    """Load a camset and compute its assessment diagnostics off the GUI thread."""

    ready = Signal(object, object)   # key, CalibrationDiagnostics
    failed = Signal(object, str)     # key, message

    def __init__(self, key: tuple, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._key = key

    def run(self) -> None:
        try:
            from pyCamSet.utils.saving import load_CameraSet
            from pyCamSet.utils.visualisation import CalibrationDiagnostics

            cams = load_CameraSet(Path(self._key[0]))
            handler = getattr(cams, "calibration_handler", None)
            results = assess._build_o_results(cams)
            if handler is None or results is None:
                raise ValueError("this camset carries no calibration results to assess")
            self.ready.emit(self._key, CalibrationDiagnostics.from_results(results, handler))
        except Exception as exc:  # reported in the page, never raised into Qt
            _LOGGER.exception("Assessment diagnostics failed")
            self.failed.emit(self._key, str(exc) or type(exc).__name__)


class ResponsiveGrid(QWidget):
    """Lay children out in as many columns as the width allows."""

    def __init__(self, min_column_width: int, max_columns: int,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._min_width = min_column_width
        self._max_columns = max_columns
        self._items: list[QWidget] = []
        self._columns = 0
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(10)

    def set_widgets(self, widgets: list[QWidget]) -> None:
        for widget in self._items:
            self._grid.removeWidget(widget)
        self._items = list(widgets)
        self._columns = 0
        self._relayout()

    def widgets(self) -> list[QWidget]:
        return list(self._items)

    def columns_for(self, width: int) -> int:
        return max(1, min(self._max_columns, width // self._min_width, len(self._items) or 1))

    def _relayout(self) -> None:
        columns = self.columns_for(self.width())
        if columns == self._columns and all(
                self._grid.indexOf(widget) >= 0 for widget in self._items):
            return
        self._columns = columns
        for widget in self._items:
            self._grid.removeWidget(widget)
        for index, widget in enumerate(self._items):
            self._grid.addWidget(widget, index // columns, index % columns)
        for column in range(self._max_columns):
            self._grid.setColumnStretch(column, 1 if column < columns else 0)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._relayout()


class AssessCalibrationPanel(QWidget):
    """Assess Calibration for one diagnostics tab.

    :param phase: ``"phase3"`` or ``"phase4"``; keys the saved styles and sizes
    :param choose_run: returns the run to assess, or None when none is selected
    """

    def __init__(self, phase: str, choose_run: Callable[[], Optional[dict]],
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._phase = phase
        self._choose_run = choose_run
        self._run: Optional[dict] = None
        self._key: Optional[tuple] = None
        self._diagnostics = None
        self._diagnostics_key: Optional[tuple] = None
        self._worker: Optional[_DiagnosticsWorker] = None
        self._want_3d = False
        self._interactors: list = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # ── Header: what is shown, and the few actions that act on all of it.
        header = QHBoxLayout()
        self.run_label = QLabel("No run selected")
        set_text_role(self.run_label, "muted")
        self.run_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        header.addWidget(self.run_label, 1)
        self.export_button = QToolButton()
        self.export_button.setText("Export ▾")
        self.export_button.setObjectName("toolbarMenuButton")
        self.export_button.setToolTip("Save the figures and 3D views for reports")
        self.export_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.export_button.setMenu(self._build_export_menu())
        header.addWidget(self.export_button)
        self.style_button = QToolButton()
        self.style_button.setText("3D style ▾")
        self.style_button.setObjectName("toolbarMenuButton")
        self.style_button.setCheckable(True)
        self.style_button.setToolTip("Background, point size, view, axes and legend of the 3D views")
        header.addWidget(self.style_button)
        root.addLayout(header)

        from pyCamSet.gui.three_d_style import ThreeDStyleControls

        self.three_d_style = ThreeDStyleControls(
            self, visual_id=f"assessment:{phase}", show_open3d_note=False)
        self.three_d_style.setVisible(False)
        self.three_d_style.changed.connect(self._restyle_3d)
        self.style_button.toggled.connect(self.three_d_style.setVisible)
        root.addWidget(self.three_d_style)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        self._body = QVBoxLayout(body)
        self._body.setContentsMargins(0, 0, 4, 0)
        self._body.setSpacing(8)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        # ── 2D figures: drawn automatically.
        self._body.addWidget(self._heading("Figures"))
        self.figure_status = QLabel("")
        self.figure_status.setWordWrap(True)
        set_text_role(self.figure_status, "muted")
        self._body.addWidget(self.figure_status)
        self.figure_grid = ResponsiveGrid(min_column_width=420, max_columns=3)
        self._body.addWidget(self.figure_grid)

        # ── 3D views: on request.
        row = QHBoxLayout()
        row.addWidget(self._heading("3D views"))
        row.addStretch()
        self.backend = QComboBox()
        self.backend.setAccessibleName("Where to show the 3D views")
        if embedding_available():
            self.backend.addItem("In this tab (PyVista)", BACKEND_EMBEDDED)
        self.backend.addItem("Separate window (PyVista)", BACKEND_PYVISTA_WINDOW)
        self.backend.addItem("Separate window (Open3D)", BACKEND_OPEN3D_WINDOW)
        self.backend.setToolTip(
            "In this tab: interactive views below.\n"
            "Separate window: a viewer of its own, which keeps working if you switch tabs.")
        row.addWidget(self.backend)
        from pyCamSet.gui.shared_functions import make_blue_button

        self.visualise_button = make_blue_button("Visualise Calibration", self.visualise_3d)
        self.visualise_button.setToolTip(
            "Load the interactive 3D views. They use more memory than the figures, "
            "so they wait for this button.")
        row.addWidget(self.visualise_button)
        self._body.addLayout(row)
        self.three_d_status = QLabel(
            "The reconstructed points and cameras, in the scene and in the target's own frame. "
            "Click Visualise Calibration to load them.")
        self.three_d_status.setWordWrap(True)
        set_text_role(self.three_d_status, "muted")
        self._body.addWidget(self.three_d_status)
        self.view_grid = ResponsiveGrid(min_column_width=480, max_columns=2)
        self._body.addWidget(self.view_grid)
        self._body.addStretch()

    # ------------------------------------------------------------------
    # Building blocks

    @staticmethod
    def _heading(text: str) -> QLabel:
        label = QLabel(text)
        set_text_role(label, "subheading")
        return label

    def _build_export_menu(self) -> QMenu:
        menu = QMenu(self)
        menu.addAction("Save the figures (PNG, SVG, PDF and CSV)…", self._save_figures)
        menu.addAction("Save the 3D views as PNG…", self._save_3d_png)
        menu.addAction("Export 3D geometry (glTF, OBJ, PLY)…", self._export_3d)
        menu.addSeparator()
        sizes = menu.addMenu("Export size")
        # A hidden combo holds the choice so the existing per-phase
        # preference binding persists it; the menu mirrors it.
        self.export_size = QComboBox(self)
        self.export_size.setVisible(False)
        for label, data in EXPORT_SIZES:
            self.export_size.addItem(label, data)
        from pyCamSet.gui.preferences import bind_export_preset

        bind_export_preset(self.export_size, f"{self._phase}:assessment-export")
        group = QActionGroup(sizes)
        for index, (label, _data) in enumerate(EXPORT_SIZES):
            action = sizes.addAction(label)
            action.setCheckable(True)
            action.setChecked(index == self.export_size.currentIndex())
            action.triggered.connect(lambda _c=False, i=index: self.export_size.setCurrentIndex(i))
            group.addAction(action)
        return menu

    # ------------------------------------------------------------------
    # Which run

    def set_run(self, run: Optional[dict]) -> None:
        """Show *run*; the figures load now if the page is visible, else when it is."""
        key = _run_key(run)
        if key == self._key and run is not None:
            return
        self._run, self._key = run, key
        self._clear_3d()
        self._want_3d = False
        if run is None:
            self.run_label.setText("No run selected")
        else:
            self.run_label.setText(
                f"Showing {run.get('phase', 'unknown')} run {run.get('run_id', 'unknown')}")
        if key is None:
            self.figure_grid.set_widgets([])
            self.figure_status.setText(
                "Select a run with a saved camset to assess it." if run is None
                else "This run has no readable camset to assess.")
            return
        if self.isVisible():
            self._load()
        else:
            self.figure_status.setText("The figures load when this page is opened.")

    def refresh_run(self) -> None:
        """Re-read the chosen run from the tab."""
        self.set_run(self._choose_run())

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self._key is not None and self._diagnostics_key != self._key:
            self._load()

    def _load(self) -> None:
        if self._diagnostics_key == self._key:
            self._draw_figures()
            return
        if self._worker is not None and self._worker.isRunning():
            # The finished worker's result is ignored if the run changed.
            self._worker.finished.connect(self._load_if_needed)
            return
        self.figure_status.setText("Computing the assessment…")
        self.figure_grid.set_widgets([])
        self._worker = _DiagnosticsWorker(self._key, self)
        self._worker.ready.connect(self._on_ready)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _load_if_needed(self) -> None:
        if self._key is not None and self._diagnostics_key != self._key and self.isVisible():
            self._load()

    def _on_ready(self, key: tuple, diagnostics) -> None:
        if key != self._key or not shiboken6.isValid(self):
            return
        self._diagnostics, self._diagnostics_key = diagnostics, key
        self._draw_figures()
        if self._want_3d:
            self._build_3d()

    def _on_failed(self, key: tuple, message: str) -> None:
        if key != self._key or not shiboken6.isValid(self):
            return
        self.figure_status.setText(f"Could not assess this run: {message}")
        set_text_role(self.figure_status, "danger")
        if self._want_3d:
            self.three_d_status.setText("The 3D views need the same assessment, which failed.")

    # ------------------------------------------------------------------
    # 2D figures

    def _figures(self, diagnostics) -> list[tuple[str, str, object]]:
        import matplotlib.pyplot as plt

        from pyCamSet.utils.visualisation import (
            accuracy_precision_plot, cluster_plot, per_camera_coverage,
        )
        built = {
            "error_distribution": cluster_plot([diagnostics.residuals]),
            "per_camera_coverage": per_camera_coverage(diagnostics),
            "accuracy_precision": accuracy_precision_plot(diagnostics),
        }
        # Drawn through pyplot by the library; the cards own them from here.
        for figure in built.values():
            plt.close(figure)
        return [(key, title, built[key]) for key, title in FIGURES]

    def _draw_figures(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        from pyCamSet.gui.shared_functions import MatplotlibFigureCard
        from pyCamSet.utils.visualisation import _assessment_csv_payloads

        diagnostics = self._diagnostics
        try:
            figures = self._figures(diagnostics)
            payloads = _assessment_csv_payloads(diagnostics)
        except Exception as exc:
            _LOGGER.exception("Assessment figures failed")
            self.figure_status.setText(f"Could not draw the figures: {exc}")
            set_text_role(self.figure_status, "danger")
            return
        for card in self.figure_grid.widgets():
            card.deleteLater()
        run = self._run or {}
        cards = []
        for key, title, figure in figures:
            columns, rows = payloads.get(key, ([], []))
            cards.append(MatplotlibFigureCard(
                title, figure, FigureCanvasQTAgg, parent=self.figure_grid,
                min_height=320, visual_id=f"assessment:{key}",
                csv_export={"columns": columns, "rows": rows, "metadata": {
                    "run_id": run.get("run_id"), "phase": run.get("phase", self._phase),
                    "diagnostic": key, "data_kind": "calibration assessment source arrays",
                }},
            ))
        self.figure_grid.set_widgets(cards)
        self.figure_status.setText(
            "Residuals, where on each sensor the error falls, and how consistently each "
            "target feature was recovered.")
        set_text_role(self.figure_status, "muted")

    # ------------------------------------------------------------------
    # 3D views

    def visualise_3d(self) -> None:
        """Show the 3D views where the backend chooser says."""
        run = self._run if self._run is not None else self._choose_run()
        if run is None:
            QMessageBox.information(self, "Assess Calibration", "Select a run first.")
            return
        if self._run is None:
            self.set_run(run)
        backend = self.backend.currentData()
        if backend == BACKEND_EMBEDDED:
            self._want_3d = True
            if self._diagnostics_key == self._key and self._diagnostics is not None:
                self._build_3d()
            else:
                self.three_d_status.setText("Loading the 3D views…")
                self._load()
            return
        if backend == BACKEND_OPEN3D_WINDOW:
            ok, message = assess.launch_visualise_calibration_open3d_for_run(run)
        else:
            ok, message = assess.launch_visualise_calibration_for_run(
                run, theme_name=_active_theme(),
                three_d_arguments=[*self.three_d_style.viewer_arguments(), "--3d-only"])
        if not ok:
            QMessageBox.warning(self, "Assess Calibration", message)
        else:
            self.three_d_status.setText(message)

    def _build_3d(self) -> None:
        self._clear_3d()
        self._want_3d = False
        try:
            views = [_make_interactor(self.view_grid) for _ in range(2)]
        except Exception as exc:
            _LOGGER.exception("Embedded 3D view failed")
            self.three_d_status.setText(
                f"The 3D views could not be embedded ({exc}); choose a separate window instead.")
            return
        for view in views:
            widget = getattr(view, "interactor", view)
            widget.setMinimumHeight(380)
        self._interactors = views
        self.view_grid.set_widgets(views)
        self._restyle_3d()
        self.three_d_status.setText(
            "Drag to rotate, scroll to zoom, shift-drag to pan. "
            "Left: the scene with its cameras. Right: the target's own frame.")

    def _restyle_3d(self) -> None:
        """Redraw the embedded views with the current style and theme."""
        if not self._interactors or self._diagnostics is None:
            return
        from pyCamSet.utils.visualisation import (
            _apply_3d_cosmetics, reconstruction_scene, target_space_scene,
        )
        style = self.three_d_style.values()
        reconstruction, target = self._interactors
        for view in self._interactors:
            view.clear()
            if hasattr(view, "hide_axes"):
                view.hide_axes()
        reconstruction_scene(self._diagnostics, style["point_size"], style["legend"],
                             plotter=reconstruction)
        target_space_scene(self._diagnostics, point_size=style["point_size"],
                           show_legend=style["legend"], plotter=target)
        for view in self._interactors:
            _apply_3d_cosmetics(view, _active_theme(), style["background"],
                                style["point_size"], style["view"], style["axes"])
            view.render()

    def _clear_3d(self) -> None:
        for view in self._interactors:
            try:
                view.close()
            except Exception:  # already torn down with its window
                pass
            view.deleteLater()
        self._interactors = []
        self.view_grid.set_widgets([])
        self.three_d_status.setText(
            "The reconstructed points and cameras, in the scene and in the target's own frame. "
            "Click Visualise Calibration to load them.")

    def closeEvent(self, event) -> None:  # noqa: N802
        self._clear_3d()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Exports

    def _require_run(self, title: str) -> Optional[dict]:
        run = self._run if self._run is not None else self._choose_run()
        if run is None:
            QMessageBox.warning(self, title, "Select a run first.")
        return run

    def _save_figures(self) -> None:
        run = self._require_run("Save figures")
        if run is None:
            return
        directory = QFileDialog.getExistingDirectory(self, "Save the assessment figures")
        if not directory:
            return
        theme = _active_theme()
        width_mm, dpi = self.export_size.currentData()
        ok, message = assess.launch_save_assessment_pngs_for_run(
            run, Path(directory), theme, width_mm, dpi, (theme,) * len(FIGURES))
        if ok:
            QMessageBox.information(self, "Save figures", message or "Saved the figures.")
        else:
            QMessageBox.warning(self, "Save figures", f"Could not save the figures:\n{message}")

    def _save_3d_png(self) -> None:
        run = self._require_run("Save 3D PNG")
        if run is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the 3D views as PNG", "calibration_assessment.png", "PNG Files (*.png)")
        if not path:
            return
        width_mm, dpi = self.export_size.currentData()
        ok, message = assess.launch_save_pyvista_png_for_run(
            run, Path(path), width_mm=width_mm, dpi=dpi, theme_name=_active_theme(),
            three_d_arguments=self.three_d_style.viewer_arguments())
        if ok:
            QMessageBox.information(self, "Save 3D PNG", message)
        else:
            QMessageBox.warning(self, "Save 3D PNG", f"Could not save the PNG:\n{message}")

    def _export_3d(self) -> None:
        run = self._require_run("3D export")
        if run is None:
            return
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export 3D geometry", "calibration_scene.gltf",
            "glTF scene (*.gltf);;Wavefront geometry (*.obj);;PLY point cloud (*.ply)")
        if not path:
            return
        if not Path(path).suffix:
            path += {"Wavefront geometry (*.obj)": ".obj",
                     "PLY point cloud (*.ply)": ".ply"}.get(selected_filter, ".gltf")
        ok, message = assess.launch_export_3d_for_run(run, Path(path))
        if ok:
            QMessageBox.information(self, "3D export", message)
        else:
            QMessageBox.warning(self, "3D export", message)
