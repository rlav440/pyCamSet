"""
pyCamSet GUI — main application window (PySide6).

Launch with::

    python -m pyCamSet.gui.main
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pyCamSet.gui.shared_functions import TAB_PHASE0, TAB_PHASE1, TAB_PHASE1_DIAG, WorkspaceManager


class PyCamSetApp(QMainWindow):
    """Top-level application window.

    Responsibilities
    ----------------
    - Two global checkboxes wired into every child tab:
        1. **Enable Informational Windows** — enables Qt tool-tips.
        2. **Show Terminal Output** — shows/hides the per-tab terminal pane.
    - A ``QTabWidget`` with four top-level tabs:
        ``Phase 0`` | ``Phase 0 Diagnostics`` | ``Phase 1`` | ``Phase 1 Diagnostics``
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("pyCamSet — Multi-Camera Calibration")
        self.resize(1140, 820)
        self.setMinimumSize(860, 640)

        # Shared state injected into child tabs
        self._info_cb = QCheckBox("Enable Informational Windows")
        self._info_cb.setChecked(True)
        self._info_cb.stateChanged.connect(self._on_info_toggle)

        self._terminal_cb = QCheckBox("Show Terminal Output")
        self._terminal_cb.setChecked(True)

        # Shared workspace manager (path updated when f_loc is set)
        self._workspace_mgr = WorkspaceManager(Path(".pycamset_workspace"))

        self._build_ui()
        self._on_info_toggle()  # apply initial tooltip state

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Deferred imports so module is importable without a display server
        from pyCamSet.gui.phase_0_input import Phase0Tab
        from pyCamSet.gui.phase_1_detection import Phase1DiagnosticsTab, Phase1Tab

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(4)

        # ── Global controls ────────────────────────────────────────────
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(self._info_cb)
        ctrl_row.addSpacing(16)
        ctrl_row.addWidget(self._terminal_cb)
        ctrl_row.addStretch()
        root_layout.addLayout(ctrl_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        root_layout.addWidget(sep)

        # ── Tab widget ─────────────────────────────────────────────────
        ws = self._workspace_mgr

        self._notebook = QTabWidget()
        root_layout.addWidget(self._notebook)

        self.phase0_tab = Phase0Tab(
            notebook=self._notebook,
            info_cb=self._info_cb,
            terminal_cb=self._terminal_cb,
            workspace_mgr=ws,
        )
        self._notebook.addTab(self.phase0_tab, TAB_PHASE0)

        self.phase1_tab = Phase1Tab(
            notebook=self._notebook,
            info_cb=self._info_cb,
            terminal_cb=self._terminal_cb,
            workspace_mgr=ws,
        )
        self._notebook.addTab(self.phase1_tab, TAB_PHASE1)

        self.phase1_diag_tab = Phase1DiagnosticsTab(
            notebook=self._notebook,
            info_cb=self._info_cb,
            workspace_mgr=ws,
        )
        diag_idx = self._notebook.addTab(self.phase1_diag_tab, TAB_PHASE1_DIAG)
        self._notebook.tabBar().setTabVisible(diag_idx, False)

        # Cross-tab wiring
        self.phase1_tab.set_diagnostics_tab(self.phase1_diag_tab)
        self.phase0_tab.set_phase1_path_callback(self.phase1_tab.set_image_folder)

    def _on_info_toggle(self) -> None:
        """Enable or disable all Qt tool-tips application-wide."""
        QApplication.instance().setProperty(
            "tooltipsEnabled", self._info_cb.isChecked()
        )

    def switch_to_tab(self, name: str) -> None:
        """Switch to the named tab by its display text."""
        for i in range(self._notebook.count()):
            if self._notebook.tabText(i) == name:
                self._notebook.setCurrentIndex(i)
                return


def main() -> None:
    """Launch the pyCamSet GUI."""
    import sys

    app = QApplication.instance() or QApplication(sys.argv)
    window = PyCamSetApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
