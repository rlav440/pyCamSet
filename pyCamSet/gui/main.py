"""
pyCamSet GUI — main application window.

Launch with::

    python -m pyCamSet.gui.main

or from the package entry-point if configured in ``setup.cfg``.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from pathlib import Path

from pyCamSet.gui.shared_functions import WorkspaceManager, TAB_PHASE0_DIAG, TAB_PHASE1


class PyCamSetApp(tk.Tk):
    """Top-level application window.

    Responsibilities
    ----------------
    - Two global checkboxes wired into every child tab:
        1. **Enable Informational Windows** — gates hover tooltips.
        2. **Show Terminal Output** — shows/hides the per-tab terminal pane.
    - A ``ttk.Notebook`` with four top-level tabs:
        ``Phase 0`` | ``Phase 0 Diagnostics`` | ``Phase 1`` | ``Phase 1 Diagnostics``
    """

    def __init__(self) -> None:
        super().__init__()
        self.title("pyCamSet — Multi-Camera Calibration")
        self.geometry("1140x820")
        self.minsize(860, 640)

        # Shared state injected into child tabs
        self.info_var = tk.BooleanVar(value=True)
        self.show_terminal_var = tk.BooleanVar(value=True)

        # Workspace manager — updated by Phase0Tab once f_loc is set
        self._workspace_mgr = WorkspaceManager(Path(".pycamset_workspace"))

        self._build_global_controls()
        self._build_tabs()

    # ------------------------------------------------------------------

    def _build_global_controls(self) -> None:
        ctrl = ttk.Frame(self, padding=(8, 4))
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Checkbutton(
            ctrl,
            text="Enable Informational Windows",
            variable=self.info_var,
        ).pack(side=tk.LEFT, padx=(0, 16))

        ttk.Checkbutton(
            ctrl,
            text="Show Terminal Output",
            variable=self.show_terminal_var,
        ).pack(side=tk.LEFT)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X)

    def _build_tabs(self) -> None:
        # Deferred imports keep the module importable without a display server
        from pyCamSet.gui.phase_0_input import Phase0Tab, Phase0DiagnosticsTab
        from pyCamSet.gui.phase_1_detection import Phase1Tab, Phase1DiagnosticsTab

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        ws = self._workspace_mgr

        self.phase0_tab = Phase0Tab(
            self.notebook,
            notebook=self.notebook,
            info_var=self.info_var,
            show_terminal_var=self.show_terminal_var,
            workspace_mgr=ws,
        )
        self.notebook.add(self.phase0_tab, text="Phase 0")

        self.phase0_diag_tab = Phase0DiagnosticsTab(
            self.notebook,
            notebook=self.notebook,
            info_var=self.info_var,
            workspace_mgr=ws,
        )
        self.notebook.add(self.phase0_diag_tab, text=TAB_PHASE0_DIAG)

        self.phase1_tab = Phase1Tab(
            self.notebook,
            notebook=self.notebook,
            info_var=self.info_var,
            show_terminal_var=self.show_terminal_var,
            workspace_mgr=ws,
        )
        self.notebook.add(self.phase1_tab, text=TAB_PHASE1)

        self.phase1_diag_tab = Phase1DiagnosticsTab(
            self.notebook,
            notebook=self.notebook,
            info_var=self.info_var,
            workspace_mgr=ws,
        )
        self.notebook.add(self.phase1_diag_tab, text="Phase 1 Diagnostics")

        # Cross-tab wiring
        self.phase0_tab.set_diagnostics_tab(self.phase0_diag_tab)
        self.phase1_tab.set_diagnostics_tab(self.phase1_diag_tab)


def main() -> None:
    """Launch the pyCamSet GUI."""
    app = PyCamSetApp()
    app.mainloop()


if __name__ == "__main__":
    main()
