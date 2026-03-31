"""Export Calibration tab.

Thin GUI wrapper around :func:`pyCamSet.utils.saving.camset_to_colmap`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pyCamSet.gui.assess_calibration import merge_phase3_phase4_runs, resolve_run_camset_artifact
from pyCamSet.gui.shared_functions import RunSelectorWidget, TerminalWidget, WorkspaceManager, make_blue_button, make_section_label, make_separator

try:
    from pyCamSet.utils.saving import camset_to_colmap, load_CameraSet

    _PYCAMSET_OK = True
except ImportError:
    camset_to_colmap = None
    load_CameraSet = None
    _PYCAMSET_OK = False


class ExportCalibrationTab(QWidget):
    """Export selected Phase 3/4 calibrations to COLMAP sparse/0 format."""

    def __init__(
        self,
        notebook: QTabWidget,
        info_cb: QCheckBox,
        terminal_cb: QCheckBox,
        workspace_mgr: WorkspaceManager,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._notebook = notebook
        self._info_cb = info_cb
        self._workspace_mgr = workspace_mgr
        self._all_runs: list[dict] = []
        self._build_ui(terminal_cb)

    def _build_ui(self, terminal_cb: QCheckBox) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        root.addWidget(make_section_label("Workspace and Calibration Runs"))

        self._ws_edit = QLineEdit()
        self._ws_edit.setReadOnly(True)
        root.addWidget(self._ws_edit)

        run_row = QHBoxLayout()
        self._run_selector = RunSelectorWidget(runs=[])
        run_row.addWidget(self._run_selector)
        root.addLayout(run_row)

        root.addWidget(make_separator())

        actions = QHBoxLayout()
        refresh_btn = QPushButton("Refresh Runs")
        refresh_btn.clicked.connect(self.refresh)
        actions.addWidget(refresh_btn)
        actions.addWidget(make_blue_button("Export to COLMAP Format", self._export_selected))
        actions.addStretch()
        root.addLayout(actions)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #2e7d32;")
        root.addWidget(self._status_lbl)

        self._terminal = TerminalWidget(terminal_cb, parent=self)
        root.addWidget(self._terminal)

        self.refresh()

    def refresh(self) -> None:
        ws = self._workspace_mgr.workspace_path
        self._ws_edit.setText(str(ws) if ws is not None else "Workspace not set.")

        phase3_runs = self._workspace_mgr.load_runs("phase3")
        phase4_runs = self._workspace_mgr.load_runs("phase4")
        merged = merge_phase3_phase4_runs(phase3_runs, phase4_runs)

        display_runs: list[dict] = []
        for run in merged:
            copied = dict(run)
            camset_path = resolve_run_camset_artifact(copied)
            base_name = copied.get("display_name") or copied.get("run_id", "unknown")
            if camset_path is None:
                copied["display_name"] = f"{base_name}  [missing camset]"
            display_runs.append(copied)

        self._all_runs = display_runs
        self._run_selector.refresh(display_runs)
        self._terminal.append_line(
            f"Loaded {len(display_runs)} run(s): phase3={len(phase3_runs)}, phase4={len(phase4_runs)}"
        )

    def _export_selected(self) -> None:
        if not _PYCAMSET_OK or camset_to_colmap is None or load_CameraSet is None:
            QMessageBox.critical(self, "Unavailable", "COLMAP export dependencies are unavailable.")
            return

        ws = self._workspace_mgr.workspace_path
        if ws is None:
            QMessageBox.critical(self, "Workspace Required", "Set the workspace first (Phase 0 image folder).")
            return

        selected = self._run_selector.get_selected()
        if not selected:
            QMessageBox.information(self, "No Runs Selected", "Select one or more phase3/phase4 runs first.")
            return

        success = 0
        failures = 0
        for run in selected:
            run_id = str(run.get("run_id", "unknown"))
            phase = str(run.get("phase", ""))
            if phase not in {"phase3", "phase4"}:
                failures += 1
                self._terminal.append_line(f"SKIP {run_id}: unsupported phase '{phase}'.")
                continue

            camset_path = resolve_run_camset_artifact(run)
            if camset_path is None:
                failures += 1
                self._terminal.append_line(f"FAIL {run_id}: no camset artifact found.")
                continue

            run_dir = Path(ws) / f"{phase}_runs" / run_id
            out_dir = run_dir / "sparse" / "0"

            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                cams = load_CameraSet(camset_path)
                camset_to_colmap(cams, out_dir)
                success += 1
                self._terminal.append_line(f"OK   {run_id}: wrote cameras.txt and rig_config.json to {out_dir}")
            except Exception as exc:
                failures += 1
                self._terminal.append_line(f"FAIL {run_id}: {exc}")

        self._status_lbl.setText(f"Export completed. Success: {success}, Failures: {failures}")

