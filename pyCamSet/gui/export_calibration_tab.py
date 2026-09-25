"""Export Calibration tab.

Thin GUI wrapper around :func:`pyCamSet.utils.saving.camset_to_colmap` and
:func:`pyCamSet.utils.saving.camset_to_apde`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pyCamSet.gui.theme import set_text_role
from pyCamSet.gui.assess_calibration import merge_phase3_phase4_runs, resolve_run_camset_artifact
from pyCamSet.gui.shared_functions import RunSelectorWidget, TerminalWidget, WorkspaceManager, make_blue_button, make_green_button, make_section_label, make_separator

try:
    from pyCamSet.utils.saving import camset_to_apde, camset_to_colmap, load_CameraSet

    _PYCAMSET_OK = True
except ImportError:
    camset_to_apde = None
    camset_to_colmap = None
    load_CameraSet = None
    _PYCAMSET_OK = False

#: The export formats offered, by the name shown for each -- same
#: display-name -> internal-key idiom as ``create_target.py``'s
#: ``_EXPORT_CHOICES``.
_FORMAT_CHOICES = {
    "COLMAP": "colmap",
    "APDe-MVS": "apde",
}


class ExportCalibrationTab(QWidget):
    """Export selected Phase 3/4 calibrations to COLMAP or APDe-MVS format."""

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

        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("Export format:"))
        self._format_combo = QComboBox()
        self._format_combo.addItems(list(_FORMAT_CHOICES))
        self._format_combo.currentIndexChanged.connect(self._on_format_changed)
        format_row.addWidget(self._format_combo)
        format_row.addStretch()
        root.addLayout(format_row)

        # APDe-MVS needs a depth range the calibration itself cannot supply;
        # COLMAP users have no use for it, so it is hidden until selected.
        self._depth_row_widget = QWidget()
        depth_row = QHBoxLayout(self._depth_row_widget)
        depth_row.setContentsMargins(0, 0, 0, 0)
        depth_row.addWidget(QLabel("Depth min:"))
        self._depth_min_edit = QLineEdit("0.1")
        self._depth_min_edit.setFixedWidth(70)
        depth_row.addWidget(self._depth_min_edit)
        depth_row.addWidget(QLabel("Depth max:"))
        self._depth_max_edit = QLineEdit("0.8")
        self._depth_max_edit.setFixedWidth(70)
        depth_row.addWidget(self._depth_max_edit)
        depth_row.addWidget(QLabel("Depth steps:"))
        self._depth_num_edit = QLineEdit("192")
        self._depth_num_edit.setFixedWidth(70)
        depth_row.addWidget(self._depth_num_edit)
        depth_row.addStretch()
        root.addWidget(self._depth_row_widget)

        actions = QHBoxLayout()
        refresh_btn = QPushButton("Refresh Runs")
        refresh_btn.clicked.connect(self.refresh)
        actions.addWidget(refresh_btn)
        self._export_btn = make_green_button("Export Selected Runs", self._export_selected)
        actions.addWidget(self._export_btn)
        # Enabled only with a selection, and the tooltip says why when not.
        self._run_selector.selection_changed.connect(self._update_export_enabled)
        actions.addStretch()
        root.addLayout(actions)

        self._status_lbl = QLabel("")
        set_text_role(self._status_lbl, "success")
        root.addWidget(self._status_lbl)
        # Spare height collects here, not in gaps between the rows above.
        root.addStretch(1)

        self._terminal = TerminalWidget(terminal_cb, parent=self)
        root.addWidget(self._terminal)

        self._on_format_changed()
        self.refresh()
        self._update_export_enabled()

    def _update_export_enabled(self, *_args) -> None:
        """Enable export only when at least one run is selected."""
        has_selection = bool(self._run_selector.get_selected())
        reason = ("Export the selected runs in the chosen format." if has_selection
                  else "Select one or more Phase 3/4 runs above to export them.")
        self._export_btn.setEnabled(has_selection)
        # A disabled button leaves the tab order, so the reason is also given
        # to assistive technology, not only as a hover tooltip.
        self._export_btn.setToolTip(reason)
        self._export_btn.setAccessibleDescription(reason)

    def _selected_format(self) -> str:
        """The internal key -- ``"colmap"`` or ``"apde"`` -- for the combo's current text."""
        return _FORMAT_CHOICES[self._format_combo.currentText()]

    def _on_format_changed(self) -> None:
        """Show the depth-range inputs only when they mean something (APDe-MVS)."""
        self._depth_row_widget.setVisible(self._selected_format() == "apde")

    def _read_depth_params(self) -> Optional[tuple[float, float, int]]:
        """Parse and validate the depth-range boxes, or report why not.

        Returns None (after showing the error) rather than raising, so the
        caller can bail out before touching a single run.
        """
        try:
            depth_min = float(self._depth_min_edit.text().strip())
            depth_max = float(self._depth_max_edit.text().strip())
            depth_num = int(self._depth_num_edit.text().strip())
        except ValueError:
            QMessageBox.critical(
                self, "Validation Error",
                "Depth min/max must be numbers and depth steps must be an integer.")
            return None

        if not depth_min < depth_max:
            QMessageBox.critical(
                self, "Validation Error", "Depth min must be less than depth max.")
            return None
        if depth_num < 1:
            QMessageBox.critical(
                self, "Validation Error", "Depth steps must be at least 1.")
            return None

        return depth_min, depth_max, depth_num

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
        if (not _PYCAMSET_OK or camset_to_colmap is None
                or camset_to_apde is None or load_CameraSet is None):
            QMessageBox.critical(self, "Unavailable", "Export dependencies are unavailable.")
            return

        ws = self._workspace_mgr.workspace_path
        if ws is None:
            QMessageBox.critical(self, "Workspace Required", "Set the workspace first (Phase 0 image folder).")
            return

        selected = self._run_selector.get_selected()
        if not selected:
            QMessageBox.information(self, "No Runs Selected", "Select one or more phase3/phase4 runs first.")
            return

        export_format = self._selected_format()
        depth_params: Optional[tuple[float, float, int]] = None
        if export_format == "apde":
            depth_params = self._read_depth_params()
            if depth_params is None:
                return  # the validation error is already on screen

        success = 0
        failures = 0
        for run in selected:
            run_id = str(run.get("run_id", "unknown"))
            phase = str(run.get("phase", ""))
            if phase not in {"phase3", "phase4"}:
                failures += 1
                self._terminal.append_line(f"SKIP {run_id}: unsupported phase '{phase}'.")
                continue

            camset_path = resolve_run_camset_artifact(run, accepted_only=True)
            if camset_path is None:
                failures += 1
                if phase == "phase4" and run.get("status") != "complete":
                    self._terminal.append_line(
                        f"FAIL {run_id}: Phase 4 status is not complete; "
                        "incomplete results are diagnostic-only.")
                else:
                    self._terminal.append_line(f"FAIL {run_id}: no camset artifact found.")
                continue

            run_dir = Path(ws) / f"{phase}_runs" / run_id
            out_dir = run_dir / "sparse" / "0" if export_format == "colmap" else run_dir / "apde"

            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                cams = load_CameraSet(camset_path)
                if export_format == "colmap":
                    camset_to_colmap(cams, out_dir)
                    success += 1
                    self._terminal.append_line(f"OK   {run_id}: wrote cameras.txt and rig_config.json to {out_dir}")
                else:
                    depth_min, depth_max, depth_num = depth_params
                    camset_to_apde(cams, out_dir, depth_min=depth_min, depth_max=depth_max, depth_num=depth_num)
                    success += 1
                    self._terminal.append_line(
                        f"OK   {run_id}: wrote cams/, cam_index_map.txt and pair.txt to {out_dir}")
                    # ACMMP/APDe-MVS also needs an images/ folder next to cams/, with
                    # files indexed identically to cam_index_map.txt -- this exporter
                    # does not write one, so the terminal has to say so explicitly
                    # rather than leaving the "OK" line implying a complete export.
                    self._terminal.append_line(
                        "     -> supply images/ yourself: one image per line of "
                        "cam_index_map.txt, in that same order.")
                    # Each reference view's scores are normalised to its best
                    # candidate (see normalise_pair_scores), so the numbers are
                    # relative within a row and mean nothing between rows. Said
                    # here so nobody reads a 1 as a co-visibility measurement.
                    self._terminal.append_line(
                        "     -> pair.txt ranks each view's neighbours, scored relative to its "
                        "best candidate (1): use the order, not the values.")
                    distorted = [
                        name for name in cams.get_names()
                        if np.any(np.abs(np.asarray(cams[name].distortion_coefs)) > 1e-9)
                    ]
                    if distorted:
                        self._terminal.append_line(
                            f"     -> {len(distorted)} camera(s) have non-negligible distortion "
                            f"({', '.join(distorted)}): those images must be undistorted first.")
            except Exception as exc:
                failures += 1
                self._terminal.append_line(f"FAIL {run_id}: {exc}")

        self._status_lbl.setText(f"Export completed. Success: {success}, Failures: {failures}")
