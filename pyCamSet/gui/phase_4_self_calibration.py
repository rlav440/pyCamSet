"""
Phase 4 - Self-calibration GUI.

The form and the figures, plus Assess Calibration; the solve itself is
:mod:`pyCamSet.workflow.phase4`.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pyCamSet.gui.theme import set_text_role
from pyCamSet.gui.shared_functions import (
    CollapsibleSection,
    IMAGE_FOLDER_SCHEMATIC,
    PhaseWorker,
    RunSelectorWidget,
    TAB_PHASE4,
    TerminalWidget,
    WorkspaceManager,
    make_blue_button,
    make_green_button,
    make_orange_button,
    make_warning_button,
    make_section_label,
    make_separator,
    render_predecessor_chain_section,
    show_tab,
)
from pyCamSet.workflow import phase4 as phase4_workflow
from pyCamSet.gui.three_d_style import ThreeDStyleControls
from pyCamSet.workflow.params import (
    ParamError,
    as_int,
    as_json_object,
    as_outlier_mode,
    as_positive_int,
    require_image_folder,
)
from pyCamSet.workflow.workspace import (
    path_exists,
    resolve_artifact,
)
from pyCamSet.gui.assess_calibration import (
    launch_visualise_calibration_for_run,
    launch_visualise_calibration_open3d_for_run,
    launch_save_pyvista_png_for_run,
    launch_export_3d_for_run,
    launch_save_assessment_pngs_for_run,
    merge_phase3_phase4_runs,
    select_latest_visualisation_run,
)

_LOG = logging.getLogger(__name__)

_PYCAMSET_OK = phase4_workflow.BACKEND_OK



class Phase4Tab(QWidget):
    def __init__(self, notebook: QTabWidget, info_cb: QCheckBox, terminal_cb: QCheckBox, workspace_mgr: WorkspaceManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._notebook = notebook
        self._info_cb = info_cb
        self._workspace_mgr = workspace_mgr
        self._diagnostics_tab: Optional["Phase4DiagnosticsTab"] = None
        self._worker: Optional[PhaseWorker] = None
        self._preferred_phase3_run_id: Optional[str] = None
        self._preferred_phase3_camset_path: Optional[str] = None
        self._build_ui(terminal_cb)

    def set_diagnostics_tab(self, tab: "Phase4DiagnosticsTab") -> None:
        self._diagnostics_tab = tab

    def set_phase3_run_id(self, run_id: str) -> None:
        self._preferred_phase3_run_id = (run_id or "").strip() or None
        self._refresh_phase3_sources()

    def set_selected_phase3_run_id(self, run_id: str) -> None:
        self.set_phase3_run_id(run_id)

    def set_phase3_camset_path(self, path: str) -> None:
        self._preferred_phase3_camset_path = (path or "").strip() or None
        self._phase3_camset_edit.setText(self._preferred_phase3_camset_path or "")

    def set_image_folder(self, path: str) -> None:
        self._floc_edit.setText(path)

    def _build_ui(self, terminal_cb: QCheckBox) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        top_row = QHBoxLayout()
        root.addLayout(top_row)

        form_widget = QWidget()
        form_root = QVBoxLayout(form_widget)
        # Pack sections at the top; spare height must not open gaps between them.
        form_root.setAlignment(Qt.AlignmentFlag.AlignTop)
        form_root.setContentsMargins(0, 0, 0, 0)
        form_root.setSpacing(4)
        form_scroll = QScrollArea()
        form_scroll.setWidgetResizable(True)
        form_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        form_scroll.setWidget(form_widget)
        top_row.addWidget(form_scroll, stretch=1)

        # ── Paths (collapsible) ────────────────────────────────────────
        paths_sect = CollapsibleSection("Paths", expanded=False)
        form_root.addWidget(paths_sect)

        floc_row = QHBoxLayout()
        self._floc_edit = QLineEdit()
        self._floc_edit.setPlaceholderText("Root folder with per-camera sub-folders")
        self._floc_edit.setToolTip(IMAGE_FOLDER_SCHEMATIC)
        self._floc_edit.textChanged.connect(self._sync_workspace_from_floc)
        floc_btn = QPushButton("Browse…")
        floc_btn.setMinimumWidth(70)
        floc_btn.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        floc_btn.clicked.connect(self._browse_floc)
        floc_row.addWidget(self._floc_edit)
        floc_row.addWidget(floc_btn)
        paths_sect.addRow("Image folder (f_loc):", floc_row)

        src_row = QHBoxLayout()
        self._phase3_run_combo = QComboBox()
        self._phase3_run_combo.currentIndexChanged.connect(self._on_phase3_source_changed)
        src_refresh = QPushButton("Refresh")
        src_refresh.setFixedWidth(70)
        src_refresh.clicked.connect(self._refresh_phase3_sources)
        src_row.addWidget(self._phase3_run_combo)
        src_row.addWidget(src_refresh)
        paths_sect.addRow("Phase 3 run:", src_row)

        p3cam_row = QHBoxLayout()
        self._phase3_camset_edit = QLineEdit()
        self._phase3_camset_edit.setPlaceholderText("Optional override: optimised phase3 camset")
        self._phase3_camset_edit.textChanged.connect(self._update_source_label)
        p3cam_btn = QPushButton("Browse…")
        p3cam_btn.setMinimumWidth(70)
        p3cam_btn.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        p3cam_btn.clicked.connect(self._browse_phase3_camset)
        p3cam_clear = QPushButton("Clear")
        p3cam_clear.setFixedWidth(55)
        p3cam_clear.clicked.connect(lambda: self._phase3_camset_edit.setText(""))
        p3cam_row.addWidget(self._phase3_camset_edit)
        p3cam_row.addWidget(p3cam_btn)
        p3cam_row.addWidget(p3cam_clear)
        paths_sect.addRow("Phase 3 camset path:", p3cam_row)

        self._source_lbl = QLabel("Source: auto")
        self._source_lbl.setWordWrap(True)
        set_text_role(self._source_lbl, "muted")
        paths_sect.addRow("", self._source_lbl)

        # ── Self-Calibration Options ───────────────────────────────────
        form_root.addWidget(make_separator())
        form_root.addWidget(make_section_label("Self-Calibration Options"))

        opts_form = QFormLayout()
        opts_form.setContentsMargins(0, 0, 0, 0)
        opts_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form_root.addLayout(opts_form)

        self._threads_edit = QLineEdit("1")
        self._threads_edit.setFixedWidth(100)
        self._threads_edit.setToolTip(
            "Concept: number of Jacobian evaluation threads used by scipy.\n\n"
            "Default: 1\n"
            "Range: positive integer\n"
            "Guidance: start with 1; increase for large problems if memory allows."
        )
        opts_form.addRow("Threads:", self._threads_edit)

        self._max_nfev_spin = QSpinBox()
        self._max_nfev_spin.setRange(5, 5000)
        self._max_nfev_spin.setValue(1000)
        self._max_nfev_spin.setFixedWidth(110)
        self._max_nfev_spin.setToolTip(
            "Concept: maximum cost-function evaluations for the solver.\n\n"
            "Default: 1000\n"
            "Range: 5–5000\n"
            "Guidance: increase above 1000 if the solver reports non-convergence.\n"
            "Self-calibration is sensitive; prefer more evaluations over fewer."
        )
        opts_form.addRow("max_nfev:", self._max_nfev_spin)

        self._verbosity_spin = QSpinBox()
        self._verbosity_spin.setRange(0, 2)
        self._verbosity_spin.setValue(2)
        self._verbosity_spin.setFixedWidth(110)
        self._verbosity_spin.setToolTip(
            "Concept: solver verbosity level.\n\n"
            "Default: 2\n"
            "Range: 0 (silent), 1 (summary), 2 (per-iteration)\n"
            "Guidance: keep at 2 to monitor self-calibration convergence."
        )
        opts_form.addRow("verbosity:", self._verbosity_spin)

        self._loss_combo = QComboBox()
        self._loss_combo.addItems(["linear", "soft_l1", "huber", "cauchy", "arctan"])
        self._loss_combo.setCurrentText("soft_l1")
        self._loss_combo.setFixedWidth(110)
        self._loss_combo.setToolTip(
            "Concept: scipy least_squares robust loss function -- downweights large\n"
            "residuals (e.g. outlier target points/poses) instead of letting them\n"
            "dominate the sum-of-squares cost.\n\n"
            "Default: soft_l1 (matches this project's validated D1 Phase 4 recipe;\n"
            "see run_dataset1_phase1_to_phase4_headless.py P4_PROBLEM_OPTIONS)\n"
            "Choices: linear (scipy default, no robustness), soft_l1, huber, cauchy, arctan\n"
            "Guidance: soft_l1 is the validated choice for self-calibration.  cauchy/arctan\n"
            "are more aggressive at suppressing outliers but can also suppress real signal."
        )
        opts_form.addRow("loss:", self._loss_combo)

        self._f_scale_spin = QDoubleSpinBox()
        # Minimum kept representable at 4 decimals (1e-6 would silently round to 0.0,
        # which would make the widget accept 0 despite scipy requiring f_scale > 0).
        self._f_scale_spin.setRange(0.0001, 1e6)
        self._f_scale_spin.setDecimals(4)
        self._f_scale_spin.setSingleStep(0.1)
        self._f_scale_spin.setValue(1.0)
        self._f_scale_spin.setFixedWidth(110)
        self._f_scale_spin.setToolTip(
            "Concept: soft threshold (in px) separating inlier from outlier residuals\n"
            "for the soft_l1/huber/cauchy/arctan loss functions.  Has no effect when\n"
            "loss=linear.\n\n"
            "Default: 1.0 (matches this project's validated D1 Phase 4 recipe)\n"
            "Range: > 0\n"
            "Guidance: leave at 1.0 unless residuals are known to be scaled differently."
        )
        opts_form.addRow("f_scale:", self._f_scale_spin)
        self._loss_combo.currentTextChanged.connect(self._on_loss_changed)
        self._on_loss_changed(self._loss_combo.currentText())

        self._outliers_combo = QComboBox()
        self._outliers_combo.setObjectName("outliers_combo")
        # Size to its short choices, like the numeric fields beside it.
        self._outliers_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._outliers_combo.addItems(["n", "y"])
        self._outliers_combo.setCurrentText("n")
        self._outliers_combo.setToolTip(
            "Concept: whether to reject outlier poses before self-calibration.\n\n"
            "Default: n (disabled)\n"
            "Choices: n (disabled), y (enabled)\n"
            "Guidance: enable only if Phase 3 already had clean data.  Outlier\n"
            "rejection in Phase 4 can discard useful target-shape information."
        )
        opts_form.addRow("outliers:", self._outliers_combo)

        self._fixed_pose_edit = QLineEdit("0")
        self._fixed_pose_edit.setFixedWidth(110)
        self._fixed_pose_edit.setToolTip(
            "Concept: pose index whose extrinsic is held fixed as the anchor.\n\n"
            "Default: 0\n"
            "Range: 0 to (num_poses − 1)\n"
            "Guidance: should match the value used in Phase 3."
        )
        opts_form.addRow("fixed_pose:", self._fixed_pose_edit)

        self._ref_cam_edit = QLineEdit("0")
        self._ref_cam_edit.setFixedWidth(110)
        self._ref_cam_edit.setToolTip(
            "Concept: camera index used as the metric reference.\n\n"
            "Default: 0\n"
            "Range: 0 to (num_cameras − 1)"
        )
        opts_form.addRow("ref_cam:", self._ref_cam_edit)

        self._ref_pose_edit = QLineEdit("0")
        self._ref_pose_edit.setFixedWidth(110)
        self._ref_pose_edit.setToolTip(
            "Concept: pose index used as the metric reference for the gauge.\n\n"
            "Default: 0\n"
            "Range: 0 to (num_poses − 1)"
        )
        opts_form.addRow("ref_pose:", self._ref_pose_edit)

        self._fp_edit = QLineEdit()
        self._fp_edit.setPlaceholderText('e.g. {"cam0": "ext"}')
        self._fp_edit.setToolTip(
            "Concept: JSON dict that pins specific parameters to fixed values.\n\n"
            "Default: blank (all parameters free)\n"
            "Range: valid JSON object\n"
            "Guidance: leave blank unless you need to freeze a reference camera."
        )
        opts_form.addRow("Fixed params (JSON):", self._fp_edit)

        # ── Action buttons ─────────────────────────────────────────────
        form_root.addWidget(make_separator())
        btn_row = QHBoxLayout()
        self._run_btn = make_blue_button("▶  Run Phase 4", self._run_phase4)
        self._run_btn.setToolTip("Run self-calibration.")
        btn_row.addWidget(self._run_btn)
        self._cancel_btn = make_orange_button("Cancel", self._cancel_phase4)
        self._cancel_btn.setToolTip(
            "Request cancellation; the active SciPy solve finishes its current step.")
        self._cancel_btn.setEnabled(False)
        btn_row.addWidget(self._cancel_btn)
        self._retry_btn = make_green_button("Retry", self._run_phase4)
        self._retry_btn.setEnabled(False)
        btn_row.addWidget(self._retry_btn)
        btn_row.addWidget(make_warning_button("Diagnostics ▼", self._open_diagnostics))
        btn_row.addWidget(make_green_button("Assess Calibration", self._open_assess_calibration))
        btn_row.addStretch()
        root.addLayout(btn_row)
        self._status_lbl = QLabel("Ready")
        set_text_role(self._status_lbl, "muted")
        root.addWidget(self._status_lbl)

        self._terminal = TerminalWidget(terminal_cb, parent=self)
        root.addWidget(self._terminal)

        self._refresh_phase3_sources()

    def _on_loss_changed(self, loss_name: str) -> None:
        # f_scale only affects soft_l1/huber/cauchy/arctan; grey it out for linear
        # (scipy's default) so it's clear the value is a no-op in that mode.
        self._f_scale_spin.setEnabled(loss_name != "linear")

    def _browse_floc(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select image folder")
        if path:
            self._floc_edit.setText(path)

    def _browse_phase3_camset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Phase 3 camset", "", "Camset files (*.camset);;All files (*)")
        if path:
            self._phase3_camset_edit.setText(path)

    def _sync_workspace_from_floc(self, text: str) -> None:
        floc = (text or "").strip()
        if not floc:
            return
        f_loc = Path(floc)
        if not f_loc.exists() or not f_loc.is_dir():
            return
        ws_path = f_loc / ".pycamset_workspace"
        if self._workspace_mgr.workspace_path != ws_path:
            self._workspace_mgr.set_workspace_path(ws_path, ensure=True)
            if self._diagnostics_tab is not None:
                self._diagnostics_tab.refresh()
        self._refresh_phase3_sources()

    def _load_phase3_run(self) -> Optional[dict]:
        runs = self._workspace_mgr.load_runs("phase3")
        if not runs:
            return None

        selected_run_id = self._phase3_run_combo.currentData()
        if selected_run_id:
            for run in runs:
                if run.get("run_id") == selected_run_id:
                    return run

        if self._preferred_phase3_run_id:
            for run in runs:
                if run.get("run_id") == self._preferred_phase3_run_id:
                    return run

        ws = self._workspace_mgr.workspace_path
        if ws is not None:
            handoff = ws / "handoff.json"
            if handoff.exists():
                try:
                    payload = json.loads(handoff.read_text(encoding="utf-8"))
                    if payload.get("phase") == "phase3" and payload.get("runs"):
                        wanted = payload["runs"][0].get("run_id")
                        for run in runs:
                            if run.get("run_id") == wanted:
                                return run
                except Exception:
                    pass
        return runs[-1]

    def _refresh_phase3_sources(self) -> None:
        runs = self._workspace_mgr.load_runs("phase3")
        keep = self._phase3_run_combo.currentData() if hasattr(self, "_phase3_run_combo") else None
        self._phase3_run_combo.blockSignals(True)
        self._phase3_run_combo.clear()
        self._phase3_run_combo.addItem("Auto (handoff else latest)", None)
        for run in runs:
            rid = run.get("run_id", "unknown")
            self._phase3_run_combo.addItem(str(rid), str(rid))

        if keep is not None:
            idx = self._phase3_run_combo.findData(keep)
            self._phase3_run_combo.setCurrentIndex(idx if idx >= 0 else 0)
        else:
            self._phase3_run_combo.setCurrentIndex(0)
        self._phase3_run_combo.blockSignals(False)
        self._update_source_label()

    def _on_phase3_source_changed(self) -> None:
        self._update_source_label()

    def _update_source_label(self) -> None:
        override = self._phase3_camset_edit.text().strip()
        if override:
            p = Path(override)
            self._source_lbl.setText(f"Source: override camset ({'exists' if path_exists(p) else 'missing'})")
            return

        run = self._load_phase3_run()
        if run is None:
            self._source_lbl.setText("Source: auto (no Phase 3 run found)")
            return

        rid = run.get("run_id", "unknown")
        ws = self._workspace_mgr.workspace_path
        resolved = resolve_artifact(run, "phase3", ws) if ws is not None else None
        p = str(resolved) if resolved is not None else (run.get("artifacts") or {}).get("optimised_camset")
        self._source_lbl.setText(f"Source: run {rid} -> {p or 'missing camset artifact'}")

    def _collect_params(self) -> Optional[dict]:
        """Read the form, or say which field is wrong and return None."""
        try:
            return self._read_params()
        except ParamError as exc:
            QMessageBox.critical(self, "Validation Error", str(exc))
            return None

    def _read_params(self) -> dict:
        """The form as a phase 4 parameter dict.

        :raises pyCamSet.workflow.ParamError: for a field that cannot be used as it stands
        """
        return {
            "f_loc": require_image_folder(self._floc_edit.text()),
            "threads": as_positive_int(
                self._threads_edit.text().strip() or "1", "Threads"),
            "fixed_params": as_json_object(
                self._fp_edit.text(), "Fixed params JSON"),
            "problem_options": {
                "verbosity": int(self._verbosity_spin.value()),
                "fixed_pose": as_int(
                    self._fixed_pose_edit.text(), "fixed_pose"),
                "ref_cam": as_int(self._ref_cam_edit.text(), "ref_cam"),
                "ref_pose": as_int(self._ref_pose_edit.text(), "ref_pose"),
                "outliers": as_outlier_mode(self._outliers_combo.currentText()),
                "max_nfev": int(self._max_nfev_spin.value()),
                "loss": self._loss_combo.currentText(),
                "f_scale": float(self._f_scale_spin.value()),
            },
        }

    def _resolve_phase3_camset(self, phase3_run: Optional[dict]) -> Optional[Path]:
        override = self._phase3_camset_edit.text().strip()
        if override:
            p = Path(override)
            return p if path_exists(p) else None

        if self._preferred_phase3_camset_path:
            p = Path(self._preferred_phase3_camset_path)
            if path_exists(p):
                return p

        if phase3_run is None:
            return None
        ws = self._workspace_mgr.workspace_path
        if ws is None:
            return None
        return resolve_artifact(phase3_run, "phase3", ws)

    def _run_phase4(self) -> None:
        params = self._collect_params()
        if params is None:
            return
        if not _PYCAMSET_OK:
            QMessageBox.critical(
                self, "Import error",
                "pyCamSet optimisation modules are unavailable.")
            return
        if self._worker is not None and self._worker.isRunning():
            if self._info_cb.isChecked():
                QMessageBox.information(
                    self, "Phase 4 running", "Phase 4 is already running.")
            return

        self._sync_workspace_from_floc(params["f_loc"])
        if self._workspace_mgr.workspace_path is None:
            QMessageBox.critical(
                self, "Workspace", "Could not initialize workspace.")
            return

        phase3_run = self._load_phase3_run()
        phase3_camset = self._resolve_phase3_camset(phase3_run)
        if phase3_camset is None:
            if self._info_cb.isChecked():
                QMessageBox.information(
                    self, "No Phase 3 source",
                    "Select a valid Phase 3 run/camset first.")
            return

        params["selected_cameras"] = list(
            ((phase3_run or {}).get("params") or {}).get("selected_cameras") or [])

        self._run_btn.setEnabled(False)
        self._retry_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._status_lbl.setText("Running…")
        self._terminal.clear_terminal()
        self._terminal.append_line("=== Phase 4: Self-Calibration ===")
        self._terminal.append_line(f"Image folder : {params['f_loc']}")
        self._terminal.append_line(
            f"Phase 3 run  : "
            f"{phase3_run.get('run_id', 'unknown') if phase3_run else 'override'}")
        self._terminal.append_line(f"Camset       : {phase3_camset}")
        self._terminal.append_line(
            f"selected cams: {params['selected_cameras'] or 'all'}")
        self._terminal.append_line("Starting…")

        workspace_mgr = self._workspace_mgr

        def work_fn(log: Callable[[str], None]) -> dict:
            return phase4_workflow.run(
                params, workspace_mgr, log,
                phase3_run=phase3_run, phase3_camset=phase3_camset)

        self._worker = PhaseWorker(work_fn, parent=self)
        self._worker.line_ready.connect(self._terminal.append_line)
        self._worker.finished.connect(self._on_run_finished)
        self._worker.error.connect(
            lambda msg: self._terminal.append_line(f"ERROR: {msg}"))
        self._worker.start()

    def _cancel_phase4(self) -> None:
        """Request a cooperative stop without claiming the solver was killed."""
        if self._worker is None or not self._worker.isRunning():
            return
        self._worker.requestInterruption()
        self._cancel_btn.setEnabled(False)
        self._status_lbl.setText(
            "Cancellation requested; waiting for the active solver step…")
        self._terminal.append_line(
            "Cancellation requested; the active optimisation is not interrupted mid-step.")

    def _on_run_finished(self, metadata: dict) -> None:
        status = str(metadata.get("status", "failed"))
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._retry_btn.setEnabled(status != "complete")
        if status == "complete":
            self._status_lbl.setText("Complete — quality gate passed")
        elif status == "incomplete":
            flags = ((metadata.get("diagnostics") or {}).get("quality_gate") or {}).get(
                "blocking_flags", [])
            detail = "; ".join(map(str, flags[:2]))
            suffix = f": {detail}" if detail else ""
            self._status_lbl.setText(
                f"Incomplete — quality gate blocked hand-off{suffix}")
        else:
            self._status_lbl.setText(
                f"Failed — {metadata.get('error', 'unknown error')}")
        if self._diagnostics_tab is not None:
            self._diagnostics_tab.refresh()

    def _open_diagnostics(self) -> None:
        if self._diagnostics_tab is not None:
            self._diagnostics_tab.refresh()
            show_tab(self._notebook, self._diagnostics_tab)

    def _open_assess_calibration(self) -> None:
        """Open the Phase 4 diagnostics tab and trigger Assess Calibration."""
        if self._diagnostics_tab is not None:
            self._diagnostics_tab.refresh()
            show_tab(self._notebook, self._diagnostics_tab)
            self._diagnostics_tab.visualise_from_primary()


class Phase4DiagnosticsTab(QWidget):
    def __init__(self, notebook: QTabWidget, info_cb: QCheckBox, workspace_mgr: WorkspaceManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._notebook = notebook
        self._info_cb = info_cb
        self._workspace_mgr = workspace_mgr
        self._all_runs: list[dict] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        top_btn_row = QHBoxLayout()
        top_btn_row.addWidget(make_warning_button("▲ Self-Calibration Settings", self._go_to_settings))
        top_btn_row.addStretch()
        root.addLayout(top_btn_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        left = QWidget()
        left.setMinimumWidth(190)
        left.setMaximumWidth(320)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)

        self._run_selector = RunSelectorWidget(runs=[], preselect=1)
        self._run_selector.selection_changed.connect(self._on_selection_changed)
        left_layout.addWidget(self._run_selector)

        refresh_btn = QPushButton("⟳  Refresh")
        refresh_btn.clicked.connect(self.refresh)
        left_layout.addWidget(refresh_btn)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.addWidget(make_section_label("Phase 4 Diagnostics"))
        right_layout.addWidget(make_separator())

        self._sub_tabs = QTabWidget()
        right_layout.addWidget(self._sub_tabs)
        splitter.addWidget(right)
        splitter.setSizes([240, 740])

        self._summary_widget = QWidget()
        self._summary_scroll = QScrollArea()
        self._summary_scroll.setWidgetResizable(True)
        self._summary_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._summary_inner = QWidget()
        self._summary_layout = QVBoxLayout(self._summary_inner)
        self._summary_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._summary_scroll.setWidget(self._summary_inner)
        summary_root = QVBoxLayout(self._summary_widget)
        summary_root.addWidget(self._summary_scroll)
        self._sub_tabs.addTab(self._summary_widget, "Summary (D4.1-D4.10)")

        self._visual_widget = QWidget()
        visual_layout = QVBoxLayout(self._visual_widget)
        visual_btn_row = QHBoxLayout()
        self._pyvista_cb = QCheckBox("PyVista")
        self._pyvista_cb.setChecked(True)
        self._pyvista_cb.setToolTip("Use PyVista backend (opens native window).")
        visual_btn_row.addWidget(self._pyvista_cb)
        self._open3d_cb = QCheckBox("Open3D")
        self._open3d_cb.setChecked(False)
        self._open3d_cb.setToolTip("Use Open3D backend (opens a separate interactive window).")
        visual_btn_row.addWidget(self._open3d_cb)
        # Enforce mutual exclusivity via QButtonGroup.
        self._backend_group = QButtonGroup(self)
        self._backend_group.setExclusive(True)
        self._backend_group.addButton(self._pyvista_cb)
        self._backend_group.addButton(self._open3d_cb)
        self._backend_group.buttonClicked.connect(self._on_backend_changed)
        self._visual_btn = QPushButton("Assess Calibration")
        self._visual_btn.clicked.connect(self._run_visualise_target)
        visual_btn_row.addWidget(self._visual_btn)
        self._save_png_btn = QPushButton("Save PNG")
        self._save_png_btn.setToolTip("Save the current visualisation as PNG.")
        self._save_png_btn.clicked.connect(self._save_visualisation_png)
        visual_btn_row.addWidget(self._save_png_btn)
        self._three_d_export_preset = QComboBox()
        self._three_d_export_preset.addItem("3D screen · 160 mm · 150 dpi", (160.0, 150))
        self._three_d_export_preset.addItem("3D single-column · 85 mm · 300 dpi", (85.0, 300))
        self._three_d_export_preset.addItem("3D double-column · 180 mm · 300 dpi", (180.0, 300))
        self._three_d_export_preset.setToolTip(
            "PNG pixel dimensions follow this generic width/DPI preset; no journal compliance is implied.")
        from pyCamSet.gui.preferences import bind_export_preset
        bind_export_preset(self._three_d_export_preset, "phase4:3d-export")
        visual_btn_row.addWidget(self._three_d_export_preset)
        export_3d_btn = QPushButton("Export 3D geometry…")
        export_3d_btn.setToolTip("PyVista: GLTF scene, OBJ geometry, or PLY target-frame point cloud.")
        export_3d_btn.clicked.connect(self._export_visualisation_3d)
        visual_btn_row.addWidget(export_3d_btn)
        self._assessment_export_preset = QComboBox()
        self._assessment_export_preset.addItem("Screen template · 160 mm · 150 dpi", (160.0, 150))
        self._assessment_export_preset.addItem("Single-column template · 85 mm · 300 dpi", (85.0, 300))
        self._assessment_export_preset.addItem("Double-column template · 180 mm · 300 dpi", (180.0, 300))
        self._assessment_export_preset.setToolTip("Generic templates; no named-journal compliance is implied.")
        bind_export_preset(self._assessment_export_preset, "phase4:assessment-export")
        visual_btn_row.addWidget(self._assessment_export_preset)
        save_2d_btn = QPushButton("Save 2D assessment exports…")
        save_2d_btn.setToolTip("Save the three child-process Matplotlib figures as PNG, SVG and PDF.")
        save_2d_btn.clicked.connect(self._save_assessment_2d_pngs)
        visual_btn_row.addWidget(save_2d_btn)
        visual_btn_row.addStretch()
        visual_layout.addLayout(visual_btn_row)
        style_row = QHBoxLayout()
        self._assessment_figure_themes = []
        for figure_label in ("Error distribution", "Camera coverage", "Accuracy / precision"):
            style_row.addWidget(QLabel(f"{figure_label} chrome:"))
            theme_combo = QComboBox()
            theme_combo.addItems(("Inherit", "Light", "Dark", "Sepia"))
            theme_combo.setToolTip("Cosmetic figure chrome only; quantitative colours are unchanged.")
            self._assessment_figure_themes.append(theme_combo)
            style_row.addWidget(theme_combo)
        style_row.addStretch()
        visual_layout.addLayout(style_row)
        self._three_d_style = ThreeDStyleControls(
            self._visual_widget, visual_id="assessment:phase4")
        visual_layout.addWidget(self._three_d_style)
        # Shows which run/phase the most recent Assess Calibration click actually
        # resolved to -- lets a user comparing PyVista vs. Open3D (or comparing this
        # tab against Phase 3's own Assess Calibration tab) immediately see whether
        # they are looking at two different camsets/runs on purpose, rather than
        # mistaking a run-selection mismatch for a rendering disagreement.
        self._current_run_label = QLabel("")
        set_text_role(self._current_run_label, "muted")
        visual_layout.addWidget(self._current_run_label)
        self._open3d_output = QLabel("")
        self._open3d_output.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._open3d_output.setMinimumHeight(400)
        self._open3d_output.setObjectName("viewportPlaceholder")
        self._open3d_output.setText("Select Open3D backend and click Assess Calibration to render here.")
        self._open3d_output.setWordWrap(True)
        self._open3d_output.setVisible(False)
        visual_layout.addWidget(self._open3d_output)
        visual_layout.addStretch()
        self._sub_tabs.addTab(self._visual_widget, "Assess Calibration")

        self.refresh()

    def _go_to_settings(self) -> None:
        for i in range(self._notebook.count()):
            if self._notebook.tabText(i) == TAB_PHASE4:
                self._notebook.setCurrentIndex(i)
                return

    @staticmethod
    def _clear_layout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                Phase4DiagnosticsTab._clear_layout(item.layout())

    def _combined_runs(self) -> list[dict]:
        return merge_phase3_phase4_runs(self._workspace_mgr.load_runs("phase3"), self._workspace_mgr.load_runs("phase4"))

    def refresh(self) -> None:
        self._all_runs = self._combined_runs()
        self._run_selector.refresh(self._all_runs)
        self._render_summary(self._run_selector.get_selected())

    def _on_selection_changed(self, _runs: list[dict]) -> None:
        self._run_selector.enforce_max_selection(4)
        self._render_summary(self._run_selector.get_selected())

    def _render_summary(self, runs: list[dict]) -> None:
        while self._summary_layout.count():
            item = self._summary_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

        if not runs:
            lbl = QLabel("Select one or more runs to inspect diagnostics.")
            set_text_role(lbl, "muted")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._summary_layout.addWidget(lbl)
            return

        for run in runs:
            phase = str(run.get("phase", "unknown"))
            hdr = QLabel(f"Run: {run.get('run_id', 'unknown')} ({phase})")
            set_text_role(hdr, "subheading")
            self._summary_layout.addWidget(hdr)
            form = QFormLayout()
            form.setContentsMargins(16, 0, 0, 0)
            if phase != "phase4":
                form.addRow("Note:", QLabel("This run is from Phase 3; D4 metrics are not available."))
            else:
                d = run.get("diagnostics", {})
                gate = d.get("quality_gate") or {}
                status = str(run.get("status", "unknown"))
                # The mark and the word carry the outcome; colour only reinforces it.
                complete = status == "complete"
                status_label = QLabel(("✓ " if complete else "✗ ") + status)
                set_text_role(status_label, "success" if complete else "danger")
                form.addRow("Disposition:", status_label)
                flags = gate.get("blocking_flags", [])
                form.addRow(
                    "Quality gate:",
                    QLabel("passed" if not flags else "; ".join(map(str, flags))))
                form.addRow(
                    "Observation coverage:",
                    QLabel(f"cameras={gate.get('observed_cameras', '—')} | "
                           f"images={len(gate.get('observed_images', []))} observed, "
                           f"{len(gate.get('missing_images', []))} explicitly missing"))
                form.addRow(
                    "Gauge accounting:",
                    QLabel(str(gate.get("gauge", "—"))))
                form.addRow("D4.1 free target points:", QLabel(str(d.get("D4.1_n_free_target_points", "—"))))
                form.addRow("D4.2 gauge-fixed points:", QLabel(str(d.get("D4.2_gauge_fixed_points", "—"))))
                form.addRow("D4.3 initial euclid (px):", QLabel(f"{float(d.get('D4.3_initial_euclid_px', float('nan'))):.5f}"))
                form.addRow("D4.3 final euclid (px):", QLabel(f"{float(d.get('D4.3_final_euclid_px', float('nan'))):.5f}"))
                form.addRow("D4.4 vs Phase 3 delta (px):", QLabel(f"{float(d.get('D4.4_vs_phase3_delta_px', float('nan'))):.5f}"))
                form.addRow("D4.5 gauge scale factor:", QLabel(f"{float(d.get('D4.5_gauge_scale_factor', float('nan'))):.6f}"))
                form.addRow("D4.7 mean target displacement (mm):", QLabel(f"{float(d.get('D4.7_mean_target_displacement_mm', float('nan'))):.5f}"))

                # D4.12 — per-camera mean reprojection error
                d412 = d.get("D4.12_per_camera_mean_reprojection", {})
                if d412 and isinstance(d412, dict):
                    valid = {k: v for k, v in d412.items() if isinstance(v, (int, float)) and np.isfinite(v)}
                    if valid:
                        worst_cam = max(valid, key=valid.get)
                        worst_val = valid[worst_cam]
                        best_cam = min(valid, key=valid.get)
                        best_val = valid[best_cam]
                        worst_lbl = QLabel(f"worst={worst_cam} ({worst_val:.2f} px), best={best_cam} ({best_val:.2f} px)")
                        if worst_val > 3 * best_val and best_val > 0:
                            worst_lbl.setText("✗ " + worst_lbl.text() + " — worst is over 3× the best")
                            set_text_role(worst_lbl, "danger")
                        form.addRow("D4.12 per-camera reprojection:", worst_lbl)
                        for cam_name in sorted(d412.keys()):
                            val = d412[cam_name]
                            if isinstance(val, (int, float)) and np.isfinite(val):
                                form.addRow(f"  {cam_name}:", QLabel(f"{val:.2f} px"))
                            else:
                                form.addRow(f"  {cam_name}:", QLabel("—"))
                    else:
                        form.addRow("D4.12 per-camera reprojection:", QLabel("no valid cameras"))
                else:
                    form.addRow("D4.12 per-camera reprojection:", QLabel("—"))
                per_image = d.get("D4.13_per_image_mean_reprojection", {})
                form.addRow(
                    "D4.13 per-image reprojection:",
                    QLabel(f"{len(per_image)} images" if isinstance(per_image, dict) else "—"))
            if run.get("error"):
                form.addRow("Error:", QLabel(str(run["error"])))
            self._summary_layout.addLayout(form)
            self._summary_layout.addWidget(make_separator())
            render_predecessor_chain_section(self._summary_layout, self._workspace_mgr, run)
        self._summary_layout.addStretch()

    def _on_backend_changed(self, btn) -> None:
        """Handle backend selector toggle — update Open3D output visibility."""
        if self._open3d_cb.isChecked():
            self._three_d_style.setEnabled(False)
            self._open3d_output.setText("Click Assess Calibration to open an interactive Open3D window.")
            self._open3d_output.setVisible(True)
        else:
            self._three_d_style.setEnabled(True)
            self._open3d_output.setVisible(False)

    def _on_pyvista_toggled(self, state: int) -> None:
        # Kept for backwards compatibility; QButtonGroup handles exclusivity.
        self._open3d_output.setVisible(False)

    def _on_open3d_toggled(self, state: int) -> None:
        # Kept for backwards compatibility; QButtonGroup handles exclusivity.
        self._open3d_output.setVisible(bool(state))

    def _run_visualise_target(self) -> None:
        selected = self._run_selector.get_selected()
        chosen = select_latest_visualisation_run(selected, self._all_runs)
        if chosen is None:
            if self._info_cb.isChecked():
                QMessageBox.information(self, "Select run", "Select at least one run first.")
            return
        self._current_run_label.setText(
            f"Currently showing: {chosen.get('phase', 'unknown')} | {chosen.get('run_id', 'unknown')}"
        )
        if self._open3d_cb.isChecked():
            # On Linux, EGL offscreen rendering is typically available, so we
            # can embed the Open3D view in the GUI. On Windows, EGL support is
            # missing so we fall back to a separate native Open3D window.
            self._three_d_style.setEnabled(False)
            _open3d_widget = self._open3d_output if os.name != "nt" else None
            ok, msg = launch_visualise_calibration_open3d_for_run(chosen, output_widget=_open3d_widget)
        else:
            self._three_d_style.setEnabled(True)
            app = QApplication.instance()
            active_theme = app.property("pycamsetTheme") if app else "Light"
            figure_themes = tuple(
                theme.currentText() if theme.currentText() != "Inherit" else active_theme
                for theme in self._assessment_figure_themes
            )
            ok, msg = launch_visualise_calibration_for_run(
                chosen, theme_name=active_theme, figure_themes=figure_themes,
                three_d_arguments=self._three_d_style.viewer_arguments())
        if not ok:
            QMessageBox.warning(self, "Assess Calibration", msg)

    def _save_visualisation_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Visualisation as PNG", "calibration_assessment.png", "PNG Files (*.png)"
        )
        if not path:
            return
        if self._open3d_cb.isChecked():
            # Open3D opens a separate native window — no embedded pixmap to save.
            QMessageBox.information(
                self, "Save PNG",
                "PNG export is not available with the Open3D backend.\n"
                "Switch to the PyVista backend to save a PNG export.",
            )
        else:
            # PyVista: offscreen render to PNG.
            selected = self._run_selector.get_selected()
            chosen = select_latest_visualisation_run(selected, self._all_runs)
            if chosen is None:
                QMessageBox.warning(self, "Save PNG", "Select at least one run first.")
                return
            self._current_run_label.setText(
                f"Currently showing: {chosen.get('phase', 'unknown')} | {chosen.get('run_id', 'unknown')}"
            )
            from pathlib import Path
            width_mm, dpi = self._three_d_export_preset.currentData()
            app = QApplication.instance()
            active_theme = app.property("pycamsetTheme") if app else "Light"
            ok, msg = launch_save_pyvista_png_for_run(
                chosen, Path(path), width_mm=width_mm, dpi=dpi, theme_name=active_theme,
                three_d_arguments=self._three_d_style.viewer_arguments())
            if ok:
                QMessageBox.information(self, "Save PNG", msg)
            else:
                QMessageBox.warning(self, "Save PNG", f"Could not save PNG:\n{msg}")

    def _export_visualisation_3d(self) -> None:
        """Offer an actual PyVista geometry export, independent of PNG capture."""
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export 3D Geometry", "calibration_scene.gltf",
            "glTF scene (*.gltf);;Wavefront geometry (*.obj);;PLY point cloud (*.ply)")
        if not path:
            return
        if not Path(path).suffix:
            extension = {"Wavefront geometry (*.obj)": ".obj",
                         "PLY point cloud (*.ply)": ".ply"}.get(selected_filter, ".gltf")
            path = f"{path}{extension}"
        selected = self._run_selector.get_selected()
        chosen = select_latest_visualisation_run(selected, self._all_runs)
        if chosen is None:
            QMessageBox.warning(self, "3D Export", "Select at least one run first.")
            return
        self._current_run_label.setText(
            f"Currently exporting: {chosen.get('phase', 'unknown')} | {chosen.get('run_id', 'unknown')}")
        ok, message = launch_export_3d_for_run(chosen, Path(path))
        if ok:
            QMessageBox.information(self, "3D Export", message)
        else:
            QMessageBox.warning(self, "3D Export", message)

    def _save_assessment_2d_pngs(self) -> None:
        """Save the child process's numerical 2D assessment figures as PNGs."""
        directory = QFileDialog.getExistingDirectory(self, "Save 2D Assessment Figures")
        if not directory:
            return
        selected = self._run_selector.get_selected()
        chosen = select_latest_visualisation_run(selected, getattr(self, "_all_runs", []))
        if chosen is None:
            QMessageBox.warning(self, "Assess Calibration", "Select at least one run first.")
            return
        app = QApplication.instance()
        theme_name = app.property("pycamsetTheme") if app else "Light"
        width_mm, dpi = self._assessment_export_preset.currentData()
        figure_themes = tuple(
            theme.currentText() if theme.currentText() != "Inherit" else theme_name
            for theme in self._assessment_figure_themes
        )
        ok, message = launch_save_assessment_pngs_for_run(
            chosen, Path(directory), theme_name, width_mm, dpi, figure_themes)
        if ok:
            QMessageBox.information(self, "Assess Calibration", message or "Saved 2D assessment PNGs.")
        else:
            QMessageBox.warning(self, "Assess Calibration", f"Could not save 2D assessment PNGs:\n{message}")

    def visualise_from_primary(self) -> None:
        self._sub_tabs.setCurrentWidget(self._visual_widget)
        self._run_visualise_target()
