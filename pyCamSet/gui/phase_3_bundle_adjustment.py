"""
Phase 3 - Template bundle adjustment GUI.

The form, the lockbox controls and the figures; the solve itself is
:mod:`pyCamSet.workflow.phase3`.  The diagnostics tab additionally re-solves
from detections it has pruned by image, which is why the backend names are
still imported here.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pyCamSet.workflow import phase3 as phase3_workflow
from pyCamSet.gui.three_d_style import ThreeDStyleControls
from pyCamSet.workflow.params import (
    ParamError,
    as_int,
    as_json_object,
    as_outlier_mode,
    as_positive_float,
    as_positive_int,
    require_image_folder,
    require_detector_available,
    require_target_match,
)
from pyCamSet.gui.shared_functions import (
    CollapsibleSection,
    DETECTOR_INHERIT,
    IMAGE_FOLDER_SCHEMATIC,
    MatplotlibFigureCard,
    PhaseWorker,
    RunSelectorWidget,
    TAB_PHASE3,
    TAB_PHASE4,
    TerminalWidget,
    WorkspaceManager,
    TargetSettingsForm,
    gate_continue_button,
    make_blue_button,
    make_continue_button,
    make_green_button,
    make_warning_button,
    make_scrollable_tab,
    make_section_label,
    make_separator,
    render_predecessor_chain_section,
    show_tab,
)
from pyCamSet.workflow.detections import DetectionFilter
from pyCamSet.calibration_targets.core.target_registry import build_target
from pyCamSet.workflow.targets import (
    TARGET_KEY,
    target_params_of_run,
)
from pyCamSet.workflow.workspace import (
    as_io_path,
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
from pyCamSet.gui.phase_3_lockbox_editor import Phase3LockboxEditor

_LOG = logging.getLogger(__name__)

# The lockbox controls load camsets to check their camera names against the
# active one.  The flag is the phase runner's: the tab and the phase fail
# together, and nothing here reaches for the optimisation backend any more.
try:
    from pyCamSet.utils.saving import load_CameraSet
except ImportError:
    load_CameraSet = None

_PYCAMSET_OK = phase3_workflow.BACKEND_OK



class Phase3Tab(QWidget):
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
        self._diagnostics_tab: Optional["Phase3DiagnosticsTab"] = None
        self._worker: Optional[PhaseWorker] = None
        self._preferred_phase2_run_id: Optional[str] = None
        self._adopted_target_run_id: Optional[str] = None
        self._preferred_phase2_camset_path: Optional[str] = None
        self._edited_lockbox_camset_path: Optional[str] = None
        self._edited_lockbox_metadata_path: Optional[str] = None
        self._lockbox_edit_summary: str = "Using original source"
        self._build_ui(terminal_cb)

    def set_diagnostics_tab(self, tab: "Phase3DiagnosticsTab") -> None:
        self._diagnostics_tab = tab

    def _build_ui(self, terminal_cb: QCheckBox) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        top_row = QHBoxLayout()
        root.addLayout(top_row)

        form_widget = QWidget()
        form_root = QVBoxLayout(form_widget)
        form_root.setContentsMargins(0, 0, 0, 0)
        form_root.setSpacing(4)
        form_scroll = QScrollArea()
        form_scroll.setWidgetResizable(True)
        form_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        form_scroll.setWidget(form_widget)
        top_row.addWidget(form_scroll, stretch=1)

        side = QWidget()
        side.setFixedWidth(240)
        side_layout = QVBoxLayout(side)
        side_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        top_row.addWidget(side)

        # ── Paths (collapsible) ────────────────────────────────────────
        paths_sect = CollapsibleSection("Paths", expanded=False)
        form_root.addWidget(paths_sect)

        floc_row = QHBoxLayout()
        self._floc_edit = QLineEdit()
        self._floc_edit.setPlaceholderText("Root folder with per-camera sub-folders")
        self._floc_edit.setToolTip(IMAGE_FOLDER_SCHEMATIC)
        self._floc_edit.textChanged.connect(self._sync_workspace_from_floc)
        floc_btn = QPushButton("Browse…")
        floc_btn.setFixedWidth(70)
        floc_btn.clicked.connect(self._browse_floc)
        floc_row.addWidget(self._floc_edit)
        floc_row.addWidget(floc_btn)
        paths_sect.addRow("Image folder (f_loc):", floc_row)

        src_row = QHBoxLayout()
        self._phase2_run_combo = QComboBox()
        self._phase2_run_combo.setToolTip(
            "Concept: Phase 2 initial-camera run used as input to Phase 3 bundle adjustment.\n"
            "Default: Auto (handoff selection from Phase 2, else latest run)."
        )
        self._phase2_run_combo.currentIndexChanged.connect(self._on_phase2_source_changed)
        src_refresh_btn = QPushButton("Refresh")
        src_refresh_btn.setFixedWidth(70)
        src_refresh_btn.setToolTip("Reload available Phase 2 runs from workspace.")
        src_refresh_btn.clicked.connect(self._refresh_phase2_sources)
        src_row.addWidget(self._phase2_run_combo)
        src_row.addWidget(src_refresh_btn)
        paths_sect.addRow("Phase 2 run:", src_row)

        self._src_lbl = QLabel("Inputs: auto Phase 2 + linked Phase 1")
        self._src_lbl.setWordWrap(True)
        self._src_lbl.setStyleSheet("color: #666;")
        paths_sect.addRow("Input runs:", self._src_lbl)

        # ── Calibration Target (collapsible) ───────────────────────────
        form_root.addWidget(make_separator())
        target_sect = CollapsibleSection("Calibration Target", expanded=False)
        form_root.addWidget(target_sect)

        # The detector is the linked Phase 1 run's: its detections are
        # what this phase reads.
        self._target_form = TargetSettingsForm(detector_mode=DETECTOR_INHERIT)
        target_sect.addRow(self._target_form)

        # ── Bundle Adjustment Options ──────────────────────────────────
        form_root.addWidget(make_separator())
        form_root.addWidget(make_section_label("Bundle Adjustment Options"))

        opts_form = QFormLayout()
        opts_form.setContentsMargins(0, 0, 0, 0)
        opts_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form_root.addLayout(opts_form)

        self._threads_edit = QLineEdit("1")
        self._threads_edit.setFixedWidth(110)
        self._threads_edit.setToolTip(
            "Concept: number of Jacobian evaluation threads used by scipy.\n\n"
            "Default: 1\n"
            "Range: positive integer\n"
            "Guidance: values > 1 can speed up large problems but increase\n"
            "memory usage.  Start with 1 for reproducibility."
        )
        opts_form.addRow("Threads:", self._threads_edit)

        self._max_nfev_spin = QSpinBox()
        self._max_nfev_spin.setRange(5, 5000)
        self._max_nfev_spin.setValue(300)
        self._max_nfev_spin.setFixedWidth(110)
        self._max_nfev_spin.setToolTip(
            "Concept: maximum number of cost-function evaluations the\n"
            "Levenberg-Marquardt solver is allowed to perform.\n\n"
            "Default: 300\n"
            "Range: 5–5000\n"
            "Guidance: if the solver reports it did not converge, try\n"
            "increasing to 1000.  Values > 2000 rarely improve results\n"
            "and greatly increase runtime."
        )
        opts_form.addRow("max_nfev:", self._max_nfev_spin)

        self._verbosity_spin = QSpinBox()
        self._verbosity_spin.setRange(0, 2)
        self._verbosity_spin.setValue(2)
        self._verbosity_spin.setFixedWidth(110)
        self._verbosity_spin.setToolTip(
            "Concept: verbosity level passed to the scipy least_squares solver.\n\n"
            "Default: 2\n"
            "Range: 0 (silent), 1 (summary only), 2 (per-iteration output)\n"
            "Guidance: keep at 2 to monitor convergence; set to 0 for\n"
            "automated / batch runs."
        )
        opts_form.addRow("verbosity:", self._verbosity_spin)

        self._outliers_combo = QComboBox()
        self._outliers_combo.setObjectName("outliers_combo")
        self._outliers_combo.addItems(["y", "n", "ask"])
        self._outliers_combo.setCurrentText("n")
        self._outliers_combo.setToolTip(
            "Concept: whether to reject outlier poses before bundle adjustment.\n"
            "Outlier poses are those whose initial reprojection error exceeds\n"
            "a threshold (median + 20 px by default).\n\n"
            "Default: n (disabled)\n"
            "Choices: n (disabled), y (enabled), ask (auto-resolved to y in GUI)\n"
            "Guidance: enable if you see a few images with very high initial\n"
            "reprojection error (> median + 20 px in D3.4).  Disabling keeps\n"
            "all poses, which is safer when data is sparse."
        )
        opts_form.addRow("outliers:", self._outliers_combo)

        self._fixed_pose_edit = QLineEdit("0")
        self._fixed_pose_edit.setFixedWidth(110)
        self._fixed_pose_edit.setToolTip(
            "Concept: the pose (image) index whose extrinsic is held fixed\n"
            "as the coordinate-system anchor during bundle adjustment.\n\n"
            "Default: 0\n"
            "Range: 0 to (num_poses − 1)\n"
            "Guidance: leave at 0 unless you have a specific reference image\n"
            "that defines a known world frame."
        )
        opts_form.addRow("fixed_pose:", self._fixed_pose_edit)

        self._ref_cam_edit = QLineEdit("0")
        self._ref_cam_edit.setFixedWidth(110)
        self._ref_cam_edit.setToolTip(
            "Concept: the camera index used as the metric reference.\n\n"
            "Default: 0\n"
            "Range: 0 to (num_cameras − 1)\n"
            "Guidance: leave at 0 unless camera 0 is unusable."
        )
        opts_form.addRow("ref_cam:", self._ref_cam_edit)

        self._ref_pose_edit = QLineEdit("0")
        self._ref_pose_edit.setFixedWidth(110)
        self._ref_pose_edit.setToolTip(
            "Concept: the pose (image) index used as the metric reference\n"
            "for the gauge constraint.\n\n"
            "Default: 0\n"
            "Range: 0 to (num_poses − 1)\n"
            "Guidance: leave at 0 for most setups."
        )
        opts_form.addRow("ref_pose:", self._ref_pose_edit)

        self._fp_edit = QLineEdit()
        self._fp_edit.setPlaceholderText('e.g. {"cam0": "ext"}')
        self._fp_edit.setToolTip(
            "Concept: JSON dict that pins specific camera parameters to\n"
            "fixed values, preventing them from being optimised.\n\n"
            "Default: blank (all parameters free)\n"
            "Range: valid JSON object, e.g. {\"cam0\": \"ext\"}\n"
            "Guidance: use to hold one camera's extrinsics fixed when you\n"
            "have a known reference.  Leave blank otherwise."
        )
        opts_form.addRow("Fixed params (JSON):", self._fp_edit)

        # ── Camera Lockbox Priors ──────────────────────────────────────
        form_root.addWidget(make_separator())
        lockbox_sect = CollapsibleSection("Camera Lockbox Prior (optional)", expanded=False)
        form_root.addWidget(lockbox_sect)

        self._lockbox_enabled_cb = QCheckBox("Enable Phase 3 extrinsic lockbox")
        self._lockbox_enabled_cb.setChecked(False)
        self._lockbox_enabled_cb.setToolTip(
            "Constrain free camera extrinsics to a bounded Gaussian prior from a previous camset.\n"
            "This applies only to Phase 3 Rodrigues+translation extrinsic parameters."
        )
        self._lockbox_enabled_cb.toggled.connect(self._set_lockbox_controls_enabled)
        lockbox_sect.addRow("Enable:", self._lockbox_enabled_cb)

        source_row = QHBoxLayout()
        self._lockbox_source_edit = QLineEdit()
        self._lockbox_source_edit.setPlaceholderText("Original source camset (.camset)")
        self._lockbox_source_edit.setToolTip(
            "Original source camset. This input is immutable; edited lockbox copies are saved separately."
        )
        self._lockbox_source_edit.textChanged.connect(self._reset_edited_lockbox_copy)
        source_btn = QPushButton("Browse…")
        source_btn.setFixedWidth(70)
        source_btn.clicked.connect(self._browse_lockbox_source)
        source_row.addWidget(self._lockbox_source_edit)
        source_row.addWidget(source_btn)
        self._lockbox_source_btn = source_btn
        lockbox_sect.addRow("Source camset:", source_row)

        editor_row = QHBoxLayout()
        self._lockbox_view_source_btn = QPushButton("View Source")
        self._lockbox_view_source_btn.setToolTip("Inspect the immutable original source camset without editing it.")
        self._lockbox_view_source_btn.clicked.connect(self._view_lockbox_source)
        self._lockbox_edit_btn = QPushButton("Edit Lockbox Prior…")
        self._lockbox_edit_btn.setToolTip("Create an edited lockbox copy; the original source camset is never overwritten.")
        self._lockbox_edit_btn.clicked.connect(self._open_lockbox_editor)
        self._lockbox_reset_btn = QPushButton("Reset Edited Copy")
        self._lockbox_reset_btn.setToolTip("Discard the saved edited-copy selection and use the original source camset.")
        self._lockbox_reset_btn.clicked.connect(self._reset_edited_lockbox_copy)
        editor_row.addWidget(self._lockbox_view_source_btn)
        editor_row.addWidget(self._lockbox_edit_btn)
        editor_row.addWidget(self._lockbox_reset_btn)
        lockbox_sect.addRow("Editor:", editor_row)

        self._lockbox_status_lbl = QLabel("Using original source")
        self._lockbox_status_lbl.setWordWrap(True)
        self._lockbox_status_lbl.setStyleSheet("color: #666;")
        lockbox_sect.addRow("Status:", self._lockbox_status_lbl)

        lockbox_banner = QLabel("Editing lockbox prior centres only. Original source camset will not be modified.")
        lockbox_banner.setWordWrap(True)
        lockbox_banner.setStyleSheet("font-weight: bold; color: #9a5b00;")
        lockbox_sect.addRow("Guardrail:", lockbox_banner)

        self._lockbox_warm_start_cb = QCheckBox("Warm-start constrained extrinsics from source camset")
        self._lockbox_warm_start_cb.setChecked(True)
        self._lockbox_warm_start_cb.setToolTip(
            "When enabled, Phase 3 starts bounded extrinsics at the prior centre.\n"
            "When disabled, the source camset still supplies bounds and soft priors."
        )
        lockbox_sect.addRow("Warm start:", self._lockbox_warm_start_cb)

        self._lockbox_rotation_half_spin = QDoubleSpinBox()
        self._lockbox_rotation_half_spin.setRange(1e-9, 10.0)
        self._lockbox_rotation_half_spin.setDecimals(6)
        self._lockbox_rotation_half_spin.setValue(0.1)
        self._lockbox_rotation_half_spin.setSingleStep(0.01)
        self._lockbox_rotation_half_spin.setToolTip("Hard Rodrigues half-width in radians.")
        lockbox_sect.addRow("Rotation half-width (rad):", self._lockbox_rotation_half_spin)

        self._lockbox_translation_half_spin = QDoubleSpinBox()
        self._lockbox_translation_half_spin.setRange(1e-9, 1e6)
        self._lockbox_translation_half_spin.setDecimals(6)
        self._lockbox_translation_half_spin.setValue(0.1)
        self._lockbox_translation_half_spin.setSingleStep(0.01)
        self._lockbox_translation_half_spin.setToolTip("Hard translation half-width in camera-set length units.")
        lockbox_sect.addRow("Translation half-width:", self._lockbox_translation_half_spin)

        self._lockbox_rotation_sigma_spin = QDoubleSpinBox()
        self._lockbox_rotation_sigma_spin.setRange(1e-9, 10.0)
        self._lockbox_rotation_sigma_spin.setDecimals(6)
        self._lockbox_rotation_sigma_spin.setValue(0.05)
        self._lockbox_rotation_sigma_spin.setSingleStep(0.005)
        self._lockbox_rotation_sigma_spin.setToolTip("Gaussian prior sigma for Rodrigues parameters.")
        lockbox_sect.addRow("Rotation sigma (rad):", self._lockbox_rotation_sigma_spin)

        self._lockbox_translation_sigma_spin = QDoubleSpinBox()
        self._lockbox_translation_sigma_spin.setRange(1e-9, 1e6)
        self._lockbox_translation_sigma_spin.setDecimals(6)
        self._lockbox_translation_sigma_spin.setValue(0.01)
        self._lockbox_translation_sigma_spin.setSingleStep(0.005)
        self._lockbox_translation_sigma_spin.setToolTip("Gaussian prior sigma for translation parameters.")
        lockbox_sect.addRow("Translation sigma:", self._lockbox_translation_sigma_spin)

        self._lockbox_center_sigma_spin = QDoubleSpinBox()
        self._lockbox_center_sigma_spin.setRange(0.0, 1e6)
        self._lockbox_center_sigma_spin.setDecimals(6)
        self._lockbox_center_sigma_spin.setValue(0.0)
        self._lockbox_center_sigma_spin.setSingleStep(0.005)
        self._lockbox_center_sigma_spin.setToolTip(
            "Gaussian sigma for world-space camera center C = −R^T @ t (camset length units).\n"
            "0 = disabled. When enabled, adds a soft prior directly on the 3D camera position,\n"
            "preventing large world-space drifts caused by Rodrigues near-π amplification.\n"
            "Recommended: ~0.02–0.05 (roughly the max tolerable camera center shift)."
        )
        lockbox_sect.addRow("Center position sigma:", self._lockbox_center_sigma_spin)

        self._lockbox_controls = [
            self._lockbox_source_edit,
            self._lockbox_source_btn,
            self._lockbox_view_source_btn,
            self._lockbox_edit_btn,
            self._lockbox_reset_btn,
            self._lockbox_warm_start_cb,
            self._lockbox_rotation_half_spin,
            self._lockbox_translation_half_spin,
            self._lockbox_rotation_sigma_spin,
            self._lockbox_translation_sigma_spin,
            self._lockbox_center_sigma_spin,
        ]
        self._set_lockbox_controls_enabled(False)

        # ── Action buttons ─────────────────────────────────────────────

        btn_row = QHBoxLayout()
        run_btn = make_blue_button("▶  Run Phase 3", self._run_phase3)
        run_btn.setToolTip("Run template bundle adjustment.")
        btn_row.addWidget(run_btn)
        btn_row.addWidget(make_warning_button("Diagnostics ▼", self._open_diagnostics))
        self._continue_btn = make_continue_button(
            self._continue_to_phase4, text="Phase 4 - Self-Calibration")
        btn_row.addWidget(self._continue_btn)
        btn_row.addWidget(make_green_button("Assess Calibration", self._visualise_target_from_primary))
        btn_row.addStretch()
        root.addLayout(btn_row)

        side_layout.addStretch()

        self._terminal = TerminalWidget(terminal_cb, parent=self)
        root.addWidget(self._terminal)

        self._refresh_phase2_sources()

    def _browse_floc(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select image folder")
        if path:
            self._floc_edit.setText(path)

    def _browse_lockbox_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select original source camset",
            "",
            "Camera Sets (*.camset *.json);;All Files (*)",
        )
        if path:
            self._lockbox_source_edit.setText(path)

    def _set_lockbox_controls_enabled(self, enabled: bool) -> None:
        for widget in getattr(self, "_lockbox_controls", []):
            widget.setEnabled(bool(enabled))

    def _reset_edited_lockbox_copy(self, *_args) -> None:
        self._edited_lockbox_camset_path = None
        self._edited_lockbox_metadata_path = None
        self._lockbox_edit_summary = "Using original source"
        if hasattr(self, "_lockbox_status_lbl"):
            self._lockbox_status_lbl.setText(self._lockbox_edit_summary)

    def _view_lockbox_source(self) -> None:
        source_path = self._lockbox_source_edit.text().strip()
        if not source_path:
            QMessageBox.information(self, "Original source camset", "Choose an original source camset first.")
            return
        if not path_exists(source_path):
            QMessageBox.critical(self, "Original source camset", f"Source camset does not exist: {source_path}")
            return
        try:
            cams = load_CameraSet(as_io_path(source_path))
            names = cams.get_names()
        except Exception as exc:
            QMessageBox.critical(self, "Original source camset", f"Could not load source camset:\n{exc}")
            return
        QMessageBox.information(
            self,
            "Original source camset",
            "Immutable original source camset loaded successfully.\n"
            f"Path: {source_path}\n"
            f"Cameras ({len(names)}): {', '.join(map(str, names))}",
        )

    def _open_lockbox_editor(self) -> None:
        source_path = self._lockbox_source_edit.text().strip()
        if not source_path:
            QMessageBox.information(self, "Lockbox editor", "Choose an original source camset first.")
            return
        if not path_exists(source_path):
            QMessageBox.critical(self, "Lockbox editor", f"Original source camset does not exist: {source_path}")
            return
        if self._workspace_mgr.workspace_path is None:
            self._sync_workspace_from_floc(self._floc_edit.text().strip())
        if self._workspace_mgr.workspace_path is None:
            QMessageBox.critical(self, "Lockbox editor", "Could not initialise a workspace-local lockbox_priors folder.")
            return
        active_names: list[str] = []
        phase2_run = self._load_phase2_run()
        if phase2_run is not None:
            camset_path = resolve_artifact(phase2_run, "phase2", self._workspace_mgr.workspace_path)
            if camset_path is not None and path_exists(str(camset_path)):
                try:
                    active_names = list(load_CameraSet(as_io_path(str(camset_path))).get_names())
                except Exception:
                    active_names = []
        try:
            target = build_target(self._target_form.spec())
        except Exception:
            target = None
        fixed_params = None
        if self._fp_edit.text().strip():
            try:
                fixed_params = json.loads(self._fp_edit.text().strip())
            except json.JSONDecodeError:
                fixed_params = None
        dialog = Phase3LockboxEditor(
            source_camset_path=source_path,
            workspace_path=self._workspace_mgr.workspace_path,
            target=target,
            active_camera_names=active_names,
            fixed_params=fixed_params,
            lockbox_params={
                "enabled": bool(self._lockbox_enabled_cb.isChecked()),
                "rotation_half_width": float(self._lockbox_rotation_half_spin.value()),
                "translation_half_width": float(self._lockbox_translation_half_spin.value()),
                "rotation_sigma": float(self._lockbox_rotation_sigma_spin.value()),
                "translation_sigma": float(self._lockbox_translation_sigma_spin.value()),
            },
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.result is None:
            return
        self._edited_lockbox_camset_path = dialog.result.edited_source_camset
        self._edited_lockbox_metadata_path = dialog.result.metadata_path
        self._lockbox_edit_summary = (
            "Using edited lockbox copy: "
            f"{dialog.result.edited_count} cameras edited, max shift {dialog.result.max_shift:.6g}"
        )
        self._lockbox_status_lbl.setText(self._lockbox_edit_summary)
        self._lockbox_enabled_cb.setChecked(True)

    def set_lockbox_source_camset_path(self, path: str) -> None:
        self._lockbox_source_edit.setText(path)
        if path:
            self._lockbox_enabled_cb.setChecked(True)

    def set_lockbox_source_path(self, path: str) -> None:
        self.set_lockbox_source_camset_path(path)

    def set_image_folder(self, path: str) -> None:
        self._floc_edit.setText(path)

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
        self._refresh_phase2_sources()

    def _collect_params(self) -> Optional[dict]:
        """Read the form, or say which field is wrong and return None."""
        try:
            return self._read_params()
        except ParamError as exc:
            QMessageBox.critical(self, "Validation Error", str(exc))
            return None

    def _read_params(self) -> dict:
        """The form as a phase 3 parameter dict.

        :raises pyCamSet.workflow.ParamError: for a field that cannot be used as it stands
        """
        return {
            "f_loc": require_image_folder(self._floc_edit.text()),
            "threads": as_positive_int(
                self._threads_edit.text().strip() or "1", "Threads"),
            "fixed_params": as_json_object(
                self._fp_edit.text(), "Fixed params JSON"),
            "target": self._target_form.spec(),
            "lockbox": self._read_lockbox_params(),
            "problem_options": {
                "verbosity": int(self._verbosity_spin.value()),
                "fixed_pose": as_int(
                    self._fixed_pose_edit.text(), "fixed_pose"),
                "ref_cam": as_int(self._ref_cam_edit.text(), "ref_cam"),
                "ref_pose": as_int(self._ref_pose_edit.text(), "ref_pose"),
                "outliers": as_outlier_mode(self._outliers_combo.currentText()),
                "max_nfev": int(self._max_nfev_spin.value()),
            },
        }

    def _read_lockbox_params(self) -> dict:
        """The lockbox block of the parameters, checked only when it is on.

        The edited copy the lockbox editor writes takes precedence over the
        camset it was derived from, and both are recorded: the run should say
        what it actually solved against and where that came from.

        :raises pyCamSet.workflow.ParamError: when the lockbox is on but its source is not there
        """
        enabled = self._lockbox_enabled_cb.isChecked()
        original = self._lockbox_source_edit.text().strip()
        edited = self._edited_lockbox_camset_path
        effective = edited or original

        if enabled:
            if not original:
                raise ParamError(
                    "Original source camset is required when lockbox is "
                    "enabled.")
            if not path_exists(original):
                raise ParamError(
                    f"Original source camset does not exist: {original}")
            if not effective or not path_exists(effective):
                raise ParamError(
                    f"Effective lockbox source does not exist: {effective}")

        return {
            "enabled": bool(enabled),
            "original_source_camset": original if enabled else None,
            "edited_source_camset": edited if enabled else None,
            "edited_source_metadata": (
                self._edited_lockbox_metadata_path if enabled else None),
            "source_camset": effective if enabled else None,
            "warm_start": bool(self._lockbox_warm_start_cb.isChecked()),
            "rotation_half_width": float(
                self._lockbox_rotation_half_spin.value()),
            "translation_half_width": float(
                self._lockbox_translation_half_spin.value()),
            "rotation_sigma": float(self._lockbox_rotation_sigma_spin.value()),
            "translation_sigma": float(
                self._lockbox_translation_sigma_spin.value()),
            "center_sigma": float(self._lockbox_center_sigma_spin.value()),
            "implementation_mode": "extrinsic_parameter_mvp",
        }

    def set_phase2_run_id(self, run_id: str) -> None:
        self._preferred_phase2_run_id = (run_id or "").strip() or None
        self._refresh_phase2_sources()
        if self._preferred_phase2_run_id:
            idx = self._phase2_run_combo.findData(self._preferred_phase2_run_id)
            if idx >= 0:
                self._phase2_run_combo.setCurrentIndex(idx)

    def set_selected_phase2_run_id(self, run_id: str) -> None:
        self.set_phase2_run_id(run_id)

    def set_phase2_run(self, run_id: str) -> None:
        self.set_phase2_run_id(run_id)

    def set_phase2_camset_path(self, path: str) -> None:
        self._preferred_phase2_camset_path = (path or "").strip() or None
        self._update_source_label()

    def set_camset_path(self, path: str) -> None:
        self.set_phase2_camset_path(path)

    def _resolve_handoff_phase2_run_id(self, runs: list[dict]) -> Optional[str]:
        ws = self._workspace_mgr.workspace_path
        if ws is None:
            return None
        handoff = ws / "handoff.json"
        if not handoff.exists():
            return None
        try:
            payload = json.loads(handoff.read_text(encoding="utf-8"))
            if payload.get("phase") == "phase2" and payload.get("runs"):
                wanted = payload["runs"][0].get("run_id")
                if wanted and any(run.get("run_id") == wanted for run in runs):
                    return str(wanted)
        except Exception:
            pass
        return None

    def _refresh_phase2_sources(self) -> None:
        if not hasattr(self, "_phase2_run_combo"):
            return
        runs = self._workspace_mgr.load_runs("phase2")
        keep = self._phase2_run_combo.currentData()
        self._phase2_run_combo.blockSignals(True)
        self._phase2_run_combo.clear()
        self._phase2_run_combo.addItem("Auto (handoff else latest)", None)
        for run in runs:
            rid = run.get("run_id", "unknown")
            self._phase2_run_combo.addItem(str(rid), str(rid))

        preferred = self._preferred_phase2_run_id
        if preferred is not None:
            idx = self._phase2_run_combo.findData(preferred)
            self._phase2_run_combo.setCurrentIndex(idx if idx >= 0 else 0)
        elif keep is not None:
            idx = self._phase2_run_combo.findData(keep)
            self._phase2_run_combo.setCurrentIndex(idx if idx >= 0 else 0)
        else:
            self._phase2_run_combo.setCurrentIndex(0)
        self._phase2_run_combo.blockSignals(False)
        self._update_source_label()

    def _on_phase2_source_changed(self) -> None:
        selected = self._phase2_run_combo.currentData() if hasattr(self, "_phase2_run_combo") else None
        self._preferred_phase2_run_id = str(selected) if selected else None
        self._update_source_label()

    def _update_source_label(self) -> None:
        if not hasattr(self, "_src_lbl"):
            return
        runs = self._workspace_mgr.load_runs("phase2")
        if not runs:
            self._src_lbl.setText("Inputs: auto (no Phase 2 run found)")
            self._adopt_target_from_phase1_run(None)
            return
        run = self._load_phase2_run()
        if run is None:
            self._src_lbl.setText("Inputs: auto (no Phase 2 run found)")
            self._adopt_target_from_phase1_run(None)
            return
        rid = run.get("run_id", "unknown")
        phase1_id = (run.get("inputs") or {}).get("phase1_run_id", "?")
        ws = self._workspace_mgr.workspace_path
        resolved = resolve_artifact(run, "phase2", ws) if ws is not None else None
        p = str(resolved) if resolved is not None else (run.get("artifacts") or {}).get("initial_camset")
        mode = "selected" if self._preferred_phase2_run_id else "auto"
        self._src_lbl.setText(f"Inputs: {mode} Phase 2 {rid} + linked Phase 1 {phase1_id} -> {p or 'missing camset artifact'}")
        self._adopt_target_from_phase1_run(self._load_phase1_run_for_phase2(run))

    def _adopt_target_from_phase1_run(self, run: Optional[dict]) -> None:
        """Match the target to the run whose detections Phase 3 will read.

        Phase 3 chooses a Phase 2 run; the detections come from the Phase 1
        run linked to it.  Only applied when that resolved run changes, so a
        target edited after choosing a run survives the next refresh.
        """
        run_id = (run or {}).get("run_id")
        if run_id is None:
            # No run applies any more (another workspace, say): its detector
            # must not linger as if it still did.
            if self._adopted_target_run_id is not None:
                self._adopted_target_run_id = None
                self._target_form.clear_inherited()
            return
        if run_id == self._adopted_target_run_id:
            return
        self._adopted_target_run_id = run_id
        self._target_form.apply_spec(
            target_params_of_run(run).get(TARGET_KEY, {}))

    def _load_phase2_run(self) -> Optional[dict]:
        runs = self._workspace_mgr.load_runs("phase2")
        if not runs:
            return None

        selected = self._phase2_run_combo.currentData() if hasattr(self, "_phase2_run_combo") else None
        wanted_id = str(selected) if selected else self._preferred_phase2_run_id
        if wanted_id:
            for run in runs:
                if run.get("run_id") == wanted_id:
                    return run

        wanted_id = self._resolve_handoff_phase2_run_id(runs)
        if wanted_id:
            for run in runs:
                if run.get("run_id") == wanted_id:
                    return run
        return runs[-1]

    def _load_phase1_run_for_phase2(self, phase2_run: dict) -> Optional[dict]:
        p1_runs = self._workspace_mgr.load_runs("phase1")
        if not p1_runs:
            return None
        wanted = phase2_run.get("inputs", {}).get("phase1_run_id")
        if wanted is None:
            return p1_runs[-1]
        for run in p1_runs:
            if run.get("run_id") == wanted:
                return run
        return p1_runs[-1]

    def _run_phase3(self) -> None:
        params = self._collect_params()
        if params is None:
            return
        try:
            require_detector_available(params)
        except ParamError as exc:
            QMessageBox.warning(self, "Detector unavailable", str(exc))
            return
        if not _PYCAMSET_OK:
            QMessageBox.critical(
                self, "Import error",
                "pyCamSet optimisation modules are unavailable.")
            return
        if self._worker is not None and self._worker.isRunning():
            if self._info_cb.isChecked():
                QMessageBox.information(
                    self, "Phase 3 running", "Phase 3 is already running.")
            return

        self._sync_workspace_from_floc(params["f_loc"])
        if self._workspace_mgr.workspace_path is None:
            QMessageBox.critical(
                self, "Workspace", "Could not initialize workspace.")
            return

        phase2_run = self._load_phase2_run()
        if phase2_run is None:
            if self._info_cb.isChecked():
                QMessageBox.information(
                    self, "No Phase 2 run", "Run Phase 2 first.")
            return

        phase1_run = self._load_phase1_run_for_phase2(phase2_run)
        if phase1_run is None:
            if self._info_cb.isChecked():
                QMessageBox.information(
                    self, "No Phase 1 run",
                    "No Phase 1 run found for detections.")
            return

        try:
            require_target_match(phase1_run, params)
        except ParamError as exc:
            QMessageBox.critical(
                self, "Target does not match the detections", str(exc))
            return

        params["selected_cameras"] = list(
            ((phase2_run.get("params") or {}).get("selected_cameras") or [])
            or ((phase1_run.get("params") or {}).get("selected_cameras") or []))

        override = self._preferred_phase2_camset_path
        camset_override = Path(override) if override else None

        self._src_lbl.setText(
            f"Phase 2: {phase2_run.get('run_id', '?')} | "
            f"Phase 1: {phase1_run.get('run_id', '?')}")

        self._terminal.clear_terminal()
        self._terminal.append_line("=== Phase 3: Template Bundle Adjustment ===")
        self._terminal.append_line(f"Image folder : {params['f_loc']}")
        self._terminal.append_line(
            f"Phase 2 run  : {phase2_run.get('run_id', 'unknown')}")
        self._terminal.append_line(
            f"Phase 1 run  : {phase1_run.get('run_id', 'unknown')}")
        self._terminal.append_line(
            f"selected cams: {params['selected_cameras'] or 'all'}")
        self._terminal.append_line("Starting…")

        workspace_mgr = self._workspace_mgr

        def work_fn(log: Callable[[str], None]) -> dict:
            return phase3_workflow.run(
                params, workspace_mgr, log,
                phase2_run=phase2_run, phase1_run=phase1_run,
                camset_override=camset_override)

        self._worker = PhaseWorker(work_fn, parent=self)
        self._worker.line_ready.connect(self._terminal.append_line)
        self._worker.finished.connect(self._on_run_finished)
        self._worker.error.connect(
            lambda msg: self._terminal.append_line(f"ERROR: {msg}"))
        self._worker.start()

    def _on_run_finished(self, metadata: dict) -> None:
        gate_continue_button(self._continue_btn, self._terminal, metadata)
        if self._diagnostics_tab is not None:
            self._diagnostics_tab.refresh()

    def _open_diagnostics(self) -> None:
        if self._diagnostics_tab is not None:
            self._diagnostics_tab.refresh()
            show_tab(self._notebook, self._diagnostics_tab)

    def _continue_to_phase4(self) -> None:
        runs = self._workspace_mgr.load_runs("phase3")
        if not runs:
            QMessageBox.information(self, "No runs", "Run Phase 3 first.")
            return

        chosen = runs[-1]
        f_loc = (chosen.get("params") or {}).get("f_loc")
        run_id = chosen.get("run_id")
        ws = self._workspace_mgr.workspace_path
        camset_resolved = resolve_artifact(chosen, "phase3", ws) if ws is not None else None
        camset_path = str(camset_resolved) if camset_resolved is not None else (chosen.get("artifacts") or {}).get("optimised_camset")

        self._workspace_mgr.write_handoff(
            {
                "phase": "phase3",
                "runs": [chosen],
                "image_folder": f_loc,
                "phase3_run_id": run_id,
                "optimised_camset": camset_path,
                "selected_cameras": ((chosen.get("params") or {}).get("selected_cameras") or []),
            }
        )

        for i in range(self._notebook.count()):
            if self._notebook.tabText(i) != TAB_PHASE4:
                continue
            phase4_tab = self._notebook.widget(i)

            def _try_call(obj, names: tuple[str, ...], value: str) -> bool:
                for meth in names:
                    if hasattr(obj, meth):
                        try:
                            getattr(obj, meth)(value)
                            return True
                        except Exception:
                            pass
                return False

            if f_loc:
                _try_call(phase4_tab, ("set_image_folder", "set_floc", "set_image_path"), str(f_loc))
            if run_id:
                _try_call(phase4_tab, ("set_phase3_run_id", "set_selected_phase3_run_id"), str(run_id))
            if camset_path:
                _try_call(phase4_tab, ("set_phase3_camset_path", "set_camset_path"), str(camset_path))

            self._notebook.setCurrentIndex(i)
            return

    def _visualise_target_from_primary(self) -> None:
        if self._diagnostics_tab is None:
            return
        self._diagnostics_tab.refresh()
        show_tab(self._notebook, self._diagnostics_tab)
        self._diagnostics_tab.visualise_from_primary()

    def _continue_to_next(self) -> None:
        self._continue_to_phase4()


class Phase3DiagnosticsTab(QWidget):
    def __init__(
        self,
        notebook: QTabWidget,
        info_cb: QCheckBox,
        workspace_mgr: WorkspaceManager,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._notebook = notebook
        self._info_cb = info_cb
        self._workspace_mgr = workspace_mgr
        self._build_ui()

    def _build_ui(self) -> None:
        self._threshold_worker = None  # PhaseWorker for threshold-based re-run
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        top_btn_row = QHBoxLayout()
        top_btn_row.addWidget(make_warning_button("▲ Bundle Settings", self._go_to_settings))
        top_btn_row.addStretch()
        root.addLayout(top_btn_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        left = QWidget()
        left.setMinimumWidth(180)
        left.setMaximumWidth(280)
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
        right_layout.addWidget(make_section_label("Phase 3 Diagnostics"))
        right_layout.addWidget(make_separator())

        self._sub_tabs = QTabWidget()
        right_layout.addWidget(self._sub_tabs)
        splitter.addWidget(right)

        splitter.setSizes([210, 760])

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
        self._sub_tabs.addTab(self._summary_widget, "Summary (D3.1-D3.10)")

        self._initial_widget, self._initial_layout, self._initial_scroll = make_scrollable_tab()
        self._sub_tabs.addTab(self._initial_widget, "Initial Per-image Error (D3.4)")

        self._residual_widget, self._residual_layout, self._residual_scroll = make_scrollable_tab()
        self._sub_tabs.addTab(self._residual_widget, "Residuals / Per-camera (D3.11-D3.12)")

        # D3.13 — Camera extrinsic poses sub-tab
        self._poses_widget, self._poses_layout, self._poses_scroll = make_scrollable_tab()
        self._sub_tabs.addTab(self._poses_widget, "Camera Poses (D3.13)")

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
            self._visual_widget, visual_id="assessment:phase3")
        visual_layout.addWidget(self._three_d_style)
        # Shows which run/phase the most recent Assess Calibration click actually
        # resolved to -- lets a user comparing PyVista vs. Open3D (or comparing this
        # tab against Phase 4's own Assess Calibration tab) immediately see whether
        # they are looking at two different camsets/runs on purpose, rather than
        # mistaking a run-selection mismatch for a rendering disagreement.
        self._current_run_label = QLabel("")
        self._current_run_label.setStyleSheet("color: #888; font-style: italic;")
        visual_layout.addWidget(self._current_run_label)
        self._open3d_output = QLabel("")
        self._open3d_output.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._open3d_output.setMinimumHeight(400)
        self._open3d_output.setStyleSheet("background: #1a1a2e; color: #666;")
        self._open3d_output.setText("Select Open3D backend and click Assess Calibration to render here.")
        self._open3d_output.setWordWrap(True)
        self._open3d_output.setVisible(False)
        visual_layout.addWidget(self._open3d_output)
        visual_layout.addStretch()
        self._sub_tabs.addTab(self._visual_widget, "Assess Calibration")

        bottom_btn_row = QHBoxLayout()
        bottom_btn_row.addStretch()
        bottom_btn_row.addWidget(make_green_button("Phase 4 - Self-Calibration ▶", self._go_to_phase4))
        root.addLayout(bottom_btn_row)

        self.refresh()

    def refresh(self) -> None:
        runs = merge_phase3_phase4_runs(
            self._workspace_mgr.load_runs("phase3"),
            self._workspace_mgr.load_runs("phase4"),
        )
        self._all_runs = runs
        self._run_selector.refresh(runs)
        selected = self._run_selector.get_selected()
        self._render_summary(selected)
        self._render_initial_plot(selected)
        self._render_residuals(selected)
        self._render_poses(selected)

    def _on_selection_changed(self, runs: list[dict]) -> None:
        self._run_selector.enforce_max_selection(4)
        selected = self._run_selector.get_selected()
        self._render_summary(selected)
        self._render_initial_plot(selected)
        self._render_residuals(selected)
        self._render_poses(selected)

    def _go_to_settings(self) -> None:
        for i in range(self._notebook.count()):
            if self._notebook.tabText(i) == TAB_PHASE3:
                self._notebook.setCurrentIndex(i)
                return

    def _go_to_phase4(self) -> None:
        """Navigate to Phase 4, pre-populating inputs from the selected run."""
        selected = self._run_selector.get_selected()
        chosen = next(
            (r for r in reversed(selected) if str(r.get("phase", "phase3")) == "phase3"),
            None,
        ) or (selected[-1] if selected else None)

        for i in range(self._notebook.count()):
            if self._notebook.tabText(i) != TAB_PHASE4:
                continue
            phase4_tab = self._notebook.widget(i)
            if chosen is not None:
                f_loc = (chosen.get("params") or {}).get("f_loc")
                run_id = chosen.get("run_id")
                ws = self._workspace_mgr.workspace_path
                camset_resolved = resolve_artifact(chosen, "phase3", ws) if ws is not None else None
                camset_path = str(camset_resolved) if camset_resolved is not None else (chosen.get("artifacts") or {}).get("optimised_camset")
                if f_loc and hasattr(phase4_tab, "set_image_folder"):
                    phase4_tab.set_image_folder(str(f_loc))
                if run_id and hasattr(phase4_tab, "set_phase3_run_id"):
                    phase4_tab.set_phase3_run_id(str(run_id))
                if camset_path and hasattr(phase4_tab, "set_phase3_camset_path"):
                    phase4_tab.set_phase3_camset_path(str(camset_path))
            self._notebook.setCurrentIndex(i)
            return

    @staticmethod
    def _clear_layout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                Phase3DiagnosticsTab._clear_layout(item.layout())

    def _render_summary(self, runs: list[dict]) -> None:
        while self._summary_layout.count():
            item = self._summary_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

        if not runs:
            lbl = QLabel("Select one or more runs to compare diagnostics.")
            lbl.setStyleSheet("color: gray;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._summary_layout.addWidget(lbl)
            return

        for run in runs:
            phase = str(run.get("phase", "phase3"))
            d = run.get("diagnostics", {})
            hdr = QLabel(f"Run: {run.get('run_id', 'unknown')} ({phase})")
            hdr.setStyleSheet("font-weight: bold; margin-top: 8px;")
            self._summary_layout.addWidget(hdr)

            form = QFormLayout()
            form.setContentsMargins(16, 0, 0, 0)
            if phase != "phase3":
                form.addRow("Note:", QLabel("This run is from Phase 4; D3 metrics are not available."))
            else:
                status = str(run.get("status", "unknown"))
                status_label = QLabel(status)
                status_label.setStyleSheet(
                    "font-weight: bold; color: "
                    + ("#228b22" if status == "complete" else "#b22222")
                )
                form.addRow("Disposition:", status_label)
                form.addRow("D3.1 missing poses:", QLabel(str(d.get("D3.1_n_missing_poses", "—"))))
                form.addRow("D3.2 outlier-removed poses:", QLabel(str(d.get("D3.2_n_outlier_removed", "—"))))
                form.addRow("D3.5 initial euclid (px):", QLabel(f"{float(d.get('D3.5_initial_euclid_px', float('nan'))):.5f}"))
                form.addRow("D3.6 final euclid (px):", QLabel(f"{float(d.get('D3.6_final_euclid_px', float('nan'))):.5f}"))
                form.addRow("D3.7 reduction ratio:", QLabel(f"{float(d.get('D3.7_error_reduction_ratio', float('nan'))):.5f}"))

                s = d.get("D3.8_solver_status", {})
                form.addRow("D3.8 solver status:", QLabel(f"{s.get('status', '—')} | success={s.get('success', '—')}"))
                msg_lbl = QLabel(str(s.get("message", "—")))
                msg_lbl.setWordWrap(True)
                form.addRow("D3.8 solver message:", msg_lbl)
                form.addRow("D3.9 nfev:", QLabel(str(d.get("D3.9_nfev", "—"))))

                ratio = d.get("D3.10_parameter_observation_ratio", {})
                form.addRow(
                    "D3.10 params/obs:",
                    QLabel(
                        f"{ratio.get('param_count', '—')} / {ratio.get('observation_count', '—')}"
                        f" (ratio={float(ratio.get('ratio', float('nan'))):.6f})"
                    ),
                )
                gate = d.get("quality_gate") or {}
                flags = gate.get("blocking_flags") or []
                if flags:
                    gate_label = QLabel("\n".join(str(flag) for flag in flags))
                    gate_label.setWordWrap(True)
                    gate_label.setStyleSheet("color: #b22222;")
                    form.addRow("Quality gate:", gate_label)

            if run.get("error"):
                form.addRow("Error:", QLabel(str(run["error"])))

            self._summary_layout.addLayout(form)
            self._summary_layout.addWidget(make_separator())
            render_predecessor_chain_section(self._summary_layout, self._workspace_mgr, run)

        self._summary_layout.addStretch()

    def _render_initial_plot(self, runs: list[dict]) -> None:
        while self._initial_layout.count():
            item = self._initial_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not runs:
            self._initial_layout.addWidget(QLabel("Select at least one Phase 3 run to view D3.4."))
            return

        run = next((r for r in reversed(runs) if str(r.get("phase", "phase3")) == "phase3"), None)
        if run is None:
            self._initial_layout.addWidget(QLabel("Select at least one Phase 3 run to view D3.4."))
            return

        vals = run.get("diagnostics", {}).get("D3.3_per_image_initial_reprojection", [])
        if not vals:
            self._initial_layout.addWidget(QLabel("No D3.3 data in selected run."))
            return

        try:
            import matplotlib
            matplotlib.use("QtAgg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except ImportError:
            self._initial_layout.addWidget(QLabel("matplotlib not available."))
            return

        arr = np.array(vals, dtype=float)
        valid_mask = ~np.isnan(arr)
        valid_arr = arr[valid_mask]

        # MAD threshold: same rule as TemplateBundleHandler.find_and_exclude_transform_outliers
        # out_thresh = 20; mad_thresh = median + 20 * MAD
        if valid_arr.size > 0:
            med_val = float(np.median(valid_arr))
            mad_val = float(np.median(np.abs(valid_arr - med_val)))
            mad_thresh = med_val + 20.0 * mad_val
        else:
            med_val = 0.0
            mad_val = 0.0
            mad_thresh = 0.0

        # Initial user threshold: mean + 2σ (a sensible starting point)
        if valid_arr.size > 0:
            init_user_thresh = float(np.nanmean(valid_arr) + 2.0 * np.nanstd(valid_arr))
            init_user_thresh = max(0.001, init_user_thresh)
        else:
            init_user_thresh = max(0.001, mad_thresh)

        # Build figure
        fig = Figure(figsize=(9, 4.8), tight_layout=True)
        ax = fig.add_subplot(111)
        ax.bar(np.arange(arr.size), arr, color="#1f77b4", alpha=0.88)

        # MAD threshold (backend rule, static)
        ax.axhline(
            mad_thresh, color="#d62728", linestyle="--", linewidth=1.1,
            label=f"MAD threshold (median + 20·MAD = {mad_thresh:.2f} px)",
        )
        # User threshold (interactive)
        user_hline = ax.axhline(
            init_user_thresh, color="#ff7f0e", linestyle="-", linewidth=1.8,
            label=f"User threshold ({init_user_thresh:.3f} px)", zorder=5,
        )

        _above_scatter = ax.scatter([], [], color="#ff7f0e", s=28, zorder=6,
                                    label="Above threshold")

        ax.set_title(f"D3.4 Per-image initial reprojection error ({run.get('run_id', '?')})")
        ax.set_xlabel("Image index")
        ax.set_ylabel("Initial error (aggregated px)")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)

        from pyCamSet.gui.theme import apply_matplotlib_theme
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        apply_matplotlib_theme(fig, app.property("pycamsetTheme") if app else "Light")
        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumHeight(320)

        # Helper: image indices whose value exceeds threshold
        def _indices_above(thresh: float) -> list[int]:
            return [i for i, v in enumerate(arr) if not np.isnan(v) and v > thresh]

        # Controls row
        ctrl = QWidget()
        crow = QHBoxLayout(ctrl)
        crow.setContentsMargins(4, 2, 4, 2)
        crow.addWidget(QLabel("User threshold (px):"))

        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1e8)
        spin.setDecimals(3)
        spin.setSingleStep(1.0)
        spin.setValue(init_user_thresh)
        spin.setFixedWidth(115)
        crow.addWidget(spin)

        stat_lbl = QLabel("")
        crow.addWidget(stat_lbl, stretch=1)

        n_sel = len(self._run_selector.get_selected())
        create_btn = QPushButton("⊖  Create new run with images above threshold removed")
        create_btn.setEnabled(n_sel == 1)
        create_btn.setToolTip(
            "Creates a new Phase 3 run excluding all images whose\n"
            "D3.3 initial reprojection error exceeds the user threshold.\n"
            "Select exactly one Phase 3 run to enable this action."
        )
        crow.addWidget(create_btn)

        def _update_threshold(v: float) -> None:
            user_hline.set_ydata([v, v])
            user_hline.set_label(f"User threshold ({v:.3f} px)")
            ax.legend(fontsize=8)
            canvas.draw_idle()
            idxs = _indices_above(v)
            stat_lbl.setText(f"  {len(idxs)} image(s) above threshold will be removed  ")
            create_btn.setEnabled(len(self._run_selector.get_selected()) == 1)

        spin.valueChanged.connect(_update_threshold)
        _update_threshold(init_user_thresh)

        # ── Drag-to-move threshold line ──────────────────────────────────
        _drag = {"active": False}

        def _on_press(event):
            if event.inaxes is not ax or event.button != 1:
                return
            thresh_display = ax.transData.transform((0, spin.value()))[1]
            if abs(event.y - thresh_display) < 5:
                _drag["active"] = True

        def _on_motion(event):
            if event.inaxes is ax:
                thresh_display = ax.transData.transform((0, spin.value()))[1]
                if abs(event.y - thresh_display) < 5:
                    canvas.setCursor(Qt.CursorShape.SizeVerCursor)
                else:
                    canvas.unsetCursor()
            else:
                canvas.unsetCursor()
            if not _drag["active"] or event.inaxes is not ax:
                return
            new_val = max(spin.minimum(), float(event.ydata))
            spin.blockSignals(True)
            spin.setValue(new_val)
            spin.blockSignals(False)
            _update_threshold(new_val)

        def _on_release(event):
            _drag["active"] = False

        canvas.mpl_connect("button_press_event", _on_press)
        canvas.mpl_connect("motion_notify_event", _on_motion)
        canvas.mpl_connect("button_release_event", _on_release)

        def _on_create() -> None:
            selected = self._run_selector.get_selected()
            if len(selected) != 1:
                QMessageBox.warning(
                    self, "Selection", "Select exactly one Phase 3 run to create a new run."
                )
                return
            src_run = next(
                (r for r in selected if str(r.get("phase", "phase3")) == "phase3"),
                None,
            )
            if src_run is None:
                QMessageBox.warning(self, "Selection", "The selected run is not a Phase 3 run.")
                return

            thresh = spin.value()
            idxs = _indices_above(thresh)
            if not idxs:
                QMessageBox.information(
                    self, "Nothing to remove",
                    "No images exceed the threshold. Lower threshold to exclude some images.",
                )
                return

            idx_preview = str(idxs[:10]) + ("…" if len(idxs) > 10 else "")
            reply = QMessageBox.question(
                self, "Confirm — create new Phase 3 run",
                f"User threshold: {thresh:.4f} px\n"
                f"MAD threshold (backend): {mad_thresh:.4f} px\n"
                f"Images to remove: {len(idxs)}   (indices: {idx_preview})\n\n"
                "A new Phase 3 run will be created with those images excluded.\n"
                "The source run will NOT be modified.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

            create_btn.setEnabled(False)
            old_txt = create_btn.text()
            create_btn.setText("⟳  Running Phase 3…")
            stat_lbl.setText("  Creating new run — please wait…  ")

            def _restore() -> None:
                create_btn.setText(old_txt)
                create_btn.setEnabled(len(self._run_selector.get_selected()) == 1)

            self._create_phase3_run_from_threshold(src_run, thresh, mad_thresh, idxs, _restore)

        create_btn.clicked.connect(_on_create)

        self._initial_layout.addWidget(ctrl)
        self._initial_layout.addWidget(MatplotlibFigureCard(
            f"D3.4 Per-image initial reprojection error ({run.get('run_id', '?')})",
            fig, FigureCanvasQTAgg, parent=self._initial_widget, min_height=320,
            canvas=canvas,
            csv_export={
                "columns": ["image_index", "initial_reprojection_error_px"],
                "rows": [(i, float(value)) for i, value in enumerate(arr) if np.isfinite(value)],
                "metadata": {
                    "run_id": run.get("run_id"), "phase": "phase3",
                    "diagnostic": "D3.3_per_image_initial_reprojection",
                    "data_kind": "observed per-image diagnostic; thresholds are display/interaction overlays",
                    "units": {"image_index": "index", "initial_reprojection_error_px": "px"},
                    "x_axis": "image_index", "y_axis": "initial_reprojection_error_px",
                },
            },
        ))

    def _create_phase3_run_from_threshold(
        self,
        source_run: dict,
        threshold: float,
        mad_threshold: float,
        image_indices: list[int],
        on_done_cb,
    ) -> None:
        """Solve the selected run again without its worst images."""
        if self._workspace_mgr.workspace_path is None:
            QMessageBox.critical(self, "Workspace", "No active workspace.")
            on_done_cb()
            return

        inputs = source_run.get("inputs") or {}
        prune = DetectionFilter(
            images=image_indices,
            record={
                "source_phase3_run_id": source_run.get("run_id"),
                "phase2_run_id": inputs.get("phase2_run_id"),
                "phase1_run_id": inputs.get("phase1_run_id"),
                "user_threshold_px": threshold,
                "mad_threshold_px": mad_threshold,
                "removed_global_image_indices": image_indices,
                "n_removed": len(image_indices),
                "removal_mode": "global via global_im_num",
            },
        )
        workspace_mgr = self._workspace_mgr

        def work_fn(log: Callable[[str], None]) -> dict:
            return phase3_workflow.rerun(source_run, workspace_mgr, log, prune)

        self._threshold_worker = PhaseWorker(work_fn, parent=self)
        self._send_output_to_settings_tab(self._threshold_worker)

        def _on_finished(metadata: dict) -> None:
            on_done_cb()
            if metadata.get("error"):
                QMessageBox.critical(
                    self, "Phase 3 failed",
                    f"New run encountered an error:\n{metadata['error']}")
            else:
                removed = (metadata.get("threshold_pruning") or {}).get(
                    "n_removed", len(image_indices))
                QMessageBox.information(
                    self, "New Phase 3 run created",
                    f"Run ID: {metadata.get('run_id', '?')}\n"
                    f"Removed {removed} image(s) with initial "
                    f"RPE > {threshold:.4f} px.")
            self.refresh()

        self._threshold_worker.finished.connect(_on_finished)
        self._threshold_worker.start()

    def _send_output_to_settings_tab(self, worker: PhaseWorker) -> None:
        """Write the worker's output to the settings tab's terminal.

        The run is started from the diagnostics tab, which has no terminal of
        its own, so its output would otherwise go nowhere someone can read it.
        """
        for index in range(self._notebook.count()):
            if self._notebook.tabText(index) != TAB_PHASE3:
                continue
            settings_tab = self._notebook.widget(index)
            terminal = getattr(settings_tab, "_terminal", None)
            if terminal is not None:
                worker.line_ready.connect(terminal.append_line)
            return

    def _render_residuals(self, runs: list[dict]) -> None:
        while self._residual_layout.count():
            item = self._residual_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not runs:
            self._residual_layout.addWidget(QLabel("Select at least one Phase 3 run to view D3.11/D3.12."))
            return

        run = next((r for r in reversed(runs) if str(r.get("phase", "phase3")) == "phase3"), None)
        if run is None:
            self._residual_layout.addWidget(QLabel("Select at least one Phase 3 run to view D3.11/D3.12."))
            return

        d = run.get("diagnostics", {})
        per_cam = d.get("D3.12_per_camera_mean_reprojection", {})

        if not per_cam:
            self._residual_layout.addWidget(QLabel("No D3.12 data in selected run."))
            return

        try:
            import matplotlib
            matplotlib.use("QtAgg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except ImportError:
            self._residual_layout.addWidget(QLabel("matplotlib not available."))
            return

        cams = list(per_cam.keys())
        vals = np.array([float(per_cam[c]) for c in cams], dtype=float)
        med = np.nanmedian(vals)

        fig = Figure(figsize=(9, 4.8), tight_layout=True)
        ax = fig.add_subplot(111)
        bars = ax.bar(np.arange(len(cams)), vals, color="#2ca02c", alpha=0.9)
        ax.axhline(med, color="#ff7f0e", linestyle="--", linewidth=1.2, label=f"median={med:.4f}")
        ax.set_xticks(np.arange(len(cams)))
        ax.set_xticklabels(cams, rotation=25, ha="right")
        ax.set_ylabel("Mean reprojection error (px)")
        ax.set_title(f"D3.12 Per-camera mean reprojection error ({run.get('run_id', '?')})")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        self._residual_layout.addWidget(
            MatplotlibFigureCard(
                f"D3.12 Per-camera mean reprojection error ({run.get('run_id', '?')})",
                fig,
                FigureCanvasQTAgg,
                parent=self._residual_widget,
                min_height=360,
                csv_export={
                    "columns": ["camera", "mean_reprojection_error_px"],
                    "rows": [(cam, float(per_cam[cam])) for cam in cams],
                    "metadata": {
                        "run_id": run.get("run_id"), "phase": "phase3",
                        "diagnostic": "D3.12_per_camera_mean_reprojection",
                        "data_kind": "observed diagnostic summary", "units": {"mean_reprojection_error_px": "px"},
                        "x_axis": "camera", "y_axis": "mean_reprojection_error_px",
                    },
                },
            )
        )

        # D3.11 — Residual x/y scatter (one point per detection)
        d311 = d.get("D3.11_residual_xy_scatter", [])
        if d311 and isinstance(d311, list) and len(d311) > 0:
            try:
                res_arr = np.array(d311, dtype=float)
                if res_arr.ndim == 2 and res_arr.shape[1] == 2:
                    fig2 = Figure(figsize=(8.2, 6.0), tight_layout=True)
                    ax2 = fig2.add_subplot(111)
                    ax2.scatter(res_arr[:, 0], res_arr[:, 1], s=4, alpha=0.4, c="#1f77b4")
                    ax2.axhline(0, color="#888", linewidth=0.5)
                    ax2.axvline(0, color="#888", linewidth=0.5)
                    ax2.set_xlabel("Residual x (px)")
                    ax2.set_ylabel("Residual y (px)")
                    ax2.set_title(
                        f"D3.11 Residual x/y scatter ({run.get('run_id', '?')})\n"
                        f"({res_arr.shape[0]} detections)"
                    )
                    ax2.set_aspect("equal", adjustable="box")
                    ax2.grid(alpha=0.2)
                    self._residual_layout.addWidget(
                        MatplotlibFigureCard(
                            f"D3.11 Residual x/y scatter ({run.get('run_id', '?')})",
                            fig2,
                            FigureCanvasQTAgg,
                            parent=self._residual_widget,
                            min_height=360,
                            csv_export={
                                "columns": ["residual_x_px", "residual_y_px"],
                                "rows": [(float(x), float(y)) for x, y in res_arr],
                                "metadata": {
                                    "run_id": run.get("run_id"), "phase": "phase3",
                                    "diagnostic": "D3.11_residual_xy_scatter",
                                    "data_kind": "observed residual coordinates", "units": {"residual_x_px": "px", "residual_y_px": "px"},
                                    "x_axis": "residual_x_px", "y_axis": "residual_y_px",
                                },
                            },
                        )
                    )
            except Exception:
                pass  # D3.11 rendering is best-effort

    def _render_poses(self, runs: list[dict]) -> None:
        while self._poses_layout.count():
            item = self._poses_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not runs:
            self._poses_layout.addWidget(QLabel("Select at least one Phase 3 run to view D3.13."))
            return

        run = next((r for r in reversed(runs) if str(r.get("phase", "phase3")) == "phase3"), None)
        if run is None:
            self._poses_layout.addWidget(QLabel("Select at least one Phase 3 run to view D3.13."))
            return

        # Resolved, not read straight out of the record, as every other
        # artifact lookup in this file is: a run whose workspace moved
        # still has its camset, and a bare lookup reports it missing.
        ws = self._workspace_mgr.workspace_path
        camset_path = (resolve_artifact(run, "phase3", ws) if ws is not None
                       else run.get("artifacts", {}).get("optimised_camset"))
        if not camset_path:
            self._poses_layout.addWidget(QLabel("No optimised camset artifact in selected run."))
            return

        try:
            cams = load_CameraSet(Path(camset_path))
        except Exception as exc:
            self._poses_layout.addWidget(QLabel(f"Could not load camset: {exc}"))
            return

        try:
            import matplotlib
            matplotlib.use("QtAgg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except ImportError:
            self._poses_layout.addWidget(QLabel("matplotlib not available."))
            return

        fig = Figure(figsize=(8.2, 6.2), tight_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        xs, ys, zs, names = [], [], [], []
        for cam in cams:
            p = np.array(cam.position).reshape(-1)
            if p.size >= 3:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
                zs.append(float(p[2]))
                names.append(cam.name)

        if xs:
            ax.scatter(xs, ys, zs, c="#1f77b4", s=36)
            for x, y, z, name in zip(xs, ys, zs, names):
                ax.text(x, y, z, name, fontsize=8)

        ax.set_title(f"D3.13 Camera extrinsic positions ({run.get('run_id', '?')})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        from pyCamSet.gui.theme import apply_matplotlib_theme
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        apply_matplotlib_theme(fig, app.property("pycamsetTheme") if app else "Light")
        pose_rows = [(cam.name, *[float(v) for v in np.asarray(cam.position).reshape(-1)[:3]])
                     for cam in cams if np.asarray(cam.position).size >= 3]
        self._poses_layout.addWidget(MatplotlibFigureCard(
            f"D3.13 Camera extrinsic positions ({run.get('run_id', '?')})",
            fig, FigureCanvasQTAgg, parent=self._poses_widget, min_height=360,
            csv_export={
                "columns": ["camera", "x", "y", "z"], "rows": pose_rows,
                "metadata": {
                    "run_id": run.get("run_id"), "phase": "phase3",
                    "diagnostic": "camera position derived from selected camset extrinsics",
                    "data_kind": "derived model output, not observed image coordinates",
                    "coordinate_frame": "camset world frame",
                    "units": "not declared by source camset; values retain camset coordinate units",
                    "x_axis": "camera", "y_axis": "x,y,z position components",
                },
            },
        ))

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
        chosen = select_latest_visualisation_run(selected, getattr(self, "_all_runs", []))
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
            chosen = select_latest_visualisation_run(selected, getattr(self, "_all_runs", []))
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
        chosen = select_latest_visualisation_run(selected, getattr(self, "_all_runs", []))
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

    def _continue_to_next(self) -> None:
        selected = self._run_selector.get_selected()
        if len(selected) != 1:
            if self._info_cb.isChecked():
                QMessageBox.information(
                    self,
                    "Select one run",
                    "Please select exactly one Phase 3 run for handoff.",
                )
            return
        chosen = selected[0]
        self._workspace_mgr.write_handoff(
            {
                "phase": "phase3",
                "runs": [chosen],
                "selected_cameras": ((chosen.get("params") or {}).get("selected_cameras") or []),
            }
        )
        if self._info_cb.isChecked():
            QMessageBox.information(self, "Handoff written", "handoff.json written for Phase 3.")
