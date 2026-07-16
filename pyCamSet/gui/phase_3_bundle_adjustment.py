"""
Phase 3 - Template bundle adjustment GUI.

Implements Phase 3 by orchestrating core pyCamSet
bundle-adjustment functions and persisting run diagnostics.
"""
from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
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

from pyCamSet.gui.shared_functions import (
    as_io_path,
    IMAGE_FOLDER_SCHEMATIC,
    TAB_PHASE3,
    TAB_PHASE4,
    CollapsibleSection,
    EmitLogHandler,
    EmitStream,
    MatplotlibFigureCard,
    PhaseWorker,
    RunSelectorWidget,
    TerminalWidget,
    WorkspaceManager,
    build_target,
    extract_detection,
    make_blue_button,
    make_orange_button,
    make_green_button,
    make_run_id,
    make_scrollable_tab,
    make_section_label,
    make_separator,
    ensure_directory,
    render_predecessor_chain_section,
    resolve_phase1_pickle_artifact,
    resolve_phase2_camset_artifact,
    resolve_phase3_camset_artifact,
    path_exists,
    suppress_matplotlib_gui,
)
from pyCamSet.gui.assess_calibration import (
    launch_visualise_calibration_for_run,
    launch_visualise_calibration_open3d_for_run,
    launch_save_pyvista_png_for_run,
    merge_phase3_phase4_runs,
    select_latest_visualisation_run,
)
from pyCamSet.gui.phase_3_lockbox_editor import Phase3LockboxEditor

try:
    from pyCamSet.optimisation.camera_lockbox import CameraLockboxConfig
    from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment_with_stats
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler
    from pyCamSet.utils.saving import load_CameraSet, load_pickle

    _PYCAMSET_OK = True
except ImportError:
    CameraLockboxConfig = None
    run_bundle_adjustment_with_stats = None
    TemplateBundleHandler = None
    load_CameraSet = None
    load_pickle = None
    _PYCAMSET_OK = False

_TARGET_CHOICES = ["Ccube", "ChArUco"]


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
        top_row.addWidget(form_widget, stretch=1)

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

        self._target_combo = QComboBox()
        self._target_combo.addItems(_TARGET_CHOICES)
        self._target_combo.setFixedWidth(140)
        self._target_combo.setToolTip(
            "Concept: the physical calibration target type.\n\n"
            "Default: Ccube\n"
            "Guidance: must match the target used in Phase 1 detection."
        )
        target_sect.addRow("Target type:", self._target_combo)

        self._npts_spin = QSpinBox()
        self._npts_spin.setRange(2, 30)
        self._npts_spin.setValue(6)
        self._npts_spin.setFixedWidth(90)
        self._npts_spin.setToolTip(
            "Concept: grid density of the calibration target.\n\n"
            "Default: 6\n"
            "Range: 2–30\n"
            "Guidance: must exactly match the value used in Phase 1."
        )
        target_sect.addRow("n_points / squares_x:", self._npts_spin)

        self._length_edit = QLineEdit("30.0")
        self._length_edit.setFixedWidth(110)
        self._length_edit.setToolTip(
            "Concept: physical size of one feature on the target (mm).\n\n"
            "Default: 30.0 mm\n"
            "Guidance: must exactly match the value used in Phase 1."
        )
        target_sect.addRow("Length / square size (mm):", self._length_edit)

        self._border_label = QLabel("Border fraction (Ccube):")
        self._border_spin = QDoubleSpinBox()
        self._border_spin.setRange(0.0, 0.9)
        self._border_spin.setDecimals(3)
        self._border_spin.setSingleStep(0.01)
        self._border_spin.setValue(0.1)
        target_sect.addRow(self._border_label, self._border_spin)

        self._marker_label = QLabel("Marker fraction (ChArUco):")
        self._marker_spin = QDoubleSpinBox()
        self._marker_spin.setRange(0.1, 1.0)
        self._marker_spin.setDecimals(3)
        self._marker_spin.setSingleStep(0.05)
        self._marker_spin.setValue(0.8)
        target_sect.addRow(self._marker_label, self._marker_spin)

        self._target_combo.currentTextChanged.connect(self._on_target_type_changed)
        self._on_target_type_changed(self._target_combo.currentText())

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
        btn_row.addWidget(make_orange_button("Diagnostics ▼", self._open_diagnostics))
        btn_row.addWidget(make_green_button("Phase 4 - Self-Calibration", self._continue_to_phase4))
        btn_row.addWidget(make_green_button("Assess Calibration", self._visualise_target_from_primary))
        btn_row.addStretch()
        form_root.addLayout(btn_row)
        form_root.addStretch()

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
            camset_path = resolve_phase2_camset_artifact(phase2_run, self._workspace_mgr.workspace_path)
            if camset_path is not None and path_exists(str(camset_path)):
                try:
                    active_names = list(load_CameraSet(as_io_path(str(camset_path))).get_names())
                except Exception:
                    active_names = []
        try:
            target = build_target(
                self._target_combo.currentText(),
                self._npts_spin.value(),
                float(self._length_edit.text().strip()),
                border_fraction=self._border_spin.value(),
                marker_fraction=self._marker_spin.value(),
            )
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

    def _on_target_type_changed(self, target_type: str) -> None:
        is_ccube = target_type == "Ccube"
        self._border_label.setVisible(is_ccube)
        self._border_spin.setVisible(is_ccube)
        self._marker_label.setVisible(not is_ccube)
        self._marker_spin.setVisible(not is_ccube)

    def _collect_params(self) -> Optional[dict]:
        floc = self._floc_edit.text().strip()
        if not floc:
            QMessageBox.critical(self, "Validation Error", "Image folder is required.")
            return None

        try:
            threads = int(self._threads_edit.text().strip()) if self._threads_edit.text().strip() else 1
            fixed_pose = int(self._fixed_pose_edit.text().strip())
            ref_cam = int(self._ref_cam_edit.text().strip())
            ref_pose = int(self._ref_pose_edit.text().strip())
            length = float(self._length_edit.text().strip())
        except ValueError as exc:
            QMessageBox.critical(self, "Validation Error", f"Invalid numeric value: {exc}")
            return None

        fixed_params = None
        if self._fp_edit.text().strip():
            try:
                fixed_params = json.loads(self._fp_edit.text().strip())
            except json.JSONDecodeError as exc:
                QMessageBox.critical(self, "Validation Error", f"Fixed params JSON: {exc}")
                return None

        raw_outlier_mode = (self._outliers_combo.currentText() or "").strip().lower()
        if raw_outlier_mode in {"y", "yes", "true", "1", "on", "enabled"}:
            outlier_mode = "y"
        elif raw_outlier_mode in {"n", "no", "false", "0", "off", "none", "disabled"}:
            outlier_mode = "n"
        elif raw_outlier_mode == "ask":
            outlier_mode = "y"  # GUI worker cannot do stdin prompts safely
        else:
            outlier_mode = "n"

        lockbox_enabled = self._lockbox_enabled_cb.isChecked()
        original_lockbox_source_path = self._lockbox_source_edit.text().strip()
        edited_lockbox_source_path = self._edited_lockbox_camset_path
        effective_lockbox_source_path = edited_lockbox_source_path or original_lockbox_source_path
        if lockbox_enabled:
            if not original_lockbox_source_path:
                QMessageBox.critical(self, "Validation Error", "Original source camset is required when lockbox is enabled.")
                return None
            if not path_exists(original_lockbox_source_path):
                QMessageBox.critical(self, "Validation Error", f"Original source camset does not exist: {original_lockbox_source_path}")
                return None
            if not effective_lockbox_source_path or not path_exists(effective_lockbox_source_path):
                QMessageBox.critical(self, "Validation Error", f"Effective lockbox source does not exist: {effective_lockbox_source_path}")
                return None

        return {
            "f_loc": floc,
            "threads": threads,
            "fixed_params": fixed_params,
            "target_type": self._target_combo.currentText(),
            "n_points": self._npts_spin.value(),
            "length": length,
            "border_fraction": self._border_spin.value(),
            "marker_fraction": self._marker_spin.value(),
            "lockbox": {
                "enabled": bool(lockbox_enabled),
                "original_source_camset": original_lockbox_source_path if lockbox_enabled else None,
                "edited_source_camset": edited_lockbox_source_path if lockbox_enabled else None,
                "edited_source_metadata": self._edited_lockbox_metadata_path if lockbox_enabled else None,
                "source_camset": effective_lockbox_source_path if lockbox_enabled else None,
                "warm_start": bool(self._lockbox_warm_start_cb.isChecked()),
                "rotation_half_width": float(self._lockbox_rotation_half_spin.value()),
                "translation_half_width": float(self._lockbox_translation_half_spin.value()),
                "rotation_sigma": float(self._lockbox_rotation_sigma_spin.value()),
                "translation_sigma": float(self._lockbox_translation_sigma_spin.value()),
                "center_sigma": float(self._lockbox_center_sigma_spin.value()),
                "implementation_mode": "extrinsic_parameter_mvp",
            },
            "problem_options": {
                "verbosity": int(self._verbosity_spin.value()),
                "fixed_pose": fixed_pose,
                "ref_cam": ref_cam,
                "ref_pose": ref_pose,
                "outliers": outlier_mode,
                "max_nfev": int(self._max_nfev_spin.value()),
            },
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
            payload = json.loads(handoff.read_text())
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
            return
        run = self._load_phase2_run()
        if run is None:
            self._src_lbl.setText("Inputs: auto (no Phase 2 run found)")
            return
        rid = run.get("run_id", "unknown")
        phase1_id = (run.get("inputs") or {}).get("phase1_run_id", "?")
        ws = self._workspace_mgr.workspace_path
        resolved = resolve_phase2_camset_artifact(run, ws) if ws is not None else None
        p = str(resolved) if resolved is not None else (run.get("artifacts") or {}).get("initial_camset")
        mode = "selected" if self._preferred_phase2_run_id else "auto"
        self._src_lbl.setText(f"Inputs: {mode} Phase 2 {rid} + linked Phase 1 {phase1_id} -> {p or 'missing camset artifact'}")

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
        if not _PYCAMSET_OK:
            QMessageBox.critical(self, "Import error", "pyCamSet optimisation modules are unavailable.")
            return
        if self._worker is not None and self._worker.isRunning():
            if self._info_cb.isChecked():
                QMessageBox.information(self, "Phase 3 running", "Phase 3 is already running.")
            return

        self._sync_workspace_from_floc(params["f_loc"])
        if self._workspace_mgr.workspace_path is None:
            QMessageBox.critical(self, "Workspace", "Could not initialize workspace.")
            return

        phase2_run = self._load_phase2_run()
        if phase2_run is None:
            if self._info_cb.isChecked():
                QMessageBox.information(self, "No Phase 2 run", "Run Phase 2 first.")
            return

        phase1_run = self._load_phase1_run_for_phase2(phase2_run)
        if phase1_run is None:
            if self._info_cb.isChecked():
                QMessageBox.information(self, "No Phase 1 run", "No Phase 1 run found for detections.")
            return

        selected_cameras = list(
            ((phase2_run.get("params") or {}).get("selected_cameras") or [])
            or ((phase1_run.get("params") or {}).get("selected_cameras") or [])
        )
        params["selected_cameras"] = selected_cameras

        self._src_lbl.setText(
            f"Phase 2: {phase2_run.get('run_id', '?')} | Phase 1: {phase1_run.get('run_id', '?')}"
        )

        self._terminal.clear_terminal()
        self._terminal.append_line("=== Phase 3: Template Bundle Adjustment ===")
        self._terminal.append_line(f"Image folder : {params['f_loc']}")
        self._terminal.append_line(f"Phase 2 run  : {phase2_run.get('run_id', 'unknown')}")
        self._terminal.append_line(f"Phase 1 run  : {phase1_run.get('run_id', 'unknown')}")
        self._terminal.append_line(f"selected cams: {selected_cameras if selected_cameras else 'all'}")
        self._terminal.append_line("Starting…")

        def work_fn(emit: Callable[[str], None]) -> dict:
            ws_path = self._workspace_mgr.workspace_path
            assert ws_path is not None

            run_id = make_run_id()
            run_dir = ws_path / "phase3_runs" / run_id
            ensure_directory(run_dir)

            diagnostics: dict = {}
            camset_path: Optional[str] = None
            p1_pickle: Optional[Path] = None

            stream = EmitStream(emit)
            log_handler = EmitLogHandler(emit)
            log_handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
            root_logger = logging.getLogger()
            root_logger.addHandler(log_handler)

            try:
                with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream), suppress_matplotlib_gui():
                    emit("Phase 3 running in non-interactive plotting mode (thread-safe).")

                    camset_path = self._preferred_phase2_camset_path
                    if camset_path and not path_exists(camset_path):
                        camset_path = None
                    if not camset_path:
                        camset_resolved = resolve_phase2_camset_artifact(phase2_run, ws_path)
                        camset_path = str(camset_resolved) if camset_resolved is not None else None
                    if not camset_path:
                        raise RuntimeError("Phase 2 run is missing initial_camset artifact.")
                    if not path_exists(camset_path):
                        raise RuntimeError(f"Phase 2 camset path does not exist: {camset_path}")

                    p1_pickle = resolve_phase1_pickle_artifact(phase1_run, ws_path)
                    if not p1_pickle:
                        raise RuntimeError("Could not resolve Phase 1 detected_datapoints.pickle artifact.")

                    cams = load_CameraSet(as_io_path(camset_path))
                    payload = load_pickle(as_io_path(p1_pickle))
                    detections = extract_detection(payload)
                    if detections is None:
                        raise RuntimeError("Could not extract TargetDetection from Phase 1 pickle.")

                    selected = list(params.get("selected_cameras") or [])
                    if selected:
                        selected_set = set(selected)
                        camset_names = set(cams.get_names())
                        det_names = set(getattr(detections, "cam_names", []) or [])
                        if camset_names != selected_set:
                            raise RuntimeError(
                                "Phase 2 camset cameras do not match selected camera subset. "
                                "Re-run Phase 2 with the same selected cameras."
                            )
                        if det_names != selected_set:
                            raise RuntimeError(
                                "Phase 1 detections do not match selected camera subset. "
                                "Re-run Phase 1/2 with the same selected cameras."
                            )

                    target = build_target(
                        params["target_type"],
                        params["n_points"],
                        params["length"],
                        border_fraction=params.get("border_fraction", 0.1),
                        marker_fraction=params.get("marker_fraction", 0.8),
                    )
                    lockbox_params = dict(params.get("lockbox") or {})
                    lockbox_config = CameraLockboxConfig(
                        enabled=bool(lockbox_params.get("enabled", False)),
                        rotation_half_width=float(lockbox_params.get("rotation_half_width", 0.1)),
                        translation_half_width=float(lockbox_params.get("translation_half_width", 0.1)),
                        rotation_sigma=float(lockbox_params.get("rotation_sigma", 0.05)),
                        translation_sigma=float(lockbox_params.get("translation_sigma", 0.01)),
                        center_sigma=float(lockbox_params.get("center_sigma", 0.0)),
                    )
                    lockbox_source_camset = None
                    lockbox_source_path = lockbox_params.get("source_camset")
                    if lockbox_config.enabled:
                        if not lockbox_source_path:
                            raise RuntimeError("Lockbox is enabled but no effective lockbox source path was provided.")
                        emit(f"Loading effective lockbox source camset: {lockbox_source_path}")
                        if lockbox_params.get("original_source_camset") and lockbox_params.get("edited_source_camset"):
                            emit(f"Original source camset: {lockbox_params.get('original_source_camset')}")
                            emit(f"Edited lockbox copy: {lockbox_params.get('edited_source_camset')}")
                        lockbox_source_camset = load_CameraSet(as_io_path(lockbox_source_path))
                        if set(lockbox_source_camset.get_names()) != set(cams.get_names()):
                            raise RuntimeError(
                                "Effective lockbox source camera names do not match the active Phase 2 camset. "
                                "This would break TemplateBundleHandler."
                            )

                    handler = TemplateBundleHandler(
                        camset=cams,
                        target=target,
                        detection=detections,
                        fixed_params=params["fixed_params"],
                        options=params["problem_options"],
                        lockbox_config=lockbox_config,
                        lockbox_source_camset=lockbox_source_camset,
                        lockbox_warm_start=bool(lockbox_params.get("warm_start", True)),
                    )
                    optimisation, out_cams, stats = run_bundle_adjustment_with_stats(  # type: ignore[arg-type]
                        handler,
                        threads=params["threads"],
                    )

                    init_euclid = float(stats.get("initial_euclid", float("nan")))
                    final_euclid = float(stats.get("final_euclid", float("nan")))
                    dt = float(stats.get("elapsed_sec", float("nan")))

                    emit(f"D3.5  Initial Euclidean reprojection error: {init_euclid:.4f} px")
                    emit(f"D3.6  Final Euclidean reprojection error: {final_euclid:.4f} px")
                    if not bool(stats.get("success", optimisation.success)):
                        emit(f"Solver note: {stats.get('message', optimisation.message)}")
                    emit(f"Optimisation finished in {dt:.2f}s")

                    camset_out_path = run_dir / "optimised_cameras.camset"
                    out_cams.save(camset_out_path)

                    missing_before = np.array(
                        getattr(handler, "missing_poses_before_outlier_rejection", []),
                        dtype=bool,
                    )
                    missing_after = np.array(
                        getattr(
                            handler,
                            "missing_poses_after_outlier_rejection",
                            handler.missing_poses if handler.missing_poses is not None else [],
                        ),
                        dtype=bool,
                    )
                    per_im_init = np.array(getattr(handler, "initial_per_im_error", []), dtype=float)

                    # Robust D3.12 (do not fail run if diagnostic sizing differs)
                    per_cam_err: dict[str, float] = {}
                    try:
                        dd = np.asarray(handler.get_detection_data(flatten=True))
                        residual_xy = np.reshape(np.asarray(optimisation.fun, dtype=float), (-1, 2))
                        residual_norm = np.linalg.norm(residual_xy, axis=1)

                        if dd.ndim == 2 and dd.shape[1] >= 1:
                            cam_idx = dd[:, 0].astype(int)
                            if cam_idx.size != residual_norm.size:
                                n = min(cam_idx.size, residual_norm.size)
                                emit(
                                    f"Warning: D3.12 alignment mismatch (cam_idx={cam_idx.size}, residuals={residual_norm.size}); truncating to {n}."
                                )
                                cam_idx = cam_idx[:n]
                                residual_norm = residual_norm[:n]

                            for idx, name in enumerate(handler.cam_names):
                                mask = cam_idx == idx
                                per_cam_err[name] = (
                                    float(np.mean(residual_norm[mask])) if np.any(mask) else float("nan")
                                )
                        else:
                            emit("Warning: D3.12 skipped (unexpected detection-data shape).")
                    except Exception as diag_exc:
                        emit(f"Warning: D3.12 skipped due to diagnostics error: {diag_exc}")

                    param_count = int(stats.get("param_count", 0))
                    obs_count = int(stats.get("observation_count", len(optimisation.fun) // 2))

                    n_missing_before = int(np.sum(missing_before))
                    n_missing_after = int(np.sum(missing_after))
                    diagnostics["D3.1_n_missing_poses"] = n_missing_before
                    diagnostics["D3.2_n_outlier_removed"] = int(max(0, n_missing_after - n_missing_before))
                    diagnostics["D3.3_per_image_initial_reprojection"] = per_im_init.tolist()
                    diagnostics["D3.4_initial_error_plot"] = "rendered in diagnostics tab"
                    diagnostics["D3.5_initial_euclid_px"] = init_euclid
                    diagnostics["D3.6_final_euclid_px"] = final_euclid
                    diagnostics["D3.7_error_reduction_ratio"] = (
                        float(init_euclid / final_euclid) if final_euclid > 0 else float("inf")
                    )
                    diagnostics["D3.8_solver_status"] = {
                        "status": int(stats.get("status", optimisation.status)),
                        "message": str(stats.get("message", optimisation.message)),
                        "success": bool(stats.get("success", optimisation.success)),
                    }
                    diagnostics["D3.9_nfev"] = int(stats.get("nfev", optimisation.nfev))
                    diagnostics["D3.10_parameter_observation_ratio"] = {
                        "param_count": param_count,
                        "observation_count": obs_count,
                        "ratio": float(param_count / max(obs_count, 1)),
                    }
                    diagnostics["D3.11_residual_xy_scatter"] = "rendered in diagnostics tab"
                    diagnostics["D3.12_per_camera_mean_reprojection"] = per_cam_err
                    diagnostics["D3.13_extrinsic_pose_view"] = "rendered in diagnostics tab"

                    metadata = {
                        "run_id": run_id,
                        "phase": "phase3",
                        "params": params,
                        "diagnostics": diagnostics,
                        "error": None,
                        "inputs": {
                            "phase2_run_id": phase2_run.get("run_id"),
                            "phase1_run_id": phase1_run.get("run_id"),
                        },
                        "artifacts": {
                            "optimised_camset": str(camset_out_path),
                            "phase2_initial_camset_used": str(camset_path),
                            "phase1_detection_pickle_used": str(p1_pickle),
                        },
                    }
                    self._workspace_mgr.save_run("phase3", run_id, metadata)
                    emit(f"Run saved: {run_id}")
                    return metadata

            except Exception as exc:
                msg = str(exc)
                emit(f"ERROR: {msg}")
                metadata = {
                    "run_id": run_id,
                    "phase": "phase3",
                    "params": params,
                    "diagnostics": diagnostics,
                    "error": msg,
                    "inputs": {
                        "phase2_run_id": phase2_run.get("run_id"),
                        "phase1_run_id": phase1_run.get("run_id"),
                    },
                    "artifacts": {
                        "phase2_initial_camset_used": str(camset_path) if camset_path else None,
                        "phase1_detection_pickle_used": str(p1_pickle) if p1_pickle is not None else None,
                    },
                }
                self._workspace_mgr.save_run("phase3", run_id, metadata)
                return metadata
            finally:
                stream.flush()
                root_logger.removeHandler(log_handler)

        self._worker = PhaseWorker(work_fn, parent=self)
        self._worker.line_ready.connect(self._terminal.append_line)
        self._worker.finished.connect(self._on_run_finished)
        self._worker.error.connect(lambda msg: self._terminal.append_line(f"ERROR: {msg}"))
        self._worker.start()

    def _on_run_finished(self, metadata: dict) -> None:
        if self._diagnostics_tab is not None:
            self._diagnostics_tab.refresh()

    def _open_diagnostics(self) -> None:
        if self._diagnostics_tab is not None:
            self._diagnostics_tab.refresh()
            self._notebook.setCurrentWidget(self._diagnostics_tab)

    def _continue_to_phase4(self) -> None:
        runs = self._workspace_mgr.load_runs("phase3")
        if not runs:
            QMessageBox.information(self, "No runs", "Run Phase 3 first.")
            return

        chosen = runs[-1]
        f_loc = (chosen.get("params") or {}).get("f_loc")
        run_id = chosen.get("run_id")
        ws = self._workspace_mgr.workspace_path
        camset_resolved = resolve_phase3_camset_artifact(chosen, ws) if ws is not None else None
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
        self._notebook.setCurrentWidget(self._diagnostics_tab)
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
        top_btn_row.addWidget(make_orange_button("▲ Bundle Settings", self._go_to_settings))
        top_btn_row.addStretch()
        root.addLayout(top_btn_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        left = QWidget()
        left.setMinimumWidth(180)
        left.setMaximumWidth(280)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)

        self._run_selector = RunSelectorWidget(runs=[])
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
        visual_btn_row.addStretch()
        visual_layout.addLayout(visual_btn_row)
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

    def _on_selection_changed(self, runs: list[dict]) -> None:
        self._run_selector.enforce_max_selection(4)
        selected = self._run_selector.get_selected()
        self._render_summary(selected)
        self._render_initial_plot(selected)
        self._render_residuals(selected)

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
                camset_resolved = resolve_phase3_camset_artifact(chosen, ws) if ws is not None else None
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
        self._initial_layout.addWidget(canvas)

    def _create_phase3_run_from_threshold(
        self,
        source_run: dict,
        threshold: float,
        mad_threshold: float,
        image_indices: list[int],
        on_done_cb,
    ) -> None:
        """Run Phase 3 with globally filtered detections (by image index) in a background thread."""
        ws = self._workspace_mgr.workspace_path
        if ws is None:
            QMessageBox.critical(self, "Workspace", "No active workspace.")
            on_done_cb()
            return

        def work_fn(emit) -> dict:
            try:
                import dill as _pkl
            except ImportError:
                import pickle as _pkl

            run_id = make_run_id()
            run_dir = ws / "phase3_runs" / run_id
            ensure_directory(run_dir)
            diagnostics: dict = {}

            try:
                src_artifacts = source_run.get("artifacts") or {}
                src_inputs = source_run.get("inputs") or {}
                src_params = source_run.get("params") or {}

                # ── Resolve Phase 1 detection pickle ───────────────────
                p1_pickle_str = src_artifacts.get("phase1_detection_pickle_used")
                if p1_pickle_str and path_exists(p1_pickle_str):
                    p1_pickle = Path(p1_pickle_str)
                else:
                    p1_runs = self._workspace_mgr.load_runs("phase1")
                    p1_run_id = src_inputs.get("phase1_run_id")
                    p1_run = next((r for r in p1_runs if r.get("run_id") == p1_run_id), None)
                    if p1_run is None and p1_runs:
                        p1_run = p1_runs[-1]
                    p1_pickle = resolve_phase1_pickle_artifact(p1_run, ws) if p1_run else None

                if p1_pickle is None or not path_exists(p1_pickle):
                    raise RuntimeError("Could not resolve Phase 1 detected_datapoints.pickle.")

                # ── Resolve Phase 2 initial camset ─────────────────────
                camset_path_str = src_artifacts.get("phase2_initial_camset_used")
                if not camset_path_str or not path_exists(camset_path_str):
                    p2_run_id = src_inputs.get("phase2_run_id")
                    p2_runs = self._workspace_mgr.load_runs("phase2")
                    p2_run = next((r for r in p2_runs if r.get("run_id") == p2_run_id), None)
                    if p2_run is None and p2_runs:
                        p2_run = p2_runs[-1]
                    if p2_run:
                        camset_path_str = (p2_run.get("artifacts") or {}).get("initial_camset")

                if not camset_path_str or not path_exists(camset_path_str):
                    raise RuntimeError("Could not resolve Phase 2 initial camset.")

                camset_path = Path(camset_path_str)

                emit(f"Loading Phase 1 detections: {p1_pickle}")
                payload = load_pickle(as_io_path(p1_pickle))
                detections = extract_detection(payload)
                if detections is None:
                    raise RuntimeError("Could not extract TargetDetection from Phase 1 pickle.")

                # ── Filter detections (global by image index) ───────────
                emit(f"Removing {len(image_indices)} image(s) via global_im_num filter…")
                filtered_det = detections.delete_row(global_im_num=image_indices)
                emit("Filtering complete.")

                # ── Save filtered detection pickle ──────────────────────
                filt_pickle_path = run_dir / "filtered_detected_datapoints.pickle"
                with open(filt_pickle_path, "wb") as fh:
                    _pkl.dump(filtered_det, fh)
                emit(f"Saved filtered detections: {filt_pickle_path}")

                # ── Load camset and run Phase 3 ─────────────────────────
                emit(f"Loading Phase 2 camset: {camset_path}")
                cams = load_CameraSet(as_io_path(camset_path))
                target = build_target(
                    src_params.get("target_type", "Ccube"),
                    src_params.get("n_points", 6),
                    src_params.get("length", 30.0),
                    border_fraction=src_params.get("border_fraction", 0.1),
                    marker_fraction=src_params.get("marker_fraction", 0.8),
                )
                problem_options = dict(src_params.get("problem_options") or {})
                threads = src_params.get("threads", 1)
                lockbox_params = dict(src_params.get("lockbox") or {})
                lockbox_config = CameraLockboxConfig(
                    enabled=bool(lockbox_params.get("enabled", False)),
                    rotation_half_width=float(lockbox_params.get("rotation_half_width", 0.1)),
                    translation_half_width=float(lockbox_params.get("translation_half_width", 0.1)),
                    rotation_sigma=float(lockbox_params.get("rotation_sigma", 0.05)),
                    translation_sigma=float(lockbox_params.get("translation_sigma", 0.01)),
                    center_sigma=float(lockbox_params.get("center_sigma", 0.0)),
                )
                lockbox_source_camset = None
                lockbox_source_path = lockbox_params.get("source_camset")
                if lockbox_config.enabled:
                    if not lockbox_source_path:
                        raise RuntimeError("Lockbox is enabled but no effective lockbox source path was provided.")
                    emit(f"Loading effective lockbox source camset: {lockbox_source_path}")
                    if lockbox_params.get("original_source_camset") and lockbox_params.get("edited_source_camset"):
                        emit(f"Original source camset: {lockbox_params.get('original_source_camset')}")
                        emit(f"Edited lockbox copy: {lockbox_params.get('edited_source_camset')}")
                    lockbox_source_camset = load_CameraSet(as_io_path(lockbox_source_path))
                    if set(lockbox_source_camset.get_names()) != set(cams.get_names()):
                        raise RuntimeError(
                            "Effective lockbox source camera names do not match the active Phase 2 camset. "
                            "This would break TemplateBundleHandler."
                        )

                handler = TemplateBundleHandler(
                    camset=cams,
                    target=target,
                    detection=filtered_det,
                    fixed_params=src_params.get("fixed_params"),
                    options=problem_options,
                    lockbox_config=lockbox_config,
                    lockbox_source_camset=lockbox_source_camset,
                    lockbox_warm_start=bool(lockbox_params.get("warm_start", True)),
                )

                stream = EmitStream(emit)
                log_handler = EmitLogHandler(emit)
                log_handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
                root_logger = logging.getLogger()
                root_logger.addHandler(log_handler)
                try:
                    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream), suppress_matplotlib_gui():
                        optimisation, out_cams, stats = run_bundle_adjustment_with_stats(
                            handler, threads=threads
                        )
                finally:
                    root_logger.removeHandler(log_handler)

                init_euclid = float(stats.get("initial_euclid", float("nan")))
                final_euclid = float(stats.get("final_euclid", float("nan")))
                emit(f"Initial Euclidean error: {init_euclid:.4f} px")
                emit(f"Final Euclidean error: {final_euclid:.4f} px")
                if not bool(stats.get("success", optimisation.success)):
                    emit(f"Solver note: {stats.get('message', optimisation.message)}")

                camset_out_path = run_dir / "optimised_cameras.camset"
                out_cams.save(camset_out_path)

                missing_before = np.array(
                    getattr(handler, "missing_poses_before_outlier_rejection", []), dtype=bool
                )
                missing_after = np.array(
                    getattr(
                        handler,
                        "missing_poses_after_outlier_rejection",
                        handler.missing_poses if handler.missing_poses is not None else [],
                    ),
                    dtype=bool,
                )
                per_im_init = np.array(getattr(handler, "initial_per_im_error", []), dtype=float)

                per_cam_err: dict = {}
                try:
                    dd = np.asarray(handler.get_detection_data(flatten=True))
                    residual_xy = np.reshape(np.asarray(optimisation.fun, dtype=float), (-1, 2))
                    residual_norm = np.linalg.norm(residual_xy, axis=1)
                    if dd.ndim == 2 and dd.shape[1] >= 1:
                        cam_idx = dd[:, 0].astype(int)
                        if cam_idx.size != residual_norm.size:
                            n = min(cam_idx.size, residual_norm.size)
                            cam_idx = cam_idx[:n]
                            residual_norm = residual_norm[:n]
                        for idx, name in enumerate(handler.cam_names):
                            mask = cam_idx == idx
                            per_cam_err[name] = (
                                float(np.mean(residual_norm[mask])) if np.any(mask) else float("nan")
                            )
                except Exception as _de:
                    emit(f"Warning: D3.12 skipped: {_de}")

                n_missing_before = int(np.sum(missing_before))
                n_missing_after = int(np.sum(missing_after))
                param_count = int(stats.get("param_count", 0))
                obs_count = int(stats.get("observation_count", len(optimisation.fun) // 2))

                diagnostics["D3.1_n_missing_poses"] = n_missing_before
                diagnostics["D3.2_n_outlier_removed"] = int(max(0, n_missing_after - n_missing_before))
                diagnostics["D3.3_per_image_initial_reprojection"] = per_im_init.tolist()
                diagnostics["D3.4_initial_error_plot"] = "rendered in diagnostics tab"
                diagnostics["D3.5_initial_euclid_px"] = init_euclid
                diagnostics["D3.6_final_euclid_px"] = final_euclid
                diagnostics["D3.7_error_reduction_ratio"] = (
                    float(init_euclid / final_euclid) if final_euclid > 0 else float("inf")
                )
                diagnostics["D3.8_solver_status"] = {
                    "status": int(stats.get("status", optimisation.status)),
                    "message": str(stats.get("message", optimisation.message)),
                    "success": bool(stats.get("success", optimisation.success)),
                }
                diagnostics["D3.9_nfev"] = int(stats.get("nfev", optimisation.nfev))
                diagnostics["D3.10_parameter_observation_ratio"] = {
                    "param_count": param_count,
                    "observation_count": obs_count,
                    "ratio": float(param_count / max(obs_count, 1)),
                }
                diagnostics["D3.11_residual_xy_scatter"] = "rendered in diagnostics tab"
                diagnostics["D3.12_per_camera_mean_reprojection"] = per_cam_err
                diagnostics["D3.13_extrinsic_pose_view"] = "rendered in diagnostics tab"

                metadata = {
                    "run_id": run_id,
                    "phase": "phase3",
                    "params": dict(src_params),
                    "diagnostics": diagnostics,
                    "error": None,
                    "inputs": {
                        "phase2_run_id": src_inputs.get("phase2_run_id"),
                        "phase1_run_id": src_inputs.get("phase1_run_id"),
                        "phase3_run_id": source_run.get("run_id"),
                    },
                    "artifacts": {
                        "optimised_camset": str(camset_out_path),
                        "phase2_initial_camset_used": str(camset_path),
                        "phase1_detection_pickle_used": str(p1_pickle),
                        "filtered_detection_pickle": str(filt_pickle_path),
                    },
                    "threshold_pruning": {
                        "source_phase3_run_id": source_run.get("run_id"),
                        "phase2_run_id": src_inputs.get("phase2_run_id"),
                        "phase1_run_id": src_inputs.get("phase1_run_id"),
                        "user_threshold_px": threshold,
                        "mad_threshold_px": mad_threshold,
                        "removed_global_image_indices": image_indices,
                        "n_removed": len(image_indices),
                        "removal_mode": "global via global_im_num",
                    },
                }
                self._workspace_mgr.save_run("phase3", run_id, metadata)
                emit(f"New Phase 3 run saved: {run_id}")
                return metadata

            except Exception as exc:
                err = str(exc)
                emit(f"ERROR: {err}")
                metadata = {
                    "run_id": run_id,
                    "phase": "phase3",
                    "params": dict(src_params),
                    "diagnostics": diagnostics,
                    "error": err,
                    "inputs": {
                        "phase2_run_id": (source_run.get("inputs") or {}).get("phase2_run_id"),
                        "phase1_run_id": (source_run.get("inputs") or {}).get("phase1_run_id"),
                        "phase3_run_id": source_run.get("run_id"),
                    },
                }
                self._workspace_mgr.save_run("phase3", run_id, metadata)
                return metadata

        self._threshold_worker = PhaseWorker(work_fn, parent=self)

        # Connect line_ready to the Phase 3 settings tab terminal so output
        # is visible when the user navigates back to that tab.
        for _i in range(self._notebook.count()):
            if self._notebook.tabText(_i) == TAB_PHASE3:
                _settings_tab = self._notebook.widget(_i)
                if hasattr(_settings_tab, "_terminal"):
                    self._threshold_worker.line_ready.connect(_settings_tab._terminal.append_line)
                break

        def _on_finished(metadata: dict) -> None:
            on_done_cb()
            if metadata.get("error"):
                QMessageBox.critical(
                    self, "Phase 3 failed",
                    f"New run encountered an error:\n{metadata['error']}"
                )
            else:
                new_id = metadata.get("run_id", "?")
                n_rem = (metadata.get("threshold_pruning") or {}).get("n_removed", len(image_indices))
                QMessageBox.information(
                    self, "New Phase 3 run created",
                    f"Run ID: {new_id}\n"
                    f"Removed {n_rem} image(s) with initial RPE > {threshold:.4f} px."
                )
            self.refresh()

        self._threshold_worker.finished.connect(_on_finished)
        self._threshold_worker.start()

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
            )
        )

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

        camset_path = run.get("artifacts", {}).get("optimised_camset")
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
        self._poses_layout.addWidget(FigureCanvasQTAgg(fig))

    def _on_backend_changed(self, btn) -> None:
        """Handle backend selector toggle — update Open3D output visibility."""
        if self._open3d_cb.isChecked():
            self._open3d_output.setText("Click Assess Calibration to open an interactive Open3D window.")
            self._open3d_output.setVisible(True)
        else:
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
            # Pass output_widget=None so visualise_calibration_open3d opens a
            # separate native Open3D window instead of attempting offscreen
            # rendering (which fails on Windows due to missing EGL support).
            ok, msg = launch_visualise_calibration_open3d_for_run(chosen, output_widget=None)
        else:
            ok, msg = launch_visualise_calibration_for_run(chosen)
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
            ok, msg = launch_save_pyvista_png_for_run(chosen, Path(path))
            if ok:
                QMessageBox.information(self, "Save PNG", msg)
            else:
                QMessageBox.warning(self, "Save PNG", f"Could not save PNG:\n{msg}")

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

