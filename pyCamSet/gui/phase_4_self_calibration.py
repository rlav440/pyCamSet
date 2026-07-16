"""
Phase 4 - Self-calibration GUI.

Implements Phase 4 using SelfBundleHandler with an
explicit user-triggered flow and diagnostics, including Assess Calibration.
"""
from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pyCamSet.gui.shared_functions import (
    as_io_path,
    IMAGE_FOLDER_SCHEMATIC,
    TAB_PHASE4,
    CollapsibleSection,
    EmitLogHandler,
    EmitStream,
    PhaseWorker,
    RunSelectorWidget,
    TerminalWidget,
    WorkspaceManager,
    make_blue_button,
    make_green_button,
    make_orange_button,
    make_run_id,
    make_section_label,
    make_separator,
    ensure_directory,
    path_exists,
    render_predecessor_chain_section,
    resolve_phase3_camset_artifact,
)
from pyCamSet.gui.assess_calibration import (
    launch_visualise_calibration_for_run,
    launch_visualise_calibration_open3d_for_run,
    launch_save_pyvista_png_for_run,
    merge_phase3_phase4_runs,
    select_latest_visualisation_run,
)

try:
    from pyCamSet.optimisation.optimisation_handling import run_bundle_adjustment_with_stats
    from pyCamSet.optimisation.standard_bundle_handler import SelfBundleHandler
    from pyCamSet.utils.saving import load_CameraSet

    _PYCAMSET_OK = True
except ImportError:
    run_bundle_adjustment_with_stats = None
    SelfBundleHandler = None
    load_CameraSet = None
    _PYCAMSET_OK = False

_TARGET_CHOICES = ["Ccube", "ChArUco"]


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
        form_root.setContentsMargins(0, 0, 0, 0)
        form_root.setSpacing(4)
        top_row.addWidget(form_widget, stretch=1)

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
        p3cam_btn.setFixedWidth(70)
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
        self._source_lbl.setStyleSheet("color: #666;")
        paths_sect.addRow("", self._source_lbl)

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
        self._max_nfev_spin.setValue(300)
        self._max_nfev_spin.setFixedWidth(110)
        self._max_nfev_spin.setToolTip(
            "Concept: maximum cost-function evaluations for the solver.\n\n"
            "Default: 300\n"
            "Range: 5–5000\n"
            "Guidance: increase to 1000 if the solver reports non-convergence.\n"
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
        run_btn = make_blue_button("▶  Run Phase 4", self._run_phase4)
        run_btn.setToolTip("Run self-calibration.")
        btn_row.addWidget(run_btn)
        btn_row.addWidget(make_orange_button("Diagnostics ▼", self._open_diagnostics))
        btn_row.addWidget(make_green_button("Assess Calibration", self._open_assess_calibration))
        btn_row.addStretch()
        form_root.addLayout(btn_row)
        form_root.addStretch()

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
                    payload = json.loads(handoff.read_text())
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
        resolved = resolve_phase3_camset_artifact(run, ws) if ws is not None else None
        p = str(resolved) if resolved is not None else (run.get("artifacts") or {}).get("optimised_camset")
        self._source_lbl.setText(f"Source: run {rid} -> {p or 'missing camset artifact'}")

    def _collect_params(self) -> Optional[dict]:
        floc = self._floc_edit.text().strip()
        if not floc:
            QMessageBox.critical(self, "Validation Error", "Image folder is required.")
            return None

        try:
            threads = int(self._threads_edit.text().strip() or "1")
            fixed_pose = int(self._fixed_pose_edit.text().strip())
            ref_cam = int(self._ref_cam_edit.text().strip())
            ref_pose = int(self._ref_pose_edit.text().strip())
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

        return {
            "f_loc": floc,
            "threads": threads,
            "fixed_params": fixed_params,
            "problem_options": {
                "verbosity": int(self._verbosity_spin.value()),
                "fixed_pose": fixed_pose,
                "ref_cam": ref_cam,
                "ref_pose": ref_pose,
                "outliers": outlier_mode,
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
        return resolve_phase3_camset_artifact(phase3_run, ws)

    def _run_phase4(self) -> None:
        params = self._collect_params()
        if params is None:
            return
        if not _PYCAMSET_OK:
            QMessageBox.critical(self, "Import error", "pyCamSet optimisation modules are unavailable.")
            return
        if self._worker is not None and self._worker.isRunning():
            if self._info_cb.isChecked():
                QMessageBox.information(self, "Phase 4 running", "Phase 4 is already running.")
            return

        self._sync_workspace_from_floc(params["f_loc"])
        if self._workspace_mgr.workspace_path is None:
            QMessageBox.critical(self, "Workspace", "Could not initialize workspace.")
            return

        phase3_run = self._load_phase3_run()
        phase3_camset = self._resolve_phase3_camset(phase3_run)
        if phase3_camset is None:
            if self._info_cb.isChecked():
                QMessageBox.information(self, "No Phase 3 source", "Select a valid Phase 3 run/camset first.")
            return

        selected_cameras = list(((phase3_run.get("params") or {}).get("selected_cameras") or []) if phase3_run else [])
        params["selected_cameras"] = selected_cameras

        self._terminal.clear_terminal()
        self._terminal.append_line("=== Phase 4: Self-Calibration ===")
        self._terminal.append_line(f"Image folder : {params['f_loc']}")
        self._terminal.append_line(f"Phase 3 run  : {phase3_run.get('run_id', 'unknown') if phase3_run else 'override'}")
        self._terminal.append_line(f"Camset       : {phase3_camset}")
        self._terminal.append_line(f"selected cams: {selected_cameras if selected_cameras else 'all'}")
        self._terminal.append_line("Starting…")

        def work_fn(emit: Callable[[str], None]) -> dict:
            ws_path = self._workspace_mgr.workspace_path
            assert ws_path is not None

            # Keep run ids compact to reduce path length pressure on Windows.
            run_id = make_run_id()
            run_dir = ws_path / "phase4_runs" / run_id
            ensure_directory(run_dir)
            diagnostics: dict = {}

            stream = EmitStream(emit)
            log_handler = EmitLogHandler(emit)
            log_handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
            root_logger = logging.getLogger()
            root_logger.addHandler(log_handler)

            try:
                with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                    prev_cams = load_CameraSet(as_io_path(phase3_camset))
                    selected = list(params.get("selected_cameras") or [])
                    if selected:
                        camset_names = set(prev_cams.get_names())
                        if camset_names != set(selected):
                            raise RuntimeError(
                                "Phase 3 camset cameras do not match selected camera subset. "
                                "Re-run Phase 3 with the same selected cameras."
                            )
                    prev_handler = getattr(prev_cams, "calibration_handler", None)
                    if prev_handler is None:
                        raise RuntimeError("Selected Phase 3 camset has no calibration handler metadata.")

                    handler = SelfBundleHandler(
                        camset=prev_cams,
                        target=prev_handler.target,
                        detection=prev_handler.detection,
                        fixed_params=params["fixed_params"],
                        options=params["problem_options"],
                    )
                    handler.set_from_templated_camset(prev_cams)

                    optimisation, out_cams, stats = run_bundle_adjustment_with_stats(handler, threads=params["threads"])  # type: ignore[arg-type]
                    init_euclid = float(stats.get("initial_euclid", float("nan")))
                    final_euclid = float(stats.get("final_euclid", float("nan")))
                    emit(f"D4.3  Initial Euclidean reprojection error: {init_euclid:.4f} px")
                    emit(f"D4.3  Final Euclidean reprojection error: {final_euclid:.4f} px")

                    p3_final = float((phase3_run.get("diagnostics") or {}).get("D3.6_final_euclid_px", float("nan"))) if phase3_run else float("nan")
                    d44 = float(p3_final - final_euclid) if np.isfinite(p3_final) else float("nan")
                    emit(f"D4.4  Improvement vs Phase 3 final error: {d44:.4f} px")

                    # Use a fixed short filename so long source folders do not exceed MAX_PATH.
                    out_path = run_dir / "self_calibrated_cameras.camset"
                    out_cams.save(out_path)

                    visible = np.array(getattr(handler, "visible_feature_mask", []), dtype=bool)
                    fixed_inds = list(getattr(handler, "fixed_inds", []))
                    updated_target = np.array(handler.get_updated_target(optimisation.x), dtype=float)
                    ref_target = np.array(handler.target.point_data, dtype=float).reshape(-1, 3)

                    with np.errstate(invalid="ignore", divide="ignore"):
                        ref_norm = np.linalg.norm(ref_target, axis=1)
                        upd_norm = np.linalg.norm(updated_target, axis=1)
                        ratio = ref_norm / np.where(upd_norm == 0.0, np.nan, upd_norm)
                    scale_est = float(np.nanmedian(ratio)) if ratio.size else float("nan")
                    displacement = np.linalg.norm(updated_target - ref_target, axis=1)
                    mean_disp_mm = float(np.nanmean(displacement) * 1000.0) if displacement.size else float("nan")

                    per_im_init = np.array(getattr(handler, "initial_per_im_error", []), dtype=float)
                    per_cam_err: dict[str, float] = {}
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
                            cam_names = list(getattr(handler, "cam_names", []))
                            for idx, name in enumerate(cam_names):
                                mask = cam_idx == idx
                                per_cam_err[name] = float(np.mean(residual_norm[mask])) if np.any(mask) else float("nan")
                    except Exception as diag_exc:
                        emit(f"Warning: D4.12 skipped due to diagnostics error: {diag_exc}")

                    diagnostics["D4.1_n_free_target_points"] = int(np.sum(visible))
                    diagnostics["D4.2_gauge_fixed_points"] = {"count": len(fixed_inds), "indices": fixed_inds}
                    diagnostics["D4.3_per_image_initial_reprojection"] = per_im_init.tolist()
                    diagnostics["D4.3_initial_euclid_px"] = init_euclid
                    diagnostics["D4.3_final_euclid_px"] = final_euclid
                    diagnostics["D4.4_vs_phase3_delta_px"] = d44
                    diagnostics["D4.5_gauge_scale_factor"] = scale_est
                    diagnostics["D4.7_mean_target_displacement_mm"] = mean_disp_mm
                    diagnostics["D4.8_shape_change_arrows"] = "available in Assess Calibration"
                    diagnostics["D4.9_planarity_rms_mm"] = "available in backend special_plots"
                    diagnostics["D4.10_accuracy_precision"] = "available in Assess Calibration"
                    diagnostics["D4.12_per_camera_mean_reprojection"] = per_cam_err

                    metadata = {
                        "run_id": run_id,
                        "phase": "phase4",
                        "params": params,
                        "diagnostics": diagnostics,
                        "error": None,
                        "inputs": {"phase3_run_id": phase3_run.get("run_id") if phase3_run else None},
                        "artifacts": {
                            "self_calibrated_camset": str(out_path),
                            "optimised_camset": str(out_path),
                            "phase3_camset_used": str(phase3_camset),
                        },
                    }
                    self._workspace_mgr.save_run("phase4", run_id, metadata)
                    emit(f"Run saved: {run_id}")
                    return metadata
            except Exception as exc:
                msg = str(exc)
                emit(f"ERROR: {msg}")
                metadata = {
                    "run_id": run_id,
                    "phase": "phase4",
                    "params": params,
                    "diagnostics": diagnostics,
                    "error": msg,
                    "inputs": {"phase3_run_id": phase3_run.get("run_id") if phase3_run else None},
                    "artifacts": {
                        "phase3_camset_used": str(phase3_camset),
                    },
                }
                self._workspace_mgr.save_run("phase4", run_id, metadata)
                return metadata
            finally:
                stream.flush()
                root_logger.removeHandler(log_handler)

        self._worker = PhaseWorker(work_fn, parent=self)
        self._worker.line_ready.connect(self._terminal.append_line)
        self._worker.finished.connect(self._on_run_finished)
        self._worker.error.connect(lambda msg: self._terminal.append_line(f"ERROR: {msg}"))
        self._worker.start()

    def _on_run_finished(self, _metadata: dict) -> None:
        if self._diagnostics_tab is not None:
            self._diagnostics_tab.refresh()

    def _open_diagnostics(self) -> None:
        if self._diagnostics_tab is not None:
            self._diagnostics_tab.refresh()
            self._notebook.setCurrentWidget(self._diagnostics_tab)

    def _open_assess_calibration(self) -> None:
        """Open the Phase 4 diagnostics tab and trigger Assess Calibration."""
        if self._diagnostics_tab is not None:
            self._diagnostics_tab.refresh()
            self._notebook.setCurrentWidget(self._diagnostics_tab)
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
        top_btn_row.addWidget(make_orange_button("▲ Self-Calibration Settings", self._go_to_settings))
        top_btn_row.addStretch()
        root.addLayout(top_btn_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        left = QWidget()
        left.setMinimumWidth(190)
        left.setMaximumWidth(320)
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
        visual_btn_row.addStretch()
        visual_layout.addLayout(visual_btn_row)
        # Shows which run/phase the most recent Assess Calibration click actually
        # resolved to -- lets a user comparing PyVista vs. Open3D (or comparing this
        # tab against Phase 3's own Assess Calibration tab) immediately see whether
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
            lbl.setStyleSheet("color: gray;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._summary_layout.addWidget(lbl)
            return

        for run in runs:
            phase = str(run.get("phase", "unknown"))
            hdr = QLabel(f"Run: {run.get('run_id', 'unknown')} ({phase})")
            hdr.setStyleSheet("font-weight: bold; margin-top: 8px;")
            self._summary_layout.addWidget(hdr)
            form = QFormLayout()
            form.setContentsMargins(16, 0, 0, 0)
            if phase != "phase4":
                form.addRow("Note:", QLabel("This run is from Phase 3; D4 metrics are not available."))
            else:
                d = run.get("diagnostics", {})
                form.addRow("D4.1 free target points:", QLabel(str(d.get("D4.1_n_free_target_points", "—"))))
                form.addRow("D4.2 gauge-fixed points:", QLabel(str(d.get("D4.2_gauge_fixed_points", "—"))))
                form.addRow("D4.3 initial euclid (px):", QLabel(f"{float(d.get('D4.3_initial_euclid_px', float('nan'))):.5f}"))
                form.addRow("D4.3 final euclid (px):", QLabel(f"{float(d.get('D4.3_final_euclid_px', float('nan'))):.5f}"))
                form.addRow("D4.4 vs Phase 3 delta (px):", QLabel(f"{float(d.get('D4.4_vs_phase3_delta_px', float('nan'))):.5f}"))
                form.addRow("D4.5 gauge scale factor:", QLabel(f"{float(d.get('D4.5_gauge_scale_factor', float('nan'))):.6f}"))
                form.addRow("D4.7 mean target displacement (mm):", QLabel(f"{float(d.get('D4.7_mean_target_displacement_mm', float('nan'))):.5f}"))
            if run.get("error"):
                form.addRow("Error:", QLabel(str(run["error"])))
            self._summary_layout.addLayout(form)
            self._summary_layout.addWidget(make_separator())
            render_predecessor_chain_section(self._summary_layout, self._workspace_mgr, run)
        self._summary_layout.addStretch()

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
        chosen = select_latest_visualisation_run(selected, self._all_runs)
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
            chosen = select_latest_visualisation_run(selected, self._all_runs)
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

