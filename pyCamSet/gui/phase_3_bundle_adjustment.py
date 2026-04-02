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
    render_predecessor_chain_section,
    resolve_phase1_pickle_artifact,
    suppress_matplotlib_gui,
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
    from pyCamSet.optimisation.template_handler import TemplateBundleHandler
    from pyCamSet.utils.saving import load_CameraSet, load_pickle

    _PYCAMSET_OK = True
except ImportError:
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

        self._src_lbl = QLabel("Inputs: latest Phase 2 + linked Phase 1")
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

        # ── Action buttons ─────────────────────────────────────────────
        form_root.addWidget(make_separator())
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

    def _browse_floc(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select image folder")
        if path:
            self._floc_edit.setText(path)

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

        return {
            "f_loc": floc,
            "threads": threads,
            "fixed_params": fixed_params,
            "target_type": self._target_combo.currentText(),
            "n_points": self._npts_spin.value(),
            "length": length,
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

    def set_selected_phase2_run_id(self, run_id: str) -> None:
        self.set_phase2_run_id(run_id)

    def set_phase2_run(self, run_id: str) -> None:
        self.set_phase2_run_id(run_id)

    def set_phase2_camset_path(self, path: str) -> None:
        self._preferred_phase2_camset_path = (path or "").strip() or None

    def set_camset_path(self, path: str) -> None:
        self.set_phase2_camset_path(path)

    def _load_phase2_run(self) -> Optional[dict]:
        runs = self._workspace_mgr.load_runs("phase2")
        if not runs:
            return None

        if self._preferred_phase2_run_id:
            for r in runs:
                if r.get("run_id") == self._preferred_phase2_run_id:
                    return r

        ws = self._workspace_mgr.workspace_path
        if ws is not None:
            handoff = ws / "handoff.json"
            if handoff.exists():
                try:
                    payload = json.loads(handoff.read_text())
                    if payload.get("phase") == "phase2" and payload.get("runs"):
                        wanted = payload["runs"][0].get("run_id")
                        for r in runs:
                            if r.get("run_id") == wanted:
                                return r
                except Exception:
                    pass
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

        self._src_lbl.setText(
            f"Phase 2: {phase2_run.get('run_id', '?')} | Phase 1: {phase1_run.get('run_id', '?')}"
        )

        self._terminal.clear_terminal()
        self._terminal.append_line("=== Phase 3: Template Bundle Adjustment ===")
        self._terminal.append_line(f"Image folder : {params['f_loc']}")
        self._terminal.append_line(f"Phase 2 run  : {phase2_run.get('run_id', 'unknown')}")
        self._terminal.append_line(f"Phase 1 run  : {phase1_run.get('run_id', 'unknown')}")
        self._terminal.append_line("Starting…")

        def work_fn(emit: Callable[[str], None]) -> dict:
            ws_path = self._workspace_mgr.workspace_path
            assert ws_path is not None

            run_id = make_run_id()
            run_dir = ws_path / "phase3_runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            diagnostics: dict = {}

            stream = EmitStream(emit)
            log_handler = EmitLogHandler(emit)
            log_handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
            root_logger = logging.getLogger()
            root_logger.addHandler(log_handler)

            try:
                with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream), suppress_matplotlib_gui():
                    emit("Phase 3 running in non-interactive plotting mode (thread-safe).")

                    camset_path = self._preferred_phase2_camset_path or phase2_run.get("artifacts", {}).get("initial_camset")
                    if not camset_path:
                        raise RuntimeError("Phase 2 run is missing initial_camset artifact.")
                    if not Path(camset_path).exists():
                        raise RuntimeError(f"Phase 2 camset path does not exist: {camset_path}")

                    p1_pickle = resolve_phase1_pickle_artifact(phase1_run, ws_path)
                    if not p1_pickle:
                        raise RuntimeError("Could not resolve Phase 1 detected_datapoints.pickle artifact.")

                    cams = load_CameraSet(Path(camset_path))
                    payload = load_pickle(Path(p1_pickle))
                    detections = extract_detection(payload)
                    if detections is None:
                        raise RuntimeError("Could not extract TargetDetection from Phase 1 pickle.")

                    target = build_target(params["target_type"], params["n_points"], params["length"])
                    handler = TemplateBundleHandler(
                        camset=cams,
                        target=target,
                        detection=detections,
                        fixed_params=params["fixed_params"],
                        options=params["problem_options"],
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
        camset_path = (chosen.get("artifacts") or {}).get("optimised_camset")

        self._workspace_mgr.write_handoff(
            {
                "phase": "phase3",
                "runs": [chosen],
                "image_folder": f_loc,
                "phase3_run_id": run_id,
                "optimised_camset": camset_path,
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
        self._open3d_cb.setToolTip("Use Open3D backend (renders embedded in GUI).")
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
                camset_path = (chosen.get("artifacts") or {}).get("optimised_camset")
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
        fig = Figure(figsize=(9, 4.8), tight_layout=True)
        ax = fig.add_subplot(111)
        ax.bar(np.arange(arr.size), arr, color="#1f77b4", alpha=0.88)
        ax.axhline(float(np.nanmedian(arr)) + 20.0, color="#d62728", linestyle="--", linewidth=1.1, label="median + 20")
        ax.set_title(f"D3.4 Per-image initial reprojection error ({run.get('run_id', '?')})")
        ax.set_xlabel("Image index")
        ax.set_ylabel("Initial error (aggregated px)")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
        self._initial_layout.addWidget(
            MatplotlibFigureCard(
                f"D3.4 Per-image initial reprojection error ({run.get('run_id', '?')})",
                fig,
                FigureCanvasQTAgg,
                parent=self._initial_widget,
                min_height=360,
            )
        )

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
        self._open3d_output.setVisible(self._open3d_cb.isChecked())

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
        if self._open3d_cb.isChecked():
            ok, msg = launch_visualise_calibration_open3d_for_run(chosen, self._open3d_output)
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
            pm = self._open3d_output.pixmap()
            if pm and not pm.isNull():
                pm.save(path, "PNG")
            else:
                QMessageBox.warning(self, "Save PNG", "No Open3D image rendered yet.")
        else:
            # PyVista: offscreen render to PNG.
            selected = self._run_selector.get_selected()
            chosen = select_latest_visualisation_run(selected, getattr(self, "_all_runs", []))
            if chosen is None:
                QMessageBox.warning(self, "Save PNG", "Select at least one run first.")
                return
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
        self._workspace_mgr.write_handoff({"phase": "phase3", "runs": [selected[0]]})
        if self._info_cb.isChecked():
            QMessageBox.information(self, "Handoff written", "handoff.json written for Phase 3.")

