"""
Phase 2 - Per-camera initial calibration (intrinsics) GUI.

Implements Phase 2 using existing pyCamSet functions:
- run_initial_calibration
- detect_datapoints_in_imfile (for optional high-distortion re-detection)
"""
from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Optional
import shutil

import cv2
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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
    TAB_PHASE2,
    TAB_PHASE3,
    CollapsibleSection,
    EmitLogHandler,
    EmitStream,
    MatplotlibFigureCard,
    PhaseWorker,
    RunSelectorWidget,
    TerminalWidget,
    WorkspaceManager,
    build_target,
    extract_detection_and_cam_res,
    make_blue_button,
    make_continue_button,
    make_orange_button,
    make_run_id,
    make_scrollable_tab,
    make_section_label,
    make_separator,
    render_predecessor_chain_section,
    resolve_phase1_pickle_artifact,
)

try:
    from pyCamSet.calibration.camera_calibrator import detect_datapoints_in_imfile, run_initial_calibration
    from pyCamSet.calibration_targets.abstract_target import get_keys
    from pyCamSet.utils.saving import load_CameraSet, load_pickle

    _PYCAMSET_OK = True
except ImportError:
    detect_datapoints_in_imfile = None
    get_keys = None
    run_initial_calibration = None
    load_CameraSet = None
    load_pickle = None
    _PYCAMSET_OK = False

_TARGET_CHOICES = ["Ccube", "ChArUco"]


def _normalise_per_view_series(payload: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-view arrays as (indices, rms_px, n_points, has_detection, valid_pose)."""
    if isinstance(payload, dict):
        indices = np.asarray(payload.get("image_indices", []), dtype=float).reshape(-1)
        rms = np.asarray(payload.get("rms_px", []), dtype=float).reshape(-1)
        n_points = np.asarray(payload.get("n_points", []), dtype=float).reshape(-1)
        has_detection = np.asarray(payload.get("has_detection", []), dtype=bool).reshape(-1)
        valid_pose = np.asarray(payload.get("valid_pose", []), dtype=bool).reshape(-1)
        if has_detection.size == 0 and indices.size:
            has_detection = ~np.isnan(rms)
        if valid_pose.size == 0 and indices.size:
            valid_pose = ~np.isnan(rms)
        return indices, rms, n_points, has_detection, valid_pose

    rms = np.asarray(payload if payload is not None else [], dtype=float).reshape(-1)
    indices = np.arange(rms.size, dtype=float)
    valid = ~np.isnan(rms)
    return indices, rms, np.full(rms.shape, np.nan, dtype=float), valid, valid


def _compute_true_per_view_reprojection(detections, calibration_target, cams) -> tuple[dict[str, dict], dict[str, float]]:
    """Compute true per-image RMS reprojection error for each camera/image index."""
    if get_keys is None:
        raise RuntimeError("pyCamSet detection helpers are unavailable.")

    d26_per_view: dict[str, dict] = {}
    d21_rms: dict[str, float] = {}
    max_ims = int(detections.max_ims)

    for cam_name in cams.get_names():
        cam = cams[cam_name]
        cam_detection = detections.get(cam=cam_name)
        cam_has_any = cam_detection.has_data()

        image_indices: list[int] = []
        rms_px: list[float] = []
        n_points: list[int] = []
        has_detection: list[bool] = []
        valid_pose: list[bool] = []

        weighted_sq_sum = 0.0
        total_points = 0

        for im_idx in range(max_ims):
            image_indices.append(im_idx)
            if not cam_has_any:
                rms_px.append(float("nan"))
                n_points.append(0)
                has_detection.append(False)
                valid_pose.append(False)
                continue

            im_detect = cam_detection.get(global_im_num=im_idx)
            data = im_detect.get_data()
            if data is None or len(data) == 0:
                rms_px.append(float("nan"))
                n_points.append(0)
                has_detection.append(False)
                valid_pose.append(False)
                continue

            has_detection.append(True)
            n_pts = int(data.shape[0])
            n_points.append(n_pts)

            try:
                pose = calibration_target.target_pose_in_cam_image(im_detect, cam, mode="nan")
            except Exception:
                pose = np.ones((4, 4), dtype=float) * np.nan

            pose_arr = np.asarray(pose, dtype=float)
            if pose_arr.shape != (4, 4) or np.any(np.isnan(pose_arr)):
                rms_px.append(float("nan"))
                valid_pose.append(False)
                continue

            valid_pose.append(True)
            keys = get_keys(data).astype(int)
            object_points = np.asarray(calibration_target.point_data[tuple(keys.T)], dtype=np.float32).reshape(-1, 3)
            image_points = np.asarray(data[:, -2:], dtype=np.float32).reshape(-1, 2)
            rvec, _ = cv2.Rodrigues(pose_arr[:3, :3].astype(np.float64))
            tvec = pose_arr[:3, 3].astype(np.float64)
            projected, _ = cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                np.asarray(cam.intrinsic, dtype=np.float64),
                np.asarray(cam.distortion_coefs, dtype=np.float64).reshape(-1),
            )
            projected = projected.reshape(-1, 2).astype(np.float32)
            sq_err = np.sum((projected - image_points) ** 2, axis=1)
            rms = float(np.sqrt(np.mean(sq_err))) if sq_err.size else float("nan")
            rms_px.append(rms)
            if sq_err.size:
                weighted_sq_sum += float(np.sum(sq_err))
                total_points += int(sq_err.size)

        d26_per_view[cam_name] = {
            "image_indices": image_indices,
            "rms_px": rms_px,
            "n_points": n_points,
            "has_detection": has_detection,
            "valid_pose": valid_pose,
        }
        d21_rms[cam_name] = float(np.sqrt(weighted_sq_sum / total_points)) if total_points else float("nan")

    return d26_per_view, d21_rms


def _make_grid_image(width: int, height: int, step: int = 48) -> np.ndarray:
    """Create a synthetic BGR grid image for undistortion diagnostics."""
    w = max(32, int(width))
    h = max(32, int(height))
    s = max(12, int(step))

    img = np.full((h, w, 3), 245, dtype=np.uint8)

    # Minor grid
    for x in range(0, w, s):
        cv2.line(img, (x, 0), (x, h - 1), (205, 205, 205), 1, lineType=cv2.LINE_AA)
    for y in range(0, h, s):
        cv2.line(img, (0, y), (w - 1, y), (205, 205, 205), 1, lineType=cv2.LINE_AA)

    # Major grid every 4 cells
    major = s * 4
    for x in range(0, w, major):
        cv2.line(img, (x, 0), (x, h - 1), (150, 150, 150), 2, lineType=cv2.LINE_AA)
    for y in range(0, h, major):
        cv2.line(img, (0, y), (w - 1, y), (150, 150, 150), 2, lineType=cv2.LINE_AA)

    # Center guides
    cx, cy = w // 2, h // 2
    cv2.line(img, (cx, 0), (cx, h - 1), (70, 120, 255), 2, lineType=cv2.LINE_AA)
    cv2.line(img, (0, cy), (w - 1, cy), (70, 120, 255), 2, lineType=cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(6, s // 4), (40, 40, 220), 2, lineType=cv2.LINE_AA)

    return img


class _SuppressDtctConfigFilter(logging.Filter):
    """Suppress repeated non-fatal dtct_config warnings from diagnostics camset loading."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "Failed to load detections with reason 'dtct_config'" not in msg


class Phase2Tab(QWidget):
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
        self._diagnostics_tab: Optional["Phase2DiagnosticsTab"] = None
        self._worker: Optional[PhaseWorker] = None
        self._build_ui(terminal_cb)
        self._refresh_phase1_sources()

    def set_diagnostics_tab(self, tab: "Phase2DiagnosticsTab") -> None:
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
        side.setFixedWidth(200)
        side_layout = QVBoxLayout(side)
        side_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        top_row.addWidget(side)

        # ── Paths (collapsible) ────────────────────────────────────────
        paths_sect = CollapsibleSection("Paths", expanded=False)
        form_root.addWidget(paths_sect)

        floc_row = QHBoxLayout()
        self._floc_edit = QLineEdit()
        self._floc_edit.setPlaceholderText("Root folder with per-camera sub-folders")
        self._floc_edit.setToolTip(
            "Concept: root image directory used for calibration.\n"
            "Default: propagated from Phase 0 if available; otherwise empty."
        )
        self._floc_edit.textChanged.connect(self._sync_workspace_from_floc)
        floc_btn = QPushButton("Browse…")
        floc_btn.setFixedWidth(70)
        floc_btn.setToolTip("Choose the image root folder manually.")
        floc_btn.clicked.connect(self._browse_floc)
        floc_row.addWidget(self._floc_edit)
        floc_row.addWidget(floc_btn)
        paths_sect.addRow("Image folder (f_loc):", floc_row)

        src_row = QHBoxLayout()
        self._phase1_run_combo = QComboBox()
        self._phase1_run_combo.setToolTip(
            "Concept: Phase 1 detection run used as input to Phase 2.\n"
            "Default: Auto (handoff selection from Phase 1, else latest run)."
        )
        self._phase1_run_combo.currentIndexChanged.connect(self._on_phase1_source_changed)
        src_refresh_btn = QPushButton("Refresh")
        src_refresh_btn.setFixedWidth(70)
        src_refresh_btn.setToolTip("Reload available Phase 1 runs from workspace.")
        src_refresh_btn.clicked.connect(self._refresh_phase1_sources)
        src_row.addWidget(self._phase1_run_combo)
        src_row.addWidget(src_refresh_btn)
        paths_sect.addRow("Phase 1 run:", src_row)

        det_row = QHBoxLayout()
        self._det_pickle_edit = QLineEdit()
        self._det_pickle_edit.setPlaceholderText("Optional override: detected_datapoints.pickle")
        self._det_pickle_edit.setToolTip(
            "Concept: manual detection artifact override.\n"
            "Default: blank (auto-resolve from selected/auto Phase 1 run)."
        )
        self._det_pickle_edit.textChanged.connect(lambda _: self._update_detection_source_label())
        det_browse_btn = QPushButton("Browse…")
        det_browse_btn.setFixedWidth(70)
        det_browse_btn.setToolTip("Select a specific detected_datapoints.pickle file.")
        det_browse_btn.clicked.connect(self._browse_detection_pickle)
        det_clear_btn = QPushButton("Clear")
        det_clear_btn.setFixedWidth(55)
        det_clear_btn.setToolTip("Clear manual override and return to auto source resolution.")
        det_clear_btn.clicked.connect(lambda: self._det_pickle_edit.setText(""))
        det_row.addWidget(self._det_pickle_edit)
        det_row.addWidget(det_browse_btn)
        det_row.addWidget(det_clear_btn)
        paths_sect.addRow("Detection source path:", det_row)

        self._phase1_lbl = QLabel("Detection source: auto")
        self._phase1_lbl.setStyleSheet("color: #666;")
        self._phase1_lbl.setWordWrap(True)
        paths_sect.addRow("", self._phase1_lbl)

        # ── Calibration Target (collapsible) ───────────────────────────
        form_root.addWidget(make_separator())
        target_sect = CollapsibleSection("Calibration Target", expanded=False)
        form_root.addWidget(target_sect)

        self._target_combo = QComboBox()
        self._target_combo.addItems(_TARGET_CHOICES)
        self._target_combo.setFixedWidth(140)
        self._target_combo.setToolTip(
            "Concept: calibration target family.\n"
            "Default: Ccube."
        )
        target_sect.addRow("Target type:", self._target_combo)

        self._npts_spin = QSpinBox()
        self._npts_spin.setRange(2, 30)
        self._npts_spin.setValue(6)
        self._npts_spin.setFixedWidth(90)
        self._npts_spin.setToolTip(
            "Concept: target discretization (points/squares along x).\n"
            "Default: 6."
        )
        target_sect.addRow("n_points / squares_x:", self._npts_spin)

        self._length_edit = QLineEdit("30.0")
        self._length_edit.setFixedWidth(110)
        self._length_edit.setToolTip(
            "Concept: physical target size parameter in millimetres.\n"
            "Default: 30.0 mm."
        )
        target_sect.addRow("Length / square size (mm):", self._length_edit)

        # ── Initial Calibration Options ────────────────────────────────
        form_root.addWidget(make_separator())
        form_root.addWidget(make_section_label("Initial Calibration Options"))

        opts_form = QFormLayout()
        opts_form.setContentsMargins(0, 0, 0, 0)
        opts_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form_root.addLayout(opts_form)

        self._cache_cb = QCheckBox("Use detection cache when fallback-detecting")
        self._cache_cb.setChecked(True)
        self._cache_cb.setToolTip(
            "Concept: reuse cached detections during fallback detection.\n"
            "Default: enabled."
        )
        opts_form.addRow(self._cache_cb)

        self._hd_cb = QCheckBox("High Distortion Mode (2c re-detection)")
        self._hd_cb.setToolTip(
            "Concept: perform an extra detection+calibration pass using initial intrinsics.\n"
            "Default: disabled."
        )
        opts_form.addRow(self._hd_cb)

        self._nlim_edit = QLineEdit()
        self._nlim_edit.setPlaceholderText("blank = no limit")
        self._nlim_edit.setFixedWidth(110)
        self._nlim_edit.setToolTip(
            "Concept: cap number of images processed per camera.\n"
            "Default: blank (no limit)."
        )
        opts_form.addRow("Max images per camera (n_lim):", self._nlim_edit)

        self._fp_edit = QLineEdit()
        self._fp_edit.setPlaceholderText('e.g. {"cam0": "int"}')
        self._fp_edit.setToolTip(
            "Concept: JSON map of fixed camera parameters during optimization.\n"
            "Default: blank (no fixed params)."
        )
        opts_form.addRow("Fixed params (JSON):", self._fp_edit)

        # ── Action buttons ─────────────────────────────────────────────
        form_root.addWidget(make_separator())
        btn_row = QHBoxLayout()
        run_btn = make_blue_button("▶  Run Phase 2", self._run_phase2)
        run_btn.setToolTip("Run per-camera initial intrinsics calibration.")
        btn_row.addWidget(run_btn)
        diag_btn = make_orange_button("Diagnostics ▼", self._open_diagnostics)
        diag_btn.setToolTip("Open Phase 2 diagnostics view.")
        btn_row.addWidget(diag_btn)
        btn_row.addWidget(make_continue_button(self._continue_to_next))
        btn_row.addStretch()
        form_root.addLayout(btn_row)
        form_root.addStretch()

        side_layout.addStretch()

        self._terminal = TerminalWidget(terminal_cb, parent=self)
        root.addWidget(self._terminal)

    def set_selected_phase1_run_id(self, run_id: str) -> None:
        """Select the given Phase 1 run in the combo box (called from Phase 1 tab)."""
        self._refresh_phase1_sources()
        idx = self._phase1_run_combo.findData(str(run_id))
        if idx >= 0:
            self._phase1_run_combo.setCurrentIndex(idx)

    def _browse_floc(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select image folder")
        if path:
            self._floc_edit.setText(path)

    def set_image_folder(self, path: str) -> None:
        self._floc_edit.setText(path)

    def _browse_detection_pickle(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select detected_datapoints.pickle",
            "",
            "Pickle files (*.pickle *.pkl);;All files (*)",
        )
        if path:
            self._det_pickle_edit.setText(path)

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
        self._refresh_phase1_sources()

    def _resolve_handoff_phase1_run_id(self, runs: list[dict]) -> Optional[str]:
        ws = self._workspace_mgr.workspace_path
        if ws is None:
            return None
        handoff = ws / "handoff.json"
        if not handoff.exists():
            return None
        try:
            payload = json.loads(handoff.read_text())
            if payload.get("phase") == "phase1" and payload.get("runs"):
                wanted = payload["runs"][0].get("run_id")
                if wanted and any(r.get("run_id") == wanted for r in runs):
                    return str(wanted)
        except Exception:
            pass
        return None

    def _refresh_phase1_sources(self) -> None:
        runs = self._workspace_mgr.load_runs("phase1")
        keep = self._phase1_run_combo.currentData() if hasattr(self, "_phase1_run_combo") else None
        self._phase1_run_combo.blockSignals(True)
        self._phase1_run_combo.clear()
        self._phase1_run_combo.addItem("Auto (handoff else latest)", None)
        for r in runs:
            rid = r.get("run_id", "unknown")
            self._phase1_run_combo.addItem(str(rid), str(rid))

        if keep is not None:
            idx = self._phase1_run_combo.findData(keep)
            self._phase1_run_combo.setCurrentIndex(idx if idx >= 0 else 0)
        else:
            self._phase1_run_combo.setCurrentIndex(0)

        self._phase1_run_combo.blockSignals(False)
        self._update_detection_source_label()

    def _on_phase1_source_changed(self) -> None:
        self._update_detection_source_label()

    def _collect_params(self) -> Optional[dict]:
        floc = self._floc_edit.text().strip()
        if not floc:
            QMessageBox.critical(self, "Validation Error", "Image folder is required.")
            return None

        n_lim = None
        if self._nlim_edit.text().strip():
            try:
                n_lim = int(self._nlim_edit.text().strip())
            except ValueError:
                QMessageBox.critical(self, "Validation Error", "n_lim must be an integer.")
                return None

        try:
            length = float(self._length_edit.text().strip())
        except ValueError:
            QMessageBox.critical(self, "Validation Error", "Length must be numeric.")
            return None

        fixed_params = None
        if self._fp_edit.text().strip():
            try:
                fixed_params = json.loads(self._fp_edit.text().strip())
            except json.JSONDecodeError as exc:
                QMessageBox.critical(self, "Validation Error", f"Fixed params JSON: {exc}")
                return None

        return {
            "f_loc": floc,
            "caching": self._cache_cb.isChecked(),
            "high_distortion": self._hd_cb.isChecked(),
            "n_lim": n_lim,
            "fixed_params": fixed_params,
            "target_type": self._target_combo.currentText(),
            "n_points": self._npts_spin.value(),
            "length": length,
        }

    def _load_phase1_run(self) -> Optional[dict]:
        runs = self._workspace_mgr.load_runs("phase1")
        if not runs:
            return None

        selected_run_id = self._phase1_run_combo.currentData() if hasattr(self, "_phase1_run_combo") else None
        if selected_run_id:
            for r in runs:
                if r.get("run_id") == selected_run_id:
                    return r

        wanted = self._resolve_handoff_phase1_run_id(runs)
        if wanted:
            for r in runs:
                if r.get("run_id") == wanted:
                    return r

        return runs[-1]

    def _update_detection_source_label(self) -> None:
        override = self._det_pickle_edit.text().strip() if hasattr(self, "_det_pickle_edit") else ""
        if override:
            p = Path(override)
            self._phase1_lbl.setText(f"Detection source: override file ({'exists' if p.exists() else 'missing'})")
            return

        run = self._load_phase1_run()
        ws = self._workspace_mgr.workspace_path
        if run is None or ws is None:
            self._phase1_lbl.setText("Detection source: auto (no Phase 1 run found)")
            return

        det = resolve_phase1_pickle_artifact(run, ws)
        rid = run.get("run_id", "unknown")
        src = str(det) if det is not None else "no pickle (fallback detect)"
        self._phase1_lbl.setText(f"Detection source: run {rid} -> {src}")

    def _run_phase2(self) -> None:
        params = self._collect_params()
        if params is None:
            return
        if not _PYCAMSET_OK:
            QMessageBox.critical(self, "Import error", "pyCamSet calibration modules are unavailable.")
            return

        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Phase 2 running", "Phase 2 is already running.")
            return

        self._sync_workspace_from_floc(params["f_loc"])
        if self._workspace_mgr.workspace_path is None:
            QMessageBox.critical(self, "Workspace", "Could not initialize workspace.")
            return

        override_pickle_text = self._det_pickle_edit.text().strip()
        override_pickle = Path(override_pickle_text) if override_pickle_text else None

        phase1_run = self._load_phase1_run()
        if phase1_run is None and override_pickle is None:
            QMessageBox.information(self, "No Phase 1 run", "Run Phase 1 first or choose a detection pickle override.")
            return

        selected_cameras = []
        if phase1_run is not None:
            selected_cameras = list(((phase1_run.get("params") or {}).get("selected_cameras") or []))
        params["selected_cameras"] = selected_cameras

        self._update_detection_source_label()

        self._terminal.clear_terminal()
        self._terminal.append_line("=== Phase 2: Per-camera Initial Calibration ===")
        self._terminal.append_line(f"Image folder : {params['f_loc']}")
        self._terminal.append_line(f"Phase 1 run  : {phase1_run.get('run_id', 'none') if phase1_run else 'none'}")
        self._terminal.append_line(f"selected cams: {selected_cameras if selected_cameras else 'all'}")
        if override_pickle is not None:
            self._terminal.append_line(f"Override det : {override_pickle}")
        self._terminal.append_line("Starting…")

        def work_fn(emit: Callable[[str], None]) -> dict:
            diagnostics: dict = {}
            error_msg: Optional[str] = None
            ws_path = self._workspace_mgr.workspace_path
            assert ws_path is not None

            run_id = make_run_id()
            run_dir = ws_path / "phase2_runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            det_pickle = None
            if override_pickle is not None:
                if override_pickle.exists():
                    det_pickle = override_pickle
                    emit(f"Using override detections: {det_pickle}")
                else:
                    emit(f"Override path missing: {override_pickle} (falling back to Phase 1 source)")

            if det_pickle is None and phase1_run is not None:
                det_pickle = resolve_phase1_pickle_artifact(phase1_run, ws_path)

            detections = None
            cam_res = None

            stream = EmitStream(emit)
            log_handler = EmitLogHandler(emit)
            log_handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
            root_logger = logging.getLogger()
            root_logger.addHandler(log_handler)

            try:
                with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                    target = build_target(params["target_type"], params["n_points"], params["length"])
                    selected = list(params.get("selected_cameras") or [])
                    selected_set = set(selected)
                    f_loc = Path(params["f_loc"])

                    cam_folders = [
                        p for p in sorted(f_loc.iterdir())
                        if p.is_dir() and p.name != "sparse" and not p.name.startswith(".")
                    ]
                    if selected_set:
                        cam_folders = [p for p in cam_folders if p.name in selected_set]
                        if len(cam_folders) < 2:
                            raise RuntimeError("Need at least two selected camera folders for Phase 2.")

                    detect_root = f_loc
                    tmp_ctx: Optional[TemporaryDirectory] = None
                    root_entries = list(f_loc.iterdir())
                    allowed = {p.name for p in cam_folders}
                    has_extra_entries = any(p.name not in allowed for p in root_entries)
                    if has_extra_entries:
                        tmp_ctx = TemporaryDirectory(prefix="pycamset_phase2_")
                        detect_root = Path(tmp_ctx.name)
                        emit("2a  Using filtered staging folder (selected camera subfolders only).")
                        for cam in cam_folders:
                            dst = detect_root / cam.name
                            try:
                                dst.symlink_to(cam, target_is_directory=True)
                            except Exception:
                                shutil.copytree(cam, dst)

                    try:
                        if det_pickle is not None and det_pickle.exists():
                            emit(f"Using detections: {det_pickle}")
                            payload = load_pickle(det_pickle)
                            detections, cam_res = extract_detection_and_cam_res(payload)
                            if selected_set:
                                det_cam_names = list(getattr(detections, "cam_names", []) or [])
                                if set(det_cam_names) != selected_set:
                                    emit(
                                        "Detection artifact camera set does not match selected cameras; "
                                        "running fresh detection on selected subset."
                                    )
                                    detections = None
                                    cam_res = None

                        if detections is None or cam_res is None:
                            emit("Detection artifact missing/incompatible, falling back to detection pass.")
                            detections, cam_res = detect_datapoints_in_imfile(
                                f_loc=detect_root,
                                calibration_target=target,
                                caching=params["caching"],
                                draw=False,
                                n_lim=params["n_lim"],
                            )

                        cams, _, per_im = run_initial_calibration(
                            detection=detections,
                            calibration_target=target,
                            cam_res=cam_res,
                            save=False,
                            fixed_params=params["fixed_params"],
                            return_poses_and_costs=True,
                        )
                        emit("2b  Initial calibration completed.")

                        if params["high_distortion"]:
                            emit("2c  High-distortion mode: re-running detection with initial intrinsics…")
                            detections_hd, _ = detect_datapoints_in_imfile(
                                f_loc=detect_root,
                                calibration_target=target,
                                caching=False,
                                draw=False,
                                n_lim=params["n_lim"],
                                camset=cams,
                            )
                            cams, _, per_im = run_initial_calibration(
                                detection=detections_hd,
                                calibration_target=target,
                                cam_res=cam_res,
                                save=False,
                                fixed_params=params["fixed_params"],
                                return_poses_and_costs=True,
                            )
                            detections = detections_hd
                            emit("2c  High-distortion refinement completed.")
                    finally:
                        if tmp_ctx is not None:
                            tmp_ctx.cleanup()

                    camset_path = run_dir / (
                        "initial_cameras_high_distortion.camset"
                        if params["high_distortion"]
                        else "initial_cameras.camset"
                    )
                    cams.save(camset_path)

                    d21_rms = {}
                    d22_intr = {}
                    d23_dst = {}
                    cam_names = list(cams.get_names())

                    d26_per_view, d21_rms = _compute_true_per_view_reprojection(detections, target, cams)

                    for cam_name, cam in zip(cam_names, cams):

                        intr = np.array(cam.intrinsic)
                        d22_intr[cam_name] = {
                            "fx": float(intr[0, 0]),
                            "fy": float(intr[1, 1]),
                            "cx": float(intr[0, 2]),
                            "cy": float(intr[1, 2]),
                            "res": np.array(cam.res).astype(float).tolist(),
                        }
                        dst = np.array(cam.distortion_coefs).reshape(-1)
                        d23_dst[cam_name] = {
                            "coeffs": dst.astype(float).tolist(),
                            "l2_norm": float(np.linalg.norm(dst)),
                        }

                    diagnostics["D2.1_per_camera_rms_reprojection"] = d21_rms
                    diagnostics["D2.2_intrinsics"] = d22_intr
                    diagnostics["D2.3_distortion"] = d23_dst
                    diagnostics["D2.5_intrinsic_stddev"] = "not available in current pyCamSet API"
                    diagnostics["D2.6_per_view_reprojection"] = d26_per_view
                    diagnostics["D2.7_per_view_error_plot"] = "rendered in diagnostics tab (true per-image reprojection RMS)"

                    emit("Diagnostics computed (D2.1-D2.7).")

                    metadata = {
                        "run_id": run_id,
                        "phase": "phase2",
                        "params": params,
                        "diagnostics": diagnostics,
                        "error": None,
                        "inputs": {
                            "phase1_run_id": phase1_run.get("run_id") if phase1_run else None,
                        },
                        "artifacts": {
                            "initial_camset": str(camset_path),
                            "phase1_detection_pickle": str(det_pickle) if det_pickle is not None else None,
                            "detection_source_override": str(override_pickle) if override_pickle is not None else None,
                        },
                    }
                    self._workspace_mgr.save_run("phase2", run_id, metadata)
                    emit(f"Run saved: {run_id}")
                    return metadata

            except Exception as exc:
                error_msg = str(exc)
                emit(f"ERROR: {error_msg}")
                metadata = {
                    "run_id": run_id,
                    "phase": "phase2",
                    "params": params,
                    "diagnostics": diagnostics,
                    "error": error_msg,
                    "inputs": {"phase1_run_id": phase1_run.get("run_id") if phase1_run else None},
                }
                self._workspace_mgr.save_run("phase2", run_id, metadata)
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

    def _continue_to_next(self) -> None:
        runs = self._workspace_mgr.load_runs("phase2")
        if not runs:
            if self._info_cb.isChecked():
                QMessageBox.information(self, "No runs", "Run Phase 2 first.")
            return

        chosen = runs[-1]
        f_loc = (chosen.get("params") or {}).get("f_loc")
        run_id = chosen.get("run_id")
        camset_path = (chosen.get("artifacts") or {}).get("initial_camset")

        self._workspace_mgr.write_handoff(
            {
                "phase": "phase2",
                "runs": [chosen],
                "image_folder": f_loc,
                "phase2_run_id": run_id,
                "initial_camset": camset_path,
                "selected_cameras": ((chosen.get("params") or {}).get("selected_cameras") or []),
            }
        )

        for i in range(self._notebook.count()):
            if self._notebook.tabText(i) != TAB_PHASE3:
                continue

            phase3_tab = self._notebook.widget(i)

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
                if not _try_call(phase3_tab, ("set_image_folder", "set_floc", "set_image_path"), str(f_loc)):
                    for attr in ("_floc_edit", "_image_folder_edit", "_f_loc_edit"):
                        if hasattr(phase3_tab, attr):
                            try:
                                getattr(phase3_tab, attr).setText(str(f_loc))
                                break
                            except Exception:
                                pass

            if run_id:
                if not _try_call(
                    phase3_tab,
                    ("set_phase2_run_id", "set_selected_phase2_run_id", "set_phase2_run"),
                    str(run_id),
                ):
                    for combo in phase3_tab.findChildren(QComboBox):
                        idx = combo.findData(str(run_id))
                        if idx < 0:
                            idx = combo.findText(str(run_id))
                        if idx >= 0:
                            try:
                                combo.setCurrentIndex(idx)
                                break
                            except Exception:
                                pass

            if camset_path:
                if not _try_call(phase3_tab, ("set_phase2_camset_path", "set_camset_path"), str(camset_path)):
                    for attr in ("_camset_edit", "_phase2_camset_edit"):
                        if hasattr(phase3_tab, attr):
                            try:
                                getattr(phase3_tab, attr).setText(str(camset_path))
                                break
                            except Exception:
                                pass

            self._notebook.setCurrentIndex(i)
            return


class Phase2DiagnosticsTab(QWidget):
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
        self._camset_cache: dict[str, tuple[object, Optional[str]]] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        self._threshold_worker = None  # PhaseWorker for threshold-based re-run
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        top_btn_row = QHBoxLayout()
        top_btn_row.addWidget(make_orange_button("▲ Intrinsics Settings", self._go_to_settings))
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
        right_layout.addWidget(make_section_label("Phase 2 Diagnostics"))
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
        self._sub_tabs.addTab(self._summary_widget, "Summary (D2.1-D2.3, D2.5)")

        self._per_view_widget, self._per_view_layout, self._per_view_scroll = make_scrollable_tab()
        self._sub_tabs.addTab(self._per_view_widget, "Per-view Errors (D2.6-D2.7)")

        self._distortion_widget, self._distortion_layout, self._distortion_scroll = make_scrollable_tab()
        self._sub_tabs.addTab(self._distortion_widget, "Distortion Field")

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(make_continue_button(self._continue_to_next))
        root.addLayout(btn_row)

        self.refresh()

    def refresh(self) -> None:
        runs = self._workspace_mgr.load_runs("phase2")
        self._run_selector.refresh(runs)
        chosen = self._run_selector.get_selected()
        self._render_summary(chosen)
        self._render_per_view(chosen)
        self._render_distortion(chosen)

    def _on_selection_changed(self, runs: list[dict]) -> None:
        self._run_selector.enforce_max_selection(4)
        chosen = self._run_selector.get_selected()
        self._render_summary(chosen)
        self._render_per_view(chosen)
        self._render_distortion(chosen)

    def _go_to_settings(self) -> None:
        for i in range(self._notebook.count()):
            if self._notebook.tabText(i) == TAB_PHASE2:
                self._notebook.setCurrentIndex(i)
                return

    @staticmethod
    def _clear_layout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                Phase2DiagnosticsTab._clear_layout(item.layout())

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
            d = run.get("diagnostics", {})
            hdr = QLabel(f"Run: {run.get('run_id', 'unknown')}")
            hdr.setStyleSheet("font-weight: bold; margin-top: 8px;")
            self._summary_layout.addWidget(hdr)

            form = QFormLayout()
            form.setContentsMargins(16, 0, 0, 0)

            d21 = d.get("D2.1_per_camera_rms_reprojection", {})
            for cam, val in d21.items():
                form.addRow(f"D2.1  {cam} RMS (px):", QLabel(f"{float(val):.4f}"))

            d22 = d.get("D2.2_intrinsics", {})
            for cam, iv in d22.items():
                fx = iv.get("fx", float("nan"))
                fy = iv.get("fy", float("nan"))
                cx = iv.get("cx", float("nan"))
                cy = iv.get("cy", float("nan"))
                form.addRow(f"D2.2  {cam} [fx fy cx cy]:", QLabel(f"[{fx:.2f}, {fy:.2f}, {cx:.2f}, {cy:.2f}]"))

            d23 = d.get("D2.3_distortion", {})
            for cam, dv in d23.items():
                form.addRow(f"D2.3  {cam} |k|:", QLabel(f"{float(dv.get('l2_norm', 0.0)):.6f}"))

            form.addRow("D2.5  intrinsic stddev:", QLabel(str(d.get("D2.5_intrinsic_stddev", "—"))))

            if run.get("error"):
                form.addRow("Error:", QLabel(str(run["error"])))

            self._summary_layout.addLayout(form)
            self._summary_layout.addWidget(make_separator())
            render_predecessor_chain_section(self._summary_layout, self._workspace_mgr, run)

        self._summary_layout.addStretch()

    def _render_per_view(self, runs: list[dict]) -> None:
        while self._per_view_layout.count():
            item = self._per_view_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

        if not runs:
            lbl = QLabel("Select a run to view per-image reprojection errors.")
            lbl.setStyleSheet("color: gray;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._per_view_layout.addWidget(lbl)
            return

        run = runs[-1]
        d = run.get("diagnostics", {})
        d26 = d.get("D2.6_per_view_reprojection", {})
        if not d26:
            self._per_view_layout.addWidget(QLabel("No D2.6 data in selected run."))
            return

        try:
            import matplotlib
            matplotlib.use("QtAgg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except ImportError:
            self._per_view_layout.addWidget(QLabel("matplotlib not available."))
            return

        # Compute initial threshold: mean + 2σ of all valid per-view RPE values
        all_rpe: list[float] = []
        for _cam, _vals in d26.items():
            _, _arr, _, _, _ = _normalise_per_view_series(_vals)
            all_rpe.extend(float(v) for v in _arr[~np.isnan(_arr)])
        rpe_arr = np.array(all_rpe, dtype=float) if all_rpe else np.array([1.0])
        init_thresh = float(np.nanmean(rpe_arr) + 2.0 * np.nanstd(rpe_arr))
        init_thresh = max(0.001, init_thresh)

        # Build figure
        fig = Figure(figsize=(9, 4.6), tight_layout=True)
        ax = fig.add_subplot(111)
        for cam, vals in d26.items():
            indices, arr, _, _, _ = _normalise_per_view_series(vals)
            if arr.size == 0:
                continue
            ax.plot(indices, arr, marker="o", linewidth=1.2, markersize=3, label=cam)

        hline = ax.axhline(
            init_thresh, color="#d62728", linestyle="--", linewidth=1.5,
            label=f"User threshold ({init_thresh:.3f} px)", zorder=5,
        )
        ax.set_title(f"D2.6/D2.7 per-image reprojection error ({run.get('run_id', '?')})")
        ax.set_xlabel("Image / view index")
        ax.set_ylabel("RMS reprojection error (px)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumHeight(320)

        # Helper: (camera, image_index) pairs whose RPE exceeds threshold
        def _pairs_above(thresh: float) -> list[tuple[str, int]]:
            out: list[tuple[str, int]] = []
            for _c, _v in d26.items():
                _inds, _arr2, _, _, _ = _normalise_per_view_series(_v)
                for _idx, _rpe in zip(_inds, _arr2):
                    if not np.isnan(_rpe) and _rpe > thresh:
                        out.append((_c, int(_idx)))
            return out

        # Controls row
        ctrl = QWidget()
        crow = QHBoxLayout(ctrl)
        crow.setContentsMargins(4, 2, 4, 2)
        crow.addWidget(QLabel("User threshold (px):"))

        spin = QDoubleSpinBox()
        spin.setRange(0.0, 100_000.0)
        spin.setDecimals(3)
        spin.setSingleStep(0.05)
        spin.setValue(init_thresh)
        spin.setFixedWidth(115)
        crow.addWidget(spin)

        stat_lbl = QLabel("")
        crow.addWidget(stat_lbl, stretch=1)

        n_sel = len(self._run_selector.get_selected())
        create_btn = QPushButton("⊖  Create new run with points above threshold removed")
        create_btn.setEnabled(n_sel == 1)
        create_btn.setToolTip(
            "Creates a new Phase 2 run with all (camera, image) pairs whose\n"
            "D2.6 RPE exceeds the threshold removed from the detection artifact.\n"
            "Select exactly one Phase 2 run to enable this action."
        )
        crow.addWidget(create_btn)

        def _update_threshold(v: float) -> None:
            hline.set_ydata([v, v])
            hline.set_label(f"User threshold ({v:.3f} px)")
            ax.legend(fontsize=8)
            canvas.draw_idle()
            pairs = _pairs_above(v)
            stat_lbl.setText(f"  {len(pairs)} (camera, image) pair(s) above threshold  ")
            create_btn.setEnabled(len(self._run_selector.get_selected()) == 1)

        spin.valueChanged.connect(_update_threshold)
        _update_threshold(init_thresh)

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
                    self, "Selection", "Select exactly one Phase 2 run to create a new run."
                )
                return
            src_run = selected[0]
            thresh = spin.value()
            pairs = _pairs_above(thresh)
            if not pairs:
                QMessageBox.information(
                    self, "Nothing to remove",
                    "No (camera, image) pairs exceed the threshold.\n"
                    "Lower the threshold to remove some observations.",
                )
                return
            cam_im_dict: dict[str, list[int]] = {}
            for _c, _i in pairs:
                cam_im_dict.setdefault(_c, []).append(_i)
            summary = "\n".join(
                f"  {c}: {len(v)} image(s)" for c, v in sorted(cam_im_dict.items())
            )
            reply = QMessageBox.question(
                self, "Confirm — create new Phase 2 run",
                f"Threshold: {thresh:.4f} px\n"
                f"Total (camera, image) pairs to remove: {len(pairs)}\n{summary}\n\n"
                "A new Phase 2 run will be created with those detections excluded.\n"
                "The source run will NOT be modified.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            create_btn.setEnabled(False)
            old_txt = create_btn.text()
            create_btn.setText("⟳  Running Phase 2…")
            stat_lbl.setText("  Creating new run — please wait…  ")

            def _restore() -> None:
                create_btn.setText(old_txt)
                create_btn.setEnabled(len(self._run_selector.get_selected()) == 1)

            self._create_phase2_run_from_threshold(src_run, thresh, pairs, cam_im_dict, _restore)

        create_btn.clicked.connect(_on_create)

        self._per_view_layout.addWidget(ctrl)
        self._per_view_layout.addWidget(canvas)

    def _create_phase2_run_from_threshold(
        self,
        source_run: dict,
        threshold: float,
        pairs: list[tuple[str, int]],
        cam_im_dict: dict[str, list[int]],
        on_done_cb,
    ) -> None:
        """Run Phase 2 with camera-specific filtered detections in a background thread."""
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
            run_dir = ws / "phase2_runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            diagnostics: dict = {}
            error_msg: Optional[str] = None

            try:
                # ── Resolve Phase 1 detection pickle ───────────────────
                source_pickle_str = (source_run.get("artifacts") or {}).get("phase1_detection_pickle")
                if source_pickle_str and Path(source_pickle_str).exists():
                    source_pickle = Path(source_pickle_str)
                else:
                    p1_runs = self._workspace_mgr.load_runs("phase1")
                    p1_run_id = (source_run.get("inputs") or {}).get("phase1_run_id")
                    p1_run = next((r for r in p1_runs if r.get("run_id") == p1_run_id), None)
                    if p1_run is None and p1_runs:
                        p1_run = p1_runs[-1]
                    source_pickle = resolve_phase1_pickle_artifact(p1_run, ws) if p1_run else None

                if source_pickle is None or not Path(source_pickle).exists():
                    raise RuntimeError(
                        "Could not resolve Phase 1 detected_datapoints.pickle for the source run."
                    )

                emit(f"Loading Phase 1 detections: {source_pickle}")
                payload = load_pickle(Path(source_pickle))
                detections, cam_res = extract_detection_and_cam_res(payload)

                # ── Filter detections (per-camera) ──────────────────────
                emit(f"Removing {len(pairs)} (camera, image) pair(s) via cam_im_num filter…")
                filtered_det = detections.delete_row(cam_im_num=cam_im_dict)
                emit("Filtering complete.")

                # ── Save filtered detection pickle ──────────────────────
                filt_pickle_path = run_dir / "filtered_detected_datapoints.pickle"
                with open(filt_pickle_path, "wb") as fh:
                    _pkl.dump((filtered_det, cam_res), fh)
                emit(f"Saved filtered detections: {filt_pickle_path}")

                # ── Rebuild target and run Phase 2 ──────────────────────
                src_params = source_run.get("params") or {}
                target = build_target(
                    src_params.get("target_type", "Ccube"),
                    src_params.get("n_points", 6),
                    src_params.get("length", 30.0),
                )
                emit("Running Phase 2 initial calibration on filtered detections…")
                stream = EmitStream(emit)
                log_handler = EmitLogHandler(emit)
                log_handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
                root_logger = logging.getLogger()
                root_logger.addHandler(log_handler)
                try:
                    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                        cams, _, per_im = run_initial_calibration(
                            detection=filtered_det,
                            calibration_target=target,
                            cam_res=cam_res,
                            save=False,
                            fixed_params=src_params.get("fixed_params"),
                            return_poses_and_costs=True,
                        )
                finally:
                    root_logger.removeHandler(log_handler)
                emit("Phase 2 calibration completed.")

                camset_path = run_dir / "initial_cameras.camset"
                cams.save(camset_path)

                # ── Compute diagnostics ─────────────────────────────────
                d26_per_view, d21_rms = _compute_true_per_view_reprojection(filtered_det, target, cams)
                d22_intr: dict = {}
                d23_dst: dict = {}
                for cam_name, cam in zip(list(cams.get_names()), cams):
                    intr = np.array(cam.intrinsic)
                    d22_intr[cam_name] = {
                        "fx": float(intr[0, 0]), "fy": float(intr[1, 1]),
                        "cx": float(intr[0, 2]), "cy": float(intr[1, 2]),
                        "res": np.array(cam.res).astype(float).tolist(),
                    }
                    dst = np.array(cam.distortion_coefs).reshape(-1)
                    d23_dst[cam_name] = {
                        "coeffs": dst.astype(float).tolist(),
                        "l2_norm": float(np.linalg.norm(dst)),
                    }

                diagnostics = {
                    "D2.1_per_camera_rms_reprojection": d21_rms,
                    "D2.2_intrinsics": d22_intr,
                    "D2.3_distortion": d23_dst,
                    "D2.5_intrinsic_stddev": "not available in current pyCamSet API",
                    "D2.6_per_view_reprojection": d26_per_view,
                    "D2.7_per_view_error_plot": "rendered in diagnostics tab (true per-image reprojection RMS)",
                }

                # ── Build and save metadata ─────────────────────────────
                src_params2 = source_run.get("params") or {}
                metadata = {
                    "run_id": run_id,
                    "phase": "phase2",
                    "params": dict(src_params2),
                    "diagnostics": diagnostics,
                    "error": None,
                    "inputs": {
                        "phase1_run_id": (source_run.get("inputs") or {}).get("phase1_run_id"),
                        "phase2_run_id": source_run.get("run_id"),
                    },
                    "artifacts": {
                        "initial_camset": str(camset_path),
                        "phase1_detection_pickle": str(source_pickle),
                        "filtered_detection_pickle": str(filt_pickle_path),
                        "detection_source_override": str(filt_pickle_path),
                    },
                    "threshold_pruning": {
                        "source_phase2_run_id": source_run.get("run_id"),
                        "phase1_run_id": (source_run.get("inputs") or {}).get("phase1_run_id"),
                        "threshold_px": threshold,
                        "diagnostic_key": "D2.6_per_view_reprojection",
                        "removed_pairs": {c: list(v) for c, v in cam_im_dict.items()},
                        "n_removed_observations": len(pairs),
                        "removal_mode": "camera-specific via cam_im_num",
                    },
                }
                self._workspace_mgr.save_run("phase2", run_id, metadata)
                emit(f"New Phase 2 run saved: {run_id}")
                return metadata

            except Exception as exc:
                error_msg = str(exc)
                emit(f"ERROR: {error_msg}")
                metadata = {
                    "run_id": run_id,
                    "phase": "phase2",
                    "diagnostics": diagnostics,
                    "error": error_msg,
                }
                self._workspace_mgr.save_run("phase2", run_id, metadata)
                return metadata

        self._threshold_worker = PhaseWorker(work_fn, parent=self)

        # Connect line_ready to the Phase 2 settings tab terminal so output
        # is visible when the user navigates back to that tab.
        for _i in range(self._notebook.count()):
            if self._notebook.tabText(_i) == TAB_PHASE2:
                _settings_tab = self._notebook.widget(_i)
                if hasattr(_settings_tab, "_terminal"):
                    self._threshold_worker.line_ready.connect(_settings_tab._terminal.append_line)
                break

        def _on_finished(metadata: dict) -> None:
            on_done_cb()
            if metadata.get("error"):
                QMessageBox.critical(
                    self, "Phase 2 failed",
                    f"New run encountered an error:\n{metadata['error']}"
                )
            else:
                new_id = metadata.get("run_id", "?")
                n_rem = (metadata.get("threshold_pruning") or {}).get("n_removed_observations", len(pairs))
                QMessageBox.information(
                    self, "New Phase 2 run created",
                    f"Run ID: {new_id}\n"
                    f"Removed {n_rem} (camera, image) pair(s) with RPE > {threshold:.4f} px."
                )
            self.refresh()

        self._threshold_worker.finished.connect(_on_finished)
        self._threshold_worker.start()

    def _load_camset_cached(self, camset_path: Path) -> tuple[Optional[object], Optional[str], Optional[str]]:
        key = str(camset_path)
        if key in self._camset_cache:
            cams, note = self._camset_cache[key]
            return cams, note, None

        note: Optional[str] = None
        try:
            root_logger = logging.getLogger()
            filt = _SuppressDtctConfigFilter()
            for h in root_logger.handlers:
                h.addFilter(filt)
            try:
                cams = load_CameraSet(camset_path)
            finally:
                for h in root_logger.handlers:
                    h.removeFilter(filt)

            note = (
                "Note: CameraSet loaded without embedded detections (dtct_config missing)."
            )
            self._camset_cache[key] = (cams, note)
            return cams, note, None
        except Exception as exc:
            return None, None, str(exc)

    def _render_distortion(self, runs: list[dict]) -> None:
        """Render per-camera distortion vector field (D2.8) from saved camset."""
        while self._distortion_layout.count():
            item = self._distortion_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

        if not runs:
            lbl = QLabel("Select a run to view the distortion field.")
            lbl.setStyleSheet("color: gray;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._distortion_layout.addWidget(lbl)
            return

        if load_CameraSet is None:
            self._distortion_layout.addWidget(QLabel("pyCamSet camera loading is unavailable (import error)."))
            return

        run = runs[-1]
        camset_path = run.get("artifacts", {}).get("initial_camset")
        if not camset_path:
            self._distortion_layout.addWidget(QLabel("No camset artifact found for selected run."))
            return

        cams, note, err = self._load_camset_cached(Path(camset_path))
        if err is not None or cams is None:
            self._distortion_layout.addWidget(QLabel(f"Could not load camset: {err}"))
            return

        try:
            import matplotlib
            matplotlib.use("QtAgg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except ImportError:
            self._distortion_layout.addWidget(QLabel("matplotlib not available."))
            return

        n = len(cams)
        if n <= 0:
            self._distortion_layout.addWidget(QLabel("Camera set is empty."))
            return

        import math
        n_cols = 2
        n_rows = int(math.ceil(n / n_cols))
        fig = Figure(figsize=(12.0, max(4.0, 4.2 * n_rows)), tight_layout=True)
        for i, cam in enumerate(cams, start=1):
            res = np.array(cam.res).astype(int).reshape(-1)
            w = int(res[0]) if res.size >= 2 else 1280
            h = int(res[1]) if res.size >= 2 else 720
            w = max(64, min(1600, w))
            h = max(64, min(1000, h))

            K = np.array(cam.intrinsic, dtype=np.float64)
            D = np.array(cam.distortion_coefs, dtype=np.float64).reshape(-1)

            # Forward distortion field: for each ideal (undistorted) grid point,
            # compute where it ends up after applying lens distortion.
            # Displacement = distorted_position − ideal_position.
            step = max(16, min(w, h) // 20)
            xs = np.arange(step // 2, w, step, dtype=np.float32)
            ys = np.arange(step // 2, h, step, dtype=np.float32)
            gx, gy = np.meshgrid(xs, ys)
            pts_ideal = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)

            # Normalise to camera-space rays
            cx, cy = float(K[0, 2]), float(K[1, 2])
            fx, fy = float(K[0, 0]), float(K[1, 1])
            pts_norm = np.stack(
                [(pts_ideal[:, 0] - cx) / fx,
                 (pts_ideal[:, 1] - cy) / fy,
                 np.ones(len(pts_ideal))],
                axis=1,
            ).astype(np.float32)

            # Project with distortion (identity pose) → distorted pixel coordinates
            R_eye = np.eye(3, dtype=np.float32)
            t_zero = np.zeros(3, dtype=np.float32)
            pts_distorted, _ = cv2.projectPoints(pts_norm, R_eye, t_zero, K, D)
            pts_distorted = pts_distorted.reshape(-1, 2)

            # Displacement vectors: distorted − ideal
            u = pts_distorted[:, 0] - pts_ideal[:, 0]
            v = pts_distorted[:, 1] - pts_ideal[:, 1]
            mag = np.hypot(u, v)

            ax = fig.add_subplot(n_rows, n_cols, i)
            sc = ax.quiver(pts_ideal[:, 0], pts_ideal[:, 1], u, -v, mag,
                           cmap="plasma", angles="xy", scale_units="xy",
                           scale=0.25, width=0.002)
            fig.colorbar(sc, ax=ax, label="displacement (px)")
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.set_aspect("equal")
            ax.set_title(f"{cam.name} -- forward distortion field (distorted - ideal)")
            ax.set_xlabel("x (px)")
            ax.set_ylabel("y (px)")

        self._distortion_layout.addWidget(
            MatplotlibFigureCard(
                f"D2.8 Distortion Field ({run.get('run_id', '?')})",
                fig,
                FigureCanvasQTAgg,
                parent=self._distortion_widget,
                min_height=460,
            )
        )

    def _continue_to_next(self) -> None:
        selected = self._run_selector.get_selected()
        if len(selected) != 1:
            if self._info_cb.isChecked():
                QMessageBox.information(
                    self,
                    "Select one run",
                    "Please select exactly one Phase 2 run to continue to Phase 3.",
                )
            return

        chosen = selected[0]
        f_loc = (chosen.get("params") or {}).get("f_loc")
        run_id = chosen.get("run_id")
        camset_path = (chosen.get("artifacts") or {}).get("initial_camset")
        self._workspace_mgr.write_handoff(
            {
                "phase": "phase2",
                "runs": [chosen],
                "image_folder": f_loc,
                "phase2_run_id": run_id,
                "initial_camset": camset_path,
                "selected_cameras": ((chosen.get("params") or {}).get("selected_cameras") or []),
            }
        )
        for i in range(self._notebook.count()):
            if self._notebook.tabText(i) == TAB_PHASE3:
                self._notebook.setCurrentIndex(i)
                return

