"""
Phase 2 - Per-camera initial calibration (intrinsics) GUI.

Implements Phase 2 from phase_planning.md using existing pyCamSet functions:
- run_initial_calibration
- detect_datapoints_in_imfile (for optional high-distortion re-detection)
"""
from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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
    TAB_PHASE2,
    TAB_PHASE2_DIAG,
    TAB_PHASE3,
    EmitLogHandler,
    EmitStream,
    PhaseWorker,
    RunSelectorWidget,
    TerminalWidget,
    WorkspaceManager,
    build_target,
    extract_detection_and_cam_res,
    make_continue_button,
    make_orange_button,
    make_run_id,
    make_section_label,
    make_separator,
    render_predecessor_chain_section,
    resolve_phase1_pickle_artifact,
)

try:
    from pyCamSet.calibration.camera_calibrator import detect_datapoints_in_imfile, run_initial_calibration
    from pyCamSet.utils.saving import load_CameraSet, load_pickle

    _PYCAMSET_OK = True
except ImportError:
    detect_datapoints_in_imfile = None
    run_initial_calibration = None
    load_CameraSet = None
    load_pickle = None
    _PYCAMSET_OK = False

_TARGET_CHOICES = ["Ccube", "ChArUco"]


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
        form = QFormLayout(form_widget)
        form.setContentsMargins(0, 0, 0, 0)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        top_row.addWidget(form_widget, stretch=1)

        side = QWidget()
        side.setFixedWidth(200)
        side_layout = QVBoxLayout(side)
        side_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        top_row.addWidget(side)

        form.addRow(make_section_label("Paths"))

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
        form.addRow("Image folder (f_loc):", floc_row)

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
        form.addRow("Phase 1 run:", src_row)

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
        form.addRow("Detection source path:", det_row)

        self._phase1_lbl = QLabel("Detection source: auto")
        self._phase1_lbl.setStyleSheet("color: #666;")
        self._phase1_lbl.setWordWrap(True)
        form.addRow("", self._phase1_lbl)

        form.addRow(make_separator())
        form.addRow(make_section_label("Initial Calibration Options"))

        self._cache_cb = QCheckBox("Use detection cache when fallback-detecting")
        self._cache_cb.setChecked(True)
        self._cache_cb.setToolTip(
            "Concept: reuse cached detections during fallback detection.\n"
            "Default: enabled."
        )
        form.addRow(self._cache_cb)

        self._hd_cb = QCheckBox("High Distortion Mode (2c re-detection)")
        self._hd_cb.setToolTip(
            "Concept: perform an extra detection+calibration pass using initial intrinsics.\n"
            "Default: disabled."
        )
        form.addRow(self._hd_cb)

        self._nlim_edit = QLineEdit()
        self._nlim_edit.setPlaceholderText("blank = no limit")
        self._nlim_edit.setFixedWidth(110)
        self._nlim_edit.setToolTip(
            "Concept: cap number of images processed per camera.\n"
            "Default: blank (no limit)."
        )
        form.addRow("Max images per camera (n_lim):", self._nlim_edit)

        self._fp_edit = QLineEdit()
        self._fp_edit.setPlaceholderText('e.g. {"cam0": "int"}')
        self._fp_edit.setToolTip(
            "Concept: JSON map of fixed camera parameters during optimization.\n"
            "Default: blank (no fixed params)."
        )
        form.addRow("Fixed params (JSON):", self._fp_edit)

        form.addRow(make_separator())
        form.addRow(make_section_label("Calibration Target"))

        self._target_combo = QComboBox()
        self._target_combo.addItems(_TARGET_CHOICES)
        self._target_combo.setFixedWidth(140)
        self._target_combo.setToolTip(
            "Concept: calibration target family.\n"
            "Default: Ccube."
        )
        form.addRow("Target type:", self._target_combo)

        self._npts_spin = QSpinBox()
        self._npts_spin.setRange(2, 30)
        self._npts_spin.setValue(6)
        self._npts_spin.setFixedWidth(90)
        self._npts_spin.setToolTip(
            "Concept: target discretization (points/squares along x).\n"
            "Default: 6."
        )
        form.addRow("n_points / squares_x:", self._npts_spin)

        self._length_edit = QLineEdit("30.0")
        self._length_edit.setFixedWidth(110)
        self._length_edit.setToolTip(
            "Concept: physical target size parameter in millimetres.\n"
            "Default: 30.0 mm."
        )
        form.addRow("Length / square size (mm):", self._length_edit)

        btn_row = QHBoxLayout()
        run_btn = QPushButton("▶  Run Phase 2")
        run_btn.setToolTip("Run per-camera initial intrinsics calibration.")
        run_btn.clicked.connect(self._run_phase2)
        btn_row.addWidget(run_btn)
        diag_btn = make_orange_button("Diagnostics ▼", self._open_diagnostics)
        diag_btn.setToolTip("Open Phase 2 diagnostics view.")
        btn_row.addWidget(diag_btn)
        btn_row.addStretch()
        form.addRow(btn_row)

        side_layout.addWidget(make_continue_button(self._continue_to_next))

        self._terminal = TerminalWidget(terminal_cb, parent=self)
        root.addWidget(self._terminal)

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

        self._update_detection_source_label()

        self._terminal.clear_terminal()
        self._terminal.append_line("=== Phase 2: Per-camera Initial Calibration ===")
        self._terminal.append_line(f"Image folder : {params['f_loc']}")
        self._terminal.append_line(f"Phase 1 run  : {phase1_run.get('run_id', 'none') if phase1_run else 'none'}")
        if override_pickle is not None:
            self._terminal.append_line(f"Override det : {override_pickle}")
        self._terminal.append_line("Starting…")

        def work_fn(emit: callable) -> dict:
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

                    if det_pickle is not None and det_pickle.exists():
                        emit(f"Using detections: {det_pickle}")
                        payload = load_pickle(det_pickle)
                        detections, cam_res = extract_detection_and_cam_res(payload)

                    if detections is None or cam_res is None:
                        emit("Detection artifact missing/incompatible, falling back to detection pass.")
                        detections, cam_res = detect_datapoints_in_imfile(
                            f_loc=Path(params["f_loc"]),
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
                            f_loc=Path(params["f_loc"]),
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

                    camset_path = run_dir / (
                        "initial_cameras_high_distortion.camset"
                        if params["high_distortion"]
                        else "initial_cameras.camset"
                    )
                    cams.save(camset_path)

                    d21_rms = {}
                    d22_intr = {}
                    d23_dst = {}
                    d26_per_view = {}

                    per_im_list = per_im if isinstance(per_im, list) else [per_im]
                    cam_names = list(cams.get_names())

                    for cam_name, cam, per_view in zip(cam_names, cams, per_im_list):
                        pv = np.array(per_view).reshape(-1).astype(float)
                        d26_per_view[cam_name] = pv.tolist()
                        d21_rms[cam_name] = float(np.sqrt(np.mean(np.square(pv)))) if pv.size else float("nan")

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
                    diagnostics["D2.4_undistorted_grid"] = "rendered in diagnostics tab from saved camset"
                    diagnostics["D2.5_intrinsic_stddev"] = "not available in current pyCamSet API"
                    diagnostics["D2.6_per_view_reprojection"] = d26_per_view
                    diagnostics["D2.7_per_view_error_plot"] = "rendered in diagnostics tab"

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

        self._per_view_widget = QWidget()
        self._per_view_layout = QVBoxLayout(self._per_view_widget)
        self._sub_tabs.addTab(self._per_view_widget, "Per-view Errors (D2.6-D2.7)")

        self._grid_widget = QWidget()
        self._grid_layout = QVBoxLayout(self._grid_widget)
        self._sub_tabs.addTab(self._grid_widget, "Undistorted Grid (D2.4)")

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
        self._render_grids(chosen)

    def _on_selection_changed(self, runs: list[dict]) -> None:
        self._run_selector.enforce_max_selection(4)
        chosen = self._run_selector.get_selected()
        self._render_summary(chosen)
        self._render_per_view(chosen)
        self._render_grids(chosen)

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

        fig = Figure(figsize=(9, 4.6), tight_layout=True)
        ax = fig.add_subplot(111)
        for cam, vals in d26.items():
            arr = np.array(vals, dtype=float).reshape(-1)
            ax.plot(np.arange(arr.size), arr, marker="o", linewidth=1.2, markersize=3, label=cam)

        ax.set_title(f"D2.6/D2.7 per-view reprojection error ({run.get('run_id', '?')})")
        ax.set_xlabel("Image index")
        ax.set_ylabel("RMS reprojection error (px)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        self._per_view_layout.addWidget(FigureCanvasQTAgg(fig))

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
                "Note: CameraSet loaded without embedded detections (dtct_config missing). "
                "This is non-fatal for D2.4."
            )
            self._camset_cache[key] = (cams, note)
            return cams, note, None
        except Exception as exc:
            return None, None, str(exc)

    def _render_grids(self, runs: list[dict]) -> None:
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

        if not runs:
            lbl = QLabel("Select a run to view undistortion diagnostics.")
            lbl.setStyleSheet("color: gray;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._grid_layout.addWidget(lbl)
            return

        if load_CameraSet is None:
            self._grid_layout.addWidget(QLabel("pyCamSet camera loading is unavailable (import error)."))
            return

        run = runs[-1]
        camset_path = run.get("artifacts", {}).get("initial_camset")
        if not camset_path:
            self._grid_layout.addWidget(QLabel("No camset artifact found for selected run."))
            return

        cams, note, err = self._load_camset_cached(Path(camset_path))
        if err is not None or cams is None:
            self._grid_layout.addWidget(QLabel(f"Could not load camset: {err}"))
            return

        if note:
            info_lbl = QLabel(note)
            info_lbl.setWordWrap(True)
            info_lbl.setStyleSheet("color: #666;")
            self._grid_layout.addWidget(info_lbl)

        try:
            import matplotlib
            matplotlib.use("QtAgg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except ImportError:
            self._grid_layout.addWidget(QLabel("matplotlib not available."))
            return

        n = len(cams)
        if n <= 0:
            self._grid_layout.addWidget(QLabel("Camera set is empty."))
            return

        fig = Figure(figsize=(10.5, max(3.5, 2.8 * n)), tight_layout=True)
        for i, cam in enumerate(cams, start=1):
            res = np.array(cam.res).astype(int).reshape(-1)
            w = int(res[0]) if res.size >= 2 else 1280
            h = int(res[1]) if res.size >= 2 else 720
            w = max(320, min(1600, w))
            h = max(240, min(1000, h))
            grid = _make_grid_image(w, h)
            und = cv2.undistort(grid, np.array(cam.intrinsic), np.array(cam.distortion_coefs))

            ax0 = fig.add_subplot(n, 2, 2 * i - 1)
            ax0.imshow(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
            ax0.set_title(f"{cam.name} distorted grid")
            ax0.axis("off")

            ax1 = fig.add_subplot(n, 2, 2 * i)
            ax1.imshow(cv2.cvtColor(und, cv2.COLOR_BGR2RGB))
            ax1.set_title(f"{cam.name} undistorted")
            ax1.axis("off")

        self._grid_layout.addWidget(FigureCanvasQTAgg(fig))

    def _continue_to_next(self) -> None:
        selected = self._run_selector.get_selected()
        if len(selected) != 1:
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
            }
        )
        for i in range(self._notebook.count()):
            if self._notebook.tabText(i) == TAB_PHASE3:
                self._notebook.setCurrentIndex(i)
                return

