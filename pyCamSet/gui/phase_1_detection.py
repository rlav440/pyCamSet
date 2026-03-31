"""
Phase 1 — Target Detection tab and diagnostics (PySide6).

Calls existing pyCamSet functions:
- :func:`~pyCamSet.calibration.camera_calibrator.detect_datapoints_in_imfile`
- :func:`~pyCamSet.calibration.camera_calibrator.validate_detections`
- :class:`~pyCamSet.calibration_targets.target_Ccube.Ccube`
- :class:`~pyCamSet.calibration_targets.target_charuco.ChArUco`
- :meth:`TargetDetection.features_per_im_per_cam`
- :meth:`TargetDetection.get_cam_list`

Diagnostics implemented (D1.1–D1.7)
-------------------------------------
D1.1 Total detections per camera.
D1.2 Detection rate per camera (%).
D1.3 Board completeness per camera (%).
D1.4 Features-per-image-per-camera heatmap (matplotlib + FigureCanvasQTAgg).
D1.5 Per-camera detection overlay montage (pickle path info).
D1.6 Per-camera detection spatial coverage (convex hull / image area).
D1.7 Minimum features in any image–camera pair.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
import contextlib
import logging
import math
import pickle
import shutil

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
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
    TAB_PHASE1,
    TAB_PHASE1_DIAG,
    TAB_PHASE2,
    CollapsibleSection,
    EmitLogHandler,
    EmitStream,
    PhaseWorker,
    RunSelectorWidget,
    TerminalWidget,
    WorkspaceManager,
    build_target,
    count_images_in_folder,
    get_camera_subfolders,
    make_continue_button,
    make_orange_button,
    make_run_id,
    make_section_label,
    make_separator,
    render_predecessor_chain_section,
)

# pyCamSet guarded imports
try:
    from pyCamSet.calibration.camera_calibrator import (
        detect_datapoints_in_imfile,
        validate_detections,
    )
    _PYCAMSET_OK = True
except ImportError:
    detect_datapoints_in_imfile = None
    validate_detections = None
    _PYCAMSET_OK = False

_TARGET_CHOICES = ["Ccube", "ChArUco"]
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _build_target(target_type: str, n_points: int, length: float):
    """Construct the calibration target object from existing pyCamSet classes."""
    if not _PYCAMSET_OK:
        raise RuntimeError("pyCamSet calibration targets are not importable.")
    return build_target(target_type, n_points, length)


class Phase1Tab(QWidget):
    """Phase 1 — Target Detection tab.

    Calls existing ``detect_datapoints_in_imfile`` and ``validate_detections``.
    """

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
        self._diagnostics_tab: Optional["Phase1DiagnosticsTab"] = None
        self._worker: Optional[PhaseWorker] = None
        self._build_ui(terminal_cb)

    def set_diagnostics_tab(self, tab: "Phase1DiagnosticsTab") -> None:
        self._diagnostics_tab = tab

    # ------------------------------------------------------------------

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
        paths_sect = CollapsibleSection("Paths", expanded=True)
        form_root.addWidget(paths_sect)

        floc_row = QHBoxLayout()
        self._floc_edit = QLineEdit()
        self._floc_edit.setPlaceholderText("Root folder with per-camera sub-folders")
        self._floc_edit.setToolTip(IMAGE_FOLDER_SCHEMATIC)
        self._floc_edit.textChanged.connect(self._on_floc_changed)
        floc_btn = QPushButton("Browse…")
        floc_btn.setFixedWidth(70)
        floc_btn.setToolTip("Select the root folder that contains one subfolder per camera.")
        floc_btn.clicked.connect(self._browse_floc)
        floc_row.addWidget(self._floc_edit)
        floc_row.addWidget(floc_btn)
        paths_sect.addRow("Image folder (f_loc):", floc_row)

        # ── Detection options ──────────────────────────────────────────
        form_root.addWidget(make_separator())
        form_root.addWidget(make_section_label("Detection Options"))

        detect_form = QFormLayout()
        detect_form.setContentsMargins(0, 0, 0, 0)
        detect_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form_root.addLayout(detect_form)

        self._cache_cb = QCheckBox("Cache detections (caching)")
        self._cache_cb.setChecked(True)
        self._cache_cb.setToolTip(
            "Concept: when enabled pyCamSet can reuse a previously saved\n"
            "detected_datapoints.pickle rather than reprocessing every image.\n\n"
            "Default: enabled\n"
            "Range: on / off\n"
            "Guidance: disable only if you suspect a stale cache is masking\n"
            "a real change (e.g. new images added to an existing folder)."
        )
        detect_form.addRow(self._cache_cb)

        self._hd_cb = QCheckBox("High Distortion Mode")
        self._hd_cb.setToolTip(
            "Concept: enables a second detection pass that accounts for heavy\n"
            "radial / fisheye distortion.  The detector uses a coarser initial\n"
            "search, which is slower but more robust on wide-angle lenses.\n\n"
            "Default: disabled\n"
            "Range: on / off\n"
            "Guidance: enable for cameras with an FOV > ~120 degrees or if\n"
            "standard detection finds very few corners near the image edges."
        )
        detect_form.addRow(self._hd_cb)

        self._nlim_edit = QLineEdit()
        self._nlim_edit.setPlaceholderText("blank = no limit")
        self._nlim_edit.setFixedWidth(100)
        self._nlim_edit.setToolTip(
            "Concept: cap the number of images processed per camera.  Useful\n"
            "for quick preview runs or when RAM is limited.\n\n"
            "Default: blank (process all images)\n"
            "Range: positive integer or blank\n"
            "Guidance: for a quality calibration use at least 30–50 images;\n"
            "n_lim < 15 may produce poor intrinsics."
        )
        detect_form.addRow("Max images per camera (n_lim):", self._nlim_edit)

        self._threads_edit = QLineEdit()
        self._threads_edit.setPlaceholderText("blank = auto")
        self._threads_edit.setFixedWidth(100)
        self._threads_edit.setToolTip(
            "Concept: number of parallel worker threads for image processing.\n\n"
            "Default: blank (auto-detect from CPU core count)\n"
            "Range: positive integer or blank\n"
            "Guidance: set to 1 for debugging; leave blank for normal use."
        )
        detect_form.addRow("Threads:", self._threads_edit)

        self._fp_edit = QLineEdit()
        self._fp_edit.setPlaceholderText('e.g. {"cam0": "int"} or blank')
        self._fp_edit.setToolTip(
            "Concept: JSON dict that pins specific camera parameters to fixed\n"
            "values during calibration, preventing them from being optimised.\n\n"
            "Default: blank (all parameters free)\n"
            "Range: valid JSON object, e.g. {\"cam0\": \"ext\"}\n"
            "Guidance: use to hold extrinsics fixed for a known reference\n"
            "camera.  Leave blank unless you have a specific reason."
        )
        detect_form.addRow("Fixed params (JSON):", self._fp_edit)

        self._po_edit = QLineEdit()
        self._po_edit.setPlaceholderText("JSON dict or blank")
        self._po_edit.setToolTip(
            "Concept: JSON dict passed directly to the pyCamSet detection\n"
            "back-end (e.g. to override corner-refinement window size).\n\n"
            "Default: blank (use built-in defaults)\n"
            "Range: valid JSON object\n"
            "Guidance: advanced option — leave blank unless instructed by\n"
            "the pyCamSet documentation."
        )
        detect_form.addRow("Problem options (JSON):", self._po_edit)

        # ── Calibration Target (collapsible) ───────────────────────────
        form_root.addWidget(make_separator())
        target_sect = CollapsibleSection("Calibration Target", expanded=True)
        form_root.addWidget(target_sect)

        self._target_combo = QComboBox()
        self._target_combo.addItems(_TARGET_CHOICES)
        self._target_combo.setFixedWidth(140)
        self._target_combo.setToolTip(
            "Concept: the physical calibration target type.\n\n"
            "Ccube — corner-cube target with coded markers; robust to partial\n"
            "  occlusion and suitable for most multi-camera setups.\n"
            "ChArUco — charuco board (chessboard + ArUco markers); widely\n"
            "  supported and easy to print.\n\n"
            "Default: Ccube\n"
            "Guidance: match this exactly to the physical target you are using."
        )
        target_sect.addRow("Target type:", self._target_combo)

        self._npts_spin = QSpinBox()
        self._npts_spin.setRange(2, 20)
        self._npts_spin.setValue(6)
        self._npts_spin.setFixedWidth(80)
        self._npts_spin.setToolTip(
            "Concept: the grid density of the calibration target.\n"
            "For Ccube: number of points per face edge.\n"
            "For ChArUco: number of squares along the x-axis.\n\n"
            "Default: 6\n"
            "Range: 2–20\n"
            "Guidance: must exactly match the physical target you are using.\n"
            "Higher values give more feature constraints per image."
        )
        target_sect.addRow("n_points / squares_x:", self._npts_spin)

        self._length_edit = QLineEdit("30.0")
        self._length_edit.setFixedWidth(100)
        self._length_edit.setToolTip(
            "Concept: the physical size of one feature on the calibration\n"
            "target, in millimetres.  This sets the metric scale of the\n"
            "calibration.\n\n"
            "Default: 30.0 mm\n"
            "Range: any positive float (mm)\n"
            "Guidance: measure the actual printed/machined target — even a\n"
            "1% error here propagates directly into reconstructed distances."
        )
        target_sect.addRow("Length / square size (mm):", self._length_edit)

        # ── Action buttons ─────────────────────────────────────────────
        form_root.addWidget(make_separator())
        btn_row = QHBoxLayout()
        run_btn = QPushButton("▶  Run Phase 1")
        run_btn.setToolTip("Run target detection for the selected image folder.")
        run_btn.clicked.connect(self._run_phase1)
        btn_row.addWidget(run_btn)
        diag_btn = make_orange_button("Diagnostics ▼", self._open_diagnostics)
        diag_btn.setToolTip("Open Phase 1 diagnostics (hidden tab).")
        btn_row.addWidget(diag_btn)
        btn_row.addStretch()
        form_root.addLayout(btn_row)
        form_root.addStretch()

        # ── Side panel ────────────────────────────────────────────────
        side_layout.addWidget(make_continue_button(self._continue_to_next))

        # ── Terminal ─────────────────────────────────────────────────
        self._terminal = TerminalWidget(terminal_cb, parent=self)
        root.addWidget(self._terminal)

    # ------------------------------------------------------------------

    def _browse_floc(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select image folder")
        if path:
            self._floc_edit.setText(path)

    def set_image_folder(self, path: str) -> None:
        self._floc_edit.setText(path)
        self._sync_workspace_from_floc(path, create_if_missing=True)

    def _on_floc_changed(self, text: str) -> None:
        self._sync_workspace_from_floc(text, create_if_missing=True)

    def _sync_workspace_from_floc(self, floc_text: str, create_if_missing: bool = True) -> None:
        floc = (floc_text or "").strip()
        if not floc:
            return
        f_loc = Path(floc)
        if not f_loc.exists() or not f_loc.is_dir():
            return

        ws_path = f_loc / ".pycamset_workspace"
        if self._workspace_mgr.workspace_path != ws_path:
            self._workspace_mgr.set_workspace_path(ws_path, ensure=create_if_missing)
            if self._diagnostics_tab is not None:
                self._diagnostics_tab.refresh()

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
            QMessageBox.critical(self, "Validation Error", "Length must be a number.")
            return None

        threads = None
        if self._threads_edit.text().strip():
            try:
                threads = int(self._threads_edit.text().strip())
            except ValueError:
                QMessageBox.critical(self, "Validation Error", "Threads must be an integer.")
                return None

        import json
        fixed_params = None
        if self._fp_edit.text().strip():
            try:
                fixed_params = json.loads(self._fp_edit.text().strip())
            except json.JSONDecodeError as exc:
                QMessageBox.critical(self, "Validation Error", f"Fixed params JSON: {exc}")
                return None

        problem_options = None
        if self._po_edit.text().strip():
            try:
                problem_options = json.loads(self._po_edit.text().strip())
            except json.JSONDecodeError as exc:
                QMessageBox.critical(self, "Validation Error", f"Problem options JSON: {exc}")
                return None

        return {
            "f_loc": floc,
            "caching": self._cache_cb.isChecked(),
            "high_distortion": self._hd_cb.isChecked(),
            "n_lim": n_lim,
            "threads": threads,
            "fixed_params": fixed_params,
            "problem_options": problem_options,
            "target_type": self._target_combo.currentText(),
            "n_points": self._npts_spin.value(),
            "length": length,
        }

    def _run_phase1(self) -> None:
        params = self._collect_params()
        if params is None:
            return

        f_loc = Path(params["f_loc"])
        canonical_ws = f_loc / ".pycamset_workspace"
        if self._workspace_mgr.workspace_path is None or self._workspace_mgr.workspace_path != canonical_ws:
            self._workspace_mgr.set_workspace_path(canonical_ws, ensure=True)

        self._terminal.clear_terminal()
        self._terminal.append_line("=== Phase 1: Target Detection ===")
        self._terminal.append_line(f"Image folder : {params['f_loc']}")
        self._terminal.append_line(
            f"Target       : {params['target_type']} "
            f"(n={params['n_points']}, length={params['length']} mm)"
        )
        self._terminal.append_line(f"caching      : {params['caching']}")
        self._terminal.append_line(f"high_distort : {params['high_distortion']}")
        self._terminal.append_line(f"threads      : {params['threads'] or 'auto'}")
        self._terminal.append_line("Starting detection…")

        def work_fn(emit: callable) -> dict:
            diagnostics: dict = {}
            error_msg: Optional[str] = None
            det_pickle_src: Optional[Path] = None

            stream = EmitStream(emit)
            log_handler = EmitLogHandler(emit)
            log_handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
            root_logger = logging.getLogger()
            root_logger.addHandler(log_handler)

            try:
                with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                    f_loc = Path(params["f_loc"])

                    cam_folders = get_camera_subfolders(f_loc)
                    cam_names = [p.name for p in cam_folders]
                    cam_img_counts = {p.name: count_images_in_folder(p) for p in cam_folders}
                    emit(f"1a  Camera sub-folders: {cam_names}")
                    if len(cam_folders) < 2:
                        raise RuntimeError("Need at least two camera sub-folders.")
                    counts = [count_images_in_folder(p) for p in cam_folders]
                    if not counts or any(c <= 0 for c in counts) or len(set(counts)) != 1:
                        raise RuntimeError("Camera folders must contain equal non-zero image counts.")

                    target = _build_target(
                        params["target_type"],
                        params["n_points"],
                        params["length"],
                    )

                    detect_root = f_loc
                    tmp_ctx: Optional[TemporaryDirectory] = None
                    root_entries = list(f_loc.iterdir())
                    allowed = {p.name for p in cam_folders}
                    has_extra_entries = any(p.name not in allowed for p in root_entries)
                    if has_extra_entries:
                        tmp_ctx = TemporaryDirectory(prefix="pycamset_phase1_")
                        detect_root = Path(tmp_ctx.name)
                        emit("1a  Using filtered staging folder (camera subfolders only).")
                        for cam in cam_folders:
                            dst = detect_root / cam.name
                            try:
                                dst.symlink_to(cam, target_is_directory=True)
                            except Exception:
                                shutil.copytree(cam, dst)

                    try:
                        detections, cam_res = detect_datapoints_in_imfile(
                            f_loc=detect_root,
                            calibration_target=target,
                            caching=params["caching"],
                            draw=False,
                            n_lim=params["n_lim"],
                        )
                        emit("1b  Detection complete.")

                        if detect_root != f_loc:
                            for artifact_name in ("detected_datapoints.pickle",):
                                src = detect_root / artifact_name
                                if src.exists():
                                    dst = f_loc / artifact_name
                                    shutil.copy2(src, dst)
                                    if artifact_name == "detected_datapoints.pickle":
                                        det_pickle_src = dst
                        else:
                            cand = f_loc / "detected_datapoints.pickle"
                            if cand.exists():
                                det_pickle_src = cand

                    finally:
                        if tmp_ctx is not None:
                            tmp_ctx.cleanup()

                    validate_detections(detections, target)
                    emit("1c  Validation complete.")

                    try:
                        # D1.1 unchanged
                        total_per_cam: dict[str, int] = {}
                        for cam_det in detections.get_cam_list():
                            data = cam_det.get_data()
                            if data is None or len(data) == 0:
                                continue
                            cam_idx = int(data[0, 0])
                            cam_name = detections.cam_names[cam_idx]
                            total_per_cam[cam_name] = len(data)
                        diagnostics["D1.1_total_detections"] = total_per_cam

                        # D1.2 / D1.3 revised
                        corners_per_face = int(target.point_data.shape[-2])
                        det_rate: dict[str, float] = {}
                        completeness: dict[str, float] = {}

                        for cam_det in detections.get_cam_list():
                            data0 = cam_det.get_data()
                            if data0 is None or len(data0) == 0:
                                continue
                            cam_idx = int(data0[0, 0])
                            cam_name = detections.cam_names[cam_idx]
                            expected = int(cam_img_counts.get(cam_name, detections.max_ims))
                            if params["n_lim"] is not None:
                                expected = min(expected, int(params["n_lim"]))
                            expected = max(expected, 1)

                            detected_images = 0
                            per_image_frac: list[float] = []

                            for im_det in cam_det.get_image_list():
                                datum = im_det.get_data()
                                if datum is None or len(datum) == 0:
                                    continue
                                detected_images += 1

                                id_cols = datum[:, 2:-2]
                                if id_cols.ndim == 1:
                                    id_cols = id_cols.reshape(-1, 1)

                                if id_cols.shape[1] <= 0:
                                    continue

                                if id_cols.shape[1] == 1:
                                    # point-id only
                                    n_unique_points = len(np.unique(id_cols[:, 0]))
                                    per_image_frac.append(n_unique_points / max(corners_per_face, 1))
                                else:
                                    # board-id columns + final point-id column
                                    board_cols = id_cols[:, :-1]
                                    point_col = id_cols[:, -1]
                                    board_ids = np.unique(board_cols, axis=0)
                                    board_fracs: list[float] = []
                                    for b in board_ids:
                                        mask = np.all(board_cols == b, axis=1)
                                        n_unique_points = len(np.unique(point_col[mask]))
                                        board_fracs.append(n_unique_points / max(corners_per_face, 1))
                                    if board_fracs:
                                        per_image_frac.append(float(np.mean(board_fracs)))

                            det_rate[cam_name] = float(detected_images) / float(expected)
                            completeness[cam_name] = float(np.mean(per_image_frac)) if per_image_frac else 0.0

                        diagnostics["D1.2_detection_rate"] = det_rate
                        diagnostics["D1.3_board_completeness"] = completeness

                        for cam in detections.cam_names:
                            r = det_rate.get(cam, 0.0) * 100.0
                            c = completeness.get(cam, 0.0) * 100.0
                            emit(f"D1.2/D1.3  {cam}: rate={r:.1f}% completeness={c:.1f}%")

                        # D1.4 — existing features_per_im_per_cam
                        fpm = detections.features_per_im_per_cam()
                        diagnostics["D1.4_features_matrix"] = fpm.tolist()

                        # D1.6 — spatial coverage
                        try:
                            from scipy.spatial import ConvexHull
                            coverage: dict[str, float] = {}
                            for cam_det, res in zip(detections.get_cam_list(), cam_res):
                                cam_idx = int(cam_det.get_data()[0, 0])
                                cam_name = detections.cam_names[cam_idx]
                                pts = cam_det.get_data()[:, -2:]
                                img_area = float(res[0]) * float(res[1])
                                if len(pts) >= 3:
                                    try:
                                        hull_area = ConvexHull(pts).volume
                                        coverage[cam_name] = hull_area / img_area
                                    except Exception:
                                        coverage[cam_name] = float("nan")
                                else:
                                    coverage[cam_name] = float("nan")
                            diagnostics["D1.6_spatial_coverage"] = coverage
                        except ImportError:
                            pass

                        # D1.7 — min features
                        min_feat = int(np.min(fpm[fpm > 0])) if np.any(fpm > 0) else 0
                        diagnostics["D1.7_min_features"] = min_feat
                        emit(f"D1.7  Min features in any image–camera: {min_feat}")

                        diagnostics["cam_names"] = detections.cam_names
                        diagnostics["n_images"] = int(detections.max_ims)

                    except Exception as diag_exc:
                        emit(f"  (partial diagnostics: {diag_exc})")

                    emit("Phase 1 complete.")

            except Exception as exc:
                error_msg = str(exc)
                diagnostics["error"] = error_msg
            finally:
                stream.flush()
                root_logger.removeHandler(log_handler)

            run_id = make_run_id()
            metadata = {
                "run_id": run_id,
                "phase": "phase1",
                "params": params,
                "diagnostics": diagnostics,
                "error": error_msg,
            }

            # Defensive fallback: always guarantee workspace is set before save.
            if self._workspace_mgr.workspace_path is None:
                self._workspace_mgr.set_workspace_path(Path(params["f_loc"]) / ".pycamset_workspace", ensure=True)

            meta_path = self._workspace_mgr.save_run("phase1", run_id, metadata)
            run_dir = meta_path.parent
            run_pickle = run_dir / "detected_datapoints.pickle"

            if det_pickle_src is None:
                cand = Path(params["f_loc"]) / "detected_datapoints.pickle"
                if cand.exists():
                    det_pickle_src = cand

            if det_pickle_src is not None and det_pickle_src.exists():
                try:
                    shutil.copy2(det_pickle_src, run_pickle)
                    metadata.setdefault("artifacts", {})["detected_datapoints_pickle"] = str(run_pickle)
                    self._workspace_mgr.save_run("phase1", run_id, metadata)
                    emit(f"Artifact saved: {run_pickle}")
                except Exception as copy_exc:
                    emit(f"Warning: could not save run-local pickle: {copy_exc}")

            emit(f"Run saved: {run_id}")
            return metadata

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
        runs = self._workspace_mgr.load_runs("phase1")
        if not runs:
            if self._info_cb.isChecked():
                QMessageBox.information(self, "No runs", "Run Phase 1 first.")
            return
        chosen = runs[-1]
        self._workspace_mgr.write_handoff({"phase": "phase1", "runs": [chosen]})
        run_id = chosen.get("run_id", "")
        for i in range(self._notebook.count()):
            if self._notebook.tabText(i) == TAB_PHASE2:
                tab = self._notebook.widget(i)
                if hasattr(tab, "set_selected_phase1_run_id"):
                    tab.set_selected_phase1_run_id(run_id)
                self._notebook.setCurrentIndex(i)
                return


# ---------------------------------------------------------------------------
# Phase 1 Diagnostics tab
# ---------------------------------------------------------------------------


class Phase1DiagnosticsTab(QWidget):
    """Phase 1 Diagnostics — multi-run comparison view.

    Three sub-tabs:
    1. Summary (D1.1–D1.3, D1.6–D1.7)
    2. Heatmap (D1.4) — matplotlib ``imshow`` via ``FigureCanvasQTAgg``
    3. Montage (D1.5) — pickle-path info
    """

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

        self._draw_state: dict = {}
        self._draw_index: int = 0
        self._draw_status_lbl: Optional[QLabel] = None

        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        top_btn_row = QHBoxLayout()
        top_btn_row.addWidget(
            make_orange_button("▲ Detection Settings", self._go_to_detection_settings)
        )
        top_btn_row.addStretch()
        root.addLayout(top_btn_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        # ── Left: run selector ─────────────────────────────────────────
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

        # ── Right: sub-tabs ────────────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.addWidget(make_section_label("Phase 1 Diagnostics"))
        right_layout.addWidget(make_separator())

        self._sub_tabs = QTabWidget()
        right_layout.addWidget(self._sub_tabs)
        splitter.addWidget(right)

        splitter.setSizes([200, 700])

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
        self._sub_tabs.addTab(self._summary_widget, "Summary (D1.1–D1.3, D1.6–D1.7)")
        self._sub_tabs.tabBar().setTabToolTip(
            0,
            "Summary metrics:\n"
            "D1.1 total detections per camera\n"
            "D1.2 detection rate (%)\n"
            "D1.3 board completeness (%)\n"
            "D1.6 spatial coverage\n"
            "D1.7 minimum features",
        )

        self._heatmap_widget = QWidget()
        self._heatmap_layout = QVBoxLayout(self._heatmap_widget)
        self._sub_tabs.addTab(self._heatmap_widget, "Heatmap (D1.4)")
        self._sub_tabs.tabBar().setTabToolTip(
            1, "D1.4 features-per-image-per-camera heatmap."
        )

        self._montage_widget = QWidget()
        self._montage_layout = QVBoxLayout(self._montage_widget)
        self._sub_tabs.addTab(self._montage_widget, "Draw Detections")
        self._sub_tabs.tabBar().setTabToolTip(
            2, "Render an in-GUI montage with detected points overlaid (D1.5)."
        )

        # Keyboard navigation for Draw Detections tab
        self._prev_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self._montage_widget)
        self._prev_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._prev_shortcut.activated.connect(lambda: self._step_draw_image(-1))

        self._next_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self._montage_widget)
        self._next_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._next_shortcut.activated.connect(lambda: self._step_draw_image(+1))

        # ── Bottom: continue button ────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(make_continue_button(self._continue_to_next))
        root.addLayout(btn_row)

        self.refresh()

    def refresh(self) -> None:
        runs = self._workspace_mgr.load_runs("phase1")
        self._run_selector.refresh(runs)
        selected = self._run_selector.get_selected()
        self._render_summary(selected)
        self._render_heatmap(selected)
        self._render_montage(selected)

    def _on_selection_changed(self, runs: list[dict]) -> None:
        self._run_selector.enforce_max_selection(5)
        runs = self._run_selector.get_selected()
        self._render_summary(runs)
        self._render_heatmap(runs)
        self._render_montage(runs)

    def _go_to_detection_settings(self) -> None:
        """Return from hidden diagnostics tab to main Phase 1 tab."""
        for i in range(self._notebook.count()):
            if self._notebook.tabText(i) == TAB_PHASE1:
                self._notebook.setCurrentIndex(i)
                return

    # ------------------------------------------------------------------

    def _render_summary(self, runs: list[dict]) -> None:
        while self._summary_layout.count():
            item = self._summary_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

        if not runs:
            lbl = QLabel("Select one or more runs (up to 5) from the list to compare.")
            lbl.setStyleSheet("color: gray;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._summary_layout.addWidget(lbl)
            return

        try:
            import matplotlib
            matplotlib.use("QtAgg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        except ImportError:
            lbl = QLabel("matplotlib not available — cannot render summary plots.")
            lbl.setStyleSheet("color: gray;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._summary_layout.addWidget(lbl)
            return

        runs = runs[-5:]
        run_ids = [r.get("run_id", "unknown") for r in runs]
        short_ids = [rid[-8:] if len(rid) > 8 else rid for rid in run_ids]

        d11_vals, d12_vals, d13_vals, d16_vals, d17_vals = [], [], [], [], []
        for run in runs:
            d = run.get("diagnostics", {})
            det_dict = d.get("D1.1_total_detections", {}) or {}
            rate_dict = d.get("D1.2_detection_rate", {}) or {}
            comp_dict = d.get("D1.3_board_completeness", {}) or {}
            cov_dict = d.get("D1.6_spatial_coverage", {}) or {}

            d11_vals.append(float(sum(det_dict.values())) if det_dict else 0.0)
            d12_vals.append(float(np.mean(list(rate_dict.values())) * 100.0) if rate_dict else 0.0)
            d13_vals.append(float(np.mean(list(comp_dict.values())) * 100.0) if comp_dict else 0.0)
            cov_vals = [float(v) for v in cov_dict.values() if np.isfinite(v)]
            d16_vals.append(float(np.mean(cov_vals) * 100.0) if cov_vals else 0.0)
            try:
                d17_vals.append(float(d.get("D1.7_min_features", 0) or 0))
            except Exception:
                d17_vals.append(0.0)

        metric_specs = [
            ("D1.1 Total Detections", d11_vals, "count"),
            ("D1.2 Detection Rate %", d12_vals, "%"),
            ("D1.3 Completeness %", d13_vals, "%"),
            ("D1.6 Coverage %", d16_vals, "%"),
            ("D1.7 Min Features", d17_vals, "count"),
        ]

        self._summary_layout.addWidget(make_section_label("Summary plots"))

        plots_host = QWidget()
        plots_grid = QGridLayout(plots_host)
        plots_grid.setContentsMargins(0, 0, 0, 0)
        plots_grid.setHorizontalSpacing(14)
        plots_grid.setVerticalSpacing(14)

        # Larger plots + wrap rows to avoid squishing.
        avail_w = max(1, self._summary_scroll.viewport().width())
        target_plot_w_px = 520
        n_cols = max(1, min(3, avail_w // target_plot_w_px))

        cmap = matplotlib.colormaps.get_cmap("tab10")
        x = np.arange(len(runs))

        for i, (title, vals, ylab) in enumerate(metric_specs):
            fig = Figure(figsize=(5.4, 3.8), tight_layout=True)
            ax = fig.add_subplot(111)
            colors = [cmap(k % 10) for k in range(len(runs))]
            bars = ax.bar(x, vals, color=colors, edgecolor="#222222", linewidth=0.4)
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(ylab, fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(short_ids, rotation=25, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.25)

            # Value labels for readability
            for b, v in zip(bars, vals):
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    b.get_height(),
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

            canvas = FigureCanvasQTAgg(fig)
            canvas.setMinimumSize(500, 330)

            r = i // n_cols
            c = i % n_cols
            plots_grid.addWidget(canvas, r, c)

        self._summary_layout.addWidget(plots_host)
        self._summary_layout.addWidget(make_separator())
        self._summary_layout.addWidget(make_section_label("Per-run tables"))

        # Restored earlier compact table-style layout per run
        for run in runs:
            run_id = run.get("run_id", "unknown")
            d = run.get("diagnostics", {})

            det_dict = d.get("D1.1_total_detections", {}) or {}
            rate_dict = d.get("D1.2_detection_rate", {}) or {}
            comp_dict = d.get("D1.3_board_completeness", {}) or {}
            cov_dict = d.get("D1.6_spatial_coverage", {}) or {}

            hdr = QLabel(f"Run: {run_id}")
            hdr.setStyleSheet("font-weight: bold; margin-top: 8px;")
            self._summary_layout.addWidget(hdr)

            # Top summary lines (same content, compact style)
            cov_vals = [float(v) for v in cov_dict.values() if np.isfinite(v)]
            summary_form = QFormLayout()
            summary_form.setContentsMargins(16, 0, 0, 0)
            summary_form.addRow("D1.1 Total detections (sum):", QLabel(str(int(sum(det_dict.values())) if det_dict else 0)))
            summary_form.addRow(
                "D1.2 Detection Rate % (mean):",
                QLabel(f"{(float(np.mean(list(rate_dict.values())) * 100.0) if rate_dict else 0.0):.2f}"),
            )
            summary_form.addRow(
                "D1.3 Completeness % (mean):",
                QLabel(f"{(float(np.mean(list(comp_dict.values())) * 100.0) if comp_dict else 0.0):.2f}"),
            )
            summary_form.addRow(
                "D1.6 Coverage % (mean):",
                QLabel(f"{(float(np.mean(cov_vals) * 100.0) if cov_vals else 0.0):.2f}"),
            )
            summary_form.addRow("D1.7 Min features:", QLabel(str(d.get("D1.7_min_features", "—"))))
            self._summary_layout.addLayout(summary_form)

            cams = sorted(set(det_dict.keys()) | set(rate_dict.keys()) | set(comp_dict.keys()) | set(cov_dict.keys()))
            if cams:
                head = QHBoxLayout()
                for text, stretch in [
                    ("Camera", 2),
                    ("D1.1 Detections", 1),
                    ("D1.2 Detection Rate %", 1),
                    ("D1.3 Completeness %", 1),
                    ("D1.6 Coverage %", 1),
                ]:
                    lbl = QLabel(text)
                    lbl.setStyleSheet("font-weight: bold;")
                    head.addWidget(lbl, stretch=stretch)
                self._summary_layout.addLayout(head)

                # Data rows
                for cam in cams:
                    row = QHBoxLayout()
                    d11 = int(det_dict.get(cam, 0))
                    d12 = float(rate_dict.get(cam, 0.0) * 100.0)
                    d13 = float(comp_dict.get(cam, 0.0) * 100.0)
                    c16 = cov_dict.get(cam, float("nan"))
                    d16s = f"{(float(c16) * 100.0):.1f}" if np.isfinite(c16) else "—"

                    row.addWidget(QLabel(cam), stretch=2)
                    row.addWidget(QLabel(str(d11)), stretch=1)
                    row.addWidget(QLabel(f"{d12:.1f}"), stretch=1)
                    row.addWidget(QLabel(f"{d13:.1f}"), stretch=1)
                    row.addWidget(QLabel(d16s), stretch=1)
                    self._summary_layout.addLayout(row)

            self._summary_layout.addWidget(make_separator())
            render_predecessor_chain_section(self._summary_layout, self._workspace_mgr, run)

    def _render_heatmap(self, runs: list[dict]) -> None:
        while self._heatmap_layout.count():
            item = self._heatmap_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not runs:
            lbl = QLabel("Select a run to view the detection heatmap.")
            lbl.setStyleSheet("color: gray;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._heatmap_layout.addWidget(lbl)
            return

        try:
            import matplotlib
            matplotlib.use("QtAgg")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        except ImportError:
            lbl = QLabel("matplotlib not available — cannot render heatmap.")
            lbl.setStyleSheet("color: gray;")
            self._heatmap_layout.addWidget(lbl)
            return

        matrix_data = None
        cam_names: list[str] = []
        selected_run: dict = {}
        for run in reversed(runs):
            d = run.get("diagnostics", {})
            if "D1.4_features_matrix" in d:
                matrix_data = np.array(d["D1.4_features_matrix"])
                cam_names = d.get("cam_names", [])
                selected_run = run
                break

        if matrix_data is None:
            lbl = QLabel(
                "No heatmap data in selected run(s).\nRe-run Phase 1 to generate it."
            )
            lbl.setStyleSheet("color: gray;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._heatmap_layout.addWidget(lbl)
            return

        n_ims, n_cams = matrix_data.shape
        fig_w = min(max(5.0, n_cams * 0.6), 14.0)
        fig_h = min(max(3.0, n_ims * 0.15), 8.0)

        fig = Figure(figsize=(fig_w, fig_h), tight_layout=True)
        ax = fig.add_subplot(111)
        im = ax.imshow(matrix_data, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_xlabel("Camera index")
        ax.set_ylabel("Image index")
        ax.set_title(
            f"D1.4  Features per image per camera\n(run: {selected_run.get('run_id', '?')})"
        )
        if cam_names:
            ax.set_xticks(range(n_cams))
            ax.set_xticklabels(cam_names, rotation=30, ha="right", fontsize=8)
        fig.colorbar(im, ax=ax, label="features detected")

        canvas = FigureCanvasQTAgg(fig)
        self._heatmap_layout.addWidget(canvas)

    def _render_montage(self, runs: list[dict]) -> None:
        while self._montage_layout.count():
            item = self._montage_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        title = make_section_label("Draw Detections (D1.5)")
        title.setToolTip("Render camera images with detected feature points overlaid.")
        self._montage_layout.addWidget(title)

        nav_row = QHBoxLayout()
        prev_btn = QPushButton("◀")
        prev_btn.setToolTip("Previous detected image (Left Arrow key).")
        prev_btn.clicked.connect(lambda: self._step_draw_image(-1))
        nav_row.addWidget(prev_btn)

        next_btn = QPushButton("▶")
        next_btn.setToolTip("Next detected image (Right Arrow key).")
        next_btn.clicked.connect(lambda: self._step_draw_image(+1))
        nav_row.addWidget(next_btn)

        self._draw_status_lbl = QLabel("0/0")
        nav_row.addWidget(self._draw_status_lbl)
        nav_row.addStretch()
        self._montage_layout.addLayout(nav_row)

        info = QLabel(
            "Select one or more runs on the left, then click 'Draw Detections'. "
            "Use ◀/▶ buttons or keyboard Left/Right to cycle images."
        )
        info.setWordWrap(True)
        self._montage_layout.addWidget(info)

        draw_btn = QPushButton("Draw Detections")
        draw_btn.setToolTip("Render the latest selected run with available detection data.")
        draw_btn.clicked.connect(self._draw_detections_clicked)
        self._montage_layout.addWidget(draw_btn)

        for run in runs:
            pkl_path = self._resolve_pickle_path_for_run(run)
            if pkl_path is None:
                continue
            exists = "✓ exists" if pkl_path.exists() else "✗ not found"
            color = "#2e7d32" if pkl_path.exists() else "#b71c1c"
            path_lbl = QLabel(f"{pkl_path}  [{exists}]")
            path_lbl.setStyleSheet(f"color: {color};")
            path_lbl.setWordWrap(True)
            self._montage_layout.addWidget(path_lbl)

        self._montage_layout.addStretch()

    def open_draw_detections_for_latest(self) -> None:
        """Programmatically open D1.5 tab and render latest run with detections."""
        self._sub_tabs.setCurrentIndex(2)  # Draw Detections tab
        runs = self._workspace_mgr.load_runs("phase1")
        if not runs:
            return
        chosen = None
        for run in reversed(runs):
            p = self._resolve_pickle_path_for_run(run)
            if p is not None and p.exists():
                chosen = run
                break
        if chosen is not None:
            self._draw_detections_for_run(chosen, show_errors=False)

    def _draw_detections_clicked(self) -> None:
        runs = self._run_selector.get_selected()
        if not runs:
            if self._info_cb.isChecked():
                QMessageBox.information(self, "No run selected", "Select at least one run.")
            return

        chosen = None
        for run in reversed(runs):
            p = self._resolve_pickle_path_for_run(run)
            if p is not None and p.exists():
                chosen = run
                break

        if chosen is None:
            QMessageBox.warning(
                self,
                "No detections file",
                "No selected run has an available detected_datapoints.pickle.",
            )
            return

        self._draw_detections_for_run(chosen, show_errors=True)

    def _draw_detections_for_run(self, chosen: dict, show_errors: bool = True) -> None:
        try:
            import matplotlib
            matplotlib.use("QtAgg")
            import matplotlib.image as mpimg
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        except ImportError:
            if show_errors:
                QMessageBox.warning(self, "Missing dependency", "matplotlib is required.")
            return

        floc = Path(chosen["params"]["f_loc"])
        det_path = self._resolve_pickle_path_for_run(chosen)
        if det_path is None or not det_path.exists():
            if show_errors:
                QMessageBox.warning(self, "No detections file", "No pickle found for this run.")
            return

        try:
            with open(det_path, "rb") as fh:
                payload = pickle.load(fh)
            detections = self._extract_detections_obj(payload)
            if detections is None:
                raise TypeError("Unsupported pickle payload (no object with get_cam_list).")
        except Exception as exc:
            if show_errors:
                QMessageBox.critical(self, "Load error", f"Could not load detections pickle:\n{exc}")
            return

        cam_map = {}
        for cam_det in detections.get_cam_list():
            data = cam_det.get_data()
            if data is not None and len(data):
                cam_idx = int(data[0, 0])
                cam_map[detections.cam_names[cam_idx]] = cam_det

        cam_folders = {p.name: p for p in get_camera_subfolders(floc)}
        cams = [c for c in detections.cam_names if c in cam_folders]
        if not cams:
            if show_errors:
                QMessageBox.warning(self, "No cameras", "No matching camera folders found.")
            return

        # Build fast draw state
        cam_images: dict[str, list[Path]] = {}
        cam_points: dict[str, dict[int, np.ndarray]] = {}
        max_images = 0

        for cam in cams:
            ims = sorted([p for p in cam_folders[cam].iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS])
            cam_images[cam] = ims
            max_images = max(max_images, len(ims))
            per_im: dict[int, np.ndarray] = {}
            cam_det = cam_map.get(cam, None)
            data = cam_det.get_data() if cam_det is not None else None
            if data is not None and len(data) and data.shape[1] >= 2:
                for im_idx in np.unique(data[:, 1].astype(int)):
                    pts = data[data[:, 1].astype(int) == int(im_idx)][:, -2:]
                    per_im[int(im_idx)] = pts
            cam_points[cam] = per_im

        if max_images <= 0:
            if show_errors:
                QMessageBox.warning(self, "No images", "No images found in camera folders.")
            return

        while self._montage_layout.count():
            item = self._montage_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # top controls
        title = make_section_label("Draw Detections (D1.5)")
        self._montage_layout.addWidget(title)
        nav_row = QHBoxLayout()
        prev_btn = QPushButton("◀")
        prev_btn.clicked.connect(lambda: self._step_draw_image(-1))
        nav_row.addWidget(prev_btn)
        next_btn = QPushButton("▶")
        next_btn.clicked.connect(lambda: self._step_draw_image(+1))
        nav_row.addWidget(next_btn)
        self._draw_status_lbl = QLabel("0/0")
        nav_row.addWidget(self._draw_status_lbl)
        nav_row.addStretch()
        self._montage_layout.addLayout(nav_row)

        n = len(cams)
        cols = min(3, n)
        rows = int(math.ceil(n / cols))
        fig = Figure(figsize=(5 * cols, 3.5 * rows), tight_layout=True)
        canvas = FigureCanvasQTAgg(fig)

        axes: dict[str, object] = {}
        im_art: dict[str, object] = {}
        sc_art: dict[str, object] = {}
        empty = np.empty((0, 2))

        for i, cam in enumerate(cams, start=1):
            ax = fig.add_subplot(rows, cols, i)
            axes[cam] = ax
            ims = cam_images[cam]
            if not ims:
                ax.set_title(f"{cam} (no images)")
                ax.axis("off")
                continue
            img0 = mpimg.imread(ims[0])
            im_artist = ax.imshow(img0, cmap="gray" if getattr(img0, "ndim", 3) == 2 else None)
            sc_artist = ax.scatter([], [], s=10, c="lime", marker="o", linewidths=0.4)
            ax.axis("off")
            im_art[cam] = im_artist
            sc_art[cam] = sc_artist

        self._montage_layout.addWidget(canvas)
        self._montage_layout.addStretch()

        self._draw_state = {
            "fig": fig,
            "canvas": canvas,
            "cams": cams,
            "cam_images": cam_images,
            "cam_points": cam_points,
            "axes": axes,
            "im_art": im_art,
            "sc_art": sc_art,
            "max_images": max_images,
            "empty": empty,
            "mpimg": mpimg,
        }
        self._draw_index = 0
        self._update_draw_frame()

    def _step_draw_image(self, delta: int) -> None:
        if not self._draw_state:
            return
        if self._sub_tabs.currentIndex() != 2:
            return
        n = int(self._draw_state.get("max_images", 0))
        if n <= 0:
            return
        self._draw_index = (self._draw_index + delta) % n
        self._update_draw_frame()

    def _update_draw_frame(self) -> None:
        if not self._draw_state:
            return

        cams = self._draw_state["cams"]
        cam_images = self._draw_state["cam_images"]
        cam_points = self._draw_state["cam_points"]
        axes = self._draw_state["axes"]
        im_art = self._draw_state["im_art"]
        sc_art = self._draw_state["sc_art"]
        max_images = int(self._draw_state["max_images"])
        empty = self._draw_state["empty"]
        mpimg = self._draw_state["mpimg"]

        idx = self._draw_index % max_images
        for cam in cams:
            ims = cam_images.get(cam, [])
            ax = axes.get(cam)
            if ax is None:
                continue
            if not ims or cam not in im_art:
                ax.set_title(f"{cam} (no images)")
                continue

            im_idx = idx % len(ims)
            img = mpimg.imread(ims[im_idx])
            im_art[cam].set_data(img)

            pts = cam_points.get(cam, {}).get(im_idx, empty)
            sc_art[cam].set_offsets(pts if len(pts) else empty)
            ax.set_title(f"{cam} | im {im_idx} | pts {len(pts)}")

        if self._draw_status_lbl is not None:
            self._draw_status_lbl.setText(f"{idx + 1}/{max_images}")
        self._draw_state["canvas"].draw_idle()

    def _resolve_pickle_path_for_run(self, run: dict) -> Optional[Path]:
        """Resolve detected_datapoints.pickle path for a Phase 1 run."""
        art_path = run.get("artifacts", {}).get("detected_datapoints_pickle")
        if art_path:
            return Path(art_path)

        run_id = run.get("run_id")
        ws = self._workspace_mgr.workspace_path
        if run_id and ws is not None:
            p = ws / "phase1_runs" / run_id / "detected_datapoints.pickle"
            if p.exists():
                return p

        f_loc = run.get("params", {}).get("f_loc")
        if f_loc:
            return Path(f_loc) / "detected_datapoints.pickle"

        return None

    @staticmethod
    def _extract_detections_obj(payload):
        if hasattr(payload, "get_cam_list"):
            return payload
        if isinstance(payload, (tuple, list)):
            for item in payload:
                if hasattr(item, "get_cam_list"):
                    return item
        if isinstance(payload, dict):
            for key in ("detections", "target_detection", "target_detections", "data"):
                item = payload.get(key)
                if hasattr(item, "get_cam_list"):
                    return item
            for item in payload.values():
                if hasattr(item, "get_cam_list"):
                    return item
        return None

    @staticmethod
    def _clear_layout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                Phase1DiagnosticsTab._clear_layout(item.layout())

    def _continue_to_next(self) -> None:
        selected = self._run_selector.get_selected()

        if not selected:
            if self._info_cb.isChecked():
                QMessageBox.information(
                    self,
                    "Select one run",
                    "Please select exactly one Phase 1 run to continue to Phase 2.",
                )
            return

        if len(selected) > 1:
            if self._info_cb.isChecked():
                QMessageBox.information(
                    self,
                    "Select one run",
                    "Multiple runs are selected. Please select exactly one run to continue to Phase 2.",
                )
            return

        chosen = selected[0]
        self._workspace_mgr.write_handoff({"phase": "phase1", "runs": [chosen]})
        run_id = chosen.get("run_id", "")
        for i in range(self._notebook.count()):
            if self._notebook.tabText(i) == TAB_PHASE2:
                tab = self._notebook.widget(i)
                if hasattr(tab, "set_selected_phase1_run_id"):
                    tab.set_selected_phase1_run_id(run_id)
                self._notebook.setCurrentIndex(i)
                return

