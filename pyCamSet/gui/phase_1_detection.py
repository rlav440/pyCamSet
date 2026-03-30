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
from typing import Optional

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
    TAB_PHASE1_DIAG,
    PhaseWorker,
    RunSelectorWidget,
    TerminalWidget,
    WorkspaceManager,
    make_continue_button,
    make_orange_button,
    make_run_id,
    make_section_label,
    make_separator,
)

# pyCamSet guarded imports
try:
    from pyCamSet.calibration.camera_calibrator import (
        detect_datapoints_in_imfile,
        validate_detections,
    )
    from pyCamSet.calibration_targets.target_Ccube import Ccube
    from pyCamSet.calibration_targets.target_charuco import ChArUco
    from pyCamSet.utils.general_utils import get_subfolder_names
    _PYCAMSET_OK = True
except ImportError:
    _PYCAMSET_OK = False

_TARGET_CHOICES = ["Ccube", "ChArUco"]


def _build_target(target_type: str, n_points: int, length: float):
    """Construct the calibration target object from existing pyCamSet classes."""
    if not _PYCAMSET_OK:
        raise RuntimeError("pyCamSet calibration targets are not importable.")
    if target_type == "Ccube":
        return Ccube(n_points=n_points, length=length)
    if target_type == "ChArUco":
        return ChArUco(
            num_squares_x=n_points,
            num_squares_y=n_points,
            square_size=length,
        )
    raise ValueError(f"Unknown target type: {target_type!r}")


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
        form = QFormLayout(form_widget)
        form.setContentsMargins(0, 0, 0, 0)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        top_row.addWidget(form_widget, stretch=1)

        side = QWidget()
        side.setFixedWidth(200)
        side_layout = QVBoxLayout(side)
        side_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        top_row.addWidget(side)

        # ── Paths ──────────────────────────────────────────────────────
        form.addRow(make_section_label("Paths"))

        floc_row = QHBoxLayout()
        self._floc_edit = QLineEdit()
        self._floc_edit.setPlaceholderText("Root folder with per-camera sub-folders")
        floc_btn = QPushButton("Browse…")
        floc_btn.setFixedWidth(70)
        floc_btn.clicked.connect(self._browse_floc)
        floc_row.addWidget(self._floc_edit)
        floc_row.addWidget(floc_btn)
        form.addRow("Image folder (f_loc):", floc_row)

        # ── Detection options ──────────────────────────────────────────
        form.addRow(make_separator())
        form.addRow(make_section_label("Detection Options"))

        self._draw_cb = QCheckBox("Draw detections (draw)")
        self._draw_cb.setToolTip(
            "Render corner overlays on each image as it is processed."
        )
        form.addRow(self._draw_cb)

        self._cache_cb = QCheckBox("Cache detections (caching)")
        self._cache_cb.setChecked(True)
        self._cache_cb.setToolTip(
            "Save/load detected_datapoints.pickle.  "
            "Disable to force re-detection."
        )
        form.addRow(self._cache_cb)

        self._nlim_edit = QLineEdit()
        self._nlim_edit.setPlaceholderText("blank = no limit")
        self._nlim_edit.setFixedWidth(100)
        self._nlim_edit.setToolTip("Max images per camera folder.")
        form.addRow("Max images per camera (n_lim):", self._nlim_edit)

        # ── Target configuration ───────────────────────────────────────
        form.addRow(make_separator())
        form.addRow(make_section_label("Calibration Target"))

        self._target_combo = QComboBox()
        self._target_combo.addItems(_TARGET_CHOICES)
        self._target_combo.setFixedWidth(140)
        self._target_combo.setToolTip(
            "Ccube = corner-cube target; ChArUco = charuco board."
        )
        form.addRow("Target type:", self._target_combo)

        self._npts_spin = QSpinBox()
        self._npts_spin.setRange(2, 20)
        self._npts_spin.setValue(6)
        self._npts_spin.setFixedWidth(80)
        self._npts_spin.setToolTip(
            "For Ccube: points per face edge.  For ChArUco: squares in x."
        )
        form.addRow("n_points / squares_x:", self._npts_spin)

        self._length_edit = QLineEdit("30.0")
        self._length_edit.setFixedWidth(100)
        self._length_edit.setToolTip("Physical size of the target feature in millimetres.")
        form.addRow("Length / square size (mm):", self._length_edit)

        # ── Action buttons ─────────────────────────────────────────────
        btn_row = QHBoxLayout()
        run_btn = QPushButton("▶  Run Phase 1")
        run_btn.clicked.connect(self._run_phase1)
        btn_row.addWidget(run_btn)
        btn_row.addWidget(make_orange_button("Diagnostics ▼", self._open_diagnostics))
        btn_row.addStretch()
        form.addRow(btn_row)

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

        return {
            "f_loc": floc,
            "draw": self._draw_cb.isChecked(),
            "caching": self._cache_cb.isChecked(),
            "n_lim": n_lim,
            "target_type": self._target_combo.currentText(),
            "n_points": self._npts_spin.value(),
            "length": length,
        }

    def _run_phase1(self) -> None:
        params = self._collect_params()
        if params is None:
            return

        self._terminal.clear_terminal()
        self._terminal.append_line("=== Phase 1: Target Detection ===")
        self._terminal.append_line(f"Image folder : {params['f_loc']}")
        self._terminal.append_line(
            f"Target       : {params['target_type']} "
            f"(n={params['n_points']}, length={params['length']} mm)"
        )
        self._terminal.append_line(f"caching      : {params['caching']}")
        self._terminal.append_line(f"draw         : {params['draw']}")
        self._terminal.append_line(f"n_lim        : {params['n_lim']}")
        self._terminal.append_line("Starting detection…")

        def work_fn(emit: callable) -> dict:
            diagnostics: dict = {}
            error_msg: Optional[str] = None

            try:
                f_loc = Path(params["f_loc"])

                if not _PYCAMSET_OK:
                    raise RuntimeError(
                        "pyCamSet calibration module not importable.  "
                        "Check your installation."
                    )

                # 1a — existing get_subfolder_names
                cam_names = get_subfolder_names(f_loc)
                emit(f"1a  Camera sub-folders: {cam_names}")

                # Build target from existing pyCamSet classes
                target = _build_target(
                    params["target_type"],
                    params["n_points"],
                    params["length"],
                )

                # 1b — existing detect_datapoints_in_imfile
                detections, cam_res = detect_datapoints_in_imfile(
                    f_loc=f_loc,
                    calibration_target=target,
                    caching=params["caching"],
                    draw=params["draw"],
                    n_lim=params["n_lim"],
                )
                emit("1b  Detection complete.")

                # 1c — existing validate_detections
                validate_detections(detections, target)
                emit("1c  Validation complete.")

                # ── Diagnostics ──────────────────────────────────────
                try:
                    # D1.1 — existing get_cam_list
                    total_per_cam: dict[str, int] = {}
                    for cam_det in detections.get_cam_list():
                        cam_idx = int(cam_det.get_data()[0, 0])
                        cam_name = detections.cam_names[cam_idx]
                        total_per_cam[cam_name] = len(cam_det.get_data())
                    diagnostics["D1.1_total_detections"] = total_per_cam
                    for cam, n in total_per_cam.items():
                        emit(f"D1.1  {cam}: {n} detections")

                    # D1.2 + D1.3 — replicate validate_detections logic
                    corners_per_face = target.point_data.shape[-2]
                    det_rate: dict[str, float] = {}
                    completeness: dict[str, float] = {}
                    for cam_det in detections.get_cam_list():
                        cam_idx = int(cam_det.get_data()[0, 0])
                        cam_name = detections.cam_names[cam_idx]
                        detected_boards = 0
                        fracs: list[float] = []
                        for im_det in cam_det.get_image_list():
                            datum = im_det.get_data()
                            if datum is not None:
                                detected_boards += 1
                                n_keys = datum.shape[1] - 4
                                if n_keys == 1:
                                    fracs.append(datum.shape[0] / corners_per_face)
                                else:
                                    n_boards = len(np.unique(datum[:, 2:-2], axis=0))
                                    fracs.append(
                                        datum.shape[0] / corners_per_face / max(n_boards, 1)
                                    )
                        det_rate[cam_name] = detected_boards / detections.max_ims
                        completeness[cam_name] = float(np.mean(fracs)) if fracs else 0.0
                    diagnostics["D1.2_detection_rate"] = det_rate
                    diagnostics["D1.3_board_completeness"] = completeness
                    for cam in detections.cam_names:
                        r = det_rate.get(cam, 0) * 100
                        c = completeness.get(cam, 0) * 100
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

            run_id = make_run_id()
            metadata = {
                "run_id": run_id,
                "phase": "phase1",
                "params": params,
                "diagnostics": diagnostics,
                "error": error_msg,
            }
            self._workspace_mgr.save_run("phase1", run_id, metadata)
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
        for i in range(self._notebook.count()):
            if self._notebook.tabText(i) == TAB_PHASE1_DIAG:
                self._notebook.setCurrentIndex(i)
                return

    def _continue_to_next(self) -> None:
        runs = self._workspace_mgr.load_runs("phase1")
        if not runs:
            QMessageBox.information(self, "No runs", "Run Phase 1 first.")
            return
        self._workspace_mgr.write_handoff({"phase": "phase1", "runs": [runs[-1]]})
        QMessageBox.information(
            self,
            "Handoff written",
            "handoff.json written to workspace.\nProceed to Phase 2.",
        )


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
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

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

        # Summary sub-tab
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

        # Heatmap sub-tab
        self._heatmap_widget = QWidget()
        self._heatmap_layout = QVBoxLayout(self._heatmap_widget)
        self._sub_tabs.addTab(self._heatmap_widget, "Heatmap (D1.4)")

        # Montage sub-tab
        self._montage_widget = QWidget()
        self._montage_layout = QVBoxLayout(self._montage_widget)
        self._sub_tabs.addTab(self._montage_widget, "Montage (D1.5)")

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
        self._render_summary(runs)
        self._render_heatmap(runs)
        self._render_montage(runs)

    # ------------------------------------------------------------------

    def _render_summary(self, runs: list[dict]) -> None:
        while self._summary_layout.count():
            item = self._summary_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

        if not runs:
            lbl = QLabel("Select one or more runs from the list to compare.")
            lbl.setStyleSheet("color: gray;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._summary_layout.addWidget(lbl)
            return

        for run in runs:
            run_id = run.get("run_id", "unknown")
            hdr = QLabel(f"Run: {run_id}")
            hdr.setStyleSheet("font-weight: bold; margin-top: 8px;")
            self._summary_layout.addWidget(hdr)

            d = run.get("diagnostics", {})
            p = run.get("params", {})
            cam_names = d.get("cam_names", list(d.get("D1.1_total_detections", {}).keys()))

            form = QFormLayout()
            form.setContentsMargins(16, 0, 0, 0)
            form.addRow("f_loc:", QLabel(str(p.get("f_loc", "—"))))
            form.addRow(
                "target:",
                QLabel(
                    f"{p.get('target_type','—')} n={p.get('n_points','—')}"
                    f" L={p.get('length','—')} mm"
                ),
            )
            form.addRow("D1.7  Min features:", QLabel(str(d.get("D1.7_min_features", "—"))))
            self._summary_layout.addLayout(form)

            if cam_names:
                # Table header
                hdr_row = QHBoxLayout()
                for hdr_text, stretch in [
                    ("Camera", 2),
                    ("D1.1 Detections", 1),
                    ("D1.2 Rate %", 1),
                    ("D1.3 Completeness %", 1),
                    ("D1.6 Coverage", 1),
                ]:
                    lbl = QLabel(hdr_text)
                    lbl.setStyleSheet("font-weight: bold;")
                    hdr_row.addWidget(lbl, stretch=stretch)
                self._summary_layout.addLayout(hdr_row)

                det = d.get("D1.1_total_detections", {})
                rate = d.get("D1.2_detection_rate", {})
                comp = d.get("D1.3_board_completeness", {})
                cov = d.get("D1.6_spatial_coverage", {})

                for cam in cam_names:
                    cam_row = QHBoxLayout()
                    for val, stretch in [
                        (cam, 2),
                        (str(det.get(cam, "—")), 1),
                        (f"{rate.get(cam, 0)*100:.1f}" if cam in rate else "—", 1),
                        (f"{comp.get(cam, 0)*100:.1f}" if cam in comp else "—", 1),
                        (f"{cov.get(cam, float('nan')):.3f}" if cam in cov else "—", 1),
                    ]:
                        cam_row.addWidget(QLabel(val), stretch=stretch)
                    self._summary_layout.addLayout(cam_row)

            if run.get("error"):
                err_lbl = QLabel(f"Error: {run['error']}")
                err_lbl.setStyleSheet("color: red;")
                err_lbl.setWordWrap(True)
                self._summary_layout.addWidget(err_lbl)

            self._summary_layout.addWidget(make_separator())

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

        self._montage_layout.addWidget(make_section_label("D1.5 — Detection overlay montage"))

        msg = QLabel(
            "Detection overlays are rendered live when the Phase 1 run is executed "
            "with 'Draw detections' enabled (draw=True).  Each frame is shown via "
            "OpenCV's imshow() as it is processed.\n\n"
            "The raw detection data is saved to:\n"
            "  <f_loc>/detected_datapoints.pickle\n\n"
            "You can reload and visualise detections at any time by loading that "
            "file via pyCamSet.utils.saving.load_pickle() and calling "
            "target.find_in_imfolder(..., draw=True) again with caching=False."
        )
        msg.setWordWrap(True)
        self._montage_layout.addWidget(msg)

        for run in runs:
            floc = run.get("params", {}).get("f_loc", "")
            if floc:
                pickle_path = Path(floc) / "detected_datapoints.pickle"
                exists = "✓ exists" if pickle_path.exists() else "✗ not found"
                color = "#2e7d32" if pickle_path.exists() else "#b71c1c"
                path_lbl = QLabel(f"{pickle_path}  [{exists}]")
                path_lbl.setStyleSheet(f"color: {color};")
                path_lbl.setWordWrap(True)
                self._montage_layout.addWidget(path_lbl)

        self._montage_layout.addStretch()

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
            runs = self._workspace_mgr.load_runs("phase1")
            selected = runs[-1:] if runs else []
        if not selected:
            QMessageBox.information(self, "No runs", "No Phase 1 runs available.")
            return
        self._workspace_mgr.write_handoff({"phase": "phase1", "runs": selected})
        QMessageBox.information(
            self,
            "Handoff written",
            "handoff.json written to workspace.\nProceed to Phase 2.",
        )
