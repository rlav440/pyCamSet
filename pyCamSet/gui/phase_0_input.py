"""
Phase 0 — Input Validation & Setup tab and diagnostics (PySide6).

Wraps the top section of
:func:`pyCamSet.calibration.camera_calibrator.calibrate_cameras` (lines 49–66)
into a GUI tab that validates parameters and computes diagnostics D0.1–D0.3.

Diagnostics implemented
-----------------------
- **D0.1** Number of camera sub-folders (``get_subfolder_names``).
- **D0.2** Number of images per camera (``glob_ims``).
- **D0.3** Image-count consistency (``sanitise_input_images``).
"""
from __future__ import annotations

import json
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pyCamSet.gui.shared_functions import (
    TAB_PHASE0_DIAG,
    TAB_PHASE1,
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

# pyCamSet helpers — guarded import
try:
    from pyCamSet.utils.general_utils import get_subfolder_names, glob_ims
    from pyCamSet.calibration.camera_calibrator import sanitise_input_images
    _PYCAMSET_OK = True
except ImportError:
    _PYCAMSET_OK = False


class Phase0Tab(QWidget):
    """Phase 0 — Input Validation & Setup tab.

    Presents all nine user-tunable parameters and runs diagnostics D0.1–D0.3
    using existing pyCamSet helpers.
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
        self._diagnostics_tab: Optional["Phase0DiagnosticsTab"] = None
        self._worker: Optional[PhaseWorker] = None
        self._build_ui(terminal_cb)

    def set_diagnostics_tab(self, tab: "Phase0DiagnosticsTab") -> None:
        self._diagnostics_tab = tab

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self, terminal_cb: QCheckBox) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Main form + side button panel ─────────────────────────────
        top_row = QHBoxLayout()
        root.addLayout(top_row)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setContentsMargins(0, 0, 0, 0)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        top_row.addWidget(form_widget, stretch=1)

        # Side panel with buttons
        side = QWidget()
        side.setFixedWidth(200)
        side_layout = QVBoxLayout(side)
        side_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        top_row.addWidget(side)

        # ── Paths section ──────────────────────────────────────────────
        form.addRow(make_section_label("Paths"))

        # f_loc
        floc_row = QHBoxLayout()
        self._floc_edit = QLineEdit()
        self._floc_edit.setPlaceholderText("Root folder with per-camera sub-folders")
        self._floc_edit.textChanged.connect(self._on_floc_change)
        floc_btn = QPushButton("Browse…")
        floc_btn.setFixedWidth(70)
        floc_btn.clicked.connect(self._browse_floc)
        floc_row.addWidget(self._floc_edit)
        floc_row.addWidget(floc_btn)
        form.addRow("Image folder (f_loc):", floc_row)

        # save_loc
        saveloc_row = QHBoxLayout()
        self._saveloc_edit = QLineEdit()
        self._saveloc_edit.setPlaceholderText("Defaults to f_loc when blank")
        saveloc_btn = QPushButton("Browse…")
        saveloc_btn.setFixedWidth(70)
        saveloc_btn.clicked.connect(self._browse_saveloc)
        saveloc_row.addWidget(self._saveloc_edit)
        saveloc_row.addWidget(saveloc_btn)
        form.addRow("Save location:", saveloc_row)

        # workspace
        self._ws_edit = QLineEdit()
        self._ws_edit.setPlaceholderText("<f_loc>/.pycamset_workspace")
        form.addRow("Workspace:", self._ws_edit)

        # ── Options ────────────────────────────────────────────────────
        form.addRow(make_separator())
        form.addRow(make_section_label("Options"))

        self._save_cb = QCheckBox("Save artefacts")
        self._save_cb.setChecked(True)
        self._save_cb.setToolTip("Cache detection pickle and camera-set files to disk.")
        form.addRow(self._save_cb)

        self._draw_cb = QCheckBox("Draw detections")
        self._draw_cb.setToolTip("Render detection overlays as each image is processed.")
        form.addRow(self._draw_cb)

        self._hd_cb = QCheckBox("High distortion mode")
        self._hd_cb.setToolTip(
            "Re-detect with undistorted images after an initial calibration pass."
        )
        form.addRow(self._hd_cb)

        self._nlim_edit = QLineEdit()
        self._nlim_edit.setPlaceholderText("blank = no limit")
        self._nlim_edit.setFixedWidth(100)
        self._nlim_edit.setToolTip("Max images to use per camera.")
        form.addRow("Max images per camera (n_lim):", self._nlim_edit)

        auto_threads = min(max(1, (cpu_count() or 1) - 2), 20)
        self._threads_edit = QLineEdit()
        self._threads_edit.setPlaceholderText(f"blank = auto ({auto_threads})")
        self._threads_edit.setFixedWidth(100)
        self._threads_edit.setToolTip(
            f"Thread count.  Auto = min(max(1, cpu-2), 20) = {auto_threads} here."
        )
        form.addRow("Threads:", self._threads_edit)

        self._fp_edit = QLineEdit()
        self._fp_edit.setPlaceholderText('e.g. {"cam0": "int"} or blank')
        self._fp_edit.setToolTip(
            "JSON dict to lock camera parameters.  Leave blank for none."
        )
        form.addRow("Fixed params (JSON):", self._fp_edit)

        self._po_edit = QLineEdit()
        self._po_edit.setPlaceholderText("JSON dict or blank")
        self._po_edit.setToolTip("JSON dict of options forwarded to the optimiser.")
        form.addRow("Problem options (JSON):", self._po_edit)

        # ── Action buttons (bottom of form) ───────────────────────────
        btn_row = QHBoxLayout()
        run_btn = QPushButton("▶  Run Phase 0")
        run_btn.clicked.connect(self._run_phase0)
        btn_row.addWidget(run_btn)
        btn_row.addWidget(make_orange_button("Diagnostics ▼", self._open_diagnostics))
        btn_row.addStretch()
        form.addRow(btn_row)

        # ── Side panel continue button ─────────────────────────────────
        side_layout.addWidget(make_continue_button(self._continue_to_next))

        # ── Terminal ──────────────────────────────────────────────────
        self._terminal = TerminalWidget(terminal_cb, parent=self)
        root.addWidget(self._terminal)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _browse_floc(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select image folder")
        if path:
            self._floc_edit.setText(path)

    def _browse_saveloc(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select save location")
        if path:
            self._saveloc_edit.setText(path)

    def _on_floc_change(self, text: str) -> None:
        if text.strip() and not self._ws_edit.text().strip():
            self._ws_edit.setText(str(Path(text.strip()) / ".pycamset_workspace"))

    def _collect_params(self) -> Optional[dict]:
        floc = self._floc_edit.text().strip()
        if not floc:
            QMessageBox.critical(self, "Validation Error", "Image folder (f_loc) is required.")
            return None

        save_loc = self._saveloc_edit.text().strip() or None

        n_lim = None
        if self._nlim_edit.text().strip():
            try:
                n_lim = int(self._nlim_edit.text().strip())
            except ValueError:
                QMessageBox.critical(self, "Validation Error", "n_lim must be an integer.")
                return None

        threads = None
        if self._threads_edit.text().strip():
            try:
                threads = int(self._threads_edit.text().strip())
            except ValueError:
                QMessageBox.critical(self, "Validation Error", "Threads must be an integer.")
                return None

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
            "save_loc": save_loc,
            "save": self._save_cb.isChecked(),
            "draw": self._draw_cb.isChecked(),
            "n_lim": n_lim,
            "threads": threads,
            "high_distortion": self._hd_cb.isChecked(),
            "fixed_params": fixed_params,
            "problem_options": problem_options,
        }

    def _run_phase0(self) -> None:
        params = self._collect_params()
        if params is None:
            return

        ws_str = self._ws_edit.text().strip()
        if not ws_str:
            ws_str = str(Path(params["f_loc"]) / ".pycamset_workspace")
        self._workspace_mgr.workspace_path = Path(ws_str)
        self._workspace_mgr.ensure_dirs()

        self._terminal.clear_terminal()
        self._terminal.append_line("=== Phase 0: Input Validation & Setup ===")
        self._terminal.append_line(f"Image folder : {params['f_loc']}")
        self._terminal.append_line(f"Save         : {params['save']}")
        self._terminal.append_line(f"draw         : {params['draw']}")
        self._terminal.append_line(f"n_lim        : {params['n_lim']}")
        self._terminal.append_line(f"threads      : {params['threads'] or 'auto'}")
        self._terminal.append_line(f"high_distort : {params['high_distortion']}")
        self._terminal.append_line("Running diagnostics…")

        def work_fn(emit: callable) -> dict:
            diagnostics: dict = {}
            error_msg: Optional[str] = None
            try:
                f_loc = Path(params["f_loc"])

                if _PYCAMSET_OK:
                    # D0.1 — existing get_subfolder_names
                    cam_folders = get_subfolder_names(f_loc, return_full_path=True)
                    cam_names = get_subfolder_names(f_loc)
                    diagnostics["D0.1_n_cameras"] = len(cam_folders)
                    emit(f"D0.1  Camera sub-folders : {len(cam_folders)}")

                    # D0.2 — existing glob_ims
                    imgs_per_cam: dict[str, int] = {}
                    for name, path in zip(cam_names, cam_folders):
                        imgs_per_cam[name] = len(glob_ims(path))
                    diagnostics["D0.2_images_per_camera"] = imgs_per_cam
                    for name, n in imgs_per_cam.items():
                        emit(f"      {name}: {n} images")

                    # D0.3 — existing sanitise_input_images
                    try:
                        sanitise_input_images(cam_folders)
                        consistent = True
                    except ValueError:
                        consistent = False
                    diagnostics["D0.3_consistent"] = consistent
                    emit(
                        f"D0.3  Image count consistent: "
                        f"{'Yes ✓' if consistent else 'No ✗ (WARNING)'}"
                    )
                else:
                    # Fallback: plain filesystem scan
                    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
                    cam_folders = [p for p in f_loc.iterdir() if p.is_dir()]
                    diagnostics["D0.1_n_cameras"] = len(cam_folders)
                    imgs_per_cam = {
                        cf.name: len([p for p in cf.iterdir() if p.suffix.lower() in exts])
                        for cf in cam_folders
                    }
                    diagnostics["D0.2_images_per_camera"] = imgs_per_cam
                    counts = list(imgs_per_cam.values())
                    diagnostics["D0.3_consistent"] = len(set(counts)) <= 1
                    emit("(pyCamSet not found — basic scan only)")

                emit("Phase 0 diagnostics complete.")

            except Exception as exc:
                error_msg = str(exc)
                diagnostics["error"] = error_msg

            run_id = make_run_id()
            metadata = {
                "run_id": run_id,
                "phase": "phase0",
                "params": params,
                "diagnostics": diagnostics,
                "error": error_msg,
            }
            self._workspace_mgr.save_run("phase0", run_id, metadata)
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
            if self._notebook.tabText(i) == TAB_PHASE0_DIAG:
                self._notebook.setCurrentIndex(i)
                return

    def _continue_to_next(self) -> None:
        runs = self._workspace_mgr.load_runs("phase0")
        if not runs:
            QMessageBox.information(self, "No runs", "Run Phase 0 first.")
            return
        self._workspace_mgr.write_handoff({"phase": "phase0", "runs": [runs[-1]]})
        for i in range(self._notebook.count()):
            if self._notebook.tabText(i) == TAB_PHASE1:
                self._notebook.setCurrentIndex(i)
                return


# ---------------------------------------------------------------------------
# Phase 0 Diagnostics tab
# ---------------------------------------------------------------------------


class Phase0DiagnosticsTab(QWidget):
    """Phase 0 Diagnostics — run comparison view.

    Left pane: :class:`~pyCamSet.gui.shared_functions.RunSelectorWidget`.
    Right pane: scrollable table of D0.1–D0.3 results for the selected run(s).
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
        self._run_selector.selection_changed.connect(self._render_diagnostics)
        left_layout.addWidget(self._run_selector)

        refresh_btn = QPushButton("⟳  Refresh")
        refresh_btn.clicked.connect(self.refresh)
        left_layout.addWidget(refresh_btn)
        splitter.addWidget(left)

        # ── Right: scrollable diagnostics ─────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)

        right_layout.addWidget(make_section_label("Phase 0 Diagnostics"))
        right_layout.addWidget(make_separator())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._diag_container = QWidget()
        self._diag_layout = QVBoxLayout(self._diag_container)
        self._diag_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self._diag_container)
        right_layout.addWidget(scroll)
        splitter.addWidget(right)

        splitter.setSizes([200, 700])

        # ── Bottom: continue button ────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(make_continue_button(self._continue_to_next))
        root.addLayout(btn_row)

        self.refresh()

    def refresh(self) -> None:
        runs = self._workspace_mgr.load_runs("phase0")
        self._run_selector.refresh(runs)
        self._render_diagnostics(self._run_selector.get_selected())

    def _render_diagnostics(self, runs: list[dict]) -> None:
        # Clear existing widgets
        while self._diag_layout.count():
            item = self._diag_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not runs:
            lbl = QLabel("Select one or more runs from the list to compare.")
            lbl.setStyleSheet("color: gray;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._diag_layout.addWidget(lbl)
            return

        for run in runs:
            run_id = run.get("run_id", "unknown")
            hdr = QLabel(f"Run: {run_id}")
            hdr.setStyleSheet("font-weight: bold; margin-top: 8px;")
            self._diag_layout.addWidget(hdr)

            diag = run.get("diagnostics", {})
            params = run.get("params", {})

            entries = [
                ("f_loc", params.get("f_loc", "—")),
                ("D0.1  Camera sub-folders", diag.get("D0.1_n_cameras", "—")),
                ("D0.3  Image count consistent", diag.get("D0.3_consistent", "—")),
            ]
            for cam, n in diag.get("D0.2_images_per_camera", {}).items():
                entries.append((f"  D0.2  {cam}", f"{n} images"))
            if run.get("error"):
                entries.append(("Error", run["error"]))

            form = QFormLayout()
            form.setContentsMargins(16, 0, 0, 0)
            for key, val in entries:
                form.addRow(f"{key}:", QLabel(str(val)))
            self._diag_layout.addLayout(form)

            self._diag_layout.addWidget(make_separator())

    def _continue_to_next(self) -> None:
        selected = self._run_selector.get_selected()
        if not selected:
            runs = self._workspace_mgr.load_runs("phase0")
            selected = runs[-1:] if runs else []
        if not selected:
            QMessageBox.information(self, "No runs", "No Phase 0 runs available.")
            return
        self._workspace_mgr.write_handoff({"phase": "phase0", "runs": selected})
        for i in range(self._notebook.count()):
            if self._notebook.tabText(i) == TAB_PHASE1:
                self._notebook.setCurrentIndex(i)
                return
