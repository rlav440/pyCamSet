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
from typing import Callable, Optional

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
    BLUE_BTN_STYLE,
    IMAGE_FOLDER_SCHEMATIC,
    TAB_PHASE1,
    RunSelectorWidget,
    TerminalWidget,
    WorkspaceManager,
    count_images_in_folder,
    get_camera_subfolders,
    make_blue_button,
    make_continue_button,
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
        self._phase1_path_cb: Optional[Callable[[str], None]] = None
        self._cameras_cb: Optional[Callable[..., None]] = None
        self._confirmed_floc: Optional[str] = None
        self._camera_names: list[str] = []
        self._cam_checkboxes: dict[str, QCheckBox] = {}
        self._rebuilding_cameras = False
        self._build_ui(terminal_cb)

    def set_phase1_path_callback(self, cb: Callable[[str], None]) -> None:
        self._phase1_path_cb = cb

    def set_cameras_callback(self, cb: Callable[..., None]) -> None:
        """Register a callback that receives camera names and selected subset."""
        self._cameras_cb = cb

    def get_camera_names(self) -> list[str]:
        """Return the list of camera names discovered in the last successful validation."""
        return list(self._camera_names)

    def get_selected_cameras(self) -> list[str]:
        """Return currently checked camera names in visual order."""
        return [name for name in self._camera_names if self._cam_checkboxes.get(name, None) and self._cam_checkboxes[name].isChecked()]

    def set_image_folder(self, path: str) -> None:
        """Set the image folder path (for global sync; does not re-validate)."""
        if self._floc_edit.text() != path:
            self._floc_edit.setText(path)

    def set_camera_names(self, names: list[str], selected: Optional[list[str]] = None) -> None:
        """Update camera checkboxes without re-running validation.

        Called from main_window when camera names are propagated cross-tab.
        Preserves existing checked state for cameras that remain in the list.
        """
        if names == self._camera_names and selected is None:
            return  # no change — avoid redundant rebuilds
        if names == self._camera_names and selected is not None:
            selected_set = set(selected)
            current_selected = set(self.get_selected_cameras())
            if current_selected == selected_set:
                return
        # Preserve checked state for cameras that remain unless explicit selected set is supplied.
        if selected is None:
            old_states = {n: cb.isChecked() for n, cb in self._cam_checkboxes.items()}
        else:
            selected_set = set(selected)
            old_states = {n: (n in selected_set) for n in names}
        self._camera_names = list(names)
        self._rebuild_camera_checkboxes(restore_states=old_states)

    def _rebuild_camera_checkboxes(self, restore_states: Optional[dict] = None) -> None:
        """Rebuild the camera checkbox list from the current _camera_names."""
        self._rebuilding_cameras = True
        while self._cameras_layout.count():
            item = self._cameras_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cam_checkboxes.clear()
        if not self._camera_names:
            lbl = QLabel("(no cameras found)")
            lbl.setStyleSheet("color: gray; font-size: 10px;")
            self._cameras_layout.addWidget(lbl)
            self._rebuilding_cameras = False
            return
        for name in self._camera_names:
            cb = QCheckBox(name)
            # Restore previous checked state, or default to True for new cameras.
            cb.setChecked(True if restore_states is None else restore_states.get(name, True))
            cb.stateChanged.connect(lambda _state: self._emit_cameras_changed())
            self._cam_checkboxes[name] = cb
            self._cameras_layout.addWidget(cb)
        self._rebuilding_cameras = False

    def _emit_cameras_changed(self) -> None:
        if self._cameras_cb is None:
            return
        if self._rebuilding_cameras:
            return
        names = list(self._camera_names)
        selected = self.get_selected_cameras()
        try:
            self._cameras_cb(names, selected)
        except TypeError:
            self._cameras_cb(names)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self, terminal_cb: QCheckBox) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setContentsMargins(0, 0, 0, 0)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        root.addWidget(form_widget)

        form.addRow(make_section_label("Paths"))

        floc_row = QHBoxLayout()
        self._floc_edit = QLineEdit()
        self._floc_edit.setPlaceholderText("Root folder with per-camera sub-folders")
        self._floc_edit.setToolTip(IMAGE_FOLDER_SCHEMATIC)
        self._floc_edit.textChanged.connect(self._on_floc_change)
        floc_btn = QPushButton("Browse…")
        floc_btn.setFixedWidth(70)
        floc_btn.setToolTip(IMAGE_FOLDER_SCHEMATIC)
        floc_btn.clicked.connect(self._browse_floc)
        floc_row.addWidget(self._floc_edit)
        floc_row.addWidget(floc_btn)
        form.addRow("Image folder (f_loc):", floc_row)

        self._ws_edit = QLineEdit()
        self._ws_edit.setPlaceholderText("<f_loc>/.pycamset_workspace")
        self._ws_edit.setToolTip(
            "Workspace is fixed to <image_folder>/.pycamset_workspace."
        )
        self._ws_edit.setReadOnly(True)
        self._ws_edit.setEnabled(False)
        form.addRow("Workspace:", self._ws_edit)

        btn_row = QHBoxLayout()
        self._confirm_btn = make_blue_button("Confirm Image Folder Validity", self._confirm_image_folder_validity)
        self._confirm_btn.setToolTip(
            "Validate that camera subfolders are present and each has "
            "the same non-zero image count."
        )
        btn_row.addWidget(self._confirm_btn)

        self._ok_lbl = QLabel("")
        self._ok_lbl.setStyleSheet("color: #2e7d32; font-size: 16px;")
        btn_row.addWidget(self._ok_lbl)

        self._continue_btn = make_continue_button(self._continue_to_next)
        self._continue_btn.setToolTip(
            "Proceed to Phase 1 after successful folder validation."
        )
        self._continue_btn.setEnabled(False)
        btn_row.addWidget(self._continue_btn)
        btn_row.addStretch()
        form.addRow(btn_row)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #2e7d32; font-size: 11px;")
        self._status_lbl.setWordWrap(True)
        form.addRow("", self._status_lbl)

        # ── Camera checkboxes (populated after validation) ─────────────
        form.addRow(make_separator())
        form.addRow(make_section_label("Discovered Cameras"))
        self._cameras_area = QWidget()
        self._cameras_layout = QVBoxLayout(self._cameras_area)
        self._cameras_layout.setContentsMargins(0, 0, 0, 0)
        self._cameras_layout.setSpacing(2)
        self._cameras_placeholder = QLabel("(confirm image folder to populate)")
        self._cameras_placeholder.setStyleSheet("color: gray; font-size: 10px;")
        self._cameras_layout.addWidget(self._cameras_placeholder)
        cam_scroll = QScrollArea()
        cam_scroll.setWidgetResizable(True)
        cam_scroll.setFixedHeight(100)
        cam_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        cam_scroll.setWidget(self._cameras_area)
        form.addRow(cam_scroll)

        self._terminal = TerminalWidget(terminal_cb, parent=self)
        root.addWidget(self._terminal)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _browse_floc(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select image folder")
        if path:
            self._floc_edit.setText(path)

    def _on_floc_change(self, text: str) -> None:
        floc = text.strip()
        self._ws_edit.setText(str(Path(floc) / ".pycamset_workspace") if floc else "")

    def _confirm_image_folder_validity(self) -> None:
        floc = self._floc_edit.text().strip()
        if not floc:
            QMessageBox.critical(self, "Validation Error", "Image folder (f_loc) is required.")
            return

        f_loc = Path(floc)
        if not f_loc.exists() or not f_loc.is_dir():
            QMessageBox.critical(self, "Validation Error", "Selected image folder does not exist.")
            return

        cam_folders = get_camera_subfolders(f_loc)
        img_counts = {p.name: count_images_in_folder(p) for p in cam_folders}
        counts = list(img_counts.values())
        is_valid = len(cam_folders) >= 2 and counts and all(c > 0 for c in counts) and len(set(counts)) == 1

        self._terminal.clear_terminal()
        self._terminal.append_line("=== Phase 0: Data Input Validation ===")
        self._terminal.append_line(f"Image folder : {f_loc}")
        self._terminal.append_line(f"Candidate camera folders: {[p.name for p in cam_folders]}")
        for name, n in img_counts.items():
            self._terminal.append_line(f"  {name}: {n} image(s)")

        if not is_valid:
            self._confirmed_floc = None
            self._ok_lbl.setText("")
            self._continue_btn.setEnabled(False)
            self._status_lbl.setText("")
            QMessageBox.warning(
                self,
                "Image Folder Not Valid",
                "Could not confirm folder structure.\n\n"
                "Requirements:\n"
                "- Multiple camera subfolders\n"
                "- Same number of images in each camera folder\n"
                "- At least one image per camera folder\n\n"
                f"{IMAGE_FOLDER_SCHEMATIC}",
            )
            return

        self._confirmed_floc = str(f_loc)
        self._ok_lbl.setText("✅")
        self._continue_btn.setEnabled(True)
        self._status_lbl.setText(
            "Image path appears to be organized correctly. Ready for next phase."
        )

        ws_path = f_loc / ".pycamset_workspace"
        self._ws_edit.setText(str(ws_path))
        self._workspace_mgr.set_workspace_path(ws_path, ensure=True)

        # Rebuild camera checkboxes while preserving any existing user selection.
        old_states = {n: cb.isChecked() for n, cb in self._cam_checkboxes.items()}
        self._camera_names = [p.name for p in cam_folders]
        self._rebuild_camera_checkboxes(restore_states=old_states)

        if self._phase1_path_cb is not None:
            self._phase1_path_cb(str(f_loc))

        self._emit_cameras_changed()

        self._terminal.append_line("Validation passed ✓")


    def _continue_to_next(self) -> None:
        if not self._confirmed_floc:
            QMessageBox.information(
                self,
                "Validation Required",
                "Confirm image folder validity first.",
            )
            return
        self._workspace_mgr.write_handoff(
            {
                "phase": "phase0",
                "runs": [
                    {
                        "params": {
                            "f_loc": self._confirmed_floc,
                            "selected_cameras": self.get_selected_cameras(),
                        }
                    }
                ],
                "selected_cameras": self.get_selected_cameras(),
            }
        )
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
