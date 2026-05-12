"""
pyCamSet GUI — main application window (PySide6).

Launch with::

    python -m pyCamSet.gui
"""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pyCamSet.gui.shared_functions import (
    TAB_EXPORT_CALIBRATION,
    TAB_CREATE_TARGET,
    TAB_PHASE0,
    TAB_PHASE1,
    TAB_PHASE1_DIAG,
    TAB_PHASE2,
    TAB_PHASE2_DIAG,
    TAB_PHASE3,
    TAB_PHASE3_DIAG,
    TAB_PHASE4,
    TAB_PHASE4_DIAG,
    WorkspaceManager,
)


class _TooltipFilter(QObject):
    """Application-level event filter that suppresses tooltip events when info_cb is unchecked."""

    def __init__(self, info_cb: QCheckBox, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._info_cb = info_cb

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if not self._info_cb.isChecked() and event.type() == QEvent.Type.ToolTip:
            return True  # consume / block the tooltip event
        return False


class PyCamSetApp(QMainWindow):
    """Top-level application window.

    Responsibilities
    ----------------
    - Two global checkboxes wired into every child tab:
        1. **Enable Informational Windows** - enables Qt tool-tips.
        2. **Show Terminal Output** - shows/hides the per-tab terminal pane.
    - A ``QTabWidget`` with phase tabs and hidden diagnostics companions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("pyCamSet — Multi-Camera Calibration")
        self.resize(1140, 820)
        self.setMinimumSize(860, 640)

        # Shared state injected into child tabs
        self._info_cb = QCheckBox("Enable Informational Windows")
        self._info_cb.setChecked(True)
        self._info_cb.stateChanged.connect(self._on_info_toggle)

        # Install an application-level event filter that blocks hover tooltip
        # events while "Enable Informational Windows" is unchecked.
        self._tooltip_filter = _TooltipFilter(self._info_cb, self)
        QApplication.instance().installEventFilter(self._tooltip_filter)

        self._terminal_cb = QCheckBox("Show Terminal Output")
        self._terminal_cb.setChecked(True)

        # Lazy workspace manager: do not create any workspace dir at startup.
        self._workspace_mgr = WorkspaceManager(None)

        self._build_ui()
        self._on_info_toggle()  # apply initial tooltip state

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Deferred imports so module is importable without a display server
        from pyCamSet.gui.create_target import CreateTargetTab
        from pyCamSet.gui.export_calibration_tab import ExportCalibrationTab
        from pyCamSet.gui.phase_0_input import Phase0Tab
        from pyCamSet.gui.phase_1_detection import Phase1DiagnosticsTab, Phase1Tab
        from pyCamSet.gui.phase_2_intrinsics import Phase2DiagnosticsTab, Phase2Tab
        from pyCamSet.gui.phase_3_bundle_adjustment import Phase3DiagnosticsTab, Phase3Tab
        from pyCamSet.gui.phase_4_self_calibration import Phase4DiagnosticsTab, Phase4Tab

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(4)

        # ── Global controls ────────────────────────────────────────────
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(self._info_cb)
        ctrl_row.addSpacing(16)
        ctrl_row.addWidget(self._terminal_cb)
        ctrl_row.addStretch()
        root_layout.addLayout(ctrl_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        root_layout.addWidget(sep)

        # ── Tab widget ─────────────────────────────────────────────────
        ws = self._workspace_mgr

        self._notebook = QTabWidget()
        root_layout.addWidget(self._notebook)

        self.create_target_tab = CreateTargetTab(
            notebook=self._notebook,
            info_cb=self._info_cb,
            terminal_cb=self._terminal_cb,
            workspace_mgr=ws,
        )
        self._notebook.addTab(self.create_target_tab, TAB_CREATE_TARGET)

        self.phase0_tab = Phase0Tab(
            notebook=self._notebook,
            info_cb=self._info_cb,
            terminal_cb=self._terminal_cb,
            workspace_mgr=ws,
        )
        self._notebook.addTab(self.phase0_tab, TAB_PHASE0)

        self.phase1_tab = Phase1Tab(
            notebook=self._notebook,
            info_cb=self._info_cb,
            terminal_cb=self._terminal_cb,
            workspace_mgr=ws,
        )
        self._notebook.addTab(self.phase1_tab, TAB_PHASE1)

        self.phase1_diag_tab = Phase1DiagnosticsTab(
            notebook=self._notebook,
            info_cb=self._info_cb,
            workspace_mgr=ws,
        )
        diag1_idx = self._notebook.addTab(self.phase1_diag_tab, TAB_PHASE1_DIAG)
        self._notebook.tabBar().setTabVisible(diag1_idx, False)

        self.phase2_tab = Phase2Tab(
            notebook=self._notebook,
            info_cb=self._info_cb,
            terminal_cb=self._terminal_cb,
            workspace_mgr=ws,
        )
        self._notebook.addTab(self.phase2_tab, TAB_PHASE2)

        self.phase2_diag_tab = Phase2DiagnosticsTab(
            notebook=self._notebook,
            info_cb=self._info_cb,
            workspace_mgr=ws,
        )
        diag2_idx = self._notebook.addTab(self.phase2_diag_tab, TAB_PHASE2_DIAG)
        self._notebook.tabBar().setTabVisible(diag2_idx, False)

        self.phase3_tab = Phase3Tab(
            notebook=self._notebook,
            info_cb=self._info_cb,
            terminal_cb=self._terminal_cb,
            workspace_mgr=ws,
        )
        self._notebook.addTab(self.phase3_tab, TAB_PHASE3)

        self.phase3_diag_tab = Phase3DiagnosticsTab(
            notebook=self._notebook,
            info_cb=self._info_cb,
            workspace_mgr=ws,
        )
        diag3_idx = self._notebook.addTab(self.phase3_diag_tab, TAB_PHASE3_DIAG)
        self._notebook.tabBar().setTabVisible(diag3_idx, False)

        self.phase4_tab = Phase4Tab(
            notebook=self._notebook,
            info_cb=self._info_cb,
            terminal_cb=self._terminal_cb,
            workspace_mgr=ws,
        )
        self._notebook.addTab(self.phase4_tab, TAB_PHASE4)

        self.phase4_diag_tab = Phase4DiagnosticsTab(
            notebook=self._notebook,
            info_cb=self._info_cb,
            workspace_mgr=ws,
        )
        diag4_idx = self._notebook.addTab(self.phase4_diag_tab, TAB_PHASE4_DIAG)
        self._notebook.tabBar().setTabVisible(diag4_idx, False)

        self.export_calibration_tab = ExportCalibrationTab(
            notebook=self._notebook,
            info_cb=self._info_cb,
            terminal_cb=self._terminal_cb,
            workspace_mgr=ws,
        )
        self._notebook.addTab(self.export_calibration_tab, TAB_EXPORT_CALIBRATION)

        # Cross-tab wiring
        self.phase1_tab.set_diagnostics_tab(self.phase1_diag_tab)
        self.phase2_tab.set_diagnostics_tab(self.phase2_diag_tab)
        self.phase3_tab.set_diagnostics_tab(self.phase3_diag_tab)
        self.phase4_tab.set_diagnostics_tab(self.phase4_diag_tab)

        # ── Global sync: camera list ───────────────────────────────────
        self._syncing_cameras = False

        def _propagate_cameras(names: list[str], selected: list[str] | None = None) -> None:
            if self._syncing_cameras:
                return
            self._syncing_cameras = True
            try:
                self.phase0_tab.set_camera_names(names, selected=selected)
                self.phase1_tab.set_cameras(names, selected_cameras=selected)
            finally:
                self._syncing_cameras = False

        self.phase0_tab.set_cameras_callback(_propagate_cameras)
        self.phase1_tab.set_cameras_callback(_propagate_cameras)

        # ── Global sync: image folder ──────────────────────────────────
        self._syncing_floc = False

        def _propagate_image_folder(path: str) -> None:
            if self._syncing_floc:
                return
            self._syncing_floc = True
            try:
                self.phase0_tab.set_image_folder(path)
                self.phase1_tab.set_image_folder(path)
                self.phase2_tab.set_image_folder(path)
                self.phase3_tab.set_image_folder(path)
                self.phase4_tab.set_image_folder(path)
            finally:
                self._syncing_floc = False

        self.phase0_tab.set_phase1_path_callback(_propagate_image_folder)

        def _make_floc_changed(source_tab) -> None:
            def _handler(text: str) -> None:
                if text.strip():
                    _propagate_image_folder(text.strip())
            source_tab._floc_edit.textChanged.connect(_handler)

        for tab in (self.phase1_tab, self.phase2_tab, self.phase3_tab, self.phase4_tab):
            _make_floc_changed(tab)

        # ── Global sync: calibration target (Phase 1/2/3/4) ──────────
        self._syncing_target = False

        def _propagate_target(source_tab) -> None:
            if self._syncing_target:
                return
            self._syncing_target = True
            try:
                t_type = source_tab._target_combo.currentText()
                n_pts = source_tab._npts_spin.value()
                length = source_tab._length_edit.text()
                for tab in (self.phase1_tab, self.phase2_tab, self.phase3_tab, self.phase4_tab):
                    if tab is source_tab:
                        continue
                    if not hasattr(tab, "_target_combo"):
                        continue
                    tab._target_combo.blockSignals(True)
                    tab._npts_spin.blockSignals(True)
                    tab._length_edit.blockSignals(True)
                    idx = tab._target_combo.findText(t_type)
                    if idx >= 0:
                        tab._target_combo.setCurrentIndex(idx)
                    tab._npts_spin.setValue(n_pts)
                    tab._length_edit.setText(length)
                    tab._target_combo.blockSignals(False)
                    tab._npts_spin.blockSignals(False)
                    tab._length_edit.blockSignals(False)
            finally:
                self._syncing_target = False

        for src in (self.phase1_tab, self.phase2_tab, self.phase3_tab, self.phase4_tab):
            if hasattr(src, "_target_combo"):
                src._target_combo.currentIndexChanged.connect(lambda _v, s=src: _propagate_target(s))
                src._npts_spin.valueChanged.connect(lambda _v, s=src: _propagate_target(s))
                src._length_edit.textChanged.connect(lambda _v, s=src: _propagate_target(s))

        # Fallback tooltips for any controls missing explicit help text.
        self._apply_generic_option_tooltips(self.phase2_tab, "Phase 2")
        self._apply_generic_option_tooltips(self.phase3_tab, "Phase 3")

        # Enforce binary outlier selection for outlier-related combo controls.
        self._normalize_outlier_combos(self.phase2_tab)
        self._normalize_outlier_combos(self.phase3_tab)

        self._notebook.currentChanged.connect(self._on_tab_changed)
        self._apply_phase3_handoff()
        self._apply_phase4_handoff()

    def _apply_generic_option_tooltips(self, root: QWidget, phase_name: str) -> None:
        for w in root.findChildren(QWidget):
            if hasattr(w, "toolTip") and w.toolTip():
                continue

            if isinstance(w, QCheckBox):
                default = "Yes" if w.isChecked() else "No"
                w.setToolTip(
                    f"{phase_name} option.\n"
                    f"Concept: enable/disable this behavior.\n"
                    f"Default: {default}."
                )
            elif isinstance(w, QComboBox):
                items = [w.itemText(i) for i in range(w.count())]
                default = w.currentText() or "(none)"
                w.setToolTip(
                    f"{phase_name} option.\n"
                    f"Concept: choose one mode.\n"
                    f"Choices: {items}\n"
                    f"Default/current: {default}."
                )
            elif isinstance(w, QSpinBox):
                w.setToolTip(
                    f"{phase_name} option.\n"
                    f"Concept: integer tuning parameter.\n"
                    f"Default/current: {w.value()}."
                )
            elif isinstance(w, QLineEdit):
                default = w.text().strip() or "blank"
                w.setToolTip(
                    f"{phase_name} option.\n"
                    f"Concept: numeric/text tuning input.\n"
                    f"Default/current: {default}."
                )

    def _normalize_outlier_combos(self, root: QWidget) -> None:
        yes_tokens = {"yes", "true", "1", "on", "enabled"}
        no_tokens = {"no", "false", "0", "off", "none", "disabled"}
        outlier_tokens = {
            "outlier", "robust", "loss", "huber", "cauchy", "soft_l1", "arctan", "tukey", "ransac"
        }

        for combo in root.findChildren(QComboBox):
            text_blob = " ".join(
                [
                    combo.objectName() or "",
                    combo.accessibleName() or "",
                    combo.toolTip() or "",
                    *[combo.itemText(i) for i in range(combo.count())],
                ]
            ).lower()

            if not any(tok in text_blob for tok in outlier_tokens):
                continue

            current = (combo.currentText() or "").strip().lower()
            is_yes = current in yes_tokens or (current not in no_tokens and combo.currentIndex() > 0)

            combo.blockSignals(True)
            combo.clear()
            combo.addItems(["No", "Yes"])
            combo.setCurrentIndex(1 if is_yes else 0)
            combo.setToolTip(
                "Concept: outlier handling on/off.\n"
                "Default: No.\n"
                "No = disable outlier rejection; Yes = enable it."
            )
            combo.blockSignals(False)

    def _read_handoff(self) -> dict | None:
        ws = self._workspace_mgr.workspace_path
        if ws is None:
            return None
        p = Path(ws) / "handoff.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def _apply_phase3_handoff(self) -> None:
        payload = self._read_handoff()
        if not payload or payload.get("phase") != "phase2":
            return

        run = None
        runs = payload.get("runs") or []
        if runs:
            run = runs[0]

        f_loc = payload.get("image_folder") or ((run or {}).get("params") or {}).get("f_loc")
        run_id = payload.get("phase2_run_id") or (run or {}).get("run_id")
        camset_path = payload.get("initial_camset") or ((run or {}).get("artifacts") or {}).get("initial_camset")
        tab = self.phase3_tab

        if f_loc:
            for meth in ("set_image_folder", "set_floc", "set_image_path"):
                if hasattr(tab, meth):
                    try:
                        getattr(tab, meth)(str(f_loc))
                        break
                    except Exception:
                        pass
            for attr in ("_floc_edit", "_image_folder_edit", "_f_loc_edit"):
                if hasattr(tab, attr):
                    try:
                        getattr(tab, attr).setText(str(f_loc))
                        break
                    except Exception:
                        pass

        if run_id:
            for meth in ("set_phase2_run_id", "set_selected_phase2_run_id", "set_phase2_run"):
                if hasattr(tab, meth):
                    try:
                        getattr(tab, meth)(str(run_id))
                        break
                    except Exception:
                        pass
            for combo in tab.findChildren(QComboBox):
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
            for meth in ("set_phase2_camset_path", "set_camset_path"):
                if hasattr(tab, meth):
                    try:
                        getattr(tab, meth)(str(camset_path))
                        break
                    except Exception:
                        pass
            for attr in ("_camset_edit", "_phase2_camset_edit"):
                if hasattr(tab, attr):
                    try:
                        getattr(tab, attr).setText(str(camset_path))
                        break
                    except Exception:
                        pass

    def _apply_phase4_handoff(self) -> None:
        payload = self._read_handoff()
        if not payload or payload.get("phase") != "phase3":
            return

        run = None
        runs = payload.get("runs") or []
        if runs:
            run = runs[0]

        f_loc = payload.get("image_folder") or ((run or {}).get("params") or {}).get("f_loc")
        run_id = payload.get("phase3_run_id") or (run or {}).get("run_id")
        camset_path = payload.get("optimised_camset") or ((run or {}).get("artifacts") or {}).get("optimised_camset")
        tab = self.phase4_tab

        if f_loc:
            for meth in ("set_image_folder", "set_floc", "set_image_path"):
                if hasattr(tab, meth):
                    try:
                        getattr(tab, meth)(str(f_loc))
                        break
                    except Exception:
                        pass

        if run_id:
            for meth in ("set_phase3_run_id", "set_selected_phase3_run_id"):
                if hasattr(tab, meth):
                    try:
                        getattr(tab, meth)(str(run_id))
                        break
                    except Exception:
                        pass
            for combo in tab.findChildren(QComboBox):
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
            for meth in ("set_phase3_camset_path", "set_camset_path"):
                if hasattr(tab, meth):
                    try:
                        getattr(tab, meth)(str(camset_path))
                        break
                    except Exception:
                        pass

    def _on_tab_changed(self, index: int) -> None:
        name = self._notebook.tabText(index)
        if name == TAB_PHASE3:
            self._apply_phase3_handoff()
            self._normalize_outlier_combos(self.phase3_tab)
        elif name == TAB_PHASE4:
            self._apply_phase4_handoff()
            self._normalize_outlier_combos(self.phase4_tab)
        elif name == TAB_PHASE2:
            self._normalize_outlier_combos(self.phase2_tab)
        elif name == TAB_EXPORT_CALIBRATION:
            self.export_calibration_tab.refresh()

    def _on_info_toggle(self) -> None:
        """Enable or disable all Qt tool-tips application-wide."""
        QApplication.instance().setProperty(
            "tooltipsEnabled", self._info_cb.isChecked()
        )

    def switch_to_tab(self, name: str) -> None:
        """Switch to the named tab by its display text."""
        for i in range(self._notebook.count()):
            if self._notebook.tabText(i) == name:
                self._notebook.setCurrentIndex(i)
                return


def main_window() -> None:
    """Launch the pyCamSet GUI."""
    import sys

    app = QApplication.instance() or QApplication(sys.argv)
    window = PyCamSetApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main_window()
