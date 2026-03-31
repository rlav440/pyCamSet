"""Create Target tab.

Thin GUI wrapper around printable target generators.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

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
    QStackedWidget,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pyCamSet.calibration_targets.create_Ccube import build_ccube, generate_ccube_target
from pyCamSet.calibration_targets.create_charuco import build_charuco, generate_charuco_target
from pyCamSet.gui.shared_functions import TerminalWidget, WorkspaceManager, make_blue_button, make_section_label, make_separator

_EXPORT_CHOICES = {
    "PDF (Raster)": "pdf_raster",
    "PDF (Vector)": "pdf_vector",
    "SVG": "svg",
}
_TARGET_CCUBE = "Ccube"
_TARGET_CHARUCO = "ChArUco"
_ARUCO_DICT_CHOICES = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_5X5_1000",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_6X6_1000",
    "DICT_7X7_50",
    "DICT_7X7_100",
    "DICT_7X7_250",
    "DICT_7X7_1000",
]


class CreateTargetTab(QWidget):
    """Generate printable Ccube or ChArUco targets and optionally visualise them externally."""

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
        self._build_ui(terminal_cb)

    def _build_ui(self, terminal_cb: QCheckBox) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        root.addWidget(make_section_label("Target Parameters"))

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        root.addLayout(form)

        self._target_combo = QComboBox()
        self._target_combo.addItems([_TARGET_CCUBE, _TARGET_CHARUCO])
        self._target_combo.currentIndexChanged.connect(self._on_target_changed)
        form.addRow("Target type:", self._target_combo)

        self._param_stack = QStackedWidget()
        form.addRow("Parameters:", self._param_stack)

        ccube_params = QWidget()
        ccube_form = QFormLayout(ccube_params)
        ccube_form.setContentsMargins(0, 0, 0, 0)

        self._npts_spin = QSpinBox()
        self._npts_spin.setRange(2, 40)
        self._npts_spin.setValue(5)
        self._npts_spin.setFixedWidth(110)
        self._npts_spin.valueChanged.connect(self._sync_default_name)
        ccube_form.addRow("n_points:", self._npts_spin)

        self._length_edit = QLineEdit("10")
        self._length_edit.setFixedWidth(110)
        self._length_edit.textChanged.connect(self._sync_default_name)
        ccube_form.addRow("length (mm):", self._length_edit)

        self._param_stack.addWidget(ccube_params)

        charuco_params = QWidget()
        charuco_form = QFormLayout(charuco_params)
        charuco_form.setContentsMargins(0, 0, 0, 0)

        self._charuco_x_spin = QSpinBox()
        self._charuco_x_spin.setRange(2, 100)
        self._charuco_x_spin.setValue(5)
        self._charuco_x_spin.setFixedWidth(110)
        self._charuco_x_spin.valueChanged.connect(self._sync_default_name)
        charuco_form.addRow("num_squares_x:", self._charuco_x_spin)

        self._charuco_y_spin = QSpinBox()
        self._charuco_y_spin.setRange(2, 100)
        self._charuco_y_spin.setValue(7)
        self._charuco_y_spin.setFixedWidth(110)
        self._charuco_y_spin.valueChanged.connect(self._sync_default_name)
        charuco_form.addRow("num_squares_y:", self._charuco_y_spin)

        self._charuco_square_edit = QLineEdit("10")
        self._charuco_square_edit.setFixedWidth(110)
        self._charuco_square_edit.textChanged.connect(self._sync_default_name)
        charuco_form.addRow("square_size (mm):", self._charuco_square_edit)

        self._charuco_marker_fraction_edit = QLineEdit("0.8")
        self._charuco_marker_fraction_edit.setFixedWidth(110)
        self._charuco_marker_fraction_edit.textChanged.connect(self._sync_default_name)
        charuco_form.addRow("marker_fraction:", self._charuco_marker_fraction_edit)

        self._charuco_dict_combo = QComboBox()
        self._charuco_dict_combo.addItems(_ARUCO_DICT_CHOICES)
        self._charuco_dict_combo.setCurrentText("DICT_4X4_1000")
        self._charuco_dict_combo.currentIndexChanged.connect(self._sync_default_name)
        charuco_form.addRow("dictionary:", self._charuco_dict_combo)

        self._param_stack.addWidget(charuco_params)

        self._format_combo = QComboBox()
        self._format_combo.addItems(list(_EXPORT_CHOICES.keys()))
        self._format_combo.setCurrentText("SVG")
        self._format_combo.currentIndexChanged.connect(self._sync_default_name)
        form.addRow("Export format:", self._format_combo)

        out_row = QHBoxLayout()
        self._out_dir_edit = QLineEdit(str(Path.cwd()))
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse_output_dir)
        out_row.addWidget(self._out_dir_edit)
        out_row.addWidget(browse_btn)
        form.addRow("Output directory:", out_row)

        self._name_edit = QLineEdit()
        form.addRow("Output file name:", self._name_edit)

        self._on_target_changed()
        self._sync_default_name()

        root.addWidget(make_separator())
        btn_row = QHBoxLayout()
        btn_row.addWidget(make_blue_button("Save Target", self._save_target))
        btn_row.addWidget(make_blue_button("Visualise Target", self._visualise_target))
        btn_row.addStretch()
        root.addLayout(btn_row)

        self._status = QLabel("")
        self._status.setStyleSheet("color: #2e7d32;")
        root.addWidget(self._status)

        self._terminal = TerminalWidget(terminal_cb, parent=self)
        root.addWidget(self._terminal)

    def _browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self._out_dir_edit.setText(path)

    def _on_target_changed(self) -> None:
        if self._target_combo.currentText() == _TARGET_CHARUCO:
            self._param_stack.setCurrentIndex(1)
        else:
            self._param_stack.setCurrentIndex(0)
        self._sync_default_name()

    def _collect(self) -> dict | None:
        target_type = self._target_combo.currentText()
        payload: dict = {"target_type": target_type}

        try:
            if target_type == _TARGET_CCUBE:
                payload["n_points"] = int(self._npts_spin.value())
                payload["length"] = float(self._length_edit.text().strip())
            else:
                payload["num_squares_x"] = int(self._charuco_x_spin.value())
                payload["num_squares_y"] = int(self._charuco_y_spin.value())
                payload["square_size"] = float(self._charuco_square_edit.text().strip())
                payload["marker_fraction"] = float(self._charuco_marker_fraction_edit.text().strip())
                payload["aruco_dict"] = self._charuco_dict_combo.currentText()
        except ValueError as exc:
            QMessageBox.critical(self, "Validation Error", f"Invalid numeric value: {exc}")
            return None

        if target_type == _TARGET_CHARUCO:
            marker_fraction = float(payload["marker_fraction"])
            if marker_fraction <= 0.0 or marker_fraction >= 1.0:
                QMessageBox.critical(self, "Validation Error", "marker_fraction must be between 0 and 1.")
                return None

        out_dir_text = self._out_dir_edit.text().strip()
        if not out_dir_text:
            QMessageBox.critical(self, "Validation Error", "Output directory is required.")
            return None

        out_dir = Path(out_dir_text)
        file_name = self._name_edit.text().strip()
        if not file_name:
            QMessageBox.critical(self, "Validation Error", "Output filename is required.")
            return None

        export_label = self._format_combo.currentText()
        export_kind = _EXPORT_CHOICES.get(export_label)
        if export_kind is None:
            QMessageBox.critical(self, "Validation Error", "Unknown export format selected.")
            return None

        payload["out_dir"] = out_dir
        payload["file_name"] = file_name
        payload["export_kind"] = export_kind
        return payload

    def _sync_default_name(self) -> None:
        export_kind = _EXPORT_CHOICES.get(self._format_combo.currentText(), "svg")
        suffix = ".svg" if export_kind == "svg" else ".pdf"

        if self._target_combo.currentText() == _TARGET_CCUBE:
            n_points = int(self._npts_spin.value())
            try:
                length = float(self._length_edit.text().strip())
            except ValueError:
                return
            self._name_edit.setText(f"ccube_{n_points}points_{length:g}mm{suffix}")
            return

        num_squares_x = int(self._charuco_x_spin.value())
        num_squares_y = int(self._charuco_y_spin.value())
        try:
            square_size = float(self._charuco_square_edit.text().strip())
        except ValueError:
            return
        self._name_edit.setText(f"charuco_{num_squares_x}x{num_squares_y}_{square_size:g}mm{suffix}")

    def _save_target(self) -> None:
        collected = self._collect()
        if collected is None:
            return

        try:
            if collected["target_type"] == _TARGET_CCUBE:
                _, out_path = generate_ccube_target(
                    n_points=int(collected["n_points"]),
                    length=float(collected["length"]),
                    output_dir=collected["out_dir"],
                    file_name=collected["file_name"],
                    export_kind=collected["export_kind"],
                )
            else:
                _, out_path = generate_charuco_target(
                    num_squares_x=int(collected["num_squares_x"]),
                    num_squares_y=int(collected["num_squares_y"]),
                    square_size=float(collected["square_size"]),
                    marker_fraction=float(collected["marker_fraction"]),
                    aruco_dict=str(collected["aruco_dict"]),
                    output_dir=collected["out_dir"],
                    file_name=collected["file_name"],
                    export_kind=collected["export_kind"],
                )
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))
            self._terminal.append_line(f"ERROR: {exc}")
            return

        self._status.setText(f"Saved target: {out_path}")
        self._terminal.append_line(f"Saved target: {out_path}")

    def _visualise_target(self) -> None:
        collected = self._collect()
        if collected is None:
            return

        try:
            if collected["target_type"] == _TARGET_CCUBE:
                cube = build_ccube(
                    n_points=int(collected["n_points"]),
                    length=float(collected["length"]),
                )
                cube.plot()  # External pyvista/matplotlib window
            else:
                board = build_charuco(
                    num_squares_x=int(collected["num_squares_x"]),
                    num_squares_y=int(collected["num_squares_y"]),
                    square_size=float(collected["square_size"]),
                    marker_fraction=float(collected["marker_fraction"]),
                    aruco_dict=str(collected["aruco_dict"]),
                )
                board.plot()
            self._terminal.append_line("Opened external target visualisation window.")
        except Exception as exc:
            QMessageBox.critical(self, "Visualise Failed", str(exc))
            self._terminal.append_line(f"ERROR: {exc}")


