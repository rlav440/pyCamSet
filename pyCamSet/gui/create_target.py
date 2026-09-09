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
from pyCamSet.calibration_targets.create_puzzleboard import build_puzzleboard, generate_puzzleboard_target
from pyCamSet.calibration_targets.create_puzzleboard_cube import build_puzzleboard_cube, generate_puzzleboard_cube_target
from pyCamSet.calibration_targets.target_puzzleboard_cube import MAX_FACE_SQUARES
from pyCamSet.gui.shared_functions import (
    ARUCO1_DICT_NAMES,
    MARKER_BACKEND_LABELS,
    TerminalWidget,
    WorkspaceManager,
    make_blue_button,
    make_section_label,
    make_separator,
    marker_backend_availability_text,
    marker_backend_available,
    repopulate_dict_combo,
)

_EXPORT_CHOICES = {
    "PDF (Raster)": "pdf_raster",
    "PDF (Vector)": "pdf_vector",
    "SVG": "svg",
}
_TARGET_CCUBE = "Ccube"
_TARGET_CHARUCO = "ChArUco"
_TARGET_PUZZLEBOARD = "PuzzleBoard"
_TARGET_PUZZLEBOARD_CUBE = "PuzzleBoard Cube"
_ARUCO_DICT_CHOICES = ARUCO1_DICT_NAMES


class CreateTargetTab(QWidget):
    """Generate printable Ccube, ChArUco, or PuzzleBoard targets and visualise them externally."""

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
        self._repopulating = False
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
        self._target_combo.addItems([_TARGET_CCUBE, _TARGET_CHARUCO, _TARGET_PUZZLEBOARD, _TARGET_PUZZLEBOARD_CUBE])
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

        self._ccube_dict_combo = QComboBox()
        self._ccube_dict_combo.addItems(_ARUCO_DICT_CHOICES)
        self._ccube_dict_combo.setCurrentText("DICT_4X4_1000")
        ccube_form.addRow("dictionary:", self._ccube_dict_combo)

        self._ccube_backend_combo = QComboBox()
        for label, value in MARKER_BACKEND_LABELS.items():
            self._ccube_backend_combo.addItem(label, value)
        self._ccube_backend_combo.setCurrentText("ArUco 1 (OpenCV)")
        self._ccube_backend_combo.currentIndexChanged.connect(self._on_ccube_backend_changed)
        ccube_form.addRow("Marker backend:", self._ccube_backend_combo)
        # FIX 8(g): small availability label near the combo (plan v4 D12),
        # refreshed on combo change so availability is honoured immediately.
        self._ccube_backend_status = QLabel(marker_backend_availability_text("aruco1"))
        self._ccube_backend_status.setStyleSheet("color: #2a7a2a;")
        ccube_form.addRow("", self._ccube_backend_status)

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

        self._charuco_backend_combo = QComboBox()
        for label, value in MARKER_BACKEND_LABELS.items():
            self._charuco_backend_combo.addItem(label, value)
        self._charuco_backend_combo.setCurrentText("ArUco 1 (OpenCV)")
        self._charuco_backend_combo.currentIndexChanged.connect(self._on_charuco_backend_changed)
        charuco_form.addRow("Marker backend:", self._charuco_backend_combo)
        # FIX 8(g): small availability label near the combo (plan v4 D12).
        self._charuco_backend_status = QLabel(marker_backend_availability_text("aruco1"))
        self._charuco_backend_status.setStyleSheet("color: #2a7a2a;")
        charuco_form.addRow("", self._charuco_backend_status)

        self._param_stack.addWidget(charuco_params)

        puzzleboard_params = QWidget()
        puzzleboard_form = QFormLayout(puzzleboard_params)
        puzzleboard_form.setContentsMargins(0, 0, 0, 0)

        self._puzzleboard_x_spin = QSpinBox()
        self._puzzleboard_x_spin.setRange(2, 501)
        self._puzzleboard_x_spin.setValue(105)
        self._puzzleboard_x_spin.setFixedWidth(110)
        self._puzzleboard_x_spin.valueChanged.connect(self._sync_default_name)
        puzzleboard_form.addRow("num_squares_x:", self._puzzleboard_x_spin)

        self._puzzleboard_y_spin = QSpinBox()
        self._puzzleboard_y_spin.setRange(2, 501)
        self._puzzleboard_y_spin.setValue(148)
        self._puzzleboard_y_spin.setFixedWidth(110)
        self._puzzleboard_y_spin.valueChanged.connect(self._sync_default_name)
        puzzleboard_form.addRow("num_squares_y:", self._puzzleboard_y_spin)

        self._puzzleboard_square_edit = QLineEdit("2")
        self._puzzleboard_square_edit.setFixedWidth(110)
        self._puzzleboard_square_edit.textChanged.connect(self._sync_default_name)
        puzzleboard_form.addRow("square_size (mm):", self._puzzleboard_square_edit)

        self._puzzleboard_start_x_spin = QSpinBox()
        self._puzzleboard_start_x_spin.setRange(0, 500)
        self._puzzleboard_start_x_spin.setValue(0)
        self._puzzleboard_start_x_spin.setFixedWidth(110)
        puzzleboard_form.addRow("start_x:", self._puzzleboard_start_x_spin)

        self._puzzleboard_start_y_spin = QSpinBox()
        self._puzzleboard_start_y_spin.setRange(0, 500)
        self._puzzleboard_start_y_spin.setValue(0)
        self._puzzleboard_start_y_spin.setFixedWidth(110)
        puzzleboard_form.addRow("start_y:", self._puzzleboard_start_y_spin)

        self._puzzleboard_page_width_edit = QLineEdit("210")
        self._puzzleboard_page_width_edit.setFixedWidth(110)
        puzzleboard_form.addRow("paper_width (mm):", self._puzzleboard_page_width_edit)

        self._puzzleboard_page_height_edit = QLineEdit("297")
        self._puzzleboard_page_height_edit.setFixedWidth(110)
        puzzleboard_form.addRow("paper_height (mm):", self._puzzleboard_page_height_edit)

        self._puzzleboard_min_width_spin = QSpinBox()
        self._puzzleboard_min_width_spin.setRange(1, 501)
        self._puzzleboard_min_width_spin.setValue(4)
        self._puzzleboard_min_width_spin.setFixedWidth(110)
        puzzleboard_form.addRow("detector min_width:", self._puzzleboard_min_width_spin)

        self._param_stack.addWidget(puzzleboard_params)

        puzzleboard_cube_params = QWidget()
        puzzleboard_cube_form = QFormLayout(puzzleboard_cube_params)
        puzzleboard_cube_form.setContentsMargins(0, 0, 0, 0)

        self._puzzleboard_cube_size_spin = QSpinBox()
        self._puzzleboard_cube_size_spin.setRange(2, MAX_FACE_SQUARES)
        self._puzzleboard_cube_size_spin.setValue(20)
        self._puzzleboard_cube_size_spin.setFixedWidth(110)
        self._puzzleboard_cube_size_spin.valueChanged.connect(self._sync_default_name)
        puzzleboard_cube_form.addRow(f"n_points / pieces per face (max {MAX_FACE_SQUARES}):", self._puzzleboard_cube_size_spin)

        self._puzzleboard_cube_square_edit = QLineEdit("200")
        self._puzzleboard_cube_square_edit.setFixedWidth(110)
        self._puzzleboard_cube_square_edit.textChanged.connect(self._sync_default_name)
        puzzleboard_cube_form.addRow("length (mm):", self._puzzleboard_cube_square_edit)

        self._puzzleboard_cube_min_width_spin = QSpinBox()
        self._puzzleboard_cube_min_width_spin.setRange(1, 501)
        self._puzzleboard_cube_min_width_spin.setValue(4)
        self._puzzleboard_cube_min_width_spin.setFixedWidth(110)
        puzzleboard_cube_form.addRow("detector min_width:", self._puzzleboard_cube_min_width_spin)

        self._puzzleboard_cube_border_edit = QLineEdit("10")
        self._puzzleboard_cube_border_edit.setFixedWidth(110)
        puzzleboard_cube_form.addRow("net border (mm):", self._puzzleboard_cube_border_edit)

        self._puzzleboard_cube_outline_check = QCheckBox()
        self._puzzleboard_cube_outline_check.setChecked(True)
        puzzleboard_cube_form.addRow("draw cut outlines:", self._puzzleboard_cube_outline_check)

        self._puzzleboard_cube_ids_check = QCheckBox()
        self._puzzleboard_cube_ids_check.setChecked(True)
        puzzleboard_cube_form.addRow("draw face labels:", self._puzzleboard_cube_ids_check)

        self._param_stack.addWidget(puzzleboard_cube_params)

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
        elif self._target_combo.currentText() == _TARGET_PUZZLEBOARD:
            self._param_stack.setCurrentIndex(2)
        elif self._target_combo.currentText() == _TARGET_PUZZLEBOARD_CUBE:
            self._param_stack.setCurrentIndex(3)
        else:
            self._param_stack.setCurrentIndex(0)
        self._sync_default_name()

    def _backend_value(self, combo: QComboBox) -> str:
        """Return the marker-backend value for a backend combo (default aruco1)."""
        return str(combo.currentData() or "aruco1")

    def _on_ccube_backend_changed(self) -> None:
        self._repopulating = True
        try:
            repopulate_dict_combo(self._ccube_dict_combo, self._backend_value(self._ccube_backend_combo))
        finally:
            self._repopulating = False
        # FIX 8(g): honour availability on combo change, not only at save/start.
        self._ccube_backend_status.setText(
            marker_backend_availability_text(self._backend_value(self._ccube_backend_combo))
        )
        self._ccube_backend_status.setStyleSheet(
            "color: #2a7a2a;" if marker_backend_available(
                self._backend_value(self._ccube_backend_combo)
            ) else "color: #8a4a00;"
        )

    def _on_charuco_backend_changed(self) -> None:
        self._repopulating = True
        try:
            repopulate_dict_combo(self._charuco_dict_combo, self._backend_value(self._charuco_backend_combo))
        finally:
            self._repopulating = False
        # FIX 8(g): honour availability on combo change, not only at save/start.
        self._charuco_backend_status.setText(
            marker_backend_availability_text(self._backend_value(self._charuco_backend_combo))
        )
        self._charuco_backend_status.setStyleSheet(
            "color: #2a7a2a;" if marker_backend_available(
                self._backend_value(self._charuco_backend_combo)
            ) else "color: #8a4a00;"
        )

    def _collect(self) -> dict | None:
        target_type = self._target_combo.currentText()
        payload: dict = {"target_type": target_type}

        try:
            if target_type == _TARGET_CCUBE:
                payload["n_points"] = int(self._npts_spin.value())
                payload["length"] = float(self._length_edit.text().strip())
                payload["aruco_dict"] = self._ccube_dict_combo.currentText()
                payload["marker_backend"] = self._backend_value(self._ccube_backend_combo)
            elif target_type == _TARGET_CHARUCO:
                payload["num_squares_x"] = int(self._charuco_x_spin.value())
                payload["num_squares_y"] = int(self._charuco_y_spin.value())
                payload["square_size"] = float(self._charuco_square_edit.text().strip())
                payload["marker_fraction"] = float(self._charuco_marker_fraction_edit.text().strip())
                payload["aruco_dict"] = self._charuco_dict_combo.currentText()
                payload["marker_backend"] = self._backend_value(self._charuco_backend_combo)
            elif target_type == _TARGET_PUZZLEBOARD:
                payload["num_squares_x"] = int(self._puzzleboard_x_spin.value())
                payload["num_squares_y"] = int(self._puzzleboard_y_spin.value())
                payload["square_size"] = float(self._puzzleboard_square_edit.text().strip())
                payload["start_x"] = int(self._puzzleboard_start_x_spin.value())
                payload["start_y"] = int(self._puzzleboard_start_y_spin.value())
                payload["paper_width"] = float(self._puzzleboard_page_width_edit.text().strip())
                payload["paper_height"] = float(self._puzzleboard_page_height_edit.text().strip())
                payload["min_width"] = int(self._puzzleboard_min_width_spin.value())
            else:
                payload["n_points"] = int(self._puzzleboard_cube_size_spin.value())
                payload["length"] = float(self._puzzleboard_cube_square_edit.text().strip())
                payload["min_width"] = int(self._puzzleboard_cube_min_width_spin.value())
                payload["border_width"] = float(self._puzzleboard_cube_border_edit.text().strip())
                payload["draw_cut_outline"] = self._puzzleboard_cube_outline_check.isChecked()
                payload["draw_face_ids"] = self._puzzleboard_cube_ids_check.isChecked()
        except ValueError as exc:
            QMessageBox.critical(self, "Validation Error", f"Invalid numeric value: {exc}")
            return None

        if target_type in (_TARGET_CCUBE, _TARGET_CHARUCO):
            if not marker_backend_available(payload.get("marker_backend", "aruco1")):
                QMessageBox.warning(
                    self,
                    "Marker backend unavailable",
                    "ArUco 2 (aruco2) is selected but the 'aruco2' package is not "
                    "installed. Install it with `pip install aruco2` or switch the "
                    "marker backend to ArUco 1 (OpenCV).",
                )
                return None

        if target_type == _TARGET_CHARUCO:
            marker_fraction = float(payload.get("marker_fraction", 0.8))
            if marker_fraction <= 0.0 or marker_fraction >= 1.0:
                QMessageBox.critical(self, "Validation Error", "marker_fraction must be between 0 and 1.")
                return None
        elif target_type == _TARGET_PUZZLEBOARD:
            if payload["square_size"] <= 0.0:
                QMessageBox.critical(self, "Validation Error", "square_size must be greater than zero.")
                return None
            if payload["paper_width"] <= 0.0 or payload["paper_height"] <= 0.0:
                QMessageBox.critical(self, "Validation Error", "paper dimensions must be greater than zero.")
                return None
            if payload["start_x"] + payload["num_squares_x"] > 501:
                QMessageBox.critical(self, "Validation Error", "start_x + num_squares_x must not exceed 501.")
                return None
            if payload["start_y"] + payload["num_squares_y"] > 501:
                QMessageBox.critical(self, "Validation Error", "start_y + num_squares_y must not exceed 501.")
                return None
        elif target_type == _TARGET_PUZZLEBOARD_CUBE:
            if payload["length"] <= 0.0:
                QMessageBox.critical(self, "Validation Error", "length must be greater than zero.")
                return None
            if payload["border_width"] < 0.0:
                QMessageBox.critical(self, "Validation Error", "net border must not be negative.")
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
        if self._repopulating:
            return
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

        if self._target_combo.currentText() == _TARGET_PUZZLEBOARD:
            num_squares_x = int(self._puzzleboard_x_spin.value())
            num_squares_y = int(self._puzzleboard_y_spin.value())
            try:
                square_size = float(self._puzzleboard_square_edit.text().strip())
            except ValueError:
                return
            self._name_edit.setText(f"puzzleboard_{num_squares_x}x{num_squares_y}_{square_size:g}mm{suffix}")
            return

        if self._target_combo.currentText() == _TARGET_PUZZLEBOARD_CUBE:
            n_points = int(self._puzzleboard_cube_size_spin.value())
            try:
                length = float(self._puzzleboard_cube_square_edit.text().strip())
            except ValueError:
                return
            self._name_edit.setText(f"puzzleboard_cube_{n_points}points_{length:g}mm{suffix}")
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
                    aruco_dict=str(collected["aruco_dict"]),
                    marker_backend=collected.get("marker_backend", "aruco1"),
                    output_dir=collected["out_dir"],
                    file_name=collected["file_name"],
                    export_kind=collected["export_kind"],
                )
            elif collected["target_type"] == _TARGET_CHARUCO:
                _, out_path = generate_charuco_target(
                    num_squares_x=int(collected["num_squares_x"]),
                    num_squares_y=int(collected["num_squares_y"]),
                    square_size=float(collected["square_size"]),
                    marker_fraction=float(collected.get("marker_fraction", 0.8)),
                    aruco_dict=str(collected["aruco_dict"]),
                    marker_backend=collected.get("marker_backend", "aruco1"),
                    output_dir=collected["out_dir"],
                    file_name=collected["file_name"],
                    export_kind=collected["export_kind"],
                )
            elif collected["target_type"] == _TARGET_PUZZLEBOARD:
                _, out_path = generate_puzzleboard_target(
                    num_squares_x=int(collected["num_squares_x"]),
                    num_squares_y=int(collected["num_squares_y"]),
                    square_size=float(collected["square_size"]),
                    start_x=int(collected["start_x"]),
                    start_y=int(collected["start_y"]),
                    paper_width=float(collected["paper_width"]),
                    paper_height=float(collected["paper_height"]),
                    min_width=int(collected["min_width"]),
                    output_dir=collected["out_dir"],
                    file_name=collected["file_name"],
                    export_kind=collected["export_kind"],
                )
            else:
                _, out_path = generate_puzzleboard_cube_target(
                    n_points=int(collected["n_points"]),
                    length=float(collected["length"]),
                    min_width=int(collected["min_width"]),
                    border_width=float(collected["border_width"]),
                    draw_cut_outline=bool(collected["draw_cut_outline"]),
                    draw_face_ids=bool(collected["draw_face_ids"]),
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
                    aruco_dict=str(collected["aruco_dict"]),
                    marker_backend=collected.get("marker_backend", "aruco1"),
                )
                cube.plot()  # External pyvista/matplotlib window
            elif collected["target_type"] == _TARGET_CHARUCO:
                board = build_charuco(
                    num_squares_x=int(collected["num_squares_x"]),
                    num_squares_y=int(collected["num_squares_y"]),
                    square_size=float(collected["square_size"]),
                    marker_fraction=float(collected.get("marker_fraction", 0.8)),
                    aruco_dict=str(collected["aruco_dict"]),
                    marker_backend=collected.get("marker_backend", "aruco1"),
                )
                board.plot()
            elif collected["target_type"] == _TARGET_PUZZLEBOARD:
                board = build_puzzleboard(
                    num_squares_x=int(collected["num_squares_x"]),
                    num_squares_y=int(collected["num_squares_y"]),
                    square_size=float(collected["square_size"]),
                    start_x=int(collected["start_x"]),
                    start_y=int(collected["start_y"]),
                    paper_width=float(collected["paper_width"]),
                    paper_height=float(collected["paper_height"]),
                    min_width=int(collected["min_width"]),
                )
                board.plot()
            else:
                cube = build_puzzleboard_cube(
                    n_points=int(collected["n_points"]),
                    length=float(collected["length"]),
                    min_width=int(collected["min_width"]),
                )
                cube.plot()
            self._terminal.append_line("Opened external target visualisation window.")
        except Exception as exc:
            QMessageBox.critical(self, "Visualise Failed", str(exc))
            self._terminal.append_line(f"ERROR: {exc}")
