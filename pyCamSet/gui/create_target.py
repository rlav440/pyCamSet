"""Making a printable calibration target.

A thin wrapper around the targets' own export, and the one thing in the
window that is not part of a calibration: a target is drawn once, printed,
and then lived with for months of runs.  It sat as the first of the phase
tabs, ahead of Phase 0, so every session opened on the one step almost no
session takes.

It is a dialog for that reason, opened from the button in the corner.

Both of its forms are built from what the selected target declares: the
arguments that decide what it is, and the options that decide how it is
drawn.  It offered four targets in a stack of four hand-written pages
before, and validated each of them against rules the targets now state.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pyCamSet.calibration_targets.core.target_registry import build_target, target_class
from pyCamSet.gui.viewer_process import spawn_viewer
from pyCamSet.gui.shared_functions import (
    DETECTOR_NONE,
    TargetSettingsForm,
    TerminalWidget,
    build_parameter_widget,
    make_blue_button,
    make_section_label,
    make_separator,
    read_parameter_widget,
)
from pyCamSet.workflow.params import ParamError

#: The formats offered, by the name shown for each.
_EXPORT_CHOICES = {
    "SVG": "svg",
    "PDF (Vector)": "pdf_vector",
    "PDF (Raster)": "pdf_raster",
}

_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
    "COM¹", "COM²", "COM³", "LPT¹", "LPT²", "LPT³",
}


class CreateTargetDialog(QDialog):
    """Generate printable Ccube, ChArUco, or PuzzleBoard targets and visualise them externally.

    Modeless: the target opens in a window of its own -- see
    :mod:`pyCamSet.gui.viewer_process` -- and comparing it against the form
    that drew it means having both.

    :param terminal_cb: the window's "Show Terminal Output" checkbox
    :param parent: the main window, which keeps the one instance
    """

    def __init__(
        self,
        terminal_cb: QCheckBox,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Create Target")
        screen = self.screen() or QApplication.primaryScreen()
        available = screen.availableGeometry().size() if screen else None
        if available is None:
            self.setMinimumSize(480, 520)
            self.resize(620, 720)
        else:
            # Keep this secondary form inside the usable display area while
            # retaining a sensible minimum on ordinary-sized screens.
            minimum_width = min(480, available.width())
            minimum_height = min(520, available.height())
            self.setMinimumSize(minimum_width, minimum_height)
            self.resize(min(620, available.width()), min(720, available.height()))
        #: Set once the person has typed their own name into the field;
        #: :meth:`_sync_default_name` leaves it alone from then on.
        self._name_is_user_set = False
        self._build_ui(terminal_cb)

    def _build_ui(self, terminal_cb: QCheckBox) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        root.addWidget(make_section_label("Target"))
        # No detector here: a board prints the same whichever detector
        # reads it, and that choice is made where it is detected.
        self._target_form = TargetSettingsForm(detector_mode=DETECTOR_NONE)
        self._target_form.changed.connect(self._on_target_changed)
        # A structural change (the target type) goes through
        # _on_target_changed, which also rebuilds the export-options rows;
        # a value edit (n_points, square size, ...) only needs the name
        # re-synced, so it is wired straight to that instead of through
        # the more expensive handler.
        self._target_form.values_changed.connect(self._sync_default_name)
        root.addWidget(self._target_form)

        root.addWidget(make_separator())
        root.addWidget(make_section_label("Printing"))
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        root.addLayout(form)

        self._format_combo = QComboBox()
        self._format_combo.setAccessibleName("Target export format")
        self._format_combo.addItems(list(_EXPORT_CHOICES))
        self._format_combo.currentIndexChanged.connect(self._sync_default_name)
        form.addRow("Export format:", self._format_combo)

        # One row per option the selected target says changes how it draws.
        self._export_rows = QWidget()
        self._export_form = QFormLayout(self._export_rows)
        self._export_form.setContentsMargins(0, 0, 0, 0)
        self._export_widgets: dict[str, QWidget] = {}
        form.addRow(self._export_rows)

        out_row = QHBoxLayout()
        self._out_dir_edit = QLineEdit(str(Path.cwd()))
        self._out_dir_edit.setAccessibleName("Target output directory")
        browse_btn = QPushButton("Browse\u2026")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse_output_dir)
        out_row.addWidget(self._out_dir_edit)
        out_row.addWidget(browse_btn)
        form.addRow("Output directory:", out_row)

        self._name_edit = QLineEdit()
        self._name_edit.setAccessibleName("Target output file name")
        self._name_edit.textEdited.connect(self._on_name_edited)
        form.addRow("Output file name:", self._name_edit)

        self._on_target_changed()

        root.addWidget(make_separator())
        btn_row = QHBoxLayout()
        save_button = make_blue_button("Save Target", self._save_target)
        save_button.setAccessibleName("Save calibration target")
        btn_row.addWidget(save_button)
        visualise_button = make_blue_button("Visualise Target", self._visualise_target)
        visualise_button.setAccessibleName("Visualise calibration target")
        btn_row.addWidget(visualise_button)
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setAccessibleName("Close target dialog")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        self._status = QLabel("")
        self._status.setAccessibleName("Target generation status")
        self._status.setAccessibleDescription("Save and visualisation feedback for the target")
        self._status.setStyleSheet("color: #2e7d32;")
        root.addWidget(self._status)

        self._terminal = TerminalWidget(terminal_cb, parent=self)
        root.addWidget(self._terminal)

    def _browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self._out_dir_edit.setText(path)

    def _target_class(self):
        """The class the target combo names."""
        return target_class(self._target_form.target_type())

    def _on_target_changed(self) -> None:
        """Offer the options this target says change how it is drawn."""
        self._export_widgets = {}
        while self._export_form.rowCount():
            self._export_form.removeRow(0)
        for parameter in self._target_class().export_parameters().settable():
            widget = build_parameter_widget(parameter)
            self._export_form.addRow(f"{parameter.label}:", widget)
            self._export_widgets[parameter.key] = widget
        self._export_rows.setVisible(bool(self._export_widgets))
        self._sync_default_name()

    def _export_kind(self) -> str:
        return _EXPORT_CHOICES[self._format_combo.currentText()]

    def _on_name_edited(self, text: str) -> None:
        """Stop auto-naming once typed into, resume it if cleared back out.

        ``textEdited`` fires only on user keystrokes, never on the
        programmatic ``setText`` :meth:`_sync_default_name` itself does, so
        there is no risk of the two fighting each other. Dropping back to
        auto-naming here has to re-suggest a name immediately -- nothing
        else is going to fire on an empty field left alone, so without
        this it would just stay blank.
        """
        self._name_is_user_set = bool(text)
        if not self._name_is_user_set:
            self._sync_default_name()

    def _sync_default_name(self) -> None:
        """Name the file after the target, until someone names it themselves.

        Called on every structural rebuild and, now, on every value edit
        too -- so a half-typed number is the common case, not the
        exception. :class:`ParamError` from a spec that will not parse yet
        is swallowed the same way it always was: stay quiet, leave the
        previous suggestion showing.
        """
        if self._name_is_user_set:
            return
        try:
            spec = self._target_form.spec()
        except ParamError:
            return
        self._name_edit.setText(
            self._target_class().printable_name(spec, self._export_kind()))

    def _collect(self) -> Optional[dict]:
        """
        The form as a target spec, its export options, and where to write it.

        Every rule the target states about its own arguments is applied by
        the form; what is left here is about the file.

        Whether the target's detector is installed is not asked: printing a
        board and reading one are different things, and a board is often
        printed on a machine that will never detect with it.
        """
        try:
            spec = self._target_form.spec()
            options = self._target_class().export_parameters().parse(
                {key: read_parameter_widget(widget)
                 for key, widget in self._export_widgets.items()})
        except (ParamError, ValueError) as exc:
            QMessageBox.critical(self, "Validation Error", str(exc))
            return None

        out_dir_text = self._out_dir_edit.text().strip()
        if not out_dir_text:
            QMessageBox.critical(self, "Validation Error",
                                 "Output directory is required.")
            return None

        file_name = self._name_edit.text().strip()
        if not file_name:
            QMessageBox.critical(self, "Validation Error",
                                 "Output filename is required.")
            return None
        if ("\x00" in file_name or "/" in file_name or "\\" in file_name
                or file_name in {".", ".."}
                or Path(file_name).stem.upper() in _WINDOWS_RESERVED_NAMES):
            QMessageBox.critical(
                self, "Validation Error",
                "Output filename must be a single, non-reserved filename.")
            return None

        return {
            "spec": spec,
            "options": options,
            "out_dir": Path(out_dir_text),
            "file_name": file_name,
            "export_kind": self._export_kind(),
        }

    def _save_target(self) -> None:
        """Build the target the form describes and write it out."""
        collected = self._collect()
        if collected is None:
            return

        out_dir = collected["out_dir"]
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            target = build_target(collected["spec"])
            out_path = target.save_printable(
                out_dir / collected["file_name"],
                collected["export_kind"],
                **collected["options"])
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))
            self._terminal.append_line(f"ERROR: {exc}")
            return

        self._status.setText(f"Saved target: {out_path}")
        self._terminal.append_line(f"Saved target: {out_path}")

    def _visualise_target(self) -> None:
        """Show the target described by the form, in a process of its own.

        ``Ccube`` and ``PuzzleBoardCube`` draw through pyvista, which
        cannot open a window inside the GUI: see
        :mod:`pyCamSet.gui.viewer_process`.  The other two draw through
        matplotlib, which cannot either, and says so less fatally.
        """
        collected = self._collect()
        if collected is None:
            return

        # spawn_viewer's own Popen call returns in milliseconds; what is
        # slow is the child importing pyCamSet before its window appears
        # (~4.7s), which this process cannot shorten or wait on. A status
        # line said before the click's work starts, forced onto the
        # screen with repaint() rather than left for the next event-loop
        # spin, is what stops the button looking dead in the meantime.
        self._status.setText("Opening the target viewer…")
        self._status.repaint()

        ok, detail = spawn_viewer(
            "pyCamSet.utils.visualise_target",
            [json.dumps(collected["spec"], default=str)],
        )
        if not ok:
            QMessageBox.critical(self, "Visualise Failed", detail)
            self._terminal.append_line(f"ERROR: {detail}")
            return
        self._terminal.append_line("Opened the target in a separate window.")
