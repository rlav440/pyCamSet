'''
Purpose: The Detection Cost tab -- time calibration-target detection on a folder
         of finished acquisition frames, and report what the streaming
         calibration plan needs (per-frame cost, the stretch's cost, the thread
         answer).
Status:  In Development
Future:  Add a live-folder mode once the producer writes frames atomically, so
         the same tab can watch an acquisition in progress.

WHY IT LIVES HERE
-----------------
This measures pyCamSet's own detectors, so it belongs in pyCamSet next to the
phases it informs, not in the acquisition software that happened to produce the
frames. The engine is `pyCamSet.workflow.detection_cost`, which imports no Qt;
this module only renders it.

THE TARGET FORM ADAPTS TO THE TARGET
------------------------------------
The parameter rows come from `TargetSettingsForm`, which builds them from the
selected target class's own `construction_parameters()`. A Ccube declares no
`num_squares_x`, so a cube is never offered one: the form cannot ask a question
the target has no answer to. Choosing a different target rebuilds the rows.
'''
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

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
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pyCamSet.gui.theme import set_text_role
from pyCamSet.gui.shared_functions import (
    DETECTOR_NONE,
    PhaseWorker,
    TargetSettingsForm,
    make_blue_button,
    make_section_label,
    make_separator,
)

TAB_DETECTION_COST = "Detection Cost"


class DetectionCostTab(QWidget):
    """Time detection on a folder of frames, and show the streaming answer.

    Runs `pyCamSet.workflow.detection_cost` on a worker thread, since a full
    acquisition folder takes minutes. The engine is Qt-free; this tab is a
    renderer for it, so the same measurement is available from a script.
    """

    def __init__(self,
                 notebook=None,
                 info_cb: Optional[QCheckBox] = None,
                 terminal_cb: Optional[QCheckBox] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._notebook = notebook
        self._worker: Optional[PhaseWorker] = None
        self._report: Optional[dict] = None
        self._build_ui()

    # ---------------------------------------------------------------- UI --

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        root.addWidget(make_section_label("Frames"))
        intro = QLabel(
            "Times detection on a folder of finished acquisition frames. "
            "Needs no cameras.\n"
            "Per frame it times the TIFF decode, the Mono16 to uint8 contrast "
            "stretch, and detection on the path the phases use.")
        intro.setWordWrap(True)
        root.addWidget(intro)

        frames_form = QFormLayout()
        frames_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        folder_row = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText(
            "The acquisition folder, holding one subfolder per camera")
        browse = QPushButton("Browse\u2026")
        browse.setFixedWidth(80)
        browse.clicked.connect(self._browse_folder)
        folder_row.addWidget(self._folder_edit)
        folder_row.addWidget(browse)
        frames_form.addRow("Frames folder:", folder_row)

        out_row = QHBoxLayout()
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText(
            "Where the report is written (defaults to the frames folder)")
        browse_out = QPushButton("Browse\u2026")
        browse_out.setFixedWidth(80)
        browse_out.clicked.connect(self._browse_out_dir)
        out_row.addWidget(self._out_edit)
        out_row.addWidget(browse_out)
        frames_form.addRow("Report folder:", out_row)
        root.addLayout(frames_form)

        root.addWidget(make_separator())
        root.addWidget(make_section_label("Target"))
        target_note = QLabel(
            "The fields below are the selected target's own settings: choose a "
            "cube and the board fields are not offered, because that target has "
            "no such setting.")
        target_note.setWordWrap(True)
        root.addWidget(target_note)
        # detector_mode=none: a board's geometry does not depend on which
        # detector reads it, and this measurement always uses the target's own
        # detector. The rows are rebuilt whenever the target changes.
        self._target_form = TargetSettingsForm(detector_mode=DETECTOR_NONE)
        root.addWidget(self._target_form)

        root.addWidget(make_separator())
        root.addWidget(make_section_label("Run"))
        options_form = QFormLayout()
        options_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._max_frames_spin = QSpinBox()
        self._max_frames_spin.setRange(0, 100000)
        self._max_frames_spin.setSpecialValueText("every frame")
        self._max_frames_spin.setValue(0)
        self._max_frames_spin.setToolTip(
            "How many frames per camera to time. Every frame of a real "
            "acquisition takes minutes; a sample is usually enough.")
        options_form.addRow("Frames per camera:", self._max_frames_spin)

        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setRange(0.1, 10000.0)
        self._fps_spin.setDecimals(1)
        self._fps_spin.setValue(16.0)
        self._fps_spin.setToolTip(
            "The whole-rig frame rate the thread answer is computed for. "
            "8 cameras at 2 fps is 16.")
        options_form.addRow("Rate (frames/s, whole rig):", self._fps_spin)

        self._single_thread_cb = QCheckBox(
            "Also time one pass with OpenCV pinned to a single thread")
        self._single_thread_cb.setChecked(True)
        self._single_thread_cb.setToolTip(
            "Separates 'expensive' from 'parallelises well': a detection spread "
            "over several cores looks cheap per frame while costing the same CPU.")
        options_form.addRow("", self._single_thread_cb)
        root.addLayout(options_form)

        button_row = QHBoxLayout()
        self._run_button = make_blue_button("Measure Detection Cost", self._run)
        button_row.addWidget(self._run_button)
        button_row.addStretch()
        root.addLayout(button_row)

        root.addWidget(make_separator())
        root.addWidget(make_section_label("Result"))
        self._summary = QTextEdit()
        self._summary.setReadOnly(True)
        self._summary.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self._summary.setPlaceholderText(
            "The summary appears here, and is also written beside the report.")
        root.addWidget(self._summary, stretch=1)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

    # ------------------------------------------------------------ actions --

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select the frames folder")
        if folder:
            self._folder_edit.setText(folder)
            if not self._out_edit.text().strip():
                self._out_edit.setText(str(Path(folder) / "detection_cost"))

    def _browse_out_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select the report folder")
        if folder:
            self._out_edit.setText(folder)

    def _target_values(self) -> dict:
        """The chosen target's own construction values, from its own form.

        `TargetSettingsForm.spec()` returns the target's declared keys only, so
        a value the target does not have cannot reach the engine from here.
        """
        spec = dict(self._target_form.spec())
        spec.pop("type", None)
        return spec

    def _options(self):
        """The engine options this form describes."""
        from pyCamSet.workflow import detection_cost as engine

        folder = self._folder_edit.text().strip()
        if not folder:
            raise ValueError("Choose the frames folder first.")
        out = self._out_edit.text().strip() or str(
            Path(folder) / "detection_cost")
        frames = self._max_frames_spin.value()
        return engine.MeasurementOptions(
            folder=Path(folder),
            out_dir=Path(out),
            target_type=self._target_form.target_type(),
            target_values=self._target_values(),
            max_frames_per_camera=(frames or None),
            measure_single_thread_too=self._single_thread_cb.isChecked(),
            expected_fps=float(self._fps_spin.value()),
        )

    def _run(self) -> None:
        from pyCamSet.workflow import detection_cost as engine

        try:
            options = self._options()
        except Exception as exc:
            self._status.setText(str(exc))
            set_text_role(self._status, "danger")
            return

        self._run_button.setEnabled(False)
        self._status.setText("Measuring\u2026")
        set_text_role(self._status, None)
        self._summary.setPlainText("")

        def work(append) -> dict:
            append(f"Target: {engine.describe_target(options.spec())}")
            append(f"Folder: {options.folder}")
            report = engine.measure_folder(options)
            append("")
            for line in engine.summarise(report).splitlines():
                append(line)
            return {"report": report}

        def done(result: dict) -> None:
            self._run_button.setEnabled(True)
            report = (result or {}).get("report")
            if report is None:
                self._status.setText("The measurement produced no report.")
                set_text_role(self._status, "danger")
                return
            self._report = report
            self._summary.setPlainText(engine.summarise(report))
            written = report.get("written") or {}
            self._status.setText(
                f"Done. Summary: {written.get('summary')}<br>"
                f"Report: {written.get('json')}")

        def failed(message: str) -> None:
            self._run_button.setEnabled(True)
            self._status.setText(message)
            set_text_role(self._status, "danger")

        worker = PhaseWorker(work, parent=self)
        worker.line_ready.connect(lambda line: self._summary.append(line))
        worker.finished.connect(done)
        worker.error.connect(failed)
        self._worker = worker
        worker.start()
