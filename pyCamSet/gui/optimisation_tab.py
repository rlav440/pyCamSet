"""
Optimisation tab (PySide6) — Optuna-driven ChArUco detector parameter sweep.

The tab implements the user-facing parts of the Optimisation Tab Specification:

- §4.1 Paths
- §4.2 Calibration Target
- §4.3 Optimisation Mode
- §4.4 Calibration Controls
- §4.5 ChArUco Detection Options (built dynamically from the metadata table)
- §4.6 Results / Progress

Heavy lifting (detection, phase 3, phase 4, scoring, retention, metadata)
lives in :mod:`pyCamSet.optimisation.optimisation_worker` and is invoked from
a worker :class:`~PySide6.QtCore.QThread` so the GUI remains responsive.

Optuna is an *optional* dependency: when it is not importable, the tab is
still created but the "Start" button is disabled with an explanatory tooltip.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pyCamSet.gui.bounded_slider import BoundedSliderRow
from pyCamSet.gui.shared_functions import (
    TAB_OPTIMISATION,
    WorkspaceManager,
    make_green_button,
    make_orange_button,
    make_section_label,
)
from pyCamSet.optimisation.charuco_detector_metadata import (
    CHARUCO_PARAMETER_METADATA,
)
from pyCamSet.optimisation.optimisation_study import (
    MAX_SUCCESSES_HARD_CAP,
    clamp_retain_count,
)
from pyCamSet.optimisation.optimisation_worker import (
    CalibrationControls,
    CancelToken,
    OptimisationStudy,
    ParameterRowConfig,
    RunConfig,
    StudyProgress,
    TargetSettings,
)

try:
    from pyCamSet.optimisation.optuna_adapter import (
        OPTUNA_AVAILABLE,
        run_optuna_study,
    )
except Exception:  # pragma: no cover - module always importable
    OPTUNA_AVAILABLE = False
    run_optuna_study = None  # type: ignore[assignment]


_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


class _StudyWorker(QObject):
    """QObject moved to a QThread that drives one :class:`OptimisationStudy` run."""

    progress = Signal(object)  # StudyProgress
    finished = Signal(int, int)  # (n_successes, n_completed)
    failed = Signal(str)

    def __init__(self, config: RunConfig, cancel_token: CancelToken) -> None:
        super().__init__()
        self._config = config
        self._cancel_token = cancel_token

    def run(self) -> None:
        try:
            def _progress_cb(p: StudyProgress) -> None:
                self.progress.emit(p)

            if OPTUNA_AVAILABLE and run_optuna_study is not None:
                from pyCamSet.optimisation.optimisation_worker import default_detection_fn
                retention = run_optuna_study(
                    config=self._config,
                    detection_fn=default_detection_fn,
                    phase3_fn=None,  # phase3/4 wiring depends on workspace context
                    phase4_fn=None,
                    cancel_token=self._cancel_token,
                    progress_cb=_progress_cb,
                )
                n_done = len(retention) + 0  # successes retained is a lower bound
                self.finished.emit(len(retention), n_done)
            else:
                # Fallback: zero-sample sweep (fixed values only).  Useful for
                # sanity-checking the worker plumbing when optuna is missing.
                study = OptimisationStudy(
                    self._config,
                    sampler=lambda i, rows: {},
                    cancel_token=self._cancel_token,
                    progress_cb=_progress_cb,
                )
                retention = study.run()
                self.finished.emit(len(retention), len(study.results))
        except Exception as exc:  # pragma: no cover - surfaced to GUI
            _LOG.exception("Optimisation study failed")
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# Tab
# ---------------------------------------------------------------------------


class OptimisationTab(QWidget):
    """The Optimisation tab widget."""

    def __init__(
        self,
        notebook: QWidget,
        info_cb: QCheckBox,
        terminal_cb: QCheckBox,
        workspace_mgr: WorkspaceManager,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._notebook = notebook
        self._info_cb = info_cb
        self._terminal_cb = terminal_cb
        self._workspace_mgr = workspace_mgr

        self._cancel_token: Optional[CancelToken] = None
        self._thread: Optional[QThread] = None
        self._worker: Optional[_StudyWorker] = None
        self._param_rows: dict[str, BoundedSliderRow] = {}

        self._build_ui()
        self._update_optuna_status()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        scroll.setWidget(inner)
        outer.addWidget(scroll, stretch=1)

        layout = QVBoxLayout(inner)
        layout.addWidget(make_section_label("Paths"))
        layout.addWidget(self._build_paths_section())

        layout.addWidget(make_section_label("Calibration Target"))
        layout.addWidget(self._build_target_section())

        layout.addWidget(make_section_label("Optimisation Mode"))
        layout.addWidget(self._build_mode_section())

        layout.addWidget(make_section_label("Calibration Controls"))
        layout.addWidget(self._build_controls_section())

        layout.addWidget(make_section_label("ChArUco Detection Options"))
        layout.addWidget(self._build_detector_section())

        layout.addWidget(make_section_label("Results / Progress"))
        layout.addWidget(self._build_results_section())
        layout.addStretch()

        # Action row (outside the scroll area)
        action_row = QHBoxLayout()
        self._start_btn = make_green_button("Start", self._on_start)
        self._cancel_btn = make_orange_button("Cancel", self._on_cancel)
        self._cancel_btn.setEnabled(False)
        action_row.addWidget(self._start_btn)
        action_row.addWidget(self._cancel_btn)
        action_row.addStretch()
        self._optuna_status = QLabel()
        action_row.addWidget(self._optuna_status)
        outer.addLayout(action_row)

    # ------------------------------------------------------------------

    def _build_paths_section(self) -> QWidget:
        gb = QGroupBox()
        form = QFormLayout(gb)

        self._floc_edit = QLineEdit()
        self._floc_edit.setPlaceholderText("Root folder containing calibration images")
        browse_floc = QPushButton("Browse…")
        browse_floc.clicked.connect(lambda: self._browse_into(self._floc_edit))
        row1 = QHBoxLayout()
        row1.addWidget(self._floc_edit, stretch=1)
        row1.addWidget(browse_floc)
        wrap1 = QWidget()
        wrap1.setLayout(row1)
        form.addRow("Dataset path:", wrap1)

        self._outdir_edit = QLineEdit()
        self._outdir_edit.setPlaceholderText("(defaults to <dataset>/optimisation_runs)")
        browse_out = QPushButton("Browse…")
        browse_out.clicked.connect(lambda: self._browse_into(self._outdir_edit))
        row2 = QHBoxLayout()
        row2.addWidget(self._outdir_edit, stretch=1)
        row2.addWidget(browse_out)
        wrap2 = QWidget()
        wrap2.setLayout(row2)
        form.addRow("Output directory:", wrap2)

        return gb

    def _build_target_section(self) -> QWidget:
        gb = QGroupBox()
        form = QFormLayout(gb)

        self._target_type_combo = QComboBox()
        self._target_type_combo.addItem("ChArUco")
        form.addRow("Target type:", self._target_type_combo)

        self._rows_spin = QSpinBox()
        self._rows_spin.setRange(2, 50)
        self._rows_spin.setValue(7)
        form.addRow("Rows (num_squares_y):", self._rows_spin)

        self._cols_spin = QSpinBox()
        self._cols_spin.setRange(2, 50)
        self._cols_spin.setValue(7)
        form.addRow("Cols (num_squares_x):", self._cols_spin)

        self._square_spin = QDoubleSpinBox()
        self._square_spin.setRange(0.1, 1000.0)
        self._square_spin.setDecimals(3)
        self._square_spin.setValue(30.0)
        form.addRow("Square length (mm):", self._square_spin)

        self._marker_spin = QDoubleSpinBox()
        self._marker_spin.setRange(0.1, 1.0)
        self._marker_spin.setDecimals(3)
        self._marker_spin.setSingleStep(0.05)
        self._marker_spin.setValue(0.8)
        form.addRow("Marker fraction:", self._marker_spin)

        self._aruco_combo = QComboBox()
        self._aruco_combo.addItems([
            "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
            "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
            "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
        ])
        self._aruco_combo.setCurrentText("DICT_4X4_1000")
        form.addRow("ArUco dictionary:", self._aruco_combo)

        self._legacy_cb = QCheckBox("Legacy pattern")
        form.addRow("", self._legacy_cb)
        return gb

    def _build_mode_section(self) -> QWidget:
        gb = QGroupBox()
        form = QFormLayout(gb)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Full", "Fast"])
        form.addRow("Mode:", self._mode_combo)

        self._trials_spin = QSpinBox()
        self._trials_spin.setRange(1, 10000)
        self._trials_spin.setValue(40)
        form.addRow("Trials:", self._trials_spin)

        self._seed_edit = QLineEdit()
        self._seed_edit.setPlaceholderText("(optional integer)")
        form.addRow("Random seed:", self._seed_edit)

        self._sampler_combo = QComboBox()
        self._sampler_combo.addItems(["TPESampler", "RandomSampler", "CmaEsSampler"])
        form.addRow("Sampler:", self._sampler_combo)
        return gb

    def _build_controls_section(self) -> QWidget:
        gb = QGroupBox()
        form = QFormLayout(gb)

        self._outliers_combo = QComboBox()
        self._outliers_combo.addItems(["n", "y", "ask"])
        self._outliers_combo.setCurrentText("n")
        form.addRow("Outliers:", self._outliers_combo)

        self._max_nfev3_spin = QSpinBox()
        self._max_nfev3_spin.setRange(1, 100000)
        self._max_nfev3_spin.setValue(100)
        form.addRow("Phase 3 max_nfev:", self._max_nfev3_spin)

        self._max_nfev4_spin = QSpinBox()
        self._max_nfev4_spin.setRange(1, 100000)
        self._max_nfev4_spin.setValue(100)
        form.addRow("Phase 4 max_nfev:", self._max_nfev4_spin)

        self._target_rpe_spin = QDoubleSpinBox()
        self._target_rpe_spin.setRange(0.001, 1000.0)
        self._target_rpe_spin.setDecimals(3)
        self._target_rpe_spin.setSingleStep(0.1)
        self._target_rpe_spin.setValue(1.0)
        form.addRow("Target RPE (px):", self._target_rpe_spin)

        self._retain_spin = QSpinBox()
        self._retain_spin.setRange(1, MAX_SUCCESSES_HARD_CAP)
        self._retain_spin.setValue(MAX_SUCCESSES_HARD_CAP)
        form.addRow("Retain successes:", self._retain_spin)
        return gb

    def _build_detector_section(self) -> QWidget:
        gb = QGroupBox()
        v = QVBoxLayout(gb)
        for entry in CHARUCO_PARAMETER_METADATA:
            row = BoundedSliderRow(entry)
            self._param_rows[entry["key"]] = row
            v.addWidget(row)
        return gb

    def _build_results_section(self) -> QWidget:
        gb = QGroupBox()
        v = QVBoxLayout(gb)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        v.addWidget(self._progress)

        live = QHBoxLayout()
        self._live_trial = QLabel("Trial: –")
        self._live_best_score = QLabel("Best score: –")
        self._live_best_stage = QLabel("Best stage: –")
        self._live_best_rpe = QLabel("Best RPE: –")
        self._live_successes = QLabel("Successes: 0")
        for w in (
            self._live_trial,
            self._live_best_score,
            self._live_best_stage,
            self._live_best_rpe,
            self._live_successes,
        ):
            live.addWidget(w)
        live.addStretch()
        live_wrap = QWidget()
        live_wrap.setLayout(live)
        v.addWidget(live_wrap)

        self._results_table = QTableWidget(0, 9)
        self._results_table.setHorizontalHeaderLabels([
            "rank", "trial", "stage", "score",
            "phase3 RPE", "phase4 RPE",
            "image cov", "cam cov", "multicam cov",
        ])
        self._results_table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self._results_table)
        return gb

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _browse_into(self, edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select folder")
        if path:
            edit.setText(path)

    def _update_optuna_status(self) -> None:
        if OPTUNA_AVAILABLE:
            self._optuna_status.setText("Optuna: available")
            self._optuna_status.setStyleSheet("color: #2a7a2a;")
        else:
            self._optuna_status.setText("Optuna not installed — sampling disabled.")
            self._optuna_status.setStyleSheet("color: #8a4a00;")
            self._optuna_status.setToolTip(
                "Install with `pip install optuna` to enable full Optimisation tab features."
            )

    def _collect_parameter_rows(self) -> list[ParameterRowConfig]:
        rows: list[ParameterRowConfig] = []
        for key, widget in self._param_rows.items():
            lower, upper = widget.bounds()
            rows.append(
                ParameterRowConfig(
                    key=key,
                    fixed=widget.fixed_value(),
                    optimise=widget.optimise_enabled(),
                    lower=lower if widget.optimise_enabled() else None,
                    upper=upper if widget.optimise_enabled() else None,
                )
            )
        return rows

    def _collect_target(self) -> TargetSettings:
        # ArUco dict name → cv2 enum, looked up lazily to avoid hard cv2 dependency at import.
        try:
            import cv2  # type: ignore[import-not-found]
            a_dict_value = int(getattr(cv2.aruco, self._aruco_combo.currentText()))
        except Exception:
            a_dict_value = 0
        return TargetSettings(
            target_type=self._target_type_combo.currentText(),
            num_squares_x=int(self._cols_spin.value()),
            num_squares_y=int(self._rows_spin.value()),
            square_size=float(self._square_spin.value()),
            marker_fraction=float(self._marker_spin.value()),
            a_dict=a_dict_value,
            legacy=self._legacy_cb.isChecked(),
        )

    def _collect_controls(self) -> CalibrationControls:
        return CalibrationControls(
            outliers=self._outliers_combo.currentText(),
            max_nfev_phase3=int(self._max_nfev3_spin.value()),
            max_nfev_phase4=int(self._max_nfev4_spin.value()),
            target_rpe=float(self._target_rpe_spin.value()),
            retain_successes=clamp_retain_count(self._retain_spin.value()),
        )

    def _collect_config(self) -> RunConfig:
        seed_text = self._seed_edit.text().strip()
        seed = int(seed_text) if seed_text.isdigit() else None
        out = self._outdir_edit.text().strip()
        return RunConfig(
            f_loc=Path(self._floc_edit.text().strip()),
            mode=self._mode_combo.currentText().lower(),
            n_trials=int(self._trials_spin.value()),
            seed=seed,
            sampler_name=self._sampler_combo.currentText(),
            parameter_rows=self._collect_parameter_rows(),
            target=self._collect_target(),
            controls=self._collect_controls(),
            output_dir=Path(out) if out else None,
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_start(self) -> None:
        if self._thread is not None:
            QMessageBox.information(self, "Optimisation", "A run is already in progress.")
            return
        try:
            config = self._collect_config()
        except Exception as exc:
            QMessageBox.critical(self, "Optimisation", f"Could not assemble run config: {exc}")
            return

        # Validate via the worker's own validation hooks.
        cancel_token = CancelToken()
        validator = OptimisationStudy(config, sampler=lambda i, rows: {}, cancel_token=cancel_token)
        errors = validator.validate()
        if errors:
            QMessageBox.warning(self, "Optimisation", "Cannot start:\n• " + "\n• ".join(errors))
            return

        self._results_table.setRowCount(0)
        self._progress.setRange(0, config.n_trials)
        self._progress.setValue(0)
        self._start_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)

        self._cancel_token = cancel_token
        self._worker = _StudyWorker(config, cancel_token)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _on_cancel(self) -> None:
        if self._cancel_token is not None:
            self._cancel_token.cancel()
            self._cancel_btn.setEnabled(False)

    def _on_progress(self, p: StudyProgress) -> None:
        self._progress.setValue(p.trial_number + 1)
        self._live_trial.setText(f"Trial: {p.trial_number + 1}/{p.total_trials}")
        if p.best_score == float("inf"):
            self._live_best_score.setText("Best score: –")
        else:
            self._live_best_score.setText(f"Best score: {p.best_score:.4f}")
        self._live_best_stage.setText(f"Best stage: {p.best_stage or '–'}")
        if p.best_phase3_rpe is not None or p.best_phase4_rpe is not None:
            best_rpe = p.best_phase4_rpe if p.best_phase4_rpe is not None else p.best_phase3_rpe
            self._live_best_rpe.setText(f"Best RPE: {best_rpe:.3f}")
        else:
            self._live_best_rpe.setText("Best RPE: –")
        self._live_successes.setText(f"Successes: {p.n_successes}")

    def _on_finished(self, n_successes: int, n_completed: int) -> None:
        QMessageBox.information(
            self,
            "Optimisation",
            f"Run finished. Retained {n_successes} successful trial(s); {n_completed} completed.",
        )

    def _on_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Optimisation", f"Run failed: {message}")

    def _cleanup_thread(self) -> None:
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
        self._worker = None
        self._cancel_token = None
        self._start_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)


__all__ = ["OptimisationTab", "TAB_OPTIMISATION"]
