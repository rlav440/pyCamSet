"""PySide6 Optimisation tab for detector sweeps and retained-run promotion.

The tab implements the user-facing parts of the Optimisation tab: paths,
calibration target, optimisation mode, calibration controls, ChArUco
detection options (built dynamically from the metadata table), and results /
progress.

Heavy lifting (detection, phase 3, phase 4, scoring, retention, metadata)
lives in :mod:`pyCamSet.optimisation.optimisation_worker` and is invoked from
a worker :class:`~PySide6.QtCore.QThread` so the GUI remains responsive.

Optuna is an *optional* dependency: when it is not importable, the tab is
still created but the "Start" button is disabled with an explanatory tooltip.
"""
from __future__ import annotations

import logging
from functools import partial
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
    ARUCO1_DICT_NAMES,
    MARKER_BACKEND_LABELS,
    TAB_OPTIMISATION,
    WorkspaceManager,
    make_green_button,
    make_orange_button,
    make_section_label,
    marker_backend_availability_text,
    marker_backend_available,
    repopulate_dict_combo,
)
from pyCamSet.optimisation.charuco_detector_metadata import (
    CHARUCO_PARAMETER_METADATA,
)
from pyCamSet.optimisation.charuco_detection_profiles import (
    CHARUCO_DETECTION_PROFILE_NAMES,
    get_charuco_detection_profile,
    make_profile_tooltip,
)
from pyCamSet.optimisation.optimisation_study import (
    MAX_SUCCESSES_HARD_CAP,
    TRIAL_GATING_PROFILE_NAMES,
    TrialGatingSettings,
    TrialResult,
    clamp_retain_count,
    make_trial_gating_settings,
)
from pyCamSet.optimisation.optimisation_promotion import promote_retained_trial
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
_TARGET_CHOICES = ("ChArUco", "Ccube")
_MIN_BOARD_DIMENSION = 2
_MAX_BOARD_DIMENSION = 50


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


class _StudyWorker(QObject):
    """QObject moved to a QThread that drives one :class:`OptimisationStudy` run."""

    progress = Signal(object)  # StudyProgress
    finished = Signal(object, int)  # (retention, n_completed)
    failed = Signal(str)

    def __init__(
        self,
        config: RunConfig,
        cancel_token: CancelToken,
        *,
        phase3_fn=None,
        phase4_fn=None,
    ) -> None:
        super().__init__()
        self._config = config
        self._cancel_token = cancel_token
        self._phase3_fn = phase3_fn
        self._phase4_fn = phase4_fn

    def run(self) -> None:
        try:
            def _progress_cb(p: StudyProgress) -> None:
                self.progress.emit(p)

            if OPTUNA_AVAILABLE and run_optuna_study is not None:
                from pyCamSet.optimisation.optimisation_worker import default_detection_fn
                retention = run_optuna_study(
                    config=self._config,
                    detection_fn=default_detection_fn,
                    phase3_fn=self._phase3_fn,
                    phase4_fn=self._phase4_fn,
                    cancel_token=self._cancel_token,
                    progress_cb=_progress_cb,
                )
                self.finished.emit(retention, getattr(retention, "n_completed", self._config.n_trials))
            else:
                # Fallback: zero-sample sweep (fixed values only).  Useful for
                # sanity-checking the worker plumbing when optuna is missing.
                study = OptimisationStudy(
                    self._config,
                    sampler=lambda i, rows: {},
                    phase3_fn=self._phase3_fn,
                    phase4_fn=self._phase4_fn,
                    cancel_token=self._cancel_token,
                    progress_cb=_progress_cb,
                )
                retention = study.run()
                self.finished.emit(retention, len(study.results))
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
        self._retained_results: list[TrialResult] = []
        self._applying_trial_gating_profile = False
        self._applying_detection_profile = False

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

        layout.addWidget(make_section_label("Trial Gating"))
        layout.addWidget(self._build_trial_gating_section())

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
        self._outdir_edit.setPlaceholderText("(defaults to <dataset>/.pycamset_workspace/optimisation_runs)")
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
        self._target_type_combo.addItems(list(_TARGET_CHOICES))
        self._target_type_combo.currentTextChanged.connect(self._update_target_visibility)
        form.addRow("Target type:", self._target_type_combo)

        self._rows_label = QLabel("Rows (num_squares_y):")
        self._rows_spin = QSpinBox()
        self._rows_spin.setRange(2, 50)
        self._rows_spin.setValue(7)
        form.addRow(self._rows_label, self._rows_spin)

        self._cols_label = QLabel("Cols (num_squares_x):")
        self._cols_spin = QSpinBox()
        self._cols_spin.setRange(2, 50)
        self._cols_spin.setValue(7)
        form.addRow(self._cols_label, self._cols_spin)

        self._square_label = QLabel("Square length (mm):")
        self._square_spin = QDoubleSpinBox()
        self._square_spin.setRange(0.1, 1000.0)
        self._square_spin.setDecimals(3)
        self._square_spin.setValue(30.0)
        form.addRow(self._square_label, self._square_spin)

        self._marker_label = QLabel("Marker fraction:")
        self._marker_spin = QDoubleSpinBox()
        self._marker_spin.setRange(0.1, 1.0)
        self._marker_spin.setDecimals(3)
        self._marker_spin.setSingleStep(0.05)
        self._marker_spin.setValue(0.8)
        form.addRow(self._marker_label, self._marker_spin)

        self._border_label = QLabel("Border fraction:")
        self._border_spin = QDoubleSpinBox()
        self._border_spin.setRange(0.001, 0.9)
        self._border_spin.setDecimals(3)
        self._border_spin.setSingleStep(0.01)
        self._border_spin.setValue(0.1)
        form.addRow(self._border_label, self._border_spin)

        self._aruco_combo = QComboBox()
        self._aruco_combo.addItems(ARUCO1_DICT_NAMES)
        self._aruco_combo.setCurrentText("DICT_4X4_1000")
        form.addRow("ArUco dictionary:", self._aruco_combo)

        self._backend_combo = QComboBox()
        for label, value in MARKER_BACKEND_LABELS.items():
            self._backend_combo.addItem(label, value)
        self._backend_combo.setCurrentText("ArUco 1 (OpenCV)")
        self._backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        form.addRow("Marker backend:", self._backend_combo)
        # FIX 8(g): small availability label near the combo (plan v4 D12),
        # refreshed on combo change so availability is honoured immediately.
        self._backend_status = QLabel(marker_backend_availability_text("aruco1"))
        self._backend_status.setStyleSheet("color: #2a7a2a;")
        form.addRow("", self._backend_status)

        self._legacy_cb = QCheckBox("Legacy pattern")
        form.addRow("", self._legacy_cb)
        self._update_target_visibility(self._target_type_combo.currentText())
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

    def _build_trial_gating_section(self) -> QWidget:
        gb = QGroupBox()
        form = QFormLayout(gb)

        # Let the user start from a named preset before editing values manually.
        self._trial_gating_profile_combo = QComboBox()
        # Keep the dropdown aligned with the requested preset names.
        self._trial_gating_profile_combo.addItems(list(TRIAL_GATING_PROFILE_NAMES))
        # Re-apply the chosen preset whenever the selection changes.
        self._trial_gating_profile_combo.currentTextChanged.connect(self._on_trial_gating_profile_changed)
        form.addRow("Profile:", self._trial_gating_profile_combo)

        # Expose the minimum number of cameras that must contribute detections.
        self._min_gating_cameras_spin = QSpinBox()
        # Keep the backend-safe lower bound of two cameras.
        self._min_gating_cameras_spin.setRange(2, 1000)
        form.addRow("Minimum cameras with detections:", self._min_gating_cameras_spin)

        # Expose the minimum number of valid poses/images.
        self._min_gating_images_spin = QSpinBox()
        # Leave room for both strict and flexible presets.
        self._min_gating_images_spin.setRange(1, 10000)
        form.addRow("Minimum valid images / poses:", self._min_gating_images_spin)

        # Expose the image-coverage threshold as a simple fraction.
        self._min_image_coverage_spin = self._make_fraction_spin()
        form.addRow("Minimum image coverage:", self._min_image_coverage_spin)

        # Expose the camera-coverage threshold as a simple fraction.
        self._min_camera_coverage_spin = self._make_fraction_spin()
        form.addRow("Minimum camera coverage:", self._min_camera_coverage_spin)

        # Expose the multi-camera image-coverage threshold as a simple fraction.
        self._min_multicam_coverage_spin = self._make_fraction_spin()
        form.addRow("Minimum multicam image coverage:", self._min_multicam_coverage_spin)

        # Expose the point-ratio threshold as a user-editable numeric value.
        self._min_point_ratio_spin = QDoubleSpinBox()
        # Allow values above 1.0 because trials can exceed the baseline point count.
        self._min_point_ratio_spin.setRange(0.0, 10.0)
        # Keep enough precision for small manual adjustments.
        self._min_point_ratio_spin.setDecimals(3)
        # Match the other fractional controls' edit step.
        self._min_point_ratio_spin.setSingleStep(0.05)
        form.addRow("Minimum point ratio:", self._min_point_ratio_spin)

        # Flip the preset selector to Custom whenever the user edits a threshold.
        for widget in (
            self._min_gating_cameras_spin,
            self._min_gating_images_spin,
            self._min_image_coverage_spin,
            self._min_camera_coverage_spin,
            self._min_multicam_coverage_spin,
            self._min_point_ratio_spin,
        ):
            # Watch each threshold widget for manual changes.
            widget.valueChanged.connect(self._on_trial_gating_field_changed)

        # Populate the fields with a predictable default preset on first load.
        self._apply_trial_gating_profile("Moderate")
        return gb

    def _make_fraction_spin(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setDecimals(3)
        spin.setSingleStep(0.05)
        return spin

    def _build_detector_section(self) -> QWidget:
        gb = QGroupBox()
        v = QVBoxLayout(gb)
        # Keep a stable key->label map for profile hover/help text.
        key_to_label = self._parameter_key_to_label_map()
        # Add a profile selector so users can pre-fill bounds quickly.
        self._detection_profile_combo = QComboBox()
        # Use the fixed display ordering defined in the profile module.
        self._detection_profile_combo.addItems(list(CHARUCO_DETECTION_PROFILE_NAMES))
        # Re-apply profile bounds whenever the selected profile changes.
        self._detection_profile_combo.currentTextChanged.connect(self._on_detection_profile_changed)
        # Attach per-item and combo-level hover text in the existing tooltip pattern.
        for idx in range(self._detection_profile_combo.count()):
            profile_name = self._detection_profile_combo.itemText(idx)
            tooltip = make_profile_tooltip(profile_name, key_to_label)
            self._detection_profile_combo.setItemData(idx, tooltip, Qt.ItemDataRole.ToolTipRole)
        self._detection_profile_combo.setToolTip(make_profile_tooltip("Balanced", key_to_label))
        form = QFormLayout()
        form.addRow("Detection Profile:", self._detection_profile_combo)
        form_wrap = QWidget()
        form_wrap.setLayout(form)
        v.addWidget(form_wrap)
        # Build one editable parameter row for each metadata entry.
        for entry in CHARUCO_PARAMETER_METADATA:
            row = BoundedSliderRow(entry)
            # Watch bound edits so manual changes can flip the selector to Custom.
            row.boundsChanged.connect(self._on_detection_profile_bounds_changed)
            self._param_rows[entry["key"]] = row
            v.addWidget(row)
        # Populate initial bounds from the default profile on first load.
        self._apply_detection_profile("Balanced")
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

        self._latest_failure_reason = QLabel("Latest trial failure: –")
        self._latest_failure_reason.setWordWrap(True)
        v.addWidget(self._latest_failure_reason)

        counts_wrap = QWidget()
        counts_form = QFormLayout(counts_wrap)
        self._rejected_detection_count = QLabel("0")
        self._rejected_gating_count = QLabel("0")
        self._failed_phase2_count = QLabel("0")
        self._failed_phase3_count = QLabel("0")
        self._failed_phase4_count = QLabel("0")
        self._succeeded_phase3_count = QLabel("0")
        self._succeeded_phase4_count = QLabel("0")
        counts_form.addRow("Rejected at detection:", self._rejected_detection_count)
        counts_form.addRow("Rejected by gating:", self._rejected_gating_count)
        counts_form.addRow("Failed in phase 2:", self._failed_phase2_count)
        counts_form.addRow("Failed in phase 3:", self._failed_phase3_count)
        counts_form.addRow("Failed in phase 4:", self._failed_phase4_count)
        counts_form.addRow("Succeeded in phase 3:", self._succeeded_phase3_count)
        counts_form.addRow("Succeeded in phase 4:", self._succeeded_phase4_count)
        v.addWidget(counts_wrap)

        self._results_table = QTableWidget(0, 10)
        self._results_table.setHorizontalHeaderLabels([
            "rank", "trial", "stage", "score",
            "phase3 RPE", "phase4 RPE",
            "image cov", "cam cov", "multicam cov", "Save",
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

    def _update_target_visibility(self, target_type: str) -> None:
        is_ccube = target_type == "Ccube"
        self._rows_label.setText("Points (n_points):" if is_ccube else "Rows (num_squares_y):")
        self._rows_spin.setRange(_MIN_BOARD_DIMENSION, _MAX_BOARD_DIMENSION)
        self._cols_label.setVisible(not is_ccube)
        self._cols_spin.setVisible(not is_ccube)
        self._square_label.setText("Length (mm):" if is_ccube else "Square length (mm):")
        self._marker_label.setVisible(not is_ccube)
        self._marker_spin.setVisible(not is_ccube)
        self._border_label.setVisible(is_ccube)
        self._border_spin.setVisible(is_ccube)

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

    def _on_backend_changed(self) -> None:
        # Note: the ArUco dictionary combo has no change listeners, so no
        # re-entrancy guard is needed while it is repopulated here.
        repopulate_dict_combo(self._aruco_combo, str(self._backend_combo.currentData() or "aruco1"))
        # FIX 8(g): honour availability on combo change, not only at start.
        backend = str(self._backend_combo.currentData() or "aruco1")
        self._backend_status.setText(marker_backend_availability_text(backend))
        self._backend_status.setStyleSheet(
            "color: #2a7a2a;" if marker_backend_available(backend) else "color: #8a4a00;"
        )

    def _collect_target(self) -> TargetSettings:
        # ArUco dict name → enum, looked up lazily to avoid hard cv2/aruco2
        # dependencies at import. The lookup is backend-aware (plan v4 D9):
        # aruco2 names resolve through the aruco2 package, aruco1 through
        # OpenCV. An unresolvable name is a hard validation error, never a
        # silent fallback to dictionary 0.
        marker_backend = str(self._backend_combo.currentData() or "aruco1")
        dict_name = self._aruco_combo.currentText()
        try:
            if marker_backend == "aruco2":
                import aruco2  # type: ignore[import-not-found]
                a_dict_value = int(getattr(aruco2, dict_name))
            else:
                import cv2  # type: ignore[import-not-found]
                a_dict_value = int(getattr(cv2.aruco, dict_name))
        except (AttributeError, ImportError) as exc:
            # Single-dialog contract: raise with the full detail and let the
            # caller's existing error dialog present it - showing one here as
            # well would double-report the same failure (P2 fix).
            raise ValueError(
                f"Could not resolve ArUco dictionary {dict_name!r} for marker "
                f"backend {marker_backend!r}: {exc}"
            ) from exc
        return TargetSettings(
            target_type=self._target_type_combo.currentText(),
            num_squares_x=int(self._cols_spin.value()),
            num_squares_y=int(self._rows_spin.value()),
            square_size=float(self._square_spin.value()),
            marker_fraction=float(self._marker_spin.value()),
            a_dict=a_dict_value,
            n_points=int(self._rows_spin.value()),
            length=float(self._square_spin.value()),
            border_fraction=float(self._border_spin.value()),
            legacy=self._legacy_cb.isChecked(),
            marker_backend=marker_backend,
        )

    def _collect_controls(self) -> CalibrationControls:
        return CalibrationControls(
            outliers=self._outliers_combo.currentText(),
            max_nfev_phase3=int(self._max_nfev3_spin.value()),
            max_nfev_phase4=int(self._max_nfev4_spin.value()),
            target_rpe=float(self._target_rpe_spin.value()),
            retain_successes=clamp_retain_count(self._retain_spin.value()),
        )

    def _collect_trial_gating(self) -> TrialGatingSettings:
        return TrialGatingSettings(
            profile_name=self._trial_gating_profile_combo.currentText(),
            min_cameras_with_detections=int(self._min_gating_cameras_spin.value()),
            min_valid_images=int(self._min_gating_images_spin.value()),
            min_image_coverage=float(self._min_image_coverage_spin.value()),
            min_camera_coverage=float(self._min_camera_coverage_spin.value()),
            min_multicam_image_coverage=float(self._min_multicam_coverage_spin.value()),
            min_point_ratio=float(self._min_point_ratio_spin.value()),
        )

    def _on_trial_gating_profile_changed(self, profile_name: str) -> None:
        # Ignore recursive signal traffic while a preset is being copied into the fields.
        if self._applying_trial_gating_profile:
            return
        # Leave the current field values untouched when the user selects Custom.
        if profile_name == "Custom":
            return
        # Copy the selected preset values into the editable threshold widgets.
        self._apply_trial_gating_profile(profile_name)

    def _apply_trial_gating_profile(self, profile_name: str) -> None:
        # Resolve the preset name into a concrete settings object.
        settings = make_trial_gating_settings(profile_name)
        # Suppress Custom flip-backs while the preset values are being applied.
        self._applying_trial_gating_profile = True
        try:
            # Keep the dropdown text aligned with the applied preset.
            self._trial_gating_profile_combo.setCurrentText(profile_name)
            # Copy the camera threshold into the UI.
            self._min_gating_cameras_spin.setValue(settings.min_cameras_with_detections)
            # Copy the image/pose threshold into the UI.
            self._min_gating_images_spin.setValue(settings.min_valid_images)
            # Copy the image coverage threshold into the UI.
            self._min_image_coverage_spin.setValue(settings.min_image_coverage)
            # Copy the camera coverage threshold into the UI.
            self._min_camera_coverage_spin.setValue(settings.min_camera_coverage)
            # Copy the multicam coverage threshold into the UI.
            self._min_multicam_coverage_spin.setValue(settings.min_multicam_image_coverage)
            # Copy the point-ratio threshold into the UI.
            self._min_point_ratio_spin.setValue(settings.min_point_ratio)
        finally:
            # Re-enable normal field-change handling after the preset copy finishes.
            self._applying_trial_gating_profile = False

    def _on_trial_gating_field_changed(self, _value) -> None:
        # Ignore signal traffic caused by applying a preset programmatically.
        if self._applying_trial_gating_profile:
            return
        # Keep the selector unchanged if it is already showing Custom.
        if self._trial_gating_profile_combo.currentText() == "Custom":
            return
        # Switch the selector to Custom while preserving the edited field values.
        self._applying_trial_gating_profile = True
        try:
            # Update only the profile name so the manual edits remain visible.
            self._trial_gating_profile_combo.setCurrentText("Custom")
        finally:
            # Re-enable normal preset handling after the combo update.
            self._applying_trial_gating_profile = False

    def _collect_config(self) -> RunConfig:
        seed_text = self._seed_edit.text().strip()
        try:
            seed = int(seed_text) if seed_text else None
        except ValueError as exc:
            # Refuse malformed seeds instead of silently changing the study's
            # reproducibility contract by falling back to an unseeded run.
            raise ValueError(f"Random seed must be an integer, got {seed_text!r}.") from exc
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
            trial_gating=self._collect_trial_gating(),
            output_dir=Path(out) if out else None,
        )

    def _parameter_key_to_label_map(self) -> dict[str, str]:
        """Return a stable key->label mapping for detector parameter UI text."""
        return {entry["key"]: entry.get("label", entry["key"]) for entry in CHARUCO_PARAMETER_METADATA}

    def _on_detection_profile_changed(self, profile_name: str) -> None:
        # Ignore recursive signal traffic while profile bounds are being copied in.
        if self._applying_detection_profile:
            return
        # Keep the current bounds untouched when the selector is set to Custom.
        if profile_name == "Custom":
            return
        # Copy the selected profile's lower/upper bounds into supported rows.
        self._apply_detection_profile(profile_name)

    def _apply_detection_profile(self, profile_name: str) -> None:
        # Resolve profile payload once to keep copies deterministic.
        profile = get_charuco_detection_profile(profile_name)
        # Prepare labels for hover/help text formatting.
        key_to_label = self._parameter_key_to_label_map()
        # Block recursive state flips while bounds are applied row by row.
        self._applying_detection_profile = True
        try:
            # Keep the selector text aligned with the applied profile.
            self._detection_profile_combo.setCurrentText(profile_name)
            # Keep combo hover text aligned with the active profile.
            self._detection_profile_combo.setToolTip(make_profile_tooltip(profile_name, key_to_label))
            # Copy only lower/upper bound values into each supported row.
            for key, row in self._param_rows.items():
                lower = profile["lower_bounds"].get(key)
                upper = profile["upper_bounds"].get(key)
                # Skip rows that are not explicitly covered by this profile.
                if lower is None or upper is None:
                    continue
                row.set_bounds(lower, upper)
        finally:
            # Re-enable normal profile-change handling after copy finishes.
            self._applying_detection_profile = False

    def _on_detection_profile_bounds_changed(self, _key: str, _lower, _upper) -> None:
        # Ignore bound signals emitted while a profile is being applied.
        if self._applying_detection_profile:
            return
        # Keep current selector text when already in Custom mode.
        if self._detection_profile_combo.currentText() == "Custom":
            return
        # Switch selector state to Custom while preserving edited bounds.
        self._applying_detection_profile = True
        try:
            # Update only the selector label and its hover/help text.
            self._detection_profile_combo.setCurrentText("Custom")
            key_to_label = self._parameter_key_to_label_map()
            self._detection_profile_combo.setToolTip(make_profile_tooltip("Custom", key_to_label))
        finally:
            # Re-enable normal selector handling after the state flip.
            self._applying_detection_profile = False

    def _sync_workspace_from_floc(self, f_loc: Path) -> None:
        if f_loc.exists() and f_loc.is_dir():
            ws_path = f_loc / ".pycamset_workspace"
            if self._workspace_mgr.workspace_path != ws_path:
                self._workspace_mgr.set_workspace_path(ws_path, ensure=True)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_start(self) -> None:
        if self._thread is not None:
            QMessageBox.information(self, "Optimisation", "A run is already in progress.")
            return
        if not marker_backend_available(str(self._backend_combo.currentData() or "aruco1")):
            QMessageBox.warning(
                self,
                "Marker backend unavailable",
                "ArUco 2 (aruco2) is selected but the 'aruco2' package is not "
                "installed. Install it with `pip install aruco2` or switch the "
                "marker backend to ArUco 1 (OpenCV).",
            )
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

        self._sync_workspace_from_floc(config.f_loc)
        self._results_table.setRowCount(0)
        self._retained_results = []
        self._progress.setRange(0, config.n_trials)
        self._progress.setValue(0)
        self._latest_failure_reason.setText("Latest trial failure: –")
        self._rejected_detection_count.setText("0")
        self._rejected_gating_count.setText("0")
        self._failed_phase2_count.setText("0")
        self._failed_phase3_count.setText("0")
        self._failed_phase4_count.setText("0")
        self._succeeded_phase3_count.setText("0")
        self._succeeded_phase4_count.setText("0")
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
        self._latest_failure_reason.setText(
            f"Latest trial failure: {p.latest_failure_reason or '–'}"
        )
        self._rejected_detection_count.setText(str(p.rejected_at_detection))
        self._rejected_gating_count.setText(str(p.rejected_by_gating))
        self._failed_phase2_count.setText(str(p.failed_phase2))
        self._failed_phase3_count.setText(str(p.failed_phase3))
        self._failed_phase4_count.setText(str(p.failed_phase4))
        self._succeeded_phase3_count.setText(str(p.succeeded_phase3))
        self._succeeded_phase4_count.setText(str(p.succeeded_phase4))

    def _on_finished(self, retention, n_completed: int) -> None:
        self._populate_results_table(retention.ranked())
        QMessageBox.information(
            self,
            "Optimisation",
            f"Run finished. Retained {len(retention)} successful trial(s); {n_completed} completed.",
        )

    def _populate_results_table(self, results: list[TrialResult]) -> None:
        self._retained_results = list(results)
        self._results_table.setRowCount(len(results))
        for rank, result in enumerate(results, start=1):
            row = rank - 1
            values = [
                rank,
                result.trial_number,
                result.success_stage or "–",
                result.score,
                result.phase3_rpe,
                result.phase4_rpe,
                result.image_coverage,
                result.camera_coverage,
                result.multicam_image_coverage,
            ]
            for col, value in enumerate(values):
                text = "–" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))
                self._results_table.setItem(row, col, QTableWidgetItem(text))
            btn = QPushButton("Save")
            btn.setEnabled(bool(result.saved_metadata_path))
            btn.clicked.connect(partial(self._save_retained_result, row))
            self._results_table.setCellWidget(row, 9, btn)

    def _save_retained_result(self, row: int, _checked: bool = False) -> None:
        if row < 0 or row >= len(self._retained_results):
            return
        result = self._retained_results[row]
        if not result.saved_metadata_path:
            QMessageBox.warning(self, "Optimisation", "This retained run has no saved metadata to promote.")
            return
        reply = QMessageBox.question(
            self,
            "Promote retained run",
            "Promote this retained optimisation run into the normal phase outputs?",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            promoted = promote_retained_trial(self._workspace_mgr, result.saved_metadata_path)
        except Exception as exc:
            QMessageBox.critical(self, "Optimisation", f"Could not promote retained run: {exc}")
            return
        _LOG.info("Promoted optimisation trial %s into workspace runs: %s", result.trial_number, promoted)
        QMessageBox.information(
            self,
            "Optimisation",
            "Promoted retained run: " + ", ".join(f"{phase}={run_id}" for phase, run_id in promoted.items()),
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
