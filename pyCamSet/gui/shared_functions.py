"""
Shared utilities, mixins, widget factories, and constants for the pyCamSet GUI.

All UI is built with PySide6 (Qt6).

Conventions
-----------
- Hover tooltips are gated by a shared ``QCheckBox`` (``info_cb``) via its
  ``isChecked()`` state — set as ``setToolTipsEnabled(bool)`` on every widget.
- The terminal pane is a dark, read-only ``QTextEdit`` written to via
  :meth:`TerminalWidget.append_line`.
- :class:`WorkspaceManager` owns all disk I/O; nothing else writes to the
  workspace directory.
- Background work runs inside :class:`PhaseWorker`, a ``QThread`` subclass
  that emits ``line_ready(str)`` and ``finished(dict)`` signals.

Backwards-compatibility policy
-------------------------------
This module intentionally does **not** support legacy data formats or API
aliases from earlier pyCamSet builds.  If legacy input is detected (e.g. a
``phase5`` folder name or an unrecognised detection-pickle shape), a
``ValueError`` or ``RuntimeError`` is raised immediately with a clear message
so the user can migrate their workspace rather than silently producing
incorrect results.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Tab-name constants (shared across modules)
# ---------------------------------------------------------------------------

TAB_PHASE0 = "Phase 0 - Data Input"
TAB_PHASE1 = "Phase 1 - Detection"
TAB_PHASE1_DIAG = "Phase 1 Diagnostics"
TAB_PHASE2 = "Phase 2 - Intrinsics"
TAB_PHASE2_DIAG = "Phase 2 Diagnostics"
TAB_PHASE3 = "Phase 3 - Bundle Adjustment"
TAB_PHASE3_DIAG = "Phase 3 Diagnostics"
TAB_PHASE4 = "Phase 4 - Self-Calibration"
TAB_PHASE4_DIAG = "Phase 4 Diagnostics"
TAB_OPTIMISATION = "Optimisation"
TAB_CREATE_TARGET = "Create Target"
TAB_PRINT_TARGET = TAB_CREATE_TARGET
TAB_EXPORT_CALIBRATION = "Export Calibration"

# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

ORANGE = "#e07b00"
DARK_ORANGE = "#c06000"
GREEN = "#2e7d32"
DARK_GREEN = "#1b5e20"
SECTION_COLOR = "#1976d2"
BABY_BLUE = "#8fd3ff"
DARK_BABY_BLUE = "#67bde8"

ORANGE_BTN_STYLE = (
    f"QPushButton {{ background-color: {ORANGE}; color: white; font-weight: bold;"
    f" border-radius: 4px; padding: 4px 10px; }}"
    f"QPushButton:hover {{ background-color: {DARK_ORANGE}; }}"
    f"QPushButton:pressed {{ background-color: {DARK_ORANGE}; }}"
)

GREEN_BTN_STYLE = (
    f"QPushButton {{ background-color: {GREEN}; color: white; font-weight: bold;"
    f" border-radius: 4px; padding: 4px 10px; }}"
    f"QPushButton:hover {{ background-color: {DARK_GREEN}; }}"
    f"QPushButton:pressed {{ background-color: {DARK_GREEN}; }}"
)

BLUE_BTN_STYLE = (
    f"QPushButton {{ background-color: {BABY_BLUE}; color: #083b5c; font-weight: bold;"
    f" border-radius: 4px; padding: 4px 10px; }}"
    f"QPushButton:hover {{ background-color: {DARK_BABY_BLUE}; }}"
    f"QPushButton:pressed {{ background-color: {DARK_BABY_BLUE}; }}"
)

SECTION_STYLE = "QLabel { color: #1976d2; font-weight: bold; margin-top: 6px; }"

TERMINAL_STYLE = (
    "QTextEdit { background: #1e1e1e; color: #d4d4d4; font-family: Courier, monospace;"
    " font-size: 10pt; border: none; }"
)

# ---------------------------------------------------------------------------
# Section label factory
# ---------------------------------------------------------------------------


def make_section_label(text: str) -> QLabel:
    """Return a styled section-header label."""
    lbl = QLabel(text)
    lbl.setStyleSheet(SECTION_STYLE)
    return lbl


def make_separator() -> QFrame:
    """Return a horizontal separator line."""
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setFrameShadow(QFrame.Shadow.Sunken)
    return sep


def make_orange_button(text: str, callback: Callable) -> QPushButton:
    """Return an orange push button connected to *callback*."""
    btn = QPushButton(text)
    btn.setStyleSheet(ORANGE_BTN_STYLE)
    btn.clicked.connect(callback)
    return btn


def make_green_button(text: str, callback: Callable) -> QPushButton:
    """Return a green push button connected to *callback*."""
    btn = QPushButton(text)
    btn.setStyleSheet(GREEN_BTN_STYLE)
    btn.clicked.connect(callback)
    return btn


def make_blue_button(text: str, callback: Callable) -> QPushButton:
    """Return a baby-blue action button connected to *callback*."""
    btn = QPushButton(text)
    btn.setStyleSheet(BLUE_BTN_STYLE)
    btn.clicked.connect(callback)
    return btn


def make_continue_button(callback: Callable) -> QPushButton:
    """Return a green "Continue to Next Phase" button."""
    btn = QPushButton("Continue to Next Phase ▶")
    btn.setStyleSheet(GREEN_BTN_STYLE)
    btn.clicked.connect(callback)
    return btn


class MatplotlibFigureCard(QWidget):
    """A labelled matplotlib card with per-figure Expand and Save PNG buttons."""

    def __init__(
        self,
        title: str,
        fig,
        canvas_cls,
        parent: Optional[QWidget] = None,
        min_height: int = 300,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._fig = fig
        self._canvas_cls = canvas_cls

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 8)

        header = QHBoxLayout()
        header.addWidget(make_section_label(title))
        header.addStretch()
        save_btn = QPushButton("Save PNG")
        save_btn.setFixedWidth(82)
        save_btn.clicked.connect(self._save_png)
        header.addWidget(save_btn)
        expand_btn = QPushButton("Expand")
        expand_btn.setFixedWidth(78)
        expand_btn.clicked.connect(self._open_expanded)
        header.addWidget(expand_btn)
        layout.addLayout(header)

        self._canvas = self._canvas_cls(self._fig)
        self._canvas.setMinimumHeight(min_height)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout.addWidget(self._canvas)

    def _save_png(self) -> None:
        import re
        safe_title = re.sub(r'[^\w\s-]', '', self._title).strip().replace(' ', '_')
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Figure as PNG", f"{safe_title}.png", "PNG Files (*.png)"
        )
        if path:
            self._fig.savefig(path, dpi=150, bbox_inches="tight")

    def _open_expanded(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(self._title)
        dlg.resize(1100, 780)
        root = QVBoxLayout(dlg)
        canvas = self._canvas_cls(self._fig)
        canvas.setMinimumHeight(700)
        root.addWidget(canvas)
        dlg.exec()


def make_scrollable_tab() -> tuple[QWidget, QVBoxLayout, QScrollArea]:
    """Return (tab_widget, inner_layout, scroll_area) for vertically scrollable tab content."""
    tab = QWidget()
    root = QVBoxLayout(tab)
    root.setContentsMargins(0, 0, 0, 0)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)

    inner = QWidget()
    inner_layout = QVBoxLayout(inner)
    inner_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
    scroll.setWidget(inner)

    root.addWidget(scroll)
    return tab, inner_layout, scroll


# ---------------------------------------------------------------------------
# Collapsible section widget
# ---------------------------------------------------------------------------


class CollapsibleSection(QWidget):
    """A labelled section with a toggle header button and collapsible QFormLayout body.

    Usage::

        section = CollapsibleSection("Paths", expanded=True)
        section.addRow("Image folder:", edit)
        layout.addWidget(section)
    """

    def __init__(self, title: str, expanded: bool = True, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._title = title
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 2, 0, 2)
        root.setSpacing(0)

        self._btn = QPushButton()
        self._btn.setCheckable(True)
        self._btn.setChecked(expanded)
        self._btn.setStyleSheet(
            "QPushButton { text-align: left; font-weight: bold; color: #1976d2;"
            " background: transparent; border: none; padding: 2px 0px; font-size: 10pt; }"
            "QPushButton:hover { color: #0d47a1; }"
        )
        self._btn.clicked.connect(self._on_toggle)
        root.addWidget(self._btn)

        self._body = QWidget()
        self._form = QFormLayout(self._body)
        self._form.setContentsMargins(4, 0, 0, 4)
        self._form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        root.addWidget(self._body)

        # Keep visual state and body visibility in sync at startup.
        self._body.setVisible(expanded)
        self._update_label(expanded)

    def _update_label(self, expanded: bool) -> None:
        arrow = "▼" if expanded else "▶"
        self._btn.setText(f"{arrow}  {self._title}")

    def _on_toggle(self, checked: bool) -> None:
        self._body.setVisible(checked)
        self._update_label(checked)
        self._btn.setChecked(checked)

    def addRow(self, *args) -> None:  # noqa: N802
        """Proxy for the internal :class:`QFormLayout.addRow`."""
        self._form.addRow(*args)

    def form(self) -> QFormLayout:
        """Return the internal :class:`QFormLayout`."""
        return self._form


def make_run_id() -> str:
    """Return a unique, timestamp-ordered run identifier.

    Format: ``YYYYMMDD_HHMMSS_<6-char hex>``
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"


CHARUCO_DETECTION_OPTION_METADATA: list[dict[str, Any]] = [
    {
        "key": "DetectorParameters.minMarkerPerimeterRate",
        "label": "minMarkerPerimeterRate",
        "priority": "A",
        "default": 0.03,
        "range": "Not explicitly specified in source; practically should be > 0",
        "range_source": "Estimated by us",
        "suggested": "0.01–0.02",
        "concept": (
            "Concept: minimum candidate marker size, expressed relative to image size. "
            "Detection: rejects contours whose perimeter is too small before decoding. "
            "Calibration: if valid small markers are rejected, ChArUco corners never exist, "
            "so calibration fails from lack of correspondences."
        ),
        "widget_type": "float",
        "parser_type": "positive_float",
    },
    {
        "key": "DetectorParameters.adaptiveThreshWinSizeMin",
        "label": "adaptiveThreshWinSizeMin",
        "priority": "A",
        "default": 3,
        "range": "Not explicitly specified in source; should be a positive integer",
        "range_source": "Estimated by us",
        "suggested": "3",
        "concept": (
            "Concept: smallest local window used for adaptive thresholding. "
            "Detection: controls the finest local binarisation scale tested before contour extraction. "
            "Calibration: affects whether marker edges survive thresholding; poor thresholding reduces usable observations."
        ),
        "widget_type": "int",
        "parser_type": "positive_int",
    },
    {
        "key": "DetectorParameters.adaptiveThreshWinSizeMax",
        "label": "adaptiveThreshWinSizeMax",
        "priority": "A",
        "default": 23,
        "range": "Not explicitly specified in source; should be >= min",
        "range_source": "Estimated by us",
        "suggested": "31 or 41",
        "concept": (
            "Concept: largest local window used for adaptive thresholding. "
            "Detection: controls the coarsest local binarisation scale tested. "
            "Calibration: if the thresholding scale is mismatched to marker scale or illumination structure, "
            "detections become sparse or unstable."
        ),
        "widget_type": "int",
        "parser_type": "positive_int",
    },
    {
        "key": "DetectorParameters.adaptiveThreshWinSizeStep",
        "label": "adaptiveThreshWinSizeStep",
        "priority": "A",
        "default": 10,
        "range": "Not explicitly specified in source; should be a positive integer",
        "range_source": "Estimated by us",
        "suggested": "4 or 6",
        "concept": (
            "Concept: spacing between tested adaptive-threshold window sizes. "
            "Detection: determines how densely OpenCV samples threshold scales. "
            "Calibration: coarse sampling can miss the useful threshold regime and reduce "
            "the number of stable points available for calibration."
        ),
        "widget_type": "int",
        "parser_type": "positive_int",
    },
    {
        "key": "DetectorParameters.adaptiveThreshConstant",
        "label": "adaptiveThreshConstant",
        "priority": "A",
        "default": 7,
        "range": "Not explicitly specified in source",
        "range_source": "Estimated by us",
        "suggested": "3–7",
        "concept": (
            "Concept: constant subtracted in adaptive thresholding. "
            "Detection: shifts the black/white decision boundary during binarisation. "
            "Calibration: indirectly controls whether markers decode consistently across views, "
            "which affects point count and repeatability."
        ),
        "widget_type": "float",
        "parser_type": "float",
    },
    {
        "key": "CharucoParameters.minMarkers",
        "label": "minMarkers",
        "priority": "A",
        "default": 2,
        "range": "0–2 (reported in earlier investigation; not independently verified here)",
        "range_source": "Estimated by us",
        "suggested": "1",
        "concept": (
            "Concept: required local marker support for accepting an interpolated ChArUco corner. "
            "Detection: filters out corners that do not have enough adjacent detected markers. "
            "Calibration: lower values increase point count but may lower point reliability; "
            "this is a count-versus-quality trade-off."
        ),
        "widget_type": "int",
        "parser_type": "int_range",
        "min_value": 0,
        "max_value": 2,
    },
    {
        "key": "DetectorParameters.cornerRefinementMethod",
        "label": "cornerRefinementMethod",
        "priority": "B",
        "default": "CORNER_REFINE_NONE",
        "range": "Enum: CORNER_REFINE_NONE, CORNER_REFINE_SUBPIX, CORNER_REFINE_CONTOUR, CORNER_REFINE_APRILTAG",
        "range_source": "OpenCV-provided",
        "suggested": "CORNER_REFINE_SUBPIX",
        "concept": (
            "Concept: method for refining detected ArUco marker corners. "
            "Detection: improves marker-corner localisation before ChArUco interpolation. "
            "Calibration: better marker corners generally improve interpolated chessboard corners "
            "and reduce reprojection residuals."
        ),
        "widget_type": "enum",
        "parser_type": "enum",
        "choices": [
            "CORNER_REFINE_NONE",
            "CORNER_REFINE_SUBPIX",
            "CORNER_REFINE_CONTOUR",
            "CORNER_REFINE_APRILTAG",
        ],
    },
    {
        "key": "DetectorParameters.cornerRefinementWinSize",
        "label": "cornerRefinementWinSize",
        "priority": "B",
        "default": 5,
        "range": "Not explicitly specified in source",
        "range_source": "Estimated by us",
        "suggested": "3–5",
        "concept": (
            "Concept: local search window for marker-corner refinement. "
            "Detection: sets the neighbourhood over which corner refinement operates. "
            "Calibration: too small may under-refine; too large may drift on blurred or aliased edges, "
            "affecting geometric precision."
        ),
        "widget_type": "int",
        "parser_type": "positive_int",
    },
    {
        "key": "DetectorParameters.cornerRefinementMaxIterations",
        "label": "cornerRefinementMaxIterations",
        "priority": "B",
        "default": 30,
        "range": "Not explicitly specified in source",
        "range_source": "Estimated by us",
        "suggested": "30",
        "concept": (
            "Concept: maximum iteration count for marker-corner refinement. "
            "Detection: limits the refinement solver effort. "
            "Calibration: mainly affects convergence robustness of localised corners rather than "
            "whether markers are found at all."
        ),
        "widget_type": "int",
        "parser_type": "positive_int",
    },
    {
        "key": "DetectorParameters.cornerRefinementMinAccuracy",
        "label": "cornerRefinementMinAccuracy",
        "priority": "B",
        "default": 0.1,
        "range": "Not explicitly specified in source",
        "range_source": "Estimated by us",
        "suggested": "0.05–0.1",
        "concept": (
            "Concept: stopping tolerance for marker-corner refinement. "
            "Detection: determines when refinement is considered converged. "
            "Calibration: tighter values may improve localisation slightly, but usually with "
            "diminishing returns if the image data are weak."
        ),
        "widget_type": "float",
        "parser_type": "positive_float",
    },
    {
        "key": "DetectorParameters.minOtsuStdDev",
        "label": "minOtsuStdDev",
        "priority": "B",
        "default": 5.0,
        "range": "Not explicitly specified in source",
        "range_source": "Estimated by us",
        "suggested": "2–4",
        "concept": (
            "Concept: minimum local intensity variation required before Otsu thresholding is used in decoding. "
            "Detection: affects how marker bits are binarised during decoding in low-contrast patches. "
            "Calibration: relevant only if markers are found but fail decoding due to weak contrast."
        ),
        "widget_type": "float",
        "parser_type": "float",
    },
    {
        "key": "RefineParameters.minRepDistance",
        "label": "minRepDistance",
        "priority": "B",
        "default": 10.0,
        "range": "Not explicitly specified in source; should be non-negative in practice",
        "range_source": "Estimated by us",
        "suggested": "10–20",
        "concept": (
            "Concept: tolerance for recovering rejected markers using board-guided refinement. "
            "Detection: larger values allow more rejected candidates to be recovered as valid markers. "
            "Calibration: can increase point count when detections are sparse, but may admit worse "
            "correspondences if too loose."
        ),
        "widget_type": "float",
        "parser_type": "non_negative_float",
    },
    {
        "key": "DetectorParameters.errorCorrectionRate",
        "label": "errorCorrectionRate",
        "priority": "C",
        "default": 0.6,
        "range": "Not explicitly specified in header comment",
        "range_source": "Estimated by us",
        "suggested": "0.6–0.8",
        "concept": (
            "Concept: tolerance for correcting bit errors during marker identification. "
            "Detection: increases permissiveness during dictionary matching. "
            "Calibration: may recover weak markers, but if too permissive may increase false IDs, "
            "which is geometrically harmful."
        ),
        "widget_type": "float",
        "parser_type": "float",
    },
    {
        "key": "DetectorParameters.perspectiveRemoveIgnoredMarginPerCell",
        "label": "perspectiveRemoveIgnoredMarginPerCell",
        "priority": "C",
        "default": 0.13,
        "range": "Not explicitly specified in source",
        "range_source": "Estimated by us",
        "suggested": "0.08–0.13",
        "concept": (
            "Concept: ignored fraction at cell borders during marker bit sampling. "
            "Detection: reduces contamination from cell-border transitions when decoding. "
            "Calibration: matters only indirectly through decode stability; usually secondary unless "
            "markers are very small in pixels."
        ),
        "widget_type": "float",
        "parser_type": "float",
    },
    {
        "key": "DetectorParameters.polygonalApproxAccuracyRate",
        "label": "polygonalApproxAccuracyRate",
        "priority": "C",
        "default": 0.03,
        "range": "Not explicitly specified in source",
        "range_source": "Estimated by us",
        "suggested": "0.03–0.05",
        "concept": (
            "Concept: contour simplification tolerance when deciding whether a candidate looks square. "
            "Detection: affects quad fitting from thresholded contours. "
            "Calibration: only matters if contour extraction is the limiting stage; otherwise it is "
            "secondary to thresholding and size rejection."
        ),
        "widget_type": "float",
        "parser_type": "float",
    },
    {
        "key": "CharucoParameters.cameraMatrix",
        "label": "cameraMatrix",
        "priority": "C",
        "default": "",
        "range": "Matrix presence/absence, not a scalar range",
        "range_source": "OpenCV-provided",
        "suggested": "leave empty initially",
        "concept": (
            "Concept: optional camera model for camera-aware ChArUco interpolation. "
            "Detection: switches interpolation from local homography to a pose/projection-based route when provided. "
            "Calibration: can improve interpolation in some regimes, but is not the first lever when "
            "the main failure is missing markers."
        ),
        "widget_type": "json_matrix",
        "parser_type": "optional_json_matrix_3x3",
        "drop_if_none": True,
    },
    {
        "key": "CharucoParameters.distCoeffs",
        "label": "distCoeffs",
        "priority": "C",
        "default": "",
        "range": "Vector presence/absence, not a scalar range",
        "range_source": "OpenCV-provided",
        "suggested": "leave empty initially",
        "concept": (
            "Concept: optional distortion model paired with cameraMatrix. "
            "Detection: used only in the camera-aware interpolation path. "
            "Calibration: relevant if distortion is non-negligible and interpolation quality, rather than "
            "raw marker count, is the limiting issue."
        ),
        "widget_type": "json_vector",
        "parser_type": "optional_json_vector",
        "drop_if_none": True,
    },
]


def build_charuco_option_tooltip(meta: dict[str, Any]) -> str:
    """Build canonical tooltip/info text from a ChArUco option metadata row."""
    return (
        f"{meta['concept']}\n\n"
        f"Default: {meta['default']}\n"
        f"Range: {meta['range']}\n"
        f"Range source: {meta['range_source']}\n"
        f"Suggested value(s): {meta['suggested']}"
    )


def _parse_charuco_value(meta: dict[str, Any], raw_value: Any):
    key = meta["key"]
    parser = meta["parser_type"]
    value = raw_value
    if isinstance(value, str):
        value = value.strip()
    if value in (None, ""):
        value = meta["default"]

    if parser == "enum":
        choices = list(meta.get("choices", []))
        if value not in choices:
            raise ValueError(f"{key} must be one of {choices}.")
        return value

    if parser in {"int", "positive_int", "int_range"}:
        try:
            out = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be an integer.") from exc
        if parser == "positive_int" and out <= 0:
            raise ValueError(f"{key} must be >= 1.")
        if parser == "int_range":
            min_value = int(meta.get("min_value", out))
            max_value = int(meta.get("max_value", out))
            if out < min_value or out > max_value:
                raise ValueError(f"{key} must be between {min_value} and {max_value}.")
        return out

    if parser in {"float", "positive_float", "non_negative_float"}:
        try:
            out = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a number.") from exc
        if parser == "positive_float" and out <= 0:
            raise ValueError(f"{key} must be > 0.")
        if parser == "non_negative_float" and out < 0:
            raise ValueError(f"{key} must be >= 0.")
        return out

    if parser == "optional_json_matrix_3x3":
        if value in ("", None):
            return None
        try:
            data = json.loads(value) if isinstance(value, str) else value
        except json.JSONDecodeError as exc:
            raise ValueError(f"{key} must be valid JSON.") from exc
        if not isinstance(data, list) or len(data) != 3:
            raise ValueError(f"{key} must be a 3x3 JSON array.")
        for row in data:
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError(f"{key} must be a 3x3 JSON array.")
            for elem in row:
                try:
                    float(elem)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{key} entries must be numeric.") from exc
        return data

    if parser == "optional_json_vector":
        if value in ("", None):
            return None
        try:
            data = json.loads(value) if isinstance(value, str) else value
        except json.JSONDecodeError as exc:
            raise ValueError(f"{key} must be valid JSON.") from exc
        if not isinstance(data, list):
            raise ValueError(f"{key} must be a JSON array.")
        for elem in data:
            if isinstance(elem, list):
                for sub_elem in elem:
                    try:
                        float(sub_elem)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"{key} entries must be numeric.") from exc
            else:
                try:
                    float(elem)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{key} entries must be numeric.") from exc
        return data

    raise ValueError(f"Unsupported parser type {parser!r} for {key}.")


def collect_charuco_detection_options(raw_options: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate and normalise ChArUco detector options from raw GUI values."""
    parsed_by_key: dict[str, Any] = {}
    for meta in CHARUCO_DETECTION_OPTION_METADATA:
        key = meta["key"]
        parsed_by_key[key] = _parse_charuco_value(meta, raw_options.get(key))

    min_key = "DetectorParameters.adaptiveThreshWinSizeMin"
    max_key = "DetectorParameters.adaptiveThreshWinSizeMax"
    if parsed_by_key[max_key] < parsed_by_key[min_key]:
        raise ValueError(f"{max_key} must be >= {min_key}.")

    options: dict[str, dict[str, Any]] = {
        "DetectorParameters": {},
        "CharucoParameters": {},
        "RefineParameters": {},
    }
    meta_by_key = {meta["key"]: meta for meta in CHARUCO_DETECTION_OPTION_METADATA}
    for key, value in parsed_by_key.items():
        group, name = key.split(".", 1)
        if value is None and meta_by_key.get(key, {}).get("drop_if_none", False):
            continue
        options[group][name] = value
    return options


def build_target(
    target_type: str,
    n_points: int,
    length: float,
    charuco_detection_options: dict[str, dict[str, Any]] | None = None,
):
    """Construct a calibration target from canonical GUI options."""
    from pyCamSet.calibration_targets.target_Ccube import Ccube
    from pyCamSet.calibration_targets.target_charuco import ChArUco

    if target_type == "Ccube":
        return Ccube(
            n_points=n_points,
            length=length,
            detection_options=charuco_detection_options,  # Ccube detection also runs through ChArUco boards.
        )
    if target_type == "ChArUco":
        return ChArUco(
            num_squares_x=n_points,
            num_squares_y=n_points,
            square_size=length,
            detection_options=charuco_detection_options,
        )
    raise ValueError(f"Unknown target type: {target_type!r}")


def extract_detection_and_cam_res(payload):
    """Extract ``(TargetDetection, cam_res)`` from a persisted detection payload.

    Accepted formats
    ----------------
    1. A bare ``TargetDetection`` object (has ``get_cam_list``).
    2. A ``(TargetDetection, cam_res)`` tuple/list produced by
       ``detect_datapoints_in_imfile``.

    Raises
    ------
    ValueError
        If the payload is a legacy dict format or any other unrecognised shape.
        Migrate the workspace by re-running Phase 1 to produce a current-format
        pickle.
    """
    if hasattr(payload, "get_cam_list"):
        return payload, None

    if isinstance(payload, (tuple, list)) and len(payload) >= 1:
        detections = payload[0]
        cam_res = payload[1] if len(payload) > 1 else None
        if hasattr(detections, "get_cam_list"):
            return detections, cam_res

    if isinstance(payload, dict):
        raise ValueError(
            "Legacy dict-format detection pickle detected.  "
            "Re-run Phase 1 to produce a current-format detected_datapoints.pickle."
        )

    raise ValueError(
        f"Unrecognised detection payload type: {type(payload).__name__!r}.  "
        "Re-run Phase 1 to produce a current-format detected_datapoints.pickle."
    )


def extract_detection(payload):
    """Extract only TargetDetection from common persisted payload shapes."""
    detections, _ = extract_detection_and_cam_res(payload)
    return detections


# ---------------------------------------------------------------------------
# Terminal widget
# ---------------------------------------------------------------------------


_ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

class TerminalWidget(QTextEdit):
    """A dark, append-only terminal pane.

    Controlled by a :class:`QCheckBox`; hidden when the box is unchecked.
    """

    def __init__(self, show_cb: QCheckBox, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet(TERMINAL_STYLE)
        self.setMinimumHeight(110)
        self.setMaximumHeight(200)
        self._show_cb = show_cb
        show_cb.stateChanged.connect(self._on_toggle)
        self._on_toggle()

    def append_line(self, text: str) -> None:
        """Append *text* + newline and scroll to bottom."""
        clean = _ANSI_RE.sub("", str(text)).replace("\r", "")
        self.moveCursor(QTextCursor.MoveOperation.End)
        self.insertPlainText(clean + "\n")
        self.ensureCursorVisible()

    def clear_terminal(self) -> None:
        """Erase all terminal content."""
        self.clear()

    def _on_toggle(self) -> None:
        self.setVisible(self._show_cb.isChecked())


# ---------------------------------------------------------------------------
# Workspace manager
# ---------------------------------------------------------------------------


class WorkspaceManager:
    """Manages workspace directory and run metadata.

    Supported phases: ``phase0`` through ``phase4``.  Legacy phase names such
    as ``phase5``, ``visualise_target``, or ``assess_calibration`` are **not**
    supported; passing them raises ``ValueError`` immediately.
    """

    _KNOWN_PHASES = {"phase0", "phase1", "phase2", "phase3", "phase4"}
    _PHASE_ORDER = ["phase0", "phase1", "phase2", "phase3", "phase4"]

    def __init__(self, workspace_path: Optional[Path] = None) -> None:
        self.workspace_path: Optional[Path] = Path(workspace_path) if workspace_path else None
        if self.workspace_path is not None:
            self.ensure_dirs()

    @classmethod
    def canonical_phase_name(cls, phase: str) -> str:
        """Normalise *phase* to a canonical lower-case name.

        Raises
        ------
        ValueError
            If *phase* is a known legacy alias (e.g. ``phase5``,
            ``visualise_target``) so the caller gets a clear migration hint
            instead of silently reading stale data.
        """
        p = (phase or "").strip().lower()
        _LEGACY = {
            "phase5", "phase_5", "phase-5", "phase 5",
            "visualise_target", "assess_calibration",
        }
        if p in _LEGACY:
            raise ValueError(
                f"Legacy phase name {phase!r} is no longer supported.  "
                "Re-run calibration to produce phase4 output in the current workspace layout."
            )
        return p

    @classmethod
    def _phase_run_dirs(cls, phase: str) -> list[str]:
        canonical = cls.canonical_phase_name(phase)
        return [f"{canonical}_runs"]

    def set_workspace_path(self, workspace_path: Path, ensure: bool = True) -> None:
        """Set workspace path; optionally create standard sub-directories."""
        self.workspace_path = Path(workspace_path)
        if ensure:
            self.ensure_dirs()

    def ensure_dirs(self) -> None:
        """Create standard sub-directories if workspace path is set."""
        if self.workspace_path is None:
            return
        for sub in (
            "phase0_runs",
            "phase1_runs",
            "phase2_runs",
            "phase3_runs",
            "phase4_runs",
        ):
            (self.workspace_path / sub).mkdir(parents=True, exist_ok=True)

    def save_run(self, phase: str, run_id: str, metadata: dict) -> Path:
        """Persist metadata to <workspace>/<phase>_runs/<run_id>/metadata.json."""
        if self.workspace_path is None:
            raise RuntimeError("Workspace path is not set.")
        phase_dir = self._phase_run_dirs(phase)[0]
        run_dir = self.workspace_path / phase_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        meta_path = run_dir / "metadata.json"
        with open(meta_path, "w") as fh:
            json.dump(metadata, fh, indent=2, default=str)
        return meta_path

    def load_runs(self, phase: str) -> list[dict]:
        """Return all saved runs for phase sorted oldest-first."""
        if self.workspace_path is None:
            return []

        results: list[dict] = []
        for phase_dir in self._phase_run_dirs(phase):
            runs_dir = self.workspace_path / phase_dir
            if not runs_dir.exists():
                continue
            for run_dir in sorted(runs_dir.iterdir()):
                meta_path = run_dir / "metadata.json"
                if meta_path.exists():
                    try:
                        with open(meta_path) as fh:
                            data = json.load(fh)
                        data.setdefault("run_id", run_dir.name)
                        results.append(data)
                    except (json.JSONDecodeError, OSError):
                        pass

        results.sort(key=lambda d: str(d.get("run_id", "")))
        return results

    def build_predecessor_chain(self, run: dict) -> list[dict]:
        """Return deep copies of all predecessor runs for *run*, oldest-first.

        Follows ``inputs.phaseN_run_id`` links through persisted workspace
        metadata.  At each step the direct parent is the highest-phase entry in
        the current run's ``inputs`` dict.  The chain terminates when no
        further parent can be resolved.

        Returns an empty list if *run* has no tracked predecessors.
        """
        chain: list[dict] = []
        visited: set[str] = set()
        current = run

        for _ in range(len(self._PHASE_ORDER)):
            inputs = current.get("inputs") or {}

            best_phase_idx = -1
            parent_phase: Optional[str] = None
            parent_run_id: Optional[str] = None

            for key, val in inputs.items():
                if not (key.endswith("_run_id") and val):
                    continue
                phase = key[: -len("_run_id")]
                try:
                    idx = self._PHASE_ORDER.index(phase)
                except ValueError:
                    continue
                if idx > best_phase_idx:
                    best_phase_idx = idx
                    parent_phase = phase
                    parent_run_id = str(val)

            if parent_run_id is None or parent_run_id in visited:
                break

            visited.add(parent_run_id)
            try:
                parent_runs = self.load_runs(parent_phase)  # type: ignore[arg-type]
            except ValueError:
                break
            parent = next(
                (r for r in parent_runs if r.get("run_id") == parent_run_id),
                None,
            )
            if parent is None:
                break

            chain.append(copy.deepcopy(parent))
            current = parent

        chain.reverse()
        return chain

    def write_handoff(self, payload: dict) -> None:
        """Write payload to <workspace>/handoff.json."""
        if self.workspace_path is None:
            raise RuntimeError("Workspace path is not set.")
        with open(self.workspace_path / "handoff.json", "w") as fh:
            json.dump(payload, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# Run selector widget
# ---------------------------------------------------------------------------


class RunSelectorWidget(QWidget):
    """A ``QListWidget``-based multi-select of saved runs.

    The most recent ``min(3, n)`` runs are pre-selected on every
    :meth:`refresh`.

    :param on_select: Optional callback invoked with ``list[dict]`` on change.
    """

    selection_changed = Signal(object)  # emits list[dict]

    def __init__(
        self,
        runs: list[dict],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(make_section_label("Saved runs"))

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self._list.itemSelectionChanged.connect(self._emit_selection)
        layout.addWidget(self._list)

        self._empty_lbl = QLabel("No runs saved yet.")
        self._empty_lbl.setStyleSheet("color: gray;")
        layout.addWidget(self._empty_lbl)

        self._runs: list[dict] = []
        self.refresh(runs)

    def _emit_selection(self) -> None:
        self.selection_changed.emit(self.get_selected())

    def get_selected(self) -> list[dict]:
        """Return currently selected run-metadata dicts."""
        selected = []
        for item in self._list.selectedItems():
            idx = self._list.row(item)
            if 0 <= idx < len(self._runs):
                selected.append(self._runs[idx])
        return selected

    def refresh(self, runs: list[dict]) -> None:
        """Repopulate with *runs* and pre-select the most recent 1-3."""
        self._runs = runs
        self._list.clear()
        if not runs:
            self._list.hide()
            self._empty_lbl.show()
            return
        self._empty_lbl.hide()
        self._list.show()
        for run in runs:
            label = run.get("display_name") or run.get("run_id", str(run))
            self._list.addItem(QListWidgetItem(str(label)))
        n = len(runs)
        for i in range(max(0, n - 3), n):
            self._list.item(i).setSelected(True)

    def enforce_max_selection(self, max_selected: int) -> None:
        """Keep only the most recent selected rows when selection exceeds *max_selected*."""
        selected_rows = sorted(self._list.row(item) for item in self._list.selectedItems())
        if len(selected_rows) <= max_selected:
            return
        keep = set(selected_rows[-max_selected:])
        self._list.blockSignals(True)
        for row in selected_rows:
            if row not in keep:
                item = self._list.item(row)
                if item is not None:
                    item.setSelected(False)
        self._list.blockSignals(False)
        self._emit_selection()


# ---------------------------------------------------------------------------
# Background worker thread
# ---------------------------------------------------------------------------


class PhaseWorker(QThread):
    """A ``QThread`` that executes a callable and emits progress signals.

    :param work_fn: ``(append_fn) -> dict`` — receives a callable to emit log
        lines; returns a diagnostics dict.

    Signals
    -------
    line_ready(str)
        Emitted for each log line the worker wishes to display.
    finished(dict)
        Emitted with the diagnostics dict when work is complete.
    error(str)
        Emitted with an error message if an exception is raised.
    """

    line_ready = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        work_fn: Callable[[Callable[[str], None]], dict],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._work_fn = work_fn

    def run(self) -> None:
        try:
            result = self._work_fn(self.line_ready.emit)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))
            self.finished.emit({"error": str(exc)})


# ---------------------------------------------------------------------------
# Shared image-folder validation helpers
# ---------------------------------------------------------------------------

IMAGE_FOLDER_SCHEMATIC = (
    "Expected image folder layout:\n\n"
    "<image_folder>/\n"
    "  cam0/\n"
    "    img_0001.png\n"
    "    img_0002.png\n"
    "  cam1/\n"
    "    img_0001.png\n"
    "    img_0002.png\n\n"
    "Also valid (extra non-camera artefacts allowed at root):\n\n"
    "<image_folder>/\n"
    "  cam0/\n"
    "  cam1/\n"
    "  detected_datapoints.pickle\n"
    "  session.camset\n"
    "  .pycamset_workspace/\n\n"
    "Ignored as camera folders:\n"
    "  - folder named 'sparse'\n"
    "  - any folder starting with '.'\n"
    "  - any individual file"
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_camera_subfolders(root: Path) -> list[Path]:
    """Return valid camera subfolders (ignores sparse, dot-folders, and files)."""
    root = Path(root)
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        [
            p for p in root.iterdir()
            if p.is_dir() and p.name != "sparse" and not p.name.startswith(".")
        ],
        key=lambda p: p.name.lower(),
    )


def count_images_in_folder(folder: Path) -> int:
    """Count image files directly inside *folder*."""
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)


def resolve_phase1_pickle_artifact(phase1_run: dict, ws_path: Path) -> Optional[Path]:
    """Resolve the ``detected_datapoints.pickle`` path for a Phase 1 run.

    Resolution order
    ----------------
    1. The path stored in ``artifacts["detected_datapoints_pickle"]``.
    2. ``<workspace>/phase1_runs/<run_id>/detected_datapoints.pickle``.

    Raises
    ------
    RuntimeError
        If neither location yields an existing file.  The caller should prompt
        the user to re-run Phase 1 rather than falling back to a legacy f_loc
        path (which would silently use stale data).
    """
    artifacts = phase1_run.get("artifacts") or {}
    artifact_path = artifacts.get("detected_datapoints_pickle")
    if artifact_path:
        p = Path(artifact_path)
        if p.exists():
            return p

    run_id = phase1_run.get("run_id")
    if run_id:
        p = ws_path / "phase1_runs" / str(run_id) / "detected_datapoints.pickle"
        if p.exists():
            return p

    return None


class EmitStream(io.TextIOBase):
    """Redirect stream writes to a line-emitting callback."""

    def __init__(self, emit: Callable[[str], None]):
        super().__init__()
        self._emit = emit
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._emit(line)
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            self._emit(self._buf.strip())
        self._buf = ""


class EmitLogHandler(logging.Handler):
    """Forward Python logging records to a line-emitting callback."""

    def __init__(self, emit: Callable[[str], None]):
        super().__init__(level=logging.INFO)
        self._emit = emit

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if msg.strip():
                self._emit(msg)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Matplotlib thread-safety helper
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def suppress_matplotlib_gui():
    """Context manager that forces Matplotlib into non-interactive ``Agg`` mode.

    Use this inside ``PhaseWorker.run()`` to prevent Matplotlib from opening
    GUI windows on a background thread, which would crash Qt.

    On exit the original ``plt.show`` callable is restored (best-effort).
    """
    plt = None
    orig_show = None
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as _plt
        plt = _plt
        orig_show = plt.show
        plt.show = lambda *args, **kwargs: None
    except Exception:
        pass
    try:
        yield
    finally:
        if plt is not None and orig_show is not None:
            try:
                plt.show = orig_show
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Predecessor-chain UI helper
# ---------------------------------------------------------------------------

def render_predecessor_chain_section(layout, workspace_mgr: "WorkspaceManager", run: dict) -> None:
    """Append a labelled *Upstream Run Chain* section to *layout*.

    For each predecessor (oldest-first) a compact read-only form is added
    showing the phase, run-id, key params, and a selection of diagnostics.
    All data is a deep copy — no destructive references.

    The section is omitted entirely when the run has no trackable predecessors.
    """
    chain = workspace_mgr.build_predecessor_chain(run)
    if not chain:
        return

    layout.addWidget(make_separator())
    hdr = make_section_label("Upstream Run Chain")
    hdr.setToolTip(
        "Metadata from all predecessor phase runs that led to this result.\n"
        "Oldest phase first.  All values are read-only copies."
    )
    layout.addWidget(hdr)

    for pred in chain:
        phase = str(pred.get("phase", "unknown"))
        rid = str(pred.get("run_id", "unknown"))
        pred_hdr = QLabel(f"  {phase}  |  {rid}")
        pred_hdr.setStyleSheet("font-weight: bold; margin-top: 4px; color: #555;")
        layout.addWidget(pred_hdr)

        form = QFormLayout()
        form.setContentsMargins(32, 0, 0, 0)

        params = pred.get("params") or {}
        interesting_params = {
            k: v for k, v in params.items()
            if k not in ("f_loc",) and v is not None
        }
        if interesting_params:
            param_txt = ", ".join(
                f"{k}={v}" for k, v in list(interesting_params.items())[:6]
            )
            plbl = QLabel(param_txt)
            plbl.setWordWrap(True)
            plbl.setStyleSheet("color: #444; font-size: 9pt;")
            form.addRow("params:", plbl)

        diag = pred.get("diagnostics") or {}
        for key, val in list(diag.items())[:4]:
            short_key = key.split("_", 1)[-1] if "_" in key else key
            dlbl = QLabel(str(val)[:120])
            dlbl.setWordWrap(True)
            dlbl.setStyleSheet("font-size: 9pt;")
            form.addRow(f"{short_key}:", dlbl)

        if pred.get("error"):
            err_lbl = QLabel(str(pred["error"])[:200])
            err_lbl.setStyleSheet("color: red; font-size: 9pt;")
            err_lbl.setWordWrap(True)
            form.addRow("error:", err_lbl)

        layout.addLayout(form)
