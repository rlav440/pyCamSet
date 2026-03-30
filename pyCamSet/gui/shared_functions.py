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
"""
from __future__ import annotations

import io
import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
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

# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

ORANGE = "#e07b00"
DARK_ORANGE = "#c06000"
GREEN = "#2e7d32"
DARK_GREEN = "#1b5e20"
SECTION_COLOR = "#1976d2"

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


def make_continue_button(callback: Callable) -> QPushButton:
    """Return a green "Continue to Next Phase" button."""
    btn = QPushButton("Continue to Next Phase ▶")
    btn.setStyleSheet(GREEN_BTN_STYLE)
    btn.clicked.connect(callback)
    return btn


def make_run_id() -> str:
    """Return a unique, timestamp-ordered run identifier.

    Format: ``YYYYMMDD_HHMMSS_<6-char hex>``
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"


def build_target(target_type: str, n_points: int, length: float):
    """Construct a calibration target from canonical GUI options."""
    from pyCamSet.calibration_targets.target_Ccube import Ccube
    from pyCamSet.calibration_targets.target_charuco import ChArUco

    if target_type == "Ccube":
        return Ccube(n_points=n_points, length=length)
    if target_type == "ChArUco":
        return ChArUco(num_squares_x=n_points, num_squares_y=n_points, square_size=length)
    raise ValueError(f"Unknown target type: {target_type!r}")


def extract_detection_and_cam_res(payload):
    """Extract (TargetDetection, cam_res) from common persisted payload shapes."""
    if hasattr(payload, "get_cam_list"):
        return payload, None

    if isinstance(payload, (tuple, list)) and len(payload) >= 1:
        detections = payload[0]
        cam_res = payload[1] if len(payload) > 1 else None
        if hasattr(detections, "get_cam_list"):
            return detections, cam_res

    if isinstance(payload, dict):
        for key in ("detections", "target_detection", "target_detections", "data"):
            det = payload.get(key)
            if hasattr(det, "get_cam_list"):
                return det, payload.get("cam_res")

    return None, None


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
    """Manages workspace directory and run metadata."""

    _PHASE_ALIASES = {
        "phase5": "assess_calibration",
        "phase_5": "assess_calibration",
        "phase-5": "assess_calibration",
        "phase 5": "assess_calibration",
        "visualise_target": "assess_calibration",
    }

    def __init__(self, workspace_path: Optional[Path] = None) -> None:
        self.workspace_path: Optional[Path] = Path(workspace_path) if workspace_path else None
        if self.workspace_path is not None:
            self.ensure_dirs()

    @classmethod
    def canonical_phase_name(cls, phase: str) -> str:
        p = (phase or "").strip().lower()
        return cls._PHASE_ALIASES.get(p, p)

    @classmethod
    def _phase_run_dirs(cls, phase: str) -> list[str]:
        canonical = cls.canonical_phase_name(phase)
        if canonical == "assess_calibration":
            # Keep reading legacy folders created by earlier builds.
            return ["assess_calibration_runs", "visualise_target_runs", "phase5_runs", "phase_5_runs"]
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
            "assess_calibration_runs",
            "visualise_target_runs",
            "phase5_runs",
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
    """Resolve detected_datapoints pickle robustly for older/newer run metadata."""
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

    f_loc = (phase1_run.get("params") or {}).get("f_loc")
    if f_loc:
        p = Path(f_loc) / "detected_datapoints.pickle"
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

