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

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QPalette, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
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


# ---------------------------------------------------------------------------
# Terminal widget
# ---------------------------------------------------------------------------


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
        self.moveCursor(QTextCursor.MoveOperation.End)
        self.insertPlainText(text + "\n")
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

    def __init__(self, workspace_path: Optional[Path] = None) -> None:
        self.workspace_path: Optional[Path] = Path(workspace_path) if workspace_path else None
        if self.workspace_path is not None:
            self.ensure_dirs()

    def set_workspace_path(self, workspace_path: Path, ensure: bool = True) -> None:
        """Set workspace path; optionally create standard sub-directories."""
        self.workspace_path = Path(workspace_path)
        if ensure:
            self.ensure_dirs()

    def ensure_dirs(self) -> None:
        """Create standard sub-directories if workspace path is set."""
        if self.workspace_path is None:
            return
        for sub in ("phase0_runs", "phase1_runs"):
            (self.workspace_path / sub).mkdir(parents=True, exist_ok=True)

    def save_run(self, phase: str, run_id: str, metadata: dict) -> Path:
        """Persist metadata to <workspace>/<phase>_runs/<run_id>/metadata.json."""
        if self.workspace_path is None:
            raise RuntimeError("Workspace path is not set.")
        run_dir = self.workspace_path / f"{phase}_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        meta_path = run_dir / "metadata.json"
        with open(meta_path, "w") as fh:
            json.dump(metadata, fh, indent=2, default=str)
        return meta_path

    def load_runs(self, phase: str) -> list[dict]:
        """Return all saved runs for phase sorted oldest-first."""
        if self.workspace_path is None:
            return []
        runs_dir = self.workspace_path / f"{phase}_runs"
        if not runs_dir.exists():
            return []
        results: list[dict] = []
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
        """Repopulate with *runs* and pre-select the most recent 1–3."""
        self._runs = runs
        self._list.clear()
        if not runs:
            self._list.hide()
            self._empty_lbl.show()
            return
        self._empty_lbl.hide()
        self._list.show()
        for run in runs:
            self._list.addItem(QListWidgetItem(run.get("run_id", str(run))))
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

