"""
Shared widget factories, mixins, and constants for the pyCamSet GUI.

All UI is built with PySide6 (Qt6).  What a phase actually does lives in
:mod:`pyCamSet.workflow`, which knows nothing about Qt; this module is the
presentation half.

Conventions
-----------
- Hover tooltips are gated by a shared ``QCheckBox`` (``info_cb``) via its
  ``isChecked()`` state -- set as ``setToolTipsEnabled(bool)`` on every widget.
- The terminal pane is a dark, read-only ``QTextEdit`` written to via
  :meth:`TerminalWidget.append_line`.
- Background work runs inside :class:`PhaseWorker`, a ``QThread`` subclass
  that emits ``line_ready(str)`` and ``finished(dict)`` signals.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Optional

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
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
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from pyCamSet.workflow.params import ParamError, as_positive_float
from pyCamSet.workflow.run_quality import blocking_reasons
from pyCamSet.workflow.workspace import WorkspaceManager

#: What a dictionary combo falls back to when the backend it is being
#: repopulated for does not offer the dictionary that was selected.
_DICT_FALLBACK_NAME = "DICT_4X4_1000"

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
TAB_EXPORT_CALIBRATION = "Export Calibration"


# ---------------------------------------------------------------------------
# Marker dictionary combo
# ---------------------------------------------------------------------------


def repopulate_dict_combo(combo, marker_backend: str) -> None:
    """Repopulate a dictionary combo for the given backend (plan v4 R2-H6/H7).

    The caller must set its ``self._repopulating`` guard around the call so
    connected slots (e.g. ``_sync_default_name``) early-return during the
    repopulation. When the currently selected dictionary name is absent from
    the new list, the combo falls back to ``DICT_4X4_1000``. Signals are
    blocked for the whole repopulation and restored in ``finally``.
    """
    from pyCamSet.calibration_targets.backend_registry import dict_names_for_backend

    current_name = combo.currentText()
    combo.blockSignals(True)
    try:
        combo.clear()
        combo.addItems(dict_names_for_backend(marker_backend))
        if combo.findText(current_name) >= 0:
            combo.setCurrentText(current_name)
        else:
            combo.setCurrentText(_DICT_FALLBACK_NAME)
    finally:
        combo.blockSignals(False)


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

ORANGE = "#e07b00"
DARK_ORANGE = "#c06000"
GREEN = "#2e7d32"
DARK_GREEN = "#1b5e20"
DULL_RED = "#8c3b3b"
DARK_DULL_RED = "#733030"
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

#: A continue button whose run produced nothing worth carrying forward. Dull
#: rather than bright, because it is a warning against going on rather than
#: an action of its own.
BLOCKED_BTN_STYLE = (
    f"QPushButton {{ background-color: {DULL_RED}; color: white; font-weight: bold;"
    f" border-radius: 4px; padding: 4px 10px; }}"
    f"QPushButton:hover {{ background-color: {DARK_DULL_RED}; }}"
    f"QPushButton:pressed {{ background-color: {DARK_DULL_RED}; }}"
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


def make_continue_button(callback: Callable,
                         text: str = "Continue to Next Phase ▶") -> QPushButton:
    """Return a green button that hands the phase's work on.

    The button carries the reasons the run behind it should not be carried
    forward: :func:`set_continue_blocked` puts them there and colours it, and
    while it is red a click explains itself and asks first.

    :param callback: what to call once the click is allowed through
    :param text: the button's label
    """
    btn = QPushButton(text)
    set_continue_blocked(btn, [])
    btn.clicked.connect(lambda: _continue_clicked(btn, callback))
    return btn


def set_continue_blocked(btn: QPushButton, reasons: list[str]) -> None:
    """Colour a continue button by whether its run is worth carrying on.

    The button stays enabled: a run being unusable is a strong statement, and
    the person holding the images may know something this does not, so it
    says so and asks rather than locking them out.

    :param btn: the button, as :func:`make_continue_button` built it
    :param reasons: why the run cannot be carried forward, empty when it can
    """
    btn._blocked_reasons = list(reasons)
    if reasons:
        if not hasattr(btn, "_unblocked_tooltip"):
            btn._unblocked_tooltip = btn.toolTip()
        btn.setStyleSheet(BLOCKED_BTN_STYLE)
        btn.setToolTip("This run cannot be carried forward:\n\n"
                       + "\n\n".join(reasons))
        return
    btn.setStyleSheet(GREEN_BTN_STYLE)
    btn.setToolTip(getattr(btn, "_unblocked_tooltip", btn.toolTip()))


def gate_continue_button(btn: QPushButton, terminal, metadata: dict) -> None:
    """Judge a finished run, and say so on the button and in the terminal.

    :param btn: the phase's continue button
    :param terminal: the phase's terminal pane
    :param metadata: the run record the phase just produced
    """
    reasons = blocking_reasons(metadata)
    set_continue_blocked(btn, reasons)
    for reason in reasons:
        terminal.append_line(f"Cannot continue: {reason}")


def _continue_clicked(btn: QPushButton, callback: Callable) -> None:
    """Ask before carrying a run that said it should not be carried."""
    reasons = getattr(btn, "_blocked_reasons", [])
    if reasons:
        answer = QMessageBox.warning(
            btn.window(), "This run is not worth continuing from",
            "\n\n".join(reasons) + "\n\nContinue anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
    callback()


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




def build_charuco_option_tooltip(meta: dict[str, Any]) -> str:
    """Build canonical tooltip/info text from a ChArUco parameter row."""
    from pyCamSet.calibration_targets.charuco_parameters import label_for

    return (
        f"{meta['concept']}\n\n"
        f"Default: {label_for(meta, meta['default'])}\n"
        f"Range: {meta['range']}\n"
        f"Range source: {meta['range_source']}\n"
        f"Suggested value(s): {meta['suggested']}"
    )






#: Which of the form's controls each target takes its arguments from.  The
#: form is one set of widgets serving four targets, showing and hiding as the
#: type changes, so this is where that flat set becomes each target's own
#: constructor arguments.  It is the only mapping of its kind left: every
#: other place a target is described, it is described as a spec.
_TARGET_WIDGETS: dict[str, dict[str, tuple[str, type]]] = {
    "Ccube": {
        "n_points": ("_npts_spin", int),
        "length": ("_length_edit", float),
        "border_fraction": ("_border_spin", float),
    },
    "ChArUco": {
        "num_squares_x": ("_npts_spin", int),
        "num_squares_y": ("_npts_spin", int),
        "square_size": ("_length_edit", float),
        "marker_fraction": ("_marker_spin", float),
    },
    "PuzzleBoard": {
        "num_squares_x": ("_pb_x_spin", int),
        "num_squares_y": ("_pb_y_spin", int),
        "square_size": ("_pb_square_edit", float),
        "start_x": ("_pb_start_x_spin", int),
        "start_y": ("_pb_start_y_spin", int),
        "paper_width": ("_pb_paper_w_edit", float),
        "paper_height": ("_pb_paper_h_edit", float),
        "min_width": ("_pb_min_width_spin", int),
    },
    "PuzzleBoardCube": {
        "n_points": ("_pbc_size_spin", int),
        "length": ("_pbc_square_edit", float),
        "min_width": ("_pb_min_width_spin", int),
    },
}

#: The targets that read markers, and so take a backend from the form.
_MARKER_TARGETS = {"Ccube", "ChArUco"}


def _widget_value(tab: Any, name: str, cast: type, label: str):
    """Read one control, as the value its target's argument takes."""
    widget = getattr(tab, name, None)
    if widget is None:
        return None
    if hasattr(widget, "value"):
        return cast(widget.value())
    text = widget.text().strip()
    return as_positive_float(text, label) if cast is float else cast(text)


def read_target_spec(tab: Any, detection_options: dict | None = None) -> dict:
    """
    The target spec a phase tab's controls describe.

    :param tab: the phase tab, holding the target widgets
    :param detection_options: the detector tuning, when the tab collects it
    :raises ParamError: for a control whose value the target cannot take
    """
    target_type = tab._target_combo.currentText()
    spec: dict[str, Any] = {"type": target_type}
    for argument, (widget_name, cast) in _TARGET_WIDGETS[target_type].items():
        value = _widget_value(tab, widget_name, cast, argument)
        if value is not None:
            spec[argument] = value
    if target_type in _MARKER_TARGETS:
        spec["marker_backend"] = str(
            tab._marker_backend_combo.currentData() or "aruco1")
    if detection_options is not None:
        spec["detection_options"] = detection_options
    return spec


def apply_target_spec_to_widgets(tab: Any, spec: dict) -> None:
    """
    Set a phase tab's target controls from a saved run's target.

    Phases 2 and 3 build their own target and pair it with detections made
    by an earlier run, so the default that is right almost always is the one
    the detections were made with.

    Spin boxes clamp to their own ranges, so a value the interface cannot
    represent is silently narrowed here; :func:`describe_target_mismatch`
    is what catches that before it reaches the solver.

    :param tab: the phase tab, holding the target widgets
    :param spec: the target spec to adopt
    """
    if not spec or "type" not in spec:
        return
    combo = getattr(tab, "_target_combo", None)
    if combo is not None:
        combo.setCurrentText(str(spec["type"]))

    for argument, (widget_name, cast) in _TARGET_WIDGETS.get(spec["type"], {}).items():
        if spec.get(argument) is None:
            continue
        widget = getattr(tab, widget_name, None)
        if widget is None:
            continue
        if hasattr(widget, "setValue"):
            widget.setValue(cast(spec[argument]))
        else:
            widget.setText(f"{float(spec[argument]):g}")

    backend = spec.get("marker_backend")
    backend_combo = getattr(tab, "_marker_backend_combo", None)
    if backend_combo is not None and backend:
        index = backend_combo.findData(str(backend))
        if index >= 0:
            backend_combo.setCurrentIndex(index)


# ---------------------------------------------------------------------------
# Terminal widget
# ---------------------------------------------------------------------------


_ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

#: The colour escapes, which are painted rather than dropped.
_SGR_RE = re.compile(r"\x1B\[([0-9;]*)m")

# The xterm-256 palette the report blocks are written against.  0-15 are the
# terminal's own colours, which have no fixed values and are taken here from
# the xterm defaults; 16-231 are a 6x6x6 cube and 232-255 a grey ramp, both
# of which are defined by their formula.
_XTERM_BASIC = (
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
    (0, 0, 128), (128, 0, 128), (0, 128, 128), (192, 192, 192),
    (128, 128, 128), (255, 0, 0), (0, 255, 0), (255, 255, 0),
    (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
)
_XTERM_CUBE = (0, 95, 135, 175, 215, 255)


def xterm_colour(index: int) -> QColor:
    """
    The colour an xterm-256 palette index stands for.

    :param index: the palette index, 0 to 255
    """
    if index < 16:
        return QColor(*_XTERM_BASIC[index])
    if index < 232:
        index -= 16
        return QColor(_XTERM_CUBE[index // 36],
                      _XTERM_CUBE[(index // 6) % 6],
                      _XTERM_CUBE[index % 6])
    grey = 8 + 10 * (index - 232)
    return QColor(grey, grey, grey)


def _drop_movement(text: str) -> str:
    """
    Remove every escape that is not a colour.

    :param text: the text as it was written
    """
    return _ANSI_RE.sub(
        lambda escape: escape.group(0) if escape.group(0).endswith("m") else "",
        text)


def _sgr_format(params: str, current: QTextCharFormat) -> QTextCharFormat:
    """
    The character format an SGR escape asks for.

    Only what the report blocks emit is acted on -- a reset, bold, and an
    xterm-256 foreground -- and anything else leaves the format as it was.

    :param params: the escape's parameters, between the '[' and the 'm'
    :param current: the format in force before the escape
    """
    fmt = QTextCharFormat(current)
    codes = [int(code) for code in params.split(";") if code] or [0]
    index = 0
    while index < len(codes):
        code = codes[index]
        if code == 0:
            fmt = QTextCharFormat()
        elif code == 1:
            fmt.setFontWeight(QFont.Weight.Bold)
        elif (code == 38 and codes[index + 1:index + 2] == [5]
                and index + 2 < len(codes)):
            fmt.setForeground(xterm_colour(codes[index + 2]))
            index += 2
        index += 1
    return fmt


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
        """
        Append *text* + newline in the colours it asks for, and scroll down.

        The report blocks grade their numbers with xterm-256 escapes -- see
        :mod:`pyCamSet.utils.report_format` -- so those are painted here
        rather than thrown away.  Escapes that are not colours, the cursor
        moves a progress bar makes, are dropped: this pane only appends.

        :param text: the line, with or without escapes in it
        """
        body = _drop_movement(str(text).replace("\r", ""))
        self.moveCursor(QTextCursor.MoveOperation.End)
        cursor = self.textCursor()
        fmt = QTextCharFormat()
        written = 0
        for escape in _SGR_RE.finditer(body):
            cursor.insertText(body[written:escape.start()], fmt)
            fmt = _sgr_format(escape.group(1), fmt)
            written = escape.end()
        cursor.insertText(body[written:] + "\n", fmt)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def clear_terminal(self) -> None:
        """Erase all terminal content."""
        self.clear()

    def _on_toggle(self) -> None:
        self.setVisible(self._show_cb.isChecked())


# ---------------------------------------------------------------------------
# Tab navigation
# ---------------------------------------------------------------------------


def show_tab(notebook, widget) -> None:
    """
    Make a widget's tab the current one, showing it in the bar first.

    The diagnostics tabs are hidden in the tab bar to keep it short, and
    were then made current anyway.  Qt never leaves a hidden tab current
    by itself -- ``QTabBar::setTabVisible`` moves the current tab along
    when it hides one -- so that combination is a state Qt does not
    expect: the tab bar paints a current tab that has no geometry, and on
    macOS the native style dereferences a null context doing it.

    Every crash report from the GUI landed in
    ``QMacCGContext::QMacCGContext`` under ``QTabBar::paintEvent``, with
    the repaint driven by whatever happened to come next -- a window
    opening, the application losing focus.

    :param notebook: the ``QTabWidget`` holding the tab
    :param widget: the page to switch to
    """
    index = notebook.indexOf(widget)
    if index < 0:
        return
    bar = notebook.tabBar()
    if not bar.isTabVisible(index):
        bar.setTabVisible(index, True)
    notebook.setCurrentWidget(widget)




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


# ---------------------------------------------------------------------------
# Predecessor-chain UI helper
# ---------------------------------------------------------------------------

def render_predecessor_chain_section(layout, workspace_mgr: WorkspaceManager, run: dict) -> None:
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
