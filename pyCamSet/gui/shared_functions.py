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
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from pyCamSet.calibration_targets.markers.backend_registry import (
    MARKER_BACKEND_LABELS,
    marker_backend_availability_text,
    marker_backend_available,
)
from pyCamSet.workflow.params import ParamError
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
    from pyCamSet.calibration_targets.markers.backend_registry import dict_names_for_backend

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

    def clear(self) -> None:
        """Remove every row, so the section can be rebuilt for a new subject."""
        while self._form.rowCount():
            self._form.removeRow(0)

    def set_title(self, title: str) -> None:
        """Rename the section, keeping it expanded or collapsed as it is."""
        self._title = title
        self._update_label(self._btn.isChecked())




def detector_parameterisation_for(target_type: str, backend: str | None):
    """
    The detector a form's target selection names.

    One detector combo serves every target on a form, so only a target with
    a choice of detector takes its selection from it; the rest are read with
    the one they have.

    :param target_type: the selected target, as :data:`TARGET_NAMES` keys it
    :param backend: what the form's detector combo holds
    :raises ValueError: for an unknown target, or a detector it cannot use
    """
    from pyCamSet.calibration_targets.core.target_registry import target_class

    cls = target_class(target_type)
    if len(cls.DETECTOR_BACKENDS) < 2:
        backend = None
    return cls.detector_parameterisation(backend or None)


def build_parameter_tooltip(meta) -> str:
    """What one parameter says about itself, as hover text."""
    lines = [f"Concept: {meta.concept}", ""] if meta.concept else []
    lines.append(f"Default: {meta.label_for(meta.default)}")
    if meta.range_text:
        lines.append(f"Range: {meta.range_text}")
    elif meta.minimum is not None and meta.maximum is not None:
        lines.append(f"Range: {meta.minimum} to {meta.maximum}")
    if meta.range_source:
        lines.append(f"Range source: {meta.range_source}")
    if meta.suggested:
        lines.append(f"Suggested value(s): {meta.suggested}")
    return "\n".join(lines)


def build_parameter_widget(meta) -> QWidget:
    """
    The control one parameter is typed into.

    Named choices get a combo, an on/off gets a check box, a number with
    bounds gets a spin box that holds it inside them, and everything else
    -- a matrix, a vector, a number with no bounds -- gets a text field
    whose placeholder says the shape of the value.
    """
    if meta.choices:
        widget = QComboBox()
        widget.addItems(meta.choice_labels())
        widget.setCurrentText(meta.label_for(meta.default))
    elif meta.dtype == "bool":
        widget = QCheckBox()
        widget.setChecked(bool(meta.default))
    elif meta.dtype in ("int", "float") and meta.minimum is not None \
            and meta.maximum is not None:
        if meta.dtype == "int":
            widget = QSpinBox()
            widget.setSingleStep(int(meta.step or 1) or 1)
        else:
            widget = QDoubleSpinBox()
            widget.setDecimals(int(meta.decimals or 3))
            widget.setSingleStep(float(meta.step or 0.01) or 0.01)
        widget.setRange(meta.cast(meta.minimum), meta.cast(meta.maximum))
        widget.setValue(meta.cast(meta.default))
    else:
        widget = QLineEdit("" if meta.default == "" else str(meta.default))
        if meta.dtype == "json_matrix_3x3":
            widget.setPlaceholderText('e.g. [[fx,0,cx],[0,fy,cy],[0,0,1]] or blank')
        elif meta.dtype == "json_vector":
            widget.setPlaceholderText("e.g. [k1,k2,p1,p2,k3] or blank")
    widget.setFixedWidth(220)
    widget.setToolTip(build_parameter_tooltip(meta))
    return widget


def read_parameter_widget(widget):
    """What was typed into one of :func:`build_parameter_widget`'s controls."""
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        return widget.value()
    return widget.text().strip()


def set_parameter_widget(widget, value) -> None:
    """Show *value* in one of :func:`build_parameter_widget`'s controls."""
    if isinstance(widget, QComboBox):
        text = str(value)
        if widget.findText(text) < 0:
            # A saved spec can name a value this combo's current choices no
            # longer offer (e.g. a dictionary retired from the dropdown).
            # Add it rather than silently keeping whatever was already
            # selected -- the displayed value must always match the spec.
            widget.addItem(text)
        widget.setCurrentText(text)
    elif isinstance(widget, QCheckBox):
        widget.setChecked(bool(value))
    elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        widget.setValue(type(widget.value())(value))
    else:
        widget.setText("" if value is None else str(value))


def connect_value_changed(widget, slot: Callable[[], None]) -> None:
    """Wire *slot* to whichever signal means "the typed value moved".

    One dispatch for every control :func:`build_parameter_widget` can
    produce, so a caller that wants to hear about edits -- not just
    structural rebuilds -- does not have to know the widget classes itself.

    Every one of these Qt signals carries the new value as an argument
    (``str``, ``int``, ``bool``...); *slot* takes none, so it is wrapped
    rather than connected directly -- a direct connection does not raise
    here, it just never calls *slot* at all, and fails silently.
    """
    handler = lambda *_args: slot()
    if isinstance(widget, QComboBox):
        widget.currentIndexChanged.connect(handler)
    elif isinstance(widget, QCheckBox):
        widget.toggled.connect(handler)
    elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        widget.valueChanged.connect(handler)
    else:
        widget.textChanged.connect(handler)


class TargetSettingsForm(QWidget):
    """
    The controls a calibration target is described with.

    A target combo, a detector combo for the targets that have a choice of
    one, and a control for each argument the selected target declares --
    rebuilt when either changes.  Which is the point: the target's own
    arguments were a map of widget names, one copy per phase, so a target
    none of them named had no form at all.

    ``marker_backend`` is written here and in
    :func:`~pyCamSet.workflow.targets.detector_parameterisation_of`, and
    nowhere else outside the two targets that take one: it is what those
    constructors call the detector they are read with.
    """

    changed = Signal()

    #: A row's typed value moved -- a keystroke or a spin, not a rebuild.
    #: Kept separate from :attr:`changed` because that signal already means
    #: something to other tabs (rebuild the detection-options section,
    #: rebuild the sweep rows), and firing it on every keystroke would make
    #: them redo that work continuously rather than once per structural
    #: change.
    values_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None,
                 targets: Optional[list[str]] = None) -> None:
        super().__init__(parent)
        from pyCamSet.calibration_targets.core.target_registry import TARGET_NAMES

        self._widgets: dict[str, QWidget] = {}
        #: Raw widget values captured at the top of :meth:`_rebuild`, keyed
        #: by parameter name -- but only for a key in :attr:`_edited`. Lets
        #: a value someone actually typed survive a round trip through a
        #: target that does not have it -- flip away and back and it is
        #: still there -- without leaking into a saved spec
        #: :meth:`apply_spec` loads, which clears this outright. A value
        #: never touched is a default, and carrying a default across a
        #: flip is how one target's defaults used to overwrite another's
        #: same-named field (Ccube and PuzzleBoardCube both have
        #: ``n_points``/``length``; browsing between them silently swapped
        #: in Ccube's numbers as PuzzleBoardCube's own).
        self._retained: dict[str, Any] = {}
        #: Keys a person (or a caller writing straight into a widget) has
        #: actually moved, as opposed to a value only ever shown because
        #: it was built as a default. Populated by
        #: :meth:`_on_widget_value_changed`; never touched by
        #: :meth:`apply_spec`'s own writes, which happen while
        #: :attr:`_applying_spec` is set.
        self._edited: set[str] = set()
        #: True for the duration of :meth:`apply_spec`, so the writes it
        #: makes into widgets are never mistaken for a person's edit, and
        #: so :meth:`_rebuild` -- which it can trigger synchronously via
        #: ``setCurrentText``/``setCurrentIndex`` -- captures nothing into
        #: :attr:`_retained` from whatever was on the form a moment ago.
        self._applying_spec = False
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._target_combo = QComboBox()
        self._target_combo.addItems(list(targets or TARGET_NAMES))
        self._target_combo.setToolTip(
            "The calibration target these settings describe.")
        form.addRow("Target type:", self._target_combo)

        self._backend_label = QLabel("Detector:")
        self._backend_combo = QComboBox()
        for label, value in MARKER_BACKEND_LABELS.items():
            self._backend_combo.addItem(label, value)
        self._backend_combo.setToolTip(
            "Which library reads this target's markers.")
        form.addRow(self._backend_label, self._backend_combo)
        self._backend_status = QLabel()
        self._backend_status.setStyleSheet("font-size: 10px;")
        form.addRow("", self._backend_status)

        self._rows = QWidget()
        self._rows_form = QFormLayout(self._rows)
        self._rows_form.setContentsMargins(0, 0, 0, 0)
        self._rows_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.addRow(self._rows)

        self._target_combo.currentTextChanged.connect(self._rebuild)
        self._backend_combo.currentIndexChanged.connect(self._rebuild)
        self._rebuild()

    # -- what is selected ------------------------------------------------

    def target_type(self) -> str:
        """The selected target's name."""
        return self._target_combo.currentText()

    def backend(self) -> str:
        """The selected detector, for a target that is offered a choice."""
        return str(self._backend_combo.currentData() or "aruco1")

    def detector_parameterisation(self):
        """What the selected target's detection can be told."""
        return detector_parameterisation_for(self.target_type(), self.backend())

    def construction_parameters(self):
        """What the selected target says it is described by."""
        from pyCamSet.calibration_targets.core.target_registry import target_class

        return target_class(self.target_type()).construction_parameters(
            self.backend() if self._backend_offered() else None)

    def _backend_offered(self) -> bool:
        """Whether this target has a choice of detector to be asked about."""
        from pyCamSet.calibration_targets.core.target_registry import target_class

        return len(target_class(self.target_type()).DETECTOR_BACKENDS) > 1

    # -- the form -------------------------------------------------------

    def _rebuild(self, *_args) -> None:
        """Offer the arguments the selected target says it takes."""
        # Remember what was typed before the rows that hold it are torn
        # down -- but only for a key that was actually edited. Merged
        # rather than replaced, so a key from a target flipped away from
        # two rebuilds ago is still here.
        for key in self._edited:
            widget = self._widgets.get(key)
            if widget is None:
                continue
            try:
                self._retained[key] = read_parameter_widget(widget)
            except Exception:
                pass

        offered = self._backend_offered()
        self._backend_label.setVisible(offered)
        self._backend_combo.setVisible(offered)
        self._backend_status.setVisible(offered)
        if offered:
            backend = self.backend()
            self._backend_status.setText(marker_backend_availability_text(backend))
            self._backend_status.setStyleSheet(
                "font-size: 10px; color: "
                + ("#2a7a2a;" if marker_backend_available(backend) else "#8a4a00;"))

        self._widgets = {}
        while self._rows_form.rowCount():
            self._rows_form.removeRow(0)
        for parameter in self.construction_parameters().settable():
            widget = build_parameter_widget(parameter)
            if parameter.key in self._edited and parameter.key in self._retained:
                # A retained value can be wrong for this target -- out of a
                # spin box's range, a dictionary this backend does not
                # offer -- and restoring it must never be the reason a
                # value is lost outright. Fall back to the freshly built
                # default for this key alone.
                try:
                    set_parameter_widget(widget, self._retained[parameter.key])
                except Exception:
                    pass
            # Connected after any restore above, so putting a retained
            # value back does not itself count as a further edit.
            connect_value_changed(
                widget, lambda key=parameter.key: self._on_widget_value_changed(key))
            self._rows_form.addRow(f"{parameter.label}:", widget)
            self._widgets[parameter.key] = widget
        self.changed.emit()

    def _on_widget_value_changed(self, key: str) -> None:
        """A row's typed value moved.

        Recorded into :attr:`_edited` unless it happened while
        :meth:`apply_spec` is writing a loaded spec into place -- that is
        the call overwriting the form on purpose, not a person editing it,
        and must never be treated as one, or the very next flip would
        carry a value :meth:`apply_spec` just tried to erase.
        """
        if not self._applying_spec:
            self._edited.add(key)
        self.values_changed.emit()

    # -- reading and writing a spec --------------------------------------

    def spec(self, detection_options: dict | None = None) -> dict:
        """
        The target spec these controls describe.

        :param detection_options: the detector tuning, when the form
            collecting this also collects that
        :raises pyCamSet.workflow.ParamError: for a value the target cannot take
        """
        from pyCamSet.calibration_targets.core.target_registry import TYPE_KEY

        parameters = self.construction_parameters()
        try:
            values = parameters.parse(
                {key: read_parameter_widget(widget)
                 for key, widget in self._widgets.items()})
        except ValueError as exc:
            raise ParamError(str(exc)) from None

        spec: dict[str, Any] = {TYPE_KEY: self.target_type(), **values}
        if self._backend_offered():
            spec["marker_backend"] = self.backend()
        if detection_options is not None:
            spec["detection_options"] = detection_options
        return spec

    def apply_spec(self, spec: dict) -> None:
        """
        Set these controls from a saved run's target.

        Phases 2 and 3 build their own target and pair it with detections
        made by an earlier run, so the default that is right almost always
        is the one the detections were made with.

        A loaded spec wins outright: the form ends up as the spec, with
        every key it does not mention at the new target's own default --
        never a value left over from whatever was on the form a moment
        ago, and never a value this call's own writes are mistaken for
        someone editing.
        """
        from pyCamSet.calibration_targets.core.target_registry import TYPE_KEY

        if not spec or TYPE_KEY not in spec:
            return
        self._applying_spec = True
        try:
            # Nothing carried over from before this call may leak into a
            # key the spec does not mention.
            self._retained = {}
            self._edited = set()
            target_type = str(spec[TYPE_KEY])
            if self._target_combo.currentText() == target_type:
                # setCurrentText is a no-op when the type is already
                # selected -- Qt emits nothing and _rebuild never runs on
                # its own -- so force it, or a value already sitting in a
                # widget for a key the spec does not mention would survive
                # untouched instead of falling back to this target's
                # default.
                self._rebuild()
            else:
                # This rebuilds the rows -- with _edited already emptied
                # above, so it captures nothing from the outgoing target --
                # so it comes before the values below.
                self._target_combo.setCurrentText(target_type)
            if backend := spec.get("marker_backend"):
                if (index := self._backend_combo.findData(str(backend))) >= 0:
                    self._backend_combo.setCurrentIndex(index)

            for key, widget in self._widgets.items():
                if spec.get(key) is not None:
                    set_parameter_widget(widget, spec[key])
        finally:
            self._applying_spec = False
            # These writes must not outlive the call as if someone had
            # typed them: the very next flip should not carry a spec's
            # value into a target the spec never named.
            self._edited = set()


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

    The most recent ``min(preselect, n)`` runs are pre-selected on every
    :meth:`refresh`.

    :param on_select: Optional callback invoked with ``list[dict]`` on change.
    :param preselect: how many of the most recent runs to pre-select;
        clamped to ``[0, n]`` -- a value at or below zero pre-selects
        nothing, one past the run count selects all of them.
    """

    selection_changed = Signal(object)  # emits list[dict]

    def __init__(
        self,
        runs: list[dict],
        parent: Optional[QWidget] = None,
        preselect: int = 3,
    ) -> None:
        super().__init__(parent)
        self._preselect = preselect
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
        """Repopulate with *runs* and pre-select the most recent *preselect*."""
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
        count = max(0, min(self._preselect, n))
        for i in range(n - count, n):
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
