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
import csv
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Optional

from PySide6.QtCore import QEvent, QObject, QStandardPaths, QThread, Signal, Qt
from PySide6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QAbstractSpinBox,
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
from pyCamSet.calibration_targets.core.parameters import (
    Choice,
    Parameter,
    Parameterisation,
)
from pyCamSet.calibration_targets.markers.backend_registry import (
    ARUCO1_BACKEND,
    ARUCO2_BACKEND,
    MARKER_BACKEND_LABELS,
    marker_backend_availability_text,
    marker_backend_available,
)
from pyCamSet.workflow.params import ParamError
from pyCamSet.workflow.run_quality import blocking_reasons
from pyCamSet.workflow.workspace import WorkspaceManager

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
# Styling helpers
# ---------------------------------------------------------------------------

BUTTON_ROLES = {"orange": "secondary", "green": "success", "blue": "primary"}


class WheelMutationGuard(QObject):
    """Ignore wheel changes on numeric/choice controls unless they have focus.

    An unfocused wheel gesture is normally an attempt to scroll the page, not
    to silently alter a calibration setting. Consuming it here avoids
    re-posting wheel events (and the resulting scroll-recursion risk).
    """

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if event.type() == QEvent.Type.Wheel and isinstance(
                obj, (QAbstractSpinBox, QComboBox)):
            if not obj.hasFocus():
                event.accept()
                return True
        return False

# ---------------------------------------------------------------------------
# Section label factory
# ---------------------------------------------------------------------------


def make_section_label(text: str) -> QLabel:
    """Return a styled section-header label."""
    lbl = QLabel(text)
    lbl.setProperty("designRole", "section")
    return lbl


def make_separator() -> QFrame:
    """Return a horizontal separator line."""
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setFrameShadow(QFrame.Shadow.Sunken)
    return sep


def make_orange_button(text: str, callback: Callable) -> QPushButton:
    """Return a secondary action button connected to *callback*."""
    return _make_role_button(text, callback, BUTTON_ROLES["orange"])


def make_warning_button(text: str, callback: Callable) -> QPushButton:
    """Return an orange warning/navigation button connected to *callback*."""
    return _make_role_button(text, callback, "warning")


def make_green_button(text: str, callback: Callable) -> QPushButton:
    """Return a success/commit button connected to *callback*."""
    return _make_role_button(text, callback, BUTTON_ROLES["green"])


def make_blue_button(text: str, callback: Callable) -> QPushButton:
    """Return a primary action button connected to *callback*."""
    return _make_role_button(text, callback, BUTTON_ROLES["blue"])


def _make_role_button(text: str, callback: Callable, role: str) -> QPushButton:
    """Create a button styled by the active application theme."""
    btn = QPushButton(text)
    btn.setProperty("designRole", role)
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
        btn.setProperty("designRole", "warning")
        btn.style().unpolish(btn)
        btn.style().polish(btn)
        btn.setToolTip("This run cannot be carried forward:\n\n"
                       + "\n\n".join(reasons))
        return
    btn.setProperty("designRole", "success")
    btn.style().unpolish(btn)
    btn.style().polish(btn)
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
    """A managed Matplotlib visual with presentation and source-data exports."""

    def __init__(
        self,
        title: str,
        fig,
        canvas_cls,
        parent: Optional[QWidget] = None,
        min_height: int = 300,
        csv_export: Optional[dict[str, Any]] = None,
        csv_disabled_reason: str = "CSV is unavailable: this visual has no tabular source adapter.",
        canvas=None,
        visual_id: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._fig = fig
        self._canvas_cls = canvas_cls
        self._csv_export = csv_export
        from pyCamSet.gui.theme import apply_matplotlib_theme
        from pyCamSet.gui.visual_style import (
            VisualStyle, apply_visual_style, style_from_json, style_path_for_visual,
        )
        legacy_visual_id = "figure:" + re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")
        self._visual_id = visual_id or legacy_visual_id
        from pyCamSet.gui.preferences import config_directory
        self._style_path = style_path_for_visual(config_directory(), self._visual_id)
        self._style = VisualStyle()
        application = QApplication.instance()
        theme_name = (application.property("pycamsetTheme") if application else None) or "Light"
        apply_matplotlib_theme(fig, theme_name)
        style_source = self._style_path
        legacy_style_path = style_path_for_visual(config_directory(), legacy_visual_id)
        if not style_source.exists() and self._visual_id != legacy_visual_id and legacy_style_path.exists():
            # Read the old run-specific sidecar in place; migrate only on explicit save.
            style_source = legacy_style_path
        if style_source.exists():
            try:
                self._style = style_from_json(
                    style_source.read_text(encoding="utf-8"),
                    legacy_visual_id if style_source == legacy_style_path else self._visual_id)
                apply_visual_style(
                    fig, self._style, theme_name)
            except (OSError, ValueError):
                # Invalid preference files are ignored, never partially applied.
                self._style = VisualStyle()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 8)

        header = QHBoxLayout()
        header.addWidget(make_section_label(title))
        header.addStretch()
        save_btn = QPushButton("Save PNG")
        from pyCamSet.gui.action_icons import set_action_icon
        set_action_icon(save_btn, "snapshot")
        save_btn.setFixedWidth(82)
        save_btn.setAccessibleName(f"Save PNG for {title}")
        save_btn.clicked.connect(self._save_png)
        header.addWidget(save_btn)
        self._preset = QComboBox()
        self._preset.addItem("Screen template · 160 mm · 150 dpi", (160.0, 150))
        self._preset.addItem("Publication single-column template · 85 mm · 300 dpi", (85.0, 300))
        self._preset.addItem("Publication double-column template · 180 mm · 300 dpi", (180.0, 300))
        self._preset.setToolTip("Generic sizing templates only; not a claim of compliance with any named journal.")
        from pyCamSet.gui.preferences import bind_export_preset
        bind_export_preset(self._preset, self._visual_id)
        header.addWidget(self._preset)
        for fmt in ("SVG", "PDF"):
            vector_btn = QPushButton(f"Save {fmt}")
            vector_btn.clicked.connect(lambda _checked=False, output_format=fmt: self._save_vector(output_format))
            header.addWidget(vector_btn)
        self._csv_btn = QPushButton("Save CSV")
        set_action_icon(self._csv_btn, "chart")
        has_csv_rows = csv_export is not None and bool(csv_export.get("rows"))
        self._csv_btn.setEnabled(has_csv_rows)
        self._csv_btn.setToolTip("Export source-backed numeric data." if has_csv_rows else csv_disabled_reason)
        self._csv_btn.clicked.connect(self._save_csv)
        header.addWidget(self._csv_btn)
        style_btn = QPushButton("Style…")
        set_action_icon(style_btn, "options")
        style_btn.setAccessibleName(f"Figure style options for {title}")
        style_btn.clicked.connect(self._edit_style)
        header.addWidget(style_btn)
        expand_btn = QPushButton("Expand")
        expand_btn.setFixedWidth(78)
        expand_btn.clicked.connect(self._open_expanded)
        header.addWidget(expand_btn)
        layout.addLayout(header)

        self._canvas = canvas if canvas is not None else self._canvas_cls(self._fig)
        self._canvas.setMinimumHeight(min_height)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # Expose the figure's visible heading to assistive technology without
        # claiming a data summary that the canvas cannot reliably provide.
        self._canvas.setAccessibleName(title)
        self._canvas.setAccessibleDescription(
            f"Scientific figure: {title}. Use the figure controls to expand or save it."
        )
        layout.addWidget(self._canvas)

    def _save_png(self) -> None:
        import re
        safe_title = re.sub(r'[^\w\s-]', '', self._title).strip().replace(' ', '_')
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Figure as PNG", f"{safe_title}.png", "PNG Files (*.png)"
        )
        if not path:
            return
        try:
            from pyCamSet.gui.visual_style import _validate_user_style_filename
            _validate_user_style_filename(Path(path).name)
            width_mm, dpi = self._preset.currentData()
            original_size = self._fig.get_size_inches().copy()
            width_inches = width_mm / 25.4
            height_inches = width_inches * float(original_size[1]) / float(original_size[0])
            pixel_width = round(width_inches * dpi)
            pixel_height = round(height_inches * dpi)
            self._fig.set_size_inches(pixel_width / dpi, pixel_height / dpi, forward=False)
            try:
                self._fig.savefig(path, dpi=int(dpi), format="png")
            finally:
                self._fig.set_size_inches(original_size, forward=False)
        except Exception as exc:
            QMessageBox.warning(self, "PNG export failed", f"The figure could not be saved.\n\nTechnical detail: {exc}")

    def _save_vector(self, output_format: str) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, f"Save Figure as {output_format}", f"figure.{output_format.lower()}",
            f"{output_format} Files (*.{output_format.lower()})")
        if not path:
            return
        try:
            from pyCamSet.gui.visual_style import _validate_user_style_filename
            _validate_user_style_filename(Path(path).name)
            width_mm, dpi = self._preset.currentData()
            original_size = self._fig.get_size_inches().copy()
            width_inches = width_mm / 25.4
            height_inches = width_inches * float(original_size[1]) / float(original_size[0])
            self._fig.set_size_inches(width_inches, height_inches, forward=False)
            try:
                self._fig.savefig(path, format=output_format.lower(), dpi=int(dpi))
            finally:
                self._fig.set_size_inches(original_size, forward=False)
        except Exception as exc:
            QMessageBox.warning(self, f"{output_format} export failed",
                                f"The figure could not be saved.\n\nTechnical detail: {exc}")

    def _save_csv(self) -> None:
        if self._csv_export is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save source data as CSV", "figure-data.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            from pyCamSet.gui.visual_style import _validate_user_style_filename
            _validate_user_style_filename(Path(path).name)
            adapter = self._csv_export
            with open(path, "w", newline="", encoding="utf-8") as stream:
                stream.write("# metadata: " + json.dumps(adapter["metadata"], ensure_ascii=False) + "\n")
                writer = csv.writer(stream)
                writer.writerow(adapter["columns"])
                writer.writerows(adapter["rows"])
        except Exception as exc:
            QMessageBox.warning(self, "CSV export failed", f"Source data could not be saved.\n\nTechnical detail: {exc}")

    def _edit_style(self) -> None:
        """Preview a visual-only style and persist it outside scientific runs."""
        from PySide6.QtWidgets import QMessageBox
        from pyCamSet.gui.visual_style import (
            VisualStyle, VisualStyleDialog, _VISUAL_OVERRIDES,
            _restore_presentation_state, apply_visual_style, style_to_json,
        )

        application = QApplication.instance()
        theme_name = application.property("pycamsetTheme") if application else "Light"
        dialog = VisualStyleDialog(self._fig, self._visual_id, self._style,
                                   theme_name, self, lambda *_: self._canvas.draw_idle())
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        candidate = dialog.current
        temporary_path = None
        try:
            candidate.validate()
            self._style_path.parent.mkdir(parents=True, exist_ok=True)
            # Stage beside the sidecar so promotion is atomic on the same filesystem.
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self._style_path.parent,
                prefix=f".{self._style_path.name}.", suffix=".tmp", delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(style_to_json(candidate, self._visual_id))
            temporary_path.replace(self._style_path)
        except (OSError, ValueError) as exc:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
            # Restore the exact pre-dialog artist and override states after a failed save.
            self._style = dialog.original
            _restore_presentation_state(self._fig, dialog.original_artist_state)
            if dialog.original_override is None:
                _VISUAL_OVERRIDES.pop(self._fig, None)
            else:
                _VISUAL_OVERRIDES[self._fig] = dialog.original_override
            self._canvas.draw_idle()
            QMessageBox.warning(self, "Figure style not saved",
                                f"The style could not be saved.\n\nTechnical detail: {exc}")
            return
        self._style = candidate
        apply_visual_style(self._fig, candidate, theme_name)
        self._canvas.draw_idle()

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
    widget.setAccessibleName(meta.label)
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


#: How a :class:`TargetSettingsForm` treats the detector a target is read
#: with.  Which detector reads a board is a question for the detection, not
#: for the board: making a target asks nothing about it (``none``), the tabs
#: that run a detection choose one (``choose``), and the phases that reuse a
#: Phase 1 run's detections read them with that run's detector (``inherit``).
DETECTOR_NONE = "none"
DETECTOR_CHOOSE = "choose"
DETECTOR_INHERIT = "inherit"
DETECTOR_MODES = (DETECTOR_NONE, DETECTOR_CHOOSE, DETECTOR_INHERIT)

#: The ArUco detectors, by the short name a sentence uses for each.
_ARUCO_BACKEND_NAMES = {ARUCO1_BACKEND: "ArUco 1", ARUCO2_BACKEND: "ArUco 2"}


def aruco_backends_of(target_type: str) -> tuple[str, ...]:
    """
    The ArUco detectors a target can be read with, in the order it lists them.

    The first is the one it is read with when nobody says otherwise.

    :param target_type: the target, as :data:`TARGET_NAMES` keys it
    :raises ValueError: for an unknown target
    """
    from pyCamSet.calibration_targets.core.target_registry import target_class

    return tuple(backend for backend in target_class(target_type).DETECTOR_BACKENDS
                 if backend in _ARUCO_BACKEND_NAMES)


class _WithLoadedChoices(Parameterisation):
    """
    A target's construction parameters, also accepting values a loaded
    spec named that their choices do not offer.

    Those values build -- a target's constructor takes any dictionary its
    detector knows -- so a form holding one must read it back rather than
    refuse the run it came from.
    """

    def __init__(self, base: Parameterisation, loaded: dict[str, str]) -> None:
        self._base = base
        self.name = base.name
        self._parameters = tuple(
            replace(parameter,
                    choices=parameter.choices + (Choice(loaded[parameter.key],
                                                        loaded[parameter.key]),))
            if parameter.key in loaded and parameter.choices else parameter
            for parameter in base.parameters)

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        return self._parameters

    def validate(self, values: dict[str, Any]) -> list[str]:
        return self._base.validate(values)


class TargetSettingsForm(QWidget):
    """
    The controls a calibration target is described with.

    A target combo, a control for each argument the selected target
    declares, and -- depending on ``detector_mode`` -- a detector row.
    Which is the point: the target's own arguments were a map of widget
    names, one copy per phase, so a target none of them named had no form
    at all.  Rebuilt when the target changes; a detector change only moves
    the detector row, since what a board is does not depend on what reads
    it.

    The detector is chosen where detection happens.  ``none`` (Create
    Target) has no detector row, since a board prints the same whichever
    detector reads it.  ``choose`` (Phase 1, Optimisation) offers ArUco 1 and
    ArUco 2 for every target read with ArUco markers, greying out the one a
    target cannot use.  ``inherit`` (Phases 2 and 3) offers nothing: it shows
    the detector of the Phase 1 run :meth:`apply_spec` adopted, since those
    phases read that run's detections.

    ``marker_backend`` is written here and in
    :func:`~pyCamSet.workflow.targets.detector_parameterisation_of`, and
    nowhere else outside the two targets that take one: it is what those
    constructors call the detector they are read with.

    :param parent: the owning widget
    :param targets: the target names to offer; defaults to every one
    :param detector_mode: one of :data:`DETECTOR_MODES`
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
                 targets: Optional[list[str]] = None,
                 detector_mode: str = DETECTOR_CHOOSE) -> None:
        super().__init__(parent)
        from pyCamSet.calibration_targets.core.target_registry import (
            TARGET_NAMES,
            target_label,
        )

        if detector_mode not in DETECTOR_MODES:
            raise ValueError(
                f"detector_mode must be one of {', '.join(DETECTOR_MODES)}; "
                f"got {detector_mode!r}.")
        self._detector_mode = detector_mode

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
        #: ``setCurrentIndex`` -- captures nothing into :attr:`_retained`
        #: from whatever was on the form a moment ago.
        self._applying_spec = False
        #: The detector last chosen for a target that has a choice of two
        #: (``choose`` mode).  Kept apart from what the combo shows, because
        #: a ChArUco2 target forces ArUco 2 onto the combo, and flipping back
        #: to a ChArUco must not silently keep a choice nobody made.
        self._chosen_backend = ARUCO1_BACKEND
        #: The detector of the run :meth:`apply_spec` adopted (``inherit``
        #: mode), or None before one is -- read as the target's default.
        self._inherited_backend: Optional[str] = None
        #: The target of the run :meth:`apply_spec` adopted (``inherit``
        #: mode).  A ChArUco2 spec names no detector, so whether a run was
        #: adopted cannot be read off :attr:`_inherited_backend` alone.
        self._inherited_target: Optional[str] = None
        #: The adopted run's own detector tuning (``inherit`` mode), read
        #: back from its spec's ``detection_options`` -- there is no widget
        #: for it, so without this a Phase 2/3 fallback redetection would
        #: silently use each detector's built-in defaults instead of the
        #: settings that made the adopted Phase 1 run's detection succeed.
        self._inherited_detection_options: Optional[dict] = None
        #: True while :meth:`_sync_detector_row` moves the combo itself, so
        #: that move is not taken for a person's choice.
        self._syncing_backend = False
        #: Values :meth:`apply_spec` loaded that the row's choices do not
        #: offer, keyed by parameter -- a saved spec naming a dictionary
        #: since dropped from the list.  Accepted by :meth:`spec` until the
        #: rows are rebuilt, so an adopted run reads back as it was saved
        #: rather than being refused.
        self._loaded_choices: dict[str, str] = {}

        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        # Shown by label, known by name: the registry names are what every
        # saved spec carries, and the labels say which marker generation a
        # board uses.
        self._target_combo = QComboBox()
        for name in (targets or TARGET_NAMES):
            self._target_combo.addItem(target_label(name), name)
        self._target_combo.setToolTip(
            "The calibration target these settings describe.")
        form.addRow("Target type:", self._target_combo)

        self._backend_label: Optional[QLabel] = None
        self._backend_combo: Optional[QComboBox] = None
        self._inherited_backend_label: Optional[QLabel] = None
        self._backend_status: Optional[QLabel] = None
        if detector_mode != DETECTOR_NONE:
            self._backend_label = QLabel("Detector:")
            if detector_mode == DETECTOR_CHOOSE:
                self._backend_combo = QComboBox()
                for label, value in MARKER_BACKEND_LABELS.items():
                    self._backend_combo.addItem(label, value)
                self._backend_combo.setToolTip(
                    "Which library reads this target's markers.")
                form.addRow(self._backend_label, self._backend_combo)
            else:
                self._inherited_backend_label = QLabel()
                self._inherited_backend_label.setToolTip(
                    "The detector the Phase 1 run whose detections this "
                    "phase reads was run with.  Choose a detector in "
                    "Phase 1.")
                form.addRow(self._backend_label, self._inherited_backend_label)
            self._backend_status = QLabel()
            self._backend_status.setStyleSheet("font-size: 10px;")
            form.addRow("", self._backend_status)

        self._rows = QWidget()
        self._rows_form = QFormLayout(self._rows)
        self._rows_form.setContentsMargins(0, 0, 0, 0)
        self._rows_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.addRow(self._rows)

        self._target_combo.currentIndexChanged.connect(self._rebuild)
        if self._backend_combo is not None:
            self._backend_combo.currentIndexChanged.connect(self._on_backend_selected)
        self._rebuild()

    # -- what is selected ------------------------------------------------

    def detector_mode(self) -> str:
        """How this form treats the detector: one of :data:`DETECTOR_MODES`."""
        return self._detector_mode

    def target_type(self) -> str:
        """The selected target's name, as the registry keys it."""
        return str(self._target_combo.currentData())

    def set_target_type(self, target_type: str) -> None:
        """
        Select a target by its name, whatever label it is shown by.

        :param target_type: the target's name, as the registry keys it
        :raises ValueError: for a target this form does not offer
        """
        index = self._target_combo.findData(target_type)
        if index < 0:
            raise ValueError(
                f"This form does not offer a target named {target_type!r}.")
        self._target_combo.setCurrentIndex(index)

    def backend(self) -> str:
        """
        The detector the selected target will be read with.

        The one chosen in ``choose`` mode, or, in ``inherit`` mode, the
        adopted run's -- but only while the selected target is still the
        one that run was made with; a target picked by hand afterwards
        gets its own default, whatever the adopted run's detector was.  A
        target read without ArUco markers never asks, and gets
        ``"aruco1"``.
        """
        usable = aruco_backends_of(self.target_type())
        if self._backend_combo is not None:
            wanted = self._backend_combo.currentData()
        elif (self._detector_mode == DETECTOR_INHERIT
              and self._inherited_target == self.target_type()):
            # Only the adopted run's own target reads with its detector; a
            # different target picked by hand afterwards falls through to
            # its own default below, however the two line up.
            wanted = self._inherited_backend
        else:
            wanted = None
        if wanted in usable:
            return str(wanted)
        return usable[0] if usable else ARUCO1_BACKEND

    def detector_parameterisation(self):
        """What the selected target's detection can be told."""
        return detector_parameterisation_for(self.target_type(), self.backend())

    def construction_parameters(self):
        """What the selected target says it is described by."""
        from pyCamSet.calibration_targets.core.target_registry import target_class

        return target_class(self.target_type()).construction_parameters()

    def _writes_marker_backend(self) -> bool:
        """Whether the spec carries the detector: a detection-phase form,
        and a target whose constructor takes one."""
        import inspect

        from pyCamSet.calibration_targets.core.target_registry import target_class

        if self._detector_mode == DETECTOR_NONE:
            return False
        constructor = target_class(self.target_type()).__init__
        return "marker_backend" in inspect.signature(constructor).parameters

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
        # The rows these belonged to are about to go.
        self._loaded_choices = {}

        self._sync_detector_row()

        self._widgets = {}
        while self._rows_form.rowCount():
            self._rows_form.removeRow(0)
        for parameter in self.construction_parameters().settable():
            widget = build_parameter_widget(parameter)
            if parameter.key in self._edited and parameter.key in self._retained:
                # A retained value can be wrong for this target -- out of a
                # spin box's range, a dictionary this target does not
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

    def _sync_detector_row(self) -> None:
        """Show the detector the selected target will be read with.

        In ``choose`` mode a detector the target cannot be read with stays
        in the combo, greyed out and saying why, and the selection moves to
        one it can -- back to the last one chosen, when that is usable
        again.
        """
        from pyCamSet.calibration_targets.core.target_registry import target_label

        if self._detector_mode == DETECTOR_NONE:
            return
        target_type = self.target_type()
        usable = aruco_backends_of(target_type)
        shown = bool(usable)
        self._backend_label.setVisible(shown)
        self._backend_status.setVisible(shown)

        if self._backend_combo is not None:
            self._backend_combo.setVisible(shown)
            if shown:
                why = (f"{target_label(target_type)} targets are read with "
                       f"{' or '.join(_ARUCO_BACKEND_NAMES[b] for b in usable)} only.")
                model = self._backend_combo.model()
                wanted = (self._chosen_backend if self._chosen_backend in usable
                          else usable[0])
                self._syncing_backend = True
                try:
                    for index in range(self._backend_combo.count()):
                        allowed = self._backend_combo.itemData(index) in usable
                        # Greyed rather than removed: the other detector
                        # exists, and this says why it is not on offer.
                        model.item(index).setEnabled(allowed)
                        self._backend_combo.setItemData(
                            index, "" if allowed else why,
                            Qt.ItemDataRole.ToolTipRole)
                    self._backend_combo.setCurrentIndex(
                        self._backend_combo.findData(wanted))
                finally:
                    self._syncing_backend = False
        else:
            self._inherited_backend_label.setVisible(shown)
            if shown:
                backend = self.backend()
                label = next(label for label, value in MARKER_BACKEND_LABELS.items()
                             if value == backend)
                # The adopted run's own target is read with that run's
                # detector; any other is shown with its default as such.
                if (self._inherited_target != target_type
                        and backend == usable[0]):
                    label += " (this target's default)"
                self._inherited_backend_label.setText(label)

        if shown:
            backend = self.backend()
            self._backend_status.setText(marker_backend_availability_text(backend))
            self._backend_status.setStyleSheet(
                "font-size: 10px; color: "
                + ("#2a7a2a;" if marker_backend_available(backend) else "#8a4a00;"))

    def _on_backend_selected(self, index: int) -> None:
        """The detector combo moved: a choice, unless the form moved it."""
        if self._syncing_backend:
            return
        value = self._backend_combo.itemData(index)
        if value in aruco_backends_of(self.target_type()):
            self._chosen_backend = str(value)
        # A detector this target cannot use is only reachable from code; the
        # sync puts the selection back on one it can.  The rows are left
        # alone: what a board is does not depend on what reads it, and
        # rebuilding them would drop a spec apply_spec loaded back to the
        # target's defaults.
        self._sync_detector_row()
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
            collecting this also collects that. In ``inherit`` mode, where
            there is no such widget, the adopted run's own tuning
            (:meth:`apply_spec`) is used when this is left as ``None`` and
            the selected target is still the run's own -- otherwise a
            Phase 2/3 fallback redetection would silently drop the Phase 1
            run's tuning and use the detector's built-in defaults instead.
            A different target picked by hand gets no inherited tuning: the
            run's options are for its own target and may not even apply to
            this one.
        :raises pyCamSet.workflow.ParamError: for a value the target cannot take
        """
        from pyCamSet.calibration_targets.core.target_registry import TYPE_KEY

        parameters = self.construction_parameters()
        if self._loaded_choices:
            parameters = _WithLoadedChoices(parameters, self._loaded_choices)
        try:
            values = parameters.parse(
                {key: read_parameter_widget(widget)
                 for key, widget in self._widgets.items()})
        except ValueError as exc:
            raise ParamError(str(exc)) from None

        spec: dict[str, Any] = {TYPE_KEY: self.target_type(), **values}
        if self._writes_marker_backend():
            spec["marker_backend"] = self.backend()
        if (detection_options is None and self._detector_mode == DETECTOR_INHERIT
                and self._inherited_target == self.target_type()):
            # As with backend() above: the adopted run's own tuning is only
            # right for the target it was made with, not one swapped in by
            # hand afterwards.
            detection_options = self._inherited_detection_options
        if detection_options is not None:
            spec["detection_options"] = detection_options
        return spec

    def apply_spec(self, spec: dict) -> None:
        """
        Set these controls from a saved run's target.

        Phases 2 and 3 build their own target and pair it with detections
        made by an earlier run, so the default that is right almost always
        is the one the detections were made with -- and the detector is
        not a default there but the run's, which they show and write back.

        A loaded spec wins outright: the form ends up as the spec, with
        every key it does not mention at the new target's own default --
        never a value left over from whatever was on the form a moment
        ago, and never a value this call's own writes are mistaken for
        someone editing.
        """
        from pyCamSet.calibration_targets.core.target_registry import TYPE_KEY

        if not spec or TYPE_KEY not in spec:
            # A run recorded without a target spec has no detector to
            # adopt, so the one a previous run left must not linger.
            self.clear_inherited()
            return
        self._applying_spec = True
        try:
            # Nothing carried over from before this call may leak into a
            # key the spec does not mention.
            self._retained = {}
            self._edited = set()
            self._loaded_choices = {}
            target_type = str(spec[TYPE_KEY])

            # The detector first, so the rebuild below shows it.
            backend = spec.get("marker_backend")
            backend = str(backend) if backend else None
            try:
                usable = aruco_backends_of(target_type)
            except ValueError:
                usable = ()
            if self._detector_mode == DETECTOR_INHERIT:
                self._inherited_backend = backend if backend in usable else None
                self._inherited_target = target_type
                self._inherited_detection_options = spec.get("detection_options")
            elif (self._detector_mode == DETECTOR_CHOOSE
                  and len(usable) > 1 and backend in usable):
                # A remembered target's detector is a choice someone made.
                self._chosen_backend = backend

            index = self._target_combo.findData(target_type)
            if index < 0:
                # A target this form does not offer: the rows stay as they
                # are, as they always did, but the detector row still says
                # what the selected target will be read with.
                self._sync_detector_row()
            elif index == self._target_combo.currentIndex():
                # setCurrentIndex is a no-op when the type is already
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
                self._target_combo.setCurrentIndex(index)

            parameters = (self.construction_parameters() if index >= 0 else None)
            for key, widget in self._widgets.items():
                if spec.get(key) is not None:
                    set_parameter_widget(widget, spec[key])
                    # A choice the list no longer offers -- a dictionary
                    # dropped from it -- is still what the run was made with.
                    parameter = (parameters.parameter(key)
                                 if parameters is not None and key in parameters
                                 else None)
                    if parameter is not None and parameter.choices and not any(
                            spec[key] in (choice.label, choice.value)
                            for choice in parameter.choices):
                        self._loaded_choices[key] = str(spec[key])
        finally:
            self._applying_spec = False
            # These writes must not outlive the call as if someone had
            # typed them: the very next flip should not carry a spec's
            # value into a target the spec never named.
            self._edited = set()

    def clear_inherited(self) -> None:
        """
        Forget the adopted run's detector (``inherit`` mode).

        For when the run a phase adopted no longer applies -- another
        workspace with no run of its own, or a run recorded without a target
        spec -- so the selected target is read
        with its own default again, and says so.  The target and its rows
        are left as they are.
        """
        if self._detector_mode != DETECTOR_INHERIT:
            return
        self._inherited_backend = None
        self._inherited_target = None
        self._inherited_detection_options = None
        self._sync_detector_row()
        self.changed.emit()


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
        self.setObjectName("terminalOutput")
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

        self._heading = make_section_label("Saved runs")
        layout.addWidget(self._heading)

        self._selection_summary = QLabel("No runs selected.")
        self._selection_summary.setWordWrap(True)
        self._selection_summary.setAccessibleName("Run selection summary")
        layout.addWidget(self._selection_summary)

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self._list.setAccessibleName("Saved runs")
        self._list.setAccessibleDescription(
            "Select one or more saved runs to show their diagnostics."
        )
        self._list.itemSelectionChanged.connect(self._emit_selection)
        layout.addWidget(self._list)

        self._empty_lbl = QLabel("No saved runs are available. Run this phase, then refresh.")
        self._empty_lbl.setWordWrap(True)
        self._empty_lbl.setAccessibleName("No saved runs")
        self._empty_lbl.setAccessibleDescription(
            "The run list is empty. Complete a phase run and refresh to load it."
        )
        layout.addWidget(self._empty_lbl)

        self._runs: list[dict] = []
        self.refresh(runs)

    def _emit_selection(self) -> None:
        self._update_selection_summary()
        self.selection_changed.emit(self.get_selected())

    def _update_selection_summary(self) -> None:
        selected_count = len(self._list.selectedItems())
        total_count = len(self._runs)
        self._selection_summary.setText(
            f"{selected_count} of {total_count} runs selected. "
            "Select runs to compare their diagnostics."
        )

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
            self._update_selection_summary()
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
        self._update_selection_summary()

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
