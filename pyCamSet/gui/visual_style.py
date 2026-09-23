'''Purpose: Per-visual, presentation-only styles for GUI-managed Matplotlib output.
Status: Active; keeps explicit visual overrides outside scientific run artefacts.
Future: Extend to additional visual backends only with separate data-preservation tests.
'''
from __future__ import annotations

import json
from weakref import WeakKeyDictionary
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "pycamset.visual-style"
VERSION = 1
_VISUAL_OVERRIDES = WeakKeyDictionary()


@dataclass
class VisualStyle:
    """Theme-default fields remain unset; values are explicit per-visual overrides."""

    font_family: str | None = None
    font_size: float | None = None
    text_colour: str | None = None
    figure_background: str | None = None
    axes_background: str | None = None
    line_width: float | None = None
    marker_size: float | None = None
    grid_visible: bool | None = None
    legend_visible: bool | None = None
    overlay_size: float | None = None
    overlay_colour: str | None = None
    overlay_edge_colour: str | None = None
    series_colours: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        """Reject unknown, malformed or out-of-range style state."""
        if self.font_family is not None and (not self.font_family.strip() or len(self.font_family) > 128):
            raise ValueError("font_family must be a non-empty name up to 128 characters")
        for name in ("font_size", "marker_size", "overlay_size"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)
                                      or not 1 <= value <= 48):
                raise ValueError(f"{name} must be between 1 and 48")
        if self.line_width is not None and (
                not isinstance(self.line_width, (int, float))
                or isinstance(self.line_width, bool) or not 0.2 <= self.line_width <= 12):
            raise ValueError("line_width must be between 0.2 and 12")
        for name in ("text_colour", "figure_background", "axes_background",
                     "overlay_colour", "overlay_edge_colour"):
            _validate_colour(getattr(self, name), name)
        if len(self.series_colours) > 128:
            raise ValueError("too many series colour overrides")
        for series_id, colour in self.series_colours.items():
            if not isinstance(series_id, str) or not series_id or len(series_id) > 160:
                raise ValueError("series IDs must be non-empty strings up to 160 characters")
            _validate_colour(colour, f"series_colours[{series_id!r}]")
        if self.grid_visible is not None and not isinstance(self.grid_visible, bool):
            raise ValueError("grid_visible must be boolean or null")
        if self.legend_visible is not None and not isinstance(self.legend_visible, bool):
            raise ValueError("legend_visible must be boolean or null")


def _validate_colour(value: Any, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or len(value) != 7 or value[0] != "#":
        raise ValueError(f"{name} must be #RRGGBB or null")
    try:
        int(value[1:], 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be #RRGGBB or null") from exc


def style_to_json(style: VisualStyle, visual_id: str) -> str:
    """Serialise one visual's style, rejecting invalid state before writing."""
    style.validate()
    return json.dumps({"schema": SCHEMA, "version": VERSION, "visual_id": visual_id,
                       "style": asdict(style)}, indent=2, sort_keys=True) + "\n"


def style_from_json(text: str, expected_visual_id: str | None = None) -> VisualStyle:
    """Parse a versioned style document and fail closed on any unknown key."""
    try:
        document = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Invalid style JSON: {exc}") from exc
    if not isinstance(document, dict) or set(document) != {"schema", "version", "visual_id", "style"}:
        raise ValueError("Style document has missing or unknown top-level keys")
    if document["schema"] != SCHEMA or document["version"] != VERSION:
        raise ValueError("Unsupported style schema or version")
    if not isinstance(document["visual_id"], str) or not document["visual_id"]:
        raise ValueError("visual_id must be a non-empty string")
    if expected_visual_id is not None and document["visual_id"] != expected_visual_id:
        raise ValueError("Style belongs to a different visual")
    values = document["style"]
    if not isinstance(values, dict) or set(values) != set(VisualStyle.__dataclass_fields__):
        raise ValueError("Style has missing or unknown keys")
    style = VisualStyle(**values)
    style.validate()
    return style


def apply_visual_style(figure: Any, style: VisualStyle, theme_name: str = "Light") -> None:
    """Apply presentation properties only, leaving data, limits and colormaps intact."""
    from pyCamSet.gui.theme import THEME_TOKENS

    style.validate()
    if theme_name not in THEME_TOKENS:
        raise ValueError(f"Unknown theme: {theme_name}")
    _VISUAL_OVERRIDES[figure] = style
    tokens = THEME_TOKENS[theme_name]
    figure.set_facecolor(style.figure_background or tokens["background"])
    for axes_index, axes in enumerate(figure.axes):
        axes.set_facecolor(style.axes_background or tokens["surface"])
        axes.title.set_color(style.text_colour or tokens["text"])
        axes.xaxis.label.set_color(style.text_colour or tokens["text"])
        axes.yaxis.label.set_color(style.text_colour or tokens["text"])
        for tick in axes.get_xticklabels() + axes.get_yticklabels():
            tick.set_color(style.text_colour or tokens["text"])
        if style.grid_visible is not None:
            axes.grid(style.grid_visible)
        for line in axes.lines:
            if style.line_width is not None:
                line.set_linewidth(style.line_width)
            if style.marker_size is not None:
                line.set_markersize(style.marker_size)
            series_id = line.get_gid()
            if series_id is None and line.get_label() and not line.get_label().startswith("_"):
                series_id = f"line:{axes_index}:{line.get_label()}"
            if series_id in style.series_colours:
                line.set_color(style.series_colours[series_id])
        # Detection overlays opt in through a stable gid and are styled without
        # touching the underlying image or detection coordinate arrays.
        for collection in axes.collections:
            gid = collection.get_gid() or ""
            if gid.startswith("detection-overlay:"):
                if style.overlay_size is not None and hasattr(collection, "set_sizes"):
                    collection.set_sizes([style.overlay_size ** 2])
                if style.overlay_colour is not None and hasattr(collection, "set_facecolor"):
                    collection.set_facecolor(style.overlay_colour)
                if style.overlay_edge_colour is not None and hasattr(collection, "set_edgecolor"):
                    collection.set_edgecolor(style.overlay_edge_colour)
        legend = axes.get_legend()
        if legend is not None:
            if style.legend_visible is not None:
                legend.set_visible(style.legend_visible)
            if style.text_colour is not None:
                for text in legend.get_texts():
                    text.set_color(style.text_colour)
    # Font properties are applied to existing text artists, not data artists.
    for text in figure.findobj(match=lambda artist: hasattr(artist, "set_fontsize")):
        if style.font_size is not None:
            text.set_fontsize(style.font_size)
        if style.font_family is not None:
            text.set_fontfamily(style.font_family)
    figure.canvas.draw_idle()


def refresh_visual_style(figure: Any, theme_name: str) -> None:
    """Reapply explicit overrides after the GUI's semantic theme refresh."""
    style = _VISUAL_OVERRIDES.get(figure)
    if style is not None:
        apply_visual_style(figure, style, theme_name)


def scale_bar_unavailable(pixel_to_world: Any = None, unit: str | None = None) -> str | None:
    """Return the disabling explanation unless a physical transform and unit exist."""
    if pixel_to_world is None or not unit or not str(unit).strip():
        return "Scale bar unavailable: calibrated pixel-to-world transform and units are required."
    return None


def style_path_for_visual(app_config_dir: Path, visual_id: str) -> Path:
    """Return an isolated presentation-settings path (never a run artefact path)."""
    import hashlib

    digest = hashlib.sha256(visual_id.encode("utf-8")).hexdigest()[:16]
    return app_config_dir / "visual-styles" / f"{digest}.json"


class VisualStyleDialog:
    """Small optional-import Qt editor with preview, reset, and JSON import/export."""

    def __new__(cls, figure, visual_id: str, style: VisualStyle, theme_name: str,
                parent=None, on_preview=None):
        from PySide6.QtWidgets import (
            QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
            QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
            QVBoxLayout, QWidget,
        )
        from PySide6.QtGui import QFontDatabase

        class _Dialog(QDialog):
            def __init__(self):
                super().__init__(parent)
                self.setWindowTitle("Figure style")
                self.setModal(True)
                self.visual_id = visual_id
                self.original = VisualStyle(**asdict(style))
                self.current = VisualStyle(**asdict(style))
                self.theme_name = theme_name
                self.on_preview = on_preview
                root = QVBoxLayout(self)
                form = QFormLayout()
                root.addLayout(form)
                self.font = QComboBox()
                self.font.addItems(["Sans Serif"] + sorted(QFontDatabase.families()))
                self.font.setCurrentText(style.font_family or "Sans Serif")
                form.addRow("Typeface:", self.font)
                self.font_size = _number(QDoubleSpinBox, style.font_size, 10, 1, 48)
                form.addRow("Typography size (pt):", self.font_size)
                self.line_width = _number(QDoubleSpinBox, style.line_width, 1.5, 0.2, 12)
                form.addRow("Line width:", self.line_width)
                self.marker_size = _number(QDoubleSpinBox, style.marker_size, 6, 1, 24)
                form.addRow("Marker size:", self.marker_size)
                self.text_colour = _colour(QLineEdit, style.text_colour)
                form.addRow("Text colour (#RRGGBB):", self.text_colour)
                self.figure_background = _colour(QLineEdit, style.figure_background)
                form.addRow("Figure background:", self.figure_background)
                self.axes_background = _colour(QLineEdit, style.axes_background)
                form.addRow("Axes background:", self.axes_background)
                self.overlay_size = _number(QDoubleSpinBox, style.overlay_size, 6, 1, 24)
                form.addRow("Detection marker size:", self.overlay_size)
                self.overlay_colour = _colour(QLineEdit, style.overlay_colour)
                form.addRow("Detection marker colour:", self.overlay_colour)
                self.series_id = QComboBox()
                self.series_id.addItem("No series override", "")
                for axes_index, axes in enumerate(figure.axes):
                    for line in axes.lines:
                        label = line.get_label()
                        if label and not label.startswith("_"):
                            stable_id = line.get_gid() or f"line:{axes_index}:{label}"
                            self.series_id.addItem(f"{label} [{stable_id}]", stable_id)
                form.addRow("Series ID:", self.series_id)
                self.series_colour = _colour(QLineEdit, "")
                if self.series_id.count() < 2:
                    self.series_id.setEnabled(False)
                    self.series_colour.setEnabled(False)
                form.addRow("Series colour (#RRGGBB):", self.series_colour)
                self.grid = QCheckBox("Override grid visibility")
                self.grid.setChecked(style.grid_visible is not None)
                self.grid_value = QCheckBox("Show grid")
                self.grid_value.setChecked(bool(style.grid_visible))
                form.addRow(self.grid, self.grid_value)
                self.legend = QCheckBox("Override legend visibility")
                self.legend.setChecked(style.legend_visible is not None)
                self.legend_value = QCheckBox("Show legend")
                self.legend_value.setChecked(bool(style.legend_visible))
                form.addRow(self.legend, self.legend_value)
                scale = QCheckBox("Enable scale bar")
                scale.setEnabled(False)
                scale.setToolTip(scale_bar_unavailable())
                form.addRow("Scale bar:", scale)
                note = QLabel("Scale bars require a known pixel-to-world transform and unit. "
                              "Raw-image overlays remain unmodified.")
                note.setWordWrap(True)
                root.addWidget(note)
                row = QHBoxLayout()
                root.addLayout(row)
                save = QPushButton("Save JSON…")
                load = QPushButton("Load JSON…")
                reset = QPushButton("Reset")
                row.addWidget(save)
                row.addWidget(load)
                row.addWidget(reset)
                buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                           QDialogButtonBox.StandardButton.Cancel)
                root.addWidget(buttons)
                for widget in (self.font, self.font_size, self.line_width, self.marker_size,
                               self.text_colour, self.figure_background, self.axes_background,
                               self.overlay_size, self.overlay_colour, self.series_id,
                               self.series_colour, self.grid, self.grid_value,
                               self.legend, self.legend_value):
                    signal = getattr(widget, "currentTextChanged", None) or getattr(widget, "valueChanged", None) \
                        or getattr(widget, "textChanged", None) or getattr(widget, "toggled", None)
                    if signal is not None:
                        signal.connect(self._preview)
                self.series_id.currentIndexChanged.connect(self._load_series_colour)
                reset.clicked.connect(self._reset)
                save.clicked.connect(self._save)
                load.clicked.connect(self._load)
                buttons.accepted.connect(self.accept)
                buttons.rejected.connect(self.reject)

            def _read(self):
                series_colours = dict(self.current.series_colours)
                selected_id = self.series_id.currentData()
                if selected_id:
                    colour = self.series_colour.text().strip()
                    if colour:
                        series_colours[str(selected_id)] = colour
                    else:
                        series_colours.pop(str(selected_id), None)
                return VisualStyle(
                    font_family=(self.font.currentText()
                                 if self.font.currentText() != "Sans Serif" else None),
                    font_size=self.font_size.value() if self.font_size.value() != 10 else None,
                    text_colour=_optional(self.text_colour.text()),
                    figure_background=_optional(self.figure_background.text()),
                    axes_background=_optional(self.axes_background.text()),
                    line_width=self.line_width.value() if self.line_width.value() != 1.5 else None,
                    marker_size=self.marker_size.value() if self.marker_size.value() != 6 else None,
                    grid_visible=self.grid_value.isChecked() if self.grid.isChecked() else None,
                    legend_visible=self.legend_value.isChecked() if self.legend.isChecked() else None,
                    overlay_size=(self.overlay_size.value()
                                  if self.overlay_size.value() != 6 else None),
                    overlay_colour=_optional(self.overlay_colour.text()),
                    overlay_edge_colour=self.current.overlay_edge_colour,
                    series_colours=series_colours)

            def _load_series_colour(self, *_):
                stable_id = self.series_id.currentData()
                self.series_colour.setText(self.current.series_colours.get(stable_id, ""))

            def _preview(self, *_):
                try:
                    self.current = self._read()
                    self.current.validate()
                    apply_visual_style(figure, self.current, self.theme_name)
                    if self.on_preview:
                        self.on_preview(self.current)
                except ValueError:
                    pass

            def _reset(self):
                self.current = VisualStyle()
                self.font.setCurrentText("Sans Serif")
                self.font_size.setValue(10)
                self.line_width.setValue(1.5)
                self.marker_size.setValue(6)
                self.text_colour.clear()
                self.figure_background.clear()
                self.axes_background.clear()
                self.grid.setChecked(False)
                self.legend.setChecked(False)
                self._preview()

            def _save(self):
                from PySide6.QtWidgets import QMessageBox
                path, _ = QFileDialog.getSaveFileName(self, "Save visual style", "visual-style.json",
                                                      "JSON files (*.json)")
                if path:
                    try:
                        self.current = self._read()
                        Path(path).write_text(style_to_json(self.current, visual_id), encoding="utf-8")
                    except (OSError, ValueError) as exc:
                        QMessageBox.warning(self, "Style not saved",
                                            f"The style could not be saved.\n\nTechnical detail: {exc}")

            def _load(self):
                from PySide6.QtWidgets import QMessageBox
                path, _ = QFileDialog.getOpenFileName(self, "Load visual style", "",
                                                      "JSON files (*.json)")
                if not path:
                    return
                try:
                    candidate = style_from_json(Path(path).read_text(encoding="utf-8"), visual_id)
                except (OSError, ValueError) as exc:
                    QMessageBox.warning(self, "Style not loaded",
                                        f"The style file was rejected.\n\nTechnical detail: {exc}")
                    return
                self.current = candidate
                self.font.setCurrentText(candidate.font_family or "Sans Serif")
                self.font_size.setValue(candidate.font_size or 10)
                self.line_width.setValue(candidate.line_width or 1.5)
                self.marker_size.setValue(candidate.marker_size or 6)
                self.text_colour.setText(candidate.text_colour or "")
                self.figure_background.setText(candidate.figure_background or "")
                self.axes_background.setText(candidate.axes_background or "")
                self.grid.setChecked(candidate.grid_visible is not None)
                self.grid_value.setChecked(bool(candidate.grid_visible))
                self.legend.setChecked(candidate.legend_visible is not None)
                self.legend_value.setChecked(bool(candidate.legend_visible))
                self._preview()

            def exec(self):
                result = super().exec()
                if result != QDialog.DialogCode.Accepted:
                    apply_visual_style(figure, self.original, theme_name)
                    if self.on_preview:
                        self.on_preview(self.original)
                else:
                    self.current = self._read()
                return result

        def _number(widget_type, value, default, minimum, maximum):
            widget = widget_type()
            widget.setRange(minimum, maximum)
            widget.setValue(value if value is not None else default)
            return widget

        def _colour(widget_type, value):
            widget = widget_type(value or "")
            widget.setPlaceholderText("Theme default")
            widget.setMaxLength(7)
            return widget

        def _optional(value):
            return value.strip() or None

        return _Dialog()
