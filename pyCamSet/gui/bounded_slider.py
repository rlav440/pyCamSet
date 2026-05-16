"""Purpose: Render one Optimisation-tab detector-parameter row.

Status: Active shared widget for numeric detector sliders and the discrete corner-refinement selector.

Future: Keep special-case categorical handling narrow unless more Optimisation controls genuinely need it.
"""
from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QWidget,
)


_FLOAT_SCALE = 1000  # scaler used when rendering float ranges on integer-only QSlider


def _make_spin(entry: dict[str, Any]) -> QDoubleSpinBox | QSpinBox:
    """Build a spin box appropriate for *entry*'s dtype."""
    if entry["dtype"] == "int":
        spin = QSpinBox()
        spin.setRange(int(entry["min"]), int(entry["max"]))
        spin.setSingleStep(int(entry.get("step", 1)) or 1)
        spin.setValue(int(entry["default"]))
        return spin
    spin = QDoubleSpinBox()
    spin.setRange(float(entry["min"]), float(entry["max"]))
    step = float(entry.get("step", 0.01) or 0.01)
    spin.setSingleStep(step)
    spin.setDecimals(int(entry.get("decimals", 3)))
    spin.setValue(float(entry["default"]))
    return spin


class BoundedSliderRow(QWidget):
    """One detector-parameter row (§4.5).

    Signals
    -------
    valueChanged(str, object)
        Emitted whenever any of the row's controls change.  ``object`` is the
        fixed value (int or float) post-coercion.
    optimiseChanged(str, bool)
        Emitted when the optimise checkbox flips.
    """

    valueChanged = Signal(str, object)
    optimiseChanged = Signal(str, bool)

    def __init__(self, entry: dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._entry = entry
        self._key = entry["key"]
        self._is_float = entry["dtype"] == "float"
        self._choices = list(entry.get("choices") or [])
        self._is_choice = bool(self._choices)

        self._label = QLabel(entry.get("label", entry["key"]))
        self._label.setMinimumWidth(220)

        self._fixed_spin: QDoubleSpinBox | QSpinBox | None = None
        self._fixed_combo: QComboBox | None = None
        if self._is_choice:
            self._fixed_combo = QComboBox()
            self._fixed_combo.setFixedWidth(170)
            for choice in self._choices:
                self._fixed_combo.addItem(str(choice["label"]), choice["value"])
            default_index = self._fixed_combo.findData(entry["default"])
            if default_index < 0:
                raise ValueError(
                    f"Choice metadata for {self._key!r} is missing default value {entry['default']!r}."
                )
            self._fixed_combo.setCurrentIndex(default_index)
        else:
            self._fixed_spin = _make_spin(entry)
            self._fixed_spin.setFixedWidth(110)

        self._optimise = QCheckBox("optimise")
        self._optimise.setChecked(False)

        self._lower_spin: QDoubleSpinBox | QSpinBox | None = None
        self._upper_spin: QDoubleSpinBox | QSpinBox | None = None
        self._slider: QSlider | None = None
        if not self._is_choice:
            self._lower_spin = _make_spin(entry)
            self._lower_spin.setFixedWidth(90)
            self._lower_spin.setEnabled(False)
            self._upper_spin = _make_spin(entry)
            self._upper_spin.setFixedWidth(90)
            self._upper_spin.setEnabled(False)
            # Default search range = absolute bounds for numeric parameters.
            if self._is_float:
                self._lower_spin.setValue(float(entry["min"]))
                self._upper_spin.setValue(float(entry["max"]))
            else:
                self._lower_spin.setValue(int(entry["min"]))
                self._upper_spin.setValue(int(entry["max"]))
            self._slider = QSlider(Qt.Orientation.Horizontal)
            self._configure_slider()

        # Layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.addWidget(self._label)
        if self._is_choice:
            layout.addWidget(self._fixed_combo)
            layout.addStretch(1)
        else:
            layout.addWidget(self._fixed_spin)
            layout.addWidget(self._slider, stretch=1)
        layout.addWidget(self._optimise)
        if not self._is_choice:
            layout.addWidget(QLabel("lower"))
            layout.addWidget(self._lower_spin)
            layout.addWidget(QLabel("upper"))
            layout.addWidget(self._upper_spin)

        # Wiring
        if self._fixed_combo is not None:
            self._fixed_combo.currentIndexChanged.connect(self._on_combo_changed)
        if self._fixed_spin is not None:
            self._fixed_spin.valueChanged.connect(self._on_fixed_changed)
        if self._slider is not None:
            self._slider.valueChanged.connect(self._on_slider_changed)
        self._optimise.toggled.connect(self._on_optimise_toggled)

    # ------------------------------------------------------------------

    def _configure_slider(self) -> None:
        entry = self._entry
        if self._is_float:
            self._slider.setMinimum(int(float(entry["min"]) * _FLOAT_SCALE))
            self._slider.setMaximum(int(float(entry["max"]) * _FLOAT_SCALE))
            self._slider.setValue(int(float(entry["default"]) * _FLOAT_SCALE))
            step = float(entry.get("step", 0.01) or 0.01)
            self._slider.setSingleStep(max(1, int(step * _FLOAT_SCALE)))
        else:
            self._slider.setMinimum(int(entry["min"]))
            self._slider.setMaximum(int(entry["max"]))
            self._slider.setValue(int(entry["default"]))
            self._slider.setSingleStep(int(entry.get("step", 1)) or 1)

    # ------------------------------------------------------------------

    @property
    def key(self) -> str:
        return self._key

    def fixed_value(self) -> Any:
        if self._fixed_combo is not None:
            return self._fixed_combo.currentData()
        if self._fixed_spin is None:
            return None
        return self._fixed_spin.value()

    def optimise_enabled(self) -> bool:
        return self._optimise.isChecked()

    def bounds(self) -> tuple[Any, Any]:
        if self._lower_spin is None or self._upper_spin is None:
            return None, None
        return self._lower_spin.value(), self._upper_spin.value()

    def set_fixed(self, value: Any) -> None:
        if self._fixed_combo is not None:
            index = self._fixed_combo.findData(value)
            if index >= 0:
                self._fixed_combo.setCurrentIndex(index)
            return
        if self._fixed_spin is not None:
            self._fixed_spin.setValue(value)

    def set_optimise(self, enabled: bool) -> None:
        self._optimise.setChecked(bool(enabled))

    def set_bounds(self, lower: Any, upper: Any) -> None:
        if self._lower_spin is None or self._upper_spin is None:
            return
        self._lower_spin.setValue(lower)
        self._upper_spin.setValue(upper)

    # ------------------------------------------------------------------

    def _on_fixed_changed(self, _value) -> None:
        if self._slider is None or self._fixed_spin is None:
            return
        # Sync the slider without re-triggering the signal loop.
        self._slider.blockSignals(True)
        try:
            if self._is_float:
                self._slider.setValue(int(float(self._fixed_spin.value()) * _FLOAT_SCALE))
            else:
                self._slider.setValue(int(self._fixed_spin.value()))
        finally:
            self._slider.blockSignals(False)
        self.valueChanged.emit(self._key, self.fixed_value())

    def _on_combo_changed(self, _index: int) -> None:
        self.valueChanged.emit(self._key, self.fixed_value())

    def _on_slider_changed(self, slider_value: int) -> None:
        if self._fixed_spin is None:
            return
        self._fixed_spin.blockSignals(True)
        try:
            if self._is_float:
                self._fixed_spin.setValue(slider_value / _FLOAT_SCALE)
            else:
                self._fixed_spin.setValue(int(slider_value))
        finally:
            self._fixed_spin.blockSignals(False)
        self.valueChanged.emit(self._key, self.fixed_value())

    def _on_optimise_toggled(self, checked: bool) -> None:
        if self._lower_spin is not None:
            self._lower_spin.setEnabled(checked)
        if self._upper_spin is not None:
            self._upper_spin.setEnabled(checked)
        self.optimiseChanged.emit(self._key, checked)


__all__ = ["BoundedSliderRow"]
