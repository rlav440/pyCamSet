"""Render one Optimisation-tab detector-parameter row."""
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

from pyCamSet.calibration_targets.core.parameters import Parameter


_FLOAT_SCALE = 1000  # scaler used when rendering float ranges on integer-only QSlider


def _make_spin(entry: Parameter) -> QDoubleSpinBox | QSpinBox:
    """Build a spin box appropriate for *entry*'s dtype."""
    if entry.dtype == "int":
        spin = QSpinBox()
        spin.setRange(int(entry.minimum), int(entry.maximum))
        spin.setSingleStep(int(entry.step or 1) or 1)
        spin.setValue(int(entry.default))
        return spin
    spin = QDoubleSpinBox()
    spin.setRange(float(entry.minimum), float(entry.maximum))
    step = float(entry.step or 0.01)
    spin.setSingleStep(step)
    spin.setDecimals(int((entry.decimals or 3)))
    spin.setValue(float(entry.default))
    return spin


class BoundedSliderRow(QWidget):
    """One detector-parameter row.

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
    boundsChanged = Signal(str, object, object)

    def __init__(self, entry: Parameter, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._entry = entry
        self._key = entry.key
        self._is_float = entry.dtype == "float"
        self._choices = list(entry.choices)
        self._is_choice = bool(self._choices)

        self._label = QLabel(entry.label)
        self._label.setMinimumWidth(220)

        self._fixed_spin: QDoubleSpinBox | QSpinBox | None = None
        self._fixed_combo: QComboBox | None = None
        if self._is_choice:
            self._fixed_combo = QComboBox()
            self._fixed_combo.setFixedWidth(170)
            for choice in self._choices:
                self._fixed_combo.addItem(str(choice.label), choice.value)
            default_index = self._fixed_combo.findData(entry.default)
            if default_index < 0:
                raise ValueError(
                    f"Choice metadata for {self._key!r} is missing default value {entry.default!r}."
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
                self._lower_spin.setValue(float(entry.minimum))
                self._upper_spin.setValue(float(entry.maximum))
            else:
                self._lower_spin.setValue(int(entry.minimum))
                self._upper_spin.setValue(int(entry.maximum))
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
        if self._lower_spin is not None:
            self._lower_spin.valueChanged.connect(self._on_bounds_changed)
        if self._upper_spin is not None:
            self._upper_spin.valueChanged.connect(self._on_bounds_changed)

    # ------------------------------------------------------------------

    def _configure_slider(self) -> None:
        entry = self._entry
        if self._is_float:
            self._slider.setMinimum(round(float(entry.minimum) * _FLOAT_SCALE))
            self._slider.setMaximum(round(float(entry.maximum) * _FLOAT_SCALE))
            self._slider.setValue(self._slider_value(self._normalise_numeric(entry.default)))
            step = float(entry.step or 0.01)
            self._slider.setSingleStep(max(1, round(step * _FLOAT_SCALE)))
        else:
            self._slider.setMinimum(int(entry.minimum))
            self._slider.setMaximum(int(entry.maximum))
            self._slider.setValue(self._slider_value(self._normalise_numeric(entry.default)))
            self._slider.setSingleStep(int(entry.step or 1) or 1)

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
            self._fixed_spin.setValue(self._normalise_numeric(value))

    def set_optimise(self, enabled: bool) -> None:
        self._optimise.setChecked(bool(enabled))

    def set_bounds(self, lower: Any, upper: Any) -> None:
        if self._lower_spin is None or self._upper_spin is None:
            return
        # Normalise metadata constraints before handing values to Qt.  In
        # particular, OpenCV adaptive-threshold windows marked ``odd`` must
        # remain odd even when a caller supplies a profile with an even edge.
        self._lower_spin.setValue(self._normalise_numeric(lower))
        self._upper_spin.setValue(self._normalise_numeric(upper))

    # ------------------------------------------------------------------

    def _on_fixed_changed(self, _value) -> None:
        if self._slider is None or self._fixed_spin is None:
            return
        fixed_value = self._normalise_numeric(self._fixed_spin.value())
        if fixed_value != self._fixed_spin.value():
            self._fixed_spin.blockSignals(True)
            try:
                self._fixed_spin.setValue(fixed_value)
            finally:
                self._fixed_spin.blockSignals(False)
        # Sync the slider without re-triggering the signal loop.
        self._slider.blockSignals(True)
        try:
            self._slider.setValue(self._slider_value(fixed_value))
        finally:
            self._slider.blockSignals(False)
        self.valueChanged.emit(self._key, self.fixed_value())

    def _on_combo_changed(self, _index: int) -> None:
        self.valueChanged.emit(self._key, self.fixed_value())

    def _on_slider_changed(self, slider_value: int) -> None:
        if self._fixed_spin is None:
            return
        value = slider_value / _FLOAT_SCALE if self._is_float else int(slider_value)
        value = self._normalise_numeric(value)
        self._fixed_spin.blockSignals(True)
        try:
            self._fixed_spin.setValue(value)
        finally:
            self._fixed_spin.blockSignals(False)
        # A mouse click can land on an even slider position despite a step of
        # two; snap the slider back to the valid odd value as well as the spin.
        self._slider.blockSignals(True)
        try:
            self._slider.setValue(self._slider_value(value))
        finally:
            self._slider.blockSignals(False)
        self.valueChanged.emit(self._key, self.fixed_value())

    def _on_optimise_toggled(self, checked: bool) -> None:
        if self._lower_spin is not None:
            self._lower_spin.setEnabled(checked)
        if self._upper_spin is not None:
            self._upper_spin.setEnabled(checked)
        self.optimiseChanged.emit(self._key, checked)

    def _on_bounds_changed(self, _value) -> None:
        if self._lower_spin is None or self._upper_spin is None:
            return
        lower = self._normalise_numeric(self._lower_spin.value())
        upper = self._normalise_numeric(self._upper_spin.value())
        if lower != self._lower_spin.value() or upper != self._upper_spin.value():
            self._lower_spin.blockSignals(True)
            self._upper_spin.blockSignals(True)
            try:
                self._lower_spin.setValue(lower)
                self._upper_spin.setValue(upper)
            finally:
                self._lower_spin.blockSignals(False)
                self._upper_spin.blockSignals(False)
        self.boundsChanged.emit(self._key, lower, upper)

    def _normalise_numeric(self, value: Any) -> Any:
        """Return a value inside the declared range and metadata constraints."""
        if self._is_float:
            return min(max(float(value), float(self._entry.minimum)), float(self._entry.maximum))
        result = int(round(float(value)))
        minimum = int(self._entry.minimum)
        maximum = int(self._entry.maximum)
        result = min(max(result, minimum), maximum)
        if self._entry.odd and result % 2 == 0:
            candidates = [
                candidate
                for candidate in (result - 1, result + 1)
                if minimum <= candidate <= maximum
            ]
            if candidates:
                result = min(candidates, key=lambda candidate: abs(candidate - result))
        return result

    def _slider_value(self, value: Any) -> int:
        """Convert a normalised control value to the integer slider domain."""
        if self._is_float:
            return round(float(value) * _FLOAT_SCALE)
        return int(value)


__all__ = ["BoundedSliderRow"]
