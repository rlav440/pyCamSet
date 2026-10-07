'''Purpose: Pick figure colours from a named palette, with exact RGB as the advanced route.
Status: Active; the swatch-and-palette control shared in spirit with the optical-mapping GUI.
Future: Add palette entries only with a name a person would say out loud.
'''
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QColorDialog, QGridLayout, QHBoxLayout, QMenu, QToolButton, QWidget, QWidgetAction,
)

#: The named colours offered first.  Rows: neutrals, warm, green, blue,
#: purple.  Includes the colour-blind-safe Okabe-Ito set (orange, sky blue,
#: bluish green, yellow, blue, vermilion, reddish purple) that the lab's
#: optical-mapping GUI uses for regions of interest.
PALETTE: tuple[tuple[str, str], ...] = (
    ("Black", "#000000"), ("Dark grey", "#444444"), ("Grey", "#888888"),
    ("Light grey", "#bbbbbb"), ("White", "#ffffff"),
    ("Dark red", "#c00000"), ("Red", "#d62728"), ("Vermilion", "#d55e00"),
    ("Orange", "#e69f00"), ("Yellow", "#f0e442"),
    ("Dark green", "#2e7d32"), ("Green", "#2ca02c"), ("Bluish green", "#009e73"),
    ("Lime", "#32cd32"), ("Brown", "#8c564b"),
    ("Navy", "#1f4e9c"), ("Blue", "#0072b2"), ("Sky blue", "#56b4e9"),
    ("Cyan", "#17becf"), ("Indigo", "#4f46e5"),
    ("Purple", "#6a3d9a"), ("Violet", "#9467bd"), ("Reddish purple", "#cc79a7"),
    ("Pink", "#e377c2"), ("Olive", "#bcbd22"),
)
_NAMES = {value: name for name, value in PALETTE}


def colour_name(value: str) -> str:
    """The palette name of *value*, or its hex code when it is a custom colour."""
    return _NAMES.get(value.lower(), value.lower())


def _swatch(value: str | None, size: int = 16) -> QPixmap:
    """A square swatch; an unset colour shows a diagonal "theme default" mark."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QPen(QColor("#808080"), 1))
    if value:
        painter.setBrush(QColor(value))
        painter.drawRoundedRect(0.5, 0.5, size - 1, size - 1, 3, 3)
    else:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(0.5, 0.5, size - 1, size - 1, 3, 3)
        painter.drawLine(3, size - 3, size - 3, 3)
    painter.end()
    return pixmap


class ColourPicker(QWidget):
    """A swatch button: named palette colours first, exact RGB under "Custom colour…".

    It deliberately offers the small text-field API (``text``, ``setText``,
    ``clear``, ``textChanged``) that the style dialog and its tests use, where
    the text is ``#rrggbb`` or empty for the theme default.
    """

    textChanged = Signal(str)

    def __init__(self, value: str | None = "", parent: QWidget | None = None, *,
                 default_label: str = "Theme default", allow_default: bool = True) -> None:
        super().__init__(parent)
        self._value = ""
        self._default_label = default_label
        self._allow_default = allow_default
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._button = QToolButton(self)
        self._button.setObjectName("colourPickerButton")
        self._button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._button.setIconSize(QSize(16, 16))
        self._button.setMinimumWidth(150)
        self._button.setToolTip("Choose a colour; exact RGB values are under Custom colour…")
        layout.addWidget(self._button)
        layout.addStretch(1)
        self._menu = QMenu(self._button)
        self._button.setMenu(self._menu)
        self._build_menu()
        self.setText(value or "")

    # -- the text-field API -------------------------------------------------
    def text(self) -> str:
        return self._value

    def setText(self, value: str | None) -> None:  # noqa: N802
        value = (value or "").strip().lower()
        if value and not QColor.isValidColorName(value):
            value = ""
        if value == self._value and self._button.text():
            return
        self._value = value
        self._button.setIcon(QIcon(_swatch(value or None)))
        self._button.setText(colour_name(value) if value else self._default_label)
        self.textChanged.emit(value)

    def clear(self) -> None:
        self.setText("")

    def setPlaceholderText(self, text: str) -> None:  # noqa: N802
        self._default_label = text
        if not self._value:
            self._button.setText(text)

    def setMaxLength(self, _length: int) -> None:  # noqa: N802
        """Accepted for text-field compatibility; a picked colour is always #rrggbb."""

    def setAccessibleName(self, name: str) -> None:  # noqa: N802
        super().setAccessibleName(name)
        self._button.setAccessibleName(name)

    # -- menu -----------------------------------------------------------------
    def _build_menu(self) -> None:
        grid_host = QWidget(self._menu)
        grid = QGridLayout(grid_host)
        grid.setContentsMargins(8, 8, 8, 4)
        grid.setSpacing(4)
        columns = 5
        for index, (name, value) in enumerate(PALETTE):
            swatch = QToolButton(grid_host)
            swatch.setObjectName("colourSwatch")
            swatch.setIcon(QIcon(_swatch(value, 18)))
            swatch.setIconSize(QSize(18, 18))
            swatch.setAutoRaise(True)
            swatch.setToolTip(f"{name} ({value})")
            swatch.setAccessibleName(name)
            swatch.clicked.connect(lambda _checked=False, colour=value: self._choose(colour))
            grid.addWidget(swatch, index // columns, index % columns)
        action = QWidgetAction(self._menu)
        action.setDefaultWidget(grid_host)
        self._menu.addAction(action)
        self._menu.addSeparator()
        if self._allow_default:
            default = self._menu.addAction(QIcon(_swatch(None)), self._default_label)
            default.triggered.connect(lambda: self._choose(""))
        custom = self._menu.addAction("Custom colour… (exact RGB)")
        custom.triggered.connect(self._custom)

    def _choose(self, value: str) -> None:
        self._menu.close()
        self.setText(value)

    def _custom(self) -> None:
        initial = QColor(self._value) if self._value else QColor("#000000")
        chosen = QColorDialog.getColor(initial, self, "Custom colour")
        if chosen.isValid():
            self.setText(chosen.name())
