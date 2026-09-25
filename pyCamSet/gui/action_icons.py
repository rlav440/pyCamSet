'''Purpose: Give figure actions the lab's shared compact icon buttons.
Status: Active; options, snapshot and CSV actions use the same glyphs as the
    optical-mapping GUI, with Qt-painted vector marks where no glyph font exists.
Future: Add an action only when the GUI has a matching operation.
'''
from __future__ import annotations

import math
import weakref

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import (
    QFont, QFontDatabase, QFontMetrics, QIcon, QIconEngine, QImage, QPainter, QPainterPath,
    QPalette, QPen, QPixmap,
)
from PySide6.QtWidgets import QApplication, QPushButton

#: Glyphs shared with the optical-mapping GUI, so both lab tools read alike.
#: They are Unicode code points drawn by the host's own symbol/emoji font;
#: no image asset or font file is shipped.
ACTION_GLYPHS = {
    "options": "⚙",       # gear: per-visual style options
    "snapshot": "\U0001F4F7",  # camera: save the visual as an image
    "chart": "\U0001F4CA",     # bar chart: save the visual's data as CSV
}

#: Fixed side of a compact icon button, in logical pixels (as in the OM GUI).
ICON_BUTTON_SIZE = 28
#: Glyph size inside the button, in pixels, independent of the UI font size.
ICON_GLYPH_PX = 13


class _ActionIconEngine(QIconEngine):
    """Paint a vector mark using the button's current text colour."""

    def __init__(self, button: QPushButton, semantic: str) -> None:
        super().__init__()
        self._button = weakref.ref(button)
        self._semantic = semantic

    def clone(self) -> _ActionIconEngine:
        """Return an independent engine retaining the weak button reference."""
        button = self._button()
        if button is None:
            application = QApplication.instance()
            button = QPushButton() if application is not None else None
        if button is None:
            raise RuntimeError("A Qt application is required to clone an action icon")
        return _ActionIconEngine(button, self._semantic)

    def pixmap(self, size: QSize, mode: QIcon.Mode, state: QIcon.State) -> QPixmap:
        """Rasterise the vector into a transparent pixmap at the requested size."""
        pixmap = QPixmap(size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        self.paint(painter, QRect(0, 0, size.width(), size.height()), mode, state)
        painter.end()
        return pixmap

    def paint(self, painter: QPainter, rect: QRect, mode: QIcon.Mode, state: QIcon.State) -> None:
        """Draw the semantic icon in the supplied logical-pixel rectangle."""
        del state
        button = self._button()
        application = QApplication.instance()
        palette = button.palette() if button is not None else application.palette()
        # Disabled actions use the disabled semantic text role, not active ink.
        group = (QPalette.ColorGroup.Disabled if mode == QIcon.Mode.Disabled
                 else palette.currentColorGroup())
        colour = palette.color(group, QPalette.ColorRole.ButtonText)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.translate(rect.x(), rect.y())
        painter.scale(rect.width() / 24.0, rect.height() / 24.0)
        painter.setPen(QPen(colour, 2.0, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if self._semantic == "options":
            self._paint_gear(painter)
        elif self._semantic == "snapshot":
            self._paint_camera(painter)
        elif self._semantic == "chart":
            self._paint_chart(painter)
        painter.restore()

    @staticmethod
    def _paint_gear(painter: QPainter) -> None:
        """Draw a toothed settings wheel and centre using vector paths."""
        points = []
        for index in range(32):
            angle = index * math.pi / 16.0 - math.pi / 2.0
            radius = 10.5 if index % 4 in (0, 1) else 8.2
            points.append((12 + math.cos(angle) * radius, 12 + math.sin(angle) * radius))
        path = QPainterPath()
        path.moveTo(*points[0])
        for point in points[1:]:
            path.lineTo(*point)
        path.closeSubpath()
        painter.drawPath(path)
        painter.drawEllipse(9.0, 9.0, 6.0, 6.0)

    @staticmethod
    def _paint_camera(painter: QPainter) -> None:
        """Draw a camera body and lens for image snapshot actions."""
        body = QPainterPath()
        body.moveTo(3.0, 7.0)
        body.lineTo(8.0, 7.0)
        body.lineTo(10.0, 4.5)
        body.lineTo(15.0, 4.5)
        body.lineTo(17.0, 7.0)
        body.lineTo(21.0, 7.0)
        body.quadTo(22.0, 7.0, 22.0, 8.0)
        body.lineTo(22.0, 19.0)
        body.quadTo(22.0, 20.0, 21.0, 20.0)
        body.lineTo(3.0, 20.0)
        body.quadTo(2.0, 20.0, 2.0, 19.0)
        body.lineTo(2.0, 8.0)
        body.quadTo(2.0, 7.0, 3.0, 7.0)
        painter.drawPath(body)
        painter.drawEllipse(8.0, 9.0, 8.0, 8.0)

    @staticmethod
    def _paint_chart(painter: QPainter) -> None:
        """Draw axes and three bars for tabular figure-data export."""
        painter.drawLine(4.0, 20.0, 4.0, 4.0)
        painter.drawLine(4.0, 20.0, 21.0, 20.0)
        painter.drawLine(8.0, 17.0, 8.0, 13.0)
        painter.drawLine(13.0, 17.0, 13.0, 8.0)
        painter.drawLine(18.0, 17.0, 18.0, 5.0)


#: Answer of the render probe, kept once a QApplication exists.
_GLYPHS_RENDERABLE: bool | None = None

#: An unassigned code point: whatever a font draws for it is its "missing
#: glyph" (usually an empty box), the shape a real glyph must not match.
_MISSING_CODEPOINT = 0x0378


def _render_probe(font: QFont, codepoint: int) -> QImage:
    """Paint one code point with *font* into a small transparent image."""
    image = QImage(32, 32, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setFont(font)
    painter.setPen(Qt.GlobalColor.black)
    painter.drawText(QRect(0, 0, 32, 32), Qt.AlignmentFlag.AlignCenter, chr(codepoint))
    painter.end()
    return image


def _has_ink(image: QImage) -> bool:
    """Whether any pixel of *image* is not fully transparent."""
    return any(image.pixelColor(x, y).alpha()
               for y in range(image.height()) for x in range(image.width()))


def glyphs_renderable() -> bool:
    """Whether this host has a font that really draws every shared action glyph.

    Qt falls back across installed fonts for a missing code point, so the
    primary UI font is the wrong thing to ask.  Instead look for one symbol
    or emoji family (Segoe UI Emoji on Windows, Apple Color Emoji on macOS,
    Noto Color Emoji on most Linux desktops) that covers all three glyphs.
    Coverage in the font's character map is not enough: some FreeType builds
    list a colour-bitmap emoji yet paint nothing or a box.  So each glyph is
    also rendered, and must leave ink that differs from the font's rendering
    of an unassigned code point.  Where no family passes, the painted vector
    marks are used instead.

    The answer is cached only once a QApplication exists; asked earlier, the
    function answers False without remembering it.
    """
    global _GLYPHS_RENDERABLE
    if _GLYPHS_RENDERABLE is not None:
        return _GLYPHS_RENDERABLE
    if QApplication.instance() is None:
        return False
    codepoints = [ord(glyph) for glyph in ACTION_GLYPHS.values()]
    found = False
    for family in QFontDatabase.families():
        folded = family.casefold()
        if "emoji" not in folded and "symbol" not in folded:
            continue
        font = QFont(family)
        font.setPixelSize(ICON_GLYPH_PX * 2)
        metrics = QFontMetrics(font)
        if not all(metrics.inFontUcs4(codepoint) for codepoint in codepoints):
            continue
        missing = _render_probe(font, _MISSING_CODEPOINT)
        if all(_has_ink(image := _render_probe(font, codepoint)) and image != missing
               for codepoint in codepoints):
            found = True
            break
    _GLYPHS_RENDERABLE = found
    return found


def set_action_icon(button: QPushButton, semantic: str) -> None:
    """Turn *button* into a compact icon button for a figure action.

    The button keeps its click signal and focusability.  Its old text label
    moves to the tooltip (unless one is set) and the accessible name (unless
    one is set), and is kept in the ``actionLabel`` property so code and tests
    can still find the action by what it does.
    """
    if semantic not in ACTION_GLYPHS:
        raise ValueError(f"Unsupported action icon semantic: {semantic!r}")
    label = button.property("actionLabel") or button.text()
    button.setProperty("actionLabel", label)
    button.setProperty("designRole", "icon")
    if not button.toolTip():
        button.setToolTip(label)
    if not button.accessibleName():
        button.setAccessibleName(label)
    button.setFixedSize(ICON_BUTTON_SIZE, ICON_BUTTON_SIZE)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    if glyphs_renderable():
        button.setIcon(QIcon())
        button.setText(ACTION_GLYPHS[semantic])
        font = QFont(button.font())
        font.setPixelSize(ICON_GLYPH_PX)
        button.setFont(font)
    else:
        button.setText("")
        button.setIcon(QIcon(_ActionIconEngine(button, semantic)))
        button.setIconSize(QSize(18, 18))
    style = button.style()
    style.unpolish(button)
    style.polish(button)
