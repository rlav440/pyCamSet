'''Purpose: Draw accessible, scalable action icons for the pyCamSet GUI.
Status: Active; original Qt-painted semantics for options, snapshots and charts.
Future: Add an action only when the GUI has a matching operation.
'''
from __future__ import annotations

import math
import weakref

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QIcon, QIconEngine, QPainter, QPainterPath, QPalette, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QPushButton


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


def set_action_icon(button: QPushButton, semantic: str) -> None:
    """Set original Qt vector geometry without changing button semantics.

    The icon geometry is project-authored under pyCamSet's Apache-2.0 licence.
    No OM artwork, third-party asset, emoji, or external font is loaded.
    """
    if semantic not in {"options", "snapshot", "chart"}:
        raise ValueError(f"Unsupported action icon semantic: {semantic!r}")
    button.setIcon(QIcon(_ActionIconEngine(button, semantic)))
    button.setIconSize(QSize(18, 18))
