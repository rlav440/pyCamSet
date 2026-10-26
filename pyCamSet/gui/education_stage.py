"""The one stage layout every education phase uses (stepper, visual, caption, chips, numbers).

Layout (top to bottom), sized so that at 1400x900 and 1280x800 the whole stage fits with no
vertical scroll and the visual area gets >= 65 % of the height::

    [ (1) Photos  (2) What the detector sees  ...  ]  [pickers ...]  [L1 — overview v]
    +---------------------------------------------------------------------------+
    |  visual area (QStackedWidget: matplotlib canvas | Scene3D | any widget)   |
    +---------------------------------------------------------------------------+
    [ optional controls row, e.g. ScenePlayer ]
    caption (word-wrapped, reserved height)     [chip][chip][chip]   [Numbers >]
    [ collapsed Numbers drawer: table ]

Typical phase code::

    view = EducationStageView(["Photos", "What the detector sees", "Finding markers",
                               "Final", "Decision"], parent)
    camera = view.add_picker("Camera", QComboBox())
    view.stageChanged.connect(self._show_stage)
    view.levelChanged.connect(lambda _level: self._show_stage(view.current_stage()))

    def _show_stage(self, index):
        view.draw_figure(lambda fig, theme: self._draw_photos(fig))   # themed + constrained
        view.set_caption("Each camera's photo for image 12 ...")
        view.set_chips([Chip("63", "corners"), Chip("4 / 8", "cameras kept", tone="warn")])
        view.set_numbers(["Camera", "Points"], rows)

Themes: the view follows the application property ``pycamsetTheme`` (Light, Dark, Sepia)
live; figures drawn with :meth:`EducationStageView.draw_figure` are redrawn on a switch and
visual widgets with a ``set_theme(name)`` method are told.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractButton, QApplication, QComboBox, QFrame, QHBoxLayout, QLabel, QProgressBar,
    QSizePolicy,
    QStackedWidget, QTableWidget, QToolButton, QToolTip, QVBoxLayout, QWidget,
)

from pyCamSet.gui.education_style import active_theme_name, resolve_theme


def theme_tokens_for(theme_name: str | None = None) -> dict[str, str]:
    """The GUI theme tokens (``gui/theme.py``) for ``theme_name`` (None = active theme)."""
    from pyCamSet.gui.theme import THEME_TOKENS
    return dict(THEME_TOKENS[resolve_theme(theme_name)])


# --------------------------------------------------------------------------------------
# Camera and photo pickers (UX-4: one camera depiction everywhere)


def camera_badge_pixmap(index: int, size: int = 18, theme: str | None = None,
                        device_ratio: float = 2.0) -> QPixmap:
    """The camera badge as a Qt pixmap: a filled circle in the camera's fill for ``theme``
    (:func:`~pyCamSet.gui.education_style.camera_fill`) with the 1-based number and the thin
    outline ring of substituted cameras (same rule as the figures)."""
    from pyCamSet.gui.education_style import (
        badge_text_colour, camera_badge_label, camera_edge, camera_fill, resolve_theme,
    )
    theme = resolve_theme(theme)
    px = int(round(size * device_ratio))
    pixmap = QPixmap(px, px)
    pixmap.setDevicePixelRatio(device_ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    edge = camera_edge(index, theme)
    pen_width = 1.2 if edge != "none" else 0.0
    painter.setPen(QPen(QColor(edge), pen_width) if edge != "none" else Qt.PenStyle.NoPen)
    painter.setBrush(QColor(camera_fill(index, theme)))
    inset = 0.6 + pen_width / 2
    painter.drawEllipse(QRectF(inset, inset, size - 2 * inset, size - 2 * inset))
    font = QFont()
    font.setBold(True)
    label = camera_badge_label(index)
    font.setPixelSize(max(7, int(size * (0.58 if len(label) == 1 else 0.48))))
    painter.setFont(font)
    painter.setPen(QColor(badge_text_colour(index, theme)))
    painter.drawText(QRectF(0, 0, size, size), int(Qt.AlignmentFlag.AlignCenter), label)
    painter.end()
    return pixmap


def camera_badge_icon(index: int, size: int = 18, theme: str | None = None) -> QIcon:
    """:func:`camera_badge_pixmap` as an icon (combo boxes, buttons, list items)."""
    return QIcon(camera_badge_pixmap(index, size, theme))


#: Widest a camera picker grows to show "Camera n · name" in full (pixels); narrower than
#: that, only the name part is elided ("Camera 8 · cam8_B…08B"), never the number (UX2-3).
CAMERA_PICKER_MAX_WIDTH = 300


class CameraCombo(QComboBox):
    """The combo box behind :func:`camera_picker`: sized to its longest item (at most
    :data:`CAMERA_PICKER_MAX_WIDTH`), and when the header gives it less room it paints
    "Camera n · " in full and elides only the camera's name in the middle, with the full
    title as tooltip. Never clips mid-word (round-2 UX2-3: "Camera 8 · carr")."""

    def __init__(self, parent: QWidget | None = None, *,
                 max_width: int = CAMERA_PICKER_MAX_WIDTH) -> None:
        super().__init__(parent)
        self._max_width = int(max_width)
        self._shown = ""

    # -- text ---------------------------------------------------------------------------

    def full_text(self, row: int | None = None) -> str:
        """The full item text ("Camera 8 · cam8_Back_908B") of ``row`` (default current)."""
        row = self.currentIndex() if row is None else int(row)
        return self.itemText(row) if 0 <= row < self.count() else ""

    def _text_room(self) -> int:
        from PySide6.QtWidgets import QStyle, QStyleOptionComboBox
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        rect = self.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, opt,
                                           QStyle.SubControl.SC_ComboBoxEditField, self)
        icon = self.iconSize().width() + 4 if not self.itemIcon(
            max(self.currentIndex(), 0)).isNull() else 0
        return max(rect.width() - icon - 2, 10)

    def elided_text(self, width: int | None = None, row: int | None = None) -> str:
        """``full_text`` fitted to ``width`` pixels: the name part is elided in the middle,
        "Camera n" always stays."""
        full = self.full_text(row)
        width = self._text_room() if width is None else int(width)
        metrics = self.fontMetrics()
        if metrics.horizontalAdvance(full) <= width:
            return full
        head, sep, name = full.partition(" · ")
        if not sep:
            return metrics.elidedText(full, Qt.TextElideMode.ElideRight, width)
        room = width - metrics.horizontalAdvance(head + sep)
        if room < metrics.horizontalAdvance("a…a"):
            return head
        return head + sep + metrics.elidedText(name, Qt.TextElideMode.ElideMiddle, room)

    def displayText(self) -> str:  # noqa: N802 - Qt naming (as the transport counter)
        """The text as painted (what the screenshot checker compares with ``full_text``)."""
        return self.elided_text()

    def is_elided(self) -> bool:
        """True when the painted text is shorter than the full title."""
        return self.displayText() != self.full_text()

    # -- size and paint -------------------------------------------------------------------

    def _chrome_width(self) -> int:
        from PySide6.QtWidgets import QStyle, QStyleOptionComboBox
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        icon = self.iconSize().width() + 4
        size = self.style().sizeFromContents(QStyle.ContentsType.CT_ComboBox, opt,
                                             QSize(icon, self.fontMetrics().height()), self)
        return size.width()

    def _narrowed(self) -> bool:
        """True when the owner asked for a narrow picker (``setMinimumContentsLength`` with
        ``AdjustToMinimumContentsLengthWithIcon``): then it stays that narrow, showing at least
        "Camera n" (the name elided or left to the tooltip), so a tight header keeps its
        stage names."""
        return (self.sizeAdjustPolicy()
                == QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
                and self.minimumContentsLength() > 0)

    def _head_width(self) -> int:
        metrics = self.fontMetrics()
        heads = [self.itemText(i).partition(" · ")[0] for i in range(self.count())] or [""]
        return self._chrome_width() + max(metrics.horizontalAdvance(h) for h in heads) + 6

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        hint = super().sizeHint()
        if self._narrowed():
            return QSize(max(hint.width(), self._head_width()), hint.height())
        metrics = self.fontMetrics()
        widest = max((metrics.horizontalAdvance(self.itemText(i)) for i in range(self.count())),
                     default=0)
        width = min(self._chrome_width() + widest + 6, self._max_width)
        return QSize(max(width, 60), hint.height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        hint = super().minimumSizeHint()
        if self._narrowed():
            return QSize(max(hint.width(), self._head_width()), hint.height())
        metrics = self.fontMetrics()
        heads = [self.itemText(i).partition(" · ")[0] for i in range(self.count())] or [""]
        widest = max(metrics.horizontalAdvance(h + " · a…a") for h in heads)
        return QSize(min(self._chrome_width() + widest + 6, self.sizeHint().width()),
                     hint.height())

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        from PySide6.QtWidgets import QStyle, QStyleOptionComboBox, QStylePainter
        painter = QStylePainter(self)
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, opt)
        self._shown = self.elided_text()
        opt.currentText = self._shown
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, opt)

    def event(self, event) -> bool:  # noqa: D401 - Qt API
        if event.type() == QEvent.Type.ToolTip:
            extra = self.toolTip()
            name = self.itemData(self.currentIndex(), Qt.ItemDataRole.ToolTipRole)
            text = "\n".join(t for t in (str(name or self.full_text()), extra) if t)
            QToolTip.showText(event.globalPos(), text, self)
            return True
        return super().event(event)


def camera_picker(cameras: Any, parent: QWidget | None = None, *,
                  current: int | None = None,
                  max_width: int = CAMERA_PICKER_MAX_WIDTH) -> QComboBox:
    """The one camera picker: badge icon + ``"Camera n · name"`` per item.

    ``cameras``: a sequence of 0-based indices, or a mapping ``index -> name``. Item data
    (``currentData()``) is the 0-based index. Sized to its longest item up to ``max_width``
    pixels; with less room only the name is elided ("Camera 8 · cam8_B…08B") and the hover
    tooltip gives the full title (:class:`CameraCombo`). Do not shrink it with
    ``setMinimumContentsLength``; give the header room instead. Re-themes its badges when the
    stage view's theme changes (it is marked with the ``eduCameraPicker`` property).
    """
    from pyCamSet.gui.education_style import camera_title
    combo = CameraCombo(parent, max_width=max_width)
    combo.setProperty("eduCameraPicker", True)
    combo.setIconSize(QSize(18, 18))
    combo.setAccessibleName("Camera")
    items = list(cameras.items()) if hasattr(cameras, "items") else [(i, None) for i in cameras]
    for index, name in items:
        combo.addItem(camera_badge_icon(int(index)), camera_title(int(index), name, max_name=200),
                      int(index))
        if name:
            combo.setItemData(combo.count() - 1, f"Camera {int(index) + 1}: {name}",
                              Qt.ItemDataRole.ToolTipRole)
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
    combo.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    if current is not None:
        found = combo.findData(int(current))
        if found >= 0:
            combo.setCurrentIndex(found)
    return combo


def refresh_camera_picker(combo: QComboBox, theme: str | None = None) -> None:
    """Redraw a :func:`camera_picker`'s badges for ``theme`` (outline rule)."""
    for row in range(combo.count()):
        index = combo.itemData(row)
        if isinstance(index, int):
            combo.setItemIcon(row, camera_badge_icon(index, 18, theme))


def photo_picker(indices: Sequence[int], parent: QWidget | None = None, *,
                 details: Any = None, current: int | None = None) -> QComboBox:
    """The one photo picker: ``"Photo n"`` (1-based, :func:`photo_label`) per stored index,
    with an optional detail per index (``details[index]``, e.g. ``"1.87 px per photo"``:
    say *which* number it is, ED-3). Item data is the stored 0-based index."""
    from pyCamSet.gui.education_style import photo_label
    combo = QComboBox(parent)
    combo.setAccessibleName("Photo")
    combo.setToolTip("Photos are numbered from 1.")
    for index in indices:
        text = photo_label(int(index))
        detail = details.get(int(index)) if hasattr(details, "get") else (
            details[int(index)] if details is not None else None)
        if detail:
            text += f" · {detail}"
        combo.addItem(text, int(index))
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
    if current is not None:
        found = combo.findData(int(current))
        if found >= 0:
            combo.setCurrentIndex(found)
    return combo


class _ThemeHub(QObject):
    """The single application-wide event filter that watches ``pycamsetTheme``.

    It lives as long as the QApplication (parented to it), so it never dangles. Widgets do not
    install filters of their own: they subscribe to :attr:`themeChanged` through a
    :class:`ThemeWatcher`, and Qt drops the subscription when the widget is deleted.
    """

    themeChanged = Signal(str)

    def __init__(self, app: QApplication) -> None:
        super().__init__(app)
        self._app = app
        self._last = active_theme_name()
        app.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt API
        if (watched is self._app and event.type() == QEvent.Type.DynamicPropertyChange
                and event.propertyName() == b"pycamsetTheme"):
            name = active_theme_name()
            if name != self._last:
                self._last = name
                self.themeChanged.emit(name)
        return False


_THEME_HUB: _ThemeHub | None = None


def _theme_hub() -> _ThemeHub | None:
    """The hub for the running application, created on first use (None without an app)."""
    global _THEME_HUB
    app = QApplication.instance()
    if app is None:
        return None
    if _THEME_HUB is not None:
        try:
            if _THEME_HUB.parent() is app:
                return _THEME_HUB
        except RuntimeError:  # the previous application (and its hub) was destroyed
            pass
    _THEME_HUB = _ThemeHub(app)
    return _THEME_HUB


class ThemeWatcher(QObject):
    """Emits ``themeChanged(name)`` when the application's ``pycamsetTheme`` property changes.

    Cheap to create and safe to drop: it holds no application filter of its own, only a
    subscription to the shared hub, which :meth:`detach` (or the owner's deletion) removes.
    """

    themeChanged = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._hub = _theme_hub()
        if self._hub is not None:
            self._hub.themeChanged.connect(self._relay)
        if parent is not None:
            # Owner deletion is the normal end of life; detach explicitly as well.
            parent.destroyed.connect(self.detach)

    def current(self) -> str:
        """The active theme name (Light, Dark or Sepia)."""
        return active_theme_name()

    def _relay(self, name: str) -> None:
        self.themeChanged.emit(name)

    def detach(self) -> None:
        """Stop watching. Idempotent; safe after the owner has been destroyed."""
        hub, self._hub = self._hub, None
        if hub is None:
            return
        try:
            hub.themeChanged.disconnect(self._relay)
        except (RuntimeError, TypeError):  # already gone with the hub or never connected
            pass


# --------------------------------------------------------------------------------------
# Panel heading for Qt-hosted panels (3D scenes): the matplotlib panel_title look


class PanelHeading(QWidget):
    """Poster title ("A", :data:`~pyCamSet.gui.education_style.LETTER_GAP`, bold 13 pt title)
    and a muted 10 pt subtitle above a Qt-hosted panel (a 3D scene), drawn like
    :func:`~pyCamSet.gui.education_style.panel_title` so it shares the baseline of the
    matplotlib panel titles beside it (UX-31 / UX2-19: never "A The solver" with one space,
    never a few pixels higher). Text that does not fit ends in "…" (full text as tooltip).

    ``set_text("A  The solver at work", "subtitle")`` (letter parsed from the title) or
    ``set_heading("A", "The solver at work", "subtitle")``.
    """

    #: matplotlib draws the figures at 100 dpi: 13 pt = 18 px, 10 pt = 14 px.
    TITLE_PX = 18
    SUBTITLE_PX = 14
    #: Top of the title line, as constrained layout's h_pad (0.06 in at 100 dpi) + rounding.
    TOP_PX = 3
    #: Title baseline to subtitle baseline, as matplotlib draws it (measured on renders).
    SUBTITLE_GAP_PX = 19

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("eduPanelHeading")
        self._letter, self._title, self._subtitle = "", "", ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(self._height())

    def _fonts(self) -> tuple[QFont, QFont]:
        from pyCamSet.gui.education_style import FONT_FAMILIES
        title = QFont(list(FONT_FAMILIES)[0])
        title.setPixelSize(self.TITLE_PX)
        title.setBold(True)
        sub = QFont(list(FONT_FAMILIES)[0])
        sub.setPixelSize(self.SUBTITLE_PX)
        return title, sub

    def _height(self) -> int:
        title, sub = self._fonts()
        base = self.TOP_PX + QFontMetrics(title).ascent()
        return int(base + self.SUBTITLE_GAP_PX + QFontMetrics(sub).descent() + 6)

    def heading_text(self) -> str:
        """The title line as drawn in full: ``"A  The solver at work"``."""
        from pyCamSet.gui.education_style import LETTER_GAP
        return f"{self._letter}{LETTER_GAP}{self._title}" if self._letter else self._title

    def text(self) -> str:
        return self.heading_text() + (f"\n{self._subtitle}" if self._subtitle else "")

    def set_heading(self, letter: str | None, title: str, subtitle: str = "") -> None:
        self._letter, self._title, self._subtitle = (letter or "").strip(), str(title), \
            str(subtitle or "")
        self.setToolTip(f"{self.heading_text()}\n{self._subtitle}".strip())
        self.setAccessibleName(self.heading_text())
        self.setAccessibleDescription(self._subtitle)
        self.update()

    def set_text(self, title: str, subtitle: str = "") -> None:
        """Compatibility with the round-1 heading: ``"A  Title"`` or ``"A Title"``."""
        import re
        match = re.match(r"^\s*([A-Z])\s+(\S.*)$", str(title))
        if match:
            self.set_heading(match.group(1), match.group(2), subtitle)
        else:
            self.set_heading(None, str(title), subtitle)

    def restyle(self) -> None:
        self.update()

    def lines_shown(self) -> tuple[str, str]:
        """(title line, subtitle) as painted at the current width (elided with "…")."""
        title, sub = self._fonts()
        width = max(self.width() - 8, 40)
        return (QFontMetrics(title).elidedText(self.heading_text(), Qt.TextElideMode.ElideRight,
                                               width),
                QFontMetrics(sub).elidedText(self._subtitle, Qt.TextElideMode.ElideRight, width))

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        from pyCamSet.gui.education_style import stage_tokens
        t = stage_tokens(None)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.fillRect(self.rect(), QColor(t["stage"]))
        title_font, sub_font = self._fonts()
        head, sub = self.lines_shown()
        baseline = self.TOP_PX + QFontMetrics(title_font).ascent()
        painter.setFont(title_font)
        painter.setPen(QColor(t["title"]))
        painter.drawText(QPoint(4, baseline), head)
        if sub:
            painter.setFont(sub_font)
            painter.setPen(QColor(t["muted"]))
            painter.drawText(QPoint(4, baseline + self.SUBTITLE_GAP_PX), sub)
        painter.end()


# --------------------------------------------------------------------------------------
# Stage stepper


class _StagePill(QAbstractButton):
    """A numbered pill: a disc with the stage number and the short stage name."""

    _HEIGHT = 30
    _DISC = 20

    def __init__(self, number: int, name: str, parent: QWidget) -> None:
        super().__init__(parent)
        self.number = number
        self.name = name
        self.state = "upcoming"  # "current" | "done" | "upcoming"
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # the stepper owns the keyboard
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"Stage {number}: {name}")
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def _font(self) -> QFont:
        font = QFont(self.font())
        font.setBold(self.state == "current")
        return font

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        text = QFontMetrics(self._font()).horizontalAdvance(self.name)
        return QSize(min(8 + self._DISC + 7 + text + 12, 300), self._HEIGHT)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        if self.state == "current":  # the current stage always shows its name
            return QSize(min(self.sizeHint().width(), 210), self._HEIGHT)
        return QSize(8 + self._DISC + 8, self._HEIGHT)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        t = theme_tokens_for()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = rect.height() / 2
        hovered = self.underMouse()
        if self.state == "current":
            bg, fg, border = QColor(t["accent"]), QColor(t["on_accent"]), QColor(t["accent"])
            disc_bg, disc_fg = QColor(t["on_accent"]), QColor(t["accent"])
        elif self.state == "done":
            bg = QColor(t["accent_tint_strong"] if hovered else t["accent_tint"])
            fg, border = QColor(t["text"]), QColor(t["accent_tint_strong"])
            disc_bg, disc_fg = QColor(t["accent"]), QColor(t["on_accent"])
        else:
            bg = QColor(t["accent_tint"] if hovered else t["surface"])
            fg, border = QColor(t["text_muted"]), QColor(t["border_strong"])
            disc_bg, disc_fg = QColor(t["surface"]), QColor(t["text_muted"])
        painter.setPen(QPen(border, 1))
        painter.setBrush(bg)
        painter.drawRoundedRect(rect, radius, radius)
        parent = self.parentWidget()
        if self.state == "current" and parent is not None and parent.hasFocus():
            painter.setPen(QPen(QColor(t["focus"]), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect.adjusted(1.5, 1.5, -1.5, -1.5), radius - 1.5,
                                    radius - 1.5)
        disc = QRectF(5, (rect.height() - self._DISC) / 2 + 0.5, self._DISC, self._DISC)
        painter.setPen(QPen(disc_fg if self.state == "upcoming" else disc_bg, 1.2))
        painter.setBrush(disc_bg)
        painter.drawEllipse(disc)
        number_font = QFont(self.font())
        number_font.setBold(True)
        number_font.setPointSizeF(max(self.font().pointSizeF() - 1, 7))
        painter.setFont(number_font)
        painter.setPen(disc_fg)
        painter.drawText(disc, Qt.AlignmentFlag.AlignCenter, str(self.number))
        text_rect = QRectF(disc.right() + 7, 0, rect.width() - disc.right() - 7 - 10,
                           rect.height())
        if text_rect.width() > 14:
            font = self._font()
            painter.setFont(font)
            painter.setPen(fg)
            text = QFontMetrics(font).elidedText(self.name, Qt.TextElideMode.ElideRight,
                                                 int(text_rect.width()))
            painter.drawText(text_rect, int(Qt.AlignmentFlag.AlignVCenter
                                            | Qt.AlignmentFlag.AlignLeft), text)
        painter.end()


class StageStepper(QWidget):
    """Row of numbered stage pills; click a pill or use Left/Right/Home/End to move.

    Replaces the combo box + "Stage n of m". Emits ``currentChanged(index)`` (0-based).
    """

    currentChanged = Signal(int)

    def __init__(self, names: Sequence[str] = (), parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Stages")
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._layout.setSpacing(6)
        self._pills: list[_StagePill] = []
        self._index = -1
        self.set_stages(names)

    def set_stages(self, names: Sequence[str]) -> None:
        """Replace the stages (short names, 1-3 words each); selects the first one."""
        for pill in self._pills:
            self._layout.removeWidget(pill)
            pill.deleteLater()
        self._pills = []
        while self._layout.count():
            self._layout.takeAt(0)
        for number, name in enumerate(names, start=1):
            pill = _StagePill(number, str(name), self)
            pill.clicked.connect(lambda _checked=False, i=number - 1: self.setCurrentIndex(i))
            self._layout.addWidget(pill)
            self._pills.append(pill)
        self._layout.addStretch(1)
        self._index = -1
        if self._pills:
            self.setCurrentIndex(0)

    def count(self) -> int:
        """Number of stages."""
        return len(self._pills)

    def currentIndex(self) -> int:  # noqa: N802 - Qt naming, matches QComboBox
        """The current stage (0-based), -1 when there are none."""
        return self._index

    def stage_name(self, index: int) -> str:
        """Short name of stage ``index``."""
        return self._pills[index].name

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802 - Qt naming
        """Select stage ``index`` (clamped); emits ``currentChanged`` when it changes."""
        if not self._pills:
            return
        index = max(0, min(int(index), len(self._pills) - 1))
        if index == self._index:
            return
        self._index = index
        for i, pill in enumerate(self._pills):
            pill.state = "current" if i == index else ("done" if i < index else "upcoming")
            pill.setAccessibleName(
                f"Stage {i + 1} of {len(self._pills)}: {pill.name}"
                + (" (current)" if i == index else ""))
            pill.updateGeometry()
            pill.update()
        self.setAccessibleDescription(
            f"Stage {index + 1} of {len(self._pills)}: {self._pills[index].name}. "
            "Use Left and Right to change stage.")
        self.currentChanged.emit(index)

    def next(self) -> None:
        """Go to the next stage (no-op at the end)."""
        self.setCurrentIndex(self._index + 1)

    def previous(self) -> None:
        """Go to the previous stage (no-op at the start)."""
        self.setCurrentIndex(self._index - 1)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        key = event.key()
        if key == Qt.Key.Key_Right:
            self.next()
        elif key == Qt.Key.Key_Left:
            self.previous()
        elif key == Qt.Key.Key_Home:
            self.setCurrentIndex(0)
        elif key == Qt.Key.Key_End:
            self.setCurrentIndex(self.count() - 1)
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def focusInEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().focusInEvent(event)
        self.update()
        for pill in self._pills:
            pill.update()

    def focusOutEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().focusOutEvent(event)
        for pill in self._pills:
            pill.update()


# --------------------------------------------------------------------------------------
# Stat chips


@dataclass(frozen=True)
class Chip:
    """One stat chip: a big ``value`` with a small ``label`` under it.

    ``tone``: None (neutral), ``"good"``, ``"warn"``, ``"bad"`` or ``"accent"`` colours the
    value and a left stripe. ``camera`` (0-based) adds that camera's colour stripe instead.
    """

    value: str
    label: str
    tone: str | None = None
    camera: int | None = None
    tooltip: str = ""


_TONE_TOKEN = {"good": "success", "warn": "warning", "bad": "danger", "accent": "accent"}
_TONE_ROLE = {"good": "good", "warn": "warn", "bad": "bad", "accent": "accent"}
#: Small mark beside a chip's number for tones that need attention (UX3-9: the number
#: itself stays in the text colour; the tone shows in the left stripe and this mark).
_TONE_MARK = {"warn": "!", "bad": "✕"}


def _glossary_tooltip(label: str) -> str:
    """Plain-words tooltip for glossary terms found in a chip label ("" when none)."""
    from pyCamSet.gui.education_style import glossary, glossary_terms_in
    terms = glossary_terms_in(str(label))
    return "\n".join(f"{term}: {glossary(term)}" for term in terms)


class GlossaryButton(QToolButton):
    """A small "What's this?" button listing the stage's unavoidable terms in plain words.

    Hover shows every definition; click (or Space/Enter) shows them too, for keyboard and
    touch users. Hidden below ``min_level`` (default L2) and when there are no terms.
    """

    def __init__(self, parent: QWidget | None = None, min_level: int = 2) -> None:
        super().__init__(parent)
        self.setObjectName("eduGlossaryButton")
        self.min_level = int(min_level)
        self._terms: list[str] = []
        self._level = 1
        self.setFixedHeight(26)
        self.setCursor(Qt.CursorShape.WhatsThisCursor)
        self.clicked.connect(self.show_definitions)
        self.setVisible(False)

    def terms(self) -> list[str]:
        """The terms explained (glossary keys)."""
        return list(self._terms)

    def set_terms(self, terms: Sequence[str], level: int | None = None) -> None:
        """Explain ``terms`` (unknown ones are skipped); ``level`` updates visibility."""
        from pyCamSet.gui.education_style import glossary
        self._terms = [str(t) for t in dict.fromkeys(terms) if glossary(str(t))]
        if level is not None:
            self._level = int(level)
        self.setText(f"? Terms ({len(self._terms)})")
        text = self.definitions_html()
        self.setToolTip(text)
        self.setAccessibleName(f"What these terms mean: {', '.join(self._terms)}")
        self.setAccessibleDescription(" ".join(
            f"{term}: {glossary(term)}" for term in self._terms))
        self.setVisible(bool(self._terms) and self._level >= self.min_level)

    def set_level(self, level: int) -> None:
        self.set_terms(self._terms, level)

    def definitions_html(self) -> str:
        from html import escape
        from pyCamSet.gui.education_style import glossary
        return "".join(f"<p><b>{escape(term)}</b>: {escape(glossary(term))}</p>"
                       for term in self._terms)

    def show_definitions(self) -> None:
        QToolTip.showText(self.mapToGlobal(QPoint(0, self.height())),
                          self.definitions_html(), self)


class StatChips(QWidget):
    """A row of :class:`Chip` cards (big number, small label). Call :meth:`set_chips`."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self._chips: list[Chip] = []
        self._frames: list[QFrame] = []
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

    def chips(self) -> list[Chip]:
        """The chips shown."""
        return list(self._chips)

    def set_chips(self, chips: Sequence[Chip | tuple]) -> None:
        """Show ``chips`` (``Chip`` or ``(value, label[, tone])`` tuples); [] hides the row."""
        for frame in self._frames:
            self._layout.removeWidget(frame)
            frame.hide()  # deleteLater waits for the event loop; hide now so no ghost chips
            frame.deleteLater()
        self._frames = []
        self._chips = [c if isinstance(c, Chip) else Chip(*c) for c in chips]
        for chip in self._chips:
            frame = QFrame(self)
            frame.setObjectName("eduChip")
            box = QVBoxLayout(frame)
            box.setContentsMargins(12, 5, 12, 6)
            box.setSpacing(0)
            value = QLabel(str(chip.value), frame)
            value.setObjectName("eduChipValue")
            label = QLabel(str(chip.label), frame)
            label.setObjectName("eduChipLabel")
            mark_text = _TONE_MARK.get(chip.tone or "", "")
            if mark_text:
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(5)
                row.addWidget(value)
                mark = QLabel(mark_text, frame)
                mark.setObjectName("eduChipMark")
                mark.setAccessibleName({"!": "needs attention", "✕": "problem"}[mark_text])
                row.addWidget(mark, 0, Qt.AlignmentFlag.AlignVCenter)
                row.addStretch(1)
                box.addLayout(row)
            else:
                box.addWidget(value)
            box.addWidget(label)
            frame.setAccessibleName(f"{chip.label}: {chip.value}")
            tooltip = chip.tooltip or _glossary_tooltip(chip.label)
            if tooltip:
                frame.setToolTip(tooltip)
            frame.setProperty("chipTone", chip.tone or "")
            frame.setProperty("chipCamera", -1 if chip.camera is None else int(chip.camera))
            self._layout.addWidget(frame)
            self._frames.append(frame)
        self.setVisible(bool(self._chips))
        self.restyle()

    def restyle(self) -> None:
        """Re-apply theme colours (called by the stage view on a theme switch).

        Tone colours are the *figure* tokens (``stage_tokens``), so a "warn" chip never
        uses a camera's hue (UX-8). The number is always in the text colour; the tone shows
        only in the left stripe and, for warn / bad, a small "!" / "✕" mark (UX3-9: ochre
        numbers read as camera 2's brown)."""
        from pyCamSet.gui.education_style import camera_fill, stage_tokens
        t = theme_tokens_for()
        figure = stage_tokens(None)
        for frame in self._frames:
            tone = frame.property("chipTone") or ""
            camera = frame.property("chipCamera")
            stripe = (camera_fill(camera, figure["theme"]) if camera is not None and camera >= 0
                      else figure[_TONE_ROLE[tone]] if tone in _TONE_ROLE else t["border"])
            value_colour = t["text"]
            tone_colour = figure[_TONE_ROLE[tone]] if tone in _TONE_ROLE else t["text"]
            frame.setStyleSheet(
                f"QFrame#eduChip {{ background: {t['surface_alt']}; border: 1px solid "
                f"{t['border']}; border-left: 4px solid {stripe}; border-radius: 8px; }}"
                f"QLabel#eduChipValue {{ color: {value_colour}; font-size: 15pt; "
                "font-weight: 600; background: transparent; border: none; }"
                f"QLabel#eduChipMark {{ color: {tone_colour}; font-size: 11pt; "
                "font-weight: 700; background: transparent; border: none; }"
                f"QLabel#eduChipLabel {{ color: {t['text_muted']}; font-size: 8.5pt; "
                "background: transparent; border: none; }")


# --------------------------------------------------------------------------------------
# Numbers drawer


class NumbersDrawer(QWidget):
    """A "Numbers" toggle button plus a collapsed table for tabular detail.

    The button lives in the stage's bottom band (:attr:`button`); the table opens below it.
    """

    toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None, max_height: int = 230) -> None:
        super().__init__(parent)
        from pyCamSet.gui.education_levels import setup_evidence_table
        self.button = QToolButton(parent)
        self.button.setObjectName("eduNumbersButton")
        self.button.setCheckable(True)
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.setArrowType(Qt.ArrowType.RightArrow)
        self.button.setText("Numbers")
        self.button.setAccessibleName("Show the numbers table")
        self.button.setToolTip("Show or hide the table with the exact numbers for this stage.")
        self.button.setMinimumHeight(30)
        self.table = QTableWidget(self)
        self.table.setAccessibleName("Numbers for this stage")
        self.table.setMaximumHeight(max_height)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        setup_evidence_table(self.table)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        self.button.toggled.connect(self.set_open)
        self.setVisible(False)
        self.button.setEnabled(False)

    def is_open(self) -> bool:
        """True while the table is shown."""
        return self.button.isChecked() and self.isVisibleTo(self.parentWidget())

    def set_open(self, opened: bool) -> None:
        """Open or close the drawer."""
        opened = bool(opened) and self.table.rowCount() > 0
        if self.button.isChecked() != opened:
            self.button.blockSignals(True)
            self.button.setChecked(opened)
            self.button.blockSignals(False)
        self.button.setArrowType(Qt.ArrowType.DownArrow if opened else Qt.ArrowType.RightArrow)
        self.button.setAccessibleName("Hide the numbers table" if opened
                                      else "Show the numbers table")
        self.setVisible(opened)
        self.toggled.emit(opened)

    def set_rows(self, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        """Fill the table (cells keep full text as tooltips); no rows disables the button."""
        from pyCamSet.gui.education_levels import fill_evidence_table
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels([str(h) for h in headers])
        fill_evidence_table(self.table, rows)
        has_rows = bool(rows)
        self.button.setEnabled(has_rows)
        self.button.setText(f"Numbers ({len(rows)})" if has_rows else "Numbers")
        if not has_rows:
            self.set_open(False)


# --------------------------------------------------------------------------------------
# Stage view


class _NoHeightForWidth(QWidget):
    """Container that hides its children's height-for-width from the stage layout.

    A word-wrapped label anywhere in the stage would otherwise make a QScrollArea size the
    whole stage from preferred heights (the figure canvas's 500 px hint) and scroll for nothing.
    """

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt API
        return False


class _CaptionLabel(QLabel):
    """Word-wrapped caption whose height the stage reserves itself (see _reserve_caption)."""

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt API
        return False


#: Caption lines reserved per detail level (brief: <= 2 / 3 / 4 short sentences); also the
#: most rendered lines a caption may take at 1280 px (``education_style.CAPTION_MAX_LINES``).
CAPTION_LINES = {1: 2, 2: 3, 3: 4}

FigureDrawer = Callable[[Any, str], None]


class EducationStageView(QWidget):
    """The stage layout every education phase uses (see the module docstring).

    Signals: ``stageChanged(int)`` (0-based), ``levelChanged(int)`` (1..3),
    ``themeChanged(str)``.
    """

    stageChanged = Signal(int)
    levelChanged = Signal(int)
    themeChanged = Signal(str)

    def __init__(self, stages: Sequence[str] = (), parent: QWidget | None = None, *,
                 show_level: bool = True, phase_label: str = "") -> None:
        super().__init__(parent)
        self.setObjectName("eduStageView")
        self._drawer: FigureDrawer | None = None
        self._figure = None
        self._canvas = None
        self._level = 1
        #: The phase heading ("Phase 4"); it names the saved figure files.
        self.phase_label = phase_label
        self._watcher = ThemeWatcher(self)
        self._watcher.themeChanged.connect(self._on_theme_changed)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # Header row: stepper | pickers | level.
        header = QHBoxLayout()
        header.setSpacing(10)
        self.stepper = StageStepper(stages, self)
        if phase_label:
            self.stepper.setAccessibleName(f"{phase_label} stages")
        header.addWidget(self.stepper, 1)
        self._pickers = QHBoxLayout()
        self._pickers.setSpacing(10)
        header.addLayout(self._pickers)
        self.level_selector: QComboBox | None = None
        if show_level:
            from pyCamSet.gui.education_levels import make_level_selector
            self.level_selector = make_level_selector(self, self._level_index_changed)
            self.level_selector.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToContents)
            header.addWidget(self.level_selector)
        # Save the stage's figure (or its PyVista render) through the figure export presets.
        from pyCamSet.gui.education_export import StageFigureSaveControl
        self.save_control = StageFigureSaveControl(self, self)
        header.addWidget(self.save_control)
        root.addLayout(header)

        # Loading bar: an indeterminate bar for work that runs off the GUI thread (shown by
        # set_busy). Hidden otherwise, and it takes no height while hidden.
        self.busy_bar = QProgressBar(self)
        self.busy_bar.setObjectName("eduBusyBar")
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setTextVisible(True)
        self.busy_bar.setFixedHeight(16)
        self.busy_bar.setAccessibleName("Loading")
        self.busy_bar.hide()
        root.addWidget(self.busy_bar)

        # Visual area.
        self.visual_panel = QFrame(self)
        self.visual_panel.setObjectName("eduVisualPanel")
        panel_layout = QVBoxLayout(self.visual_panel)
        panel_layout.setContentsMargins(6, 6, 6, 6)
        self.visual = QStackedWidget(self.visual_panel)
        self.visual.setMinimumSize(240, 180)
        self.visual.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        panel_layout.addWidget(self.visual)
        root.addWidget(self.visual_panel, 1)

        # Optional controls row (e.g. a ScenePlayer's transport).
        self._controls_host = _NoHeightForWidth(self)
        self._controls_slot = QVBoxLayout(self._controls_host)
        self._controls_slot.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._controls_host)
        self._controls: QWidget | None = None

        # Bottom band: caption | chips | Numbers.
        band = QHBoxLayout()
        band.setSpacing(14)
        self.caption = _CaptionLabel(self)
        self.caption.setObjectName("eduCaption")
        self.caption.setWordWrap(True)
        self.caption.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.caption.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.caption.setAccessibleName("What this stage shows")
        self.caption.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        band.addWidget(self.caption, 1)
        self.caption.installEventFilter(self)
        self.chips = StatChips(self)
        band.addWidget(self.chips, 0, Qt.AlignmentFlag.AlignTop)
        # Numbers and "? Terms" share one column, so the glossary never narrows the caption
        # (a narrower caption wraps to more lines and could make the stage scroll).
        side = QVBoxLayout()
        side.setSpacing(4)
        self.numbers = NumbersDrawer(self)
        side.addWidget(self.numbers.button)
        self.glossary = GlossaryButton(self)
        side.addWidget(self.glossary)
        side.addStretch(1)
        band.addLayout(side)
        root.addLayout(band)
        root.addWidget(self.numbers)
        self._glossary_terms: list[str] | None = None  # None = found automatically
        #: Layout-contract problems of the current caption (too many sentences, or more
        #: rendered lines at 1280 px than the level allows; see :meth:`caption_line_count`).
        self.caption_issues: list[str] = []
        #: Rendered caption lines at the narrowest supported dialog width (None until laid out).
        self.caption_lines_at_narrowest: int | None = None
        self.caption_lines_by_width: dict[int, int] = {}
        self.caption_last_lines: dict[int, str] = {}

        self.stepper.currentChanged.connect(self.stageChanged)
        self._caption_lines = CAPTION_LINES[1]
        self._apply_theme()
        self._reserve_caption()

    # -- stages and level ---------------------------------------------------------------

    def set_stages(self, names: Sequence[str]) -> None:
        """Replace the stage names (selects stage 0 and emits ``stageChanged``)."""
        self.stepper.set_stages(names)

    def current_stage(self) -> int:
        """Current stage, 0-based."""
        return self.stepper.currentIndex()

    def set_current_stage(self, index: int) -> None:
        """Select stage ``index`` (0-based)."""
        self.stepper.setCurrentIndex(index)

    def level(self) -> int:
        """Detail level 1..3 (L1 overview, L2 details, L3 full evidence)."""
        return self._level

    def set_level(self, level: int) -> None:
        """Set the detail level 1..3 (updates the selector, emits ``levelChanged``)."""
        level = max(1, min(3, int(level)))
        if self.level_selector is not None:
            self.level_selector.setCurrentIndex(level - 1)
        else:
            self._level_index_changed(level - 1)

    def _level_index_changed(self, index: int) -> None:
        level = int(index) + 1
        if level == self._level:
            return
        self._level = level
        self._caption_lines = CAPTION_LINES.get(level, 3)
        self._reserve_caption()
        self.glossary.set_level(level)
        self.levelChanged.emit(level)

    def add_picker(self, label: str, widget: QWidget) -> QWidget:
        """Add a header control (camera / photo picker); returns ``widget``.

        ``label`` "" adds the widget without a text label (a :func:`camera_picker` already
        reads "Camera n · name")."""
        if not label:
            self._pickers.addWidget(widget)
            return widget
        from pyCamSet.gui.education_levels import labelled
        row = labelled(label, widget, self)
        self._pickers.addWidget(row)
        return widget

    # -- visual area ------------------------------------------------------------------

    def theme(self) -> str:
        """The active theme name (Light, Dark or Sepia)."""
        return self._watcher.current()

    def figure(self):
        """The built-in matplotlib Figure (created on first use; constrained layout)."""
        if self._figure is None:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            from pyCamSet.gui import education_diagnostics, education_perf
            education_diagnostics.install()  # no-op unless PYCAMSET_EDU_DEBUG=1
            self._figure = Figure(figsize=(8, 5), dpi=100, layout="constrained")
            self._canvas = FigureCanvasQTAgg(self._figure)
            education_perf.cap_canvas_pixel_ratio(self._canvas)
            self._canvas.setParent(self.visual)
            self._canvas.setMinimumSize(200, 150)
            self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Expanding)
            self._canvas.setAccessibleName("Stage figure")
            self.visual.addWidget(self._canvas)
        return self._figure

    @property
    def canvas(self):
        """The built-in figure's Qt canvas (None until :meth:`figure` is first used)."""
        return self._canvas

    def draw_figure(self, drawer: FigureDrawer) -> Any:
        """Clear the built-in figure, call ``drawer(figure, theme_name)`` inside
        ``rc_context(figure_style(theme))``, show the canvas and draw it.

        The drawer is kept and re-run on a theme switch, so draw everything from data in it.
        Returns the figure.
        """
        import matplotlib
        from pyCamSet.gui.education_style import align_panel_titles, figure_style, stage_tokens
        figure = self.figure()
        self._drawer = drawer
        theme = self.theme()
        figure.clear()
        figure._edu_theme = theme
        with matplotlib.rc_context(figure_style(theme)):
            figure.set_facecolor(stage_tokens(theme)["stage"])
            # Constrained layout + one title baseline per row, elided long titles,
            # collision-free value labels and badges (EduLayoutEngine).
            align_panel_titles(figure)
            drawer(figure, theme)
        self.visual.setCurrentWidget(self._canvas)
        self._canvas.draw_idle()
        return figure

    def set_visual(self, widget: QWidget) -> QWidget:
        """Show any widget (a Scene3D, a photo grid, ...) in the visual area; returns it.

        The widget is kept in the stack, so calling again with the same widget just shows it.
        """
        if self.visual.indexOf(widget) < 0:
            self.visual.addWidget(widget)
        self.visual.setCurrentWidget(widget)
        setter = getattr(widget, "set_theme", None)
        if callable(setter):
            setter(self.theme())
        return widget

    def set_controls(self, widget: QWidget | None) -> None:
        """Put a controls row (e.g. a ScenePlayer) under the visual area; None removes it.

        The previous controls widget is hidden (not deleted; the phase owns it).
        """
        if self._controls is widget:
            if widget is not None:
                widget.show()
            return
        if self._controls is not None:
            if hasattr(self._controls, "stop_for_context_change"):
                self._controls.stop_for_context_change()
            self._controls_slot.removeWidget(self._controls)
            self._controls.hide()
        self._controls = widget
        if widget is not None:
            self._controls_slot.addWidget(widget)
            widget.show()

    # -- caption, chips, numbers ---------------------------------------------------------

    def set_busy(self, text: str | None) -> None:
        """Show the loading bar with *text* (what is being prepared), or hide it with None."""
        bar = self.busy_bar
        if text is None:
            bar.hide()
            return
        bar.setFormat(str(text))
        bar.setAccessibleDescription(str(text))
        bar.show()

    def set_caption(self, text: str, *, lines: int | None = None) -> None:
        """Set the caption (plain text; <= 2/3/4 short sentences at L1/L2/L3).

        Its height is reserved for ``lines`` (default per level), so the visual above does
        not jump between stages. Contract problems (more sentences than
        :data:`CAPTION_MAX_SENTENCES` allows at this level) are kept in
        :attr:`caption_issues` for tests and review screenshots.
        """
        self.caption.setText(str(text))
        self._update_caption_issues()
        if lines is not None:
            self._caption_lines = int(lines)
        self._reserve_caption()
        self._update_glossary()

    def set_chips(self, chips: Sequence[Chip | tuple]) -> None:
        """Show stat chips (see :class:`Chip`); [] hides the row."""
        self.chips.set_chips(chips)
        self._update_glossary()

    def set_glossary(self, terms: Sequence[str] | None) -> None:
        """Terms the "? Terms" button explains (L2+). ``None`` (default) finds glossary
        terms in the caption and chip labels automatically; ``[]`` hides the button."""
        self._glossary_terms = None if terms is None else list(terms)
        self._update_glossary()

    def _update_glossary(self) -> None:
        from pyCamSet.gui.education_style import glossary_terms_in
        if self._glossary_terms is not None:
            terms = self._glossary_terms
        else:
            text = " ".join([self.caption.text()] + [c.label for c in self.chips.chips()])
            terms = glossary_terms_in(text)
        self.glossary.set_terms(terms, self._level)

    def set_numbers(self, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        """Fill the collapsed Numbers drawer (it stays closed until the learner opens it)."""
        self.numbers.set_rows(headers, rows)

    def caption_width_at(self, dialog_width: int) -> int | None:
        """The caption's width (pixels) in a dialog ``dialog_width`` wide: its current width
        plus the difference in window width (the chips and buttons keep their width, the
        caption takes the rest). None before the caption is laid out."""
        current = self.caption.width()
        if not self.caption.isVisible() or current <= 160:
            return None
        return current - (self.window().width() - int(dialog_width))

    def caption_lines_text(self, width: int) -> list[str]:
        """The caption's rendered lines (their text) at ``width`` pixels, wrapped as the
        label wraps them (QTextLayout, the label's font)."""
        from PySide6.QtGui import QTextLayout, QTextOption
        text = self.caption.text()
        if not text:
            return []
        margins = self.caption.contentsMargins()
        inner = int(width) - margins.left() - margins.right() - 2 * self.caption.margin()
        layout = QTextLayout(text, self.caption.font())
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WordWrap)
        layout.setTextOption(option)
        lines: list[str] = []
        layout.beginLayout()
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(max(inner, 40))
            lines.append(text[line.textStart():line.textStart() + line.textLength()])
        layout.endLayout()
        return lines or [text]

    def caption_line_count(self, width: int | None = None) -> int | None:
        """Rendered lines of the caption at ``width`` pixels (default: the width it would
        have in a dialog at the narrowest supported size, 1280 px, see
        :meth:`caption_width_at`). None before the caption is laid out."""
        from pyCamSet.gui.education_style import NARROWEST_DIALOG
        if not self.caption.text():
            return 0
        if width is None:
            width = self.caption_width_at(NARROWEST_DIALOG[0])
            if width is None:
                return None
        return max(1, len(self.caption_lines_text(width)))

    def _update_caption_issues(self) -> None:
        from PySide6.QtGui import QFontInfo
        from pyCamSet.gui.education_style import (
            CAPTION_BUDGET_FONT, NARROWEST_DIALOG, SUPPORTED_DIALOG_WIDTHS, caption_problems)
        # Lines are only judged in the font the line budgets were set in, Qt's Windows UI
        # font: another font wraps the same copy differently (DejaVu Sans, Linux's usual one,
        # is wider), so counting lines in it would test the platform rather than the copy, and
        # an offscreen Qt without a font folder has no font at all.
        by_width: dict[int, list[str]] = {}
        if QFontInfo(self.caption.font()).family() == CAPTION_BUDGET_FONT and self.caption.text():
            for dialog_width in SUPPORTED_DIALOG_WIDTHS:
                width = self.caption_width_at(dialog_width)
                if width is not None:
                    by_width[int(dialog_width)] = self.caption_lines_text(width)
        narrow = by_width.get(int(NARROWEST_DIALOG[0]))
        self.caption_lines_at_narrowest = (len(narrow) if narrow is not None else
                                           (0 if not self.caption.text() else None))
        #: Rendered caption lines per supported dialog width ({1280: 4, 1400: 4}; UX3-2).
        self.caption_lines_by_width = {w: len(v) for w, v in by_width.items()}
        #: Text of the last rendered line per supported dialog width (orphan check).
        self.caption_last_lines = {w: (v[-1] if v else "") for w, v in by_width.items()}
        self.caption_issues = caption_problems(
            self.caption.text(), self._level, self.caption_lines_at_narrowest,
            lines_by_width=self.caption_lines_by_width, last_lines=self.caption_last_lines)

    def _reserve_caption(self) -> None:
        self._update_caption_issues()
        metrics = self.caption.fontMetrics()
        reserve = metrics.lineSpacing() * self._caption_lines + 4
        width = self.caption.width()
        needed = 0
        # Only trust the width once the label is laid out; before that it is a default.
        if self.caption.isVisible() and width > 160 and self.caption.text():
            # QLabel.heightForWidth over-reserves (6 lines for text that renders in 4);
            # measure the wrapped text itself.
            margins = self.caption.contentsMargins()
            inner = width - margins.left() - margins.right() - 2 * self.caption.margin()
            rect = metrics.boundingRect(QRect(0, 0, max(inner, 1), 100000),
                                        int(Qt.TextFlag.TextWordWrap), self.caption.text())
            needed = rect.height() + margins.top() + margins.bottom() + 2 * self.caption.margin() + 2
        height = max(reserve, needed)
        if self.caption.minimumHeight() != height or self.caption.maximumHeight() != height:
            self.caption.setFixedHeight(height)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt API
        """No height-for-width: the caption reserves its own height, and a scroll area that
        asked the layout would size the stage from preferred (not minimum) heights and
        scroll for nothing."""
        return False

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API
        hint = super().minimumSizeHint()
        return QSize(min(hint.width(), 640), hint.height())

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt API
        if watched is self.caption and event.type() in (QEvent.Type.Resize,
                                                         QEvent.Type.Show):
            self._reserve_caption()
        return super().eventFilter(watched, event)

    # -- theme ------------------------------------------------------------------------

    def _on_theme_changed(self, name: str) -> None:
        self._apply_theme()
        for combo in self.findChildren(QComboBox):
            if combo.property("eduCameraPicker") is True:
                refresh_camera_picker(combo, name)
        if self._drawer is not None and self._figure is not None:
            self.draw_figure(self._drawer)
        for index in range(self.visual.count()):
            setter = getattr(self.visual.widget(index), "set_theme", None)
            if callable(setter):
                setter(name)
        if self._controls is not None and callable(getattr(self._controls, "set_theme", None)):
            self._controls.set_theme(name)
        self.themeChanged.emit(name)

    def _apply_theme(self) -> None:
        from pyCamSet.gui.education_style import stage_tokens
        t = theme_tokens_for()
        stage = stage_tokens(None)["stage"]
        self.setStyleSheet(
            f"QFrame#eduVisualPanel {{ background: {stage}; border: 1px solid {t['border']}; "
            "border-radius: 10px; }"
            f"QLabel#eduCaption {{ color: {t['text']}; font-size: 10.5pt; }}"
            f"QToolButton#eduNumbersButton {{ color: {t['text']}; padding: 4px 10px; "
            f"border: 1px solid {t['border_strong']}; border-radius: 6px; "
            f"background: {t['surface']}; }}"
            f"QToolButton#eduNumbersButton:checked {{ background: {t['accent_tint']}; }}"
            f"QToolButton#eduNumbersButton:disabled {{ color: {t['text_disabled']}; "
            f"border-color: {t['border']}; }}"
            f"QToolButton#eduGlossaryButton {{ color: {t['text_muted']}; padding: 4px 8px; "
            f"border: 1px dashed {t['border_strong']}; border-radius: 6px; "
            f"background: {t['surface']}; }}")
        self.chips.restyle()
        self.stepper.update()
        for child in self.stepper.findChildren(_StagePill):
            child.update()

    def stop_for_context_change(self) -> None:
        """Stop any playback in the controls row (the dialog calls this on run changes)."""
        if self._controls is not None and hasattr(self._controls, "stop_for_context_change"):
            self._controls.stop_for_context_change()


__all__ = [
    "Chip", "StatChips", "NumbersDrawer", "StageStepper", "EducationStageView", "ThemeWatcher",
    "CAPTION_LINES", "theme_tokens_for", "GlossaryButton", "camera_badge_pixmap",
    "camera_badge_icon", "camera_picker", "refresh_camera_picker", "photo_picker",
    "CameraCombo", "CAMERA_PICKER_MAX_WIDTH", "PanelHeading",
]
