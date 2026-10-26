"""Present immutable, run-bound education views in a learner-controlled dialog.

Developer pilot (not a product entry point)::

    python -m pyCamSet.gui.education_dialog --run <run directory>

Each optional ``education_phaseN`` module exposes
``create_education_phase_view(context, parent=None)``.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAccessible, QAccessibleAnnouncementEvent, QColor, QIcon, QKeySequence, QPainter, QPixmap,
    QPolygonF, QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QAbstractSpinBox, QApplication, QComboBox, QDialog,
    QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMenu, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QStackedWidget, QTextEdit, QToolButton, QVBoxLayout,
    QWidget,
)

from pyCamSet.gui.education_levels import phase_name
from pyCamSet.gui.education_preferences import load_preferences, save_preferences


@dataclass(frozen=True)
class RecordedStep:
    """A kept solver state with its original solver step number."""

    solver_step: int
    frame: Any


PLAYBACK_SHORTCUTS = {
    "previous": "Ctrl+Alt+Left",
    "next": "Ctrl+Alt+Right",
    "toggle": "Ctrl+Alt+Space",
}
OS_RESERVED_SHORTCUTS = frozenset({
    "Alt+F4", "Ctrl+Alt+Delete", "Ctrl+Shift+Escape", "Meta+L", "Meta+Tab",
})


def shortcut_text(action: str) -> str:
    """Return a playback chord as the platform shows it, e.g. 'Ctrl+Alt+Left'."""
    return QKeySequence(PLAYBACK_SHORTCUTS[action]).toString(
        QKeySequence.SequenceFormat.NativeText)


def shortcut_conflicts(shortcuts: Mapping[str, str] = PLAYBACK_SHORTCUTS) -> tuple[str, ...]:
    """Return duplicate or OS-reserved key sequences from the declared table."""
    sequences = [QKeySequence(chord).toString(QKeySequence.SequenceFormat.PortableText)
                 for chord in shortcuts.values()]
    reserved = {QKeySequence(chord).toString(QKeySequence.SequenceFormat.PortableText)
                for chord in OS_RESERVED_SHORTCUTS}
    duplicates = {chord for chord in sequences if sequences.count(chord) > 1}
    return tuple(sorted(duplicates | (set(sequences) & reserved)))


class PlaybackPreferences(QObject):
    """The learner's two playback preferences, shared live by every transport bar.

    Reduce motion (hide Play, no tweening) and keyboard shortcuts are persisted in the
    education preferences sidecar. They are set from the dialog header's Settings menu
    (UX-3: preferences are not part of any transport bar); every bar listens here.
    """

    reduceMotionChanged = Signal(bool)
    shortcutsChanged = Signal(bool)

    def reduce_motion(self) -> bool:
        return bool(load_preferences()["reduce_motion"])

    def shortcuts_enabled(self) -> bool:
        return bool(load_preferences()["shortcuts_enabled"])

    def set_reduce_motion(self, enabled: bool) -> None:
        """Persist and broadcast (every open bar follows)."""
        if type(enabled) is not bool:
            raise ValueError("reduce_motion must be boolean")
        save_preferences({"reduce_motion": enabled})
        self.reduceMotionChanged.emit(enabled)

    def set_shortcuts_enabled(self, enabled: bool) -> None:
        """Persist and broadcast (every open bar follows)."""
        if type(enabled) is not bool:
            raise ValueError("shortcuts_enabled must be boolean")
        save_preferences({"shortcuts_enabled": enabled})
        self.shortcutsChanged.emit(enabled)


_PLAYBACK_PREFERENCES: PlaybackPreferences | None = None


def playback_preferences() -> PlaybackPreferences:
    """The process-wide :class:`PlaybackPreferences` notifier."""
    global _PLAYBACK_PREFERENCES
    if _PLAYBACK_PREFERENCES is None:
        _PLAYBACK_PREFERENCES = PlaybackPreferences(QApplication.instance())
    return _PLAYBACK_PREFERENCES


def shortcuts_help_text() -> str:
    """One sentence naming every playback chord (tooltips, the Settings menu)."""
    return (f"Step back: {shortcut_text('previous')}; step forward: {shortcut_text('next')}; "
            f"play or pause: {shortcut_text('toggle')}. Inactive while a list, number box "
            "or 3D view has keyboard focus.")


#: Playback speeds (label, milliseconds per step) for :class:`TransportBar`.
SPEEDS = (("×0.25", 1400), ("×0.5", 700), ("×1", 350))


class _StatusLabel(QLabel):
    """The bar's compact counter ("Step 5 of 11"). ``text()`` returns the *full* sentence
    (also its tooltip, accessible description and what is announced); the label paints
    the short form, so the bar stays one row (UX-3, UX-29)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full = ""

    def set_status(self, full: str, short: str) -> None:
        self._full = str(full)
        super().setText(str(short))
        self.setToolTip(self._full)
        self.setAccessibleDescription(self._full)

    def text(self) -> str:  # noqa: D401 - Qt-style accessor
        """The full status sentence (the label shows :meth:`displayText`)."""
        return self._full

    def displayText(self) -> str:  # noqa: N802 - Qt naming
        """The compact text actually painted."""
        return super().text()


def transport_icon(kind: str, colour: str, size: int = 16) -> QIcon:
    """Theme-coloured transport glyphs drawn with QPainter (no emoji-font fallback):
    ``"first"`` |◀◀, ``"back"`` |◀, ``"play"`` ▶, ``"pause"`` ❚❚, ``"next"`` ▶|."""
    ratio = 2.0
    pixmap = QPixmap(int(size * ratio), int(size * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(colour))
    s = float(size)
    top, bottom, mid = 0.2 * s, 0.8 * s, 0.5 * s

    def triangle(x_tip: float, x_base: float) -> None:
        painter.drawPolygon(QPolygonF([QPointF(x_tip, mid), QPointF(x_base, top),
                                       QPointF(x_base, bottom)]))

    def bar(x: float, width: float = 0.12) -> None:
        painter.drawRect(QRectF(x, top, width * s, bottom - top))

    if kind == "first":
        bar(0.14 * s)
        triangle(0.27 * s, 0.57 * s)
        triangle(0.55 * s, 0.85 * s)
    elif kind == "back":
        bar(0.2 * s)
        triangle(0.34 * s, 0.8 * s)
    elif kind == "next":
        triangle(0.66 * s, 0.2 * s)
        bar(0.68 * s)
    elif kind == "pause":
        bar(0.28 * s, 0.15)
        bar(0.57 * s, 0.15)
    else:  # play
        triangle(0.8 * s, 0.25 * s)
    painter.end()
    return QIcon(pixmap)


class TransportBar(QWidget):
    """The one playback / step-through bar for every education phase (UX-3).

    One compact row, same order and icons everywhere::

        [⏮] [◀] [▶ Play] [▶]  ──●──┼──┼──┼── (or named step pills)  Step 3 of 8  [×0.5] [⚙]

    ``count`` positions 0..count-1. ``names`` (optional) shows numbered step pills in place
    of the scrubber (Phase 1 detector steps). ``describe(i)`` gives the full sentence for
    position ``i`` (tooltip, screen-reader announcement), ``short(i)`` the compact counter.
    ``speeds`` adds a compact speed choice (default :data:`SPEEDS`; None hides it).
    Extra display options (e.g. exaggeration) go behind a ⚙ menu via :meth:`add_menu_choice`.

    Never plays on its own. Reduce motion and keyboard shortcuts are global preferences
    (:func:`playback_preferences`, set from the dialog header); with reduce motion on, Play
    is hidden. Keyboard: Ctrl+Alt+Left/Right step, Ctrl+Alt+Space plays (not while a list,
    number box, slider or 3D view has focus; the scrubber keeps play/pause).

    Signals: ``positionChanged(int)``, ``playingChanged(bool)``, ``announcement(str)``,
    ``reduceMotionChanged(bool)``.
    """

    positionChanged = Signal(int)
    playingChanged = Signal(bool)
    announcement = Signal(str)
    reduceMotionChanged = Signal(bool)

    def __init__(self, count: int, parent: QWidget | None = None, *,
                 names: Sequence[str] | None = None,
                 describe: Callable[[int], str] | None = None,
                 short: Callable[[int], str] | None = None,
                 unit_label: str = "Steps", interval_ms: int = 700,
                 min_interval_ms: int = 350,
                 speeds: Sequence[tuple[str, int]] | None = SPEEDS,
                 start: int = 0, restart_at_end: bool = True,
                 reduce_motion: bool | None = None,
                 scrub_help: str = "", play_name: str | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("eduTransportBar")
        if int(count) < 1:
            raise ValueError("a transport bar needs at least one position")
        self._count = int(count)
        self._last = self._count - 1
        self._names = tuple(str(n) for n in names) if names else ()
        self._describe = describe
        self._short = short
        self._unit = unit_label
        self._min_interval = int(min_interval_ms)
        self._restart_at_end = bool(restart_at_end)
        self._play_name = play_name or f"Play {unit_label.lower()}"
        self.interval_for: Callable[[int], int] | None = None
        self._index = max(0, min(int(start), self._last))
        self._timer = QTimer(self)
        self._timer.setInterval(max(self._min_interval, int(interval_ms)))
        self._base_interval = self._timer.interval()
        self._timer.timeout.connect(self._advance_playback)
        prefs = playback_preferences()
        settings = load_preferences()
        self._reduce_motion = settings["reduce_motion"] if reduce_motion is None else reduce_motion
        self._shortcuts_enabled = settings["shortcuts_enabled"]
        prefs.reduceMotionChanged.connect(self._apply_reduce_motion)
        prefs.shortcutsChanged.connect(self._apply_shortcuts_enabled)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self._first = QToolButton(self)
        self._previous = QToolButton(self)
        self._play = QToolButton(self)
        self._next = QToolButton(self)
        for button, name, action in (
                (self._first, "Go to the start", None),
                (self._previous, "Step back", "previous"),
                (self._play, self._play_name, "toggle"),
                (self._next, "Step forward", "next")):
            button.setAccessibleName(name)
            button.setToolTip(f"{name} ({shortcut_text(action)})" if action else name)
            button.setMinimumSize(32, 30)
            button.setIconSize(QSize(16, 16))
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            row.addWidget(button)
        self._play.setObjectName("eduTransportPlay")
        self._play.setText("Play")
        self._play.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._play.setMinimumWidth(78)
        from pyCamSet.gui.education_stage import ThemeWatcher
        self._theme_watcher = ThemeWatcher(self)
        self._theme_watcher.themeChanged.connect(lambda _name: self._refresh_icons())
        self._refresh_icons()
        row.addSpacing(6)
        self._scrubber = QSlider(Qt.Orientation.Horizontal, self)
        self._scrubber.setRange(0, self._last)
        self._scrubber.setValue(self._index)
        self._scrubber.setPageStep(1)
        self._scrubber.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._scrubber.setTickInterval(max(1, self._count // 40))
        self._scrubber.setAccessibleName(unit_label)
        help_text = scrub_help or f"Each position is one of the {unit_label.lower()}."
        self._scrubber.setAccessibleDescription(
            f"{help_text} The counter beside it names the position.")
        self._scrubber.setToolTip(f"Drag to move through the {unit_label.lower()}. {help_text}")
        self._scrubber.setMinimumHeight(30)
        self._scrubber.setMinimumWidth(80)
        row.addWidget(self._scrubber, 1)
        self._pills = None
        if self._names:
            from pyCamSet.gui.education_stage import StageStepper
            self._pills = StageStepper(self._names, self)
            self._pills.setAccessibleName(unit_label)
            self._pills.setCurrentIndex(self._index)
            self._pills.currentChanged.connect(self._pill_clicked)
            row.addWidget(self._pills, 1)
            self._scrubber.hide()
        self._status = _StatusLabel(self)
        self._status.setAccessibleName(f"{unit_label} position")
        self._status.setObjectName("eduTransportStatus")
        row.addWidget(self._status)
        self.speed = QComboBox(self)
        self.speed.setAccessibleName("Playback speed")
        self.speed.setToolTip("How fast Play moves through the steps.")
        self._speeds = tuple(speeds or ())
        for label, _ms in self._speeds:
            self.speed.addItem(label)
        if self._speeds:
            closest = min(range(len(self._speeds)),
                          key=lambda i: abs(self._speeds[i][1] - self._timer.interval()))
            self.speed.setCurrentIndex(closest)
            self.speed.currentIndexChanged.connect(self._speed_changed)
            row.addWidget(self.speed)
        else:
            self.speed.hide()
        self._options = QToolButton(self)
        self._options.setText("⚙")
        self._options.setAccessibleName("Display options")
        self._options.setToolTip("Display options for this picture")
        self._options.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._options.setMinimumSize(30, 30)
        self._options_menu = QMenu(self._options)
        self._options.setMenu(self._options_menu)
        self._options.hide()
        row.addWidget(self._options)

        self._first.clicked.connect(lambda: self.go_to(0))
        self._previous.clicked.connect(self.step_back)
        self._next.clicked.connect(self.step_forward)
        self._play.clicked.connect(self.toggle_playback)
        self._scrubber.valueChanged.connect(self._scrubbed)
        self._scrubber.sliderPressed.connect(self.stop_playback)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._play.setVisible(not self._reduce_motion)
        self._install_shortcuts()
        self._reserve_status_width()
        self._sync_controls()
        self._update_status(announce=False)

    # -- icons ------------------------------------------------------------------------

    def _icon(self, kind: str) -> QIcon:
        from pyCamSet.gui.education_style import stage_tokens
        return transport_icon(kind, stage_tokens(None)["text"])

    def _refresh_icons(self) -> None:
        self._first.setIcon(self._icon("first"))
        self._previous.setIcon(self._icon("back"))
        self._next.setIcon(self._icon("next"))
        self._play.setIcon(self._icon("pause" if self._timer.isActive() else "play"))

    # -- texts ------------------------------------------------------------------------

    def describe(self, index: int) -> str:
        """The full sentence for position ``index``."""
        if self._describe is not None:
            return self._describe(index)
        if self._names:
            return f"{self._unit} {index + 1} of {self._count}: {self._names[index]}."
        return f"{self._unit} {index + 1} of {self._count}."

    def short_text(self, index: int) -> str:
        """The compact counter for position ``index``."""
        if self._short is not None:
            return self._short(index)
        return f"{index + 1} / {self._count}"

    def between_text(self, lower: int) -> tuple[str, str]:
        """(full, short) counter texts for a position between ``lower`` and ``lower + 1``
        (an interpolated frame while a scene tweens, UX3-6)."""
        full = (f"Between {self.describe(lower).rstrip('.')} and the next position "
                "(interpolated).")
        return full, "→ " + self.short_text(min(lower + 1, self._last))

    def show_fraction(self, position: float) -> None:
        """Show the counter for a possibly fractional ``position`` (a tween between two
        positions) without moving the bar; a whole position shows its usual counter."""
        lower = int(max(0, min(int(position // 1), self._last)))
        if abs(position - round(position)) < 1e-9 or lower >= self._last:
            self._status.set_status(self.describe(int(round(position))),
                                    self.short_text(int(round(position))))
            return
        full, short = self.between_text(lower)
        self._status.set_status(full, short)

    def _reserve_status_width(self) -> None:
        metrics = self._status.fontMetrics()
        texts = [self.short_text(i) for i in range(self._count)]
        texts += [self.between_text(i)[1] for i in range(self._last)]
        widest = max(metrics.horizontalAdvance(t) for t in texts)
        self._status.setFixedWidth(widest + 12)

    def changeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API name
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._reserve_status_width()
        super().changeEvent(event)

    # -- state ------------------------------------------------------------------------

    def count(self) -> int:
        """Number of positions."""
        return self._count

    def position(self) -> int:
        """The position shown (0-based)."""
        return self._index

    def is_playing(self) -> bool:
        return self._timer.isActive()

    @property
    def reduce_motion(self) -> bool:
        """The learner's reduce-motion preference (Play is hidden while it is on)."""
        return self._reduce_motion

    @property
    def playback_interval_ms(self) -> int:
        """Milliseconds between steps while playing (at the chosen speed)."""
        return self._base_interval

    def set_playback_interval(self, milliseconds: int) -> None:
        """Change the time per step (never below ``min_interval_ms``)."""
        self._base_interval = max(self._min_interval, int(milliseconds))
        self._timer.setInterval(self._interval_at(self._index))

    def _interval_at(self, index: int) -> int:
        if self.interval_for is None:
            return self._base_interval
        try:
            return max(self._min_interval, int(self.interval_for(index)))
        except Exception:  # noqa: BLE001 - pacing is a nicety
            return self._base_interval

    def _speed_changed(self, index: int) -> None:
        if 0 <= index < len(self._speeds):
            self.set_playback_interval(self._speeds[index][1])

    # -- moving -----------------------------------------------------------------------

    def go_to(self, position: int) -> None:
        """Show ``position`` (clamped); stops playback."""
        self.stop_playback()
        position = max(0, min(int(position), self._last))
        if position != self._index:
            self._set_index(position, announce=True)

    def step_back(self) -> None:
        self.stop_playback()
        if self._index > 0:
            self._set_index(self._index - 1, announce=True)

    def step_forward(self) -> None:
        self.stop_playback()
        if self._index < self._last:
            self._set_index(self._index + 1, announce=True)

    def _scrubbed(self, index: int) -> None:
        self.stop_playback()
        if index != self._index:
            self._set_index(index, announce=True)

    def _pill_clicked(self, index: int) -> None:
        if index != self._index:
            self.stop_playback()
            self._set_index(index, announce=True)

    def _set_index(self, index: int, *, announce: bool) -> None:
        self._index = int(index)
        self._sync_controls()
        self._position_changed()
        self._update_status(announce=announce)

    def _position_changed(self) -> None:
        """Hook: subclasses emit their own signals here (before the status update)."""
        self.positionChanged.emit(self._index)

    def _sync_controls(self) -> None:
        self._scrubber.blockSignals(True)
        self._scrubber.setValue(self._index)
        self._scrubber.blockSignals(False)
        if self._pills is not None:
            self._pills.blockSignals(True)
            self._pills.setCurrentIndex(self._index)
            self._pills.blockSignals(False)
        self._first.setEnabled(self._index > 0)
        self._previous.setEnabled(self._index > 0)
        self._next.setEnabled(self._index < self._last)

    def _update_status(self, *, announce: bool) -> None:
        full = self.describe(self._index)
        self._status.set_status(full, self.short_text(self._index))
        if announce:
            self.announcement.emit(full)
            event = QAccessibleAnnouncementEvent(self, full)
            event.setPoliteness(QAccessible.AnnouncementPoliteness.Polite)
            QAccessible.updateAccessibility(event)

    # -- playback ---------------------------------------------------------------------

    def toggle_playback(self) -> None:
        if self._reduce_motion:
            return
        if self._timer.isActive():
            self.stop_playback(announce=True)
            return
        if self._index >= self._last:
            if not self._restart_at_end:
                return
            self._set_index(0, announce=False)
        self._play.setText("Pause")
        self._play.setIcon(self._icon("pause"))
        self._play.setAccessibleName(f"Pause {self._play_name.removeprefix('Play ')}")
        self._timer.setInterval(self._interval_at(self._index))
        self._timer.start()
        self.playingChanged.emit(True)

    def _advance_playback(self) -> None:
        if self._index >= self._last:
            self.stop_playback()
            return
        self._set_index(self._index + 1, announce=False)
        if self._index >= self._last:
            self.stop_playback()
        else:
            self._timer.setInterval(self._interval_at(self._index))

    def stop_playback(self, announce: bool = False) -> None:
        was_playing = self._timer.isActive()
        self._timer.stop()
        self._play.setText("Play")
        self._play.setIcon(self._icon("play"))
        self._play.setAccessibleName(self._play_name)
        if was_playing:
            self.playingChanged.emit(False)
            if announce:
                self._update_status(announce=True)

    def stop_for_context_change(self) -> None:
        """Stop playback before changing the stage, camera, image, theme or run."""
        self.stop_playback()

    # -- options ----------------------------------------------------------------------

    def add_menu_choice(self, title: str, options: Sequence[str], current: int,
                        on_change: Callable[[int], None]) -> QMenu:
        """Add a radio choice (e.g. "Exaggerate": ×1 ×5 ×20) behind the bar's ⚙ button;
        ``on_change(index)`` runs on a pick. Returns the submenu."""
        from PySide6.QtGui import QActionGroup
        menu = self._options_menu.addMenu(title)
        group = QActionGroup(menu)
        group.setExclusive(True)
        for k, label in enumerate(options):
            action = menu.addAction(str(label))
            action.setCheckable(True)
            action.setChecked(k == int(current))
            group.addAction(action)
            action.triggered.connect(lambda _checked=False, k=k: on_change(k))
        self._options.show()
        return menu

    # -- preferences ------------------------------------------------------------------

    def set_reduce_motion(self, enabled: bool) -> None:
        """Persist the preference for every bar; stop and move focus before hiding Play."""
        if type(enabled) is not bool:
            raise ValueError("reduce_motion must be boolean")
        if enabled == self._reduce_motion:
            return
        self._apply_reduce_motion(enabled)
        playback_preferences().set_reduce_motion(enabled)

    def _apply_reduce_motion(self, enabled: bool) -> None:
        if enabled == self._reduce_motion:
            return
        if enabled:
            self.stop_playback()
            if self._play.hasFocus():
                self._next.setFocus(Qt.FocusReason.OtherFocusReason)
        self._reduce_motion = enabled
        self._play.setVisible(not enabled)
        self.reduceMotionChanged.emit(enabled)

    def set_shortcuts_enabled(self, enabled: bool) -> None:
        """Persist the global education shortcut switch (every bar follows)."""
        if type(enabled) is not bool:
            raise ValueError("shortcuts_enabled must be boolean")
        self._apply_shortcuts_enabled(enabled)
        playback_preferences().set_shortcuts_enabled(enabled)

    def _apply_shortcuts_enabled(self, enabled: bool) -> None:
        self._shortcuts_enabled = bool(enabled)
        self._update_shortcut_focus(None, QApplication.focusWidget())

    def _install_shortcuts(self) -> None:
        self._shortcuts: dict[str, QShortcut] = {}
        for action, chord in PLAYBACK_SHORTCUTS.items():
            shortcut = QShortcut(QKeySequence(chord), self)
            shortcut.activated.connect({"previous": self.step_back,
                                        "next": self.step_forward,
                                        "toggle": self.toggle_playback}[action])
            self._shortcuts[action] = shortcut
        app = QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._update_shortcut_focus)
        self._update_shortcut_focus(None, QApplication.focusWidget())

    @staticmethod
    def _shortcuts_blocked(widget: QWidget | None) -> bool:
        blocked = (QComboBox, QAbstractSpinBox, QLineEdit, QTextEdit,
                   QAbstractItemView, QSlider)
        while widget is not None:
            class_name = type(widget).__name__
            is_render_widget = (
                widget.property("education3DView") is True
                or class_name in {"QtInteractor", "QVTKRenderWindowInteractor"}
            )
            if isinstance(widget, blocked) or is_render_widget:
                return True
            widget = widget.parentWidget()
        return False

    def _update_shortcut_focus(self, _old: QWidget | None, new: QWidget | None) -> None:
        try:
            blocked = self._shortcuts_blocked(new)
            scrubber_focus = self._has_scrubber_focus(new)
            for action, shortcut in self._shortcuts.items():
                # Keep play/pause available while the scrubber owns arrow keys.
                shortcut.setEnabled(self._shortcuts_enabled and
                                    (not blocked or (scrubber_focus and action == "toggle")))
        except RuntimeError:  # the C++ side is already gone (deleted bar)
            pass

    def _has_scrubber_focus(self, widget: QWidget | None) -> bool:
        while widget is not None:
            if widget is self._scrubber:
                return True
            widget = widget.parentWidget()
        return False


class RecordedStepControls(TransportBar):
    """The transport bar for recorded solver states; never auto-plays on open.

    Positions: an optional starting point (``start_frame``; shown first, never counted,
    numbered or announced as a step: A1 section 8), then one per kept step. Emits
    ``frameChanged(frame)`` on every move. The counter shows "Start" / "Step k of N";
    ``_status.text()`` is the full sentence (gaps of steps that were not kept included).
    """

    frameChanged = Signal(object)

    def __init__(self, steps: Sequence[RecordedStep], total_steps: int | None = None,
                 parent: QWidget | None = None, reduce_motion: bool | None = None,
                 unit_label: str = "Recorded solver steps",
                 start_frame: Any = None,
                 speeds: Sequence[tuple[str, int]] | None = None) -> None:
        if not steps:
            raise ValueError("recorded-step controls need at least one frame")
        self._steps = tuple(steps)
        if self._steps[0].solver_step < 1:
            raise ValueError("accepted solver steps start at step 1; the starting state is not a step")
        if any(right.solver_step <= left.solver_step
               for left, right in zip(self._steps, self._steps[1:])):
            raise ValueError("solver step numbers must be strictly increasing")
        self._total_steps = max(total_steps or 0, self._steps[-1].solver_step)
        self._start_frame = start_frame
        # Positions: an optional starting point, then one per kept step.
        self._offset = 0 if start_frame is None else 1
        scrub_help = ("Position 0 is the starting point, not an accepted step; each later "
                      "position is one recorded accepted solver step."
                      if self._offset else
                      "Each position is one recorded accepted solver step.")
        super().__init__(len(self._steps) + self._offset, parent,
                         describe=self._status_text, short=self._short_status,
                         unit_label=unit_label, interval_ms=350, min_interval_ms=350,
                         speeds=speeds, restart_at_end=False, reduce_motion=reduce_motion,
                         scrub_help=scrub_help, play_name="Play recorded steps")
        self.frameChanged.emit(self.current_frame)

    @property
    def at_start(self) -> bool:
        """True while the stored starting point (not a step) is shown."""
        return self._offset == 1 and self._index == 0

    @property
    def current_step(self) -> RecordedStep | None:
        """The kept step shown, or None while the starting point is shown."""
        if self.at_start:
            return None
        return self._steps[self._index - self._offset]

    @property
    def current_frame(self) -> Any:
        """The frame shown: the starting point's or the current step's."""
        step = self.current_step
        return self._start_frame if step is None else step.frame

    def _position_changed(self) -> None:
        super()._position_changed()
        self.frameChanged.emit(self.current_frame)

    def _status_text(self, index: int) -> str:
        kept_count = len(self._steps)
        if self._offset == 1 and index == 0:
            return ("Starting point — where the solver began (not an accepted step). "
                    f"{kept_count} of {self._total_steps} steps recorded.")
        position = index - self._offset
        step = self._steps[position].solver_step
        gap = ""
        if position and step - self._steps[position - 1].solver_step > 1:
            first = self._steps[position - 1].solver_step + 1
            last = step - 1
            gap = f" Steps {first}–{last} not kept."
        return f"Step {step} of {self._total_steps} ({kept_count} recorded).{gap}"

    def between_text(self, lower: int) -> tuple[str, str]:
        """Counter texts between recorded position ``lower`` and the next one: "→ Step 1 of
        11" and "Between the start and step 1 (interpolated)."."""
        upper = min(lower + 1, self._last)
        step = self._steps[upper - self._offset].solver_step
        if self._offset == 1 and lower == 0:
            before = "the start"
        else:
            before = f"step {self._steps[lower - self._offset].solver_step}"
        return (f"Between {before} and step {step} of {self._total_steps} (interpolated).",
                f"→ Step {step} of {self._total_steps}")

    def _short_status(self, index: int) -> str:
        if self._offset == 1 and index == 0:
            return "Start"
        return f"Step {self._steps[index - self._offset].solver_step} of {self._total_steps}"


class RenderWindowKeyFilter(QObject):
    """Keep VTK navigation keys in the named render widget from hijacking Qt.

    Install this on the QtInteractor widget (not the render-window object).
    The allowlist is based on ``key()`` values only; ``text()`` is layout-dependent.
    """

    _ALLOWED_KEYS = frozenset({
        Qt.Key.Key_Tab, Qt.Key.Key_Backtab, Qt.Key.Key_Escape, Qt.Key.Key_R,
        Qt.Key.Key_Up, Qt.Key.Key_Down,
    })

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt API name
        if (event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease)
                and hasattr(event, "key") and event.key() not in self._ALLOWED_KEYS):
            event.accept()
            return True
        return False


def install_render_window_key_filter(interactor: QWidget) -> RenderWindowKeyFilter:
    """Install the Qt-side VTK key guard and retain it for the widget lifetime."""
    key_filter = RenderWindowKeyFilter(interactor)
    interactor.installEventFilter(key_filter)
    interactor.setProperty("education3DView", True)
    interactor._education_key_filter = key_filter
    return key_filter


@dataclass(frozen=True)
class EducationRunView:
    """Read-only dialog input; metadata is exposed through an immutable view."""

    phase: str
    run_id: str
    run_dir: Path
    status: str
    metadata: Mapping[str, Any]
    manager: Any

    @classmethod
    def from_run_dir(cls, run_dir: Path | str) -> "EducationRunView":
        """Read a saved metadata record without starting workflow or detection."""
        directory = Path(run_dir).expanduser().resolve()
        if not directory.is_dir() or directory.name in {"", ".", ".."}:
            raise ValueError("--run must name an existing run directory")
        phase_runs = directory.parent.name
        if not phase_runs.endswith("_runs"):
            raise ValueError("run directory must be inside a phaseN_runs folder")
        phase = phase_runs.removesuffix("_runs")
        if phase not in {f"phase{index}" for index in range(5)}:
            raise ValueError(f"unsupported education phase: {phase}")
        metadata_path = directory / "metadata.json"
        document = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or not document.get("run_id"):
            raise ValueError("run metadata is missing a run_id")
        run_id = str(document["run_id"])
        if run_id != directory.name:
            raise ValueError("metadata run_id does not match the run directory")
        # Avoid WorkspaceManager's normal constructor, which creates folders.
        from pyCamSet.workflow.workspace import WorkspaceManager
        from pyCamSet.workflow.education_trace import RunContext

        manager = WorkspaceManager()
        manager.workspace_path = directory.parent.parent
        context = RunContext.from_run(manager, {**document, "_run_dir": directory}, phase)
        return cls(context.phase, context.run_id, directory, context.status,
                   _freeze(context.metadata), manager)


def discover_phase_view(context: EducationRunView, parent: QWidget | None = None) -> QWidget:
    """Import one phase renderer only when requested, using a fixed factory name."""
    module = importlib.import_module(f"pyCamSet.gui.education_{context.phase}")
    factory = getattr(module, "create_education_phase_view")
    view = factory(context, parent=parent)
    if not isinstance(view, QWidget):
        raise TypeError("phase view factory must return a QWidget")
    return view


def _freeze(value: Any) -> Any:
    """Recursively freeze JSON containers so renderers cannot mutate run data."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


class EducationDialog(QDialog):
    """Single-run shell with chooser flow and a lazy phase-specific view."""

    #: Opening size where the screen allows: wide enough for Phase 4's two
    #: side-by-side 3D panels, tall enough for a stage's copy and its figure.
    _TARGET_SIZE = (1400, 900)  # the redesign's layout contract size (no-scroll stages)

    def __init__(self, runs: Sequence[EducationRunView], selected_run_ids: Sequence[str] = (),
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Opt-in, not an import side effect; a no-op when the user chose a cache or numba is loaded.
        from pyCamSet.gui.education_worker import configure_numba_cache
        configure_numba_cache()
        self.setWindowTitle("What’s Going On?")
        self.setMinimumSize(560, 420)
        self._runs = tuple(runs)
        self.setObjectName("educationDialog")
        self._application = QApplication.instance()
        # Theme changes arrive through the shared hub, not an application filter of this
        # dialog, so a dialog that is closed without closeEvent (Escape, done()) cannot leave
        # a dangling filter behind.
        from pyCamSet.gui.education_stage import ThemeWatcher
        self._theme_watcher = ThemeWatcher(self)
        self._theme_watcher.themeChanged.connect(lambda _name: self._apply_locked_theme())
        self._apply_locked_theme()
        self._active: EducationRunView | None = None
        self._view: QWidget | None = None
        self._stack = QStackedWidget(self)
        root = QVBoxLayout(self)
        root.addWidget(self._stack, 1)

        chooser = QWidget(self)
        chooser_layout = QVBoxLayout(chooser)
        chooser_layout.addWidget(QLabel("Choose one saved run to explain:", chooser))
        self._run_list = QListWidget(chooser)
        self._run_list.setAccessibleName("Saved calibration runs")
        self._run_list.setAccessibleDescription("Select exactly one run for its explanation.")
        for run in self._runs:
            self._run_list.addItem(f"{run.run_id} — {phase_name(run.phase)} — {run.status}")
        chooser_layout.addWidget(self._run_list, 1)
        self._open_button = QPushButton("Open explanation", chooser)
        self._open_button.setMinimumSize(96, 32)
        self._open_button.setAccessibleName("Open explanation for selected run")
        chooser_layout.addWidget(self._open_button, alignment=Qt.AlignmentFlag.AlignRight)
        self._open_button.clicked.connect(self._open_selection)
        self._stack.addWidget(chooser)

        self._detail = QWidget(self)
        detail_layout = QVBoxLayout(self._detail)
        header = QHBoxLayout()
        self._header_label = QLabel(self._detail)
        self._header_label.setWordWrap(True)
        self._header_label.setAccessibleName("Selected run")
        header.addWidget(self._header_label, 1)
        self._settings_button = self._make_settings_button(self._detail)
        header.addWidget(self._settings_button)
        self._change_button = QPushButton("Change run", self._detail)
        self._change_button.setMinimumSize(96, 32)
        self._change_button.clicked.connect(self._change_run)
        header.addWidget(self._change_button)
        detail_layout.addLayout(header)
        # Phase views are taller than many screens: scroll them rather than
        # letting Qt crush tables, plots and captions into the window height.
        self._phase_scroll = QScrollArea(self._detail)
        self._phase_scroll.setWidgetResizable(True)
        self._phase_scroll.setFrameShape(QFrame.Shape.NoFrame)
        # Transparent so phase content sits on the dialog's themed surface.
        self._phase_scroll.setObjectName("educationPhaseScroll")
        self._phase_scroll.viewport().setAutoFillBackground(False)
        self._phase_scroll.setAccessibleName("Explanation content")
        self._phase_container = QWidget()
        self._phase_container.setObjectName("educationPhaseContainer")
        self._phase_container.setAutoFillBackground(False)
        self._phase_layout = QVBoxLayout(self._phase_container)
        self._phase_layout.setContentsMargins(0, 0, 0, 0)
        self._phase_scroll.setWidget(self._phase_container)
        detail_layout.addWidget(self._phase_scroll, 1)
        self._stack.addWidget(self._detail)

        selected = [run for run in self._runs if run.run_id in set(selected_run_ids)]
        if len(selected) == 1:
            self._activate(selected[0])
        elif not self._runs:
            self._run_list.setEnabled(False)
            self._open_button.setEnabled(False)
            self._run_list.addItem("No saved runs are available.")
        else:
            self._run_list.setCurrentRow(0)
            self._stack.setCurrentIndex(0)

    def _make_settings_button(self, parent: QWidget) -> QToolButton:
        """Header "⚙ Settings" menu: the global playback preferences (UX-3: they are not
        part of any transport bar)."""
        button = QToolButton(parent)
        # UX3-17: the style's menu arrow sat under the last letter; draw our own "▾" in the
        # label with room on the right and hide the indicator.
        button.setText("⚙  Settings  ▾")
        button.setObjectName("educationSettingsButton")
        button.setStyleSheet("QToolButton#educationSettingsButton { padding: 4px 14px 4px 10px; }"
                             "QToolButton#educationSettingsButton::menu-indicator "
                             "{ image: none; width: 0px; }")
        button.setAccessibleName("Playback settings")
        button.setToolTip("Reduce motion and keyboard shortcuts for every step-through.")
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setMinimumHeight(32)
        menu = QMenu(button)
        self._reduce_motion_action = menu.addAction("Reduce motion")
        self._reduce_motion_action.setCheckable(True)
        self._reduce_motion_action.setToolTip(
            "Hide Play and skip animations; pictures change only when you step or drag.")
        self._shortcuts_action = menu.addAction("Keyboard shortcuts")
        self._shortcuts_action.setCheckable(True)
        self._shortcuts_action.setToolTip(shortcuts_help_text())
        menu.setToolTipsVisible(True)
        prefs = playback_preferences()

        def refresh() -> None:
            for action, value in ((self._reduce_motion_action, prefs.reduce_motion()),
                                  (self._shortcuts_action, prefs.shortcuts_enabled())):
                action.blockSignals(True)
                action.setChecked(value)
                action.blockSignals(False)

        refresh()
        menu.aboutToShow.connect(refresh)
        self._reduce_motion_action.toggled.connect(prefs.set_reduce_motion)
        self._shortcuts_action.toggled.connect(prefs.set_shortcuts_enabled)
        button.setMenu(menu)
        return button

    def _apply_locked_theme(self) -> None:
        """Set dialog chrome on the theme's actual surface token."""
        from pyCamSet.gui.theme import THEME_TOKENS

        theme = (self._application.property("pycamsetTheme")
                 if self._application is not None else None) or "Light"
        tokens = THEME_TOKENS.get(theme, THEME_TOKENS["Light"])
        self.setStyleSheet(
            f"QDialog#educationDialog {{ background-color: {tokens['surface']}; "
            f"color: {tokens['text']}; }}\n"
            f"QDialog#educationDialog QLabel {{ color: {tokens['text']}; }}\n"
            "QScrollArea#educationPhaseScroll, QWidget#educationPhaseContainer "
            "{ background: transparent; }")

    def _open_selection(self) -> None:
        index = self._run_list.currentRow()
        if 0 <= index < len(self._runs):
            self._activate(self._runs[index])

    def _cancel_child_preparation(self) -> None:
        """Make any phase view's background preparation stale (the dialog is closing or the
        view is being replaced). The worker thread finishes on its own; nothing is shown."""
        for child in self.findChildren(QWidget):
            cancel = getattr(child, "close_background", None)
            if callable(cancel):
                cancel()

    def _stop_child_playback(self) -> None:
        for child in self.findChildren(QWidget):
            stop = getattr(child, "stop_for_context_change", None)
            if not callable(stop):
                stop = getattr(child, "stop_playback", None)
            if callable(stop):
                stop()

    def _change_run(self) -> None:
        self._stop_child_playback()
        if self._active in self._runs:
            self._run_list.setCurrentRow(self._runs.index(self._active))
        self._stack.setCurrentIndex(0)

    def _activate(self, run: EducationRunView) -> None:
        self._stop_child_playback()
        self._active = run
        self._header_label.setText(f"Run {run.run_id} · {phase_name(run.phase)} · {run.status}")
        while self._phase_layout.count():
            item = self._phase_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                cancel = getattr(widget, "close_background", None)
                if callable(cancel):
                    cancel()  # a replaced view must not deliver its stale result
                widget.hide()  # gone at once, not when the deferred delete runs
                widget.deleteLater()
        from pyCamSet.gui import education_scene3d
        education_scene3d.forget_backend_choice()  # a note only for this run's 3D scene
        try:
            self._view = discover_phase_view(run, self._phase_container)
        except ModuleNotFoundError as exc:
            if exc.name != f"pyCamSet.gui.education_{run.phase}":
                raise
            self._view = QLabel(
                f"The Phase {run.phase.removeprefix('phase')} explanation is not installed yet.",
                self._phase_container)
            self._view.setWordWrap(True)
        self._phase_layout.addWidget(self._view)
        note = education_scene3d.backend_note()
        if note:  # one line in the header: the 3D picture is static, and how to opt in
            self._header_label.setText(f"{self._header_label.text()}\n{note}")
        self._stack.setCurrentWidget(self._detail)
        if not self.isVisible():
            self._fit_to_screen()
        event = QAccessibleAnnouncementEvent(
            self, f"Showing {phase_name(run.phase)} explanation for run {run.run_id}.")
        event.setPoliteness(QAccessible.AnnouncementPoliteness.Polite)
        QAccessible.updateAccessibility(event)

    def _fit_to_screen(self) -> None:
        """Open at a reading size, never below the content's hint, capped at 90% of the screen.

        A phase view's size hint is its *minimum* comfortable layout (wrapped
        labels, a 400 px canvas), so opening at it leaves figures cramped on any
        screen; open at least ``_TARGET_SIZE`` where the screen allows.
        """
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None or self._view is None:
            return
        available = screen.availableGeometry()
        hint = self._view.sizeHint()
        margins = 48  # header row, layout margins and the vertical scroll bar
        target_width, target_height = self._TARGET_SIZE
        width = min(max(self.minimumWidth(), hint.width() + margins, target_width),
                    int(available.width() * 0.9))
        height = min(max(self.minimumHeight(), hint.height() + 2 * margins, target_height),
                     int(available.height() * 0.9))
        self.resize(max(width, self.minimumWidth()), max(height, self.minimumHeight()))

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API name
        """First Escape returns focus to a safe control; a second closes."""
        if event.key() == Qt.Key.Key_Escape:
            if self._change_button.hasFocus():
                self.close()
            else:
                self._change_button.setFocus(Qt.FocusReason.ShortcutFocusReason)
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        """Stop child playback/derivation hooks before the shell disappears."""
        self._stop_child_playback()
        self._cancel_child_preparation()
        self._theme_watcher.detach()
        super().closeEvent(event)

    def done(self, result: int) -> None:  # noqa: N802 - Qt API name
        """Escape and the close buttons end an exec() here, not in closeEvent."""
        self._stop_child_playback()
        self._cancel_child_preparation()
        self._theme_watcher.detach()
        super().done(result)


def main(argv: Sequence[str] | None = None) -> int:
    """Open one run directory for a developer pilot; never starts calibration."""
    parser = argparse.ArgumentParser(description="Open a saved run in the education-dialog pilot")
    parser.add_argument("--run", required=True, type=Path, help="existing phaseN_runs/<run_id> directory")
    args = parser.parse_args(argv)
    from pyCamSet.gui.education_perf import configure_surface_format
    configure_surface_format()
    application = QApplication.instance() or QApplication(sys.argv[:1])
    run = EducationRunView.from_run_dir(args.run)
    dialog = EducationDialog([run], [run.run_id])
    dialog.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
