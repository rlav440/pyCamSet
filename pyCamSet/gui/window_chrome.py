'''Purpose: Make the native window title bar follow the selected GUI theme.
Status: Active; exact colours on Windows 11, light/dark on Windows 10 and macOS.
Future: Add a platform only through a documented window-manager interface.
'''
from __future__ import annotations

import ctypes
import sys
from collections.abc import Mapping

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QApplication, QWidget

# Desktop Window Manager attributes (dwmapi.h).  Windows 11 22000+ honours the
# caption, text and border colours; Windows 10 1809+ only the dark-mode flag
# (19 before build 18985, 20 after).  Unsupported attributes simply fail.
_DWMWA_USE_IMMERSIVE_DARK_MODE = (20, 19)
_DWMWA_BORDER_COLOR = 34
_DWMWA_CAPTION_COLOR = 35
_DWMWA_TEXT_COLOR = 36


def _colorref(hex_colour: str) -> int:
    """#rrggbb as a Win32 COLORREF (0x00bbggrr)."""
    red, green, blue = (int(hex_colour[index:index + 2], 16) for index in (1, 3, 5))
    return red | (green << 8) | (blue << 16)


def _dwm_set_attribute(hwnd: int, attribute: int, value: int) -> bool:
    """Set one DWM attribute; False where the platform or build lacks it."""
    try:
        dwmapi = ctypes.windll.dwmapi  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return False
    data = ctypes.c_int(value)
    result = dwmapi.DwmSetWindowAttribute(ctypes.c_void_p(hwnd), ctypes.c_uint(attribute),
                                          ctypes.byref(data), ctypes.sizeof(data))
    return result == 0


def set_colour_scheme(application, dark: bool) -> bool:
    """Ask Qt for a light or dark native frame (Qt 6.8+; Windows, macOS and some Linux desktops)."""
    hints = application.styleHints()
    if not hasattr(hints, "setColorScheme"):
        return False
    wanted = Qt.ColorScheme.Dark if dark else Qt.ColorScheme.Light
    if hints.colorScheme() != wanted:
        hints.setColorScheme(wanted)
    return True


def apply_window_chrome(window: QWidget, tokens: Mapping[str, str], dark: bool) -> bool:
    """Colour one top-level window's title bar; True if exact colours were applied.

    On Windows the caption takes the menu bar's surface colour, with the
    theme's text and hairline border, so the two read as one strip.  Where
    exact colours are unavailable the dark-mode flag still applies, and on
    other platforms the Qt colour scheme (see set_colour_scheme) governs the
    frame.  Failures are silent: the title bar is decoration, never state.
    """
    from PySide6.QtGui import QGuiApplication

    # Only a real Windows desktop has a DWM frame; the offscreen plugin used
    # by tests must not be made to create native window handles.
    if sys.platform != "win32" or QGuiApplication.platformName() != "windows" or not window.isWindow():
        return False
    hwnd = int(window.winId())
    for attribute in _DWMWA_USE_IMMERSIVE_DARK_MODE:
        if _dwm_set_attribute(hwnd, attribute, 1 if dark else 0):
            break
    exact = _dwm_set_attribute(hwnd, _DWMWA_CAPTION_COLOR, _colorref(tokens["surface"]))
    if exact:
        _dwm_set_attribute(hwnd, _DWMWA_TEXT_COLOR, _colorref(tokens["text"]))
        _dwm_set_attribute(hwnd, _DWMWA_BORDER_COLOR, _colorref(tokens["border"]))
    return exact


class _ChromeOnShow(QObject):
    """Colour each window's title bar as it first appears (dialogs included)."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if event.type() == QEvent.Type.Show and isinstance(obj, QWidget) and obj.isWindow():
            _apply_current(obj)
        return False


def _apply_current(window: QWidget) -> None:
    from pyCamSet.gui.theme import THEME_TOKENS

    application = QApplication.instance()
    theme = (application.property("pycamsetTheme") if application else None) or "Light"
    if theme in THEME_TOKENS:
        apply_window_chrome(window, THEME_TOKENS[theme], theme == "Dark")


def refresh_window_chrome(application, theme_name: str) -> None:
    """Follow a theme change: frame scheme, every open window, and future ones."""
    set_colour_scheme(application, theme_name == "Dark")
    if not getattr(application, "_pycamset_chrome_filter", None):
        chrome_filter = _ChromeOnShow(application)
        application.installEventFilter(chrome_filter)
        application._pycamset_chrome_filter = chrome_filter
    for window in application.topLevelWidgets():
        if window.isVisible():
            _apply_current(window)
