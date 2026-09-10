"""
Refusing to open a native window inside somebody else's event loop.

pyvista draws through VTK, and VTK's Cocoa render window runs
``[NSRunLoop runUntilDate:]`` while it is open.  Inside a Qt application
that nested loop re-enters Qt's event delivery and repaints the widget
tree from inside an event Qt has not finished dispatching, and Qt dies in
``QMacCGContext`` -- either there and then, or on whichever repaint comes
next.  A segmentation fault carries no traceback, so nothing points at
the call that caused it.

This turns that into an exception that names the call.  It has caught the
same mistake three times in this codebase; the fourth should not have to
be diagnosed from a crash report.

The GUI never trips it: every window it wants goes to a separate process
through :mod:`pyCamSet.gui.viewer_process`.  Set
``PYCAMSET_ALLOW_WINDOWS_IN_QT=1`` to proceed anyway, for anyone
embedding pyCamSet in a Qt application of their own who has arranged
things differently.
"""
from __future__ import annotations

import os
import sys

#: Only macOS is known to fault this way; on X11 and Windows a second
#: native loop is merely rude, and refusing there would break setups that
#: work today.
_ENFORCED_PLATFORMS = ("darwin",)

_OVERRIDE = "PYCAMSET_ALLOW_WINDOWS_IN_QT"


def qt_application_is_running() -> bool:
    """
    Whether a Qt application owns this process's event loop.

    Only asks Qt if Qt is already imported: importing it here to find out
    would be both slow and, in a headless process, harmful.

    :return: whether a ``QGuiApplication`` instance exists
    """
    for module_name in ("PySide6.QtWidgets", "PySide6.QtGui"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        application = getattr(module, "QApplication", None) or getattr(
            module, "QGuiApplication", None)
        if application is None:
            continue
        try:
            if application.instance() is not None:
                return True
        except Exception:
            continue
    return False


def refuse_window_inside_qt(what: str) -> None:
    """
    Raise rather than open a native window under a Qt event loop.

    :param what: the call being refused, for the message
    :raises RuntimeError: when a Qt application is running
    """
    if os.environ.get(_OVERRIDE):
        return
    if sys.platform not in _ENFORCED_PLATFORMS:
        return
    if not qt_application_is_running():
        return
    raise RuntimeError(
        f"{what} would open a native window inside a running Qt "
        f"application. On macOS the render window runs its own event loop, "
        f"which re-enters Qt's and crashes it in QMacCGContext.\n"
        f"Run the drawing in its own process -- see "
        f"pyCamSet.gui.viewer_process.spawn_viewer, and the viewers in "
        f"pyCamSet.utils.visualise_camset and "
        f"pyCamSet.utils.visualise_target -- or render off screen.\n"
        f"Set {_OVERRIDE}=1 to proceed anyway."
    )
