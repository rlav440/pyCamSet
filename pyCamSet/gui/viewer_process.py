"""
Starting the viewers that cannot share a process with the GUI.

pyvista draws through VTK, and VTK's Cocoa render window runs
``[NSRunLoop runUntilDate:]`` for as long as it is open.  Started from a Qt
slot, that nested loop re-enters Qt's event delivery and repaints the
widget tree from inside an event Qt has not finished dispatching; Qt has
no valid graphics context there and dies in ``QMacCGContext``.  The crash
lands either immediately, under ``vtkCocoaRenderWindow::Render()``, or on
whichever repaint comes next once the window has been and gone.

matplotlib's ``plt.show()`` asks for the same thing and merely says so:
"the event loop is already running".

None of this is a fault in the drawing code.  Both are asking for an event
loop the GUI already owns, so they get a process where they can have one.
"""
from __future__ import annotations

import logging
import subprocess
import sys

logger = logging.getLogger(__name__)

# Kept only so the viewers are not garbage collected into zombies while
# they are still on screen; nothing waits on them.
_VIEWER_PROCESSES: list[subprocess.Popen] = []


def spawn_viewer(module: str, arguments: list[str]) -> tuple[bool, str]:
    """
    Run one of the viewer modules in a process of its own.

    :param module: the module to run, as ``python -m`` would take it
    :param arguments: its command line
    :return: whether it started, and what to say if it did not
    """
    command = [sys.executable, "-m", module, *arguments]
    try:
        # not waited on: the viewer owns its window for as long as the
        # person wants it, and the GUI carries on meanwhile
        process = subprocess.Popen(command)
    except OSError as exc:
        logger.debug("Could not start %s: %s", module, exc)
        return False, f"Could not start the viewer: {exc}"

    _VIEWER_PROCESSES.append(process)
    reap_finished_viewers()
    return True, ""


def reap_finished_viewers() -> None:
    """Drop viewers that have exited, so they are not left as zombies."""
    for process in list(_VIEWER_PROCESSES):
        if process.poll() is not None:
            _VIEWER_PROCESSES.remove(process)


def run_viewer(module: str, arguments: list[str],
               timeout: float = 300.0) -> tuple[bool, str]:
    """
    Run a viewer module and wait for it.

    For the work that produces a file rather than a window: the caller
    wants the result, and it finishes on its own.  It still runs out of
    process, because off screen rendering on macOS builds a Cocoa render
    window just the same.

    :param module: the module to run, as ``python -m`` would take it
    :param arguments: its command line
    :param timeout: seconds to wait before giving up on it
    :return: whether it succeeded, and what it said if it did not
    """
    command = [sys.executable, "-m", module, *arguments]
    try:
        finished = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"The renderer did not finish within {timeout:.0f}s."
    except OSError as exc:
        return False, f"Could not start the renderer: {exc}"

    if finished.returncode != 0:
        detail = (finished.stderr or finished.stdout or "").strip()
        return False, detail or f"The renderer exited with {finished.returncode}."
    return True, (finished.stdout or "").strip()
