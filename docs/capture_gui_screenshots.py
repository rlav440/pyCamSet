"""Regenerate the GUI guide's screenshots.

Run from the repository root, in an environment with the GUI installed::

    python docs/capture_gui_screenshots.py

Widgets are rendered with ``QWidget.grab()`` rather than captured from the
screen.  That renders the window into an offscreen buffer, so it works the
same under Wayland, X11 and a headless runner, needs no compositor
cooperation, and never catches a stray notification or cursor.

Two things are pinned so that the output does not depend on the machine that
made it.  The platform plugin is forced to ``offscreen``, and the style and
palette are set explicitly -- left alone, Qt follows the desktop theme, and the
same script produces light screenshots on one machine and dark ones on the
next.  Each tab is therefore captured twice, once per theme, and the guide
shows whichever matches the reader's:

.. code-block:: rst

   .. image:: /_static/gui/phase-1--detection-light.png
      :class: only-light
   .. image:: /_static/gui/phase-1--detection-dark.png
      :class: only-dark

Setting ``QT_QPA_PLATFORM`` yourself overrides the offscreen default, and
the rendering will differ; leave it unset to reproduce the committed images.

Images land in ``docs/source/_static/gui/``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Must be set before QApplication is constructed.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "source" / "_static" / "gui"

# The window is grabbed at this size so the screenshots are a consistent shape.
WINDOW_SIZE = (1280, 860)


def _slug(tab_label: str) -> str:
    """Turn a tab's label into a stable file name."""
    out = "".join(c.lower() if c.isalnum() else "-" for c in tab_label)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


def _dark_palette():
    """The standard Fusion dark palette, so dark mode does not need a theme."""
    from PySide6.QtGui import QColor, QPalette

    p = QPalette()
    window, base, text = QColor(53, 53, 53), QColor(35, 35, 35), QColor(220, 220, 220)
    for role, colour in [
        (QPalette.Window, window),
        (QPalette.WindowText, text),
        (QPalette.Base, base),
        (QPalette.AlternateBase, window),
        (QPalette.ToolTipBase, window),
        (QPalette.ToolTipText, text),
        (QPalette.Text, text),
        (QPalette.Button, window),
        (QPalette.ButtonText, text),
        (QPalette.BrightText, QColor(255, 80, 80)),
        (QPalette.Link, QColor(90, 160, 240)),
        (QPalette.Highlight, QColor(60, 115, 175)),
        (QPalette.HighlightedText, QColor(255, 255, 255)),
    ]:
        p.setColor(role, colour)
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(130, 130, 130))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(130, 130, 130))
    return p


def _capture(app, suffix: str) -> list[str]:
    """Build a fresh window and grab every visible tab."""
    from PySide6.QtWidgets import QTabWidget

    from pyCamSet.gui.main_window import PyCamSetApp

    window = PyCamSetApp()
    window.resize(*WINDOW_SIZE)
    window.show()
    app.processEvents()

    tabs = window.findChild(QTabWidget)
    if tabs is None:
        raise SystemExit("could not find the phase tab widget")

    written = []
    for index in range(tabs.count()):
        # The diagnostics companions only appear once their phase has run.
        if not tabs.isTabVisible(index):
            continue
        tabs.setCurrentIndex(index)
        app.processEvents()

        name = f"{_slug(tabs.tabText(index))}-{suffix}.png"
        if not window.grab().save(str(OUT_DIR / name)):
            raise SystemExit(f"failed to write {name}")
        written.append(name)

    window.close()
    app.processEvents()
    return written


def main() -> int:
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QApplication, QStyleFactory

    from pyCamSet.utils.report_format import force_colour

    force_colour(True)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    for suffix, palette in [("light", QPalette()), ("dark", _dark_palette())]:
        app.setPalette(palette)
        names = _capture(app, suffix)
        total += len(names)
        print(f"  {suffix}: {len(names)} tabs")

    print(f"{total} screenshots in {OUT_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
