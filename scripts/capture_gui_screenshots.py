"""Regenerate the GUI guide's screenshots.

Run from the repository root, in an environment with the GUI installed::

    python scripts/capture_gui_screenshots.py

Widgets are rendered with ``QWidget.grab()`` rather than captured from the
screen.  That renders the window into an offscreen buffer, so it works the
same under Wayland, X11 and a headless runner, needs no compositor
cooperation, and never catches a stray notification or cursor.

Three things are pinned so that the output does not depend on the machine
that made it.  The platform plugin defaults to ``offscreen``; the application's
own Light and Dark themes (``pyCamSet.gui.theme``) are applied explicitly,
overriding whatever theme the person running the script last chose; and the
window reads a throwaway configuration directory, so no one's saved folders
or parameters appear in the documentation.  Each tab is captured twice, once
per theme, and the guide shows whichever matches the reader's:

::

   ![The Phase 1 tab.](../assets/gui/phase-1-detection-light.png#only-light)
   ![The Phase 1 tab.](../assets/gui/phase-1-detection-dark.png#only-dark)

Setting ``QT_QPA_PLATFORM`` yourself overrides the offscreen default.  On
Windows the offscreen plugin has no font database and draws every label as an
empty box; there, run with ``QT_QPA_PLATFORM=windows`` -- windows are still
rendered with ``grab()`` and never shown on screen.

Images land in ``docs/assets/gui/``.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Must be set before QApplication is constructed.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Read and write preferences in a scratch directory, never the user's own.
os.environ.setdefault("PYCAMSET_CONFIG_DIR", tempfile.mkdtemp(prefix="pycamset-screenshots-"))

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "assets" / "gui"

# The window is grabbed at this size so the screenshots are a consistent shape.
WINDOW_SIZE = (1280, 860)


def _slug(tab_label: str) -> str:
    """Turn a tab's label into a stable file name."""
    out = "".join(c.lower() if c.isalnum() else "-" for c in tab_label)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


def _capture_create_target(app, suffix: str) -> str:
    """Grab the Create Target dialog, which the corner button opens."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QCheckBox

    from pyCamSet.gui.create_target import CreateTargetDialog

    terminal_cb = QCheckBox("Show Terminal Output")
    terminal_cb.setChecked(True)

    dialog = CreateTargetDialog(terminal_cb=terminal_cb, parent=None)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    # The field defaults to the working directory, which would put whoever ran
    # this script's home directory in the documentation.
    dialog._out_dir_edit.setText("/path/to/targets")
    dialog.show()
    app.processEvents()

    name = f"create-target-{suffix}.png"
    if not dialog.grab().save(str(OUT_DIR / name)):
        raise SystemExit(f"failed to write {name}")
    dialog.close()
    app.processEvents()
    return name


def _capture(app, suffix: str, theme_name: str) -> list[str]:
    """Build a fresh window in *theme_name* and grab every visible tab."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QTabWidget

    from pyCamSet.gui.main_window import PyCamSetApp
    from pyCamSet.gui.theme import apply_theme, refresh_matplotlib_theme

    window = PyCamSetApp()
    # The window applies the last saved theme; the documented one wins.
    apply_theme(app, theme_name)
    refresh_matplotlib_theme(theme_name)
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
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
    from PySide6.QtWidgets import QApplication

    from pyCamSet.gui.theme import apply_theme
    from pyCamSet.utils.report_format import force_colour

    force_colour(True)
    app = QApplication.instance() or QApplication(sys.argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    for suffix, theme_name in [("light", "Light"), ("dark", "Dark")]:
        names = _capture(app, suffix, theme_name)
        apply_theme(app, theme_name)
        names.append(_capture_create_target(app, suffix))
        total += len(names)
        print(f"  {suffix}: {len(names)} images")

    print(f"{total} screenshots in {OUT_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
