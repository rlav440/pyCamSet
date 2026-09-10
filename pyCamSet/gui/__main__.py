"""Command-line entry point for the pyCamSet GUI.

Installed as the ``pycamset`` console script, and reachable as
``python -m pyCamSet.gui``.  PySide6 is a base dependency, so a normal
install has it -- but the lean install documented in requirements-core.txt
deliberately does not, and the import of the window therefore sits inside
:func:`main`.  Someone on that install should get a sentence telling them
what to add, not an import traceback.
"""

from __future__ import annotations

import sys


def main() -> int:
    """
    Launch the GUI, or explain why it could not start.

    :return: the process exit status
    """
    try:
        from pyCamSet.gui.main_window import main_window
    except ImportError as exc:
        print(
            f"The pyCamSet GUI could not start: {exc}.\n"
            "This is the lean install, which leaves the GUI toolkit out.\n"
            "Add it with:\n"
            "    pip install PySide6",
            file=sys.stderr,
        )
        return 1

    # main_window() runs the Qt event loop and exits the process itself.  The
    # return below is for the day that stops being true.
    main_window()
    return 0


if __name__ == "__main__":
    sys.exit(main())
