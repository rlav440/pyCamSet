"""Command-line entry point for the pyCamSet GUI.

Installed as the ``pycamset`` console script, and reachable as
``python -m pyCamSet.gui``.  PySide6 is a base dependency, so a normal
install has it -- but the lean install documented in requirements_core.txt
deliberately does not, and the import of the window therefore sits inside
:func:`main`.  Someone on that install should get a sentence telling them
what to add, not an import traceback.

Two failures reach that handler and they need different advice.  A toolkit that
is not installed raises :class:`ModuleNotFoundError`, and adding it is the fix.
A toolkit that IS installed but cannot load -- the commonest being a second copy
of Qt on the DLL search path, which is what a conda environment's
``Library/bin/Qt6Core.dll`` is to a pip-installed PySide6 -- raises a plain
:class:`ImportError` reading "DLL load failed".  Telling that person to
``pip install PySide6`` sends them to a command that answers "requirement
already satisfied" and changes nothing, so the two are told apart here.
"""

from __future__ import annotations

import sys


def advice_for(exc: ImportError) -> str:
    """What to tell someone whose GUI did not start.

    :param exc: the ``ImportError`` raised by importing the window
    :return: the explanation, as text to print
    """
    if isinstance(exc, ModuleNotFoundError):
        return (
            f"The pyCamSet GUI could not start: {exc}.\n"
            "This is the lean install, which leaves the GUI toolkit out.\n"
            "Add it with:\n"
            "    pip install PySide6"
        )

    # An installed toolkit that will not load is almost always a library
    # conflict rather than a missing install, and the advice above is actively
    # misleading here: pip reports PySide6 already satisfied and the same
    # failure returns.
    return (
        f"The pyCamSet GUI could not start: {exc}.\n"
        "\n"
        "PySide6 is installed but its Qt libraries did not load. That is usually\n"
        "a second copy of Qt being found first -- a conda environment's\n"
        "Library\\bin is a common one, and it is not ABI-compatible with a\n"
        "pip-installed PySide6.\n"
        "\n"
        "Find the PySide6 package this run is using, and check for a competing\n"
        "Qt outside it:\n"
        "    python -c \"import PySide6, pathlib; print(pathlib.Path(PySide6.__file__).parent)\"\n"
        "\n"
        "Then remove the competing Qt from that environment, or install PySide6\n"
        "into an environment that has no other Qt in it."
    )


def main() -> int:
    """
    Launch the GUI, or explain why it could not start.

    :return: the process exit status
    """
    try:
        from pyCamSet.gui.main_window import main_window
    except ImportError as exc:
        print(advice_for(exc), file=sys.stderr)
        return 1

    # main_window() runs the Qt event loop and exits the process itself.  The
    # return below is for the day that stops being true.
    main_window()
    return 0


if __name__ == "__main__":
    sys.exit(main())
