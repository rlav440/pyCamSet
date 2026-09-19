"""The ``pycamset`` console script.

PySide6 is a base dependency, but the lean install documented in
requirements_core.txt leaves it out, so these tests cover the part of the
entry point that must work either way: the launch call, and the message
someone on the lean install gets instead.
"""
from __future__ import annotations

import sys

import pytest

from pyCamSet.gui.__main__ import main


def test_it_launches_the_window(monkeypatch):
    called = []

    module = type(sys)("pyCamSet.gui.main_window")
    module.main_window = lambda: called.append(True)
    monkeypatch.setitem(sys.modules, "pyCamSet.gui.main_window", module)

    assert main() == 0
    assert called == [True]


def test_a_missing_gui_extra_explains_itself(monkeypatch, capsys):
    """Setting a sys.modules entry to None makes importing it raise.

    A toolkit that is genuinely absent raises ModuleNotFoundError, and adding
    it is the right advice.
    """
    monkeypatch.setitem(sys.modules, "pyCamSet.gui.main_window", None)

    assert main() == 1

    message = capsys.readouterr().err
    assert "pip install PySide6" in message


def test_an_installed_but_unloadable_toolkit_gets_different_advice(capsys):
    """A DLL-load failure must NOT be answered with 'pip install PySide6'.

    This is the misdiagnosis that cost a session: PySide6 was installed, its
    Qt6Core would not load against a competing Qt, and the message sent the
    reader to a command that answers "requirement already satisfied" and
    changes nothing.
    """
    from pyCamSet.gui.__main__ import advice_for

    failure = ImportError(
        "DLL load failed while importing QtCore: "
        "The specified procedure could not be found.")
    message = advice_for(failure)

    assert "DLL load failed" in message
    assert "pip install PySide6" not in message
    # It must point at the actual cause: another Qt on the search path.
    assert "Qt" in message


def test_advice_distinguishes_the_two_failure_kinds():
    """The two cases are told apart by exception type, not by message text."""
    from pyCamSet.gui.__main__ import advice_for

    missing = advice_for(ModuleNotFoundError("No module named 'PySide6'"))
    broken = advice_for(ImportError("DLL load failed while importing QtCore"))

    assert missing != broken
    assert "pip install PySide6" in missing
    assert "pip install PySide6" not in broken


def test_the_console_script_is_declared():
    """The entry point setup.cfg promises must resolve to a callable."""
    from importlib.metadata import entry_points

    scripts = entry_points(group="console_scripts")
    pycamset = [e for e in scripts if e.name == "pycamset"]
    if not pycamset:
        pytest.skip("pyCamSet is not installed in this environment")

    assert pycamset[0].value == "pyCamSet.gui.__main__:main"
    assert callable(pycamset[0].load())
