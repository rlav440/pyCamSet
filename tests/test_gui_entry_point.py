"""The ``pycamset`` console script.

PySide6 is a base dependency, but the lean install documented in
requirements-core.txt leaves it out, so these tests cover the part of the
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
    """Setting a sys.modules entry to None makes importing it raise."""
    monkeypatch.setitem(sys.modules, "pyCamSet.gui.main_window", None)

    assert main() == 1

    message = capsys.readouterr().err
    assert "pip install PySide6" in message


def test_the_console_script_is_declared():
    """The entry point setup.cfg promises must resolve to a callable."""
    from importlib.metadata import entry_points

    scripts = entry_points(group="console_scripts")
    pycamset = [e for e in scripts if e.name == "pycamset"]
    if not pycamset:
        pytest.skip("pyCamSet is not installed in this environment")

    assert pycamset[0].value == "pyCamSet.gui.__main__:main"
    assert callable(pycamset[0].load())
