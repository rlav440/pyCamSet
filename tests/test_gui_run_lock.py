"""A phase's Run button can start one run at a time."""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(qapp, worker):
    worker.wait(10_000)
    for _ in range(50):
        qapp.processEvents()


@pytest.mark.parametrize("fails", [False, True])
def test_run_button_is_held_until_the_run_ends(qapp, fails):
    from PySide6.QtWidgets import QPushButton

    from pyCamSet.gui.shared_functions import PhaseWorker, hold_run_button

    gate = threading.Event()

    def work(_log):
        gate.wait(5)
        if fails:
            raise RuntimeError("boom")
        return {}

    button = QPushButton("▶  Run Phase 1")
    worker = PhaseWorker(work)
    hold_run_button(button, worker)
    worker.start()
    try:
        assert not button.isEnabled() and button.text() == "Running…"
        # Clicking a disabled button does nothing.
        clicks = []
        button.clicked.connect(lambda: clicks.append(1))
        button.click()
        assert clicks == []
    finally:
        gate.set()
        _wait(qapp, worker)
    assert button.isEnabled() and button.text() == "▶  Run Phase 1"


def test_run_button_stays_held_until_the_last_holder_ends(qapp):
    from PySide6.QtWidgets import QPushButton

    from pyCamSet.gui.shared_functions import PhaseWorker, hold_run_button, run_button_held

    first_gate, second_gate = threading.Event(), threading.Event()
    button = QPushButton("▶  Run Phase 3")
    first = PhaseWorker(lambda _log: first_gate.wait(5) and {})
    second = PhaseWorker(lambda _log: second_gate.wait(5) and {})
    hold_run_button(button, first)
    hold_run_button(button, second)
    first.start()
    second.start()
    try:
        first_gate.set()
        _wait(qapp, first)
        assert run_button_held(button)
        assert not button.isEnabled() and button.text() == "Running…"
    finally:
        first_gate.set()
        second_gate.set()
        _wait(qapp, second)
    assert not run_button_held(button)
    assert button.isEnabled() and button.text() == "▶  Run Phase 3"


@pytest.mark.parametrize("tab_name", ["phase1_tab", "phase2_tab", "phase3_tab"])
def test_every_phase_exposes_its_run_button(qapp, tab_name, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCAMSET_CONFIG_DIR", str(tmp_path))
    from pyCamSet.gui.main_window import PyCamSetApp

    window = PyCamSetApp()
    try:
        button = getattr(window, tab_name)._run_btn
        assert button.text().startswith("▶  Run Phase")
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
