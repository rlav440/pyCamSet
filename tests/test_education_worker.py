'''Verify the education worker (keyed, stale-discarding, cancellable background jobs) and
that the Phase 3 and Phase 4 views open at once with a 'Preparing' state, then fill in.
derivation with an event so the busy state can be seen before the stages are ready.
'''
from __future__ import annotations

import os
import sys
import threading
import time

import pytest


def _app():
    pytest.importorskip("PySide6")
    os.environ.setdefault("PYCAMSET_EDU_3D", "static")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class _Recorder:
    """Collects the worker's signals; the GUI thread (this test) reads them."""

    def __init__(self, worker):
        self.ready: list[tuple] = []
        self.failed: list[tuple] = []
        self.status: list[tuple] = []
        worker.ready.connect(lambda key, value: self.ready.append((key, value)))
        worker.failed.connect(lambda key, message: self.failed.append((key, message)))
        worker.status.connect(lambda key, text: self.status.append((key, text)))


def _worker():
    _app()
    from pyCamSet.gui.education_worker import EducationWorker
    return EducationWorker()


@pytest.mark.gui
def test_result_is_delivered_on_the_gui_thread_and_the_worker_goes_idle():
    worker = _worker()
    recorder = _Recorder(worker)
    seen_threads = []

    def work(report):
        seen_threads.append(threading.current_thread())
        report("halfway")
        return 42

    worker.submit("job", work)
    assert worker.is_busy()
    assert worker.wait_until_idle(10)
    assert not worker.is_busy()
    assert recorder.ready == [("job", 42)]
    assert recorder.failed == []
    assert recorder.status == [("job", "halfway")]
    assert seen_threads and seen_threads[0] is not threading.main_thread()


@pytest.mark.gui
def test_a_newer_submit_discards_the_stale_result():
    worker = _worker()
    recorder = _Recorder(worker)
    release = threading.Event()

    def slow(_report):
        release.wait(10)
        return "stale"

    worker.submit("first", slow)
    worker.submit("second", lambda _report: "fresh")
    assert worker.wait_until_idle(10)
    release.set()
    time.sleep(0.05)
    assert worker.wait_until_idle(5)
    assert recorder.ready == [("second", "fresh")]
    assert all(key != "first" for key, _ in recorder.ready)


@pytest.mark.gui
def test_close_cancels_cleanly_and_nothing_is_delivered_afterwards():
    from pyCamSet.gui.education_worker import wait_for_background_threads
    worker = _worker()
    recorder = _Recorder(worker)
    release = threading.Event()

    def blocked(_report):
        release.wait(10)
        return "late"

    worker.submit("job", blocked)
    worker.close()  # must not raise, must not wait for the thread
    assert not worker.is_busy()
    release.set()
    assert wait_for_background_threads(10)
    assert worker.wait_until_idle(2)
    assert recorder.ready == [] and recorder.failed == []


@pytest.mark.gui
def test_an_exception_is_surfaced_as_a_message_not_a_crash():
    worker = _worker()
    recorder = _Recorder(worker)

    def broken(_report):
        raise ValueError("saved run is missing its camera file")

    worker.submit("job", broken)
    assert worker.wait_until_idle(10)
    assert recorder.ready == []
    assert recorder.failed == [("job", "saved run is missing its camera file")]


@pytest.mark.gui
def test_wait_until_idle_reports_a_timeout_while_a_job_is_blocked():
    worker = _worker()
    release = threading.Event()
    worker.submit("job", lambda _report: release.wait(10))
    assert worker.wait_until_idle(0.05) is False
    assert worker.is_busy()
    release.set()
    assert worker.wait_until_idle(10)


@pytest.mark.gui
def test_numba_cache_folder_is_per_user_and_respects_a_chosen_one(tmp_path, monkeypatch):
    _app()
    from pyCamSet.gui import education_worker

    monkeypatch.setattr(education_worker.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setenv("NUMBA_CACHE_DIR", "")  # empty counts as unset
    monkeypatch.delitem(sys.modules, "numba", raising=False)
    folder = education_worker.configure_numba_cache()
    assert folder == tmp_path / ".cache" / "pycamset" / "numba"
    assert folder.is_dir()
    assert os.environ["NUMBA_CACHE_DIR"] == str(folder)

    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "mine"))
    assert education_worker.configure_numba_cache() is None
    assert os.environ["NUMBA_CACHE_DIR"] == str(tmp_path / "mine")


def _gated_result(monkeypatch, module, name, gate, result):
    """Replace a derivation with one that waits for *gate*, so the busy state can be seen."""
    def gated(*_args, **_kwargs):
        gate.wait(15)
        return result
    monkeypatch.setattr(module, name, gated)


def _caption(view) -> str:
    return view.stage_view.caption.text()


@pytest.mark.gui
def test_phase4_view_opens_at_once_showing_the_preparing_state_then_fills_in(
        tmp_path, monkeypatch, education_view_construction):
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.workflow.education_derive import DerivationUnavailable
    from test_education_phase4 import _context

    gate = threading.Event()
    _gated_result(monkeypatch, education_phase4, "derive_phase4", gate,
                  DerivationUnavailable("test derivation held back"))
    started = time.perf_counter()
    view = education_phase4.Phase4View(_context(tmp_path))
    opened = time.perf_counter() - started
    try:
        # Mechanism, not speed: the held-back derivation waits up to 15 s on the gate, so a
        # constructor that blocked on it could not return in under 10 s. The bound is far above
        # any real start-up time and far below the block; the not-ready checks below are the proof.
        assert opened < 10.0
        assert not gate.is_set()
        assert not view.is_ready()
        assert view._worker.is_busy()
        assert "Preparing the walkthrough" in _caption(view)
        assert "Preparing the walkthrough" in view.stage_view.accessibleDescription()
        assert view.visual is None and view.narrative is None
        # a stage change before the data is ready shows the same state, and does not fail
        view.set_stage(2)
        assert "Preparing the walkthrough" in _caption(view)
        gate.set()
        assert view.wait_until_ready(15)
        assert view.visual is not None and view.narrative is not None
        assert "Preparing the walkthrough" not in _caption(view)
    finally:
        gate.set()
        view.close_background()
        view.close()


@pytest.mark.gui
def test_phase4_closing_while_preparing_cancels_cleanly(tmp_path, monkeypatch,
                                                        education_view_construction):
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.gui.education_worker import wait_for_background_threads
    from pyCamSet.workflow.education_derive import DerivationUnavailable
    from test_education_phase4 import _context

    gate = threading.Event()
    _gated_result(monkeypatch, education_phase4, "derive_phase4", gate,
                  DerivationUnavailable("never shown"))
    view = education_phase4.Phase4View(_context(tmp_path))
    view.close_background()
    view.close()
    gate.set()
    assert wait_for_background_threads(15)
    assert view._worker.is_busy() is False
    assert view.visual is None


@pytest.mark.gui
def test_phase4_preparation_failure_is_shown_to_the_learner(tmp_path, monkeypatch,
                                                            education_view_construction):
    _app()
    from pyCamSet.gui import education_phase4
    from test_education_phase4 import _context

    def broken(*_args, **_kwargs):
        raise RuntimeError("the saved camera file could not be read")

    monkeypatch.setattr(education_phase4, "derive_phase4", broken)
    view = education_phase4.Phase4View(_context(tmp_path))
    try:
        assert view.wait_until_ready(15) is False
        assert "could not be prepared" in _caption(view)
        assert "the saved camera file could not be read" in _caption(view)
    finally:
        view.close()


@pytest.mark.gui
def test_phase3_view_opens_at_once_showing_the_preparing_state_then_fills_in(
        tmp_path, monkeypatch, education_view_construction):
    _app()
    from pyCamSet.gui import education_phase3
    from pyCamSet.workflow.education_derive import DerivationUnavailable
    from test_education_phase3 import _context

    gate = threading.Event()
    _gated_result(monkeypatch, education_phase3, "derive_phase3", gate,
                  DerivationUnavailable("input not verified"))
    monkeypatch.setattr(education_phase3, "read_trace",
                        lambda *_args: {"status": "unavailable", "reason": "missing sidecar"})
    started = time.perf_counter()
    view = education_phase3.Phase3View(_context(tmp_path, {"params": {}}))
    opened = time.perf_counter() - started
    try:
        # Mechanism, not speed: the held-back derivation waits up to 15 s on the gate, so a
        # constructor that blocked on it could not return in under 10 s. The bound is far above
        # any real start-up time and far below the block; the not-ready checks below are the proof.
        assert opened < 10.0
        assert not gate.is_set()
        assert not view.is_ready()
        assert "Preparing the walkthrough" in _caption(view)
        gate.set()
        assert view.wait_until_ready(30)
        assert view.is_ready()
        assert "Preparing the walkthrough" not in _caption(view)
        assert len(view._evidence) == 6
    finally:
        gate.set()
        view.close_background()
        view.close()


@pytest.mark.gui
def test_dialog_opens_a_phase4_view_without_waiting_and_closes_it_cleanly(
        tmp_path, monkeypatch, education_view_construction):
    _app()
    from pyCamSet.gui import education_phase4
    from pyCamSet.gui.education_dialog import EducationDialog
    from pyCamSet.workflow.education_derive import DerivationUnavailable
    from test_education_phase4 import _context

    gate = threading.Event()
    _gated_result(monkeypatch, education_phase4, "derive_phase4", gate,
                  DerivationUnavailable("held back for the dialog test"))
    run = _context(tmp_path)
    run.phase = "phase4"
    started = time.perf_counter()
    dialog = EducationDialog([run], [run.run_id])
    dialog.show()
    opened = time.perf_counter() - started
    try:
        # Mechanism, not speed: the held-back derivation waits up to 15 s on the gate, so a
        # constructor that blocked on it could not return in under 10 s. The bound is far above
        # any real start-up time and far below the block; the not-ready checks below are the proof.
        assert opened < 10.0
        assert not gate.is_set()
        view = dialog._view
        assert view is not None and not view.is_ready()
        assert "Preparing the walkthrough" in _caption(view)
    finally:
        dialog.close()  # closes while the derivation is still held back
    gate.set()
    from pyCamSet.gui.education_worker import wait_for_background_threads
    assert wait_for_background_threads(15)
