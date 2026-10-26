'''Run education-mode preparation off the GUI thread, keyed, cancellable and quiet on close.
'''
from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal

# A job's callable receives one argument: report(text), which shows progress text on the GUI.
ProgressReport = Callable[[str], None]

# Strong references to running threads, so a view that is closed while its thread still runs
# cannot destroy a QThread that is running (Qt aborts in that case). Entries leave on finish.
_RUNNING: set["_Job"] = set()


def configure_numba_cache() -> Path | None:
    """Point numba's on-disk cache at a per-user folder, unless the user chose one.

    By default numba caches each compiled function beside its source file, so a moved or freshly
    copied tree compiles everything again (tens of seconds on a slow machine). The education views
    use compiled solver kernels; a per-user folder makes that a one-time cost. The variable is read
    when numba is first imported, so this only takes effect if numba has not been imported yet.
    It is not called at import time: the education dialog calls it when it opens, before its
    views start any worker. Returns the folder it set, or None when nothing was changed.
    """
    if os.environ.get("NUMBA_CACHE_DIR") or "numba" in sys.modules:
        return None
    folder = Path.home() / ".cache" / "pycamset" / "numba"
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None  # read-only home: numba falls back to its default cache location
    os.environ["NUMBA_CACHE_DIR"] = str(folder)
    return folder



class _Job(QThread):
    """One callable on its own thread. Its signals are emitted from that thread."""

    done = Signal(object, object, object)  # job, result, error text (None when it worked)
    progress = Signal(object, str)         # job, progress text

    def __init__(self, key: Any, work: Callable[[ProgressReport], Any]) -> None:
        super().__init__()
        self.key = key
        self._work = work
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def run(self) -> None:  # runs on the worker thread
        def report(text: str) -> None:
            if not self._cancelled.is_set():
                self.progress.emit(self, str(text))

        result: Any = None
        error: str | None = None
        try:
            result = self._work(report)
        except Exception as exc:  # noqa: BLE001 - the learner sees the message, not a crash
            error = str(exc) or type(exc).__name__
        self.done.emit(self, result, error)


class EducationWorker(QObject):
    """Keyed background job runner for one education view.

    ``submit(key, work)`` starts ``work`` on a worker thread. A later submit (or ``cancel``)
    makes the earlier job stale: its result is dropped and never signalled. Exactly one of
    ``ready(key, result)`` or ``failed(key, message)`` is emitted for a job that is still
    current when it finishes. ``status(key, text)`` carries the job's progress text.
    """

    ready = Signal(object, object)
    failed = Signal(object, str)
    status = Signal(object, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._job: _Job | None = None

    # -- starting and stopping -------------------------------------------------------------

    def submit(self, key: Any, work: Callable[[ProgressReport], Any]) -> None:
        """Start *work* for *key*, superseding any job already in flight."""
        self.cancel()
        job = _Job(key, work)
        job.done.connect(self._on_done)
        job.progress.connect(self._on_progress)
        job.finished.connect(lambda: _RUNNING.discard(job))
        _RUNNING.add(job)
        self._job = job
        job.start()

    def cancel(self) -> None:
        """Make the current job stale. Its thread finishes in the background; nothing is shown."""
        job, self._job = self._job, None
        if job is not None:
            job.cancel()

    def close(self) -> None:
        """Cancel for good (dialog closed). Never raises, never waits for the thread."""
        self.cancel()

    # -- state for views, tests and tools --------------------------------------------------

    def is_busy(self) -> bool:
        """True from ``submit`` until the job's result or failure has been delivered."""
        return self._job is not None

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        """Block (on the GUI thread) until no job is pending, pumping events so results arrive.

        Returns False if *timeout* seconds pass first. Intended for tests and screenshot tools;
        the GUI itself never calls it.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.is_busy():
            QCoreApplication.processEvents()
            if not self.is_busy():
                break
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.005)
        QCoreApplication.processEvents()
        return True

    # -- delivery (GUI thread) -------------------------------------------------------------

    def _on_progress(self, job: _Job, text: str) -> None:
        if job is self._job and not job.cancelled():
            self.status.emit(job.key, text)

    def _on_done(self, job: _Job, result: Any, error: str | None) -> None:
        if job is not self._job or job.cancelled():
            return  # stale: superseded, cancelled or closed
        self._job = None
        if error is None:
            self.ready.emit(job.key, result)
        else:
            self.failed.emit(job.key, error)


def wait_for_background_threads(timeout: float = 10.0) -> bool:
    """Join every education job still running (tests, and clean process exit).

    Returns True once none remain. Events are pumped while waiting, because a finished thread
    leaves ``_RUNNING`` through a queued signal on the GUI thread.
    """
    deadline = time.monotonic() + timeout
    while _RUNNING and time.monotonic() < deadline:
        for job in list(_RUNNING):
            job.wait(50)
        QCoreApplication.processEvents()
    return not _RUNNING
