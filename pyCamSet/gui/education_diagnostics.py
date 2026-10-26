"""Opt-in diagnostics for education mode, for a maintainer collecting a log from a
slow or frozen machine: the chosen 3D backend and why, the platform and Qt facts, and a
watchdog that writes the GUI thread's Python stack when the event loop stalls for over a
second.
no log file. The log goes to stderr and to ``education-debug.log`` in the user's config
directory (``config_directory()/logs``).
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

#: Environment variable that turns diagnostics on (only the value ``1`` counts).
ENV = "PYCAMSET_EDU_DEBUG"
LOGGER_NAME = "pyCamSet.education"
LOG_FILE_NAME = "education-debug.log"
#: A GUI-thread gap longer than this (seconds) is reported as a stall.
STALL_SECONDS = 1.0
#: How often the GUI thread refreshes the heartbeat, and how often the watchdog checks it.
HEARTBEAT_MS = 200
WATCHDOG_POLL_SECONDS = 0.25
WATCHDOG_THREAD_NAME = "pycamset-edu-watchdog"

_WATCHDOG: "StallWatchdog | None" = None
_HANDLER_TAG = "_pycamset_education_debug"


def enabled() -> bool:
    """True when ``PYCAMSET_EDU_DEBUG`` is exactly ``1``."""
    return (os.environ.get(ENV) or "").strip() == "1"


def log_path() -> Path:
    """Where the debug log is written (the per-user config directory's ``logs`` folder)."""
    from pyCamSet.gui.preferences import config_directory
    return config_directory() / "logs" / LOG_FILE_NAME


def logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def configure(path: Path | None = None, stream: Any = None) -> None:
    """Attach the stderr and file handlers once (idempotent). A log file that cannot be
    opened is reported on stderr and does not stop the stderr log."""
    log = logger()
    if any(getattr(h, _HANDLER_TAG, False) for h in log.handlers):
        return
    log.setLevel(logging.DEBUG)
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(message)s")
    console = logging.StreamHandler(stream if stream is not None else sys.stderr)
    console.setFormatter(fmt)
    setattr(console, _HANDLER_TAG, True)
    log.addHandler(console)
    target = path if path is not None else log_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target, encoding="utf-8")
        file_handler.setFormatter(fmt)
        setattr(file_handler, _HANDLER_TAG, True)
        log.addHandler(file_handler)
        log.info("education debug log: %s", target)
    except OSError as exc:
        log.warning("could not open the education debug log %s: %s", target, exc)


def _ram_text() -> str:
    from pyCamSet.gui.education_perf import physical_ram_bytes
    ram = physical_ram_bytes()
    return "unknown" if ram is None else f"{ram / 2**30:.1f} GiB"


def environment_lines() -> list[str]:
    """Platform, CPU, Qt and screen facts, one line each (Qt parts only once an app exists)."""
    from pyCamSet.gui.education_perf import logical_cpu_count
    lines = [f"platform {sys.platform}, Python {sys.version.split()[0]}, "
             f"logical CPUs {logical_cpu_count()}, RAM {_ram_text()}"]
    try:
        from PySide6 import __version__ as pyside_version
        from PySide6.QtCore import qVersion
        from PySide6.QtGui import QGuiApplication
        app = QGuiApplication.instance()
        if app is None:
            lines.append(f"Qt {qVersion()}, PySide6 {pyside_version}, no QGuiApplication yet")
            return lines
        screen = app.primaryScreen()
        dpr = screen.devicePixelRatio() if screen is not None else None
        lines.append(f"Qt {qVersion()}, PySide6 {pyside_version}, Qt platform plugin "
                     f"{app.platformName()}, primary screen devicePixelRatio {dpr}")
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break the GUI
        lines.append(f"Qt facts unavailable ({type(exc).__name__}: {exc})")
    return lines


def note_backend(choice: Any, probe: Mapping[str, Any] | None = None) -> None:
    """Log the 3D backend decision (``choice`` is a ``BackendChoice``) and the probe result.
    Also installs the watchdog. Does nothing unless diagnostics are enabled."""
    if not enabled():
        return
    configure()
    install()
    log = logger()
    log.info("3D backend: %s (%s; chosen automatically: %s)",
             choice.name, choice.reason, "yes" if choice.automatic else "no")
    for line in environment_lines():
        log.info("%s", line)
    if probe:
        log.info("OpenGL probe: ok=%s vendor=%s renderer=%s version=%s error=%s",
                 probe.get("ok"), probe.get("vendor"), probe.get("renderer"),
                 probe.get("version"), probe.get("error"))


class StallWatchdog:
    """Reports GUI-thread stalls. A heartbeat timer on the GUI thread calls :meth:`beat`; a
    daemon thread calls :meth:`check`, which logs the GUI thread's stack once per stall.
    ``clock`` and ``frames`` are injectable for tests."""

    def __init__(self, gui_ident: int, *, threshold: float = STALL_SECONDS,
                 clock: Callable[[], float] = time.monotonic,
                 frames: Callable[[], Mapping[int, Any]] | None = None,
                 log: logging.Logger | None = None) -> None:
        self.gui_ident = gui_ident
        self.threshold = float(threshold)
        self.clock = clock
        self.frames = frames if frames is not None else sys._current_frames
        self.log = log if log is not None else logger()
        self._last_beat = clock()
        self._stall_started: float | None = None
        self._reported = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._timer: Any = None

    def beat(self) -> None:
        """Called on the GUI thread by the heartbeat timer."""
        now = self.clock()
        if self._reported and self._stall_started is not None:
            self.log.info("GUI thread responsive again after %.1f s", now - self._stall_started)
        self._last_beat = now
        self._reported = False
        self._stall_started = None

    def check(self) -> bool:
        """Log the GUI thread's stack if it has not beaten for ``threshold`` seconds. Reports
        each stall once. Returns True when this call reported a stall."""
        now = self.clock()
        idle = now - self._last_beat
        if idle <= self.threshold or self._reported:
            return False
        self._reported = True
        self._stall_started = self._last_beat
        frame = self.frames().get(self.gui_ident)
        stack = "".join(traceback.format_stack(frame)) if frame is not None \
            else "(GUI thread frame not available)\n"
        self.log.warning("GUI thread stalled for %.1f s; Python stack of the GUI thread:\n%s",
                         idle, stack)
        return True

    def _run(self) -> None:
        while not self._stop.wait(WATCHDOG_POLL_SECONDS):
            self.check()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=WATCHDOG_THREAD_NAME,
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._timer is not None:
            self._timer.stop()


def install() -> StallWatchdog | None:
    """Start the heartbeat and watchdog once, when diagnostics are enabled. Returns the
    watchdog, or None when diagnostics are off or it cannot start. GUI thread only."""
    global _WATCHDOG
    if not enabled():
        return None
    if _WATCHDOG is not None:
        return _WATCHDOG
    try:
        from PySide6.QtCore import QCoreApplication, QThread, QTimer
        app = QCoreApplication.instance()
        if app is not None and QThread.currentThread() != app.thread():
            return None  # the heartbeat must live on the GUI thread
        configure()
        watchdog = StallWatchdog(threading.get_ident())
        timer = QTimer(app)
        timer.setInterval(HEARTBEAT_MS)
        timer.timeout.connect(watchdog.beat)
        timer.start()
        watchdog._timer = timer
        watchdog.start()
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break the GUI
        logger().warning("education watchdog did not start: %s", exc)
        return None
    _WATCHDOG = watchdog
    logger().info("education watchdog running (stall threshold %.1f s)", STALL_SECONDS)
    return watchdog


def _reset_for_tests() -> None:
    """Stop the watchdog and detach the handlers this module added (tests only)."""
    global _WATCHDOG
    if _WATCHDOG is not None:
        _WATCHDOG.stop()
    _WATCHDOG = None
    log = logger()
    for handler in list(log.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            log.removeHandler(handler)
            handler.close()
