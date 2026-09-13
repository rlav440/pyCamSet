"""
Getting a phase's output to whoever is watching it run.

A phase writes to three places at once: ``print``, the logging module, and
whatever a library it calls decides to do.  A caller wants one stream of
lines.  :func:`captured_output` collects all three and hands each line to one
callable, so a phase runner takes a ``log`` argument and nothing else.
"""
from __future__ import annotations

import contextlib
import io
import logging
from typing import Callable, Iterator

LogFn = Callable[[str], None]

LOG_FORMAT = "[%(levelname)s] %(name)s: %(message)s"


def discard(_line: str) -> None:
    """A log callable for a caller that wants no output."""


class EmitStream(io.TextIOBase):
    """A writable stream that hands each completed line to a callable."""

    def __init__(self, log: LogFn) -> None:
        super().__init__()
        self._log = log
        self._buffer = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._log(line)
        return len(s)

    def flush(self) -> None:
        if self._buffer.strip():
            self._log(self._buffer.strip())
        self._buffer = ""


class EmitLogHandler(logging.Handler):
    """A logging handler that hands each formatted record to a callable."""

    def __init__(self, log: LogFn) -> None:
        super().__init__(level=logging.INFO)
        self._log = log
        self.setFormatter(logging.Formatter(LOG_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            if message.strip():
                self._log(message)
        except Exception:
            pass


@contextlib.contextmanager
def captured_output(log: LogFn) -> Iterator[None]:
    """
    Send everything written inside the block to *log*, one line at a time.

    Covers ``stdout``, ``stderr`` and the root logger.  Whatever happens in
    the block, the partial last line is flushed, the log handler removed and
    the root logger put back at the level it was found at.

    The level is part of the job: a library's default root logger drops
    ``INFO`` before any handler sees it, and ``INFO`` is where the reports
    are -- the detection summary, the initial intrinsics, the calibration
    summary.  Leaving it alone meant a phase showed its warnings and none of
    its results.

    :param log: what to call with each line
    """
    stream = EmitStream(log)
    handler = EmitLogHandler(log)
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    if not root_logger.isEnabledFor(logging.INFO):
        root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            yield
    finally:
        stream.flush()
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_level)


@contextlib.contextmanager
def non_interactive_plotting() -> Iterator[None]:
    """
    Force matplotlib to draw into ``Agg`` and never open a window.

    A phase run from a thread or a server has no business opening a figure:
    the drawing backends want an event loop, and take the process down when
    they do not get their own.  ``plt.show`` is restored on the way out.
    """
    plt = None
    original_show = None
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as _plt
        plt = _plt
        original_show = plt.show
        plt.show = lambda *args, **kwargs: None
    except Exception:
        pass
    try:
        yield
    finally:
        if plt is not None and original_show is not None:
            try:
                plt.show = original_show
            except Exception:
                pass
