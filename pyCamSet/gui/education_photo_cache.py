"""Bounded memory for the education photo and explanation caches, and a background
prefetcher that decodes neighbouring photos off the GUI thread.

caches. The byte cap comes from the PYCAMSET_EDU_CACHE_MB environment variable (documented in
docs/how-to/gui.md); the default is 64 MB per cache.

"""
from __future__ import annotations

import dataclasses
import os
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Hashable, Optional

import numpy as np
from PySide6.QtCore import QObject, Signal

#: Environment variable that sets the byte cap of each education cache, in megabytes.
CACHE_MB_ENV = "PYCAMSET_EDU_CACHE_MB"
#: Default cap per cache. A Phase 1 view holds two caches (photos and explanations).
DEFAULT_CACHE_MB = 64
#: Clamp for the environment value: a typo cannot make the cache useless or unbounded.
MIN_CACHE_MB, MAX_CACHE_MB = 8, 1024

#: Longest side of a photo kept in the Phase 1 tiles and by the explanation display copies.
DISPLAY_MAX_SIDE = 1024


def cache_limit_bytes() -> int:
    """The byte cap for one education cache, read from the environment on each call."""
    raw = os.environ.get(CACHE_MB_ENV, "").strip()
    try:
        megabytes = int(raw) if raw else DEFAULT_CACHE_MB
    except ValueError:
        megabytes = DEFAULT_CACHE_MB
    megabytes = min(max(megabytes, MIN_CACHE_MB), MAX_CACHE_MB)
    return megabytes * 1024 * 1024


def estimate_nbytes(value: Any, _depth: int = 0, _seen: Optional[set] = None) -> int:
    """Approximate the memory a cached value holds, counting numpy arrays inside containers,
    dataclasses and plain objects. Cycles and very deep structures are cut off."""
    if _seen is None:
        _seen = set()
    if id(value) in _seen or _depth > 6:
        return 0
    _seen.add(id(value))
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return 0
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, BaseException):
        return 256
    if isinstance(value, dict):
        return sum(estimate_nbytes(v, _depth + 1, _seen) for v in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return sum(estimate_nbytes(v, _depth + 1, _seen) for v in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return sum(estimate_nbytes(getattr(value, f.name, None), _depth + 1, _seen)
                   for f in dataclasses.fields(value))
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return sum(estimate_nbytes(v, _depth + 1, _seen) for v in attributes.values())
    return 64


class ByteLRU:
    """A least-recently-used mapping capped by the estimated bytes of its values.

    The most recently stored entry is never evicted, even when it alone exceeds the cap:
    otherwise a large explanation would be recomputed every time it is shown.
    """

    def __init__(self, max_bytes: Optional[int] = None) -> None:
        self.max_bytes = cache_limit_bytes() if max_bytes is None else int(max_bytes)
        self._items: OrderedDict[Hashable, Any] = OrderedDict()
        self._sizes: dict[Hashable, int] = {}
        self._used = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __contains__(self, key: Hashable) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, key: Hashable) -> Any:
        value = self._items[key]
        self._items.move_to_end(key)
        self.hits += 1
        return value

    def __setitem__(self, key: Hashable, value: Any) -> None:
        if key in self._items:
            self._used -= self._sizes.pop(key)
            del self._items[key]
        size = estimate_nbytes(value)
        self._items[key] = value
        self._sizes[key] = size
        self._used += size
        while self._used > self.max_bytes and len(self._items) > 1:
            old_key, _ = self._items.popitem(last=False)
            self._used -= self._sizes.pop(old_key)
            self.evictions += 1

    def get(self, key: Hashable, default: Any = None) -> Any:
        """The value for ``key`` (counted as a hit), or ``default`` (counted as a miss)."""
        if key in self._items:
            return self[key]
        self.misses += 1
        return default

    def bytes_used(self) -> int:
        return self._used

    def clear(self) -> None:
        self._items.clear()
        self._sizes.clear()
        self._used = 0


class PhotoPrefetcher(QObject):
    """Decodes photos on a small thread pool and hands each result back to the GUI thread.

    Workers never touch Qt widgets or the caches: they only call ``decode`` and emit
    ``ready(key, result)``, which Qt delivers to the GUI thread as a queued signal. The
    pending table is only used from the GUI thread.
    """

    ready = Signal(object, object)

    def __init__(self, decode: Callable[[Any], Any], parent: QObject | None = None, *,
                 max_workers: int = 2, thread_name_prefix: str = "edu-photo") -> None:
        super().__init__(parent)
        self._decode = decode
        self._max_workers = max(1, int(max_workers))
        self._prefix = thread_name_prefix
        self._executor: Optional[ThreadPoolExecutor] = None
        self._pending: dict[Hashable, Future] = {}
        self._closed = False

    def request(self, key: Hashable, argument: Any) -> bool:
        """Start decoding ``argument`` for ``key`` unless it is already pending. GUI thread only."""
        if self._closed or key in self._pending:
            return False
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers,
                                                thread_name_prefix=self._prefix)
        future = self._executor.submit(self._decode, argument)
        self._pending[key] = future
        future.add_done_callback(lambda done, k=key: self._finished(k, done))
        return True

    def take(self, key: Hashable) -> Any:
        """The result for ``key`` if its decode was requested, waiting for it if it is still
        running (less work than a fresh decode); None when nothing was requested. GUI thread only."""
        future = self._pending.pop(key, None)
        if future is None or future.cancelled():
            return None
        try:
            return future.result()
        except Exception as exc:  # noqa: BLE001 - the caller decodes again itself
            return exc

    def is_pending(self, key: Hashable) -> bool:
        return key in self._pending

    def forget(self, key: Hashable) -> None:
        self._pending.pop(key, None)

    def _finished(self, key: Hashable, future: Future) -> None:
        # Runs on a worker thread: no shared state is touched, only the queued signal.
        if self._closed or future.cancelled():
            return
        try:
            result: Any = future.result()
        except Exception as exc:  # noqa: BLE001 - delivered as a value, never raised
            result = exc
        self.ready.emit(key, result)

    def shutdown(self) -> None:
        """Cancel queued decodes and stop accepting work. Safe to call more than once."""
        self._closed = True
        for future in self._pending.values():
            future.cancel()
        self._pending.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None


__all__ = [
    "CACHE_MB_ENV",
    "DEFAULT_CACHE_MB",
    "DISPLAY_MAX_SIDE",
    "ByteLRU",
    "PhotoPrefetcher",
    "cache_limit_bytes",
    "estimate_nbytes",
]
