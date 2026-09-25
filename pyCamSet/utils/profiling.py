"""Ad-hoc timing for a callable, reported at the terminal."""
from __future__ import annotations

import time

import numpy as np
from uniplot import histogram

_SCALE = {"us": 1e-3, "ms": 1e-6, "s": 1e-9}


def _allocation_stats():
    """
    Read numba's NRT allocation counters, enabling them if they are off.

    The counters are only switched on at import when NUMBA_NRT_STATS is set,
    so a process that did not set it is asked to enable them here. They are
    process wide: anything else allocating on a numba thread is counted too.

    :return: the counts, or None when they cannot be read
    """
    try:
        from numba.core.runtime import nrt, rtsys
    except ImportError:
        return None
    try:
        return rtsys.get_allocation_stats()
    except RuntimeError:
        pass
    try:
        nrt._nrt.memsys_enable_stats()
        return rtsys.get_allocation_stats()
    except (AttributeError, RuntimeError):
        return None


def benchmark(func, repeats=100, mode="ms", timer=time.time_ns, max_runtime=100):
    """
    Time a callable repeatedly, and print the distribution and its allocations.

    :param func: the call to time, taking no arguments
    :param repeats: how many times to call it
    :param mode: the unit to report in, one of "us", "ms" or "s"
    :param timer: the nanosecond clock to read
    :param max_runtime: seconds to spend before stopping early
    """
    scale = _SCALE[mode]
    before = _allocation_stats()
    times = []
    loop_start = timer()
    for _ in range(repeats):
        start = timer()
        func()
        end = timer()
        times.append(end - start)
        if (end - loop_start) * _SCALE["s"] > max_runtime:
            print(f"Exceeded given max_runtime of {max_runtime} seconds.")
            break
    after = _allocation_stats()

    times = np.array(times) * scale
    mean, stdev, median = np.mean(times), np.std(times), np.median(times)
    max_t = min(mean + 3 * stdev, np.amax(times))
    print(f"Mean: {mean:.2f} {mode}, median: {median:.2f} {mode}, stdev: {stdev:.2f} {mode}")
    if before is None or after is None:
        print("Allocations: unavailable -- numba is not counting them")
    else:
        calls = len(times)
        print(f"Allocations: {(after.alloc - before.alloc) / calls:.1f} per call, "
              f"{(after.free - before.free) / calls:.1f} freed")
    histogram(times, bins=20,
              bins_min=max(mean - 3 * stdev, 0),
              x_max=min(mean + 5 * stdev, max_t),
              height=3,
              color=True,
              y_unit=" freq",
              x_unit=mode,
              )
