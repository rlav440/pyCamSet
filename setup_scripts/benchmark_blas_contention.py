"""Measures what BLAS's thread pool costs a bundle adjustment.

A design-time tool, not library code, and not a test: what it measures is the
machine, so there is no number it could assert that would mean the same thing
on the next one.  ``tests/test_solver_blas_threads.py`` asserts that the limit
is applied; this says what applying it is worth.

An iteration of a bundle adjustment alternates the compiled kernels with a
little dense linear algebra.  Between its calls OpenBLAS leaves its threads
spinning rather than sleeping, and where a machine has more logical cores than
the solve can use, those spinners preempt the kernels -- which do no BLAS at
all.  So the cost does not show up in the linear algebra, which is why it is
easy to look straight past: it shows up in the loss and jacobian, and it grows
with the core count rather than with the problem.

Run it from the repository root::

    python setup_scripts/benchmark_blas_contention.py

It needs ``tests/test_data/calibration_ccube``.  Expect it to take a couple of
minutes: it calibrates several times over.

Quiesce the machine first.  A background ``mkdocs serve`` rebuilding on save is
enough to swamp the difference being measured -- one was found holding a third
of a core while this was being written.
"""

import argparse
import contextlib
import io
import logging
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "tests" / "test_data" / "calibration_ccube"


def _time_one(limit_blas: bool, optimise_target: bool) -> tuple[float, float]:
    """
    Calibrate once, and report how long it and its kernels took.

    :param limit_blas: whether to hold BLAS to one thread, as the solver does
    :param optimise_target: whether to solve the target's shape as well
    :return: the wall time of the calibration, and the median loss kernel call
    """
    import numpy as np
    from cv2 import aruco

    from pyCamSet import Ccube
    from pyCamSet.calibration.camera_calibrator import calibrate_cameras
    import pyCamSet.optimisation.optimisation_handling as handling

    calls: list[float] = []
    original = handling.make_optimisation_function

    def timed(param_handler, threads=1):
        loss, jac, init = original(param_handler, threads)

        def wrapped(params):
            start = time.perf_counter()
            result = loss(params)
            calls.append(time.perf_counter() - start)
            return result

        return wrapped, jac, init

    handling.make_optimisation_function = timed
    limit = handling._BLAS_THREADS_DURING_SOLVE if limit_blas else None
    handling._BLAS_THREADS_DURING_SOLVE = limit
    try:
        target = Ccube(n_points=10, length=40,
                       aruco_dict=aruco.DICT_6X6_1000, border_fraction=0.2)
        started = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            calibrate_cameras(str(DATA), target, save=False, threads=1,
                              optimise_target=optimise_target)
        elapsed = time.perf_counter() - started
    finally:
        handling.make_optimisation_function = original
        handling._BLAS_THREADS_DURING_SOLVE = 1
    # The first call is the initial error, outside the limit either way.
    return elapsed, statistics.median(calls[1:]) if len(calls) > 1 else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3,
                        help="how many times to run each arm; the best is kept")
    parser.add_argument("--no-target", action="store_true",
                        help="skip solving the target's shape, which is the "
                             "longer of the two solves")
    args = parser.parse_args()

    if not DATA.is_dir():
        print(f"missing {DATA}", file=sys.stderr)
        return 1
    logging.disable(logging.CRITICAL)

    import numpy as np
    import scipy.linalg  # noqa: F401  - loads its own BLAS, separately
    import threadpoolctl

    # Touch a decomposition first.  Both libraries load their BLAS lazily, and
    # asking before they have reports none of them, which reads as "the limit
    # will do nothing" on a machine where it does a great deal.
    np.linalg.cholesky(np.eye(8))
    pools = [p["num_threads"] for p in threadpoolctl.threadpool_info()
             if p["user_api"] == "blas"]
    print(f"BLAS pools found: {pools or 'none -- the limit will do nothing'}")

    results = {}
    for label, limited in (("BLAS unlimited", False), ("BLAS held to 1", True)):
        runs = [_time_one(limited, not args.no_target)
                for _ in range(args.repeats)]
        total = min(r[0] for r in runs)
        kernel = min(r[1] for r in runs)
        results[label] = (total, kernel)
        print(f"  {label:<16} best total {total:6.2f} s    "
              f"loss kernel {kernel * 1e3:7.2f} ms")

    slow, fast = results["BLAS unlimited"], results["BLAS held to 1"]
    print()
    print(f"calibration  {slow[0] / fast[0]:5.2f}x faster held to one thread")
    print(f"loss kernel  {slow[1] / fast[1]:5.2f}x faster held to one thread")
    print()
    print("A ratio near 1 means this machine's BLAS does not hold a spinning "
          "pool -- macOS's Accelerate does not, so the limit costs it nothing "
          "and gains it nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
