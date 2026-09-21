"""
That a bundle adjustment holds BLAS to one thread while it runs.

An iteration alternates the compiled kernels with a little dense linear
algebra.  Between its calls OpenBLAS leaves its threads spinning rather than
sleeping, and on a machine with more logical cores than the solve can use
those spinners preempt the kernels -- which do no BLAS at all.  Measured on an
8 core machine, the loss kernel of a Ccube self calibration ran in 1.9 ms with
BLAS held to one thread and 22.2 ms without.

What is asserted here is that the limit is *applied*, not how fast anything
is.  A timing assertion would be asserting the machine rather than the code,
and would fail on a machine whose BLAS does not hold a spinning pool -- macOS
does not, which is why this was invisible there.  ``setup_scripts/
benchmark_blas_contention.py`` is what measures the gain.
"""
import numpy as np
import pytest

from pyCamSet.optimisation import optimisation_handling as oh

threadpoolctl = pytest.importorskip("threadpoolctl")


def _blas_threads() -> list[int]:
    """How many threads each BLAS pool is currently allowed."""
    return [pool["num_threads"] for pool in threadpoolctl.threadpool_info()
            if pool["user_api"] == "blas"]


def test_threadpoolctl_can_see_this_blas():
    """
    The fix is a no-op against a BLAS threadpoolctl cannot reach.

    numpy and scipy each load their own OpenBLAS, and both have to be found
    for the limit to mean anything.
    """
    import scipy.linalg  # noqa: F401
    np.linalg.cholesky(np.eye(8))
    assert _blas_threads(), (
        "threadpoolctl found no BLAS pool, so limiting it would do nothing")


def test_blas_is_held_to_one_thread_inside_the_solve(monkeypatch):
    """
    The limit covers the solve, and only the solve.

    Asserted from inside the loss function, which is what the solver calls
    between its own BLAS calls, and which is exactly where the spinning pool
    does its damage.
    """
    np.linalg.cholesky(np.eye(8))
    outside = _blas_threads()
    if max(outside, default=1) < 2:
        pytest.skip("BLAS is already single threaded, so there is nothing to hold")

    seen: list[list[int]] = []

    class _Handler:
        problem_opts = {"verbosity": 0, "max_nfev": 1, "solver": "trf"}

    def fake_make_optimisation_function(param_handler, threads=1):
        def loss(params):
            seen.append(_blas_threads())
            return np.zeros(4)
        return loss, None, np.zeros(2)

    monkeypatch.setattr(
        oh, "make_optimisation_function", fake_make_optimisation_function)
    monkeypatch.setattr(oh, "can_use_schur", lambda handler: (False, "test"))
    monkeypatch.setattr(
        oh, "reprojection_residuals", lambda err, handler: (np.zeros(4), None))

    with pytest.raises(Exception):
        # The fake handler cannot finish a real solve; the loss calls that get
        # far enough are what this is about.
        oh._solve_bundle_adjustment(_Handler(), threads=1)

    assert len(seen) > 1, (
        "the solver never called the loss, so nothing about it was observed")

    # The first call is the initial error, which _solve_bundle_adjustment
    # evaluates before it starts solving and so outside the limit.  One call
    # is nothing to the cost of a solve, and leaving it out keeps the limit
    # scoped to the loop it is there for.
    initial, during = seen[0], seen[1:]
    assert max(initial) == max(outside), (
        "the initial error was evaluated under the limit, so the limit is "
        "wider than the solve it is meant to cover")
    for observed in during:
        assert max(observed) == oh._BLAS_THREADS_DURING_SOLVE, (
            f"BLAS ran {max(observed)} threads inside the solve, expected "
            f"{oh._BLAS_THREADS_DURING_SOLVE}")

    assert _blas_threads() == outside, (
        "the solve left BLAS limited after it returned")


def test_the_limit_is_one_thread():
    """
    Pinned, because the number is the whole fix.

    The systems solved here are small -- a self calibration's reduced camera
    system is a few hundred square, and the Cholesky that solves it took 4 ms
    of a 10 s solve -- so there is no threading win to trade away.
    """
    assert oh._BLAS_THREADS_DURING_SOLVE == 1
