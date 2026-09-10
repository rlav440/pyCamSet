"""
A live view of the bundle adjustment while it runs.

The solve is fast, but it sits behind two things that are not: compiling the
loss and jacobian on the first run for a problem shape, and the detection
before it. Between them a run can spend a long time with nothing on screen,
which is indistinguishable from a hang. This shows the optimiser working, and
shows the error coming down while it does.

The bar is only drawn for a person at a terminal. Redirected to a file, in a
test, or in CI it stays quiet and the per iteration log records carry the same
information.
"""
from __future__ import annotations

import sys

import numpy as np
from tqdm import tqdm


def _interactive() -> bool:
    """
    Whether there is a terminal on stderr to draw a bar on.
    """
    try:
        return bool(sys.stderr is not None and sys.stderr.isatty())
    except (AttributeError, ValueError):     # closed or replaced stderr
        return False


class OptimisationProgress:
    """
    A live counter of the solver's iterations, labelled with the error.

    Used as a context manager, and handed to a solver as its ``callback``::

        with OptimisationProgress() as progress:
            result = levenberg_marquardt(..., callback=progress.update)

    Deliberately a counter and not a bar with a total. Levenberg-Marquardt
    stops when the cost stops moving, which is usually a small fraction of
    the iteration cap -- a real solve took 15 of a permitted 100 -- so a
    proportion bar would sit at 15% and then vanish, reading as a solve that
    gave up rather than one that converged.

    :param enabled: draw the counter; defaults to whether stderr is a terminal
    """

    def __init__(self, enabled: bool | None = None):
        self.enabled = _interactive() if enabled is None else enabled
        self._bar: tqdm | None = None

    def __enter__(self) -> OptimisationProgress:
        if self.enabled:
            self._bar = tqdm(
                desc="bundle adjustment", unit=" it", leave=False,
                file=sys.stderr,
                # a whole solve can finish inside tqdm's default 0.1s refresh
                # window, which would show an empty counter and nothing else.
                mininterval=0.0,
                # tqdm prepends its own ", " to a postfix, so no space here
                bar_format="{desc}: {n_fmt}{unit} [{elapsed}]{postfix}",
            )
        return self

    def __exit__(self, *exc) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def update(self, iteration: int, cost: float,
               residuals: np.ndarray | None = None) -> None:
        """
        Advance the counter one accepted step.

        :param iteration: the solver's iteration number
        :param cost: half the sum of squared residuals
        :param residuals: the residual vector, for the mean pixel error
        """
        if self._bar is None:
            return
        postfix = f"cost {cost:.4g}"
        if residuals is not None and np.size(residuals):
            euclid = np.linalg.norm(
                np.reshape(np.asarray(residuals, dtype=float), (-1, 2)), axis=1)
            postfix += f", error {np.mean(euclid):.3f} px"
        self._bar.set_postfix_str(postfix, refresh=False)
        self._bar.update(max(1, iteration - self._bar.n))
