"""
Robust losses for the Schur complement Levenberg-Marquardt solver.

The Schur solver minimises ``0.5 * ||r(x)||^2``. A robust loss instead
minimises ``0.5 * f_scale^2 * sum(rho((f_i / f_scale)^2))`` over the raw
residuals ``f``. Both are the same problem once each residual is replaced by
its square root form

    r_i = f_scale * sign(f_i) * sqrt(rho(z_i)),   z_i = (f_i / f_scale)^2,

because ``0.5 * r_i^2`` is then exactly the robust cost of ``f_i``. The
gradient of the transformed problem equals the robust gradient, so both share
their minima, and the solver's gain ratio test compares true robust costs.
The Jacobian rows are rescaled by ``dr_i / df_i``, which is one for small
residuals and shrinks as a residual moves into the loss's tail.

The ``rho`` functions and the ``f_scale`` convention are scipy's
(``scipy.optimize.least_squares``), so a named loss means the same objective
whichever solver runs it.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

# rho(z) and rho'(z) for z = (f / f_scale)^2, as scipy.optimize.least_squares
# defines them.
_LOSSES: dict[str, tuple[Callable[[np.ndarray], np.ndarray],
                         Callable[[np.ndarray], np.ndarray]]] = {
    "soft_l1": (lambda z: 2.0 * (np.sqrt(1.0 + z) - 1.0),
                lambda z: 1.0 / np.sqrt(1.0 + z)),
    "huber": (lambda z: np.where(z <= 1.0, z, 2.0 * np.sqrt(z) - 1.0),
              lambda z: np.where(z <= 1.0, 1.0, 1.0 / np.sqrt(np.maximum(z, 1.0)))),
    "cauchy": (lambda z: np.log1p(z),
               lambda z: 1.0 / (1.0 + z)),
    "arctan": (lambda z: np.arctan(z),
               lambda z: 1.0 / (1.0 + z * z)),
}

ROBUST_LOSSES = frozenset(_LOSSES)

# Below this z every loss is quadratic to machine precision, so the transform
# is the identity and its derivative is one; evaluating the ratio there would
# divide zero by zero.
_SMALL_Z = 1e-12


def is_supported(loss) -> bool:
    """Whether the Schur solver can honour *loss* (a scipy loss name)."""
    return loss == "linear" or loss in _LOSSES


def _transform(f: np.ndarray, loss: str, f_scale: float):
    """The square root form of *f* and its derivative with respect to *f*."""
    rho, drho = _LOSSES[loss]
    with np.errstate(over="ignore", invalid="ignore"):
        z = (f / f_scale) ** 2
        root = np.sqrt(rho(z))
        small = z < _SMALL_Z
        # dr/df = rho'(z) * sqrt(z) / sqrt(rho(z)); sqrt(z) = |f| / f_scale
        scale = np.where(small, 1.0,
                         drho(z) * np.sqrt(z) / np.where(small, 1.0, root))
        residual = np.where(small, f, f_scale * np.sign(f) * root)
    # A residual so large that z overflows has a row scale of 0 in the limit,
    # for every loss here; leaving the NaN would poison the Jacobian instead.
    # The residual itself stays infinite, so the solver rejects that step.
    scale = np.where(np.isfinite(scale), scale, 0.0)
    return residual, scale


def robustify(loss_fn: Callable[[np.ndarray], np.ndarray],
              jac_blocks: Callable[[np.ndarray], np.ndarray],
              loss: str, f_scale: float = 1.0):
    """
    Wrap a residual and a block Jacobian so the Schur solver minimises *loss*.

    :param loss_fn: ``x -> (2 * n_det,)`` raw residuals
    :param jac_blocks: ``x -> (n_det, 2, row_width)`` raw Jacobian blocks
    :param loss: a scipy loss name other than ``"linear"``
    :param f_scale: the residual scale at which the loss departs from quadratic
    :return: the transformed residual and Jacobian block callables
    """
    if loss not in _LOSSES:
        raise ValueError(f"unsupported robust loss {loss!r}")
    f_scale = float(f_scale)
    if not f_scale > 0:
        raise ValueError(f"f_scale must be positive, not {f_scale}")

    def robust_loss(x):
        return _transform(loss_fn(x), loss, f_scale)[0]

    def robust_blocks(x):
        # The row scale depends on the raw residuals at x; the solver asks for
        # the blocks only at accepted points, so this costs one residual
        # evaluation per iteration.
        _, scale = _transform(loss_fn(x), loss, f_scale)
        blocks = jac_blocks(x)
        return blocks * scale.reshape((-1, 2, 1))

    return robust_loss, robust_blocks
