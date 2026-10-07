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

# rho(z), rho'(z) and rho''(z) for z = (f / f_scale)^2, as
# scipy.optimize.least_squares defines them.
_LOSSES: dict[str, tuple[Callable[[np.ndarray], np.ndarray], ...]] = {
    "soft_l1": (lambda z: 2.0 * (np.sqrt(1.0 + z) - 1.0),
                lambda z: 1.0 / np.sqrt(1.0 + z),
                lambda z: -0.5 / (1.0 + z) ** 1.5),
    "huber": (lambda z: np.where(z <= 1.0, z, 2.0 * np.sqrt(z) - 1.0),
              lambda z: np.where(z <= 1.0, 1.0, 1.0 / np.sqrt(np.maximum(z, 1.0))),
              lambda z: np.where(z <= 1.0, 0.0, -0.5 / np.maximum(z, 1.0) ** 1.5)),
    "cauchy": (lambda z: np.log1p(z),
               lambda z: 1.0 / (1.0 + z),
               lambda z: -1.0 / (1.0 + z) ** 2),
    "arctan": (lambda z: np.arctan(z),
               lambda z: 1.0 / (1.0 + z * z),
               lambda z: -2.0 * z / (1.0 + z * z) ** 2),
}

ROBUST_LOSSES = frozenset(_LOSSES)

# Below this z every loss is quadratic to machine precision, so the transform
# is the identity and its derivative is one; evaluating the ratio there would
# divide zero by zero.
_SMALL_Z = 1e-12


def is_supported(loss) -> bool:
    """Whether the Schur solver can honour *loss* (a scipy loss name)."""
    return loss == "linear" or loss in _LOSSES


def cost(f: np.ndarray, loss: str = "linear", f_scale: float = 1.0) -> float:
    """
    The objective *loss* assigns to residuals *f*, as scipy reports ``cost``.

    :param loss: a scipy loss name, or a callable taking ``z`` and returning
        scipy's ``(3, m)`` array of ``rho`` and its derivatives
    """
    f = np.asarray(f, dtype=float)
    if loss == "linear":
        return 0.5 * float(f @ f)
    z = (f / f_scale) ** 2
    rho = np.asarray(loss(z))[0] if callable(loss) else _LOSSES[loss][0](z)
    return 0.5 * f_scale ** 2 * float(np.sum(rho))


def _transform(f: np.ndarray, loss: str, f_scale: float):
    """
    The square root form of *f* and its derivative with respect to *f*.

    A non-finite raw residual stays non-finite in both, so a step that
    produces one is rejected and a start point with one is reported by the
    solver's finite-Jacobian check. A finite residual so large that ``z``
    overflows has a row scale of 0, its limit for every loss here.
    """
    rho, drho, _ = _LOSSES[loss]
    finite = np.isfinite(f)
    with np.errstate(over="ignore", invalid="ignore"):
        z = (f / f_scale) ** 2
        root = np.sqrt(rho(z))
        small = z < _SMALL_Z
        # dr/df = rho'(z) * sqrt(z) / sqrt(rho(z)); sqrt(z) = |f| / f_scale
        scale = np.where(small, 1.0,
                         drho(z) * np.sqrt(z) / np.where(small, 1.0, root))
        residual = np.where(small, f, f_scale * np.sign(f) * root)
    scale = np.where(np.isfinite(scale), scale, 0.0)
    return np.where(finite, residual, f), np.where(finite, scale, np.nan)


def _jacobian_scale(f: np.ndarray, loss: str, f_scale: float) -> np.ndarray:
    """scipy's row scale for the Jacobian it returns: ``sqrt(rho' + 2 rho'' z)``."""
    _, drho, d2rho = _LOSSES[loss]
    with np.errstate(over="ignore", invalid="ignore"):
        z = (f / f_scale) ** 2
        scale = drho(z) + 2.0 * d2rho(z) * z
    eps = np.finfo(float).eps
    return np.sqrt(np.where(np.isfinite(scale) & (scale > eps), scale, eps))


class RobustProblem:
    """
    A residual and a block Jacobian rewritten so the Schur solver minimises a
    scipy robust loss.

    The raw residuals at the last point evaluated are kept, so the Jacobian
    blocks at an accepted point reuse the residual evaluation that accepted it.

    :param loss_fn: ``x -> (2 * n_det,)`` raw residuals
    :param jac_blocks: ``x -> (n_det, 2, row_width)`` raw Jacobian blocks
    :param loss: a scipy loss name other than ``"linear"``
    :param f_scale: the residual scale at which the loss departs from quadratic
    """

    def __init__(self, loss_fn: Callable[[np.ndarray], np.ndarray],
                 jac_blocks: Callable[[np.ndarray], np.ndarray],
                 loss: str, f_scale: float = 1.0):
        if loss not in _LOSSES:
            raise ValueError(f"unsupported robust loss {loss!r}")
        f_scale = float(f_scale)
        if not f_scale > 0:
            raise ValueError(f"f_scale must be positive, not {f_scale}")
        self._loss_fn = loss_fn
        self._jac_blocks = jac_blocks
        self.loss = loss
        self.f_scale = f_scale
        self._x: np.ndarray | None = None
        #: raw residuals at the last point evaluated
        self.raw: np.ndarray | None = None

    def raw_at(self, x: np.ndarray) -> np.ndarray:
        """The raw residuals at *x*."""
        if self._x is None or not np.array_equal(x, self._x):
            self.raw = np.array(self._loss_fn(x), dtype=float)
            self._x = np.array(x, dtype=float)
        return self.raw

    def residuals(self, x: np.ndarray) -> np.ndarray:
        """The square root form of the residuals at *x*."""
        return _transform(self.raw_at(x), self.loss, self.f_scale)[0]

    def blocks(self, x: np.ndarray) -> np.ndarray:
        """The Jacobian blocks of :meth:`residuals` at *x*."""
        _, scale = _transform(self.raw_at(x), self.loss, self.f_scale)
        return self._jac_blocks(x) * scale.reshape((-1, 2, 1))

    def jacobian_scale(self, x: np.ndarray) -> np.ndarray:
        """The row scale that turns the raw Jacobian into scipy's robust one."""
        return _jacobian_scale(self.raw_at(x), self.loss, self.f_scale)
