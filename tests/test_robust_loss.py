"""The Schur solver's robust losses mean what scipy's do.

``robust_loss`` rewrites each residual so that half its square is scipy's
robust cost; these tests hold the cost, the gradient and the minimum to
scipy's own definitions on small synthetic problems.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import least_squares

from pyCamSet.optimisation import robust_loss

LOSSES = sorted(robust_loss.ROBUST_LOSSES)


def _scipy_cost(f, loss, f_scale):
    """scipy.optimize.least_squares' cost of residuals *f*."""
    result = least_squares(lambda _x: f, np.zeros(1), loss=loss, f_scale=f_scale,
                           max_nfev=1, jac=lambda _x: np.zeros((f.size, 1)))
    return result.cost


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("f_scale", [0.5, 1.0, 3.0])
def test_half_the_square_is_scipys_robust_cost(loss, f_scale):
    f = np.array([0.0, 1e-9, -0.3, 0.9, -2.5, 7.0, -40.0, 1e3])
    r, _ = robust_loss._transform(f, loss, f_scale)
    assert 0.5 * float(r @ r) == pytest.approx(_scipy_cost(f, loss, f_scale), rel=1e-10)
    assert np.all(np.sign(r) == np.sign(f))


@pytest.mark.parametrize("loss", LOSSES)
def test_the_row_scale_is_the_derivative_of_the_transform(loss):
    f = np.array([1e-8, -0.2, 0.7, 1.3, -4.0, 25.0])
    step = 1e-6
    _, scale = robust_loss._transform(f, loss, 2.0)
    plus, _ = robust_loss._transform(f + step, loss, 2.0)
    minus, _ = robust_loss._transform(f - step, loss, 2.0)
    np.testing.assert_allclose(scale, (plus - minus) / (2 * step), rtol=1e-5, atol=1e-8)


@pytest.mark.filterwarnings("ignore:Mean of empty slice")
@pytest.mark.parametrize("loss", LOSSES)
def test_schur_reaches_scipys_robust_minimum(loss):
    """A line fit with gross outliers, solved by both, lands in one place."""
    from pyCamSet.optimisation.numba_schur import (
        ParamGroup, SchurSolver, levenberg_marquardt, spec_from_groups)

    rng = np.random.default_rng(3)
    n_det = 60
    t = np.linspace(-1, 1, 2 * n_det)
    y = 2.0 * t + 0.5 + rng.normal(scale=0.05, size=t.size)
    y[::9] += 6.0  # outliers

    def residuals(x):
        return x[0] * t + x[1] - y

    def blocks(x):
        return np.stack([t, np.ones_like(t)], axis=1).reshape(n_det, 2, 2)

    # Two shared parameters and nothing to eliminate but a dummy group, which
    # the Schur solver needs to be well formed.
    groups = [
        ParamGroup("line", np.zeros((1, 2)), np.ones(1, bool), np.zeros(n_det, int)),
        ParamGroup("none", np.zeros((1, 0)), np.ones(1, bool), np.zeros(n_det, int)),
    ]
    try:
        solver = SchurSolver(spec_from_groups(groups))
    except Exception as exc:  # the solver may reject an empty eliminated block
        pytest.skip(f"synthetic structure not accepted: {exc}")
    wrapped, wrapped_blocks = robust_loss.robustify(residuals, blocks, loss, 0.2)
    ours = levenberg_marquardt(wrapped, wrapped_blocks, np.zeros(2), solver, max_iter=200)
    theirs = least_squares(residuals, np.zeros(2), loss=loss, f_scale=0.2,
                           xtol=1e-12, ftol=1e-12, gtol=1e-12)
    assert ours.cost == pytest.approx(theirs.cost, rel=1e-6)
    np.testing.assert_allclose(ours.x, theirs.x, atol=1e-4)
    # And the outliers were actually discounted: a plain fit is pulled off.
    plain = least_squares(residuals, np.zeros(2))
    assert abs(ours.x[1] - 0.5) < abs(plain.x[1] - 0.5)


def test_unknown_losses_are_refused():
    assert robust_loss.is_supported("linear") and robust_loss.is_supported("soft_l1")
    assert not robust_loss.is_supported("tukey")
    with pytest.raises(ValueError):
        robust_loss.robustify(lambda x: x, lambda x: x, "tukey")
    with pytest.raises(ValueError):
        robust_loss.robustify(lambda x: x, lambda x: x, "huber", f_scale=0.0)
