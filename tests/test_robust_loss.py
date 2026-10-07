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
    solver = SchurSolver(spec_from_groups(groups))
    problem = robust_loss.RobustProblem(residuals, blocks, loss, 0.2)
    ours = levenberg_marquardt(problem.residuals, problem.blocks, np.zeros(2),
                               solver, max_iter=200)
    theirs = least_squares(residuals, np.zeros(2), loss=loss, f_scale=0.2,
                           xtol=1e-12, ftol=1e-12, gtol=1e-12)
    assert ours.cost == pytest.approx(theirs.cost, rel=1e-6)
    np.testing.assert_allclose(ours.x, theirs.x, atol=1e-4)
    # And the outliers were actually discounted: a plain fit is pulled off.
    plain = least_squares(residuals, np.zeros(2))
    assert abs(ours.x[1] - 0.5) < abs(plain.x[1] - 0.5)


@pytest.mark.parametrize("loss", LOSSES)
def test_an_overflowing_finite_residual_scales_its_row_to_zero(loss):
    _, scale = robust_loss._transform(np.array([1e200, 3.0]), loss, 1.0)
    assert scale[0] == 0.0 and scale[1] > 0.0


@pytest.mark.parametrize("loss", LOSSES)
def test_a_non_finite_residual_stays_non_finite(loss):
    """So the step that produced it is rejected, bounded losses included."""
    r, scale = robust_loss._transform(np.array([-np.inf, np.nan, 3.0]), loss, 1.0)
    assert not np.any(np.isfinite(r[:2])) and not np.any(np.isfinite(scale[:2]))
    assert np.isfinite(r[2]) and np.isfinite(scale[2])


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("f_scale", [0.5, 2.0])
def test_cost_is_scipys_robust_cost(loss, f_scale):
    f = np.array([0.0, -0.3, 0.9, -2.5, 7.0, -40.0])
    assert robust_loss.cost(f, loss, f_scale) == pytest.approx(
        _scipy_cost(f, loss, f_scale), rel=1e-10)


@pytest.mark.parametrize("loss", LOSSES)
def test_the_jacobian_scale_is_the_one_scipy_returns(loss):
    """scipy returns its loss-scaled Jacobian at the solution; ours matches it."""
    rng = np.random.default_rng(3)
    t = np.linspace(-1, 1, 40)
    y = 2.0 * t + 0.5 + rng.normal(scale=0.05, size=t.size)
    y[::9] += 6.0

    def residuals(x):
        return x[0] * t + x[1] - y

    jac = np.stack([t, np.ones_like(t)], axis=1)
    # scipy scales the Jacobian it is handed in place, so hand it a copy
    result = least_squares(residuals, np.zeros(2), jac=lambda _x: jac.copy(),
                           loss=loss, f_scale=0.2, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    problem = robust_loss.RobustProblem(residuals, None, loss, 0.2)
    scaled = problem.jacobian_scale(result.x)[:, None] * jac
    np.testing.assert_allclose(scaled, result.jac, rtol=1e-9, atol=1e-12)


def test_the_blocks_at_an_evaluated_point_reuse_its_residuals():
    calls = []

    def residuals(x):
        calls.append(np.copy(x))
        return np.array([x[0] - 1.0, 3.0 * x[0]])

    problem = robust_loss.RobustProblem(
        residuals, lambda x: np.ones((1, 2, 1)), "cauchy", 1.0)
    x = np.array([2.0])
    problem.residuals(x)
    problem.blocks(x)
    assert len(calls) == 1
    problem.blocks(np.array([2.5]))
    assert len(calls) == 2
    np.testing.assert_array_equal(problem.raw, [1.5, 7.5])


def test_unknown_losses_are_refused():
    assert robust_loss.is_supported("linear") and robust_loss.is_supported("soft_l1")
    assert not robust_loss.is_supported("tukey")
    with pytest.raises(ValueError):
        robust_loss.RobustProblem(lambda x: x, lambda x: x, "tukey")
    with pytest.raises(ValueError):
        robust_loss.RobustProblem(lambda x: x, lambda x: x, "huber", f_scale=0.0)
