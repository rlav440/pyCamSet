"""Regression tests for the lockbox residual reshape bug (PR #18, B1).

``append_lockbox_residuals`` appends lockbox prior residuals (6 per constrained
camera, plus 3 world-centre residuals per camera when ``center_sigma > 0``)
onto the end of the reprojection residual vector.  The downstream RPE code used
to ``np.reshape(fun, (-1, 2))`` the *whole* vector, which:

  * crashes with ``ValueError`` when the total length is odd (an odd number of
    constrained cameras with centre priors enabled), and
  * folds the near-zero prior residuals into the per-row norm, diluting the
    reported RPE.

The fix splits the vector into the reprojection prefix (always even: two
residuals per observation) and the prior suffix, and computes the RPE from the
prefix only.
"""

import numpy as np

from pyCamSet.optimisation.camera_lockbox import (
    CameraLockboxPrior,
    append_lockbox_residuals,
)
from pyCamSet.optimisation.optimisation_handling import _split_residuals


class _StubHandler:
    """Minimal stand-in for a bundle handler exposing the base residual count."""

    def __init__(self, base_count: int):
        self._base_count = base_count

    def get_base_residual_count(self) -> int:
        return self._base_count


def _make_prior(n_cameras: int, center_sigma: float) -> CameraLockboxPrior:
    """Build a prior over ``n_cameras`` cameras with centre priors enabled."""
    param_len = 6 * n_cameras
    indices = np.arange(param_len, dtype=int)
    return CameraLockboxPrior(
        enabled=True,
        param_len=param_len,
        indices=indices,
        centres=np.zeros(param_len, dtype=float),
        sigmas=np.ones(param_len, dtype=float),
        lower_bounds=np.full(param_len, -np.inf, dtype=float),
        upper_bounds=np.full(param_len, np.inf, dtype=float),
        camera_names=tuple(f"cam{i}" for i in range(n_cameras)),
        world_centers=np.zeros((n_cameras, 3), dtype=float),
        cam_base_indices=np.arange(0, param_len, 6, dtype=int),
        center_sigma=center_sigma,
    )


def test_odd_total_residual_count_no_longer_crashes():
    """An odd total (even reprojection + odd centre priors) must not crash the reshape."""
    base = np.zeros(10, dtype=float)  # 5 observations -> 10 residuals (even)
    prior = _make_prior(n_cameras=1, center_sigma=1.0)
    full = append_lockbox_residuals(base, np.zeros(6), prior)
    # 10 base + 6 param + 3 centre = 19 (odd).
    assert full.size == 19
    assert full.size % 2 == 1

    reproj, priors = _split_residuals(full, _StubHandler(10))
    assert reproj.size == 10
    assert priors is not None and priors.size == 9
    # The reprojection prefix is even, so the reshape succeeds.
    np.reshape(reproj, (-1, 2))


def test_even_total_residual_count_splits_correctly():
    """An even total still splits into the reprojection prefix and prior suffix."""
    base = np.zeros(10, dtype=float)
    prior = _make_prior(n_cameras=2, center_sigma=1.0)
    full = append_lockbox_residuals(base, np.zeros(12), prior)
    # 10 base + 12 param + 6 centre = 28 (even).
    assert full.size == 28

    reproj, priors = _split_residuals(full, _StubHandler(10))
    assert reproj.size == 10
    assert priors is not None and priors.size == 18


def test_rpe_is_not_diluted_by_priors():
    """The RPE must be computed from the reprojection prefix only."""
    # 4 observations with a real reprojection error of 3 px each.
    base = np.array([3.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 3.0])
    prior = _make_prior(n_cameras=1, center_sigma=1.0)
    full = append_lockbox_residuals(base, np.zeros(6), prior)

    reproj, _ = _split_residuals(full, _StubHandler(8))
    rpe = float(np.mean(np.linalg.norm(np.reshape(reproj, (-1, 2)), axis=1)))
    assert rpe == 3.0  # exactly the reprojection error, priors excluded


def test_no_prior_returns_full_vector():
    """With no prior (disabled), the split returns the whole vector unchanged."""
    base = np.zeros(10, dtype=float)
    reproj, priors = _split_residuals(base, _StubHandler(0))
    assert reproj.size == 10
    assert priors is None
