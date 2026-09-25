"""Correctness of the Schur complement solver and the parameter group seam.

The solver tests are synthetic: they build parameter groups and jacobian blocks
directly, so they need no image corpus and run in well under a second. The one
data backed test checks that switching the solver does not change the answer a
real calibration converges to.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyCamSet.optimisation.numba_schur import (
    ParamGroup,
    SchurSolver,
    _diag_scale,
    _solve_reduced,
    levenberg_marquardt,
    spec_from_groups,
)


def _random_problem(n_det=400, n_cams=3, n_imgs=7, n_pts=40, seed=0, fix_some=True):
    """A self calibration shaped problem: cameras, poses and free points."""
    rng = np.random.default_rng(seed)
    cam = rng.integers(0, n_cams, n_det)
    img = rng.integers(0, n_imgs, n_det)
    key = rng.integers(0, n_pts, n_det)

    poses_unfixed = np.ones(n_imgs, dtype=bool)
    points_unfixed = np.ones(n_pts * 3, dtype=bool)
    if fix_some:
        poses_unfixed[0] = False          # the reference pose
        points_unfixed[:3] = False        # gauge fixing, as SelfBundleHandler does
        points_unfixed[3:5] = False

    groups = [
        ParamGroup("intr", rng.normal(size=(n_cams, 9)), np.ones(n_cams, bool), cam),
        ParamGroup("extr", rng.normal(size=(n_cams, 6)), np.ones(n_cams, bool), cam),
        ParamGroup("pose", rng.normal(size=(n_imgs, 6)), poses_unfixed, img),
        ParamGroup("point", rng.normal(size=(n_pts, 3)), points_unfixed, key),
    ]
    row_width = sum(g.n_par for g in groups)
    blocks = rng.normal(size=(n_det, 2, row_width))
    resid = rng.normal(size=(n_det, 2))
    return groups, blocks, resid


def _dense_jacobian(groups, blocks):
    """The same blocks written out as a dense jacobian over the free parameters."""
    spec = spec_from_groups(groups)
    n_det = blocks.shape[0]
    offsets, off = [], 0
    for g in groups:
        offsets.append(off)
        off += g.base.size
    full = np.zeros((2 * n_det, off))
    rows = np.arange(n_det)
    col = 0
    for g, start in zip(groups, offsets):
        for j in range(g.n_par):
            cols = start + g.index * g.n_par + j
            full[2 * rows, cols] = blocks[:, 0, col + j]
            full[2 * rows + 1, cols] = blocks[:, 1, col + j]
        col += g.n_par
    free = np.concatenate([g.element_unfixed for g in groups])
    return full[:, free], spec


@pytest.mark.parametrize("lam", [1e-6, 1e-3, 1.0])
def test_schur_step_matches_dense_solve(lam):
    """The reduced solve must reproduce a direct solve of the same system."""
    groups, blocks, resid = _random_problem()
    J, spec = _dense_jacobian(groups, blocks)
    r = resid.reshape(-1)
    solver = SchurSolver(spec, threads=1)

    # damp the reference with exactly the weights the solver uses, so this
    # tests the Schur reduction rather than the damping policy
    B, E, C, v, w = solver.normal_equations(blocks, resid)
    damping = _diag_scale(solver, B, C)
    A = J.T @ J
    reference = np.linalg.solve(A + lam * np.diag(damping), -(J.T @ r))

    step = solver.step(blocks, resid, lam)
    assert step.shape == reference.shape
    assert np.linalg.norm(step - reference) / np.linalg.norm(reference) < 1e-9


def test_schur_serial_and_parallel_agree():
    """The privately accumulating kernel must match the serial one."""
    groups, blocks, resid = _random_problem(n_det=2000)
    spec = spec_from_groups(groups)
    serial = SchurSolver(spec, threads=1).step(blocks, resid, 1e-3)
    parallel = SchurSolver(spec, threads=4).step(blocks, resid, 1e-3)
    assert np.allclose(serial, parallel, rtol=1e-9, atol=0)


def test_fixed_parameters_are_absent_from_the_step():
    """Fixed parameters have no column, so the step is only over free ones."""
    groups, blocks, resid = _random_problem()
    spec = spec_from_groups(groups)
    step = SchurSolver(spec, threads=1).step(blocks, resid, 1e-3)
    n_free = sum(g.n_free for g in groups)
    assert step.size == n_free
    assert np.all(np.isfinite(step))


def test_group_order_puts_the_eliminated_block_last():
    """The solver eliminates the last group, so it must be the local one."""
    groups, _, _ = _random_problem()
    spec = spec_from_groups(groups)
    assert spec.n_elim_blocks == groups[-1].n_rows
    assert spec.elim_size == groups[-1].n_par
    assert spec.n_keep_full == sum(g.base.size for g in groups[:-1])


def test_reduced_system_is_symmetric_and_positive_definite():
    """The premises of the Cholesky solve, asserted on the solver's own matrix.

    This calls ``reduced_system`` rather than rebuilding ``S`` locally: a
    reimplementation of the assembly would keep passing if the real one gained a
    wrong transpose or a sign error, leaving ``_solve_reduced`` fed a matrix that
    is neither symmetric nor positive definite.
    """
    groups, blocks, resid = _random_problem()
    spec = spec_from_groups(groups)
    solver = SchurSolver(spec, threads=1)
    B, E, C, v, w = solver.normal_equations(blocks, resid)

    S, g, _Cinv = solver.reduced_system(B, E, C, v, w, 1e-3)

    assert S.shape == (spec.n_keep_full, spec.n_keep_full)
    assert g.shape == (spec.n_keep_full,)

    asym = float(np.max(np.abs(S - S.T)))
    scale = float(np.max(np.abs(S)))
    assert asym < 1e-10 * scale, f"S is not symmetric: {asym:.3e} vs {scale:.3e}"
    assert float(np.min(np.linalg.eigvalsh(S))) > 0, "the damped S is not PD"

    # and the pinning of fixed parameters preserves both properties, since it
    # writes the identity over entire rows and columns
    assert spec.keep_fixed.any(), "this problem should pin something"
    pinned = np.flatnonzero(spec.keep_fixed)
    assert np.allclose(S[pinned, :], np.eye(spec.n_keep_full)[pinned, :])
    assert np.allclose(S[:, pinned], np.eye(spec.n_keep_full)[:, pinned])


def test_the_step_solves_the_system_reduced_system_returns():
    """``_step_from_parts`` must be driven by the matrix it exposes.

    Ties the two together so the exposed assembly cannot drift away from the one
    the step actually solves.
    """
    groups, blocks, resid = _random_problem()
    spec = spec_from_groups(groups)
    solver = SchurSolver(spec, threads=1)
    B, E, C, v, w = solver.normal_equations(blocks, resid)
    lam = 1e-3

    S, g, _Cinv = solver.reduced_system(B, E, C, v, w, lam)
    dc = _solve_reduced(S, g)

    step = solver.step_from_parts(B, E, C, v, w, lam)
    # the step is in free-parameter space, its retained half first, so it holds
    # the reduced solve's free entries followed by the eliminated block's
    n_keep_free = int(spec.keep_free.sum())
    assert np.allclose(step[:n_keep_free], dc[spec.keep_free], rtol=1e-12)


@pytest.mark.parametrize("n", [4, 40, 300])
def test_reduced_solve_matches_the_general_solve(n):
    """The symmetric solve must agree with LU on a symmetric positive definite S."""
    rng = np.random.default_rng(n)
    M = rng.normal(size=(n, n))
    S = M @ M.T + n * np.eye(n)
    g = rng.normal(size=n)

    symmetric = _solve_reduced(S, g)
    general = np.linalg.solve(S, g)

    assert np.allclose(symmetric, general, rtol=1e-9, atol=1e-12)
    # and it is a solution, not merely close to one
    assert np.max(np.abs(S @ symmetric - g)) < 1e-8 * max(
        1.0, float(np.max(np.abs(g))))


def test_reduced_solve_falls_back_when_the_system_is_indefinite():
    """An indefinite S has no Cholesky factor; the general solve must still answer.

    Damping makes S positive definite in the solver's own path, so this pins
    the fallback for a caller that passes a ``lam`` which does not.
    """
    S = np.array([[1.0, 2.0], [2.0, 1.0]])          # eigenvalues -1 and 3
    g = np.array([1.0, 1.0])

    assert _solve_reduced(S, g) == pytest.approx(np.linalg.solve(S, g))


def test_reduced_solve_propagates_a_non_finite_system():
    """A poisoned system must stay poisoned, not silently return a number.

    A wild iterate can overflow the reduced system; the caller treats a
    non finite step as "damp harder", so a finite answer here would hide it.
    """
    S = np.array([[2.0, 0.5], [0.5, 2.0]])
    S[0, 0] = np.nan

    out = _solve_reduced(S, np.array([1.0, 1.0]))
    assert not np.all(np.isfinite(out))


@pytest.mark.data
@pytest.mark.slow
def test_schur_and_trust_region_agree_on_a_real_calibration(data_dir):
    """Switching the solver must not change where the calibration lands."""
    from multiprocessing import cpu_count

    from cv2 import aruco

    import pyCamSet.optimisation.template_handler as th
    from pyCamSet import Ccube
    from pyCamSet.calibration.camera_calibrator import (
        detect_datapoints_in_imfile,
        run_initial_calibration,
    )
    from pyCamSet.optimisation.optimisation_handling import (
        can_use_schur,
        run_bundle_adjustment,
    )

    target = Ccube(
        n_points=10, length=40, aruco_dict=aruco.DICT_6X6_1000, border_fraction=0.2
    )
    location = data_dir / "calibration_ccube"
    detections, camera_res = detect_datapoints_in_imfile(
        f_loc=location, caching=False, calibration_target=target, threads=1
    )
    initial = run_initial_calibration(detections, target, camera_res, save=False)
    initial.set_resolutions_from_file(floc=location)

    def solve(solver):
        handler = th.TemplateBundleHandler(
            camset=initial, target=target, detection=detections,
            options={"outliers": "n", "verbosity": 0, "solver": solver},
        )
        result, _ = run_bundle_adjustment(handler, threads=cpu_count())
        return handler, result

    handler, schur = solve("schur")
    usable, reason = can_use_schur(handler)
    assert usable, f"the Ccube template handler should support the Schur solver: {reason}"

    _, trf = solve("trf")

    def rpe(residuals):
        return float(np.mean(np.linalg.norm(np.reshape(residuals, (-1, 2)), axis=1)))

    # The two take different paths to the minimum, so compare where they land
    # rather than the iterates. Schur should never be the worse of the two.
    assert rpe(schur["fun"]) <= rpe(trf["fun"]) + 0.02
    assert schur["jac"] is not None, "the result must carry a jacobian for the camera set"
    assert schur["x"].shape == trf["x"].shape


class _NoSolver:
    """Enough of a SchurSolver to reach the exits that never step."""

    spec = type("_Spec", (), {"keep_free": np.array([], dtype=int),
                              "elim_free": np.array([], dtype=int)})()


@pytest.mark.parametrize(("jac", "max_iter", "status"), [
    (lambda x: np.zeros((1, 2, 3)), 0, 1),
    (lambda x: np.full((1, 2, 3), np.nan), 5, 5),
])
def test_a_solve_that_never_steps_still_returns_a_result(jac, max_iter, status):
    """Both exits leave before a gradient exists.

    A non-finite jacobian is a documented outcome -- a point has moved onto
    or behind a camera plane -- and reporting it used to raise
    UnboundLocalError from the optimality field instead.
    """
    result = levenberg_marquardt(
        loss=lambda x: np.zeros(2), jac_blocks=jac, x0=np.zeros(3),
        solver=_NoSolver(), max_iter=max_iter)

    assert result.status == status
    assert result.success is False
    assert result.optimality == np.inf
