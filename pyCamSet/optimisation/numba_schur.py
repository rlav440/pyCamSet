"""
A Schur complement Levenberg-Marquardt solver built on numba and numpy.

This is the same algorithm as :mod:`pyCamSet.optimisation.jax_schur`, with no
JAX dependency.  Bundle adjustment normal equations split into

    | B    E | | dc |     | -v |
    | E^T  C | | dp |  =  | -w |

where ``C`` is block diagonal -- one small block per pose or per target point,
because such a parameter only appears in residuals carrying its own index.
Eliminating it gives the reduced camera system

    (B - E C^-1 E^T) dc = -v + E C^-1 w
    dp = -C^-1 (w + E^T dc)

The split of work between the two runtimes follows where each one wins:

*   the **accumulation** of B, E, C, v and w is an irregular scatter over the
    detections, which is exactly what a compiled loop is good at and what
    array libraries are bad at, so it is one ``njit`` kernel;
*   the **reduction and solve** are dense linear algebra on small matrices, so
    they are left to numpy and hence to BLAS.

Fixed parameters are pinned rather than compressed out: a fixed parameter
keeps its slot, its row and column become the identity and its gradient zero,
so its step is exactly zero.  That keeps every eliminated block the same size,
which is what makes the batched inverse possible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import logging

logger = logging.getLogger(__name__)

import numpy as np
from numba import njit, prange
from scipy.optimize import OptimizeResult


# ---------------------------------------------------------------------------
# problem structure
# ---------------------------------------------------------------------------
@dataclass
class ParamGroup:
    """
    One block of parameters, e.g. the intrinsics of every camera.

    :param name: label, used in reprs and errors
    :param base: ``(n_rows, n_par)`` full parameter values, including fixed ones
    :param unfixed: boolean mask, either per row or per individual element
    :param index: ``(n_det,)`` which row each detection reads
    """

    name: str
    base: np.ndarray
    unfixed: np.ndarray
    index: np.ndarray
    element_unfixed: np.ndarray = field(init=False)

    def __post_init__(self):
        self.base = np.ascontiguousarray(self.base, dtype=np.float64)
        self.n_rows, self.n_par = self.base.shape
        unfixed = np.asarray(self.unfixed, dtype=bool)
        if unfixed.size == self.n_rows:
            unfixed = np.repeat(unfixed, self.n_par)
        elif unfixed.size != self.base.size:
            raise ValueError(
                f"{self.name}: mask of {unfixed.size} entries fits neither "
                f"{self.n_rows} rows nor {self.base.size} elements"
            )
        self.element_unfixed = unfixed
        self.n_free = int(unfixed.sum())
        self.index = np.asarray(self.index, dtype=np.int64)


@dataclass
class SchurSpec:
    """The static structure the solver works against."""

    keep_cols: np.ndarray     # (n_det, n_keep) retained columns, full space
    elim_block: np.ndarray    # (n_det,) which eliminated block
    n_keep_full: int
    n_elim_blocks: int
    elim_size: int
    keep_fixed: np.ndarray    # (n_keep_full,) bool
    elim_fixed: np.ndarray    # (n_elim_blocks, elim_size) bool
    keep_free: np.ndarray
    elim_free: np.ndarray
    group_id: np.ndarray      # (n_det,) which distinct keep_cols pattern
    group_cols: np.ndarray    # (n_groups, n_keep) the distinct patterns

    @property
    def n_det(self) -> int:
        return self.keep_cols.shape[0]

    @property
    def n_groups(self) -> int:
        return self.group_cols.shape[0]

    @property
    def n_keep(self) -> int:
        return self.keep_cols.shape[1]

    @property
    def row_width(self) -> int:
        return self.n_keep + self.elim_size


def spec_from_groups(groups: Sequence[ParamGroup]) -> SchurSpec:
    """
    Build a :class:`SchurSpec` from the parameter groups of a problem.

    The *last* group is the one eliminated, which for every chain pyCamSet
    ships is the block diagonal one: the per image pose for a fixed target,
    the per point geometry for a self calibration.
    """
    *keep_groups, elim = groups
    cols, fixed, offset = [], [], 0
    for g in keep_groups:
        cols.append(offset + g.index[:, None] * g.n_par + np.arange(g.n_par)[None, :])
        fixed.append(~g.element_unfixed)
        offset += g.base.size
    keep_cols = np.ascontiguousarray(np.concatenate(cols, axis=1), dtype=np.int64)
    keep_fixed = np.concatenate(fixed)
    # The retained columns depend only on the camera and image index, so there
    # are far fewer distinct patterns than detections. Accumulating B into a
    # small dense block per pattern keeps the inner loop in cache and defers the
    # scattered writes to one pass over the patterns.
    group_cols, group_id = np.unique(keep_cols, axis=0, return_inverse=True)
    return SchurSpec(
        keep_cols=keep_cols,
        elim_block=np.ascontiguousarray(elim.index, dtype=np.int64),
        n_keep_full=offset,
        n_elim_blocks=elim.n_rows,
        elim_size=elim.n_par,
        keep_fixed=keep_fixed,
        elim_fixed=(~elim.element_unfixed).reshape((elim.n_rows, elim.n_par)),
        keep_free=~keep_fixed,
        elim_free=elim.element_unfixed,
        group_id=np.ascontiguousarray(group_id.reshape(-1), dtype=np.int64),
        group_cols=np.ascontiguousarray(group_cols, dtype=np.int64),
    )


def free_parameter_order(groups: Sequence[ParamGroup]) -> np.ndarray:
    """The free parameter mask, in the order ``build_param_list`` produces."""
    return np.concatenate([g.element_unfixed for g in groups])


# ---------------------------------------------------------------------------
# accumulation
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _accumulate(blocks, resid, keep_cols, elim_block, group_id, group_cols,
                n_keep_full, n_elim_blocks, n_groups):
    """
    Sum the per detection contributions into the four normal equation blocks.

    B and v go into a small dense block per distinct retained-column pattern,
    which is the hot part: it is O(n_keep**2) per residual and, accumulated
    directly into the full (n_keep_full, n_keep_full) matrix, every write is a
    scattered one. There are only as many patterns as camera and image pairs,
    so the per pattern blocks stay in cache and the scattered writes happen
    once per pattern at the end instead of once per detection.
    """
    n_det = blocks.shape[0]
    nk = keep_cols.shape[1]
    ne = blocks.shape[2] - nk

    Bg = np.zeros((n_groups, nk, nk))
    vg = np.zeros((n_groups, nk))
    E = np.zeros((n_keep_full, n_elim_blocks, ne))
    C = np.zeros((n_elim_blocks, ne, ne))
    w = np.zeros((n_elim_blocks, ne))

    for d in range(n_det):
        g = group_id[d]
        j = elim_block[d]
        for a in range(2):
            r = resid[d, a]
            for i in range(nk):
                jk = blocks[d, a, i]
                vg[g, i] += jk * r
                for k in range(nk):
                    Bg[g, i, k] += jk * blocks[d, a, k]
                ci = keep_cols[d, i]
                for l in range(ne):
                    E[ci, j, l] += jk * blocks[d, a, nk + l]
            for i in range(ne):
                je = blocks[d, a, nk + i]
                w[j, i] += je * r
                for l in range(ne):
                    C[j, i, l] += je * blocks[d, a, nk + l]

    B, v = _scatter_groups(Bg, vg, group_cols, n_keep_full)
    return B, E, C, v, w


@njit(cache=True, fastmath=True)
def _scatter_groups(Bg, vg, group_cols, n_keep_full):
    """Place the per pattern blocks into the full retained matrix."""
    n_groups, nk, _ = Bg.shape
    B = np.zeros((n_keep_full, n_keep_full))
    v = np.zeros(n_keep_full)
    for g in range(n_groups):
        for i in range(nk):
            ci = group_cols[g, i]
            v[ci] += vg[g, i]
            for k in range(nk):
                B[ci, group_cols[g, k]] += Bg[g, i, k]
    return B, v


@njit(cache=True, fastmath=True, parallel=True)
def _accumulate_parallel(blocks, resid, keep_cols, elim_block, group_id, group_cols,
                         n_keep_full, n_elim_blocks, n_groups, n_threads):
    """
    As :func:`_accumulate`, with each thread accumulating privately.

    The reduction is over the per thread copies, so it costs
    ``n_threads * (n_groups * n_keep**2 + n_keep_full * n_elim * elim)``. That
    dominates until the detection count is large; see
    ``SchurSolver.PARALLEL_MIN_DETECTIONS``.
    """
    n_det = blocks.shape[0]
    nk = keep_cols.shape[1]
    ne = blocks.shape[2] - nk

    Bgt = np.zeros((n_threads, n_groups, nk, nk))
    vgt = np.zeros((n_threads, n_groups, nk))
    Et = np.zeros((n_threads, n_keep_full, n_elim_blocks, ne))
    Ct = np.zeros((n_threads, n_elim_blocks, ne, ne))
    wt = np.zeros((n_threads, n_elim_blocks, ne))

    chunk = (n_det + n_threads - 1) // n_threads
    for t in prange(n_threads):
        lo = t * chunk
        hi = min(lo + chunk, n_det)
        for d in range(lo, hi):
            g = group_id[d]
            j = elim_block[d]
            for a in range(2):
                r = resid[d, a]
                for i in range(nk):
                    jk = blocks[d, a, i]
                    vgt[t, g, i] += jk * r
                    for k in range(nk):
                        Bgt[t, g, i, k] += jk * blocks[d, a, k]
                    ci = keep_cols[d, i]
                    for l in range(ne):
                        Et[t, ci, j, l] += jk * blocks[d, a, nk + l]
                for i in range(ne):
                    je = blocks[d, a, nk + i]
                    wt[t, j, i] += je * r
                    for l in range(ne):
                        Ct[t, j, i, l] += je * blocks[d, a, nk + l]

    Bg = Bgt[0]
    vg = vgt[0]
    E = Et[0]
    C = Ct[0]
    w = wt[0]
    for t in range(1, n_threads):
        Bg += Bgt[t]
        vg += vgt[t]
        E += Et[t]
        C += Ct[t]
        w += wt[t]
    B, v = _scatter_groups(Bg, vg, group_cols, n_keep_full)
    return B, E, C, v, w


# ---------------------------------------------------------------------------
# the solver
# ---------------------------------------------------------------------------
class SchurSolver:
    """
    Forms and solves the reduced camera system for a fixed problem structure.

    :param spec: the static structure, from :func:`spec_from_groups`
    :param threads: ``"auto"`` (the default) picks serial or parallel from the
        detection count; an int forces the thread count, where 1 is serial.

    The parallel kernel gives each thread a private set of accumulators and
    reduces at the end, so it carries both a launch cost and a reduction cost.
    Measured on a 10 core machine those dominate below roughly 15k detections
    -- at 4.9k the parallel kernel is 2x *slower*, at 313k it is 3.1x faster --
    hence the crossover rather than always threading.
    """

    PARALLEL_MIN_DETECTIONS = 15_000

    def __init__(self, spec: SchurSpec, threads: int | str = "auto"):
        self.spec = spec
        if threads == "auto":
            import numba
            threads = (numba.get_num_threads()
                       if spec.n_det >= self.PARALLEL_MIN_DETECTIONS else 1)
        self.threads = int(threads)
        self._eye_keep = np.eye(spec.n_keep_full)
        self._eye_elim = np.eye(spec.elim_size)
        self._pin_keep = spec.keep_fixed[:, None] | spec.keep_fixed[None, :]
        ef = spec.elim_fixed
        self._pin_elim = ef[:, :, None] | ef[:, None, :]
        self._diag_keep = np.arange(spec.n_keep_full)
        self._diag_elim = np.arange(spec.elim_size)

    def normal_equations(self, blocks, residuals):
        """``(B, E, C, v, w)`` of the pinned normal equations."""
        s = self.spec
        blocks = np.ascontiguousarray(blocks, dtype=np.float64)
        residuals = np.ascontiguousarray(residuals, dtype=np.float64)
        if self.threads > 1:
            B, E, C, v, w = _accumulate_parallel(
                blocks, residuals, s.keep_cols, s.elim_block, s.group_id, s.group_cols,
                s.n_keep_full, s.n_elim_blocks, s.n_groups, self.threads)
        else:
            B, E, C, v, w = _accumulate(
                blocks, residuals, s.keep_cols, s.elim_block, s.group_id, s.group_cols,
                s.n_keep_full, s.n_elim_blocks, s.n_groups)

        # pin the fixed parameters
        C = np.where(self._pin_elim, self._eye_elim[None, :, :], C)
        w = np.where(s.elim_fixed, 0.0, w)
        E = np.where(s.elim_fixed[None, :, :], 0.0, E)
        E = np.where(s.keep_fixed[:, None, None], 0.0, E)
        B = np.where(self._pin_keep, self._eye_keep, B)
        v = np.where(s.keep_fixed, 0.0, v)
        return B, E, C, v, w

    def step_from_parts(self, B, E, C, v, w, lam: float) -> np.ndarray:
        """
        The LM step in free parameter space, for an assembled system.

        A non finite step is a valid outcome, not an error: a wild iterate can
        overflow the reduced system, and the caller responds by damping harder.
        """
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return self._step_from_parts(B, E, C, v, w, lam)

    def _step_from_parts(self, B, E, C, v, w, lam: float) -> np.ndarray:
        s = self.spec
        nkf, ne, nb = s.n_keep_full, s.elim_size, s.n_elim_blocks

        db, dc = _damping_scales(B, C)
        Bd = B.copy()
        Bd[self._diag_keep, self._diag_keep] += lam * db
        Cd = C.copy()
        Cd[:, self._diag_elim, self._diag_elim] += lam * dc

        Cinv = np.linalg.inv(Cd)                                # (nb, ne, ne)

        # Y = E C^-1, as one batched matmul over the blocks rather than an
        # einsum, then flattened so the two big contractions reach BLAS as
        # single (n_keep_full x nb*ne) GEMMs
        Ef = E.reshape((nkf, nb * ne))
        Eb = np.ascontiguousarray(E.transpose(1, 0, 2))         # (nb, nkf, ne)
        Yf = np.ascontiguousarray(
            np.matmul(Eb, Cinv).transpose(1, 0, 2)).reshape((nkf, nb * ne))
        S = Bd - Yf @ Ef.T
        g = -v + Yf @ w.reshape(-1)

        S = np.where(self._pin_keep, self._eye_keep, S)
        g = np.where(s.keep_fixed, 0.0, g)

        dc = np.linalg.solve(S, g)
        rhs = w + (Ef.T @ dc).reshape((nb, ne))
        dp = -np.matmul(Cinv, rhs[:, :, None])[:, :, 0]
        return np.concatenate([dc[s.keep_free], dp.reshape(-1)[s.elim_free]])

    def step(self, blocks, residuals, lam: float) -> np.ndarray:
        """Assemble and solve in one call."""
        B, E, C, v, w = self.normal_equations(blocks, residuals)
        return self.step_from_parts(B, E, C, v, w, lam)


# ---------------------------------------------------------------------------
# Levenberg-Marquardt
# ---------------------------------------------------------------------------
def levenberg_marquardt(
    loss: Callable[[np.ndarray], np.ndarray],
    jac_blocks: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    solver: SchurSolver,
    max_iter: int = 200,
    lam0: float = 1e-3,
    lam_min: float = 1e-10,
    lam_max: float = 1e10,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
    jac_csr: Callable | None = None,
    verbose: bool = False,
    callback: Callable[[int, float, np.ndarray], None] | None = None,
) -> OptimizeResult:
    """
    Minimise ``0.5 * ||loss(x)||^2`` with the Schur complement step.

    Steps are accepted on the gain ratio (actual over predicted reduction)
    rather than on any decrease, which is what keeps the damping schedule
    stable when a pose or point is only seen once and its block is close to
    singular.

    :param loss: ``x -> (2 * n_det,)`` residuals
    :param jac_blocks: ``x -> (n_det, 2, row_width)`` per detection blocks
    :param solver: the :class:`SchurSolver` for this structure
    :param jac_csr: optional ``x -> csr_array`` used only to populate the
        ``jac`` field of the result, which pyCamSet stores on the camera set
    :param callback: called after each accepted step with the iteration
        number, the cost, and the residuals, for progress reporting
    :returns: a :class:`scipy.optimize.OptimizeResult`
    """
    x = np.array(x0, dtype=np.float64)
    r = loss(x)
    cost = _cost(r)
    lam = lam0
    nfev, njev, nit, status, message = 1, 0, 0, 2, "ftol reached"

    for nit in range(1, max_iter + 1):
        blocks = jac_blocks(x)
        njev += 1
        if not np.all(np.isfinite(blocks)):
            status, message = 5, (
                "the jacobian is not finite at the current parameters, which "
                "usually means a point has moved onto or behind a camera plane")
            logger.warning(message)
            break
        B, E, C, v, w = solver.normal_equations(blocks, r.reshape((-1, 2)))
        g_free = -np.concatenate([
            v[solver.spec.keep_free], w.reshape(-1)[solver.spec.elim_free]])

        accepted = False
        while lam < lam_max:
            dx = solver.step_from_parts(B, E, C, v, w, lam)
            if not np.all(np.isfinite(dx)):
                lam *= 10.0
                continue
            r_new = loss(x + dx)
            nfev += 1
            cost_new = _cost(r_new)
            # predicted reduction for the damped Gauss-Newton model. A badly
            # conditioned step can overflow this; a non finite prediction is
            # treated as no prediction, which rejects the step and damps more.
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                pred = 0.5 * float(dx @ (lam * dx * _diag_scale(solver, B, C) + g_free))
            rho = (cost - cost_new) / pred if np.isfinite(pred) and pred > 0 else -1.0
            if not np.isfinite(cost_new):
                rho = -1.0

            if cost_new < cost and rho > 0:
                dcost, dx_norm = cost - cost_new, float(np.linalg.norm(dx))
                x, r, cost = x + dx, r_new, cost_new
                lam = max(lam * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3), lam_min)
                accepted = True
                if verbose:
                    # debug, not info: the progress bar carries this for a
                    # person watching and the summary carries the outcome, so
                    # the raw trace is for diagnosis. Ask for it with the
                    # 'verbosity' option at 3.
                    logger.debug(
                        f"    it {nit:3d} cost {cost:.9e} "
                        f"lam {lam:.2e} rho {rho:.3f}")
                if callback is not None:
                    callback(nit, cost, r)
                if dcost < ftol * cost:
                    status, message = 2, "ftol reached"
                elif dx_norm < xtol * (xtol + float(np.linalg.norm(x))):
                    status, message = 3, "xtol reached"
                else:
                    status = 0
                break
            lam *= 10.0

        if not accepted:
            status, message = 4, "damping hit lam_max without an accepted step"
            break
        if status in (2, 3):
            break
    else:
        status, message = 1, "max_iter reached"

    return OptimizeResult(
        x=x, fun=r, cost=cost, jac=jac_csr(x) if jac_csr is not None else None,
        nfev=nfev, njev=njev, nit=nit, status=status, message=message,
        success=status in (0, 2, 3), optimality=float(np.max(np.abs(g_free))),
    )


def _cost(r) -> float:
    """
    Half the sum of squares, or infinity when that overflows.

    A rejected trial step can put a point on the camera plane and send the
    residuals to 1e150 or beyond; squaring them overflows. Infinity is the
    right answer for the caller, which is only comparing costs.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        c = 0.5 * float(r @ r)
    return c if np.isfinite(c) else np.inf


DAMPING_FLOOR = 1e-8


def _damping_scales(B, C):
    """
    The per parameter damping weights, floored away from zero.

    Marquardt damping scales a parameter's own curvature, which fails for a
    parameter that is barely constrained: a target point seen by a single
    camera has a rank 2 block, so scaling its diagonal leaves the block close
    to singular and the batched inverse overflows. Flooring each weight at a
    small fraction of the mean curvature makes the damped system safely
    invertible while leaving well constrained parameters untouched.
    """
    db = np.diag(B)
    dc = np.diagonal(C, axis1=1, axis2=2)
    b_floor = DAMPING_FLOOR * max(float(db.mean()), np.finfo(float).tiny)
    c_floor = DAMPING_FLOOR * max(float(dc.mean()), np.finfo(float).tiny)
    return np.maximum(db, b_floor), np.maximum(dc, c_floor)


def _diag_scale(solver: SchurSolver, B, C):
    """The damping diagonal, in free parameter order, matching the solve."""
    s = solver.spec
    db, dc = _damping_scales(B, C)
    return np.concatenate([db[s.keep_free], dc.reshape(-1)[s.elim_free]])
