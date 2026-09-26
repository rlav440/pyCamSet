from __future__ import annotations
import logging

logger = logging.getLogger(__name__)
import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, approx_fprime, OptimizeResult

from typing import TYPE_CHECKING

import pyCamSet.optimisation.template_handler as th
from pyCamSet.optimisation.numba_schur import (
    SchurSolver, levenberg_marquardt, spec_from_groups)

from pyCamSet.calibration_targets import TargetDetection
from pyCamSet.utils.calibration_report import (
    CalibrationReport, HIGH_INITIAL_ERROR_PX, reprojection_residuals)
from pyCamSet.utils.progress import OptimisationProgress

#: How many threads BLAS may use while a bundle adjustment runs.
#:
#: One.  An iteration alternates the compiled kernels with a little dense
#: linear algebra, and between its calls OpenBLAS leaves its threads spinning
#: rather than sleeping.  On a machine with more logical cores than the solve
#: can use, those spinners preempt the kernels -- which do no BLAS at all --
#: and cost far more than the linear algebra they belong to: the loss kernel
#: of a Ccube self calibration runs in 1.9 ms with BLAS held to one thread and
#: 22.2 ms with OpenBLAS's default of one per core, on an 8 core machine.
#:
#: Nothing is lost by it.  The systems here are small -- the reduced camera
#: system of a self calibration is a few hundred square, and the Cholesky that
#: solves it took 4 ms of a 10 s solve -- and that is the size at which
#: threading a dense solve is already a loss rather than a win.
#:
#: macOS does not show this: its BLAS is Accelerate, which does not hold a
#: spinning pool, which is why the same solve is an order of magnitude faster
#: there on lesser hardware.  Setting OPENBLAS_NUM_THREADS is the workaround
#: available to anyone already affected; limiting the pool around the solve is
#: the fix, and needs no environment variable and no knowledge of the machine.
_BLAS_THREADS_DURING_SOLVE = 1

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:  # pragma: no cover - threadpoolctl is a declared dependency
    import contextlib

    def _threadpool_limits(limits=None, user_api=None):
        """
        Stand in for threadpoolctl, so its absence costs speed and not a run.

        It is a declared dependency, so this is for an environment that
        upgraded pyCamSet without its dependencies rather than for a supported
        configuration.
        """
        logger.warning(
            "threadpoolctl is not installed, so BLAS keeps its own thread "
            "count during the solve. On a machine whose BLAS holds a spinning "
            "thread pool this is several times slower; install threadpoolctl, "
            "or set OPENBLAS_NUM_THREADS=1.")
        return contextlib.nullcontext()

if TYPE_CHECKING:
    from pyCamSet.calibration_targets import AbstractTarget
    from pyCamSet.cameras import CameraSet, Camera


def make_optimisation_function(
        param_handler: th.TemplateBundleHandler,
        threads: int = 1,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]|None, np.ndarray]:
    """
    Takes a parameter handler and creates a callable cost function that evaluates the
    cost of a parameter array.

    :param param_handler: the param handler representing the optimisation
    :param threads: number of evaluation threads
    :return fn: the cost function
    """
    #
    logger.info("getting initial params")
    init_params = param_handler.get_initial_params()
    logger.info("Compiling the loss function")
    bundle_loss_fun = param_handler.make_loss_fun(threads)

    if param_handler.can_make_jac():
        logger.info(
            "Compiling the jacobian. The generated source is cached under "
        )
        bundle_loss_jac = param_handler.make_loss_jac(threads)
        check_jacobian_is_not_degenerate(bundle_loss_jac, init_params)
    else: 
        bundle_loss_jac = None

    return bundle_loss_fun, bundle_loss_jac, init_params


def check_jacobian_is_not_degenerate(bundle_loss_jac: Callable, init_params: np.ndarray):
    """
    Raises when the analytic jacobian cannot move some of the parameters.

    :param bundle_loss_jac: the compiled jacobian callable
    :param init_params: the parameters to evaluate it at
    """
    jac = bundle_loss_jac(init_params)
    jac = np.asarray(jac.todense()) if hasattr(jac, "todense") else np.asarray(jac)

    if jac.ndim != 2 or jac.shape[1] != init_params.size:
        logger.warning(
            "Skipping the jacobian degeneracy check: expected a "
            f"(n_residuals, {init_params.size}) jacobian, got shape {jac.shape}."
        )
        return

    dead = np.flatnonzero(np.all(jac == 0, axis=0))
    if dead.size:
        raise RuntimeError(
            f"The compiled jacobian is degenerate: {dead.size} of "
            f"{init_params.size} parameters have an all-zero column, so the "
            f"optimiser cannot adjust them. Columns: {dead.tolist()}.\n"
            "This is a code generation fault. Delete the cached templates in "
            "pyCamSet/optimisation/template_functions/ to force a rebuild; if "
            "it recurs, please report it with your platform and "
            "numpy/numba versions."
        )
    logger.info("Jacobian degeneracy check passed (no all-zero columns).")


def can_use_schur(param_handler) -> tuple[bool, str]:
    """
    Whether the Schur complement solver can drive this parameter handler.

    :param param_handler: the handler to check
    :return: usable, and the reason it is not when it is not
    """
    if param_handler.problem_opts.get("solver", "schur") != "schur":
        return False, f"solver option is {param_handler.problem_opts['solver']!r}"
    # The block solver is built from parameter_groups()/make_loss_blocks(),
    # which describe the reprojection residuals alone.  Prior rows appended
    # to the loss are invisible to it, so it would quietly optimise a
    # different problem from the one the residuals report.
    if getattr(param_handler, "has_lockbox_priors", lambda: False)():
        return False, "the problem carries lockbox prior residuals"
    for method in ("parameter_groups", "make_loss_blocks"):
        if not hasattr(param_handler, method):
            return False, f"the handler does not implement {method}()"
    try:
        groups = param_handler.parameter_groups()
    except Exception as e:                                   # a custom handler
        return False, f"parameter_groups() raised {type(e).__name__}: {e}"
    if len(groups) < 2:
        return False, "there is no block diagonal group to eliminate"
    n_free = int(sum(g.n_free for g in groups))
    n_params = param_handler.get_initial_params().size
    if n_free != n_params:
        return False, (f"the parameter groups describe {n_free} free parameters "
                       f"but the handler has {n_params}")
    return True, ""


def run_schur_bundle_adjustment(param_handler, loss_fn, bundle_jac, init_params,
                                threads: int) -> OptimizeResult:
    """
    Solve with Levenberg-Marquardt on the Schur reduced camera system.

    The parameters split into the ones shared between many residuals and one
    block diagonal group -- the per image pose, or the per point geometry --
    which is eliminated, leaving a system the size of the camera parameters
    alone. See :mod:`pyCamSet.optimisation.numba_schur`.

    :param param_handler: the handler describing the problem
    :param loss_fn: the residual callable
    :param bundle_jac: the CSR jacobian, stored on the result for the camera set
    :param init_params: the starting parameters
    :param threads: evaluation threads for the compiled kernels
    """
    groups = param_handler.parameter_groups()
    spec = spec_from_groups(groups)
    solver = SchurSolver(spec)
    blocks = param_handler.make_loss_blocks(threads)
    logger.info(
        f"Schur solver: eliminating {spec.n_elim_blocks} blocks of "
        f"{spec.elim_size}x{spec.elim_size}, leaving a "
        f"{int(spec.keep_free.sum())} parameter reduced system "
        f"(from {init_params.size})"
    )
    with OptimisationProgress() as progress:
        return levenberg_marquardt(
            loss_fn, blocks, init_params, solver,
            max_iter=param_handler.problem_opts["max_nfev"],
            jac_csr=bundle_jac,
            verbose=param_handler.problem_opts["verbosity"] > 1,
            callback=progress.update,
        )


def get_bundle_adjustment_stats(
        optimisation: OptimizeResult,
        init_params: np.ndarray,
        init_err: np.ndarray,
        elapsed_sec: float,
        param_handler=None,
) -> dict:
    """
    A finished solve as a flat dictionary, for the GUI and the study driver.

    :class:`~pyCamSet.utils.calibration_report.CalibrationReport` is the
    richer view of the same run and is what a person reads.  This is the
    machine-readable one: the study driver ranks trials on ``final_euclid``
    and the phase tabs write these keys straight into their run metadata,
    so the names here are a contract with those callers.

    Reprojection residuals are separated from any lockbox priors first, so
    the two error figures stay comparable with each other and with the
    initial error logged by the solve.

    :param optimisation: the scipy style result the solver returned
    :param init_params: the parameters the solve started from
    :param init_err: the residuals at ``init_params``
    :param elapsed_sec: wall clock seconds the solve took
    :param param_handler: the handler that defined the problem.  When it
        exposes the per pose diagnostics filled in by ``calc_initial_params``
        and ``find_and_exclude_transform_outliers``, the per pose breakdowns
        are added; when it does not, those keys are simply absent.
    :return: the statistics, keyed as described above
    """
    init_reprojection, _ = reprojection_residuals(init_err, param_handler)
    init_euclid = float(np.mean(np.linalg.norm(
        np.reshape(init_reprojection, (-1, 2)), axis=1)))
    final_reprojection, final_priors = reprojection_residuals(
        optimisation.fun, param_handler)
    final_euclid = float(np.mean(np.linalg.norm(
        np.reshape(final_reprojection, (-1, 2)), axis=1)))
    initial_reprojection_cost = 0.5 * float(init_reprojection @ init_reprojection)
    final_reprojection_cost = 0.5 * float(final_reprojection @ final_reprojection)

    stats = {
        "initial_euclid": init_euclid,
        "final_euclid": final_euclid,
        "initial_reprojection_cost": initial_reprojection_cost,
        "final_reprojection_cost": final_reprojection_cost,
        "param_count": int(np.size(init_params)),
        "observation_count": int(np.size(final_reprojection) // 2),
        "prior_residual_count": (
            int(np.size(final_priors)) if final_priors is not None else 0),
        "elapsed_sec": float(elapsed_sec),
        "status": _stat_int(getattr(optimisation, "status", None)),
        "success": bool(getattr(optimisation, "success", False)),
        "message": str(getattr(optimisation, "message", "")),
        "nfev": _stat_int(getattr(optimisation, "nfev", None)),
    }

    if param_handler is None:
        return stats

    # These are indexed by "global_im_num", a pose index shared across every
    # camera in the rig, so there is no one camera to attribute them to and
    # the pose index is the identifier.
    per_pose_error = getattr(param_handler, "initial_per_im_error", None)
    if per_pose_error is not None:
        stats["per_pose_initial_error_px"] = [
            {"pose": int(i), "initial_error_px": float(v)}
            for i, v in enumerate(np.asarray(per_pose_error, dtype=float))
        ]
    for key, attr in (
            ("outlier_poses_before_rejection",
             "missing_poses_before_outlier_rejection"),
            ("outlier_poses_after_rejection",
             "missing_poses_after_outlier_rejection"),
    ):
        missing = getattr(param_handler, attr, None)
        if missing is not None:
            stats[key] = [int(i) for i in np.where(np.asarray(missing))[0]]

    return stats


def _stat_int(value) -> int:
    """A solver field as an int, with the 0 the metadata writers expect."""
    return 0 if value is None else int(value)


def _solve_bundle_adjustment(
        param_handler: th.TemplateBundleHandler,
        threads: int = 1,
) -> tuple[OptimizeResult, CameraSet, dict]:
    """
    The solve behind both public entry points.

    Kept as one path so a run reports the same numbers whether it was
    started from the library or from a GUI phase: the two wrappers differ
    only in how much of the result they hand back.

    :param param_handler: The parameter handler that represents the optimisation
    :param threads: evaluation threads for the compiled kernels
    :return: the optimisation, the argmin defined CameraSet, and the statistics
    """
    logger.info("Making optimisation problem")
    loss_fn, bundle_jac, init_params = make_optimisation_function(
        param_handler, threads
    )

    init_err = loss_fn(init_params)
    init_reprojection, _ = reprojection_residuals(init_err, param_handler)
    init_euclid = np.mean(np.linalg.norm(
        np.reshape(init_reprojection, (-1, 2)), axis=1))
    logger.info(f'found {len(init_params)} parameters')
    logger.info(f'found {len(init_reprojection) // 2} control points')
    logger.info(f'Initial Euclidean error: {init_euclid:.2f} px')

    if (init_euclid > HIGH_INITIAL_ERROR_PX) or np.isnan(init_euclid):
        logger.warning(
            f"Initial error of {init_euclid:.2f} px is above the "
            f"{HIGH_INITIAL_ERROR_PX:.0f} px this check expects: verify the "
            f"initial parametisation is sensible. This usually indicates a "
            f"camera or the target has been placed incorrectly.")
        # param_handler.check_params(init_params)

    start = time.time()
    usable, reason = can_use_schur(param_handler)
    requested_loss = param_handler.problem_opts.get("loss", "linear")
    # The custom Schur path only minimises raw residuals; use SciPy when a
    # nonlinear loss is requested so the configured loss is actually applied.
    use_schur = (
        usable and bundle_jac is not None and requested_loss == "linear"
    )
    # Held around the whole solve rather than around the linear algebra: what
    # the spinning pool costs is the kernels between the BLAS calls, not the
    # BLAS calls themselves.  See _BLAS_THREADS_DURING_SOLVE.  Both solvers
    # alternate the same way, so both are inside it.
    with _threadpool_limits(limits=_BLAS_THREADS_DURING_SOLVE, user_api="blas"):
        if use_schur:
            solver = "schur"
            optimisation = run_schur_bundle_adjustment(
                param_handler, loss_fn, bundle_jac, init_params, threads)
        else:
            solver = "trf"
            if bundle_jac is not None and param_handler.problem_opts.get(
                    "solver", "schur") == "schur":
                logger.warning(f"Falling back to the trust region solver: {reason}")
            bounds = (-np.inf, np.inf)
            if hasattr(param_handler, "get_lockbox_bounds"):
                bounds = param_handler.get_lockbox_bounds(len(init_params))
            optimisation = least_squares(
                loss_fn,
                init_params,
                verbose=param_handler.problem_opts['verbosity'],
                jac= bundle_jac if bundle_jac is not None else "2-point", #pass the function for the jacobian if it exists
                max_nfev=param_handler.problem_opts["max_nfev"],
                x_scale='jac',
                xtol=1e-4,
                loss=requested_loss,
                f_scale=float(param_handler.problem_opts.get("f_scale", 1.0)),
                bounds=bounds,
            )
    end = time.time()

    report = CalibrationReport.from_optimisation(
        optimisation, param_handler,
        initial_error_px=init_euclid, duration_s=end - start, solver=solver,
    )
    logger.info("\n" + report.summary())

    camset = param_handler.get_camset(optimisation.x)
    camset.set_calibration_history(optimisation, param_handler, report=report)

    stats = get_bundle_adjustment_stats(
        optimisation, init_params, init_err, end - start,
        param_handler=param_handler,
    )
    return optimisation, camset, stats


def run_bundle_adjustment(param_handler: th.TemplateBundleHandler,
                          threads: int = 1) -> tuple[OptimizeResult, CameraSet]:
    """
    A function that takes an abstract parameter handler, turns it into a cost function, and returns the
    optimisation results and the camera set that minimises the optimisation problem defined by the parameter handler.

    :param param_handler: The parameter handler that represents the optimisation
    :return: The output of the calibration and the argmin defined CameraSet
    """
    optimisation, camset, _stats = _solve_bundle_adjustment(
        param_handler, threads)
    return optimisation, camset


def run_bundle_adjustment_with_stats(
        param_handler: th.TemplateBundleHandler,
        threads: int = 1,
) -> tuple[OptimizeResult, CameraSet, dict]:
    """
    :func:`run_bundle_adjustment`, and the run statistics alongside it.

    The GUI phases and the optimisation study need the solve's numbers as
    data rather than as a logged report, and they need them without solving
    twice.

    :param param_handler: The parameter handler that represents the optimisation
    :param threads: evaluation threads for the compiled kernels
    :return: the optimisation, the argmin defined CameraSet, and the
        statistics described in :func:`get_bundle_adjustment_stats`
    """
    return _solve_bundle_adjustment(param_handler, threads)

