from __future__ import annotations
import logging

logger = logging.getLogger(__name__)
import time
from copy import copy
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, approx_fprime, OptimizeResult

from typing import TYPE_CHECKING

import pyCamSet.utils.general_utils as gu
import pyCamSet.optimisation.compiled_helpers as ch
import pyCamSet.optimisation.template_handler as th
from pyCamSet.optimisation.numba_schur import (
    SchurSolver, levenberg_marquardt, spec_from_groups)

from pyCamSet.calibration_targets import TargetDetection
from pyCamSet.utils.calibration_report import (
    CalibrationReport, HIGH_INITIAL_ERROR_PX)
from pyCamSet.utils.progress import OptimisationProgress
    
if TYPE_CHECKING:
    from pyCamSet.calibration_targets import AbstractTarget
    from pyCamSet.cameras import CameraSet, Camera


def _split_residuals(fun, param_handler):
    """Separate reprojection residuals from optional lockbox priors."""
    residuals = np.asarray(fun)
    base_count = 0
    if param_handler is not None:
        base_count = int(getattr(
            param_handler, "get_base_residual_count", lambda: 0)())
    if base_count <= 0 or base_count >= residuals.size:
        return residuals, None
    return residuals[:base_count], residuals[base_count:]


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
    base_data = param_handler.get_detection_data(flatten=True)
    logger.info("Compiling the loss function")
    bundle_loss_fun = param_handler.make_loss_fun(threads)

    if param_handler.can_make_jac():
        logger.info(
            "Compiling the jacobian. The generated source is cached under "
            "optimisation/template_functions, so this is slow only the first "
            "time a problem of this shape is solved.")
        bundle_loss_jac = param_handler.make_loss_jac(threads)
        check_jacobian_is_not_degenerate(bundle_loss_jac, init_params)
    else: 
        bundle_loss_jac = None

    return bundle_loss_fun, bundle_loss_jac, init_params


def check_jacobian_is_not_degenerate(bundle_loss_jac: Callable, init_params: np.ndarray):
    """
    Raises when the analytic jacobian cannot move some of the parameters.

    The jacobian is generated Python source cached on disk, so a bad one is
    reused silently: the optimiser converges to a worse answer with no error.
    On macos-14 this cost the Ccube calibration 3.5 px of reprojection error
    (6.17 px against 2.62 px on x86_64) and was only found by comparing against
    a numeric jacobian.

    matmul_map.check_all_params_reach_the_output catches this when the template
    is generated. This second check runs on the jacobian actually in use, so it
    also catches a template cached by an older version of pyCamSet or generated
    on a different machine sharing the install.

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


def run_bundle_adjustment(param_handler: TemplateBundleHandler,
                          threads: int = 1) -> tuple[OptimizeResult, CameraSet]:
    """
    A function that takes an abstract parameter handler, turns it into a cost function, and returns the
    optimisation results and the camera set that minimises the optimisation problem defined by the parameter handler.

    :param param_handler: The parameter handler that represents the optimisation
    :return: The output of the calibration and the argmin defined CameraSet
    """
    logger.info("Making optimisation problem")
    loss_fn, bundle_jac, init_params = make_optimisation_function(
        param_handler, threads
    )

    init_err = loss_fn(init_params)
    # split before measuring: with a lockbox the residual vector carries
    # prior terms after the reprojections, and averaging over both reports
    # an initial error that is neither one nor the other.
    init_reprojection, _ = _split_residuals(init_err, param_handler)
    init_euclid = np.mean(np.linalg.norm(
        np.reshape(init_reprojection, (-1, 2)), axis=1))
    logger.info(f'found {len(init_params)} parameters')
    logger.info(f'found {len(init_reprojection) // 2} control points')
    logger.info(f'Initial Euclidean error: {init_euclid:.2f} px')

    # raise ValueError
    # test = lambda : loss_fn(init_params)
    # gu.benchmark(test, repeats=100)

    # bundle_jac(init_params)
    # test = lambda : bundle_jac(init_params)
    # gu.benchmark(test, repeats=100)

    if (init_euclid > HIGH_INITIAL_ERROR_PX) or np.isnan(init_euclid):
        logger.warning(
            f"Initial error of {init_euclid:.2f} px is above the "
            f"{HIGH_INITIAL_ERROR_PX:.0f} px this check expects: verify the "
            f"initial parametisation is sensible. This usually indicates a "
            f"camera or the target has been placed incorrectly.")
        # param_handler.check_params(init_params)

    start = time.time()
    usable, reason = can_use_schur(param_handler)
    if usable and bundle_jac is not None:
        solver = "schur"
        optimisation = run_schur_bundle_adjustment(
            param_handler, loss_fn, bundle_jac, init_params, threads)
    else:
        solver = "trf"
        if bundle_jac is not None and param_handler.problem_opts.get(
                "solver", "schur") == "schur":
            logger.warning(f"Falling back to the trust region solver: {reason}")
        optimisation = least_squares(
            loss_fn,
            init_params,
            verbose=param_handler.problem_opts['verbosity'],
            jac= bundle_jac if bundle_jac is not None else "2-point", #pass the function for the jacobian if it exists
            max_nfev=param_handler.problem_opts["max_nfev"],
            x_scale='jac',
            xtol=1e-4,
        )
    end = time.time()

    report = CalibrationReport.from_optimisation(
        optimisation, param_handler,
        initial_error_px=init_euclid, duration_s=end - start, solver=solver,
    )
    # the summary carries the final error, the timing and any concern that
    # would otherwise be a line of its own, so it is the whole report of the
    # solve rather than a footer under one.
    logger.info("\n" + report.summary())

    camset = param_handler.get_camset(optimisation.x)
    camset.set_calibration_history(optimisation, param_handler, report=report)

    return optimisation, camset

