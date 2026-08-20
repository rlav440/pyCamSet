from __future__ import annotations
import logging
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

from pyCamSet.calibration_targets import TargetDetection


def _split_residuals(fun, param_handler):
    """Split a residual vector into reprojection and lockbox-prior segments.

    Lockbox prior residuals are appended after the reprojection residuals by
    ``append_lockbox_residuals``.  The reprojection segment is always an even
    length (two residuals per observation), so it can be reshaped into
    ``(x, y)`` pairs for the Euclidean RPE.  The prior segment is returned
    separately and must not be folded into the per-row norm, which would
    dilute the RPE (and crash the reshape when the total length is odd).
    """
    fun = np.asarray(fun)
    base_count = 0
    if param_handler is not None:
        base_count = int(getattr(param_handler, "get_base_residual_count", lambda: 0)())
    if base_count <= 0 or base_count >= fun.size:
        return fun, None
    return fun[:base_count], fun[base_count:]
    
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
    logging.info("getting initial params")
    init_params = param_handler.get_initial_params()
    base_data = param_handler.get_detection_data(flatten=True)
    logging.info("Compiling the loss function")
    bundle_loss_fun = param_handler.make_loss_fun(threads)

    if param_handler.can_make_jac():
        logging.info("Compiling the jacobian")
        bundle_loss_jac = param_handler.make_loss_jac(threads)
    else: 
        bundle_loss_jac = None

    return bundle_loss_fun, bundle_loss_jac, init_params


def run_bundle_adjustment(param_handler: th.TemplateBundleHandler,
                          threads: int = 1) -> tuple[OptimizeResult, CameraSet]:
    """
    A function that takes an abstract parameter handler, turns it into a cost function, and returns the
    optimisation results and the camera set that minimises the optimisation problem defined by the parameter handler.

    :param param_handler: The parameter handler that represents the optimisation
    :return: The output of the calibration and the argmin defined CameraSet
    """
    logging.info("Making optimisation problem")
    loss_fn, bundle_jac, init_params = make_optimisation_function(
        param_handler, threads
    )

    init_err = loss_fn(init_params)
    init_reproj, _ = _split_residuals(init_err, param_handler)
    init_euclid = np.mean(np.linalg.norm(np.reshape(init_reproj, (-1, 2)), axis=1))
    logging.info(f'found {len(init_params):.2e} parameters')
    logging.info(f'found {len(init_err):.2e} control points')
    logging.info(f'Initial Euclidean error: {init_euclid:.2f} px')

    # raise ValueError
    # test = lambda : loss_fn(init_params)
    # gu.benchmark(test, repeats=100)

    # bundle_jac(init_params)
    # test = lambda : bundle_jac(init_params)
    # gu.benchmark(test, repeats=100)

    if (init_euclid > 100) or (init_euclid == np.nan):
        logging.critical("Found worryingly high/NaN initial error: check that the initial parametisation is sensible")
        logging.info(
            "This can often indicate failure to place a camera or target correctly, giving nonsensical errors.")
        # param_handler.check_params(init_params)

    # bundle_jac = lambda x: approx_fprime(x, loss_fn)
    bounds = param_handler.get_lockbox_bounds(len(init_params))
    start = time.time()
    optimisation = least_squares(
        loss_fn,
        init_params,
        verbose=param_handler.problem_opts['verbosity'],
        # method="lm",
        # tr_solver='lsmr',
        jac= bundle_jac if bundle_jac is not None else "2-point", #pass the function for the jacobian if it exists
        max_nfev=param_handler.problem_opts["max_nfev"],
        loss=param_handler.problem_opts.get("loss", "linear"),
        f_scale=param_handler.problem_opts.get("f_scale", 1.0),
        x_scale='jac',
        xtol=1e-4,
        bounds=bounds,
    )
    end = time.time()

    final_reproj, _ = _split_residuals(optimisation.fun, param_handler)
    final_euclid = np.mean(np.linalg.norm(np.reshape(final_reproj, (-1, 2)), axis=1))
    logging.info(f'Final Euclidean error: {final_euclid:.2f} px')
    logging.info(f'Optimisation took {end - start: .2f} seconds.')

    if final_euclid > 5:
        logging.critical("Remaining error is very large: please check the output results")
        # param_handler.check_params(optimisation.x)

    camset = param_handler.get_camset(optimisation.x)
    camset.set_calibration_history(optimisation, param_handler)

    init_err = loss_fn(optimisation.x)
    init_reproj, _ = _split_residuals(init_err, param_handler)
    init_euclid = np.mean(np.linalg.norm(np.reshape(init_reproj, (-1, 2)), axis=1))
    logging.info(f"Check test with a result of {init_euclid:.2f}")

    return optimisation, camset


def get_bundle_adjustment_stats(
    optimisation: OptimizeResult,
    init_params: np.ndarray,
    init_err: np.ndarray,
    elapsed_sec: float,
    param_handler: "th.TemplateBundleHandler | None" = None,
) -> dict:
    """Build a compact diagnostics dictionary for a completed bundle-adjustment run.

    :param param_handler: optional handler for the run. When provided and it exposes
        the per-pose diagnostic attributes populated during ``calc_initial_params``/
        ``find_and_exclude_transform_outliers`` (``initial_per_im_error``,
        ``missing_poses_before_outlier_rejection``, ``missing_poses_after_outlier_rejection``),
        the returned dict is additionally populated with per-pose breakdowns. This is
        purely additive: if the attributes are absent (e.g. an older/alternate handler
        or a code path that never called ``calc_initial_params``), the corresponding
        keys are simply omitted and the rest of the stats dict is unaffected.
    """
    init_reproj, _ = _split_residuals(init_err, param_handler)
    init_euclid = float(np.mean(np.linalg.norm(np.reshape(init_reproj, (-1, 2)), axis=1)))
    final_reproj, final_priors = _split_residuals(optimisation.fun, param_handler)
    final_euclid = float(np.mean(np.linalg.norm(np.reshape(final_reproj, (-1, 2)), axis=1)))
    stats = {
        "initial_euclid": init_euclid,
        "final_euclid": final_euclid,
        "param_count": int(len(init_params)),
        "observation_count": int(len(final_reproj) // 2),
        "prior_residual_count": int(final_priors.size) if final_priors is not None else 0,
        "elapsed_sec": float(elapsed_sec),
        "status": int(optimisation.status),
        "success": bool(optimisation.success),
        "message": str(optimisation.message),
        "nfev": int(optimisation.nfev),
    }

    if param_handler is not None:
        # These attributes are indexed by "global_im_num" (a pose/image index shared
        # across all cameras in the rig, see TargetDetection) rather than by a single
        # (camera, pose) pair, so there is no individual camera name to resolve them
        # to here -- the raw pose index is used as the identifier, per the fallback
        # noted in the calling convention.
        per_pose_error = getattr(param_handler, "initial_per_im_error", None)
        missing_before = getattr(param_handler, "missing_poses_before_outlier_rejection", None)
        missing_after = getattr(param_handler, "missing_poses_after_outlier_rejection", None)

        if per_pose_error is not None:
            per_pose_error_arr = np.asarray(per_pose_error, dtype=float)
            stats["per_pose_initial_error_px"] = [
                {"pose": int(i), "initial_error_px": float(v)}
                for i, v in enumerate(per_pose_error_arr)
            ]

        if missing_before is not None:
            stats["outlier_poses_before_rejection"] = [
                int(i) for i in np.where(np.asarray(missing_before))[0]
            ]

        if missing_after is not None:
            stats["outlier_poses_after_rejection"] = [
                int(i) for i in np.where(np.asarray(missing_after))[0]
            ]

    return stats


def run_bundle_adjustment_with_stats(
    param_handler: th.TemplateBundleHandler,
    threads: int = 1,
) -> tuple[OptimizeResult, CameraSet, dict]:
    """Run bundle adjustment and also return structured run statistics for GUI diagnostics."""
    loss_fn, bundle_jac, init_params = make_optimisation_function(param_handler, threads)
    init_err = loss_fn(init_params)

    bounds = param_handler.get_lockbox_bounds(len(init_params))
    start = time.time()
    optimisation = least_squares(
        loss_fn,
        init_params,
        verbose=param_handler.problem_opts['verbosity'],
        jac=bundle_jac if bundle_jac is not None else "2-point",
        max_nfev=param_handler.problem_opts["max_nfev"],
        x_scale='jac',
        xtol=param_handler.problem_opts.get("xtol", 1e-4),
        loss=param_handler.problem_opts.get("loss", "linear"),
        f_scale=param_handler.problem_opts.get("f_scale", 1.0),
        bounds=bounds,
    )
    elapsed = time.time() - start

    camset = param_handler.get_camset(optimisation.x)
    camset.set_calibration_history(optimisation, param_handler)
    stats = get_bundle_adjustment_stats(optimisation, init_params, init_err, elapsed, param_handler=param_handler)
    return optimisation, camset, stats
