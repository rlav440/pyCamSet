from __future__ import annotations
import logging
import time
from copy import copy
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, approx_fprime

from typing import TYPE_CHECKING

import pyCamSet.utils.general_utils as gu
import pyCamSet.optimisation.compiled_helpers as ch
import pyCamSet.optimisation.template_handler as th

from pyCamSet.calibration_targets import TargetDetection
    
if TYPE_CHECKING:
    from pyCamSet.calibration_targets import AbstractTarget
    from pyCamSet.cameras import CameraSet, Camera


def make_optimisation_function(
        param_handler: th.TemplateBundleHandler,
        threads: int = 16,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]|None, np.ndarray]:
    """
    Takes a parameter handler and creates a callable cost function that evaluates the
    cost of a parameter array.

    :param param_handler: the param handler representing the optimisation
    :param threads: number of evaluation threads
    :return fn: the cost function
    """
    #
    init_params = param_handler.get_initial_params()
    base_data = param_handler.get_detection_data(flatten=True)
    length, width = base_data.shape
    data = np.resize(copy(base_data), (threads, int(np.ceil(length / threads)), width))
    
    bundle_loss_fun = param_handler.make_loss_fun(threads)


    if param_handler.can_make_jac():
        bundle_loss_jac = param_handler.make_loss_jac(threads)
    else: 
        bundle_loss_jac = None

    return bundle_loss_fun, bundle_loss_jac, init_params


def run_bundle_adjustment(param_handler: TemplateBundleHandler,
                          threads: int = 1) -> tuple[np.ndarray, CameraSet]:
    """
    A function that takes an abstract parameter handler, turns it into a cost function, and returns the
    optimisation results and the camera set that minimises the optimisation problem defined by the parameter handler.

    :param param_handler: The parameter handler that represents the optimisation
    :return: The output of the calibration and the argmin defined CameraSet
    """
    loss_fn, bundle_jac, init_params = make_optimisation_function(
        param_handler, threads
    )

    init_err = loss_fn(init_params)
    init_euclid = np.mean(np.linalg.norm(np.reshape(init_err, (-1, 2)), axis=1))
    logging.info(f'found {len(init_params):.2e} parameters')
    logging.info(f'found {len(init_err):.2e} control points')
    logging.info(f'Initial Euclidean error: {init_euclid:.2f} px')

    # test = lambda : loss_fn(init_params)
    # gu.benchmark(test, repeats=100)

    # test = lambda : bundle_jac(init_params)
    # gu.benchmark(test, repeats=100)

    if (init_euclid > 150) or (init_euclid == np.nan):
        logging.critical("Found worryingly high/NaN initial error: check that the initial parametisation is sensible")
        logging.info(
            "This can often indicate failure to place a camera or target correctly, giving nonsensical errors.")
        param_handler.check_params(init_params)

    # bundle_jac = lambda x: approx_fprime(x, loss_fn)
    start = time.time()
    optimisation = least_squares(
        loss_fn,
        init_params,
        # ftol=1e-4,
        verbose=param_handler.problem_opts['verbosity'],
        # method='lm', #dogbox', #trf',
        # tr_solver='lsmr',
        # jac_sparsity=sparsity,
        # loss='soft_l1',
        jac= bundle_jac if bundle_jac is not None else "2-point", #pass the function for the jacobian if it exists
        max_nfev=100,
        x_scale='jac',
    )
    end = time.time()

    final_euclid = np.mean(np.linalg.norm(np.reshape(optimisation.fun, (-1, 2)), axis=1))
    logging.info(f'Final Euclidean error: {final_euclid:.2f} px')
    logging.info(f'Optimisation took {end - start: .2f} seconds.')

    if final_euclid > 5:
        logging.critical("Remaining error is very large: please check the output results")
        param_handler.check_params(optimisation.x)

    camset = param_handler.get_camset(optimisation.x)
    camset.set_calibration_history(optimisation, param_handler)

    init_err = loss_fn(optimisation.x)
    init_euclid = np.mean(np.linalg.norm(np.reshape(init_err, (-1, 2)), axis=1))
    logging.info(f"Check test with a result of {init_euclid:.2f}")

    return optimisation, camset

