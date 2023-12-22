from __future__ import annotations
import logging
import time
from copy import copy
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from typing import TYPE_CHECKING

import pyCamSet.utils.general_utils as gu
import pyCamSet.optimisation.compiled_helpers as ch

from pyCamSet.calibration_targets import TargetDetection
    
if TYPE_CHECKING:
    from pyCamSet.calibration_targets import AbstractTarget
    from pyCamSet.cameras import CameraSet, Camera


DEFAULT_OPTIONS = {
    'verbosity': 2,
    'fixed_pose':0,
    'ref_cam':0,
    'ref_pose':0,
    'outliers':'ask'
}


def list_dict_to_np_array(d) -> dict:
    if isinstance(d, dict):
        for key, val in d.items():
            if isinstance(val, dict):
                list_dict_to_np_array(val)
            elif isinstance(val, list):
                d[key] = np.array(val)
            else:
                pass
    return d


class TemplateBundlePrimitive:
    """
    A class that contains a set of base arrays.
    These arrays contain the pose, extrinsic, intrinsic and distortion params
    that will be used to create the bundle adjustment problem.
    If a param is fixed, it can be marked as fixed in the *_fixed data structure.
    A fixed value will not be dependent on the standard parameters.
    """

    def __init__(self, poses: np.ndarray, extr: np.ndarray, intr: np.ndarray, dst: np.ndarray,
                 poses_unfixed=None, extr_unfixed=None, intr_unfixed=None, dst_unfixed=None
                 ):
        self.poses = poses
        self.poses_unfixed = poses_unfixed if poses_unfixed is not None else np.ones(poses.shape[0], dtype=bool)
        self.extr = extr
        self.extr_unfixed = extr_unfixed if extr_unfixed is not None else np.ones(extr.shape[0], dtype=bool)
        self.intr = intr
        self.intr_unfixed = intr_unfixed if intr_unfixed is not None else np.ones(intr.shape[0], dtype=bool)
        self.dst = dst
        self.dst_unfixed = dst_unfixed if dst_unfixed is not None else np.ones(dst.shape[0], dtype=bool)

        self.calc_free_poses()

    def calc_free_poses(self):

        self.free_poses = np.sum(self.poses_unfixed)
        self.free_extr = np.sum(self.extr_unfixed)
        self.free_intr = np.sum(self.intr_unfixed)
        self.free_dst = np.sum(self.dst_unfixed)
        
        
        self.pose_end = 6 * self.free_poses
        self.extr_end = 6 * self.free_extr + self.pose_end
        self.intr_end = 4 * self.free_intr + self.extr_end
        self.dst_end = 5 * self.free_dst + self.intr_end

    def return_bundle_primitives(self, params):
        """
        Takes an array of parameters and populates all unfixed parameters.

        :param params: The input parameters
        """

        pose_data = params[:self.pose_end].reshape((-1, 6))
        extr_data = params[self.pose_end:self.extr_end].reshape((-1, 6))
        intr_data = params[self.extr_end:self.intr_end].reshape((-1, 4))
        dst_data = params[self.intr_end:self.dst_end].reshape((-1, 5))

        ch.fill_pose(pose_data, self.poses, self.poses_unfixed)
        ch.fill_extr(extr_data, self.extr, self.extr_unfixed)
        ch.fill_intr(intr_data, self.intr, self.intr_unfixed)
        ch.fill_dst(dst_data, self.dst, self.dst_unfixed)
        return self.poses, self.extr, self.intr, self.dst



def make_optimisation_function(
        param_handler: TemplateBundleHandler,
        threads: int = 16,
) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
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

    def bundle_fn(x):
        proj, intr, dists, obj_points = param_handler.get_bundle_adjustment_inputs(x)
        output = ch.bundle_adj_parrallel_solver(
            data,
            im_points=obj_points,
            projection_matrixes=proj,
            intrinsics=intr,
            dists=dists,
        )
        return output.reshape((-1))[:length * 2]

    return bundle_fn, init_params


def run_bundle_adjustment(param_handler: TemplateBundleHandler,
                          threads: int = 1) -> tuple[np.ndarray, CameraSet]:
    """
    A function that takes an abstract parameter handler, turns it into a cost function, and returns the
    optimisation results and the camera set that minimises the optimisation problem defined by the parameter handler.

    :param param_handler: The parameter handler that represents the optimisation
    :return: The output of the calibration and the argmin defined CameraSet
    """
    loss_fn, init_params = make_optimisation_function(
        param_handler, threads
    )

    init_err = loss_fn(init_params)
    init_euclid = np.mean(np.linalg.norm(np.reshape(init_err, (-1, 2)), axis=1))
    logging.info(f'found {len(init_params):.2e} parameters')
    logging.info(f'found {len(init_err):.2e} control points')
    logging.info(f'Initial Euclidean error: {init_euclid:.2f} px')

    if (init_euclid > 150) or (init_euclid == np.nan):
        logging.critical("Found worryingly high/NaN initial error: check that the initial parametisation is sensible")
        logging.info(
            "This can often indicate failure to place a camera or target correctly, giving nonsensical errors.")
        param_handler.check_params(init_params)

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

