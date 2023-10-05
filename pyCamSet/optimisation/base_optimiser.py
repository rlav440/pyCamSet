import logging
import time
from copy import copy
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from pyCamSet.calibration_targets import TargetDetection, AbstractTarget
from pyCamSet.cameras import CameraSet
from pyCamSet.utils.general_utils import ext_4x4_to_rod
from pyCamSet.optimisation.compiled_helpers import fill_pose, fill_dst, fill_extr, fill_intr
from pyCamSet.optimisation.compiled_helpers import bundle_adj_parrallel_solver, n_htform_broadcast_prealloc

DEFAULT_OPTIONS = {
    'verbosity': 2,
}


def list_dict_to_np_array(d):
    if isinstance(d, dict):
        for key, val in d.items():
            if isinstance(val, dict):
                list_dict_to_np_array(val)
            elif isinstance(val, list):
                d[key] = np.array(val)
            else:
                pass
    return d


class BundlePrimitive:
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

        self.free_poses = np.sum(self.poses_unfixed)
        self.free_extr = np.sum(self.extr_unfixed)
        self.free_intr = np.sum(self.intr_unfixed)
        self.free_dst = np.sum(self.dst_unfixed)

    def return_bundle_primitives(self, params):
        """
        Takes an array of parameters and populates all unfixed parameters.

        :param params: The input parameters
        """
        pose_end = 6 * self.free_poses
        extr_end = 6 * self.free_extr + pose_end
        intr_end = 4 * self.free_intr + extr_end
        dst_end = 5 * self.free_dst + intr_end

        pose_data = params[:pose_end].reshape((-1, 6))
        extr_data = params[pose_end:extr_end].reshape((-1, 6))
        intr_data = params[extr_end:intr_end].reshape((-1, 4))
        dst_data = params[intr_end:dst_end].reshape((-1, 5))

        fill_pose(pose_data, self.poses, self.poses_unfixed)
        fill_extr(extr_data, self.extr, self.extr_unfixed)
        fill_intr(intr_data, self.intr, self.intr_unfixed)
        fill_dst(dst_data, self.dst, self.dst_unfixed)
        return self.poses, self.extr, self.intr, self.dst


class AbstractParamHandler:
    """
    The abstract param handler is a class that handles the optimisation of camera parameters.
    It is designed to be used with the numba implentation of the bundle adjustment cost function.
    It takes a CameraSet, a Target and the associated TargetDetection.
    Given these, it will return a function that takes a parameter array and returns data structures ready for
    evaluation with the bundle adjustment cost function.
    The implementation given in the abstract param handler implements a standard object pose based bundle adjustment.

    Two functions provide the ability to add extra parameters and functionality to the optimisation.
        - add_extra_params: this can be overriden to add initial estimates of additional parameters.
        - parse_extra_params_and_setup: this can be overriden to parse additional parameters given to the optimisation.
            Manipulations of the object data/state can be done here, and will be reflected in the cost function.
            As an example: if a higher level structure for camera poses is defined, self.extr_unfixed can be set to all
             false. The parameters can then be parsed, translated into specific extrinsics for each camera, written
             to self.extr, and the cost function will use these extrinsics to define the camera.
    """

    def __init__(self,
                 camset: CameraSet, target: AbstractTarget, detection: TargetDetection,
                 fixed_params: dict = None,
                 options: dict = None,
                 missing_poses=None
                 ):

        self.fixed_params = list_dict_to_np_array(fixed_params)
        if fixed_params is None:
            self.fixed_params = {}
        self.camset = camset
        self.cam_names = camset.get_names()
        self.detection = detection
        self.target = target
        self.point_data = target.point_data
        self.target_point_shape = np.array(target.point_data.shape)
        self.initial_params = None

        n_poses = detection.max_ims
        n_cams = camset.get_n_cams()

        self.poses = np.zeros((n_poses, 12))
        self.extr = np.zeros((n_cams, 4, 4))
        self.intr = np.zeros((n_cams, 3, 3))
        self.dst = np.zeros((n_cams, 5))

        self.extr_unfixed = np.array(['ext' not in self.fixed_params.get(cam_name, {}) for cam_name in self.cam_names])
        self.intr_unfixed = np.array(['int' not in self.fixed_params.get(cam_name, {}) for cam_name in self.cam_names])
        self.dst_unfixed = np.array(['dst' not in self.fixed_params.get(cam_name, {}) for cam_name in self.cam_names])
        self.populate_self_from_fixed_params()

        self.bundlePrimitive = BundlePrimitive(
            self.poses, self.extr, self.intr, self.dst,
            extr_unfixed=self.extr_unfixed, intr_unfixed=self.intr_unfixed, dst_unfixed=self.dst_unfixed
        )

        self.param_len = None
        self.jac_mask = None
        self.problem_opts = DEFAULT_OPTIONS
        if options is not None:
            self.problem_opts.update(options)
        self.missing_poses = missing_poses

    def add_extra_params(self, param_array:np.ndarray) -> np.ndarray:
        """
        A function called during the initial parameterisation to allow for the addition of extra parameters.

        :param param_array:
        :return:
        """
        return param_array

    def parse_extra_params_and_setup(self, param_array:np.ndarray) -> np.ndarray:
        """
        A function called at the start of getting the bundle adjustment inputs
        to allow the addition of different parameter structures to the optimisation.
        It also allows for the implementation of additional non-standard calculations.
        
        :param param_array:
        :return:
        """
        return param_array

    def special_plots(self, params):
        """
        A function called during a visualisaiton of the calibration results for any
        target specific plots to show.

        :param params: The parameters of the calbiration.
        """
        return

    def populate_self_from_fixed_params(self):
        """
        Populates the internal data structures from the fixed params.
        """
        for idx, cam_name in enumerate(self.cam_names):
            if 'ext' in self.fixed_params.get(cam_name, {}):
                self.extr[idx] = self.fixed_params[cam_name]['ext']
            if 'int' in self.fixed_params.get(cam_name, {}):
                self.intr[idx] = self.fixed_params[cam_name]['int']
            if 'dst' in self.fixed_params.get(cam_name, {}):
                self.dst[idx] = self.fixed_params[cam_name]['dst']

    def get_bundle_adjustment_inputs(self, x) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This function uses the state of the parameter handler, and the given params
        to build np arrays that describe:
            - the projection matrices of the cameras
            - the intrinsic matrices of the cameras
            - the distortion parameters of the cameras
            - the pose of every object point in every time point
        These are direct inputs to a bundle adjustment based loss.

        :param x: The input optimisation parameters.
        """
        x = self.parse_extra_params_and_setup(x)
        poses, extr, intr, dst = self.bundlePrimitive.return_bundle_primitives(x)
        proj = intr @ extr[:, :3, :]
        im_points = np.empty((len(poses), *self.point_data.shape))
        for idx, pose in enumerate(poses):
            n_htform_broadcast_prealloc(self.point_data, pose, im_points[idx])
        
        im_points = np.reshape(im_points, (len(poses), -1, 3))
        return proj, intr, dst, im_points


    def set_initial_params(self, x: np.ndarray):
        """
        Sets the initial params from som other parameter array.

        :param x: An appropriate param vector for the given problem
        :return: the params
        """
        self.initial_params = x

    def get_initial_params(self) -> np.ndarray:
        """
        Returns initial parameters if they exist, or starts calculating them
        and returns the result if they do not.

        :return: the params
        """

        if self.initial_params is not None:
            return self.initial_params
        return self.calc_initial_params()

    def calc_initial_params(self) -> np.ndarray:
        """
        Calculates an initial guess at the parameters for the following optimisation.
        The initial estimate gives parameters based on the provided CameraSet and a rough pose
        estimation of the target given those camera parameters.

        :return params: the initial estimate of the parameters.
        """
        cams = self.camset
        self.jac_mask = []
        param_array = []
        cam_idx = 0
        poses, detected_poses = self.target.pose_in_detections(self.detection, self.camset)
        for idp, pose_unfixed in enumerate(self.bundlePrimitive.poses_unfixed):
            # how do we handle missing poses - delete the image associated with the missing daata
            if pose_unfixed:
                ext = ext_4x4_to_rod(poses[idp])
                param_array.append(ext[0])
                param_array.append(ext[1])
        for idc, ext_unfixed in enumerate(self.bundlePrimitive.extr_unfixed):
            if ext_unfixed:
                ext = ext_4x4_to_rod(cams[idc].extrinsic)
                param_array.append(ext[0])
                param_array.append(ext[1])
        for idc, intr_unfixed in enumerate(self.bundlePrimitive.intr_unfixed):
            if intr_unfixed:
                param_array.append(cams[idc].intrinsic[[0, 0, 1, 1], [0, 2, 1, 2]])
        for idc, intr_unfixed in enumerate(self.bundlePrimitive.dst_unfixed):
            if intr_unfixed:
                param_array.append(np.squeeze(cams[idc].distortion_coefs))
        param_array = self.add_extra_params(param_array)
        return np.concatenate(param_array, axis=0)

    def get_camset(self, x, return_pose=False) -> CameraSet or tuple[CameraSet, np.ndarray]:
        """
        Given a set of parameters, returns a camera set.

        :param x: the optimisation parameters.
        :param return_pose: Optionally also return the poses of the target.
        :return: Either a CameraSet, or a CameraSet and a list of object poses.
        """

        x = self.parse_extra_params_and_setup(x)

        new_cams = copy(self.camset)
        poses, extr, intr, dst = self.bundlePrimitive.return_bundle_primitives(x)

        for idc, cam_name in enumerate(self.cam_names):
            temp_cam = new_cams[cam_name]
            temp_cam.extrinsic = extr[idc]
            temp_cam.intrinsic = intr[idc]
            temp_cam.distortion_coefs = dst[idc]
            temp_cam._update_state()
        if not return_pose:
            return new_cams

        return new_cams, poses

    def get_detection(self) -> TargetDetection:
        """
        :return targetDetection: a representation of the detection according to the param handler
        """
        return TargetDetection(cam_names=self.cam_names, data=self.get_detection_data())

    def get_detection_data(self, flatten=False) -> np.ndarray:
        """
        :return: The data as used by the calibration target.
        """
        # a function for representing any internal manipulations of the
        # detection data for external use

        #calculate the shape of the ojbect
        dims = self.target_point_shape[:-1]


        if self.missing_poses is not None:
            if np.any(self.missing_poses):
                logging.info("Missing poses required removing detected data from the optimisation")
                # delete any inds with missing pose numbers.
                missing_poses = np.where(self.missing_poses)[0]
                new_data = self.detection.delete_row(im_num=missing_poses)
            else:
                new_data = self.detection
            if flatten: 
                return new_data.return_flattened_keys(dims).get_data()
            return new_data.get_data()
        if flatten:
            return self.detection.return_flattened_keys(dims).get_data()
        return self.detection.get_data()

    def check_params(self, params):
        """
        A helper function to visualise the problem as sent to the bundle adjustment cost.
        
        :param params: The input parameters to an optimisation.
        """
        _, _, _, obj_points = self.get_bundle_adjustment_inputs(params)
        self.get_camset(params).plot_np_array(obj_points.reshape((-1, 3)))


def make_optimisation_function(
        param_handler: AbstractParamHandler,
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
        output = bundle_adj_parrallel_solver(
            data,
            im_points=obj_points,
            projection_matrixes=proj,
            intrinsics=intr,
            dists=dists,
        )
        return output.reshape((-1))[:length * 2]

    return bundle_fn, init_params


def run_bundle_adjustment(param_handler: AbstractParamHandler) -> tuple[np.ndarray, CameraSet]:
    """
    A function that takes an abstract parameter handler, turns it into a cost function, and returns the
    optimisation results and the camera set that minimises the optimisation problem defined by the parameter handler.

    :param param_handler: The parameter handler that represents the optimisation
    :return: The output of the calibration and the argmin defined CameraSet
    """
    loss_fn, init_params = make_optimisation_function(
        param_handler
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

    time.sleep(0.25)
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
