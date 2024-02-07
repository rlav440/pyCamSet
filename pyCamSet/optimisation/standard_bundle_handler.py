
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
import pyCamSet.optimisation.function_block_implementations as fb

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

class TemplateBundlePrimitive:
    """
    A class that contains a set of base arrays.
    These arrays contain the pose, extrinsic, intrinsic and distortion params
    that will be used to create the bundle adjustment problem.
    If a param is fixed, it can be marked as fixed in the *_fixed data structure.
    A fixed value will not be dependent on the standard parameters.
    """

    def __init__(self, poses: np.ndarray, extr: np.ndarray, intr: np.ndarray,
                 poses_unfixed=None, extr_unfixed=None, intr_unfixed=None, 
                 ):
        self.poses = poses
        self.poses_unfixed = poses_unfixed if poses_unfixed is not None else np.ones(poses.shape[0], dtype=bool)
        self.extr = extr
        self.extr_unfixed = extr_unfixed if extr_unfixed is not None else np.ones(extr.shape[0], dtype=bool)
        self.intr = intr
        self.intr_unfixed = intr_unfixed if intr_unfixed is not None else np.ones(intr.shape[0], dtype=bool)
        self.calc_free_poses()

    def calc_free_poses(self):

        self.free_poses = np.sum(self.poses_unfixed)
        self.free_extr = np.sum(self.extr_unfixed)
        self.free_intr = np.sum(self.intr_unfixed)

        self.intr_end = 9 * self.free_intr
        self.extr_end = 6 * self.free_extr + self.intr_end
        self.pose_end = 6 * self.free_poses + self.extr_end


    def return_bundle_primitives(self, params):
        """
        Takes an array of parameters and populates all unfixed parameters.

        :param params: The input parameters
        """


        intr_data = params[:self.intr_end].reshape((self.free_intr, 9))
        extr_data = params[self.intr_end:self.extr_end].reshape((self.free_extr, 6))
        pose_data = params[self.extr_end:self.pose_end].reshape((self.free_poses, 6))

        ch.fill_flat(pose_data, self.poses, self.poses_unfixed)
        ch.fill_flat(extr_data, self.extr, self.extr_unfixed)
        ch.fill_flat(intr_data, self.intr, self.intr_unfixed)
        return self.intr, self.extr, self.poses

class SelfBundleHandler:
    """
    The standard bundle handler is a class that handles the optimisation of camera parameters.
    It is designed to be used with the numba implentation of the bundle adjustment cost function.
    It takes a CameraSet, a Target and the associated TargetDetection.
    Given these, it will return a function that takes a parameter array and returns data structures ready for
    evaluation with the bundle adjustment cost function.
    The implementation given in the standard param handler implements a target based, but feature unconstrained
    pose based bundle adjustment.

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
                 fixed_params: dict|None = None,
                 options: dict | None = None,
                 missing_poses: list | None =None
                 ):
        
        self.problem_opts = DEFAULT_OPTIONS
        if options is not None:
            self.problem_opts.update(options)

        self.fixed_params = list_dict_to_np_array(fixed_params)
        if fixed_params is None:
            self.fixed_params : dict = {}

        self.camset = camset
        self.cam_names = camset.get_names()
        self.detection = detection
        self.target = target
        self.point_data = target.point_data
        self.flat_point_data = target.point_data.reshape((-1,3))
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
        self.pose_unfixed = np.ones(n_poses, dtype=bool)
        if "fixed_pose" in self.problem_opts:
            fixed_pose = self.problem_opts["fixed_pose"]
            self.pose_unfixed[fixed_pose] = False
            self.poses[fixed_pose, :] = [0,0,0,0,0,0]
        self.feat_unfixed = np.ones(self.flat_point_data.shape[0], dtype=bool)

        self.populate_self_from_fixed_params()

        self.bundlePrimitive = TemplateBundlePrimitive(
            self.poses, self.extr, self.intr,
            extr_unfixed=self.extr_unfixed, intr_unfixed=self.intr_unfixed, poses_unfixed=self.pose_unfixed,
        )

        self.param_len = None
        self.jac_mask = None
        self.missing_poses: list | None = missing_poses

        self.op_fun: fb.optimisation_function = fb.projection() + fb.extrinsic3D() + fb.rigidTform3d() +  fb.free_point()

    def can_make_jac(self):
        return self.op_fun.can_make_jac() 

    def make_loss_fun(self, threads):
        
        #flatten the object shape
        obj_data = self.target.point_data.reshape((-1, 3))

        temp_loss = self.op_fun.make_full_loss_fn(self.detection.get_data(), threads, template=obj_data)
        def loss_fun(params):
            inps = self.get_bundle_adjustment_inputs(params) #return proj, extr, poses
            param_str = self.op_fun.build_param_list(*inps)
            return temp_loss(param_str).flatten()
        return loss_fun

    def make_loss_jac(self, threads): 
        #TODO implement proper culling
        obj_data = self.target.point_data.reshape((-1, 3))
        temp_loss = self.op_fun.make_jacobean(self.detection.get_data(), threads, template=obj_data)
        mask = np.concatenate(
            ( 
                np.repeat(self.intr_unfixed, 9),
                np.repeat(self.extr_unfixed, 6),
                np.repeat(self.pose_unfixed, 6),
                np.repeat(self.feat_unfixed, 3),
            ), axis=0
        )
        if np.all(mask):
            def jac_fn(params):
                inps = self.get_bundle_adjustment_inputs(params) #return proj, extr, poses
                param_str = self.op_fun.build_param_list(*inps)
                return temp_loss(param_str)
            return jac_fn
        else:
            def jac_fn(params):
                inps = self.get_bundle_adjustment_inputs(params) #return proj, extr, poses
                param_str = self.op_fun.build_param_list(*inps)
                return temp_loss(param_str)[:, mask]
            return jac_fn

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
        proj, extr, poses = self.bundlePrimitive.return_bundle_primitives(x)
        free_points = np.prod(self.target_point_shape)
        im_points = x[-free_points:] #unless we use the same system for defining the fixed target points
        #reshape and return the 
        return proj, extr, poses, im_points

    def find_and_exclude_transform_outliers(self, per_im_error): 
        """
        Given a number of per image errors, finds images that may cause calibration difficulties.
        Uses MAD outlier detection. If an image/pose is found to be an outlier, it is marked as a missing pose internally.

        :param per_im_error: The total reprojection error per image.
        """
        if self.missing_poses is None:
            raise ValueError("missing poses should be initialised before calling this function")
        cyclic_outlier_detection = True
        num_loops = 0
        logging.info("Begining outlier detection")
        while cyclic_outlier_detection and num_loops < 10:
            not_missing = np.where(~np.array(self.missing_poses))[0]
            # mloc = np.mean(poses[not_missing, :3, -1], axis=0)
            mloc = per_im_error
            condensed_outlier_inds = gu.mad_outlier_detection(
                # [np.linalg.norm(p[:3,3] - mloc) for p in poses[not_missing]],
                mloc[not_missing],
                out_thresh=20,
            )
            outlier_inds = not_missing[condensed_outlier_inds]
            
            if condensed_outlier_inds is not None:
                user_in = self.problem_opts['outliers']
                while not (user_in == 'y' or user_in == 'n'):
                    print(f"Outliers detected in iteration {num_loops}.")
                    user_in = input("Do you wish to remove these outlier poses: \n y/n: ")
                if user_in == 'y':
                    self.missing_poses[outlier_inds] = True
                if user_in == 'n':
                    cyclic_outlier_detection = False
            else:
                logging.info(f"No outliers detected in iteration {num_loops}.")
                cyclic_outlier_detection = False
            num_loops += 1

    def set_initial_params(self, cam_ints, cam_extrs, target_poses):
        """
        Sets the initial params from som other parameter array.

        :param x: An appropriate param vector for the given problem
        :return: the params
        """
        param_array = []
        for idc, intr_unfixed in enumerate(self.bundlePrimitive.intr_unfixed):
            if intr_unfixed:
                param_array.append(cam_ints[idc])

        for ide, ext_unfixed in enumerate(self.bundlePrimitive.extr_unfixed):
            if ext_unfixed:
                param_array.append(cam_extrs[ide])

        for idp, pose_unfixed in enumerate(self.bundlePrimitive.poses_unfixed):
            if pose_unfixed:
                param_array.append(target_poses[idp])

        param_array.append(self.point_data.reshape((-1,3))[self.feat_unfixed].flatten())
        param_array = np.concatenate(param_array, axis=0)
        return param_array

    def get_initial_params(self) -> np.ndarray:
        """
        Returns initial parameters if they exist, or starts calculating them
        and returns the result if they do not.

        :return: the params
        """
        if self.initial_params is None:
            raise ValueError("Initial params should be set with this method")
        return self.initial_params

    def get_camset(self, x, return_pose=False) -> CameraSet | tuple[CameraSet, np.ndarray]:
        """
        Given a set of parameters, returns a camera set.

        :param x: the optimisation parameters.
        :param return_pose: Optionally also return the poses of the target.
        :return: Either a CameraSet, or a CameraSet and a list of object poses.
        """


        new_cams = copy(self.camset)
        proj, extr, poses = self.bundlePrimitive.return_bundle_primitives(x)

        for idc, cam_name in enumerate(self.cam_names):
            blank_intr = np.eye(3)
            blank_intr[0, 0] = proj[idc][0]
            blank_intr[0, 2] = proj[idc][1]
            blank_intr[1, 1] = proj[idc][2]
            blank_intr[1, 2] = proj[idc][3]
            temp_cam: Camera = new_cams[cam_name]
            temp_cam.extrinsic = gu.make_4x4h_tform(extr[idc][:3], extr[idc][3:])
            temp_cam.intrinsic = blank_intr
            temp_cam.distortion_coefs = proj[idc][4:]
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

        detection = self.detection
        if self.missing_poses is not None:
            if np.any(self.missing_poses):
                logging.info("Missing poses required removing detected data from the optimisation")
                # delete any inds with missing pose numbers.
                missing_poses = np.where(self.missing_poses)[0]
                detection = self.detection.delete_row(im_num=missing_poses)
        if flatten:
            return detection.return_flattened_keys(dims).get_data()
        return detection.get_data()

    def check_params(self, params):
        """
        A helper function to visualise the problem as sent to the bundle adjustment cost.
        
        :param params: The input parameters to an optimisation.
        """
        obj_points = self.get_bundle_adjustment_inputs(params, make_points=True)
        self.get_camset(params).plot_np_array(obj_points.reshape((-1, 3)))

