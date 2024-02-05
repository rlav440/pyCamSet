
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

        self.pose_unfixed = np.array(n_poses, dtype=bool)
        if fixed_pose := self.problem_opts.get("fixed_pose"):
            self.pose_unfixed[fixed_pose] = False
            self.poses[fixed_pose, :] = [1,0,0, 0,1,0, 0,0,1, 0,0,0]

        self.populate_self_from_fixed_params()
        self.bundlePrimitive = BundlePrimitive(
            self.poses, self.extr, self.intr, self.dst,
            extr_unfixed=self.extr_unfixed, intr_unfixed=self.intr_unfixed, dst_unfixed=self.dst_unfixed
        )

        self.param_len = None
        self.jac_mask = None
        self.missing_poses: list | None = missing_poses

        self.op_fun: fb.optimisation_function = fb.projection() + fb.extrinsic3D() + fb.rigidTform3d() +  fb.free_point()

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
            ch.n_htform_broadcast_prealloc(self.point_data, pose, im_points[idx])
        
        im_points = np.reshape(im_points, (len(poses), -1, 3))
        return proj, intr, dst, im_points

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

        cam_poses, target_poses, per_im_error = estimate_camera_relative_poses(
            detection=self.detection, cams=self.camset, calibration_target=self.target
        )
        self.missing_poses = np.array([np.isnan(t[0,0]) for t in target_poses])
        self.find_and_exclude_transform_outliers(per_im_error)
        
        for idp, pose_unfixed in enumerate(self.bundlePrimitive.poses_unfixed):
            # how do we handle missing poses - delete the image associated with the missing daata
            if pose_unfixed:
                ext = gu.ext_4x4_to_rod(target_poses[idp])
                param_array.append(ext[0])
                param_array.append(ext[1])
        for idc, ext_unfixed in enumerate(self.bundlePrimitive.extr_unfixed):
            if ext_unfixed:
                ext = gu.ext_4x4_to_rod(cam_poses[idc])
                param_array.append(ext[0])
                param_array.append(ext[1])
        for idc, intr_unfixed in enumerate(self.bundlePrimitive.intr_unfixed):
            if intr_unfixed:
                param_array.append(cams[idc].intrinsic[[0, 0, 1, 1], [0, 2, 1, 2]])
        for idc, intr_unfixed in enumerate(self.bundlePrimitive.dst_unfixed):
            if intr_unfixed:
                param_array.append(np.squeeze(cams[idc].distortion_coefs))
        param_array = np.concatenate(param_array, axis=0)
        param_array = self.add_extra_params(param_array)
        return param_array

    def get_camset(self, x, return_pose=False) -> CameraSet | tuple[CameraSet, np.ndarray]:
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
            temp_cam: Camera = new_cams[cam_name]
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
        _, _, _, obj_points = self.get_bundle_adjustment_inputs(params)
        self.get_camset(params).plot_np_array(obj_points.reshape((-1, 3)))
