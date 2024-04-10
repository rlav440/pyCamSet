
from __future__ import annotations
import logging
import time
from copy import copy
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from typing import TYPE_CHECKING
from pyCamSet.optimisation.template_handler import TemplateBundleHandler, DEFAULT_OPTIONS
import pyCamSet.utils.general_utils as gu
import pyCamSet.optimisation.compiled_helpers as ch
import pyCamSet.optimisation.function_block_implementations as fb

from pyCamSet.calibration_targets import TargetDetection
    
if TYPE_CHECKING:
    from pyCamSet.calibration_targets import AbstractTarget
    from pyCamSet.cameras import CameraSet, Camera


class StandardBundlePrimitive:
    """
    A class that contains a set of base arrays.
    These arrays contain the pose, extrinsic, intrinsic and distortion params
    that will be used to create the bundle adjustment problem.
    If a param is fixed, it can be marked as fixed in the *_fixed data structure.
    A fixed value will not be dependent on the standard parameters.
    """

    def __init__(self, poses:np.ndarray, bundle_points: np.ndarray, extr: np.ndarray, intr: np.ndarray,
                 poses_unfixed=None, bundle_points_unfixed=None, extr_unfixed=None, intr_unfixed=None, 
                 ):
        self.extr = extr
        self.extr_unfixed = extr_unfixed if extr_unfixed is not None else np.ones(extr.shape[0], dtype=bool)
        self.intr = intr
        self.intr_unfixed = intr_unfixed if intr_unfixed is not None else np.ones(intr.shape[0], dtype=bool)
        self.bundle_pts = bundle_points
        self.bdpt_unfixed = bundle_points_unfixed if bundle_points_unfixed is not None else np.ones(bundle_points.shape[0], dtype=bool)
        self.poses = poses
        self.pose_unfixed = poses_unfixed if poses_unfixed is not None else np.ones(poses.shape[0], dtype=bool)
        self.calc_type_inds()

    def calc_type_inds(self):

        self.free_extr = np.sum(self.extr_unfixed)
        self.free_intr = np.sum(self.intr_unfixed)
        self.free_pose = np.sum(self.pose_unfixed)
        self.free_bdpt = np.sum(self.bdpt_unfixed)

        self.intr_end = 9 * self.free_intr
        self.extr_end = 6 * self.free_extr + self.intr_end
        self.pose_end = 6 * self.free_pose + self.extr_end
        self.bdpt_end = 3 * self.free_bdpt + self.pose_end

    def return_bundle_primitives(self, params):
        """
        Takes an array of parameters and populates all unfixed parameters.

        :param params: The input parameters
        """


        intr_data = params[:self.intr_end].reshape((self.free_intr, 9))
        extr_data = params[self.intr_end:self.extr_end].reshape((self.free_extr, 6))
        pose_data = params[self.extr_end:self.pose_end].reshape((self.free_pose, 6))
        bdpt_data = params[self.pose_end:self.bdpt_end].reshape((self.free_bdpt, 3))

        ch.fill_flat(pose_data, self.poses, self.pose_unfixed)
        ch.fill_flat(bdpt_data, self.bundle_pts, self.bdpt_unfixed)
        ch.fill_flat(extr_data, self.extr, self.extr_unfixed)
        ch.fill_flat(intr_data, self.intr, self.intr_unfixed)
        return self.intr, self.extr, self.poses, self.bundle_pts

class SelfBundleHandler(TemplateBundleHandler):
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
        super().__init__(camset, target, detection, fixed_params, options, missing_poses) 

        self.flat_point_data = np.copy(self.point_data.reshape((-1, 3)))
        self.feat_unfixed = np.ones(self.flat_point_data.shape[0], dtype=bool)
        self.bundlePrimitive = StandardBundlePrimitive(
            self.poses, self.flat_point_data, self.extr, self.intr,
            extr_unfixed=self.extr_unfixed, intr_unfixed=self.intr_unfixed, poses_unfixed=self.pose_unfixed, bundle_points_unfixed=self.feat_unfixed
        )

        self.param_len = None
        self.jac_mask = None
        self.missing_poses: list | None = missing_poses

        self.op_fun: fb.optimisation_function = fb.projection() + fb.extrinsic3D() + fb.rigidTform3d() +  fb.free_point()

    def make_loss_fun(self, threads):
        #flatten the object shape
        temp_loss = self.op_fun.make_full_loss_fn(self.detection.get_data(), threads)
        def loss_fun(params):
            inps = self.get_bundle_adjustment_inputs(params) #return proj, extr, poses
            param_str = self.op_fun.build_param_list(*inps)
            return temp_loss(param_str).flatten()
        return loss_fun

    def make_loss_jac(self, threads): 
        #TODO implement proper culling
        temp_loss = self.op_fun.make_jacobean(self.detection.get_data(), threads)
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
                return temp_loss(param_str, )
            return jac_fn
        else:
            def jac_fn(params):
                inps = self.get_bundle_adjustment_inputs(params) #return proj, extr, poses
                param_str = self.op_fun.build_param_list(*inps)
                return temp_loss(param_str)[:, mask]
            return jac_fn

    def get_bundle_adjustment_inputs(self, x) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This function uses the state of the parameter handler, and the given params
        to build np arrays that describe:

            - the projection matrices of the cameras
            - the distortion parameters of the cameras
            - the pose of every object point in every time poin

        These are direct inputs to a bundle adjustment based loss.

        :param x: The input optimisation parameters.
        """
        proj, extr, poses, bundle_points = self.bundlePrimitive.return_bundle_primitives(x)
        return proj, extr, poses, bundle_points

    def set_initial_params(self,x):
        #get the number of params to set on the super
        self.initial_params = x

    def set_from_templated_camset(self, prev_cams: CameraSet):
        self.initial_params = np.empty(self.bundlePrimitive.bdpt_end)

        if not isinstance(prev_cams.calibration_handler, TemplateBundleHandler):
            raise ValueError("Previous camera set was not a templated adjustment")
        self.initial_params[:self.bundlePrimitive.pose_end] = prev_cams.calibration_params.copy()
        self.initial_params[ 
            self.bundlePrimitive.pose_end:
        ] = prev_cams.calibration_handler.target.point_data.copy().flatten() 
        # print(prev_cams.calibration_handler.target.point_data.flatten()[:20])


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
        proj, extr, poses, _ = self.bundlePrimitive.return_bundle_primitives(x)

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

        ps = np.empty((len(poses), 12))
        for pn, p in zip(ps, poses):
            ch.n_e4x4_flat_INPLACE(p, pn)

        return new_cams, ps


