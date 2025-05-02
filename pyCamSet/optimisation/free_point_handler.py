from __future__ import annotations
import logging
import time
from copy import copy
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_array
import pyvista as pv
from itertools import combinations

from typing import TYPE_CHECKING

from scipy.spatial.distance import cdist
from pyCamSet.optimisation.template_handler import TemplateBundleHandler, DEFAULT_OPTIONS
from pyCamSet.calibration_targets import AbstractTarget
import pyCamSet.utils.general_utils as gu
import pyCamSet.optimisation.compiled_helpers as ch
import pyCamSet.optimisation.function_block_implementations as fb

from pyCamSet.calibration_targets import TargetDetection
    
if TYPE_CHECKING:
    from pyCamSet.calibration_targets import AbstractTarget
    from pyCamSet.cameras import CameraSet, Camera


class FreePointTarget(AbstractTarget):
    def __init__(self, point_data):
        super().__init__(inputs=locals())
        self.point_data = point_data
        self._process_data()

    def find_in_image(self, image, draw=False, camera: Camera=None, wait_len = 1) -> ImageDetection:
        """
        Notes: Detects the calibration target in an image

        :param image: a mxn or mxnx3 image input
        :param draw: whether to draw the target
        :param camera: A camera object for use in camera aware detections
        :return: An ImageDetection object, containing the detected data
        """
        raise NotImplementedError

class FreePointPrimitive:
    """
    A class that contains a set of base arrays.
    These arrays contain the pose, extrinsic, intrinsic and distortion params
    that will be used to create the bundle adjustment problem.
    If a param is fixed, it can be marked as fixed in the *_fixed data structure.
    A fixed value will not be dependent on the standard parameters.
    """

    def __init__(self, bundle_points: np.ndarray, extr: np.ndarray, intr: np.ndarray,
                 bundle_points_unfixed=None, extr_unfixed=None, intr_unfixed=None, 
                 ):

        self.extr = extr
        self.extr_unfixed = extr_unfixed if extr_unfixed is not None else np.ones(extr.shape[0], dtype=bool)
        self.intr = intr
        self.intr_unfixed = intr_unfixed if intr_unfixed is not None else np.ones(intr.shape[0], dtype=bool)
        self.bundle_pts = bundle_points
        self.bdpt_unfixed = bundle_points_unfixed if bundle_points_unfixed is not None else np.ones(bundle_points.shape[0], dtype=bool)
        #we fix the bundle points on a per point basis

        self.correct_gauge = True
        self.calc_type_inds()

    def calc_type_inds(self):
        """
        Updates the internal indicies where different params are stored internally.
        """

        self.free_extr = np.sum(self.extr_unfixed)
        self.free_intr = np.sum(self.intr_unfixed)
        self.free_bdpt = np.sum(self.bdpt_unfixed)

        self.intr_end = 9 * self.free_intr
        self.extr_end = 6 * self.free_extr + self.intr_end
        self.bdpt_end = 1 * self.free_bdpt + self.extr_end

    def return_bundle_primitives(self, params):
        """
        Takes an array of parameters and populates all unfixed parameters.
        :param params: The input parameters
        :return: The intrinsics, extrinsics, poses and feature points of the calibration.
        """


        intr_data = params[:self.intr_end].reshape((self.free_intr, 9))
        extr_data = params[self.intr_end:self.extr_end].reshape((self.free_extr, 6))
        bdpt_data = params[self.extr_end:self.bdpt_end]

        ch.fill_flat(extr_data, self.extr, self.extr_unfixed)
        ch.fill_flat(intr_data, self.intr, self.intr_unfixed)
        ch.fill_flat(bdpt_data, self.bundle_pts, self.bdpt_unfixed)

        return self.intr, self.extr, self.bundle_pts.reshape((-1, 3))

class FreePointBundleHandler(TemplateBundleHandler):
    """
    The free point bundle handler is a class that handles the optimisation of camera parameters.
    It is designed to be used with the numba implentation of the bundle adjustment cost function.
    It takes a CameraSet, a Target and the associated TargetDetection.
    Given these, it will return a function that takes a parameter array and returns data structures ready for
    evaluation with the bundle adjustment cost function.
    The implementation given in the free point optimisation purely optimises the positions of points in the calibration space.

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

        self.flat_point_data = np.copy(self.point_data.reshape((-1)))
        self.feat_unfixed = np.ones(self.flat_point_data.shape[0], dtype=bool)
        self.super_primitive = self.bundlePrimitive

        self.bundlePrimitive = FreePointPrimitive(
            self.flat_point_data, self.super_primitive.extr, self.super_primitive.intr,
            extr_unfixed=self.super_primitive.extr_unfixed, intr_unfixed=self.super_primitive.intr_unfixed, bundle_points_unfixed=self.feat_unfixed
        )

        self.param_len = None
        self.jac_mask = None
        self.missing_poses: list | None = missing_poses


        self.op_fun: fb.optimisation_function = fb.projection() + fb.extrinsic3D() +  fb.free_point()

    def make_loss_fun(self, threads):
        """
        Describes and writes the loss function of the loss function represented by self.
        Wraps the loss function to account for the fixed parameters of the optimisation.

        :params threads: the number of threads to use.

        """
        target_shape = self.target.point_data.shape
        dd = self.detection.return_flattened_keys(target_shape[:-1]).get_data()
        temp_loss = self.op_fun.make_full_loss_fn(dd, threads)
        def loss_fun(params):
            inps = self.get_bundle_adjustment_inputs(params) #return proj, extr, poses
            param_str = self.op_fun.build_param_list(*inps)
            return temp_loss(param_str).flatten()
        return loss_fun

    def make_loss_jac(self, threads): 
        """
        Describes and writes the jacobian of the loss function described in self.
        Wraps the jacobian of the loss function to deal with the fixed parameters of the optimisation.

        :params threads: the number of threads to use for the optimisation.
        :returns jac_fn: a callable jacobian function that returns the jacobian of the given paramaters.
        """
        target_shape = self.target.point_data.shape
        dd = self.detection.return_flattened_keys(target_shape[:-1]).get_data()
        mask = np.concatenate(
            ( 
                np.repeat(self.bundlePrimitive.intr_unfixed, 9),
                np.repeat(self.bundlePrimitive.extr_unfixed, 6),
                np.repeat(self.bundlePrimitive.bdpt_unfixed, 1),
            ), axis=0
        )

        temp_loss = self.op_fun.make_jacobean(dd, threads, unfixed_params=mask)
        def jac_fn(params):
            inps = self.get_bundle_adjustment_inputs(params) #return proj, extr, poses
            param_str = self.op_fun.build_param_list(*inps)
            d, c, rp = temp_loss(param_str)
            return csr_array((d,c,rp), shape=(2*dd.shape[0], params.shape[0]))
        return jac_fn

    def get_bundle_adjustment_inputs(self, x) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This function uses the state of the parameter handler, and the given params
        to build np arrays that describe:

            - the projection matrices of the cameras
            - the distortion parameters of the cameras
            - the pose of every object point in every time poin

        These are direct inputs to a bundle adjustment based loss.

        :param x: The input optimisation parameters.
        """
        proj, extr, bundle_points = self.bundlePrimitive.return_bundle_primitives(x)
        return proj, extr, bundle_points

    def set_initial_params(self,x):
        #get the number of params to set on the super
        self.initial_params = x


    def set_from_camset(self, prev_cams: CameraSet, init_points: np.ndarray):
        """
        Sets the initial values of the calibration from a previous calibration of the same system.
        The previous system must have used a TemplateBundleHandler.
        :param prev_cams: The calibrated camseet to use.
        """
        self.initial_params = np.empty(self.bundlePrimitive.bdpt_end)
        self.initial_params[:self.bundlePrimitive.bdpt_end] = prev_cams.calibration_params.copy()
        self.initial_params[ 
            self.bundlePrimitive.bdpt_end:
        ] = init_points.flatten()
        # print(prev_camsk

    def get_initial_params(self) -> np.ndarray:
        """
        Returns initial parameters if they exist, or starts calculating them
        and returns the result if they do not.

        :return: the params
        """

        if self.initial_params is not None:
            return self.initial_params
        start_params = self.calc_initial_params()

        self.initial_params = np.empty(self.bundlePrimitive.bdpt_end)
        self.initial_params[:self.bundlePrimitive.pose_end] = start_params
        self.initial_params[ 
            self.bundlePrimitive.pose_end:
        ] = self.target.point_data.copy().flatten()[self.feat_unfixed]
        return self.initial_params

    def get_updated_points():
        _,_, ps = self.bundlePrimitive.return_bundle_primitives(x)
        return ps

    def get_camset(self, x) -> CameraSet:
        """
        Given a set of parameters, returns a camera set.

        :param x: the optimisation parameters.
        :param return_pose: Optionally also return the poses of the target.
        :return: Either a CameraSet, or a CameraSet and a list of object poses.
        """


        new_cams = copy(self.camset)

        standard_model = self.bundlePrimitive.return_bundle_primitives(x)
        proj, extr, ps = standard_model

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
        return new_cams
        





