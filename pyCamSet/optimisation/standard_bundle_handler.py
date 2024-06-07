
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
import pyCamSet.utils.general_utils as gu
import pyCamSet.optimisation.compiled_helpers as ch
import pyCamSet.optimisation.function_block_implementations as fb

from pyCamSet.calibration_targets import TargetDetection
    
if TYPE_CHECKING:
    from pyCamSet.calibration_targets import AbstractTarget
    from pyCamSet.cameras import CameraSet, Camera


def find_not_colinear_pts(points):
    """
    Given a set of points mxn, finds 3 points that are not co-linear.
    :param points: The points to search for.
    :return: The indicies of the returned points.
    """
    ind0 = 0
    for ind1, ind2 in combinations(np.arange(1, points.shape[0]), 2):
        AB = points[ind0] - points[ind1]
        AC = points[ind0] - points[ind2]
        score = np.linalg.norm(np.cross(AB, AC))
        if score > 1e-8:
            return ind0, ind1, ind2
    else:
        raise ValueError("No set of values that were not colinear were found in the provided data.")

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
                 always_correct_gauge=False
                 ):

        self.extr = extr
        self.extr_unfixed = extr_unfixed if extr_unfixed is not None else np.ones(extr.shape[0], dtype=bool)
        self.intr = intr
        self.intr_unfixed = intr_unfixed if intr_unfixed is not None else np.ones(intr.shape[0], dtype=bool)
        self.bundle_pts = bundle_points
        self.bdpt_unfixed = bundle_points_unfixed if bundle_points_unfixed is not None else np.ones(bundle_points.shape[0], dtype=bool)
        #we fix the bundle points on a per point basis

        self.correct_gauge = True
        self.poses = poses
        self.pose_unfixed = poses_unfixed if poses_unfixed is not None else np.ones(poses.shape[0], dtype=bool)
        self.calc_type_inds()

    def calc_type_inds(self):
        """
        Updates the internal indicies where different params are stored internally.
        """

        self.free_extr = np.sum(self.extr_unfixed)
        self.free_intr = np.sum(self.intr_unfixed)
        self.free_pose = np.sum(self.pose_unfixed)
        self.free_bdpt = np.sum(self.bdpt_unfixed)

        self.intr_end = 9 * self.free_intr
        self.extr_end = 6 * self.free_extr + self.intr_end
        self.pose_end = 6 * self.free_pose + self.extr_end
        self.bdpt_end = 1 * self.free_bdpt + self.pose_end

    def return_bundle_primitives(self, params):
        """
        Takes an array of parameters and populates all unfixed parameters.
        :param params: The input parameters
        :return: The intrinsics, extrinsics, poses and feature points of the calibration.
        """


        intr_data = params[:self.intr_end].reshape((self.free_intr, 9))
        extr_data = params[self.intr_end:self.extr_end].reshape((self.free_extr, 6))
        pose_data = params[self.extr_end:self.pose_end].reshape((self.free_pose, 6))
        bdpt_data = params[self.pose_end:self.bdpt_end]

        ch.fill_flat(pose_data, self.poses, self.pose_unfixed)
        ch.fill_flat(extr_data, self.extr, self.extr_unfixed)
        ch.fill_flat(intr_data, self.intr, self.intr_unfixed)

        ch.fill_flat(bdpt_data, self.bundle_pts, self.bdpt_unfixed)

        return self.intr, self.extr, self.poses, self.bundle_pts.reshape((-1, 3))

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

        self.flat_point_data = np.copy(self.point_data.reshape((-1)))


        # if bundle_points_unfixed is not None:
        #     logging.warning(
        #         """
        #         A list of unfixed bundle points was provided. The calibration fixes arbitrary points to break gauge symmetries. 
        #         Unless overridden with the always_correct_gauge=True, the optimisation will no longer attempt to return the output geometry to the provided scale. 
        #         """)
        #     self.correct_gauge = always_correct_gauge
        #
        #     self.bdpt_unfixed = bundle_points_unfixed
        # else:

        #fix the gauge of the optimisation by fixing 7 params of the target

        self.fixed_inds = find_not_colinear_pts(self.flat_point_data.reshape((-1,3)))
        i0, i1, i2 = self.fixed_inds
        self.feat_unfixed = np.ones(self.flat_point_data.shape[0], dtype=bool)
        self.feat_unfixed[3*i0:3*i0+3] = False
        self.feat_unfixed[3*i1:3*i1+3] = False
        self.feat_unfixed[3*i2] = False

        self.bundlePrimitive = StandardBundlePrimitive(
            self.poses, self.flat_point_data, self.extr, self.intr,
            extr_unfixed=self.extr_unfixed, intr_unfixed=self.intr_unfixed, poses_unfixed=self.pose_unfixed, bundle_points_unfixed=self.feat_unfixed
        )

        self.param_len = None
        self.jac_mask = None
        self.missing_poses: list | None = missing_poses
        self.op_fun: fb.optimisation_function = fb.projection() + fb.extrinsic3D() + fb.rigidTform3d() +  fb.free_point()

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
                np.repeat(self.intr_unfixed, 9),
                np.repeat(self.extr_unfixed, 6),
                np.repeat(self.pose_unfixed, 6),
                np.repeat(self.feat_unfixed, 1), #I unrolled feature unfixed
            ), axis=0
        )

        temp_loss = self.op_fun.make_jacobean(dd, threads, unfixed_params=mask)
        def jac_fn(params):
            inps = self.get_bundle_adjustment_inputs(params) #return proj, extr, poses
            param_str = self.op_fun.build_param_list(*inps)
            d, c, rp = temp_loss(param_str)
            return csr_array((d,c,rp), shape=(2*dd.shape[0], params.shape[0]))
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
        """
        Sets the initial values of the calibration from a previous calibration of the same system.
        The previous system must have used a TemplateBundleHandler.
        :param prev_cams: The calibrated camseet to use.
        """
        self.initial_params = np.empty(self.bundlePrimitive.bdpt_end)

        if not isinstance(prev_cams.calibration_handler, TemplateBundleHandler):
            raise ValueError("Previous camera set was not a templated adjustment")
        self.initial_params[:self.bundlePrimitive.pose_end] = prev_cams.calibration_params.copy()
        self.initial_params[ 
            self.bundlePrimitive.pose_end:
        ] = prev_cams.calibration_handler.target.point_data.copy().flatten()[self.feat_unfixed]
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

        standard_model = self.bundlePrimitive.return_bundle_primitives(x)
        proj, extr, poses, ps = self.apply_gauge_transform(*standard_model)

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

    def apply_gauge_transform(self, proj, extr, poses, point_estimate) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Maps a set of parameters from an existing representation to the scale and transformation that best matches the provided model.
        The transformation is garunteed to preserve the calibration result.
        The first pose also remains as the identity.

        :param proj: The array describing the intrinsic + distortion of the camera. Untouched by this transform.
        :param extr: The array containing the extrinsics of the camera system.
        :param poses: The array containing the extimated poses of the calibration target.
        :param point_estimate: The array containing the estimated locations of the calibration target features.
        :return: A tuple containing updated proj, extr, poses, and point estimate.
        """

        ref_points = self.target.point_data.reshape((-1,3))
        valid_map = self.target.valid_map

        if isinstance(valid_map, bool):
            if valid_map == False:
                raise ValueError("Target has given a valid map of False, which indicates no distance comparisons are valid.")
            #use cdist, take the upper
            inds = np.triu_indices(point_estimate.shape[0], k=1)
            new_map = cdist(point_estimate, point_estimate)[inds]
            ref_map = cdist(ref_points, ref_points)[inds]
        elif isinstance(valid_map, np.ndarray):
            new_map = ch.calc_distance_subset(point_estimate, point_estimate, valid_map[:,:2])
            ref_map = ch.calc_distance_subset(ref_points, ref_points, valid_map[:,:2])
        else:
            raise ValueError("The target.valid_map property either needs to be true, for all comparisons being valid, or a nx2 list of index pairs.")
        s = np.mean(ref_map/new_map)
        new_points = s * point_estimate
        
        update_tform = gu.make_4x4h_tform(*ch.n_estimate_rigid_transform(new_points, ref_points)) #this mapping from used points to a reference space
        inv_update = np.linalg.inv(update_tform)
        new_points = gu.h_tform(new_points, update_tform)
        #proj matricies never change: scale invariance!

        for i in range(len(poses)):
            ### scale change
            poses[i][3:] = poses[i][3:] * s
            ### rigid change
            pose = gu.make_4x4h_tform(poses[i][:3], poses[i][3:])
            new_pose = update_tform @ pose @ inv_update
            poses[i][:3], poses[i][3:] = gu.ext_4x4_to_rod(new_pose)

        for i in range(len(extr)):
            ### scale change
            extr[i][3:] = extr[i][3:] * s
            ### rigid change
            og_tform = gu.make_4x4h_tform(extr[i][:3], extr[i][3:])
            new_tform = og_tform @ inv_update
            extr[i][:3], extr[i][3:] = gu.ext_4x4_to_rod(new_tform)
        return proj, extr, poses, new_points

    
    def special_plots(self, x):
        """
        An additional plot called to visualise the calibration. 
        Visualises the error in the calibration target that was recovered.
        """
        og_data = self.target.point_data.reshape((-1,3))
        
        un_gauged_data = self.get_bundle_adjustment_inputs(x)
        _,_,_, final_data = self.apply_gauge_transform(*un_gauged_data)
        unfixed_points = un_gauged_data[-1].copy()

        diff = (final_data - og_data) * 1000
        print(f"found a mean difference of {np.mean(np.linalg.norm(diff, axis=-1)):.2f} mm")
        s = pv.Plotter()
        s.title = "Target Self-calibration Results."
        s.add_arrows(og_data*100, diff, label = "Recovered shape change (10x mag)")
        s.remove_scalar_bar()
        s.add_scalar_bar(title="Euclidean displacement from initial model (mm).")
        s.add_mesh(pv.PolyData(og_data*100), color='r', label = "Original Model")
        s.add_mesh(pv.PolyData(final_data*100), color='g', label = "Best-scaled Model")
        s.add_mesh(pv.PolyData(unfixed_points*100), color='b', label = "As Calibrated Model")
        s.add_mesh(pv.Line((0,0,0), unfixed_points[self.fixed_inds[0]]*100), color='k', label="Points used to fix gauge symmetry")
        s.add_mesh(pv.Line((0,0,0), unfixed_points[self.fixed_inds[1]]*100), color='k')
        s.add_mesh(pv.Line((0,0,0), unfixed_points[self.fixed_inds[2]]*100), color='k')
        s.add_legend(bcolor='w', border=True)
        s.show()






