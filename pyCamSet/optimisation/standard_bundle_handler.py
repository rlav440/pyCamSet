
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
        self.poses_unfixed = poses_unfixed if poses_unfixed is not None else np.ones(poses.shape[0], dtype=bool)
        self.calc_type_inds()

    def calc_type_inds(self):
        """
        Updates the internal indicies where different params are stored internally.
        """

        self.free_extr = np.sum(self.extr_unfixed)
        self.free_intr = np.sum(self.intr_unfixed)
        self.free_pose = np.sum(self.poses_unfixed)
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

        ch.fill_flat(pose_data, self.poses, self.poses_unfixed)
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

        # then look at the detection data - if a feature isn't seen, report it as unseen
        n_points = np.prod(self.point_data.shape[:2])
        dd = self.detection.return_flattened_keys(self.target.point_data.shape[:-1]).get_data()[:, 2]
        # cd = np.arange(n_points)//81
        # good_face_mask = (cd == 1) | (cd == 4) | (cd == 5)
        # self.visible_feature_mask = np.isin(np.arange(n_points), dd) & good_face_mask
        self.visible_feature_mask = np.isin(np.arange(n_points), dd) 
        for idf, vf in enumerate(self.visible_feature_mask): #fix all unseen features to shrink the optimisation
            if not vf:
                self.feat_unfixed[3*idf:3*idf + 3] = False
        

        superBundlePrimitive = self.bundlePrimitive

        self.bundlePrimitive = StandardBundlePrimitive(
            superBundlePrimitive.poses, self.flat_point_data, superBundlePrimitive.extr, superBundlePrimitive.intr,
            extr_unfixed=superBundlePrimitive.extr_unfixed, intr_unfixed=superBundlePrimitive.intr_unfixed, poses_unfixed=superBundlePrimitive.poses_unfixed, bundle_points_unfixed=self.feat_unfixed
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
                np.repeat(self.bundlePrimitive.intr_unfixed, 9),
                np.repeat(self.bundlePrimitive.extr_unfixed, 6),
                np.repeat(self.bundlePrimitive.poses_unfixed, 6),
                np.repeat(self.bundlePrimitive.bdpt_unfixed, 1), #I unrolled feature unfixed
            ), axis=0
        )

        temp_loss = self.op_fun.make_jacobean(dd, threads, unfixed_params=mask)
        def jac_fn(params):
            inps = self.get_bundle_adjustment_inputs(params) #return proj, extr, poses
            param_str = self.op_fun.build_param_list(*inps)
            d, c, rp = temp_loss(param_str)
            return csr_array((d,c,rp), shape=(2*dd.shape[0], params.shape[0]))
        return jac_fn

    def get_bundle_adjustment_inputs(self, x, make_points=False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        if make_points:
            im_points = np.empty((len(poses), *self.point_data.shape))
            for idx, pose in enumerate(poses):
                blank = np.zeros((12))
                ch.n_e4x4_flat_INPLACE(pose, blank)
                ch.n_htform_broadcast_prealloc(bundle_points.reshape(self.point_data.shape), blank, im_points[idx])

            im_points = np.reshape(im_points, (len(poses), -1, 3))
            return im_points

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
        self.missing_poses =  prev_cams.calibration_handler.missing_poses
        print(self.missing_poses)
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

        if self.initial_params is not None:
            return self.initial_params
        start_params = self.calc_initial_params()

        self.initial_params = np.empty(self.bundlePrimitive.bdpt_end)
        self.initial_params[:self.bundlePrimitive.pose_end] = start_params
        self.initial_params[ 
            self.bundlePrimitive.pose_end:
        ] = self.target.point_data.copy().flatten()[self.feat_unfixed]
        return self.initial_params

    def get_updated_target(self, x):
        standard_model = self.bundlePrimitive.return_bundle_primitives(x)
        proj, extr, poses, ps = self.apply_gauge_transform(*standard_model)
        return ps

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
        n_points = np.prod(self.point_data.shape[:2])
        cd = np.arange(n_points)//(n_points//6)
        good_face_mask = np.ones(n_points, dtype=bool) #(cd == 1) | (cd == 4) | (cd == 5)

        ref_points = self.target.point_data.reshape((-1,3))
        valid_map = self.target.valid_map
        vm = self.visible_feature_mask & good_face_mask
        
        if isinstance(valid_map, bool):
            if valid_map == False:
                raise ValueError("Target has given a valid map of False, which indicates no distance comparisons are valid.")
            #use cdist, take the upper
            inds = np.triu_indices(point_estimate[vm].shape[0], k=1)
            new_map = cdist(point_estimate[vm], point_estimate[vm])[inds]
            ref_map = cdist(ref_points[vm], ref_points[vm])[inds]
            dt = self.target.square_size 
            # dt = 0.0045 #hard coded for today
            mask = np.isclose(ref_map, dt)
            new_map = new_map[mask]
            ref_map = ref_map[mask]

        elif isinstance(valid_map, np.ndarray):
            new_map = ch.calc_distance_subset(point_estimate, point_estimate, valid_map[:,:2])
            ref_map = ch.calc_distance_subset(ref_points, ref_points, valid_map[:,:2])
        else:
            raise ValueError("The target.valid_map property either needs to be true, for all comparisons being valid, or a nx2 list of index pairs.")
        s = np.mean(ref_map/new_map)
        new_points = s * point_estimate

        try:
            update_tform = gu.make_4x4h_tform(*ch.n_estimate_rigid_transform(
                new_points[self.visible_feature_mask & good_face_mask],
                ref_points[self.visible_feature_mask & good_face_mask])
            ) #this mapping from used points to a reference space
        except Exception as e:
            logging.critical("Failed to find an acceptable gauge transform, returning the identity")
            logging.critical("Gave error: ")
            print(e)
            update_tform = np.eye(4)

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
        n_points = np.prod(self.point_data.shape[:2])
        cd = np.arange(n_points)//(n_points//6)

        t0, t1, t2 = 1, 4, 5
        #
        good_face_mask = (cd == t0) | (cd == t1) | (cd == t2)
        m1 = cd == t0
        m4 = cd == t1
        m5 = cd == t2

        vm = self.visible_feature_mask & good_face_mask
        # vm = np.ones_like(vm)
        # m1 = vm.copy()
        # m4 = vm.copy()
        # m5 = vm.copy()

        
        un_gauged_data = self.get_bundle_adjustment_inputs(x)
        _,_,_, final_data = self.apply_gauge_transform(*un_gauged_data)
        unfixed_points = un_gauged_data[-1].copy()

        diff = (final_data - og_data) * 1000

        #xclude difs over 2 mm
        mask = np.linalg.norm(diff, axis=1) < 2
        vm &= mask

        scale = 5
        descale = 1000//scale
        print(f"found a mean difference of {np.mean(np.linalg.norm(diff[vm], axis=-1)):.2f} mm")
        s = pv.Plotter()
        s.title = "Target Self-calibration Results."
        s.add_arrows((og_data*descale)[vm], diff[vm], label = f"Recovered shape change ({scale}x mag)", 
                # cmap='Blues',
                cmap="Greens",
                # cmap='Oranges',
        )
        s.remove_scalar_bar()
        s.add_scalar_bar(title="Euclidean displacement from initial model (mm).")
        s.add_mesh(pv.PolyData(og_data*descale), color='k', label = "Original Model", point_size=0.3)
        # s.add_mesh(pv.PolyData(final_data*descale), color='g', label = "Best-scaled Model")
        # s.add_mesh(pv.Line((0,0,0), unfixed_points[self.fixed_inds[0]]*descale), color='k', label="Points used to fix gauge symmetry")
        # s.add_mesh(pv.Line((0,0,0), unfixed_points[self.fixed_inds[1]]*descale), color='k')
        # s.add_mesh(pv.Line((0,0,0), unfixed_points[self.fixed_inds[2]]*descale), color='k')

        p1, np1, cp1 = pv.fit_plane_to_points(final_data[vm & m1]*descale, return_meta=True)
        p4, np4, cp4 = pv.fit_plane_to_points(final_data[vm & m4]*descale, return_meta=True)
        p5, np5, cp5 = pv.fit_plane_to_points(final_data[vm & m5]*descale, return_meta=True)

        s1 = pv.PolyData(og_data[m1]*descale, lines=make_connectivity(og_data[m1]))
        s.add_mesh(s1, style='wireframe', line_width=2, color='k', opacity=0.1)
        s4 = pv.PolyData(og_data[m4]*descale, lines=make_connectivity(og_data[m4]))
        s.add_mesh(s4, style='wireframe', line_width=2, color='k', opacity=0.1)
        s5 = pv.PolyData(og_data[m5]*descale, lines=make_connectivity(og_data[m5]))
        s.add_mesh(s5, style='wireframe', line_width=2, color='k', opacity=0.1)
        rms1 = rms_plane(np1, cp1, final_data[vm & m1]*descale)
        rms4 = rms_plane(np4, cp4, final_data[vm & m4]*descale)
        rms5 = rms_plane(np5, cp5, final_data[vm & m5]*descale)
        print(rms1, rms4, rms5)
        # # s.add_mesh(pv.PolyData(final_data[vm & m2]), point_size=6)
        # s.add_mesh(p1, color='lightblue', opacity=0.7)
        # s.add_mesh(p4, color='lightblue', opacity=0.7)
        # s.add_mesh(p5, color='lightblue', opacity=0.7)
        s.add_legend(bcolor='w', border=True)
        
    
        a14 = angle_between_planes(cp1, cp4)
        a15 = angle_between_planes(cp1, cp5)
        a45 = angle_between_planes(cp4, cp5)

        print(a14, a15, a45)

        camera = s.camera
        camera.position = (-60, -60, -36)
        camera.focal_point = (0,0,0)
        camera.up = (0,0,-1)
        camera.angle = 0.02

        s.show()


def make_connectivity(pts):
    n_pts = pts.shape[0]
    n_points_per_line = int(np.sqrt(pts.shape[0]))
    #take the input points
    connectivity = []
    for idp, _ in enumerate(pts):
        if (idp + n_points_per_line < n_pts):
            connectivity.extend([2, idp, idp + n_points_per_line])
        if not ((idp +1) % n_points_per_line == 0):
            connectivity.extend([2, idp, idp + 1])
    return connectivity



def rms_plane(c, n, data):
    norms = np.sum((data - c) * n, axis=1)
    return np.mean(np.abs(norms), axis=0)


def angle_between_planes(normal1, normal2):
    # Normalize the vectors
    normal1_unit = normal1 / np.linalg.norm(normal1)
    normal2_unit = normal2 / np.linalg.norm(normal2)
    
    # Calculate the dot product
    dot_product = np.dot(normal1_unit, normal2_unit)
    
    # Calculate the angle in radians and then convert to degrees
    angle = np.arccos(dot_product)
    return np.degrees(angle)
