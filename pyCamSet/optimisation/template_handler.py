from __future__ import annotations
import logging
from copy import copy, deepcopy

import matplotlib.pyplot as plt

import numpy as np

from typing import TYPE_CHECKING
from scipy.sparse import csr_array

import pyCamSet.utils.general_utils as gu
import pyCamSet.optimisation.compiled_helpers as ch
import pyCamSet.optimisation.function_block_implementations as fb
import pyCamSet.optimisation.abstract_function_blocks as afb

from pyCamSet.calibration_targets import TargetDetection
    
if TYPE_CHECKING:
    from pyCamSet.calibration_targets import AbstractTarget
    from pyCamSet.cameras import CameraSet, Camera


DEFAULT_OPTIONS = {
    'verbosity': 2,
    'fixed_pose':0,
    'ref_cam':0,
    'ref_pose':0,
    'outliers':'ask',
    'max_nfev':100,
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

class TemplateBundleHandler:
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

        self.fixed_params = gu.list_dict_to_np_array(fixed_params)
        if fixed_params is None:
            self.fixed_params : dict = {}

        self.camset = camset
        self.cam_names = camset.get_names()
        self.detection = deepcopy(detection)
        self.target = target
        self.point_data = deepcopy(target.point_data)
        self.target_point_shape = np.array(target.point_data.shape)
        self.initial_params = None

        n_poses = detection.max_ims
        n_cams = camset.get_n_cams()

        intr = np.zeros((n_cams, 9))
        extr = np.zeros((n_cams, 6))
        poses = np.zeros((n_poses, 6))

        extr_unfixed = np.array(['ext' not in self.fixed_params.get(cam_name, {}) for cam_name in self.cam_names])
        intr_unfixed = np.array(['int' not in self.fixed_params.get(cam_name, {}) for cam_name in self.cam_names])
        pose_unfixed = np.ones(n_poses, dtype=bool)
        if "fixed_pose" in self.problem_opts:
            fixed_pose = self.problem_opts["fixed_pose"]
            pose_unfixed[fixed_pose] = False
            poses[fixed_pose, :] = [0,0,0,0,0,0]

        self.bundlePrimitive = TemplateBundlePrimitive(
            poses, extr, intr,
            extr_unfixed=extr_unfixed, intr_unfixed=intr_unfixed, poses_unfixed=pose_unfixed,
        )

        self.populate_self_from_fixed_params()


        self.param_len = None
        self.jac_mask = None
        self.missing_poses: list | None = missing_poses

        # we define an abstract function block to handle the calibration
        self.op_fun: afb.optimisation_function = fb.projection() + fb.extrinsic3D() + fb.template_points()

    def can_make_jac(self):
        return self.op_fun.can_make_jac() 

    def make_loss_fun(self, threads):
        
        #flatten the object shape
        obj_data = self.target.point_data.reshape((-1, 3)) #maybe this is wrong. 

        target_shape = self.target.point_data.shape
        dd = self.detection.return_flattened_keys(target_shape[:-1]).get_data()

        temp_loss = self.op_fun.make_full_loss_fn(dd, threads) 
        def loss_fun(params):
            inps = self.get_bundle_adjustment_inputs(params) #return proj, extr, poses
            param_str = self.op_fun.build_param_list(*inps)
            return temp_loss(param_str, obj_data).flatten()
        return loss_fun

    def make_loss_jac(self, threads): 
        #TODO implement proper culling
        obj_data = self.target.point_data.reshape((-1, 3))
        target_shape = self.target.point_data.shape
        dd = self.detection.return_flattened_keys(target_shape[:-1]).get_data()
        mask = np.concatenate(
            ( 
                np.repeat(self.bundlePrimitive.intr_unfixed, 9),
                np.repeat(self.bundlePrimitive.extr_unfixed, 6),
                np.repeat(self.bundlePrimitive.poses_unfixed, 6),
            ), axis=0
        )

        # breakpoint()

        temp_loss = self.op_fun.make_jacobean(dd, threads, unfixed_params=mask)
        def jac_fn(params):
            inps = self.get_bundle_adjustment_inputs(params) #return proj, extr, poses
            param_str = self.op_fun.build_param_list(*inps)
            d, c, rp = temp_loss(param_str, obj_data)
            return csr_array((d,c,rp), shape=(2*dd.shape[0], params.shape[0]))
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
                self.bundlePrimitive.extr[idx] = self.fixed_params[cam_name]['ext']

            if 'int' in self.fixed_params.get(cam_name, {}):
                self.bundlePrimitive.intr[idx] = self.fixed_params[cam_name]['int']

    def get_bundle_adjustment_inputs(self, x, make_points=False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This function uses the state of the parameter handler, and the given params
        to build np arrays that describe:

            - the projection matrices of the cameras
            - the distortion parameters of the cameras
            - the pose of every object point in every time point

        These are direct inputs to a bundle adjustment based loss.

        :param x: The input optimisation parameters.
        """
        proj, extr, poses = self.bundlePrimitive.return_bundle_primitives(x)
        # proj = intr @ extr[:, :3, :]
        if make_points:
            im_points = np.empty((len(poses), *self.point_data.shape))
            for idx, pose in enumerate(poses):
                blank = np.zeros((12))
                ch.n_e4x4_flat_INPLACE(pose, blank)
                ch.n_htform_broadcast_prealloc(self.point_data, blank, im_points[idx])

            im_points = np.reshape(im_points, (len(poses), -1, 3))
            return im_points
        # return proj, intr, dst, im_points
        return proj, extr, poses

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
                    # max_loc = np.argmax(per_im_error[outlier_inds])
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

        cam_poses, target_poses, per_im_error = estimate_camera_relative_poses(
            detection=self.detection, cams=self.camset, calibration_target=self.target
        )

        self.missing_poses = np.array([np.isnan(t[0,0]) for t in target_poses])
        self.find_and_exclude_transform_outliers(per_im_error)
        
        for idc, intr_unfixed in enumerate(self.bundlePrimitive.intr_unfixed):
            if intr_unfixed:
                param_array.append(
                    np.concatenate(
                        (cams[idc].intrinsic[[0, 0, 1, 1], [0, 2, 1, 2]].squeeze(), 
                         cams[idc].distortion_coefs.squeeze()),
                        axis=0,
                    )
                )

        for idc, ext_unfixed in enumerate(self.bundlePrimitive.extr_unfixed):
            if ext_unfixed:
                ext = gu.ext_4x4_to_rod(cam_poses[idc])
                param_array.append(ext[0])
                param_array.append(ext[1])


        for idp, pose_unfixed in enumerate(self.bundlePrimitive.poses_unfixed):
            # how do we handle missing poses - delete the image associated with the missing daata
            if pose_unfixed:
                ext = gu.ext_4x4_to_rod(target_poses[idp])
                param_array.append(ext[0])
                param_array.append(ext[1])

        param_array = np.concatenate(param_array, axis=0)
        return param_array

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

        ps = np.empty((len(poses), 12))
        for pn, p in zip(ps, poses):
            ch.n_e4x4_flat_INPLACE(p, pn)

        return new_cams, ps

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
    
    def gauge_fixes(self):
        """
        Defines the lagrange multiplier conditions of this optimiser that fix any gauge symmetries in the optimisation.

        :returns fn or None: if no lagrange multipliers, returns None, otherwise returns a function that evals the multipliers. f
        """
        return None
        


def check_for_target_misalignment(tforms:  np.ndarray, ref_cam:int = 0):
    """
    Checks for misalignment in the viewed target by looking at the variances of the relative transformations to the first camera, which is treated as a reference.

    :param tforms: the set of transformations to evaluate. c x p x 4 x 4 array where c is the number of cameras and p the number of poses/images
    """
    
    Mrc_at = [np.linalg.inv(p) for p in tforms[ref_cam]]
    Marc_ac = np.array([
        [(Mt_c @ Mrc_t) for Mrc_t, Mt_c in zip(Mrc_at, Mat_c)] 
        for Mat_c in tforms
    ])

    for ic, Marc_c in enumerate(Marc_ac):
        if ic == ref_cam:
            continue
        angs = np.array([np.arccos((np.trace(t[:3,:3]) - 1)/2) for t in Marc_c])
        mags = [np.linalg.norm(t[:3,-1]) for t in Marc_c]
        std_ang = np.nanstd(angs)
        std_mag = np.nanstd(mags)
        if std_mag > 0.050:
            logging.critical(f"Found inconsistent relative translation positions (stdev = {std_mag:.2f} m) for camera index {ic}")
            logging.warning(f"This may indicate misordered images, temporal misalignment, or very bad detections, and is likely to cause calibration difficulties.") 
        if std_ang > 5 / 180 * np.pi:
            logging.critical(f"Found inconsistent relative angle magnitudes (stdev = {std_ang/np.pi*180:.2f} degrees) for camera index {ic}")
            logging.warning(f"This may indicate misordered images, temporal misalignment, or very bad detections, and is likely to cause calibration difficulties.") 

def check_feasiblity_and_update_refpose(Mat_ac, ref_pose: int) -> int:
    """
    This function examines a set of input transformations, and attemps to find out if there is a possible reference.
    """
    visibility = np.isnan(Mat_ac[:,:,0,0])
    visible_pose = ~np.any(visibility, axis=0)
    vrf_pose = visible_pose[ref_pose]
    if not vrf_pose:
        f_index = np.argmax(visible_pose)
        if f_index == 0 and not visibility[0]:
            raise ValueError("Couldn't find an initial pose for all cameras.")
        ref_pose = f_index
    return ref_pose

def estimate_camera_relative_poses(
        calibration_target: AbstractTarget, detection: TargetDetection,
        cams:CameraSet, ref_cam: int = 0, ref_pose: int = 0,
        max_bad_cams_iter = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given camera estimates, performs a single camera centric pose estimate.
    This reference camera is used to generate initial estimates of all other camera poses.  If the reference pose does not have full visibility, it will pick a new reference pose. This is explicitely not a graph based solver, and will throw an error if no fully shared target is visible.
    As it assesses the cost function to pick sane defaults, it also returns an initial evaluation of the per im error of the cost function.

    :param detection: the detection data that the average estimates should be extracted from
    :param cams: the list of cameras with no yet known average transformation
    """

    # a indicates that there is a full array allong a dimension p3

    img_detections = detection.get_image_list()
    Mat_ac = []
    for cam in cams:
        pose_per_img=[]
        for id in img_detections:
            pose_per_img.append(calibration_target.target_pose_in_cam_image(id, cam, mode="nan"))
        Mat_ac.append(pose_per_img)
    Mat_ac = np.array(Mat_ac)

    ref_pose = check_feasiblity_and_update_refpose(Mat_ac, ref_pose) 
    # check_for_target_misalignment(Mat_ac, ref_cam) #TODO, refactor this to a single summary.
    
    Mrt_rc = Mat_ac[ref_cam, ref_pose]
    Mrt_ac = Mat_ac[:, ref_pose]
    
    Mac_rt = np.array([np.linalg.inv(Mrt_c) for Mrt_c in Mrt_ac])

    
    Mat_rt_ac = Mac_rt[:, None, ...] @ Mat_ac
    

    # build the projection matrix as an input to the target.
    dists = np.array([cam.distortion_coefs for cam in cams]).squeeze()
    ints = np.array([cam.intrinsic for cam in cams])
    proj = ints @ Mrt_ac[:, :3, :]

    # run a bundle adjustment over the possible target positions.
    ps = calibration_target.point_data.reshape((-1, 3)) #could the flattening be failing for things that aren't flat
    target_shape = calibration_target.point_data.shape
    dd = detection.return_flattened_keys(target_shape[:-1]).get_data() #maybe this isn't in order.

    lookups =  []
    for i in range(detection.max_ims):
        lookups.append(dd[:,1] == i)

    
    # cameras_converged = False
    # while not cameras_converged:
    
    errors = []
    for Mat_rt_c in Mat_rt_ac: #project the 
        nanform = np.isnan(Mat_rt_c[:, 0,0])
        # print(np.sum(nanform), 'nan poses')
        # Mat_rt_c[nanform] = np.eye(4)#
        for idn, wasnan in enumerate(nanform):
            if idn==0 and wasnan:
                raise ValueError("No pose in first image")
            if wasnan:
                Mat_rt_c[idn] = Mat_rt_c[idn - 1]


        imlocs = np.array([gu.h_tform(ps,Mt_rt_c) for Mt_rt_c in Mat_rt_c]) 
        try:
            costs = ch.bundle_adjustment_costfn(
                dd,
                imlocs,
                proj,
                ints,
                dists,           
            )
        except ZeroDivisionError:
            costs = ch.numpy_bundle_adjustment_costfn(
                dd,
                imlocs,
                proj,
                ints,
                dists,           
            )

        costs = np.sqrt(np.sum(costs.reshape(-1,2)**2, axis=1))
        im_costs = []
        for l, wasnan in zip(lookups, nanform):
            total_costs = np.sum(costs[l])
            reasonable_bound = np.prod(costs[l].shape) * 1000 
            # if total_costs > reasonable_bound: # or wasnan:
            #     total_costs = np.nan
            im_costs.append(total_costs)
        errors.append(im_costs)

    errors = np.array(errors)

    # cam_good = np.sum(np.isnan(errors), axis=1) < 0.9 * errors.shape[1]
    
    # what to do if we think a camera is bad?

    # rederive a new estimate for the location of the camera in a different frame.
    # cam to cam, cam to target 


    # do a check to see if a camera is really bad.




    estimate_locs = np.argmin(errors, axis=0)


    Mat_rc = Mat_ac[ref_cam, :]
    Mat_rt = np.array([Mt_rt_ac[e] for e, Mt_rt_ac in zip(estimate_locs, Mat_rt_ac.transpose((1,0,2,3)))])
    

    imlocs = np.array([gu.h_tform(ps,Mt_rt) for Mt_rt in Mat_rt]) 
    costs = ch.bundle_adjustment_costfn(
        dd,
        imlocs,
        proj,
        ints,
        dists,           
    )
    costs = np.sqrt(np.sum(costs.reshape(-1,2)**2, axis=1))
    for l in lookups:
        im_costs.append(np.sum(costs[l]))


    init_per_im_reproj_err = np.array(im_costs)

    Mat_rt[ref_pose] = np.eye(4)
    return Mrt_ac, Mat_rt, init_per_im_reproj_err
