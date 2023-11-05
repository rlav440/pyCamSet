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
    from pyCamSet.cameras import CameraSet


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

        ch.fill_pose(pose_data, self.poses, self.poses_unfixed)
        ch.fill_extr(extr_data, self.extr, self.extr_unfixed)
        ch.fill_intr(intr_data, self.intr, self.intr_unfixed)
        ch.fill_dst(dst_data, self.dst, self.dst_unfixed)
        return self.poses, self.extr, self.intr, self.dst


class StandardParamHandler:
    """
    The standard param handler is a class that handles the optimisation of camera parameters.
    It is designed to be used with the numba implentation of the bundle adjustment cost function.
    It takes a CameraSet, a Target and the associated TargetDetection.
    Given these, it will return a function that takes a parameter array and returns data structures ready for
    evaluation with the bundle adjustment cost function.
    The implementation given in the standard param handler implements a standard object pose based bundle adjustment.

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
                mloc,
                out_thresh=10,
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
        param_array = self.add_extra_params(param_array)
        return np.concatenate(param_array, axis=0)

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
    visible_pose = ~np.any(visibility, axis=1)
    vrf_pose = visible_pose[ref_pose]
    if not vrf_pose:
        f_index = np.argmax(visible_pose)
        if f_index == 0 and not visibility[0]:
            raise ValueError("Couldn't find an initial pose for all cameras.")
        ref_pose = f_index
    return ref_pose

def estimate_camera_relative_poses(
        calibration_target: AbstractTarget, detection: TargetDetection,
        cams:CameraSet, ref_cam: int = 0, ref_pose: int = 0
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
    check_for_target_misalignment(Mat_ac, ref_cam)
    
    Mrt_rc = Mat_ac[ref_cam, ref_pose]
    Mrt_ac = Mat_ac[:, ref_pose]
    
    Mac_rt = np.array([np.linalg.inv(Mrt_c) for Mrt_c in Mrt_ac])
    Mat_rt_ac = Mac_rt[:, None, ...] @ Mat_ac
    

    # build the projection matrix as an input to the target.
    dists = np.array([cam.distortion_coefs for cam in cams]).squeeze()
    ints = np.array([cam.intrinsic for cam in cams])
    proj = ints @ Mrt_ac[:, :3, :]

    # run a bundle adjustment over the possible target positions.
    errors = []
    ps = calibration_target.point_data.reshape((-1, 3)) #could the flattening be failing for things that aren't flat
    target_shape = calibration_target.point_data.shape
    dd = detection.return_flattened_keys(target_shape[:-1]).get_data()

    lookups =  []
    for i in range(detection.max_ims):
        lookups.append(dd[:,1] == i)
    
    for Mat_rt_c in Mat_rt_ac:
        nanform = np.isnan(Mat_rt_c[:, 0,0])
        Mat_rt_c[nanform] = np.eye(4) #if this wins, it wins.
        imlocs = np.array([gu.h_tform(ps,Mt_rt_c) for Mt_rt_c in Mat_rt_c]) 
        costs = ch.bundle_adjustment_costfn(
            dd,
            imlocs,
            proj,
            ints,
            dists,           
        )
        costs = np.sqrt(np.sum(costs.reshape(-1,2)**2, axis=1))
        im_costs = []
        for l in lookups:
            im_costs.append(np.sum(costs[l]))
        errors.append(im_costs)

    errors = np.array(errors)
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
    im_costs = []
    for l in lookups:
        im_costs.append(np.sum(costs[l]))
    init_per_im_reproj_err = np.array(im_costs)

    Mat_rt[ref_pose] = np.eye(4)
    return Mrt_ac, Mat_rt, init_per_im_reproj_err

    # ert refcam -> other_cams
    Mat_arc = Mac_rc[:, None, ...] @ Mat_ac
    plt.imshow(np.isnan(np.array(Mat_arc)[:,:,0,0]))
    plt.show()
    Mat_rc = np.array([gu.average_tforms(Mat_rc) for Mat_rc in Mat_arc.transpose((1,0,2,3))])

    # Mat_rc = Mat_arc[ref_cam, ...] 

    #etp refcam -> cube_loc
    Mrt_rc = Mat_rc[ref_pose]
    Mrc_rt = np.linalg.inv(Mrt_rc)  #is reftarget -> refcam
    Mac_rt = Mrc_rt @ Mac_rc # cam -> refcam -> reftarget
    Mrt_ac = Mac_rc @ Mrt_rc
    Mat_rt = Mrc_rt @ Mat_rc  # cube_loc -> refcam -> refcube
    return Mrt_ac, Mat_rt

def make_optimisation_function(
        param_handler: StandardParamHandler,
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


def run_bundle_adjustment(param_handler: StandardParamHandler,
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

