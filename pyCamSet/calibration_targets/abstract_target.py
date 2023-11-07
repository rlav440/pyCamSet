from __future__ import annotations

from abc import ABC, abstractmethod

import logging
import numpy as np
import time
from pathlib import Path
from copy import copy
import cv2
from natsort import natsorted


from pyCamSet.utils.general_utils import glob_ims, h_tform, make_4x4h_tform, mad_outlier_detection, plane_fit
from pyCamSet.cameras import CameraSet, Camera
from pyCamSet.calibration_targets.target_detections import TargetDetection, ImageDetection


def get_keys(data):
    keys = data[:, 2:-2]  # this slicing is always 2d.
    if keys.shape[1] == 1:
        keys = np.concatenate((np.zeros_like(keys), keys), axis=1)
    return keys

class AbstractTarget(ABC):
    """
    This is an abstract calibration target. It implements most of the functionality
    required to calibrate an object with a calibration target.
    Inheriting targets must define:
    1. A function to detect themselves in an input image
    2. the self.point_data array. For a target this is an array of the generic form
    (u,v ... w, n, 3) where u ... w are relevant dimensions, and n,3 is "n" 3d
    and co planar points. As an example, a cube forms a (6,n,3) point_data array, as there are 6 co-planar faces with n points on each face.

    Saving is achieved by calling the __init__ of the super and passing a dictionary of
    the required parameters to initialise the target.

    Inheriting targets may also define: a printable shape, which is either the target, or can then be folded onto a base shape and a plotting function which draws some visualisation of the target.
    """

    def __init__(self, inputs: dict):
        inputs.pop('self', None)
        inputs.pop('__class__', None)

        self.point_data: np.ndarray = None # An

        self.point_local = None #: np.ndarray = self.make_local()
        self.original_points = None # = self.point_data.copy()
        self.input_args = inputs

    def _process_data(self):
        """
        A function called at the end of the __init__ of any inhereting class
        """
        self.point_local = self.make_local()
        self.original_points = self.point_data.copy()

    def plot(self):
        """
        Notes: A function to plot and view the object instance.
        """

        raise NotImplementedError

    def save_to_pdf(self):
        """
        Notes: Implemented by a method that saves a printable version of the
        target to a pdf. Printable version contains information necessary to
        recreate
        """
        raise NotImplementedError

    @abstractmethod
    def find_in_image(self, image, draw=False, camera: Camera=None, wait_len = 1) -> ImageDetection:
        """
        Notes: Detects the calibration target in an image

        :param image: a mxn or mxnx3 image input
        :param draw: whether to draw the target
        :param camera: A camera object for use in camera aware detections
        :return: An ImageDetection object, containing the detected data
        """
        raise NotImplementedError

    def find_in_imfolder(self, file:Path, cam_names, draw=False, n_lim=None, camera: Camera=None) -> TargetDetection:
        """
        Notes: A function to detect the camera results in the image folder.
        generally a process wrapper around the previous function

        :param folder: the top level folder containing the input images
        :param cam_names:
        :param draw:
        :param n_lim: limit on the number of images used
        :param camera:

        :return detections: a TargetDetection container for the detection.
                This function is responsible for giving image numbers and camera type to the detector.

        """

        cam_name = file.parts[-1]
        im_locs = [str(x) for x in glob_ims(file)]
        im_locs = natsorted(im_locs)
        if n_lim is not None:
            im_locs = im_locs[:n_lim]

        if cam_names is None:
            cam_names = [cam_name]

        detections = TargetDetection(cam_names=cam_names)
        for idx, im_file in enumerate(im_locs):
            im = cv2.imread(im_file)
            if im.ndim == 3:
                # im = np.mean(im, axis=-1).astype(np.uint8)
                im = im[:, :, 0]
            detection = self.find_in_image(im, draw=draw, camera=camera)
            detections.add_detection(cam_name, idx, detection)

        return detections


    def additional_params(self, x: np.ndarray) -> np.ndarray:
        """
        An object may have additional parameters that are, as yet,
        undefined.
        This method provides a way to pull those results, and transform the state
        of the calibration target.
        """
        return x

    def parametise_features(self, detections: TargetDetection, camset:CameraSet, ref_cam=0):
        """
        A function to parametise any non pose related parameters of the object.
        if there are no such parameters, the function returns none.
        """
        return None

    def pose_in_detections(self, detections: TargetDetection, camset: CameraSet, ref_cam=0
                           ) -> tuple[list[np.ndarray], list[bool]]:
        """
        Returns a list of the found poses of the object in each image of a detection.
        If a pose cannot be found,indicates that the pose is not good in an additional output array

        :param detections: the detections in which to find the pose
        :param camset: The camset to use to detect the poses
        :param ref_cam: if the pose defaults to a reference camera, this is the camera


        :return poses: a list of poses [4x4 homogenous transforms],
        :return pose_d: a boolean list indicating if a pose was found in an image number.

        """
        other_cams = set(range(camset.get_n_cams())) - {0}
        cam = camset[ref_cam]
        poses = []
        for im_list in detections.get_image_list():
            # first try to get the pose with the reference cam
            try:
                pose = self.target_pose_in_cam_image(im_list, cam)
                pose = (cam.cam_to_world @  #cam -> world
                    pose # cube-> cam
                )
            except:
                for other_cam in other_cams:
                    try:
                        pose = self.target_pose_in_cam_image(im_list, camset[other_cam])
                        pose = camset[other_cam].cam_to_world @ pose
                        break
                    except:
                        continue
                else:
                    pose = None
            poses.append(pose)
            # then try to get the pose with any other cams.
        p_detected = np.array([False if p is None else True for p in poses])
        poses = [p for p in poses if p is not None]
        mloc = np.mean([p[:3, 3] for p in poses], axis=0)

        cyclic_outlier_detection = True
        num_loops = 0
        print("Begining outlier detection")
        while cyclic_outlier_detection and num_loops < 10:
            ans = mad_outlier_detection([np.linalg.norm(p[:3,3] - mloc) for p in poses], out_thresh=5)
            inds = np.arange(len(p_detected))[p_detected][ans]
            if ans is not None:
                user_in = "g"
                while not (user_in == 'y' or user_in == 'n'):
                    print(f"Outliers detected in iteration {num_loops}.")
                    user_in = input("Do you wish to remove these outliers?: \n y/n: ")

                if user_in == 'y':
                    def del_list_numpy(l, id_to_del):
                        arr = np.array(l)
                        return list(np.delete(arr, id_to_del, axis=0))

                    poses = del_list_numpy(poses, ans)
                    p_detected[inds] = False
                if user_in == 'n':
                    cyclic_outlier_detection = False
            else:
                print(f"No outliers detected in iteration {num_loops}.")
                cyclic_outlier_detection = False
            num_loops += 1

        return poses, p_detected

    def make_local(self):
        """
        The self point data is of the general form: (u,v, ... w, n, 3)
        Calibration approaches assume that each face is locally flat with z = 0
        This computes, for every sub structure (u,v, ..., w) a locally flat sub structure
        representation with the z axis = 0;
        """

        if self.point_data is None:
            raise AttributeError("The self.point_data variable should be set during initialisation")

        if self.point_data.ndim == 2:
            self.point_data = self.point_data[None, ...]

        init_shape = self.point_data.shape

        n = init_shape[-2]
        local_view = np.reshape(self.point_data, (-1, n, 3))

        if local_view.shape[0] == 1:
            return copy(self.point_data)
        ref_point = local_view[:, 0, :]
        init_dir = local_view[:, 1, :] - ref_point

        normals = []
        for face in local_view:
            normals.append(plane_fit(face.T)[1])
        normals = np.array(normals)

        # create the change of basis maxtrixes
        v_3 = np.array([np.cross(v_d, v_n) for v_d, v_n in zip(init_dir, normals)])

        v_3 /= np.linalg.norm(v_3, axis=1, keepdims=True)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        init_dir /= np.linalg.norm(init_dir, axis=1, keepdims=True)

        cob_mats = [np.linalg.inv(a) for a in np.stack(
            (v_3, init_dir, normals)
        ).transpose((1, 0, 2))]
        cob_mats = np.array(cob_mats)
        ref_0 = local_view - ref_point[:, None, :]

        local_coords = (
            ref_0 @ cob_mats
        )
        return np.reshape(local_coords, init_shape)

    def initial_calibration(self, cam_name, detection: TargetDetection,
                            res: list, pose_im: int=0,
                            fixed_params: dict|None =None) -> Camera:
        """
        Takes a single camera's detections, and performs an initial
        calibration on them.
        If the object has a special model of calibration associated, this can
        be overwritten.

        :param cam_name: The name of the camera being calibrated
        :param detection: A TargetDetection of only the detections of the currently
            being calibrated camera
        :param res: The resolution of the camera being calibrated.
        :param pose_im: The image in which the Target's pose sets the coordinate system
        :param fixed_params: A dict containing any fixed params of the camera to calibrate
            accepted options are "ext", "int", and "dst" respectively.
        :return: A camera object.
        """

        detections_in_image = detection.get(cam=cam_name).get_image_list()
        object_points = []
        image_points = []

        pre_defined_camera = False
        init_cam = Camera()
        fixed_param = {}
        if fixed_params is not None:
            fixed_param = fixed_params.get(cam_name, {})
            if "int" in fixed_param and "dst" in fixed_param:
                init_cam = Camera(intrinsic=fixed_param['int'], distortion_coefs=fixed_param['dst'], res=res, name=cam_name)
                logging.info(f'Camera {cam_name} was pre determined. Skipping opencv calibration')
                return init_cam


        for im_detect in detections_in_image:
            data = im_detect.get_data()
            if data is None:
                continue  # no data here, so don't add it to the optimisation
            keys = get_keys(data) # this slicing is always 2d.
            boards, board_id, b_counts = np.unique(keys[:, :-1], return_inverse=True,   return_counts=True)
            mask = b_counts > np.prod(self.point_local.shape[:-2])
            for board in boards[mask]:
                key_mask = np.squeeze(keys[:, :-1] == board)
                if np.sum(key_mask) > 8:
                    board_obj = self.point_local[tuple(keys[key_mask].astype(int).T)][None, ...].astype('float32')
                    board_im = data[key_mask, -2:][None, ...].astype('float32')
                    object_points.append(board_obj)
                    image_points.append(board_im)

        start = time.time()
        ic = cv2.calibrateCamera(
            object_points,
            image_points,
            tuple(res[::-1]),
            None,
            None,
            None,
            None,
        )
        end = time.time()

        logging.info(f'{cam_name} took {end - start:.1f} seconds'
            f', leftover error of {ic[0]:.2f} pixels')

        # perform an initial pose estimate on the first images

        init_cam = Camera(intrinsic=ic[1], distortion_coefs=np.array(ic[2]), res=res, name=cam_name)
        if fixed_params is not None:
            if "int" in fixed_param:
                init_cam.intrinsic = fixed_param['int']
            if 'dst' in fixed_param:
                init_cam.distortion_coefs = fixed_param['dst']
            if "ext" in fixed_param:
                init_cam.set_extrinsic(fixed_param['ext'])
                return init_cam

        return init_cam

    def target_pose_in_cam_image(
            self, detection: TargetDetection, cam: Camera, 
            refine:bool = False, mode="throw") -> np.ndarray:
        """
        This function gives a pose estimate of the cube in an image as seen by a camera.

        :param detection: a detection containing data from a single image.
        :param cam: a camera model to use
        :param refine: Whether to use LM refinement of the estimate.
        :param mode: whether to throw an error or return nan arrays.

        :return a 4x4 transformation of the target giving the transformation from target to camera coordinates
        """

        datum = detection.get(cam=cam.name).get_data()
        if datum is None:
            if mode == "nan":
                return np.ones((4,4)) * np.nan
            raise ValueError(f"The detection had no data for camera {cam.name}")

        n_im = np.unique(datum[:, 0])  # check that only one camera and one image is here.
        if len(n_im) > 1:
            if mode == "nan":
                return np.ones((4,4)) * np.nan
            raise ValueError(f"passed detection contained info from {n_im} ims. \n"
                "Pose estimation only works with 1 image")

        keys = get_keys(datum)
        object_points = self.point_data[tuple(keys.astype(int).T)]
        image_points = datum[:, -2:]
        if len(object_points) < 6:
            if mode == "nan":
                return np.ones((4,4)) * np.nan
            raise ValueError("Inadequate number of corners for pose estimation")

        _, rvec, tvec, err_list = cv2.solvePnPGeneric(object_points.astype("float32"),
                                                      image_points.astype("float32"),
                                                      cam.intrinsic,
                                                      cam.distortion_coefs
                                                      )
        min_err = np.argmin(err_list)
        if (err := err_list[min_err].squeeze()) > 5:
            logging.warning(f"Initial error of {err: .2f} found for a pose detection.")
        ext = make_4x4h_tform(
            rvec[min_err],
            tvec[min_err],
        )
        
        # temp_points = np.array([h_tform(face, ext) for face in self.point_data])
        # new_obj_points = temp_points[tuple(keys.astype(int).T)]
        # proj_uv = cam.project_points(new_obj_points)
        # errors = proj_uv - image_points
        # print(f"manual error check gave {np.mean(np.abs(errors))}")

        if not refine:
            return ext # from target -> cam coordinates
        else: 
            raise NotImplementedError

