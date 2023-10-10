
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import numpy as np
from tqdm import tqdm
from pathlib import Path
from functools import reduce
import cv2
import warnings
import re
import os

from pyCamSet.cameras import CameraSet
from pyCamSet.calibration_targets import TargetDetection, AbstractTarget
from pyCamSet.optimisation.base_optimiser import run_bundle_adjustment, AbstractParamHandler
from pyCamSet.optimisation.derived_handlers import StandardBundleParameters
from pyCamSet.utils.saving import save_pickle, load_pickle, load_CameraSet
from pyCamSet.utils.general_utils import get_subfolder_names, glob_ims, mad_outlier_detection

import coloredlogs, logging

coloredlogs.install(level=logging.INFO)


class CameraCalibrator:
    """
    Notes: This camera calibrator class has a few methods
    """

    def __init__(self, cameras: CameraSet = None):
        self.detection: TargetDetection = None
        self.camera_set = cameras
        self.initial_calibration = None
        self.camera_resolutions = None
        self.optim_results = None

    def __call__(self,
                 f_loc: Path,
                 calibration_target: AbstractTarget,
                 save: bool = True,
                 save_loc: Path = None,
                 draw=False,
                 n_lim=None,
                 fixed_params=None,
                 high_distortion=False,
                 threads=max(1, cpu_count()-2)
                 ):
        """
        This function coordinates the calibration process, from detection to outputing a final camset.

        :param f_loc: the folder containing the nested cam images
        :param calibration_target: the calibration target
        :param save: should the final camset be saved
        :param save_loc: where should the final camset be saved
        :param draw: should the detection be drawn as the detections are completed
        :param n_lim: the maximum number of images to use for detection
        :param fixed_params: a dictionary of fixed parameters for the optimisation, which will not be changed
        :param high_distortion: Implements an iterative scheme for high distortion cameras.
        """
        if save_loc is None:
            save_loc = f_loc

        self.target = calibration_target

        self.detect_datapoints_in_imfile(
            f_loc=f_loc,
            calibration_target=calibration_target,
            draw=draw,
            n_lim=n_lim,
        )

        self.validate_detections()

        string_tail = '.camset'

        self.run_initial_calibration(save=save,
                                     save_loc=save_loc / ('initial_cameras' + string_tail),
                                     fixed_params=fixed_params)

        if high_distortion:
            self.detect_datapoints_in_imfile(
                f_loc=f_loc,
                calibration_target=calibration_target,
                draw=draw,
                n_lim=n_lim,
                camset=self.camera_set
            )

            self.run_initial_calibration(save=save,
                                         save_loc=save_loc / ('initial_cameras_high_distortion' + string_tail),
                                         )

            self.camera_set.draw_camera_distortions()

        self.camera_set.set_resolutions_from_file(floc=f_loc)
        self.run_stereo_calibration(
            save=save,
            save_loc=save_loc/('optimised_cameras' + string_tail),
            fixed_params=fixed_params,
        )

        return self.camera_set

    def get_resolutions_from_file(self, f_loc: Path):
        """
        Scans the file location, taking the camera resolution from the first set of images.

        :param f_loc: The top level folder containing the folders of images for each camera
        """
        names = get_subfolder_names(f_loc=f_loc)

        name_resolutions = {}

        for name in names:
            im_locs = glob_ims(f_loc / name)
            first_string = im_locs[0]
            temp_im = cv2.imread(str(first_string), -1)
            name_resolutions[name] = np.array(
                (temp_im.shape[1], temp_im.shape[0])
            ).astype(np.int)

        self.camera_resolutions = name_resolutions

        if self.camera_set is not None:
            for name, resolution in name_resolutions.items():
                self.camera_set[name].res = resolution
                self.camera_set[name]._update_state()

    def run_initial_calibration(self, detection: TargetDetection = None,
                                calibration_target: AbstractTarget = None,
                                save=True, save_loc: Path = Path('initial_estimate.camset'),
                                fixed_params: dict =None) -> CameraSet:
        """
        For all of the cameras, runs the calibration method provided by an abstract target.
        The default is an opencv calibration but may be overwritten.

        :param detection: the detection data to use for the calibration
        :param calibration_target: the calibration target to use for the calibration
        :param save: should the result be saved
        :param save_loc: where should the result be saved
        :param fixed_params: a dictionary of fixed parameters for the optimisation, which will not be changed
        :return: the camera set with the initial calibration
        """

        if save_loc.exists() and save:
            logging.info(f"Loading a previously saved initial calib from {save_loc}")
            self.camera_set = load_CameraSet(save_loc)
            return self.camera_set

        if detection is None:
            if self.detection is None:
                raise ValueError("No detections to calibrated with.")
            detection = self.detection
        if calibration_target is None:
            calibration_target = self.target

        # define the input structure to the
        # inp = data, target, intial_estimate, camera_res
        c_m = detection.features_per_im_per_cam()
        mask = ~np.any(c_m < 6, axis=1)
        score = np.sum(c_m, axis=1)
        pose_im = np.argmax(score * mask)
        # create a lambda based on the inputs

        logging.info("Pulling calibration method from target")
        work_fn = lambda datum: \
            calibration_target.initial_calibration(
                cam_name=datum[0],
                detection=datum[1],
                res=datum[2],
                pose_im=pose_im,
                fixed_params=fixed_params,
            )
        # take the data & zip it sideways
        cam_names = detection.cam_names
        cam_res = [self.camera_resolutions[name] for name in cam_names]
        cam_detections = detection.get_cam_list()
        # turn work into a pool and go zoom on it
        work_data = zip(cam_names, cam_detections, cam_res)
        raw_calibration = [work_fn(datum) for datum in tqdm(work_data)]
        cam_dict = {cam_name: cam for cam_name, cam in zip(cam_names, raw_calibration)}
        self.camera_set = CameraSet(camera_dict=cam_dict)

        if save:
            self.camera_set.save(save_loc)
        return self.camera_set


    def outlier_rejection(self, results, params: AbstractParamHandler) -> tuple[TargetDetection or None, bool]:
        """
        Takes a set of results from the optimisation and performs outlier rejection on them.
        Will identify which images are outliers, raise a warning, and return a detection set without this data.

        :param results:
        :param params:
        :return: A target detection without the outliers.
        """
        # outliers = mad_outlier_detection(results)

        detection = params.get_detection_data()
        # plot this as a boxplot
        d_list = [[] for _ in range(params.detection.max_ims)]
        for im_num, errs in zip(detection[:, 1], results):
            d_list[int(im_num)].append(errs)

        per_im_outliers = mad_outlier_detection([np.mean(datum) for datum in d_list if datum],
                                                draw=False,
                                                out_thresh=5)
        if per_im_outliers is not None:
            plt.boxplot(d_list)
            plt.ylabel("Average Pixels Reprojection error")
            plt.title(f"Images {list([per_im_outliers][0])} are likely outliers")
            plt.show()
        else:
            plt.boxplot(d_list)
            plt.ylabel("Average Pixels Reprojection error")
            plt.title("Reprojection error per image")
            plt.show()

        if True: # per_im_outliers is None:
            return None, False
        logging.info("deleting datum associated with the above outliers")
        data = params.detection
        return data.delete_row(im_num=per_im_outliers), True

    def run_stereo_calibration(
            self,
            param_handler = None,
            save: bool=True,
            save_loc: Path ='optimised_cameras.camset',
            fixed_params=None,
            floc: Path=None
    ):
        """
        This code runs a multi camera stereo calibration.
        The default behaviour is to run a standard object pose based bundle adjustment.

        :param param_handler: The parameter handler to use. If none is provided, a standard one will be created.
        :param save: should the result be saved
        :param save_loc: where should the result be saved
        :param fixed_params: a dictionary of fixed parameters for the optimisation, which will not be changed
        :param floc: the location of the images, used to update the camera resolutions
        """
        logging.info("Running the full multiview calibration")

        if param_handler is None:
            param_handler = StandardBundleParameters(
                detection=self.detection, target=self.target, camset=self.camera_set,
                fixed_params=fixed_params,
            )

        optimisation, cams = run_bundle_adjustment(
            param_handler=param_handler,
        )
        self.camera_set = cams
        # then we need to cache the relevant values of the optimisation
        # save the input params
        cache = {
            'x': optimisation.x,
            'err': optimisation.fun,
            'jac': optimisation
        }
        param_handler.camset = self.camera_set
        self.camera_set.set_calibration_history(
            optimisation_results=optimisation,
            param_handler=param_handler,
        )
        self.optim_results = cache

        if save:
            if floc is not None:
                self.camera_set.set_resolutions_from_file(floc)
            self.camera_set.save(save_loc)


    def detect_datapoints_in_imfile(
            self,
            f_loc: Path,
            calibration_target: AbstractTarget,
            caching=True,
            cache_name='detected_datapoints.pickle',
            draw=False,
            n_lim=None,
            camset = None,
    ) -> TargetDetection:
        """
        This function organises the detection of the image datapoints in a folder of images.

        :param f_loc: the file location to find the images in
        :param calibration_target: the calibration target to use for the detection
        :param caching:  should the result be cached
        :param cache_name: The name of the cache file
        :param draw: Should the detection be drawn
        :param n_lim: The maximum number of images to use for the detection
        :param camset: Optional, a camera set to use for the detections (for high distortion cameras)
        :return: A target detection.
        """

        logging.info('starting image detection')
        self.get_resolutions_from_file(f_loc)

        if camset is not None:
            cache_name = cache_name.split('.')[0] + "_with_calib.pickle"

        if not (f_loc / cache_name).exists() or not caching:
            logging.info('Not caching, starting detection')
            detected_sub_folders = get_subfolder_names(f_loc, return_full_path=True)

            if not detected_sub_folders:
                raise ValueError(f'no subfolders were found in {f_loc}')

            # checking for uneven image numbers
            sanitise_input_images(detected_sub_folders)

            cam_names = get_subfolder_names(f_loc=f_loc)
            use_cams = camset is not None
            work_fn = lambda file, cam=None: \
                calibration_target.find_in_imfolder(
                    file,
                    cam_names=cam_names,
                    draw=draw,
                    n_lim=n_lim,
                    camera=cam,
                )
            if use_cams:
                cam_zip = [camset[f.parts[-1]] for f in detected_sub_folders]
                detections = [work_fn(file, cam) for file, cam in zip(tqdm(detected_sub_folders), cam_zip)]
            else:
                detections = [work_fn(file) for file in tqdm(detected_sub_folders)]
            detected = reduce(lambda x, y: x + y, detections)

            # name the detection
            if caching:
                save_pickle(detected, f_loc / cache_name)
        else:
            logging.info('loading cached detection')
            detected = load_pickle(f_loc / cache_name)

        self.detection = detected
        return detected

    def validate_detections(self):
        """
        This function checks the detections for each camera, and prints a warning if the detection is poor.
        """
        detected = self.detection
        n_detected = {}

        board_fraction = {}

        corners_per_face = self.target.point_data.shape[-2]
        cam_names = detected.cam_names

        for cam_list in detected.get_cam_list():
            cam_ind = int(cam_list.get_data()[0,0])
            cam_name = cam_names[cam_ind]

            board_detected = 0
            im_lists = cam_list.get_image_list()
            for im_list in im_lists:
                datum = im_list.get_data()
                if datum is not None:
                    total_seen = datum.shape[0]
                    board_detected += 1
                    n_keys = datum.shape[1] - 4
                    seen = board_fraction.setdefault(cam_name, [])
                    if n_keys == 1:
                        seen.append(total_seen / corners_per_face)
                    else:
                        n_boards = len(
                            np.unique(datum[:, 2:-3], axis=0)
                        )
                        seen.append(
                            total_seen / corners_per_face / n_boards
                        )
            n_detected[cam_name] = board_detected / detected.max_ims

        for cam in cam_names:
            metric0 = n_detected[cam] * 100
            metric1 = np.mean(board_fraction[cam]) * 100
            logging.info(f'\tCamera "{cam}" detected boards: {metric0: .1f}%,'
                         f' board completeness: {metric1: .1f}%')
            if metric0 < 90:
                logging.warning(f'\tCamera "{cam}" has a high number of failed detections')
            if metric1 < 50:
                logging.warning(f'\tCamera "{cam}" struggled to detect full complete boards')
        return


def sanitise_input_images(detected_sub_folders:list[Path]):
    """
    Takes a list of detected sub folders and checks that they all have the same number of images.
    :param detected_sub_folders: A list of detected subfolders in the current location.
    :return:
    """
    equal_ims = [len(glob_ims(fol)) for fol in detected_sub_folders]
    if not len(set(equal_ims)) <= 1:
        logging.critical('Found an unequal number of images in each '
                         'sub folder')
        logging.critical("Do you wish to remove images not seen by "
                         "every camera? This will be destructive!")
        accepted_input = False
        inp_str = False
        while not accepted_input:
            inp_str = input("y/n:")
            if inp_str == 'y' or inp_str == 'n':
                accepted_input = True
        if inp_str == 'n':
            logging.critical("Aborting run, please fix images.")
        elif inp_str == 'y':
            im_nums = [glob_ims(fol) for fol in detected_sub_folders]
            found_nums = []
            found_sets = []
            for im_strings in im_nums:
                # get out numbers
                ns = [int(re.findall(r'\d+', nam.parts[-1])[-1])
                      for nam in im_strings]
                found_nums.append(ns)
                found_sets.append(set(ns))
            # get whether numbers are good by some lookup

            for id, (x, y) in enumerate(zip(found_sets, found_sets[1:])):
                found_sets[id + 1] = x.intersection(y)

            cons_ims = np.array(list(found_sets[-1]))
            for im_string, found_num in zip(im_nums, found_nums):
                mask = np.isin(found_num, cons_ims)
                for im, mask_item in zip(im_string, mask):
                    if not mask_item:
                        os.remove(im)
